"""Recollect the ENTIRE current physical netlist, optimize, apply a LAC batch,
then repeat on the NEW physical netlist. Separate from frozen old workflows.
"""
import runtime
import argparse, json, time
from collections import Counter
from pathlib import Path
import numpy as np
from logic import Library
from bacs_data import LIBRARY_PATH, load
from bacs_numeric import cleanup, save
from cutmap import CutMapper
from screen import FanoutCones
from simulate import trace, outputs, pack, validate_numpy
from prepare_resub_iscas import DEFAULT, sha, verify_bundle
from research_plan_overlay import Overlay, Action, identity
from research_c880_plan_search import context_actions, choose_frontier, losses, fitted
from research_c880_local_exact import cuts
from research_batch_iterations import iterate
import er_data


def live_actions(overlay, actions):
    amap,_=overlay.graph(actions);seen=set();live=set()
    def visit(n):
        n=overlay.base.resolve(n)
        if n in seen:return
        seen.add(n)
        if n in amap:live.add(n)
        for d in overlay.dependencies(n,amap):visit(d)
    for n in overlay.base.outputs:visit(n)
    return identity(a for a in actions if a.target in live)


def collect_physical(circuit, inputs, gold, count, mapper, stats, cap=450):
    """No inherited rankings/candidate/response cache. Every root is measured."""
    o=Overlay(circuit,inputs,gold,count,stats)
    ctx=o.refresh([],tuple(circuit.producer),'physical_netlist_full_recollection')
    cones=FanoutCones(circuit);records=[]
    for root,response in ctx['responses'].items():
        forbidden=o.forbidden(ctx,root);t=[v[root] for v in ctx['state']['values']]
        donors=[]
        for n in (*circuit.inputs,*circuit.producer):
            if n in forbidden:continue
            agreement=sum(count-(t[k]^ctx['state']['values'][k][n]).bit_count() for k in (0,1))
            choices=[]
            for inverted in (False,True):
                errors=[]
                for k in (0,1):
                    word=ctx['state']['values'][k][n]^(o.mask if inverted else 0)
                    b0,b1=response['forced'][k]
                    errors.append(((word&b1)|((word^o.mask)&b0)).bit_count())
                choices.append((max(errors),sum(errors),inverted,errors))
            _,_,inv,errors=min(choices)
            donors.append(dict(net=n,inverted=inv,errors=errors,
                agreement=agreement/(2*count),signed_agreement=2*agreement/(2*count)-1,
                absolute_correlation=abs(2*agreement/(2*count)-1)))
            stats['recollected_single_divisor_scores']+=2
        # Correlation is recorded; actual forced-error fit decides useful donors.
        donors=sorted(donors,key=lambda d:(max(d['errors']),sum(d['errors']),-d['absolute_correlation'],d['net']))[:12]
        bestgain=0;besterrors=[count,count]
        for d in donors:
            plan=mapper.lib.mapper.plans[5 if d['inverted'] else 10]
            gain=cones.saved_with_inputs(root,(d['net'],))-mapper.cost(plan)
            d['saved_units']=gain
            if (max(d['errors']),sum(d['errors'])) < (max(besterrors),sum(besterrors)):besterrors=d['errors']
            if max(d['errors'])<=cap:bestgain=max(bestgain,gain)
        observability=0
        for deltas in response['output_deltas']:
            union=0
            for word in deltas.values():union|=word
            observability+=union.bit_count()
        records.append(dict(root=root,mffc_units=cones.potential(root),
            error_sensitivity=response['sensitivity']/(2*count),
            output_observability=observability/(2*count),
            repair_overlap=response['repair_overlap'],new_error_overlap=response['new_error_overlap'],
            best_donor_errors=besterrors,best_donor_gain_units=bestgain,donors=donors))
    ranked=sorted(records,key=lambda r:(-r['best_donor_gain_units']/(1+sum(r['best_donor_errors'])),
                                        sum(r['best_donor_errors']),-r['mffc_units'],r['root']))
    large=sorted(records,key=lambda r:(-r['mffc_units'],r['root']))
    repair=sorted(records,key=lambda r:(-r['repair_overlap'],-r['mffc_units'],r['root']))
    order=[];seen=set();positions=[0,0,0];queues=[ranked,large,repair]
    while len(order)<len(records):
        for q in (0,0,1,2):
            while positions[q]<len(records) and queues[q][positions[q]]['root'] in seen:positions[q]+=1
            if positions[q]<len(records):
                n=queues[q][positions[q]]['root'];seen.add(n);order.append(n);positions[q]+=1
    stats['physical_netlist_recollections']+=1
    return dict(overlay=o,context=ctx,records=records,root_order=order,cones=cones)


def boundary_actions(info,context,args,mapper,stats,deadline):
    """Screen a small wider-boundary quota; fit a truth, never enumerate LACs."""
    if not args.wide_roots:return []
    o=info['overlay'];base=o.base;actions=[]
    roots=sorted(context['responses'],key=lambda n:(-info['cones'].potential(n),n))[:args.wide_roots]
    for root in roots:
        if time.perf_counter()>=deadline:break
        forbidden=o.forbidden(context,root)
        supports=[ns for ns in cuts(base,root,width=4,beam=2)
                  if len(ns)==4 and not(set(ns)&forbidden)]
        supports=sorted(supports,key=lambda ns:(-info['cones'].saved_with_inputs(root,ns),ns))[:2]
        for ns in supports:
            truth,loss,changes=fitted(ns,context['state']['values'],context['responses'][root]['forced'],o.mask)
            stats['wider_boundary_fits']+=1
            if max(loss)>round(args.cap*args.near_ratio):continue
            functions=[truth,*[truth^(1<<row) for _,row in sorted(changes)[:2]]]
            for fun in dict.fromkeys(functions):
                actions.append(Action(root,ns,mapper.map(len(ns),fun)))
                stats['wider_boundary_mapped_actions']+=1
    return actions


def plan_batch(info,args,mapper,stats,deadline):
    o=info['overlay'];initial=o.score([]);frontier=[initial];seen={()};finalists=[];contexts={}
    repairing=getattr(args,'objective','area')=='repair'
    caps=getattr(args,'error_caps',(args.cap,)*len(initial['bad']))
    if len(caps)!=len(initial['bad']):raise ValueError('One declared error cap is required per search group')
    within=lambda state,ratio=1.:all(e<=round(cap*ratio) for e,cap in zip(losses(state),caps))
    roots=info['root_order'][:args.roots];info['plan_log']=[]
    base_context=dict(info['context']);base_context['responses']={n:info['context']['responses'][n] for n in roots}
    contexts[()]=base_context
    for depth in range(args.plan_steps):
        if time.perf_counter()>=deadline:break
        parent=frontier[depth%len(frontier)];key=parent['actions']
        if key not in contexts:
            chosen=list(dict.fromkeys([*(a.target for a in key),*roots]))
            contexts[key]=o.refresh(key,chosen,'virtual_batch_adjustment')
            if len(contexts)>5:contexts.pop(next(iter(contexts)))
        ctx=contexts[key];actions,_=context_actions(o,ctx,mapper,info['cones'],args,deadline,stats)
        actions+=boundary_actions(info,ctx,args,mapper,stats,deadline)
        # Optional mapped larger-care proposals supplied by the generic
        # MCTS/Agent adapter; absent in the original standalone experiments.
        for action in info.get('extra_actions',()):
            if action.target in ctx['responses']:
                forbidden=o.forbidden(ctx,action.target)
                if not(set(action.names)&forbidden):actions.append(action)
        current={a.target:a for a in key};proposals=[]
        for action in actions:
            if action.target not in current and len(current)>=args.batch_size:continue
            proposed=identity({**current,action.target:action}.values())
            if proposed in seen:continue
            seen.add(proposed)
            try:
                state=o.propose(ctx,action)
                compact=live_actions(o,state['actions'])
                if compact!=state['actions']:state=dict(state,actions=compact)
            except ValueError:
                stats['batch_cycle_rejections']+=1;continue
            if not within(state,args.near_ratio):continue
            if state['units']>o.base.units+args.area_debt:continue
            stats['batch_plan_response_evaluations']+=1
            # Raw area can miss later cofactor/constant cleanup. A bounded-area
            # batch may be physically tried ONCE, then accepted only if its
            # measured cleaned area improves and the real errors pass.
            useful_repair=(sum(losses(state))<sum(losses(initial)) and max(losses(state))<=max(losses(initial)))
            if repairing and state['units']>=o.base.units:
                useful_repair=useful_repair and sum(losses(initial))-sum(losses(state))>=2*getattr(args,'min_repair_recovery',0)
            if state['units']<=o.base.units+args.area_debt and within(state) and len(state['actions'])>=args.min_batch and (not repairing or useful_repair):
                finalists.append(state)
            same=all(all(state['values'][k][o.base.resolve(n)]==parent['values'][k][o.base.resolve(n)]
                         for n in o.base.outputs) for k in (0,1))
            if same and state['units']>=parent['units']:continue
            proposals.append(state)
        # Removal is a plan edit, never an intervening physical application.
        for target in list(current)[:args.delete_quota]:
            proposed=identity(v for n,v in current.items() if n!=target)
            if proposed in seen:continue
            seen.add(proposed)
            try:state=o.score(proposed)
            except ValueError:continue
            if within(state,args.near_ratio):proposals.append(state)
        frontier=choose_frontier([*frontier,*proposals],args)
        info['plan_log'].append(dict(step=depth,proposals=len(proposals),
            frontier=[dict(area=o.base.lib.area(s['units']),errors=list(losses(s)),lacs=len(s['actions'])) for s in frontier],
            feasible_batches=len(finalists)))
        if len(frontier)==1 and not proposals:break
    if not finalists:return None
    winner=min(finalists,key=(lambda s:(sum(losses(s)),max(losses(s)),s['units'])) if repairing else
               (lambda s:(s['units'],sum(losses(s)),-len(s['actions']))))
    full=o.score(winner['actions'])
    if full['values']!=winner['values'] or full['bad']!=winner['bad']:
        raise AssertionError('Batch response prediction mismatch')
    return winner


def run(args):
    if args.min_batch<1 or args.batch_size<args.min_batch or not 1<=args.branches<=4:
        raise ValueError('Expected 1<=min_batch<=batch_size and 1..4 retained plans')
    if not 0<=args.cap<=10000 or args.near_ratio<1 or args.roots<1 or args.wide_roots<0:
        raise ValueError('Invalid error cap, near ratio, or screening quota')
    start=time.perf_counter();folder=Path(args.output).resolve()
    if False:raise ValueError('D: only')
    folder.mkdir(parents=True,exist_ok=False)
    for name in ('research_batch_iterations.py','research_c880_recollect_batches.py',
                 'research_plan_overlay.py','research_c880_plan_search.py'):
        (folder/name).write_bytes((Path(__file__).parent/name).read_bytes())
    verify_bundle(DEFAULT,'c880');lib=Library(LIBRARY_PATH);mapper=CutMapper(lib,4)
    initial=load(args.input,lib);ref=load(DEFAULT/'c880/accurate.v',lib);stats=Counter()
    x=[pack(er_data.matrix('c880',ref.inputs,k),ref.inputs) for k in (0,1)]
    gold=[outputs(ref,trace(ref,p,10000,stats)) for p in x]
    save(folder/'protocol.json',dict(vars(args),input_sha256=sha(args.input),accurate_sha256=sha(DEFAULT/'c880/accurate.v'),
        search_splits=[0,1],count_per_group=10000,metric='strict_vector_er',api_calls=0,
        workflow='physical full recollection -> plan many LACs -> one batch apply -> check -> recollect NEW physical circuit; repeat',
        candidate_policy='full node survey; bounded complex LAC candidates, no LAC enumeration',
        correlation_measure='signed bit agreement 2*P(same)-1 and its absolute value; not a Pearson coefficient',
        final_policy='freeze selected circuit before independent diagnostics; no reselection'))
    deadline=time.perf_counter()+args.seconds;surveys=[];last_errors=[];event_log=[]
    def errors(c,expected=None):
        result=[]
        for k in (0,1):
            y=outputs(c,trace(c,x[k],10000,stats));bad=0
            if expected is not None and y!=expected[k]:
                raise AssertionError('Cleanup/readback changed batch search-group outputs')
            for n in ref.outputs:bad|=y[n]^gold[k][n]
            result.append(bad.bit_count())
        return result
    if max(errors(initial))>args.cap:raise ValueError('Starting circuit exceeds declared search cap')
    def collect(c,index):
        phase_start=time.perf_counter()
        work=folder/('round_%02d'%index);work.mkdir();c.write(work/'input.v')
        info=collect_physical(c,x,gold,10000,mapper,stats,args.cap)
        record=dict(round=index,physical_input_sha256=sha(work/'input.v'),area=lib.area(c.units),
                    nodes_measured=len(info['records']),records=info['records'],root_order=info['root_order'])
        save(work/'recollected_nodes.json',record);surveys.append({k:v for k,v in record.items() if k!='records'})
        stats['physical_collection_milliseconds']+=round(1000*(time.perf_counter()-phase_start))
        print('[FULL RECOLLECT]',json.dumps({k:v for k,v in surveys[-1].items() if k!='root_order'}),flush=True)
        return info
    def plan(c,info,index):
        if time.perf_counter()>=deadline:return None
        phase_start=time.perf_counter()
        result=plan_batch(info,args,mapper,stats,min(deadline,time.perf_counter()+args.round_seconds))
        stats['plan_adjustment_milliseconds']+=round(1000*(time.perf_counter()-phase_start))
        save(folder/('round_%02d'%index)/'batch_search.json',info.get('plan_log',[]))
        if result:
            save(folder/('round_%02d'%index)/'batch_plan.json',dict(actions=[dict(target=a.target,names=a.names,plan=repr(a.plan))
                for a in result['actions']],pre_cleanup_area=lib.area(result['units']),errors=list(losses(result))))
        return result
    def apply(c,info,proposal,index):
        phase_start=time.perf_counter()
        raw=info['overlay'].commit(proposal['actions']);raw_outputs=[]
        for k in (0,1):
            actual=outputs(raw,trace(raw,x[k],10000,stats))
            raw_outputs.append(actual)
            expected={n:proposal['values'][k][c.resolve(n)] for n in c.outputs}
            if actual!=expected:raise AssertionError('Batch physical/virtual mismatch')
        candidate=cleanup(raw,mapper);physical_path=folder/('round_%02d'%index)/'batch_output.v'
        candidate.write(physical_path);readback=load(physical_path,lib)
        if readback.units!=candidate.units:raise AssertionError('Batch area changed on physical readback')
        actual_errors=errors(readback,raw_outputs);last_errors[:]=actual_errors
        stats['batch_application_milliseconds']+=round(1000*(time.perf_counter()-phase_start))
        return readback,dict(lacs=len(proposal['actions']),area=lib.area(readback.units),errors=actual_errors,
                              physical_batch_applications=1,raw_area=lib.area(raw.units),
                              readback_area_verified=True,readback_sha256=sha(physical_path))
    def eligible(c):return max(last_errors)<=args.cap
    def event(row):
        event_log.append(row);save(folder/'batch_events.json',event_log)
        print('[BATCH ITERATION]',json.dumps(row),flush=True)
    selected,events=iterate(initial,collect,plan,apply,eligible,args.outer_rounds,on_round=event)
    selected.write(folder/'selected.v');frozen=sha(folder/'selected.v')
    save(folder/'frozen_selection.json',dict(sha256=frozen,area=lib.area(selected.units),events=events))
    search_seconds=time.perf_counter()-start;checks=[]
    for split in args.splits:
        m=er_data.matrix('c880',ref.inputs,split)
        bad=int(np.any(validate_numpy(ref,m)!=validate_numpy(selected,m),axis=1).sum())
        stats['independent_numpy_circuit_evaluations']+=2
        checks.append(dict(split=split,count=10000,errors=bad,er=bad/10000,passed=bad<=args.final_limit))
    result=dict(initial_area=lib.area(initial.units),area=lib.area(selected.units),
        reduction_vs_accurate=1-selected.units/ref.units,new_area_gain=selected.units<initial.units,
        search_errors=errors(selected),events=events,surveys=surveys,checks=checks,
        strict_pass=all(c['passed'] for c in checks) if checks else None,
        search_and_setup_seconds=search_seconds,total_seconds=time.perf_counter()-start,
        selected_sha256=frozen,stats=dict(stats),api_calls=0)
    save(folder/'report.json',result);print('[RECOLLECT BATCH RESULT]',json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='runs/c880_er05_144p704_localcare_v1/best.v')
    p.add_argument('--output',required=True)
    p.add_argument('--seconds',type=float,default=120)
    p.add_argument('--round-seconds',type=float,default=30)
    p.add_argument('--outer-rounds',type=int,default=4)
    p.add_argument('--plan-steps',type=int,default=16)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--min-batch',type=int,default=2)
    p.add_argument('--roots',type=int,default=32)
    p.add_argument('--wide-roots',type=int,default=12)
    p.add_argument('--divisors',type=int,default=16)
    p.add_argument('--branches',type=int,default=4)
    p.add_argument('--delete-quota',type=int,default=8)
    p.add_argument('--cap',type=int,default=450)
    p.add_argument('--near-ratio',type=float,default=1.45)
    p.add_argument('--area-debt',type=int,default=32)
    p.add_argument('--overflow-price',type=float,default=0.15)
    p.add_argument('--final-limit',type=int,default=500)
    p.add_argument('--splits',type=int,nargs='*',default=[12800,12801,12802,12803])
    run(p.parse_args())
