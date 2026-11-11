"""Isolated strict-vector ER repair/recompression research on common baselines.

Recollect current-graph responses; rank roots by correctable input vectors,
not solely area. Accept bounded-area-debt or same-area partial repairs before
recompression. Search never uses held-out inputs or a numerical-word surrogate.
"""
import runtime
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from bacs_data import LIBRARY_PATH,load
from bacs_numeric import bits,number,save,cleanup,learned_sets
from bacs_frontier import VirtualArea,response_truth
from hybrid_identity import structural_id
from logic import Library,Plan
from netlist import Circuit,Gate,ZERO,ONE
from simulate import trace,outputs,pack,plan_value,validate_numpy
from quality_research_mcts import parser as base_parser
from research_c880_conditional import ConditionalWorker
from er_repair_tools import error_words,error_totals,development_limit
import er_resub_data as data


def detached_functions(circuit,bindings,mapper):
    """Change PO observations while preserving every original internal fanout."""
    used=set(circuit.inputs)|set(circuit.outputs)|set(circuit.producer)|set(circuit.aliases)
    rename={}
    def fresh(prefix):
        n=prefix+str(len(used))
        while n in used:n+='_'
        used.add(n);return n
    for out in bindings:rename[out]=fresh('__er_old_')
    r=lambda n:rename.get(n,n)
    gates=[Gate(g.name,g.cell,{p:r(n) for p,n in g.pins.items()}) for g in circuit.gates]
    aliases={r(k):r(v) for k,v in circuit.aliases.items()};memo={}
    def emit(plan,names):
        if plan.name=='leaf':return r(names[plan.leaf]) if plan.leaf>=0 else (ZERO,ONE)[plan.leaf+2]
        key=plan,names
        if key in memo:return memo[key]
        cell=circuit.lib.cells[plan.name];net=fresh('__er_new_')
        gates.append(Gate(fresh('__er_gate_'),cell.name,
            {**dict(zip(cell.inputs,(emit(p,names) for p in plan.children))),next(iter(cell.functions)):net}))
        memo[key]=net;return net
    for out,(names,plan) in bindings.items():aliases[out]=emit(plan,tuple(names))
    return cleanup(Circuit(circuit.lib,circuit.inputs,circuit.outputs,gates,aliases,circuit.name),mapper)


def union(words):
    result=0
    for word in words:result|=word
    return result


def vector_objective(errors,count):
    """Exact global lexicographic ER then Hamming, without surrogate leakage."""
    decoded=[bits(e,count).astype(np.int64) for e in errors]
    hd=sum(decoded,np.zeros(count,dtype=np.int64));bad=(hd>0).astype(np.int64)
    return bad*(count*len(errors)+1)+hd


def wire_repairs(w):
    c=w.current;n=w.count;mask=(1<<n)-1;values=trace(c,w.inputs[0],n,w.stats)
    out=outputs(c,values);errors=dict(zip(w.order,error_words(out,w.er_reference,w.order)))
    initial=error_totals(tuple(errors.values()));pool=list(dict.fromkeys([ZERO,ONE,*c.inputs,*c.producer]))
    bindings={};trace_rows=[]
    for sweep in range(3):
        changed=False
        # Unique wrong inputs are the primary opportunity; shared errors then
        # receive partial-bit recovery, allowing a later PO to complete repair.
        order=sorted(w.order,key=lambda o:(-(errors[o]&~union(e for p,e in errors.items() if p!=o)).bit_count(),-errors[o].bit_count(),o))
        for target in order:
            if w.remaining()<1:break
            other=union(e for p,e in errors.items() if p!=target)
            current=error_totals(tuple(errors.values()));best=None
            for signal in pool:
                for inverted in (False,True):
                    e=(values[signal]^(mask if inverted else 0))^w.er_reference[target]
                    score=((other|e).bit_count(),current[1]-errors[target].bit_count()+e.bit_count())
                    w.stats['er_global_wire_function_scores']+=1
                    key=(*score,int(inverted),c.level.get(signal,0),signal)
                    if score<current and (best is None or key<best[0]):best=key,signal,inverted,e
            if best:
                _,s,inverted,e=best;bindings[target]=((s,),w.mapper.map(1,1 if inverted else 2))
                errors[target]=e;changed=True
        trace_rows.append(dict(sweep=sweep,vector_errors=union(errors.values()).bit_count(),hamming_errors=sum(e.bit_count() for e in errors.values())))
        if not changed:break
    if not bindings:return []
    trial=detached_functions(c,bindings,w.mapper)
    predicted=error_totals(tuple(errors.values()))
    return [(trial,predicted,dict(kind='global_po_wire_repair',bindings={o:dict(signal=ns[0],inverted=plan!=w.mapper.map(1,2)) for o,(ns,plan) in bindings.items()},sweeps=trace_rows))]


def response_repairs(w):
    c=w.current;n=w.count;mask=(1<<n)-1;values=trace(c,w.inputs[0],n,w.stats)
    actual=outputs(c,values);errors=error_words(actual,w.er_reference,w.order)
    bad,hd=error_totals(errors);base_obj=vector_objective(errors,n);area=VirtualArea(c,w.mapper)
    ranked=sorted(c.producer,key=lambda s:(-area.cones.potential(s),s))
    # Mix area opportunities with the rest of the live circuit before expensive
    # truth fitting. Whole root-response scans are not LAC enumeration.
    if len(ranked)>w.args.response_roots:
        first=ranked[:w.args.response_roots//2];others=[r for r in ranked if r not in first]
        ordered=sorted(others,key=lambda s:hashlib.blake2b((structural_id(c)+s).encode(),digest_size=8).digest())
        ranked=first+ordered[:w.args.response_roots-len(first)]
    contexts=[]
    for target in ranked:
        if w.remaining()<1:break
        fv=trace(c,w.inputs[0],n,w.stats,base=values,flip=target);flipped=outputs(c,fv)
        fe=error_words(flipped,w.er_reference,w.order);new_bad=union(fe)
        recovered=(union(errors)&~new_bad).bit_count()
        hamming_potential=sum((e&~f).bit_count() for e,f in zip(errors,fe))
        w.stats['er_response_roots_screened']+=1
        if recovered or hamming_potential:contexts.append((recovered,hamming_potential,target,fe,flipped))
    contexts.sort(key=lambda q:(-q[0],-q[1],-area.cones.potential(q[2]),q[2]));proposals=[]
    for recovery,_,target,fe,flipped in contexts[:w.args.response_fit_roots]:
        if w.remaining()<1:break
        fobj=vector_objective(fe,n);tb=bits(values[target],n).astype(bool)
        l0=np.where(tb,fobj,base_obj);l1=np.where(tb,base_obj,fobj)
        change=tuple(actual[o]^flipped[o] for o in w.order)
        for names,code in learned_sets(c,target,values,l0,l1,n,w.stats):
            for penalty in (1,4):
                truth=response_truth(code,l0,l1,base_obj,len(names),penalty);plan=w.mapper.map(len(names),truth)
                word=plan_value(plan,names,c.lib,values,mask);delta=values[target]^word
                predicted=tuple(e^(d&delta) for e,d in zip(errors,change));score=error_totals(predicted)
                w.stats['er_response_truth_function_scores']+=1
                if score>=(bad,hd):continue
                gain=area.gain(target,names,plan)
                if gain < -w.args.repair_debt_units:continue
                proposals.append((score,-gain,target,names,plan,truth,penalty,recovery))
    queues=[sorted(proposals,key=lambda q:(q[0],q[1],q[2])),
            sorted(proposals,key=lambda q:(max(0,q[1])/max(1,bad-q[0][0]),q[0],q[2]))]
    selected=[];seen=set()
    for rank in range(8):
        for queue in queues:
            if rank<len(queue):
                q=queue[rank];key=q[2],q[3],q[5]
                if key not in seen:selected.append(q);seen.add(key)
            if len(selected)==8:break
        if len(selected)==8:break
    result=[]
    for score,_,target,names,plan,truth,penalty,recovery in selected:
        if w.remaining()<1:break
        trial=cleanup(c.materialize([(target,names,plan)]),w.mapper);w.stats['er_repair_materializations']+=1
        result.append((trial,score,dict(kind='recollected_internal_repair',target=target,support=names,
            truth=truth,penalty=penalty,root_vector_recovery_potential=recovery)))
    return result


def metrics(w,c,split):
    actual=outputs(c,trace(c,w.inputs[split],w.count,w.stats))
    ref=w.er_reference if split==0 else w.er_development_reference
    w.stats['er_research_actual_propagations']+=1
    return error_totals(error_words(actual,ref,w.order))


def reset_caches(w):
    w.cached=None;w.bit_cache=None;w.verified_pairs={};w.training_view=None;w.development_view=None
    if hasattr(w,'area_model'):del w.area_model


def adopt_repair(w,proposals,events):
    parent=w.current;train_base=metrics(w,parent,0);dev_base=metrics(w,parent,1);best=None
    for trial,predicted,detail in proposals:
        if w.remaining()<1:break
        train=metrics(w,trial,0)
        if train!=predicted:raise AssertionError('ER repair response mismatch')
        row=dict(detail,raw_area=w.source.lib.area(trial.units),training_errors=train,accepted=False);events.append(row)
        if train>=train_base or train[0]>w.metric.limit:continue
        dev=metrics(w,trial,1);row['development_errors']=dev
        if dev>dev_base or dev[0]>w.metric.limit:continue
        if trial.units>min(w.source.units,w.lowest_units+w.args.repair_debt_units):continue
        if structural_id(trial) in w.seen:continue
        if w.backend and w.remaining()>2:trial=w.backend.optimize(trial,w.remaining,variants=1)
        row['mapped_area']=w.source.lib.area(trial.units)
        net_recovery=max(1,train_base[0]-train[0]+dev_base[0]-dev[0])
        debt=max(0,trial.units-parent.units)
        score=(debt/net_recovery,max(train[0],dev[0]),train[1]+dev[1],trial.units)
        if best is None or score<best[0]:best=score,trial,train,dev,row
    if best is None:return False
    _,trial,train,dev,row=best;row['accepted']=True
    w.current=trial;w.loss=train[0];w.lowest_units=min(w.lowest_units,trial.units);w.seen.add(structural_id(trial));reset_caches(w)
    start=len(w.history);w.checkpoint('er_global_residual_repair',[]);w.annotate(start)
    print('[ER RESIDUAL REPAIR]',json.dumps(row),flush=True)
    return True


def run(a):
    if a.metric!='er' or not math.isfinite(a.bound) or not 0<a.bound<1:
        raise ValueError('This isolated runner supports strict-vector ER only')
    if not math.isfinite(a.seconds) or not 0<a.seconds<=600 or set(a.final_splits)&{0,1}:
        raise ValueError('Use a finite <=600-second budget and independent final splits')
    if not a.final_splits or not 0<a.training_fraction<=1 or not 0<a.development_fraction<=1:
        raise ValueError('Invalid selection fractions or empty final check')
    folder=Path(a.output).resolve()
    if False:raise ValueError('D: only')
    folder.mkdir(parents=True,exist_ok=False);started=time.perf_counter();lib=Library(LIBRARY_PATH)
    reference,order,audit=data.preflight(a.design,lib);start=load(a.start,lib) if a.start else reference
    if start.inputs!=reference.inputs or start.outputs!=reference.outputs:raise ValueError('Different circuit interface')
    inputs=[pack(data.matrix(a.design,reference.inputs,i),reference.inputs) for i in (0,1)]
    refs=[number(outputs(reference,trace(reference,x,10000)),order,10000) for x in inputs]
    a.no_word=True;remaining=lambda:max(0.,a.seconds-(time.perf_counter()-started))
    snapshot=folder/'source';snapshot.mkdir();hashes={}
    for module in tuple(sys.modules.values()):
        p=Path(getattr(module,'__file__','') or '.')
        if p.suffix=='.py' and p.resolve().parent==Path(__file__).resolve().parent:
            shutil.copy2(p,snapshot/p.name);hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    save(folder/'protocol.json',dict(args=vars(a),baseline=audit,source_hashes=hashes,
        warm_start=bool(a.start),input_count=10000,metric='strict_vector_er',selection_splits=[0,1],final_splits=a.final_splits,
        api_calls=0,exhaustive_lacs=False))
    (folder/'worker').mkdir();w=ConditionalWorker(start,order,inputs,refs,a,folder/'worker',remaining=remaining,reference_source=reference)
    starting=metrics(w,start,0),metrics(w,start,1);events=[];rounds=[]
    for iteration in range(a.rounds):
        if remaining()<2:break
        before=structural_id(w.current)
        for tool in a.tools:
            if remaining()<2:break
            print('[ER RESIDUAL TOOL]',json.dumps(dict(iteration=iteration,tool=tool,area=lib.area(w.current.units),remaining=remaining())),flush=True)
            if tool=='native':
                for policy in ('balanced','area','balanced'):
                    profile='batch' if policy=='balanced' else policy
                    if remaining()<2:break
                    if (structural_id(w.current),profile,1.) in w.blocked:continue
                    w.compress(policy,1.)
            else:
                # Leave acceptance and synthesis time after candidate generation.
                deadline=time.perf_counter()+min(a.repair_generation_seconds,max(.1,remaining()-4));old=w.global_remaining
                w.global_remaining=lambda:max(0.,min(old(),deadline-time.perf_counter()))
                try:proposals=wire_repairs(w) if tool=='wire' else response_repairs(w)
                finally:w.global_remaining=old
                adopt_repair(w,proposals,events)
        rounds.append(dict(iteration=iteration,area=lib.area(w.current.units),training_loss=w.loss,seconds=time.perf_counter()-started))
        save(folder/'events.json',events);save(folder/'trajectory.json',rounds)
        if structural_id(w.current)==before:break
    candidates=[]
    for checkpoint in w.history:
        if checkpoint['training_loss']>w.metric.limit:continue
        c=load(w.folder/checkpoint['path'],lib);dev=metrics(w,c,1)
        if dev[0]<=w.metric.limit:candidates.append((c.units,max(checkpoint['training_loss'],dev[0]),dev[1],checkpoint['path'],c))
    if not candidates:raise RuntimeError('No feasible train/development checkpoint')
    best=min(candidates,key=lambda q:q[:4]);winner=best[-1];winner.write(folder/'selected_before_checks.v')
    selected_sha=hashlib.sha256((folder/'selected_before_checks.v').read_bytes()).hexdigest()
    save(folder/'frozen_selection.json',dict(path=best[3],sha256=selected_sha,area=lib.area(winner.units),held_out_read=False))
    checks=[]
    for split in a.final_splits:
        x=data.matrix(a.design,reference.inputs,split);inp=pack(x,reference.inputs)
        actual=outputs(winner,trace(winner,inp,10000));ref=outputs(reference,trace(reference,inp,10000))
        bad=union(actual[o]^ref[o] for o in order).bit_count()
        independent=validate_numpy(winner,x);original=validate_numpy(reference,x)
        if int(np.any(independent!=original,axis=1).sum())!=bad:raise AssertionError('Independent ER mismatch')
        checks.append(dict(split=split,count=10000,wrong_vectors=bad,er=bad/10000,passed=bad<=w.formal_limit))
    passed=all(q['passed'] for q in checks);winner.write(folder/('best.v' if passed else 'failed_final.v'))
    for name,h in hashes.items():
        if hashlib.sha256((Path(__file__).resolve().parent/name).read_bytes()).hexdigest()!=h:raise RuntimeError('Source changed: '+name)
    result=dict(design=a.design,bound=a.bound,warm_start=bool(a.start),baseline_area=lib.area(reference.units),
        starting_area=lib.area(start.units),area=lib.area(winner.units),reduction=1-winner.units/reference.units,
        starting_errors=starting,strict_pass=passed,checks=checks,whole_seconds=time.perf_counter()-started,
        formal_wrong_vector_limit=w.formal_limit,training_wrong_vector_limit=w.metric.limit,
        development_wrong_vector_limit=development_limit(w.metric,a),
        stats=dict(w.stats),events=events,trajectory=rounds,selected_sha256=selected_sha,api_calls=0,
        mapping_calls=len(w.backend.rows),mapping_cec=sum(r.get('cec_equivalent',False) for r in w.backend.rows))
    save(folder/'report.json',result);print('[ER RESIDUAL RESULT]',json.dumps({k:v for k,v in result.items() if k not in ('events','trajectory')}),flush=True)
    return 0 if passed else 2


def parser():
    p=base_parser();p.add_argument('--start');p.add_argument('--tools',nargs='+',choices=('wire','response','native'),default=['wire','native','response','native'])
    p.add_argument('--rounds',type=int,default=6);p.add_argument('--response-roots',type=int,default=128)
    p.add_argument('--response-fit-roots',type=int,default=24);p.add_argument('--repair-generation-seconds',type=float,default=15.)
    p.add_argument('--final-splits',nargs='+',type=int,default=[3000,3001])
    p.set_defaults(seconds=120.,development_fraction=1.,development_confidence=0.,no_word=True)
    return p


if __name__=='__main__':
    a=parser().parse_args()
    if a.metric!='er' or not 0<a.bound<1 or not 0<a.seconds<=600 or set(a.final_splits)&{0,1}:raise ValueError('Bounded strict ER only')
    raise SystemExit(run(a))
