"""Bounded research operators on the CURRENT physical graph, for any ISCAS85 design.

The controller supplies a worker and its accurate reference. No historical run,
node name, author approximate result, or target area is read by this module.
Exact stages require whole-circuit CEC; budget-care also requires actual ER on
the two search groups. Final diagnostic inputs are never used here.
"""
import hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
from bacs_data import load
from bacs_numeric import save
from er_repair_tools import development_limit,error_totals,error_words
from simulate import outputs,trace
from hybrid_identity import structural_id
from prepare_resub_iscas import sha


def search_errors(w,c):
    return [error_totals(error_words(outputs(c,trace(c,w.inputs[k],w.count,w.stats)),ref,w.order))[0]
            for k,ref in enumerate((w.er_reference,w.er_development_reference))]


def operator_schedule(args,cap,dev,count=10000):
    # Representation diversity is retained INSIDE the exact portfolio. It is
    # essential: throwing away neutral/slightly larger implementations after
    # each mapping loses the successful ODC + port-order trajectory.
    schedule=[
        ('exact_frontier',65.,'best.v',dict(rounds=3,seed=args.seed+1)),
        ('observable',32.,'best.v',dict(trials=24,wide=True,width=6,pool=32,anchors=6,beam=4,error_inclusion=False)),
        ('local_care',30.,'best.v',dict(width=8,cuts=128,rounds=6,sampled_care=True,error_inclusion=False)),
        ('mapped_sat',16.,'best.v',dict(rounds=2,wide=False,case_seconds=3.)),
        ('budget_care',30.,'selected.v',dict(bound=args.bound,training_cap=cap/count,
            development_cap=dev/count,teacher='optimal',cuts_per_root=2,synthesis='dc',
            protected_roots=[],width=8,roots=96,cuts=96,rounds=4,splits=[])),
        ('local_exact',24.,'best.v',dict(width=8,cuts=128,rounds=6,sampled_care=False,error_inclusion=False)),
        ('local_care',24.,'best.v',dict(width=8,cuts=128,rounds=6,sampled_care=True,error_inclusion=False)),
        # Recollect on the compressed best graph. Fit a screened beam of
        # support-pruned response tables, not just one- and two-cell templates.
        ('fitted_cut',32.,'selected.v',dict(bound=args.bound,train_limit=min(cap,dev),dev_limit=dev,
            width=10,roots=144,boundaries=2,prune_depth=3,cases=256,rounds=3,
            fresh_splits=[],diagnostic_splits=[])),
        # A local rewrite changes global mapping opportunities. Revisit the
        # representation portfolio once; don't permanently mark it stalled
        # using a parent graph's information.
        ('exact_frontier',45.,'best.v',dict(rounds=3,seed=args.seed+1)),
        ('local_care',26.,'best.v',dict(width=8,cuts=128,rounds=6,sampled_care=True,error_inclusion=False)),
        ('mapped_sat',14.,'best.v',dict(rounds=2,wide=False,case_seconds=3.)),
        ('local_exact',24.,'best.v',dict(width=8,cuts=128,rounds=6,sampled_care=False,error_inclusion=False)),
    ]
    if getattr(args,'quality_fit_first',False):
        # This operator was also part of the successful historical prefix;
        # running it only after all mapping can lose a different ER function.
        schedule.insert(0,('fitted_cut',36.,'selected.v',dict(bound=args.bound,
            train_limit=min(cap,dev),dev_limit=dev,width=10,roots=144,boundaries=2,
            prune_depth=3,cases=512,rounds=4,fresh_splits=[],diagnostic_splits=[])))
    # Agent may select a tool family or the same COMPLETE package used by
    # MCTS. Defaults preserve the already verified MCTS schedule exactly.
    selected=getattr(args,'quality_operator_filter',None)
    if selected is not None:
        allowed={'exact_frontier','observable','local_care','mapped_sat','budget_care','local_exact','fitted_cut'}
        if not selected or not set(selected)<=allowed:raise ValueError('Unknown physical operator family')
        schedule=[stage for stage in schedule if stage[0] in selected]
    return schedule


def operator_signature(name,options):
    return hashlib.sha256(json.dumps([name,options],sort_keys=True).encode()).hexdigest()


def extend(w,folder,seconds,args):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    tick=time.perf_counter();deadline=tick+min(seconds,w.remaining())
    cap=w.metric.limit;dev=development_limit(w.metric,w.args)
    initial=w.current;current=initial;events=[];stage_rows=[]
    initial.write(folder/'input.v')
    common=dict(design=args.design,baseline=str(args.baseline))
    schedule=operator_schedule(args,cap,dev,w.count)
    import research_c880_exact_frontier as exact
    import research_c880_observable_exact as observable
    import research_c880_local_care_sat as care
    import research_c880_mapped_sat as sat
    import research_c880_budget_care_sat as budget
    import research_c880_cut_abc as local
    import research_c880_loss_cut as fitted
    tools=dict(exact_frontier=exact,observable=observable,local_care=care,
               mapped_sat=sat,budget_care=budget,local_exact=local,fitted_cut=fitted)
    for index,(name,allowance,filename,options) in enumerate(schedule):
        left=min(deadline-time.perf_counter(),w.remaining())
        if left<4:break
        work=folder/('%02d_%s'%(index,name));input_path=folder/('input_%02d.v'%index)
        current.write(input_path)
        input_key=sha(input_path);signature=operator_signature(name,options)
        coverage=getattr(args,'quality_no_gain_coverage',None)
        if coverage is not None and signature in coverage.get(input_key,[]):
            stage_rows.append(dict(operator=name,initial_area=current.lib.area(current.units),proposed_area=current.lib.area(current.units),
                accepted=False,seconds=0.,skipped_verified_no_gain=True,input_sha256=input_key,operator_signature=signature))
            stats_key='quality_operator_cache_hits';w.stats[stats_key]+=1
            continue
        proposal=SimpleNamespace(**common,input=str(input_path),output=str(work),
                                 seconds=min(allowance,left-1),**options)
        started=time.perf_counter()
        print('[QUALITY OPERATOR START]',json.dumps(dict(operator=name,parent_area=current.lib.area(current.units),seconds=proposal.seconds)),flush=True)
        tools[name].run(proposal)
        report=json.loads((work/'report.json').read_text(encoding='utf-8'))
        candidate=load(work/filename,current.lib)
        errors=search_errors(w,candidate)
        admissible=errors[0]<=cap and errors[1]<=dev
        row=dict(operator=name,initial_area=current.lib.area(current.units),
                 proposed_area=current.lib.area(candidate.units),errors=errors,
                 caps=[cap,dev],seconds=time.perf_counter()-started,
                 accepted=admissible and candidate.units<current.units,
                 approximate_operator=name in ('budget_care','fitted_cut'),stats=report.get('stats',{}),
                 report=str(work/'report.json'))
        row.update(input_sha256=input_key,operator_signature=signature,
            completed_configured_beam=not report.get('stats',{}).get('batch_timeout') and report.get('seconds',proposal.seconds)<proposal.seconds-2.)
        if coverage is not None and row['completed_configured_beam'] and not row['accepted']:
            bucket=coverage.setdefault(input_key,[])
            if signature not in bucket:bucket.append(signature)
        # Proxy truth/cell scores, full Monte Carlo calls and root-flip
        # propagations retain their own names; they are never lumped together.
        for key,value in report.get('stats',{}).items():
            if isinstance(value,(int,float)):w.stats['quality_'+key]+=value
        w.times['quality_'+name]+=row['seconds']
        if name in ('exact_frontier','mapped_sat'):
            w.stats['quality_mapping_cases']+=len(report.get('rows',[]))
            w.stats['quality_mapping_cec_proofs']+=sum(bool(r.get('cec_equivalent')) for r in report.get('rows',[]))
        stage_rows.append(row)
        if row['accepted']:
            current=candidate;w.current=current;w.loss=errors[0]
            w.lowest_units=min(w.lowest_units,current.units)
            w.seen.add(structural_id(current));w.preserve=False;w.stalls=0
            w.cached=None;w.bit_cache=None;w.verified_pairs={}
            w.training_view=None;w.development_view=None
            start=len(w.history);w.checkpoint(getattr(args,'quality_checkpoint_prefix','mcts_quality_')+name,[]);w.annotate(start)
            events.append(row)
        print('[QUALITY OPERATOR RESULT]',json.dumps({k:v for k,v in row.items() if k not in ('report','stats')}),flush=True)
        save(folder/'stages.json',stage_rows)
    result=dict(parent_area=initial.lib.area(initial.units),area=current.lib.area(current.units),
        accepted=current.units<initial.units,events=events,stages=stage_rows,
        seconds=time.perf_counter()-tick,error_caps=[cap,dev],api_calls=0,
        historical_netlist_imports=False,complete_lac_enumeration=False,
        search_groups=[0,1],vectors_per_group=w.count)
    save(folder/'report.json',result);return result
