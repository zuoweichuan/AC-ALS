"""Additive joint-plan operator for the original ERWorker kernel.

Fresh priors on a selected stalled current graph, bounded fitted LACs and
large-care cuts, one joint application, original train/development admission,
then the original proven exact remapper. Never imports an old approximate run.
"""
import time
from types import SimpleNamespace
from bacs_numeric import cleanup,save,number
from simulate import trace,outputs
from hybrid_identity import structural_id
from er_repair_tools import error_totals,error_words,development_limit
from research_c880_recollect_batches import collect_physical,plan_batch
from research_c880_plan_search import losses
from batch_wide import candidates as wide_candidates


def extend(worker,folder,seconds,args):
    folder.mkdir(parents=True,exist_ok=False);tick=time.perf_counter()
    deadline=min(tick+seconds,tick+worker.remaining())
    c=worker.current;graph=structural_id(c)
    cap=worker.metric.limit;dev_cap=development_limit(worker.metric,worker.args)
    baseline=[worker.response_context(k)['bad'].bit_count() for k in (0,1)]
    result=dict(parent_graph=graph,parent_area=c.lib.area(c.units),search_errors_before=baseline,
                error_caps=[cap,dev_cap],accepted=False,seconds_budget=seconds)
    if baseline[0]>cap or baseline[1]>dev_cap:
        result['reason']='native_guard_infeasible_use_original_repair';save(folder/'report.json',result);return result
    stats=worker.stats
    info=collect_physical(c,worker.inputs[:2],[worker.er_reference,worker.er_development_reference],
                          worker.count,worker.mapper,stats,cap)
    result['recollected_nodes']=len(info['records'])
    if time.perf_counter()<deadline-1:
        extra=wide_candidates(info,folder/'wide',min(cap,dev_cap),min(4.,deadline-time.perf_counter()),stats)
        info['extra_actions']=extra
        info['root_order']=list(dict.fromkeys([*(a.target for a in extra),*info['root_order']]))
    p=SimpleNamespace(cap=cap,error_caps=(cap,dev_cap),near_ratio=worker.args.near_ratio,
        area_debt=32,roots=min(32,len(info['root_order'])),wide_roots=8,divisors=16,
        batch_size=8,min_batch=1,plan_steps=12,delete_quota=6,branches=4,overflow_price=.15)
    state=plan_batch(info,p,worker.mapper,stats,deadline-(2. if seconds>=6 else 0.))
    save(folder/'plan_search.json',info.get('plan_log',[]))
    if state is None:result['reason']='screened_joint_plan_empty'
    else:
        raw=info['overlay'].commit(state['actions'])
        for k in (0,1):
            y=outputs(raw,trace(raw,worker.inputs[k],worker.count,stats))
            expected={n:state['values'][k][c.resolve(n)] for n in c.outputs}
            if y!=expected:raise AssertionError('Joint extension virtual/physical disagreement')
        trial=cleanup(raw,worker.mapper);errors=[]
        for k,ref in enumerate((worker.er_reference,worker.er_development_reference)):
            out=outputs(trial,trace(trial,worker.inputs[k],worker.count,stats))
            errors.append(error_totals(error_words(out,ref,worker.order))[0])
        result.update(proposed_area=trial.lib.area(trial.units),search_errors=errors,
                      lacs=len(state['actions']),virtual_errors=list(losses(state)))
        if errors!=list(losses(state)):raise AssertionError('Joint cleanup changed sampled function')
        if errors[0]>cap or errors[1]>dev_cap:raise AssertionError('Joint extension violated original ER guards')
        # Price the shortlisted COMPLETE plan after the same CEC-proven exact
        # remapper, before rejecting a raw area proxy. No other plan is applied.
        if worker.backend and time.perf_counter()<deadline-.5:
            mapping_start=time.perf_counter()
            trial=worker.backend.optimize(trial,lambda:min(worker.remaining(),max(0.,deadline-time.perf_counter())),
                                           variants=worker.args.map_variants)
            worker.times['exact_mapping']+=time.perf_counter()-mapping_start
            stats['mapping_calls']=len(worker.backend.rows)
            stats['mapping_cec_proofs']=sum(r.get('cec_equivalent',False) for r in worker.backend.rows)
            for k,ref in enumerate((worker.er_reference,worker.er_development_reference)):
                out=outputs(trial,trace(trial,worker.inputs[k],worker.count,stats))
                if error_totals(error_words(out,ref,worker.order))[0]!=errors[k]:
                    raise AssertionError('Proven joint remapping changed search errors')
        result['area_after_exact']=trial.lib.area(trial.units)
        if trial.units>=c.units:result['reason']='no_actual_area_gain'
        elif structural_id(trial) in worker.seen:result['reason']='known_physical_graph'
        else:
            worker.current=trial;worker.loss=errors[0];worker.lowest_units=min(worker.lowest_units,trial.units)
            worker.preserve=False;worker.stalls=0
            worker.seen.add(structural_id(trial));start=len(worker.history)
            worker.checkpoint('mcts_recollected_joint_plan_exact',[]);worker.annotate(start)
            worker.history[-1]['joint_plan_lacs']=len(state['actions'])
            worker.cached=None;worker.bit_cache=None;worker.verified_pairs={}
            worker.training_view=None;worker.development_view=None
            result.update(accepted=True,reason='measured_joint_gain',area_after_exact=worker.current.lib.area(worker.current.units))
    result['seconds']=time.perf_counter()-tick;save(folder/'report.json',result)
    return result
