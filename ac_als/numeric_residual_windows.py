"""Experimental scale-aware numerical compensation on arbitrary output windows.

Existing numeric/ER entries remain unchanged. A bounded set of windows is chosen
from numerical residual scale, not circuit names. Selected truth tables minimize
L1 loss analytically; only shortlisted functions are physically materialized.
"""
import runtime
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np

from bacs_data import LIBRARY_PATH, preflight, load, matrix
from bacs_numeric import Metric, bits, number, number_numpy, save,cleanup
from hybrid_identity import structural_id
from logic import Library
from netlist import ZERO,ONE
from numeric_agentic import Worker,sha,clone_worker
from numeric_joint_word import detached_word,detached_wires,actual_trials,parser as base_parser
from simulate import pack,trace,outputs,validate_numpy


def scaled_l1_table(code,target,width,support,step):
    """Integer per-row L1 optimum for residual target with a power-of-two step."""
    cap=(1<<width)-1
    def optimum(values):
        median=float(np.median(values))/step
        lo=int(np.clip(np.floor(median),0,cap));hi=int(np.clip(np.ceil(median),0,cap))
        return min((lo,hi),key=lambda q:(int(np.abs(values-q*step).sum()),q))
    table=np.full(1<<support,optimum(target),dtype=np.int64)
    for row in range(1<<support):
        values=target[code==row]
        if len(values):table[row]=optimum(values)
    return table,table[code]*step


def choose_windows(base,reference,order,circuit):
    error=np.abs(base-reference);nonzero=error[error>0]
    scale=int(np.floor(np.log2(np.median(nonzero)))) if len(nonzero) else max(0,len(order)//2)
    scale=min(len(order)-1,max(0,scale))
    live=[i for i,out in enumerate(order) if circuit.resolve(out) not in (ZERO,ONE)]
    anchors=list(dict.fromkeys([scale,*[scale+d for d in (-2,2,-4,4)],*(live[:1]),*(live[-1:])]))
    result=[]
    for width in (1,2,3,4):
        for anchor in anchors:
            offset=max(0,min(len(order)-width,anchor-width//2))
            if width<=len(order) and (offset,width) not in result:result.append((offset,width))
    # Round-robin widths prevents low-width fitting consuming every budget.
    return [result[i] for i in sorted(range(len(result)),key=lambda i:(i%len(anchors),i//len(anchors)))][:16]


def economic_window_tables(code,target,step,width,support,choice,limit,mapper):
    """Fit cheap real-cell templates to row L1 costs, jointly across window bits.

    At most16 output labels per row are scored analytically. This does not
    enumerate truth tables, supports, or circuit replacement combinations.
    """
    costs=np.vstack([np.bincount(code,weights=np.abs(target-label*step),minlength=1<<support)
                     for label in range(1<<width)])
    rows=np.arange(1<<support);answers=[choice]
    for order in (tuple(range(width)),tuple(reversed(range(width)))):
        current=choice.copy()
        for bit in order:
            low=current&~(1<<bit);high=low|(1<<bit);c0=costs[low,rows];c1=costs[high,rows]
            rank=[]
            for truth,plan in mapper.catalog[support].items():
                b=np.array([(truth>>row)&1 for row in rows],dtype=np.int64)
                loss=int(np.where(b,c1,c0).sum())
                if loss<=limit:rank.append((mapper.cost(plan),loss,truth,b))
            if rank:
                _,_,_,b=min(rank,key=lambda q:q[:3]);current=low|(b<<bit)
                answers.append(current.copy())
    return list({tuple(map(int,a)):a for a in answers}.values())


def significant_wire_proposals(w):
    """Coordinate L1 compensation using all output bits, including constant POs."""
    c=w.current;n=w.count;values=trace(c,w.inputs[0],n,w.stats)
    base=number(outputs(c,values),w.order,n);ref=w.reference[0];pred=base.copy();bindings={}
    pool=list(dict.fromkeys([ZERO,ONE,*c.inputs,*c.producer]));sample=np.arange(0,n,max(1,n//256))[:256]
    decoded={s:bits(values[s],n).astype(np.int64) for s in pool};initial=int(np.abs(base-ref).sum())
    # The high-impact numerical positions are checked first, but no bit index
    # is hard-coded. A full pass is cheap relative to root-resimulation.
    importance=[]
    for bit in range(len(w.order)):
        high=base-((base>>bit)&1)*(1<<bit)
        gain=initial-min(int(np.abs(high-ref).sum()),int(np.abs(high+(1<<bit)-ref).sum()))
        importance.append((gain,bit))
    order=[i for _,i in sorted(importance,reverse=True)]
    for sweep in range(3):
        changed=False
        for bit in order:
            if w.remaining()<1:break
            step=1<<bit;high=pred-((pred>>bit)&1)*step
            ranked=sorted(pool,key=lambda s:(int(np.abs(high[sample]+decoded[s][sample]*step-ref[sample]).sum()),c.level.get(s,0),s))[:12]
            best_loss=int(np.abs(pred-ref).sum());best=None
            for s in ranked:
                trial=high+decoded[s]*step;loss=int(np.abs(trial-ref).sum());w.stats['significant_wire_function_scores']+=1
                if loss<best_loss:best_loss=loss;best=s,trial
            if best:
                bindings[w.order[bit]]=best[0];pred=best[1];changed=True
        if not changed:break
    if not bindings:return []
    trial=detached_wires(c,bindings,w.mapper);loss=int(np.abs(pred-ref).sum())
    if loss>=initial or trial.units>c.units+16:return []
    return [(c.units-trial.units,loss,trial,dict(output_wires=bindings,scope='all_bit_positions'),pred)]


def residual_window_proposals(w,repair=True):
    c=w.current;n=w.count;values=trace(c,w.inputs[0],n,w.stats)
    base=number(outputs(c,values),w.order,n);ref=w.reference[0];initial=int(np.abs(base-ref).sum())
    raw=list(dict.fromkeys([*c.inputs,*c.producer]));decoded={s:bits(values[s],n).astype(np.int64) for s in raw}
    sample=np.arange(0,n,max(1,n//256))[:256];proposals=[]
    for offset,width in choose_windows(base,ref,w.order,c):
        if w.remaining()<1:break
        step=1<<offset;mask=(1<<width)-1;high=base-((base>>offset)&mask)*step;target=ref-high
        singles=[]
        for s in raw:
            _,p=scaled_l1_table(decoded[s][sample],target[sample],width,1,step)
            w.stats['window_support_proxy_scores']+=1
            singles.append((int(np.abs(p-target[sample]).sum()),c.level.get(s,0),s))
        pools=[]
        for mode in ('all','shallow'):
            pools.append((mode,[s for _,_,s in sorted(singles) if mode=='all' or c.level.get(s,0)<=1][:16]))
        for mode,pool in pools:
            names=();code=np.zeros(n,dtype=np.int64)
            for size in range(1,5):
                if w.remaining()<1:break
                best=None
                for s in pool:
                    if s in names:continue
                    cd=code|(decoded[s]<<(size-1));table,pred=scaled_l1_table(cd,target,width,size,step)
                    loss=int(np.abs(pred-target).sum());w.stats['window_truth_function_scores']+=1
                    key=(loss,c.level.get(s,0),s)
                    if best is None or key<best[0]:best=key,s,cd,table,pred,loss
                if best is None:break
                _,s,code,table,pred,loss=best;names=(*names,s)
                if loss>w.metric.limit or (repair and loss>=initial):continue
                # Only the chosen support is synthesized. Default mapper handles
                # at most four support signals and shares identical bit plans.
                tables=economic_window_tables(code,target,step,width,size,table,w.metric.limit,w.mapper) if (
                    not repair and getattr(w.args,'economic_windows',False)) else [table]
                for fitted in tables:
                    fitted_pred=fitted[code]*step;fitted_loss=int(np.abs(fitted_pred-target).sum())
                    if fitted_loss>w.metric.limit:continue
                    trial=detached_word(c,w.order[offset:offset+width],width,names,fitted,w.mapper)
                    w.stats['window_actual_materializations']+=1
                    if (repair and trial.units<=c.units+16) or (not repair and trial.units<c.units):
                        proposals.append((c.units-trial.units,fitted_loss,trial,
                            dict(offset=offset,width=width,support=names,support_mode=mode,table=fitted.tolist()),high+fitted_pred))
    unique={structural_id(q[2]):q for q in proposals}
    if repair:return sorted(unique.values(),key=lambda q:(-(initial-q[1])/max(1,1-q[0]),q[1]))[:12]
    return sorted(unique.values(),key=lambda q:(-q[0],q[1]))[:12]


def near_wire_proposals(w):
    """Six screened near branches plus global wire compensation, no parent edits."""
    original=w.current;parent_graph=structural_id(original);moves=w.scan()['moves'];limit=w.metric.limit
    near=[m for m in moves if m.gain>0 and limit<m.error<=int(limit*w.args.near_ratio)]
    queues=[sorted(near,key=lambda m:(-m.gain,m.error)),
            sorted(near,key=lambda m:((m.error-limit)/max(1,m.gain),-m.gain,m.error))]
    chosen=[];used=set()
    for rank in range(6):
        for queue in queues:
            if rank<len(queue):
                m=queue[rank];key=(m.target,m.names,m.truth)
                if key not in used:chosen.append(m);used.add(key)
            if len(chosen)==6:break
        if len(chosen)==6:break
    root=w.folder.parent/'near_candidates';root.mkdir(exist_ok=True);result=[];details=[]
    for m in chosen:
        if w.remaining()<3:break
        c=cleanup(original.materialize([m.replacement()]),w.mapper)
        y=number(outputs(c,trace(c,w.inputs[0],w.count,w.stats)),w.order,w.count)
        total,_=w.metric.exact(y,w.reference[0]);w.stats['near_global_child_verifications']+=1
        if total!=m.error:raise AssertionError('Near root response prediction mismatch')
        if c.units>=original.units:continue
        folder=root/('branch_%04d'%len(list(root.glob('branch_*'))))
        deadline=time.perf_counter()+min(10.,w.remaining()-1)
        remaining=lambda:max(0.,min(w.remaining(),deadline-time.perf_counter()))
        child=clone_worker(w,folder,remaining,time.perf_counter(),c,total,False);events=[]
        detail=dict(action=m.record(),branch_area=c.lib.area(c.units),branch_nmed=total/w.count/w.metric.denominator,
                    folder=str(folder),repaired=False)
        plans=significant_wire_proposals(child)
        ok=actual_trials(child,plans,'repair',w.inputs[1],w.reference[1],events)
        if ok:
            detail['repaired']=True
            for policy in ('balanced','area'):
                profile='batch' if policy=='balanced' else policy
                if child.remaining()<1:break
                if (structural_id(child.current),profile,1.) not in child.blocked:child.compress(policy,1.)
            out=number(outputs(child.current,trace(child.current,w.inputs[0],w.count,child.stats)),w.order,w.count)
            loss,_=child.metric.exact(out,child.reference[0])
            dev=number(outputs(child.current,trace(child.current,w.inputs[1],w.count)),w.order,w.count)
            dev_loss,_=child.metric.exact(dev,child.reference[1]);child.stats['near_branch_development_propagations']+=1
            detail.update(repaired_area=c.lib.area(child.current.units),training_nmed=loss/w.count/w.metric.denominator,
                          development_nmed=dev_loss/w.count/w.metric.denominator)
            if child.current.units<original.units and max(loss,dev_loss)<=limit:
                result.append((original.units-child.current.units,loss,child.current,
                    dict(near_action=m.record(),repair_events=events,branch_folder=str(folder)),out))
        w.stats['near_branch_mapping_calls']+=len(child.backend.rows)
        w.stats['near_branch_mapping_cec']+=sum(r.get('cec_equivalent',False) for r in child.backend.rows)
        for key,value in child.stats.items():
            if key not in ('mapping_calls','mapping_cec_proofs'):w.stats[key]+=value
        details.append(detail);save(root/'attempts.json',details)
    if structural_id(w.current)!=parent_graph:raise AssertionError('Near child mutated parent')
    return sorted(result,key=lambda q:(-q[0],q[1]))[:6]


def coarse_numeric_proposals(w):
    """Rebuild the entire numerical word on <=6 response-selected PI supports.

    Output coefficient quantization removes noisy low truth bits; a shared
    conditional median bias uses the formal numerical error scale. Supports
    grow greedily with four full-data finalists, not a combinatorial search.
    """
    c=w.current;n=w.count;reference=w.reference[0];width=len(w.order)
    values=trace(c,w.inputs[0],n,w.stats)
    raw=list(c.inputs);decoded={s:bits(values[s],n).astype(np.int64) for s in raw}
    sample=np.arange(0,n,max(1,n//512))[:512]
    names=();code=np.zeros(n,dtype=np.int64);result=[]
    scale=max(0,int(np.floor(np.log2(max(1,w.metric.limit/n)))))
    shifts=list(dict.fromkeys(max(0,min(width-1,scale+d)) for d in (0,-1,1,-2,2)))
    for support in range(1,7):
        if w.remaining()<2:break
        proxies=[]
        for s in raw:
            if s in names:continue
            cd=code[sample]|(decoded[s][sample]<<(support-1))
            _,pred=scaled_l1_table(cd,reference[sample],width,support,1)
            proxies.append((int(np.abs(pred-reference[sample]).sum()),s));w.stats['coarse_support_proxy_scores']+=1
        finalists=[]
        for _,s in sorted(proxies)[:4]:
            cd=code|(decoded[s]<<(support-1));_,pred=scaled_l1_table(cd,reference,width,support,1)
            finalists.append((int(np.abs(pred-reference).sum()),s,cd));w.stats['coarse_full_truth_scores']+=1
        if not finalists:break
        _,s,code=min(finalists,key=lambda q:q[:2]);names=(*names,s)
        models={}
        if getattr(w.args,'polynomial',False):
            for shift in shifts:
                models[shift]=rounded_quadratic_table(code,reference,support,shift,width)
        for shift in shifts:
            if w.remaining()<2:break
            step=1<<shift;bias=step//2
            for _ in range(2):
                table,pred=scaled_l1_table(code,reference-bias,width-shift,support,step)
                bias=int(np.clip(np.median(reference-pred),0,step-1))
            variants=[('median',table*step+bias)]
            if shift in models:variants.append(('rounded_quadratic',models[shift]))
            for model,numeric_table in variants:
                pred=numeric_table[code]
                loss=int(np.abs(pred-reference).sum());w.stats['coarse_quantization_scores']+=1
                if loss>w.metric.limit:continue
                trial=detached_word(c,w.order,width,names,numeric_table,w.mapper)
                w.stats['coarse_actual_materializations']+=1
                result.append((c.units-trial.units,loss,trial,
                    dict(support=names,quantization_shift=shift,shared_bias=bias,model=model,numeric_table=numeric_table.tolist()),pred))
    unique={structural_id(q[2]):q for q in result}
    return sorted(unique.values(),key=lambda q:(q[2].units,q[1]))[:12]


def rounded_quadratic_table(code,reference,support,shift,width):
    """Small degree-two model regularizes sampling noise in numerical LUT rows."""
    def features(ids):
        x=[((ids>>i)&1).astype(np.float64) for i in range(support)]
        return np.column_stack([np.ones(len(ids)),*x,*[x[i]*x[j] for i in range(support) for j in range(i)]])
    design=features(code);scale=float(1<<width)
    coefficients=np.linalg.lstsq(design,reference/scale,rcond=None)[0]*scale
    step=1<<shift;quantized=np.rint(coefficients/step)*step
    rows=features(np.arange(1<<support,dtype=np.int64))
    table=np.clip(np.rint(rows@quantized),0,(1<<width)-1).astype(np.int64)
    return table


def run(a):
    folder=Path(a.output).resolve()
    if False:raise ValueError('D: only')
    folder.mkdir(parents=True,exist_ok=False);started=time.perf_counter();lib=Library(LIBRARY_PATH)
    source,order,audit=preflight(a.design,lib);baseline=load(a.accurate_netlist,lib) if a.accurate_netlist else source
    start=load(a.start,lib) if a.start else baseline
    inputs=[pack(matrix(a.design,source.inputs,i),source.inputs) for i in (0,1)]
    refs=[number(outputs(source,trace(source,x,10000)),order,10000) for x in inputs]
    for inp,ref in zip(inputs,refs):
        if not np.array_equal(number(outputs(baseline,trace(baseline,inp,10000)),order,10000),ref):
            raise ValueError('Accurate baseline mismatch')
    if start.inputs!=source.inputs or start.outputs!=source.outputs:raise ValueError('Start interface mismatch')
    snapshot=folder/'source';snapshot.mkdir();hashes={}
    for module in tuple(sys.modules.values()):
        p=Path(getattr(module,'__file__','') or '.')
        if p.suffix=='.py' and p.resolve().parent==Path(__file__).resolve().parent:
            shutil.copy2(p,snapshot/p.name);hashes[p.name]=sha(p)
    save(folder/'protocol.json',dict(args=vars(a),source_hashes=hashes,audit=audit,
        warm_start=bool(a.start),start_sha256=sha(a.start) if a.start else None,
        api_calls=0,selection_splits=[0,1],final_splits=a.final_splits,exhaustive_lacs=False))
    remaining=lambda:max(0.,a.seconds-(time.perf_counter()-started))
    (folder/'worker').mkdir();w=Worker(start,order,inputs,refs,a,folder/'worker',remaining=remaining,reference_source=baseline)
    metric=Metric(a.metric,a.bound,order);events=[];rounds=[]
    for iteration in range(a.rounds):
        if remaining()<1:break
        before=structural_id(w.current)
        for kind in a.tools:
            if remaining()<1:break
            print('[RESIDUAL TOOL]',json.dumps(dict(iteration=iteration,kind=kind,area=lib.area(w.current.units),remaining=remaining())),flush=True)
            if kind=='native':
                for policy in ('balanced','area','balanced'):
                    profile='batch' if policy=='balanced' else policy
                    if remaining()<1:break
                    if (structural_id(w.current),profile,1.) in w.blocked:continue
                    w.compress(policy,1.)
            else:
                plans=(significant_wire_proposals(w) if kind=='wire' else near_wire_proposals(w) if kind=='near_wire'
                       else coarse_numeric_proposals(w) if kind=='coarse'
                       else residual_window_proposals(w,repair=kind=='repair_window'))
                actual_trials(w,plans,'repair' if kind in ('wire','repair_window') else kind,inputs[1],refs[1],events)
        rounds.append(dict(iteration=iteration,area=lib.area(w.current.units),loss=w.loss,seconds=time.perf_counter()-started))
        save(folder/'events.json',events);save(folder/'trajectory.json',rounds)
        if structural_id(w.current)==before:break
    eligible=[]
    for row in w.history:
        if row['training_loss']>metric.limit:continue
        candidate=load(w.folder/row['path'],lib)
        y=number(outputs(candidate,trace(candidate,inputs[1],10000)),order,10000)
        w.stats['selection_development_propagations']+=1;loss,_=metric.exact(y,refs[1])
        if loss<=metric.limit:eligible.append((candidate.units,max(row['training_loss'],loss),row['path'],candidate))
    if not eligible:raise RuntimeError('No feasible checkpoint')
    selected=min(eligible,key=lambda q:q[:3]);winner=selected[3];winner.write(folder/'selected_before_checks.v')
    digest=sha(folder/'selected_before_checks.v')
    save(folder/'frozen_selection.json',dict(path=selected[2],sha256=digest,area=lib.area(winner.units),held_out_read=False))
    checks=[]
    for split in a.final_splits:
        x=matrix(a.design,source.inputs,split);inp=pack(x,source.inputs)
        y=number(outputs(winner,trace(winner,inp,10000)),order,10000);ref=number(outputs(source,trace(source,inp,10000)),order,10000)
        if not np.array_equal(y,number_numpy(validate_numpy(winner,x),winner.outputs,order)):raise AssertionError('NumPy mismatch')
        if not np.array_equal(ref,number_numpy(validate_numpy(source,x),source.outputs,order)):raise AssertionError('Reference mismatch')
        loss,value=metric.exact(y,ref);checks.append(dict(split=split,count=10000,nmed=value,passed=loss<=metric.limit))
    passed=all(r['passed'] for r in checks);winner.write(folder/('best.v' if passed else 'failed_final.v'))
    for name,h in hashes.items():
        if sha(Path(__file__).resolve().parent/name)!=h:raise ValueError('Changed source: '+name)
    result=dict(design=a.design,bound=a.bound,baseline_area=lib.area(baseline.units),starting_area=lib.area(start.units),
        area=lib.area(winner.units),reduction=1-winner.units/baseline.units,strict_pass=passed,checks=checks,
        whole_seconds=time.perf_counter()-started,warm_start=bool(a.start),api_calls=0,stats=dict(w.stats),
        mapping_calls=len(w.backend.rows) if w.backend else 0,mapping_cec=sum(r.get('cec_equivalent',False) for r in w.backend.rows) if w.backend else 0,
        selected_sha256=digest,events=events,trajectory=rounds)
    save(folder/'report.json',result)
    print('[RESIDUAL RESULT]',json.dumps({k:v for k,v in result.items() if k not in ('events','trajectory')}),flush=True)
    return 0 if passed else 2


def parser():
    p=base_parser()
    next(arg for arg in p._actions if arg.dest=='tools').choices=('wire','repair_window','compress_window','native','near_wire','coarse')
    p.set_defaults(tools=['wire','native','repair_window','native','compress_window','native'],seconds=120.,rounds=3)
    p.add_argument('--final-splits',type=int,nargs='+',default=[1600,1601])
    p.add_argument('--economic-windows',action='store_true')
    p.add_argument('--polynomial',action='store_true')
    return p


if __name__=='__main__':
    a=parser().parse_args()
    if a.metric!='nmed' or a.seconds<=0 or a.seconds>600 or set(a.final_splits)&{0,1}:raise ValueError('Use bounded NMED exploration')
    raise SystemExit(run(a))
