"""Preserve mapped skeleton while SAT-checking observability-screened reuse.

Packed samples only screen cheap potential rewrites. They never prove
equivalence: each accepted whole circuit must pass independent ABC CEC.
"""
import runtime
import argparse,time,json
from pathlib import Path
from collections import defaultdict
from logic import Library
from bacs_data import load,LIBRARY_PATH
from bacs_numeric import cleanup,save
from simulate import trace,outputs,pack,plan_value
from frontier import forbidden_fanout,boundary_cuts
from netlist import ZERO,ONE
from bacs_frontier import VirtualArea
from cutmap import CutMapper
from research_c880_hybrid import proof
from hybrid_identity import structural_id
from prepare_resub_iscas import DEFAULT,sha
import er_data

def run(a):
    folder=Path(a.output).resolve();folder.mkdir(parents=True,exist_ok=False)
    (folder/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    lib=Library(LIBRARY_PATH);source=load(a.input,lib);source.write(folder/'input.v');best=source
    design=getattr(a,'design','c880')
    reference=load(Path(getattr(a,'baseline',DEFAULT))/design/'accurate.v',lib) if a.error_inclusion else None
    save(folder/'protocol.json',dict(vars(a),input_sha256=sha(a.input),code_sha256=sha(__file__),
        search_splits=[0,1],vectors_per_group=10000,approximate_changes=False,
        equivalence_acceptance=('whole-input error-inclusion CEC miter' if a.error_inclusion else 'whole-netlist CEC'),
        sample_equality_is_only_a_screen=True,api_calls=0))
    tick=time.perf_counter();mapper=CutMapper(lib,a.width);stats=defaultdict(int);events=[];mask=(1<<10000)-1;proofs=0
    inputs=[pack(er_data.matrix(design,source.inputs,k),source.inputs) for k in (0,1)]
    allowed=[]
    if a.error_inclusion:
        for p in inputs:
            original=outputs(source,trace(source,p,10000));gold=outputs(reference,trace(reference,p,10000));bad=0
            for o in source.outputs:bad |= original[o]^gold[o]
            allowed.append(bad)
    for iteration in range(8):
        if time.perf_counter()-tick>a.seconds-2:break
        values=[trace(best,p,10000,stats) for p in inputs];out=[outputs(best,v) for v in values]
        area=VirtualArea(best,mapper);proposals=[]
        roots=sorted(best.producer,key=lambda n:(-area.cones.potential(n),n))
        for target in roots:
            if time.perf_counter()-tick>a.seconds-2:break
            care=[]
            for k in (0,1):
                changed=outputs(best,trace(best,inputs[k],10000,stats,base=values[k],flip=target));bad=0
                for o in best.outputs:bad |= out[k][o]^changed[o]
                if a.error_inclusion:bad &= allowed[k]^mask
                care.append(bad)
            forbidden=forbidden_fanout(best,[target]);pool=[n for n in (*best.inputs,*best.producer) if n not in forbidden]
            rank=[]
            for n in pool:
                d=sum(((values[k][target]^values[k][n])&care[k]).bit_count() for k in (0,1))
                opposite=sum((((values[k][target]^values[k][n])^mask)&care[k]).bit_count() for k in (0,1))
                rank.append((min(d,opposite),n))
                for t,p in ((10,lib.mapper.plans[10]),(5,lib.mapper.plans[5])):
                    stats['single_signal_proxy_scores']+=1
                    if (d if t==10 else opposite)==0:
                        gain=area.gain(target,(n,),p)
                        if gain>0:proposals.append((gain,target,(n,),t,p))
            # Small support beam from the strongest four anchors and twelve
            # partners; finite two-input real-cell templates, no LAC product.
            ranked=[n for _,n in sorted(rank)[:a.pool]];seen=set();pair_quality=[]
            for n in ranked[:a.anchors]:
                for m in ranked:
                    names=tuple(sorted((n,m)))
                    if n==m or names in seen:continue
                    seen.add(names)
                    words={t:[plan_value(p,names,lib,values[k],mask) for k in (0,1)] for t,p in lib.mapper.plans.items()}
                    pair_quality.append((min(sum(((ww[k]^values[k][target])&care[k]).bit_count() for k in (0,1)) for ww in words.values()),names))
                    for t,ww in words.items():
                        stats['pair_template_proxy_scores']+=1
                        if any((ww[k]^values[k][target])&care[k] for k in (0,1)):continue
                        p=lib.mapper.plans[t];gain=area.gain(target,names,p)
                        if gain>0:proposals.append((gain,target,names,t,p))
            if a.wide:
                # Construct a small beam of supports; infer only care-consistent
                # tables, retaining exact real-cell don't-care completions.
                g=best.producer[target];cell=lib.cells[g.cell]
                direct=tuple(sorted({best.resolve(g.pins[p]) for p in cell.inputs}-{ZERO,ONE}))
                supports=list(boundary_cuts(best,target,width=a.width))
                if 0<len(direct)<=a.width:supports.append(direct)
                beam=[ns for _,ns in sorted(pair_quality)[:a.beam]]
                pool=list(dict.fromkeys([*direct,*ranked]))[:a.pool+4]
                cache={}
                def summary(ns):
                    if ns in cache:return cache[ns]
                    z=o=0
                    for k in (0,1):
                        regions=[care[k]]
                        for n in ns:
                            v=values[k][n];regions=[r&(v^mask) for r in regions]+[r&v for r in regions]
                        gold=values[k][target]
                        for row,r in enumerate(regions):
                            if r&gold:o |= 1<<row
                            if r&(gold^mask):z |= 1<<row
                    cache[ns]=(z&o).bit_count(),o,z|o;stats['wide_care_support_scores']+=1
                    return cache[ns]
                for width in range(3,a.width+1):
                    choices={tuple(sorted((*ns,n))) for ns in beam for n in pool if n not in ns}
                    beam=sorted(choices,key=lambda ns:(summary(ns)[0],ns))[:a.beam];supports+=beam
                for names in dict.fromkeys(supports):
                    n=len(names)
                    if not 1<=n<=a.width:continue
                    conflicts,truth,defined=summary(names)
                    if conflicts:continue
                    plans=[(p,t) for t,p in mapper.catalog[n].items() if (t&defined)==(truth&defined)]
                    if not plans and n<=4:
                        m=(1<<(1<<n))-1
                        plans=[(mapper.map(n,t),t) for t in (truth,truth|(defined^m))]
                    if not plans:continue
                    plan,t=min(plans,key=lambda r:(mapper.cost(r[0]),r[1]))
                    stats['wide_care_function_scores']+=1;gain=area.gain(target,names,plan)
                    if gain>0:proposals.append((gain,target,names,t,plan))
        proposals.sort(key=lambda r:(-r[0],r[1],r[2],r[3]));trials=[];seen=set()
        for gain,target,names,truth,plan in proposals:
            if time.perf_counter()-tick>a.seconds-1:break
            trial=cleanup(best.materialize([(target,names,plan)]),mapper);stats['actual_materializations']+=1
            key=structural_id(trial)
            if trial.units>=best.units or key in seen:continue
            seen.add(key);trials.append((trial,dict(target=target,inputs=names,truth=truth,proxy_gain=gain)))
            if len(trials)>=a.trials:break
        trials.sort(key=lambda row:(row[0].units,json.dumps(row[1],sort_keys=True)))
        accepted=None
        for trial,event in trials:
            if time.perf_counter()-tick>a.seconds-1:break
            if a.error_inclusion:
                from research_error_relation import prove_inclusion
                same=prove_inclusion(reference,source,trial,folder/('cec_%04d'%proofs))
            else:same=proof(source,trial,folder/('cec_%04d'%proofs))
            proofs+=1;stats['global_cec_attempts']+=1
            if not same:stats['sample_aliases_rejected_by_cec']+=1;continue
            accepted=(trial,event);break
        if accepted is None:break
        best,event=accepted;events.append(dict(event,area=lib.area(best.units),cec=True));best.write(folder/'best.v')
        print('[OBSERVABLE EXACT ACCEPT] '+json.dumps(events[-1]),flush=True)
    best.write(folder/'best.v');result=dict(initial_area=lib.area(source.units),area=lib.area(best.units),
        events=events,stats=dict(stats),seconds=time.perf_counter()-tick,approximate_changes=a.error_inclusion,
        proved_error_inclusion=a.error_inclusion,api_calls=0)
    save(folder/'report.json',result);print('[OBSERVABLE EXACT RESULT] '+json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seconds',type=float,default=60);p.add_argument('--trials',type=int,default=24)
    p.add_argument('--wide',action='store_true');p.add_argument('--width',type=int,choices=(4,5,6),default=4)
    p.add_argument('--pool',type=int,default=16);p.add_argument('--anchors',type=int,default=4);p.add_argument('--beam',type=int,default=2)
    p.add_argument('--error-inclusion',action='store_true',help='Permit function changes only with whole-input vector-error inclusion proof')
    run(p.parse_args())
