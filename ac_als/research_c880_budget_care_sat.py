"""Bounded error-budgeted external-care synthesis on mapped local cuts.

No enumeration of LAC truth functions. Two greedy rare-response masks and at
most two screened cofactor masks are synthesized per bounded structural cut.
Every cut input is upstream of its root. Predicted vector errors must equal
actual full-circuit errors on both search groups before a move is accepted.
Final groups are diagnostic only, after freezing the selected physical file.
"""
from portable_tools import tool_command
import runtime
import argparse,json,time,subprocess
from pathlib import Path
from collections import Counter
import numpy as np
from logic import Library
from netlist import Circuit
from bacs_data import load,LIBRARY_PATH
from bacs_numeric import save,cleanup
from bacs_frontier import VirtualArea
from cutmap import CutMapper,variable
from numeric_exact import ExactBackend,restore_ports
from research_c880_local_exact import cuts,cut_truth
from research_c880_local_care_sat import blif_truth,to_plan
from simulate import trace,outputs,pack,validate_numpy
from prepare_resub_iscas import DEFAULT,sha,verify_bundle
import er_data


def free_rows(risks,budgets,allowed=None):
    """Free whole truth rows while upper-bounding newly wrong vectors/group."""
    n=len(risks);allowed=range(n) if allowed is None else allowed
    order=sorted(allowed,key=lambda i:(max(risks[i][k]/max(1,budgets[k]) for k in range(2)),sum(risks[i]),i))
    used=[0,0];freed=0
    for i in order:
        if all(used[k]+risks[i][k]<=budgets[k] for k in range(2)):
            freed|=1<<i
            for k in range(2):used[k]+=risks[i][k]
    return freed,used


def cofactor_truth(truth,n,index,value):
    result=0
    for row in range(1<<n):
        old=(row&~(1<<index))|(value<<index)
        result|=((truth>>old)&1)<<row
    return result


def regions_for(names,values,mask):
    regions=[mask]
    for name in names:
        v=values[name];regions=[r&(v^mask) for r in regions]+[r&v for r in regions]
    return regions


def wrong(y,gold):
    bad=0
    for name in gold:bad|=y[name]^gold[name]
    return bad


def run(a):
    folder=Path(a.output).resolve()
    if False:raise ValueError('D: only')
    folder.mkdir(parents=True,exist_ok=False)
    (folder/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    design=getattr(a,'design','c880');baseline=Path(getattr(a,'baseline',DEFAULT))
    verify_bundle(baseline,design);lib=Library(LIBRARY_PATH);best=load(a.input,lib)
    source=best;ref=load(baseline/design/'accurate.v',lib);mapper=CutMapper(lib,4)
    caps=[round((a.training_cap if a.training_cap is not None else a.bound)*10000),round(a.development_cap*10000)];mask=(1<<10000)-1
    inputs=[pack(er_data.matrix(design,ref.inputs,k),ref.inputs) for k in (0,1)]
    gold=[outputs(ref,trace(ref,p,10000)) for p in inputs]
    save(folder/'protocol.json',dict(vars(a),input_sha256=sha(a.input),search_splits=[0,1],search_caps=caps,
        diagnostic_splits=a.splits,api_calls=0,selection='freeze before diagnostics; no final-based reselection',
        candidates='bounded structural cut beam, 2 greedy masks and 2 screened cofactor masks per cut; no LAC enumeration'))
    best.write(folder/'input.v');tick=time.perf_counter();stats=Counter();events=[]
    for iteration in range(a.rounds):
        if time.perf_counter()-tick>a.seconds-4:break
        work=folder/('round_%02d'%iteration);work.mkdir();area=VirtualArea(best,mapper)
        values=[trace(best,p,10000,stats) for p in inputs]
        oldbad=[wrong(outputs(best,v),g) for v,g in zip(values,gold)];loss=[b.bit_count() for b in oldbad]
        if any(loss[k]>caps[k] for k in (0,1)):
            stats['ineligible_parent']+=1;break
        available=[caps[k]-loss[k] for k in (0,1)];contexts={};short=[]
        protected_ancestors=set()
        def protect(n):
            n=best.resolve(n)
            if n in protected_ancestors:return
            g=best.producer.get(n)
            if g is None:return
            protected_ancestors.add(n)
            for pin in lib.cells[g.cell].inputs:protect(g.pins[pin])
        for root in a.protected_roots:protect(root)
        for target in sorted(best.producer,key=lambda n:(-area.cones.potential(n),n))[:a.roots]:
            if target in protected_ancestors:continue
            possible=[]
            for names in cuts(best,target,width=a.width):
                if not 3<=len(names)<=a.width:continue
                gain=area.cones.saved_with_inputs(target,names)
                if gain>=6:possible.append((gain,target,names))
            selected=[]
            for queue in (sorted(possible,key=lambda r:(-r[0],len(r[2]),r[2])),
                          sorted(possible,key=lambda r:(-r[0],-len(r[2]),r[2]))):
                for row in queue[:max(1,a.cuts_per_root//2)]:
                    if row not in selected:selected.append(row)
            if a.cuts_per_root>2:
                for width in range(3,a.width+1):
                    queue=sorted((r for r in possible if len(r[2])==width),key=lambda r:(-r[0],r[2]))
                    if queue and queue[0] not in selected:selected.append(queue[0])
                selected=selected[:a.cuts_per_root]
            short+=selected
        short=sorted(short,key=lambda r:(-r[0],r[1],r[2]))[:a.cuts]
        commands=['read_lib -w '+ExactBackend.quote(ExactBackend.linux(LIBRARY_PATH))];rows=[]
        for gain,target,names in short:
            if target not in contexts:
                flipbad=[wrong(outputs(best,trace(best,p,10000,stats,base=v,flip=target)),g) for p,v,g in zip(inputs,values,gold)]
                contexts[target]=flipbad
            flipbad=contexts[target];original=cut_truth(best,target,names);n=len(names);full=(1<<(1<<n))-1
            regions=[regions_for(names,v,mask) for v in values]
            teacher=original;teacher_loss=list(loss);flipped_rows=0
            if a.teacher=='optimal':
                # A change must improve both groups (or tie one), never rely
                # on a negative total hiding a degradation in the other group.
                for i in range(1<<n):
                    delta=[(regions[k][i]&flipbad[k]).bit_count()-(regions[k][i]&oldbad[k]).bit_count() for k in (0,1)]
                    if max(delta)<=0 and sum(delta)<0:
                        flipped_rows|=1<<i;teacher^=1<<i
                        for k in (0,1):teacher_loss[k]+=delta[k]
            risks=[]
            for i in range(1<<n):
                if flipped_rows>>i&1:
                    risks.append(tuple((regions[k][i]&oldbad[k]&(flipbad[k]^mask)).bit_count() for k in (0,1)))
                else:risks.append(tuple((regions[k][i]&flipbad[k]&(oldbad[k]^mask)).bit_count() for k in (0,1)))
            row_available=[caps[k]-teacher_loss[k] for k in (0,1)]
            zero=sum(1<<i for i,r in enumerate(risks) if not any(r));masks={}
            for fraction in (.45,1.):
                freed,used=free_rows(risks,[int(fraction*x) for x in row_available])
                masks.setdefault(freed,dict(kind='rare_row_budget',fraction=fraction,new_error_upper_bound=used))
            cofactors=[]
            for index in range(n):
                for value in (0,1):
                    alternate=cofactor_truth(teacher,n,index,value);delta=alternate^teacher
                    used=[sum(risks[i][k] for i in range(1<<n) if delta>>i&1) for k in (0,1)]
                    if delta and all(used[k]<=row_available[k] for k in (0,1)):
                        cofactors.append((sum(used),index,value,delta,used))
            for _,index,value,delta,used in sorted(cofactors)[:2]:
                masks.setdefault(zero|delta,dict(kind='screened_cofactor',index=index,value=value,new_error_upper_bound=used))
            stats['bounded_cut_masks_screened']+=len(masks)
            for freed,detail in masks.items():
                # Keep all non-freed rows of the exact original truth function.
                defined=full^freed;index=len(rows);case=work/('cut_%03d'%index);case.mkdir()
                blif_truth(case/'input.blif',n,teacher);blif_truth(case/'care.blif',n,defined)
                q=lambda name:ExactBackend.quote(ExactBackend.linux(case/name))
                mfs='mfs -a '+('-r' if a.synthesis=='dc' else '-e')+' -W 0 -F 300 -D 0 -M 1600 -L 8 -C 20000'
                commands.append('read_blif %s; strash; if -K 8 -a -C 64; care_set %s; %s; strash; dch; amap -F 4 -A 20 -C 2000 -m -s; write_verilog %s; write_blif %s'%(q('input.blif'),q('care.blif'),mfs,q('mapped.v'),q('mapped.blif')))
                rows.append(dict(index=index,target=target,names=names,original_truth=original,teacher_truth=teacher,teacher_loss=teacher_loss,care_truth=defined,potential=gain,detail=detail))
        if not rows:break
        script=work/'batch.abc';script.write_text('\n'.join(commands)+'\n',encoding='utf-8')
        try:
            p=subprocess.run(tool_command('/usr/local/bin/abc', '-c', 'source '+ExactBackend.quote(ExactBackend.linux(script)), distro='Ubuntu'),
                capture_output=True,timeout=max(1,a.seconds-(time.perf_counter()-tick)))
        except subprocess.TimeoutExpired:stats['batch_timeout']+=1;break
        log=(p.stdout+p.stderr).decode('utf-8',errors='replace');(work/'abc.log').write_text(log,encoding='utf-8')
        stats['local_synthesis_jobs']+=len(rows)
        if p.returncode or log.count('external care clauses')!=len(rows):raise RuntimeError('Incomplete local ABC batch; see '+str(work/'abc.log'))
        proposals=[]
        for row in rows:
            case=work/('cut_%03d'%row['index']);names=row['names'];target=row['target'];n=len(names)
            try:
                interface=Circuit(lib,names,('__local_result',),[],{'__local_result':"1'b0"})
                mapped=restore_ports(case/'mapped.v',interface)
                truth=outputs(mapped,trace(mapped,{name:variable(n,i) for i,name in enumerate(names)},1<<n))[mapped.outputs[0]]
                if (truth^row['teacher_truth'])&row['care_truth']:raise AssertionError('Local care synthesis violated retained truth rows')
                stats['exhaustive_local_care_truth_checks']+=1
                trial=cleanup(best.materialize([(target,names,to_plan(mapped))]),mapper)
            except ValueError as exc:row['error']=str(exc);continue
            stats['physical_local_embeddings']+=1;row['actual_area']=lib.area(trial.units)
            if any(trial.resolve(root) not in trial.producer for root in a.protected_roots):
                stats['protected_feature_removed_proposals']+=1;continue
            if trial.units>=best.units:continue
            delta=truth^row['original_truth'];predicted=[]
            for k in (0,1):
                regions=regions_for(names,values[k],mask);switch=0
                for i,r in enumerate(regions):
                    if delta>>i&1:switch|=r
                predicted.append(((oldbad[k]&(switch^mask))|(contexts[target][k]&switch)).bit_count())
            actual=[wrong(outputs(trial,trace(trial,p,10000,stats)),g).bit_count() for p,g in zip(inputs,gold)]
            stats['actual_train_dev_proposal_checks']+=1;row.update(predicted=predicted,actual=actual)
            if predicted!=actual:raise AssertionError('Local cut response disagrees with whole circuit')
            if all(actual[k]<=caps[k] for k in (0,1)):proposals.append((trial,actual,row))
        save(work/'candidates.json',rows)
        if not proposals:break
        best,loss,row=min(proposals,key=lambda r:(r[0].units,sum(r[1]),r[2]['index']))
        best.write(folder/'selected_working.v');events.append(dict(iteration=iteration,area=lib.area(best.units),train_dev=loss,action=row))
        save(folder/'events.json',events);print('[BUDGET CARE ACCEPT] '+json.dumps(events[-1]),flush=True)
    best.write(folder/'selected.v');save(folder/'frozen_selection.json',dict(sha256=sha(folder/'selected.v'),area=lib.area(best.units),splits=a.splits))
    search_seconds=time.perf_counter()-tick;checks=[]
    for split in a.splits:
        x=er_data.matrix(design,ref.inputs,split);p=pack(x,ref.inputs)
        error=wrong(outputs(best,trace(best,p,10000,stats)),outputs(ref,trace(ref,p,10000,stats))).bit_count()
        alt=int(np.count_nonzero(np.any(validate_numpy(best,x)!=validate_numpy(ref,x),axis=1)))
        if error!=alt:raise AssertionError('Independent NumPy simulator mismatch')
        checks.append(dict(split=split,count=10000,wrong_vectors=error,er=error/10000,passed=error<=round(a.bound*10000)))
    readback=load(folder/'selected.v',lib)
    if readback.units!=best.units:raise AssertionError('Physical area readback mismatch')
    final_loss=[wrong(outputs(best,trace(best,p,10000,stats)),g).bit_count() for p,g in zip(inputs,gold)]
    result=dict(initial_area=lib.area(source.units),area=lib.area(best.units),reduction=1-best.units/ref.units,events=events,
        train_dev=final_loss,checks=checks,validation_performed=bool(a.splits),
        search_caps_pass=all(final_loss[k]<=caps[k] for k in (0,1)),
        strict_pass=(all(r['passed'] for r in checks) and all(final_loss[k]<=caps[k] for k in (0,1))) if a.splits else None,
        seconds=time.perf_counter()-tick,search_seconds=search_seconds,stats=dict(stats),api_calls=0,selected_sha256=sha(folder/'selected.v'),verified_cell_area=True)
    save(folder/'report.json',result);print('[BUDGET CARE RESULT] '+json.dumps({k:v for k,v in result.items() if k!='events'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--bound',type=float,default=.05);p.add_argument('--development-cap',type=float,default=.048)
    p.add_argument('--training-cap',type=float);p.add_argument('--teacher',choices=('original','optimal'),default='original')
    p.add_argument('--cuts-per-root',type=int,default=2);p.add_argument('--synthesis',choices=('dc','resub'),default='dc')
    p.add_argument('--protected-roots',nargs='*',default=[])
    p.add_argument('--width',type=int,choices=(6,7,8),default=8);p.add_argument('--roots',type=int,default=96)
    p.add_argument('--cuts',type=int,default=96);p.add_argument('--rounds',type=int,default=8);p.add_argument('--seconds',type=float,default=180)
    p.add_argument('--splits',type=int,nargs='+',default=[2,100,2000,2001,7000,7001,13300,13301]);run(p.parse_args())
