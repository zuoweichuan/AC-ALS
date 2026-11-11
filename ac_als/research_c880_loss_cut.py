"""Response-fitted large cuts + real local synthesis, isolated warm study.

Learn a bounded beam of support-pruned tables on groups0/1. ABC only maps
specified functions. Whole-child ER is checked before adoption, and diagnostic
and fresh groups are read only after the selected netlist hash is frozen.
"""
from portable_tools import tool_command
import runtime
import argparse,time,json,subprocess
from pathlib import Path
from collections import defaultdict
import numpy as np
from logic import Library
from netlist import Circuit,ZERO
from bacs_data import load,LIBRARY_PATH
from bacs_numeric import cleanup,save
from bacs_frontier import VirtualArea
from cutmap import CutMapper
from numeric_exact import ExactBackend,restore_ports
from research_c880_local_exact import cuts
from research_c880_cut_abc import blif_truth,to_plan
from prepare_resub_iscas import DEFAULT,sha
from simulate import trace,outputs,pack,validate_numpy
import er_data

def response_tables(values,regions,wrong0,wrong1,base_bad,penalty=1):
    rows=[];a=[];b=[]
    for region in regions:
        x=(region&wrong0).bit_count();y=(region&wrong1).bit_count()
        rows.append((x,y))
        a.append(x+(penalty-1)*(region&wrong0&~base_bad).bit_count())
        b.append(y+(penalty-1)*(region&wrong1&~base_bad).bit_count())
    return rows,a,b

def run(a):
    final_limit=int(round(a.bound*10000))
    if a.train_limit is None:a.train_limit=final_limit
    if a.dev_limit is None:a.dev_limit=int(round(1.2*final_limit))
    folder=Path(a.output).resolve();folder.mkdir(parents=True,exist_ok=False);(folder/'source').mkdir()
    code={}
    for n in ('research_c880_loss_cut.py','research_c880_cut_abc.py','research_c880_local_exact.py'):
        p=Path(__file__).parent/n;(folder/'source'/n).write_bytes(p.read_bytes());code[n]=sha(p)
    save(folder/'protocol.json',dict(vars(a),input_sha256=sha(a.input),research_code_sha256=code,
        search_splits=[0,1],diagnostic_splits=list(getattr(a,'diagnostic_splits',(2,100,2000,2001,7000,7001))),fresh_splits=a.fresh_splits,
        count_per_group=10000,selection='minimum actual mapped area among train/dev admissible; freeze before any diagnostic/fresh check',
        lac_enumeration=False,api_calls=0))
    design=getattr(a,'design','c880');baseline=Path(getattr(a,'baseline',DEFAULT))
    lib=Library(LIBRARY_PATH);source=load(a.input,lib);ref=load(baseline/design/'accurate.v',lib);best=source
    inputs=[pack(er_data.matrix(design,ref.inputs,k),ref.inputs) for k in (0,1)]
    gold=[outputs(ref,trace(ref,p,10000)) for p in inputs];mask=(1<<10000)-1
    mapper=CutMapper(lib,4);stats=defaultdict(int);events=[];tick=time.perf_counter();limits=[a.train_limit,a.dev_limit]
    def totals(c):
        result=[]
        for k,p in enumerate(inputs):
            y=outputs(c,trace(c,p,10000,stats));bad=0
            for o in ref.outputs:bad |= y[o]^gold[k][o]
            result.append(bad.bit_count())
        return result
    for iteration in range(a.rounds):
        if time.perf_counter()-tick>a.seconds-3:break
        work=folder/('round_%d'%iteration);work.mkdir();area=VirtualArea(best,mapper)
        contexts=[trace(best,p,10000,stats) for p in inputs];out=[outputs(best,v) for v in contexts];base=[]
        for k in (0,1):
            bad=0
            for o in ref.outputs:bad |= out[k][o]^gold[k][o]
            base.append(bad)
        candidates=[]
        roots=sorted(best.producer,key=lambda n:(-area.cones.potential(n),n))[:a.roots]
        for target in roots:
            if time.perf_counter()-tick>a.seconds-3:break
            potential=area.cones.potential(target)
            if potential<5:continue
            wrongs=[]
            for k,p in enumerate(inputs):
                flipped=outputs(best,trace(best,p,10000,stats,base=contexts[k],flip=target));fb=0
                for o in ref.outputs:fb |= flipped[o]^gold[k][o]
                t=contexts[k][target]
                wrongs.append(((base[k]&(t^mask))|(fb&t),(base[k]&t)|(fb&(t^mask))))
            cache={}
            def fit(names):
                if names in cache:return cache[names]
                tables=[];weighted=[]
                for k in (0,1):
                    regions=[mask]
                    for n in names:
                        v=contexts[k][n];regions=[r&(v^mask) for r in regions]+[r&v for r in regions]
                    tables.append([(int((r&wrongs[k][0]).bit_count()),int((r&wrongs[k][1]).bit_count())) for r in regions])
                scores0=[x[0]+y[0] for x,y in zip(*tables)];scores1=[x[1]+y[1] for x,y in zip(*tables)]
                quality=sum(min(x,y) for x,y in zip(scores0,scores1));stats['support_response_proxies']+=1
                cache[names]=(quality,tables,scores0,scores1);return cache[names]
            boundaries=cuts(best,target,width=a.width)
            boundaries=sorted(boundaries,key=lambda ns:(-area.cones.saved_with_inputs(target,ns),-len(ns),ns))[:a.boundaries]
            # Reserve constants and low-error literal supports before the
            # larger-cut beam. Error-first pruning alone can keep redundant
            # variables and never reach a cheap rare-event constant proposal.
            supports=[(),*boundaries]
            literal_pool=list(dict.fromkeys(n for ns in boundaries for n in ns))
            supports += sorted(((n,) for n in literal_pool),key=lambda ns:(fit(ns)[0],ns))[:4]
            for boundary in boundaries:
                beam=[boundary]
                for depth in range(min(a.prune_depth,max(0,len(boundary)-2))):
                    choices={tuple(n for n in ns if n!=removed) for ns in beam for removed in ns}
                    beam=sorted(choices,key=lambda ns:(fit(ns)[0],-area.cones.saved_with_inputs(target,ns),ns))[:2];supports+=beam
            for names in dict.fromkeys(supports):
                _,tables,x,y=fit(names);truth=defined=0
                for row,(lo,hi) in enumerate(zip(x,y)):
                    if hi<lo:truth |= 1<<row
                    if hi!=lo:defined |= 1<<row
                full=(1<<(1<<len(names)))-1
                for t in dict.fromkeys((truth,truth|(defined^full))):
                    pred=[sum(z[(t>>i)&1] for i,z in enumerate(table)) for table in tables]
                    if any(p>limit for p,limit in zip(pred,limits)):continue
                    saved=area.cones.saved_with_inputs(target,names)
                    if saved<5:continue
                    candidates.append(dict(target=target,names=names,truth=t,predicted=pred,potential=saved))
        # Distinct roots get a quota before filling remaining synthesis slots.
        candidates.sort(key=lambda r:(-r['potential']/max(1,len(r['names'])),sum(r['predicted']),r['target'],r['names'],r['truth']))
        chosen=[];root_quota=defaultdict(int);identities=set()
        for row in candidates:
            key=(row['target'],row['names'],row['truth'])
            if key in identities or root_quota[row['target']]>=6:continue
            identities.add(key);root_quota[row['target']]+=1;chosen.append(row)
            if len(chosen)>=a.cases:break
        if not chosen:break
        commands=['read_lib -w '+ExactBackend.quote(ExactBackend.linux(LIBRARY_PATH))]
        for index,row in enumerate(chosen):
            row['index']=index;case=work/('cut_%03d'%index);case.mkdir();blif_truth(case/'input.blif',len(row['names']),row['truth'])
            q=lambda n:ExactBackend.quote(ExactBackend.linux(case/n))
            commands.append('read_blif %s; strash; dc2; dch; amap; write_verilog %s; write_blif %s; cec %s %s'%(
                q('input.blif'),q('mapped.v'),q('mapped.blif'),q('input.blif'),q('mapped.blif')))
        save(work/'proposals.json',chosen);script=work/'batch.abc';script.write_text('\n'.join(commands)+'\n',encoding='utf-8')
        try:
            p=subprocess.run(tool_command('/usr/local/bin/abc', '-c', 'source '+ExactBackend.quote(ExactBackend.linux(script)), distro='Ubuntu'),
                capture_output=True,timeout=max(1,a.seconds-(time.perf_counter()-tick)))
        except subprocess.TimeoutExpired:stats['batch_timeout']+=1;break
        log=(p.stdout+p.stderr).decode('utf-8',errors='replace');(work/'abc.log').write_text(log,encoding='utf-8')
        if p.returncode:raise RuntimeError('Local synthesis failed: '+str(work/'abc.log'))
        if log.count('Networks are equivalent')!=len(chosen):raise RuntimeError('Local synthesis batch incomplete: '+str(work/'abc.log'))
        stats['local_synthesis_jobs']+=len(chosen);stats['local_cec_completions']+=log.count('Networks are equivalent')
        proposals=[]
        for row in chosen:
            case=work/('cut_%03d'%row['index'])
            if not (case/'mapped.v').exists():continue
            interface=Circuit(lib,row['names'],('__local_result',),[],{'__local_result':ZERO})
            try:
                local=restore_ports(case/'mapped.v',interface);plan=to_plan(local)
                trial=cleanup(best.materialize([(row['target'],row['names'],plan)]),mapper)
            except ValueError:continue
            stats['actual_local_embeddings']+=1;row['actual_units']=trial.units
            if trial.units<best.units:proposals.append((trial,row))
        proposals.sort(key=lambda p:(p[0].units,sum(p[1]['predicted']),p[1]['index']));accepted=None
        for trial,row in proposals[:12]:
            actual=totals(trial);stats['actual_child_train_dev_checks']+=1
            if actual!=row['predicted']:raise AssertionError('Cut response prediction differs from actual circuit')
            if any(p>limit for p,limit in zip(actual,limits)):continue
            accepted=(trial,row,actual);break
        if accepted is None:break
        best,row,actual=accepted;best.write(folder/'selected_working.v')
        events.append(dict(row,area=lib.area(best.units),actual_losses=actual));save(folder/'events.json',events)
        print('[LOSS CUT ACCEPT] '+json.dumps(events[-1]),flush=True)
    best.write(folder/'selected.v');save(folder/'frozen_selection.json',dict(sha256=sha(folder/'selected.v'),area=lib.area(best.units)))
    checks=[]
    for split in (*getattr(a,'diagnostic_splits',(2,100,2000,2001,7000,7001)),*a.fresh_splits):
        x=er_data.matrix(design,ref.inputs,split);p=pack(x,ref.inputs);actual=outputs(best,trace(best,p,10000));goldfinal=outputs(ref,trace(ref,p,10000));bad=0
        for o in ref.outputs:bad |= actual[o]^goldfinal[o]
        other=int(np.count_nonzero(np.any(validate_numpy(best,x)!=validate_numpy(ref,x),axis=1)))
        if other!=bad.bit_count():raise AssertionError('Packed/NumPy final check mismatch')
        checks.append(dict(split=split,count=10000,wrong_vectors=other,er=other/10000,passed=other<=final_limit))
    result=dict(initial_area=lib.area(source.units),area=lib.area(best.units),baseline_area=lib.area(ref.units),
        reduction=1-best.units/ref.units,events=events,stats=dict(stats),seconds=time.perf_counter()-tick,
        train_development=totals(best),bound=a.bound,checks=checks,strict_pass=all(r['passed'] for r in checks) if checks else None,api_calls=0,warm_start=True)
    save(folder/'report.json',result);print('[LOSS CUT RESULT] '+json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seconds',type=float,default=180);p.add_argument('--width',type=int,choices=(6,7,8,9,10),default=10)
    p.add_argument('--roots',type=int,default=64);p.add_argument('--boundaries',type=int,default=2)
    p.add_argument('--prune-depth',type=int,default=3);p.add_argument('--cases',type=int,default=128)
    p.add_argument('--rounds',type=int,default=6);p.add_argument('--train-limit',type=int)
    p.add_argument('--dev-limit',type=int);p.add_argument('--bound',type=float,default=.005)
    p.add_argument('--fresh-splits',type=int,nargs='+',default=[8200,8201]);run(p.parse_args())
