"""Isolated PO-resynthesis validation, optional explicit trajectory proposal."""
import runtime
from pathlib import Path
import json,time,numpy as np
from er_search_resub import parser
from er_repair_tools import ERWorker,development_limit
from bacs_data import load,LIBRARY_PATH
from prepare_resub_iscas import DEFAULT,sha
from logic import Library
from bacs_numeric import VectorEROrder,number,save
from simulate import pack,trace,outputs,validate_numpy
from hybrid_identity import structural_id
from research_po_resynthesis import step as po_step
import er_data

def run(a):
    folder=Path(a.output).resolve();folder.mkdir(parents=True,exist_ok=False)
    lib=Library(LIBRARY_PATH);ref=load(DEFAULT/'c880/accurate.v',lib);source=load(a.proposal,lib) if a.proposal else ref
    order=VectorEROrder(ref.outputs);inputs=[pack(er_data.matrix('c880',ref.inputs,k),ref.inputs) for k in (0,1)]
    refs=[number(outputs(ref,trace(ref,p,10000)),order,10000) for p in inputs]
    tick=time.perf_counter();remaining=lambda:max(0,a.seconds-(time.perf_counter()-tick));events=[]
    w=ERWorker(source,order,inputs,refs,a,folder,remaining=remaining,reference_source=ref)
    for _ in range(12):
        if remaining()<2:break
        graph=structural_id(w.current)
        if (graph,'batch',1.) not in w.blocked:
            out=w.compress('balanced',1.);events.append(dict(kind='compress',result=out))
            if out['changed']:continue
        out=po_step(w,trials=a.po_trials);events.append(dict(kind='po_resynthesis',result=out))
        if not out['changed']:break
    rows=[]
    for r in w.history:
        c=load(folder/r['path'],lib);dl=w.metric.exact(number(outputs(c,trace(c,inputs[1],10000)),order,10000),refs[1])[0]
        rows.append(dict(r,development_loss=dl,eligible=r['training_loss']<=w.metric.limit and dl<=development_limit(w.metric,a)))
    selected=min((r for r in rows if r['eligible']),key=lambda r:r['units']);winner=load(folder/selected['path'],lib);checks=[]
    for split in (2,100,2000,2001):
        x=er_data.matrix('c880',ref.inputs,split);total=int(np.count_nonzero(np.any(validate_numpy(winner,x)!=validate_numpy(ref,x),axis=1)))
        checks.append(dict(split=split,er=total/10000,passed=total<=w.metric.limit))
    passed=all(c['passed'] for c in checks);winner.write(folder/('best.v' if passed else 'failed_final.v'))
    result=dict(area=lib.area(winner.units),source_area=lib.area(ref.units),reduction=1-winner.units/ref.units,strict_pass=passed,
        checks=checks,selected=selected,all_checkpoints=rows,stats=dict(w.stats),events=events,seconds=time.perf_counter()-tick,
        cold_start=not a.proposal,proposal=a.proposal,api_calls=0)
    save(folder/'report.json',result);print('[PO RESULT] '+json.dumps({k:v for k,v in result.items() if k not in ('events','all_checkpoints')}),flush=True)
    for n in ('research_c880_po.py','research_po_resynthesis.py'):
        p=Path(__file__).parent/n;(folder/n).write_bytes(p.read_bytes())

if __name__=='__main__':
    p=parser();p.add_argument('--proposal');p.add_argument('--po-trials',type=int,default=12);run(p.parse_args())
