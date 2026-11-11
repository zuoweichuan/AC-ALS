"""Isolated c880 ER control using unmodified original ResubALS search."""
import runtime
import argparse,json,re,subprocess,time,sys
from pathlib import Path
from collections import Counter
import numpy as np
from prepare_resub_iscas import DEFAULT,AUTHOR,VENDOR,verify_bundle,sha
from bacs_data import load,read_blif,LIBRARY_PATH
from logic import Library
from simulate import pack,trace,outputs,validate_numpy
from hybrid_identity import structural_id
from numeric_exact import ExactBackend
from bacs_numeric import save
import er_data

def run(a):
    verify_bundle(DEFAULT,'c880');sys.path.insert(0,str(AUTHOR))
    from author_netlist_io import read_mapped
    from run_original_nmhd import check_source
    check_source()
    folder=Path(a.output).resolve()
    if False:raise ValueError('D: only')
    folder.mkdir(parents=True,exist_ok=False)
    lib=Library(LIBRARY_PATH);reference=load(DEFAULT/'c880/accurate.v',lib)
    model=read_blif(VENDOR/'input/benchmark/iscas/c880.blif')
    protocol=dict(metric='strict_vector_er',bound=a.er,seed=a.seed,search_vectors=a.search_vectors,generation_vectors=64,
        initial_source='author accurate BLIF; original source code and binary',binary_sha256=sha(VENDOR/'resubals.out'),
        source_manifest_sha256=sha(AUTHOR/'source_manifest.json'),initial_sha256=sha(DEFAULT/'c880/accurate.v'),
        library_sha256=sha(LIBRARY_PATH),validation_splits=[2,100,2000,2001],vectors_per_group=10000,
        selection='author internal best only; never reselect using independent validation')
    save(folder/'protocol.json',protocol)
    cmd=['wsl','-d','Ubuntu','--cd',ExactBackend.linux(VENDOR),'--exec','env',
        'TMPDIR='+ExactBackend.linux(AUTHOR/'tmp'),'prlimit','--core=0:0','timeout','--signal=TERM','--kill-after=5s','90s',
        './resubals.out','--accCirc','input/benchmark/iscas/c880.blif',
        '--standCell','input/standard-cell/nangate_45nm_typ.lib','--outpPath',ExactBackend.linux(folder)+'/',
        '--metrType','ER','--errUppBound',str(a.er),'--nFrame',str(a.search_vectors),'--nFrame4ResubGen','64',
        '--nThread','4','--maxCandResub','100000','--seed',str(a.seed)]
    save(folder/'command.json',cmd);tick=time.perf_counter()
    with (folder/'solver.log').open('wb') as f:
        p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=110)
    wall=time.perf_counter()-tick
    if p.returncode:raise RuntimeError('Original solver failed/timeout: '+str(p.returncode))
    init,_=read_mapped(folder/'observed_initial.v',lib,reference,model[0],model[1],folder/'mapped_initial.v')
    best,_=read_mapped(folder/'observed_best.v',lib,reference,model[0],model[1],folder/'mapped_best.v')
    if structural_id(init)!=structural_id(reference):raise AssertionError('Original accurate baseline structure mismatch')
    checks=[]
    for split in protocol['validation_splits']:
        x=er_data.matrix('c880',reference.inputs,split);packed=pack(x,reference.inputs)
        left=outputs(reference,trace(reference,packed,10000));right=outputs(best,trace(best,packed,10000));bad=0
        for n in reference.outputs:bad |= left[n]^right[n]
        nr=validate_numpy(reference,x);nb=validate_numpy(best,x)
        if bad.bit_count()!=int(np.count_nonzero(np.any(nr!=nb,axis=1))):raise AssertionError('Independent validation mismatch')
        checks.append(dict(split=split,count=10000,wrong_vectors=bad.bit_count(),er=bad.bit_count()/10000,passed=bad.bit_count()<=round(a.er*10000)))
    text=(folder/'solver.log').read_text(encoding='utf-8',errors='replace')
    calls=Counter(int(n) for n in re.findall(r'\[OBS_SIM\] nFrame=(\d+)',text))
    meta=json.loads((folder/'observed_best.json').read_text(encoding='utf-8'))
    result=dict(design='c880',er=a.er,seed=a.seed,area=lib.area(best.units),original_area=lib.area(reference.units),
        reduction=1-best.units/reference.units,checks=checks,strict_pass=all(c['passed'] for c in checks),
        solver_wall_seconds=wall,author_selection=meta,complete_simulator_calls_by_vectors=dict(calls),
        complete_simulator_calls=sum(calls.values()),same_initial_graph=True,api_calls=0)
    save(folder/'report.json',result);print('[ORIGINAL ER RESULT] '+json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--er',type=float,choices=(.005,.05),required=True);p.add_argument('--seed',type=int,default=20261027)
    p.add_argument('--search-vectors',type=int,default=9984)
    run(p.parse_args())
