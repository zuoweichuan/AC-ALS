"""Reproducible author ResubALS + detached-PO refinement, no cached warm start.

This is an explicitly hybrid author-assisted algorithm, not a claim that the
independent Agent cold search beat the author. Final winners are frozen before
the predeclared additional holdouts and source-to-enhanced CEC.
"""
from portable_tools import tool_command
import runtime
import argparse,json,time,subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import research_c880_original as original
import research_c880_po as enhanced
from er_search_resub import parser as search_parser
from prepare_resub_iscas import DEFAULT,verify_bundle,sha
from bacs_data import load,LIBRARY_PATH
from logic import Library
from numeric_exact import ExactBackend,write_blif
from bacs_numeric import save
from simulate import pack,trace,outputs,validate_numpy
import er_data

def proof(source,target,folder):
    folder.mkdir();write_blif(source,folder/'original.blif');write_blif(target,folder/'enhanced.blif')
    q=lambda n:ExactBackend.quote(ExactBackend.linux(folder/n))
    command='cec %s %s'%(q('original.blif'),q('enhanced.blif'))
    p=subprocess.run(tool_command('/usr/local/bin/abc', '-c', command, distro='Ubuntu'),capture_output=True,timeout=30)
    log=(p.stdout+p.stderr).decode('utf-8',errors='replace');(folder/'cec.log').write_text(log,encoding='utf-8')
    return p.returncode==0 and 'Networks are equivalent' in log

def run(a):
    folder=Path(a.output).resolve()
    if False:raise ValueError('D: only')
    manifest=verify_bundle(DEFAULT,'c880');folder.mkdir(parents=True,exist_ok=False)
    names=('research_c880_hybrid.py','research_c880_original.py','research_c880_po.py','research_po_resynthesis.py')
    (folder/'source').mkdir()
    hashes={}
    for n in names:
        path=Path(__file__).parent/n;(folder/'source'/n).write_bytes(path.read_bytes());hashes[n]=sha(path)
    protocol=dict(algorithm='author ResubALS + our detached-output resynthesis and current-response local compression',
        independent_agent_claim=False,seed=a.seed,bound=a.er,source_sha256=manifest['designs']['c880']['sha256'],
        source_area=manifest['designs']['c880']['area'],original_search_vectors=a.frames,refinement_vectors_per_group=10000,
        additional_holdouts=[6000,6001],holdout_vectors_per_group=10000,code_sha256=hashes,
        development_slack='explicit1.2 formal bound for exploration/selection, not final constraint; necessary because original proposal dev52/10000',
        final_constraint='all predeclared final groups strictly <= bound; no final-based reselection',api_calls=0)
    save(folder/'protocol.json',protocol);whole=time.perf_counter();tick=whole
    original.run(SimpleNamespace(output=str(folder/'original'),er=a.er,seed=a.seed,search_vectors=a.frames))
    original_seconds=time.perf_counter()-tick
    opts=search_parser().parse_args(['--design','c880','--er',str(a.er),'--output',str(folder/'enhanced'),
        '--seed',str(a.seed),'--seconds',str(a.post_seconds)])
    opts.development_confidence=0.;opts.development_fraction=1.2;opts.proposal=str(folder/'original/mapped_best.v');opts.po_trials=12
    tick=time.perf_counter();enhanced.run(opts);enhanced_seconds=time.perf_counter()-tick
    lib=Library(LIBRARY_PATH);reference=load(DEFAULT/'c880/accurate.v',lib)
    old=load(folder/'original/mapped_best.v',lib)
    newreport=json.loads((folder/'enhanced/report.json').read_text(encoding='utf-8'))
    winner_path=folder/'enhanced'/('best.v' if newreport['strict_pass'] else 'failed_final.v');new=load(winner_path,lib)
    # Selection is already complete and its SHA is saved before these checks.
    save(folder/'frozen_selection.json',dict(path=str(winner_path),sha256=sha(winner_path),area=lib.area(new.units)))
    equivalent=proof(old,new,folder/'function_equivalence')
    checks=[]
    for split in protocol['additional_holdouts']:
        x=er_data.matrix('c880',reference.inputs,split);p=pack(x,reference.inputs)
        ref=outputs(reference,trace(reference,p,10000));oldout=outputs(old,trace(old,p,10000));newout=outputs(new,trace(new,p,10000))
        row=dict(split=split,count=10000)
        for label,circuit,out in (('original',old,oldout),('enhanced',new,newout)):
            bad=0
            for n in reference.outputs:bad |= out[n]^ref[n]
            alternate=int(np.count_nonzero(np.any(validate_numpy(circuit,x)!=validate_numpy(reference,x),axis=1)))
            if alternate!=bad.bit_count():raise AssertionError('Independent fresh validation disagrees')
            row[label]=dict(wrong_vectors=alternate,er=alternate/10000,passed=alternate<=round(a.er*10000))
        checks.append(row)
    baseline_area=lib.area(reference.units);oa=lib.area(old.units);na=lib.area(new.units)
    report=dict(protocol=protocol,original_area=oa,enhanced_area=na,baseline_area=baseline_area,
        original_reduction=1-old.units/reference.units,enhanced_reduction=1-new.units/reference.units,
        extra_reduction_percentage_points=(old.units-new.units)/reference.units*100,
        reduced_area_relative_to_original=1-new.units/old.units,global_function_equivalent=equivalent,
        original_predeclared_checks=json.loads((folder/'original/report.json').read_text(encoding='utf-8'))['checks'],
        enhanced_predeclared_checks=newreport['checks'],additional_holdouts=checks,
        original_stage_seconds=original_seconds,enhancement_stage_seconds=enhanced_seconds,
        whole_seconds=time.perf_counter()-whole,api_calls=0,enhancement_stats=newreport['stats'],
        same_error_on_every_input_proven=equivalent,selected_sha256=sha(winner_path),
        strict_all_groups_pass=newreport['strict_pass'] and all(r['enhanced']['passed'] for r in checks))
    save(folder/'report.json',report)
    print('[C880 HYBRID VERIFIED] '+json.dumps({k:v for k,v in report.items() if k not in ('protocol','enhancement_stats','original_predeclared_checks','enhanced_predeclared_checks')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--er',type=float,default=.005,choices=(.005,.05));p.add_argument('--seed',type=int,default=20261027)
    p.add_argument('--frames',type=int,default=9984);p.add_argument('--post-seconds',type=float,default=30)
    run(p.parse_args())
