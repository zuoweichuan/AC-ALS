"""Sequential, resumable BACS MCTS/Agent knowledge comparison (no search here)."""
import runtime
import argparse
from contextlib import contextmanager
import ctypes
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np
from bacs_data import DESIGNS,NMED_BOUNDS,MSE_BOUNDS,LIBRARY_PATH,AUTHOR_ROOT,source_path,preflight,load,matrix
from bacs_frontier import SOURCES
from bacs_numeric import number
from logic import Library
from netlist import Circuit,Gate
from numeric_agentic import EXTRA
from simulate import pack,trace,outputs

ROOT=Path(__file__).resolve().parent
ARMS=('mcts','agent_old','agent_learned')
OLD=ROOT/'results/teacher_transfer_mac_20261027/knowledge_before.json'
LEARNED=ROOT/'results/teacher_transfer_mac_20261027/knowledge_after_v2.json'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p,data):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    for attempt in range(12):
        try:os.replace(tmp,p);return
        except PermissionError:
            if attempt==11:raise
            time.sleep(.05*(attempt+1))


def alive(pid):
    if os.name!='nt':
        try:os.kill(pid,0);return True
        except ProcessLookupError:return False
    kernel=ctypes.windll.kernel32
    kernel.OpenProcess.restype=ctypes.c_void_p
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:return kernel.GetLastError()==5
    try:
        code=ctypes.c_ulong()
        if not kernel.GetExitCodeProcess(ctypes.c_void_p(handle),ctypes.byref(code)):return True
        return code.value==259
    finally:kernel.CloseHandle(ctypes.c_void_p(handle))


@contextmanager
def suite_lock(folder):
    path=folder/'.suite.lock'
    if path.exists():
        if alive(read(path)['pid']):raise RuntimeError('This suite already has a live launcher; do not start twice')
        path.unlink()
    with path.open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid()),f)
    try:yield
    finally:path.unlink(missing_ok=True)


def specs(designs,metrics):
    return [(d,'nmed',b) for d in designs for b in NMED_BOUNDS if 'nmed' in metrics]+[
        (d,'mse',MSE_BOUNDS[d]) for d in designs if d in MSE_BOUNDS and 'mse' in metrics]


def canonical_baseline(c,design):
    if design!='adder32':return c
    def rename(n):
        return re.sub(r'^(in1|in2|res)(?=\[)',lambda m:{'in1':'a','in2':'b','res':'r'}[m[1]],n)
    return Circuit(c.lib,tuple(map(rename,c.inputs)),tuple(map(rename,c.outputs)),
        [Gate(g.name,g.cell,{p:rename(n) for p,n in g.pins.items()}) for g in c.gates],
        {rename(n):rename(v) for n,v in c.aliases.items()},c.name)


def prepare(folder,a):
    (folder/'source').mkdir();(folder/'inputs').mkdir();(folder/'knowledge').mkdir()
    names=list(dict.fromkeys(SOURCES+EXTRA+['numeric_explore.py','numeric_menu_agent.py','numeric_suite.py']))
    code={}
    for n in names:
        shutil.copy2(ROOT/n,folder/'source'/n);code[n]=sha(ROOT/n)
    shutil.copy2(OLD,folder/'knowledge/old.json');shutil.copy2(LEARNED,folder/'knowledge/learned.json')
    lib=Library(LIBRARY_PATH);inputs={};dependencies={str(LIBRARY_PATH):sha(LIBRARY_PATH)}
    for d in a.designs:
        print('[SUITE INPUT CHECK] '+d,flush=True)
        dc,order,audit=preflight(d,lib)
        original=ROOT.parent/'resubals_paper_baseline/runs/bacs_nmed_paper_original_v1'/(d+'_0p0059')/'seed_20261028/mapped_initial.v'
        c=canonical_baseline(load(original,lib),d)
        if c.inputs!=dc.inputs or c.outputs!=dc.outputs:raise ValueError('Accurate baseline interface mismatch: '+d)
        for split in (0,1,2,100):
            p=pack(matrix(d,dc.inputs,split),dc.inputs)
            left=number(outputs(c,trace(c,p,10000)),order,10000)
            right=number(outputs(dc,trace(dc,p,10000)),order,10000)
            if not np.array_equal(left,right):raise ValueError('Accurate baseline mismatch: '+d)
        dest=folder/'inputs'/(d+'.v');c.write(dest)
        restored=load(dest,lib)
        if restored.units!=c.units or restored.inputs!=c.inputs or restored.outputs!=c.outputs:
            raise ValueError('Accurate baseline serialization mismatch: '+d)
        inputs[d]=dict(path=str(dest),sha256=sha(dest),area=lib.area(c.units),
            source=str(original),source_sha256=sha(original),reference=audit,
            adder_port_rename_only=d=='adder32')
        for p in (source_path(d),AUTHOR_ROOT/(d+'.blif')):dependencies[str(p)]=sha(p)
    protocol=dict(schema=1,designs=a.designs,metrics=a.metrics,arms=a.arms,seeds=a.seeds,
        specifications=specs(a.designs,a.metrics),seconds=a.seconds,decisions=a.decisions,
        expansions=a.expansions,worker_grace_seconds=120,code_sha256=code,inputs=inputs,dependencies=dependencies,
        knowledge_sha256={n:sha(folder/'knowledge'/(n+'.json')) for n in ('old','learned')},
        initial_views=2,map_variants=2,count_per_group=10000,input_splits=[0,1,2,100],
        selection='train+development selects before final; each seed reported, no final-based selection',
        author_solver_rerun=False,shared_api_cap_per_design_cny=5,no_silent_numeric_fallback=True,
        execution='live source hashes checked before every worker; saved copies audit the exact version; do not edit code during execution')
    save(folder/'suite_protocol.json',protocol)
    return protocol


def check_frozen(folder,p):
    for n,h in p['code_sha256'].items():
        if sha(ROOT/n)!=h or sha(folder/'source'/n)!=h:
            raise RuntimeError('Suite code changed: '+n+'; use a new output for changed code')
    for path,h in p['dependencies'].items():
        if sha(path)!=h:raise RuntimeError('Suite dependency changed: '+path)
    for r in p['inputs'].values():
        if sha(r['path'])!=r['sha256']:raise RuntimeError('Frozen accurate baseline changed')
    for n,h in p['knowledge_sha256'].items():
        if sha(folder/'knowledge'/(n+'.json'))!=h:raise RuntimeError('Frozen knowledge changed')


def command(folder,p,d,metric,bound,arm,seed,out):
    cmd=[sys.executable,'-u','-B',str(ROOT/('numeric_explore.py' if arm=='mcts' else 'numeric_menu_agent.py')),
        '--design',d,'--metric',metric,'--bound',str(bound),'--seed',str(seed),
        '--accurate-netlist',p['inputs'][d]['path'],'--seconds',str(p['seconds']),'--output',str(out)]
    if arm=='mcts':cmd+=['--expansions',str(p['expansions'])]
    else:cmd+=['--decisions',str(p['decisions']),'--knowledge',str(folder/'knowledge'/('old.json' if arm=='agent_old' else 'learned.json'))]
    return cmd


def stop_worker(proc):
    if os.name=='nt':subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True)
    else:proc.kill()
    proc.wait(timeout=15)


def worker(cmd,out,seconds):
    log=out.parent/(out.name+'.log');tick=time.perf_counter();last=tick
    save(out.parent/(out.name+'_command.json'),cmd)
    with log.open('w',encoding='utf-8') as stream:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        try:
            while proc.poll() is None:
                elapsed=time.perf_counter()-tick
                if elapsed>seconds+120:
                    stop_worker(proc);return dict(error='outer_worker_timeout',log=str(log),process_seconds=elapsed)
                if time.perf_counter()-last>=25:
                    status={}
                    try:
                        s=read(out/'state.json');status=dict(status=s['status'],api_calls=s.get('api_calls',0),
                            decisions=len(s.get('decisions',[])),expansions=s.get('expansions'))
                    except (OSError,ValueError,KeyError):pass
                    print('[SUITE RUNNING] '+json.dumps(dict(seconds=round(elapsed,1),**status)),flush=True);last=time.perf_counter()
                time.sleep(.25)
        except BaseException:
            if proc.poll() is None:stop_worker(proc)
            raise
    return dict(returncode=proc.returncode,process_seconds=time.perf_counter()-tick,log=str(log))


def extract(r):
    keys=('area','original_area','reduction','strict_pass','status','provider_error','whole_seconds',
        'whole_seconds_excluding_llm','llm_call_overhead_seconds','llm_http_request_seconds','numerical_search_seconds',
        'api_calls','api_responses','api_cost_upper_cny','checks','stats','component_seconds',
        'development_full_propagations','development_reward_propagations','final_full_propagations','netlist_sha256')
    return {k:r[k] for k in keys if k in r}


def summarize(folder,p,rows):
    aggregates=[]
    for seed in p['seeds']:
        for arm in p['arms']:
            for label in ['nmed_'+str(b) for b in NMED_BOUNDS]+['mse_paper']:
                group=[r for r in rows if r['seed']==seed and r['arm']==arm and
                    ('nmed_'+str(r['bound']) if r['metric']=='nmed' else 'mse_paper')==label]
                if not group:continue
                valid=[r for r in group if r.get('strict_pass') and not r.get('error') and not r.get('provider_error')]
                expected=sum(('nmed_'+str(b) if m=='nmed' else 'mse_paper')==label for d,m,b in p['specifications'])
                aggregates.append(dict(seed=seed,arm=arm,constraint_group=label,expected=expected,total=len(group),qualified=len(valid),
                    complete_qualified_mean_reduction=(sum(r['reduction'] for r in valid)/expected) if len(valid)==expected else None,
                    recorded_worker_seconds_sum=sum(r.get('whole_seconds',0) for r in group)))
    save(folder/'suite_report.json',dict(protocol=p,jobs=rows,aggregates=aggregates,
        caveats=['One seed by default; not statistical proof.','Failed or unavailable jobs are never silently treated as valid.',
            'No best-of-seeds selection.','Historical author results are not rerun or claimed as paired controls.']))
    lines=['# BACS numerical MCTS / Agent suite','',
        '|Design|Metric|Bound|Arm|Seed|Area|Reduction %|Total s|Excluding LLM s|Full simulations|Function scores|API responses/calls|Qualified|Status|',
        '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|']
    for r in rows:
        def fmt(key,scale=1):return ('%.3f'%(r[key]*scale)) if isinstance(r.get(key),(int,float)) else '-'
        stats=r.get('stats',{});qualified=r.get('strict_pass') and not r.get('error') and not r.get('provider_error')
        lines.append('|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s/%s|%s|%s|'%(r['design'],r['metric'],r['bound'],r['arm'],r['seed'],
            fmt('area'),fmt('reduction',100),fmt('whole_seconds'),fmt('whole_seconds_excluding_llm'),
            stats.get('full_simulations','-'),stats.get('function_error_evaluations','-'),r.get('api_responses','-'),r.get('api_calls','-'),
            'yes' if qualified else 'no',r.get('error',r.get('status','unknown'))))
    lines+=['','All seeds reported; failed jobs retained. Full and local propagation/function scoring are separate counters.',
        'A qualified group mean is emitted only if every job in that group completes and passes. See suite_report.json for errors, checks, counts, costs and all timing fields.']
    (folder/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(a):
    folder=Path(a.output).resolve()
    if False:raise ValueError('Keep this experiment output on D:')
    if folder.exists() and not a.resume:raise ValueError('Output exists; add --resume, or choose a new output')
    folder.mkdir(parents=True,exist_ok=True)
    with suite_lock(folder):
        if (folder/'suite_protocol.json').exists():p=read(folder/'suite_protocol.json')
        else:
            if a.resume:raise ValueError('No completed suite protocol; use a new output for interrupted preparation')
            p=prepare(folder,a)
        check_frozen(folder,p)
        total=len(p['specifications'])*len(p['arms'])*len(p['seeds'])
        print('[SUITE READY] '+json.dumps(dict(jobs=total,seconds_per_job=p['seconds'],output=str(folder))),flush=True)
        if a.prepare_only:return
        if any(arm!='mcts' for arm in p['arms']):
            from hybrid_secrets import load_key,store_key
            try:load_key()
            except RuntimeError:
                if not a.prompt_key:raise
                store_key(getpass.getpass('SiliconFlow API Key (hidden, saved with Windows DPAPI): ').strip())
                load_key()
            print('[AGENT CHECK] Credential loaded; actual responses/costs will be printed per job. No API probe charge.',flush=True)
            for d in p['designs']:
                ledger=ROOT/'hybrid_billing'/d/'ledger.json'
                if ledger.exists():print('[AGENT BUDGET] '+json.dumps(dict(design=d,spent_upper_cny=read(ledger)['spent_upper_cny'],cap_cny=5)),flush=True)
        rows=[]
        for d,metric,bound in p['specifications']:
            label=d+'_'+metric+'_'+str(bound).replace('.','p')
            for seed in p['seeds']:
                for arm in p['arms']:
                    check_frozen(folder,p)
                    parent=folder/label/('seed_'+str(seed))/arm;parent.mkdir(parents=True,exist_ok=True)
                    receipt=parent/'job.json'
                    row=dict(design=d,metric=metric,bound=bound,seed=seed,arm=arm)
                    if receipt.exists() and not (read(receipt).get('error') and a.retry_failed):
                        row=read(receipt);print('[SUITE REUSE] '+str(parent),flush=True)
                    else:
                        attempt=1
                        while (parent/('attempt_%03d'%attempt)).exists():attempt+=1
                        out=parent/('attempt_%03d'%attempt)
                        print('[SUITE START %d/%d] '%(len(rows)+1,total)+json.dumps(row),flush=True)
                        execution=worker(command(folder,p,d,metric,bound,arm,seed,out),out,p['seconds'])
                        row.update(path=str(out),**execution)
                        if (out/'report.json').exists():
                            r=read(out/'report.json');row.update(extract(r))
                            if r.get('provider_error'):row['error']='agent_controller_failed'
                            if execution.get('returncode',0) not in (0,2):row.setdefault('error','worker_failed')
                        else:row.setdefault('error','worker_missing_report')
                        save(receipt,row)
                    rows.append(row);save(folder/'progress.json',dict(completed=len(rows),total=total,last=row))
                    summarize(folder,p,rows)
                    print('[SUITE RESULT] '+json.dumps({k:v for k,v in row.items() if k not in ('stats','component_seconds','checks')},ensure_ascii=False),flush=True)
        failures=sum(bool(r.get('error') or not r.get('strict_pass')) for r in rows)
        print('[SUITE COMPLETE] '+json.dumps(dict(jobs=len(rows),failed_or_unqualified=failures,summary=str(folder/'SUMMARY.md'))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--prompt-key',action='store_true');p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--retry-failed',action='store_true',help='Explicitly retry technical failures; never retry merely because final validation failed')
    p.add_argument('--designs',nargs='+',choices=DESIGNS,default=list(DESIGNS))
    p.add_argument('--metrics',nargs='+',choices=('nmed','mse'),default=['nmed','mse'])
    p.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS))
    p.add_argument('--seeds',nargs='+',type=int,default=[20261027])
    p.add_argument('--seconds',type=float,default=180)
    p.add_argument('--decisions',type=int,default=128);p.add_argument('--expansions',type=int,default=64)
    a=p.parse_args()
    if min(a.seconds,a.decisions,a.expansions)<=0:p.error('Positive worker limits required')
    if not specs(a.designs,a.metrics):p.error('No constraints selected')
    run(a)
