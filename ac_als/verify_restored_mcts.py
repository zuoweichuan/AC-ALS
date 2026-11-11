"""Frozen, resumable quality checks for the restored four-child MCTS only.

No model credentials or API requests. Every job starts from the accurate
author baseline; repetition never imports a previous approximate checkpoint.
"""
from portable_tools import tool_command
import runtime
import argparse,json,math,os,shutil,subprocess,sys,time
from pathlib import Path
from numeric_suite import suite_lock,stop_worker,save,read
from prepare_resub_iscas import DEFAULT,DESIGNS,verify_bundle,sha
from bacs_data import LIBRARY_PATH
from bacs_frontier import SOURCES
from numeric_agentic import EXTRA
import er_resub_data

ROOT=Path(__file__).resolve().parent


def freeze(folder,a):
    verify_bundle(a.baseline)
    source=folder/'source';source.mkdir();(source/'assets').mkdir()
    names=set(p.name for p in ROOT.glob('*.py'))|set(SOURCES)|set(EXTRA)|set(er_resub_data.source_files)
    hashes={}
    for name in sorted(names):
        shutil.copy2(ROOT/name,source/name);hashes[name]=sha(source/name)
    shutil.copy2(LIBRARY_PATH,source/'assets/stdcells.lib')
    abc=subprocess.run(tool_command('sha256sum', '/usr/local/bin/abc', distro='Ubuntu'),capture_output=True,timeout=10)
    if abc.returncode:raise RuntimeError('Original exact ABC backend unavailable')
    p=dict(designs=a.designs,ers=a.ers,seeds=a.seeds,repeats=a.repeats,seconds=a.seconds,
        baseline=str(Path(a.baseline).resolve()),baseline_sha256=sha(Path(a.baseline)/'baseline.json'),
        code_sha256=hashes,library_sha256=sha(source/'assets/stdcells.lib'),abc_sha256=abc.stdout.decode().split()[0],
        joint_extensions=a.joint_extensions,joint_seconds=a.joint_seconds,backbone_seconds=a.backbone_seconds,
        quality_operators=getattr(a,'quality_operators',False),quality_seconds=getattr(a,'quality_seconds',225.),
        research_kernel=getattr(a,'research_kernel',False),research_seconds=getattr(a,'research_seconds',120.),
        vectors_per_group=10000,metric='strict_vector_er',training_groups=[0,1],final_groups=[2,100,2000,2001],
        api_calls=0,selection='original training+development guard; freeze selection before four final checks',
        repeats_policy='all repetitions reported, no best-of-seeds/repeats; completed failures retained')
    save(folder/'protocol.json',p);return p


def check(folder,p):
    for n,h in p['code_sha256'].items():
        if sha(folder/'source'/n)!=h:raise RuntimeError('Frozen source changed: '+n)
    if sha(folder/'source/assets/stdcells.lib')!=p['library_sha256']:raise RuntimeError('Frozen library changed')
    if sha(Path(p['baseline'])/'baseline.json')!=p['baseline_sha256']:raise RuntimeError('Baseline manifest changed')
    verify_bundle(p['baseline'])
    abc=subprocess.run(tool_command('sha256sum', '/usr/local/bin/abc', distro='Ubuntu'),capture_output=True,timeout=10)
    if abc.returncode or abc.stdout.decode().split()[0]!=p['abc_sha256']:raise RuntimeError('ABC binary changed')


def execute(folder,p,d,er,seed,output):
    entry='quality_research_mcts.py' if p.get('research_kernel') else 'quality_mcts.py' if p.get('quality_operators') else 'restored_mcts.py'
    cmd=[sys.executable,'-u','-B',str(folder/'source'/entry),
        '--design',d,'--er',str(er),'--seed',str(seed),'--seconds',str(p['seconds']),
        '--baseline',p['baseline'],'--output',str(output),'--joint-extensions',str(p['joint_extensions']),
        '--joint-seconds',str(p['joint_seconds']),'--backbone-seconds',str(p['backbone_seconds'])]
    if p.get('quality_operators'):cmd+=['--quality-seconds',str(p['quality_seconds'])]
    if p.get('research_kernel'):cmd+=['--research-seconds',str(p['research_seconds'])]
    log=output.parent/(output.name+'.log');save(output.parent/(output.name+'_command.json'),cmd)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',BACS_PROJECT_ROOT=str(ROOT),
             NETLIST_FRONTIER_ROOT=str(ROOT),BACS_LIBRARY_PATH=str(folder/'source/assets/stdcells.lib'))
    start=time.perf_counter();last=start
    with log.open('w',encoding='utf-8') as f:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None:
                elapsed=time.perf_counter()-start
                if elapsed>p['seconds']+90:stop_worker(proc);return dict(error='outer_timeout',log=str(log),process_seconds=elapsed)
                if time.perf_counter()-last>=20:
                    print('[RESTORE RUNNING]',json.dumps(dict(design=d,er=er,seed=seed,seconds=round(elapsed),log=str(log))),flush=True);last=time.perf_counter()
                time.sleep(.2)
        except BaseException:
            if proc.poll() is None:stop_worker(proc)
            raise
    r=dict(returncode=proc.returncode,process_seconds=time.perf_counter()-start,log=str(log))
    report=output/'report.json'
    if report.exists():r.update(read(report))
    else:r['error']='missing_report'
    if proc.returncode not in (0,2):r.setdefault('error','worker_returncode_'+str(proc.returncode))
    if not (r.get('restored_native_kernel') or r.get('conditional_research_kernel')):r.setdefault('error','restoration_report_incomplete')
    return r


def qualified(row):
    return bool(row.get('returncode')==0 and row.get('strict_pass') and (row.get('restored_native_kernel') or row.get('conditional_research_kernel'))
                and not row.get('error') and not row.get('provider_error'))


def summarize(folder,p,rows):
    save(folder/'summary.json',dict(protocol=p,jobs=rows))
    title=('# Conditional four-child UCT + current-run physical refinement' if p.get('research_kernel')
           else '# Restored native ER kernel + greedy spine + MCTS4')
    lines=[title,'',
        'Every job starts accurate; all repetitions and failures are retained. No Agent/API calls.','',
        '|Circuit|ER|Seed|Repeat|Greedy area|MCTS area|Reduction %|Final ER max|Pass|Whole s|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|---:|']
    for r in rows:
        def f(v):return '%.3f'%v if isinstance(v,(int,float)) else '-'
        maximum=max((c['metric'] for c in r.get('checks',[])),default=None)
        lines.append('|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|'%(
            r['design'],r['er'],r['seed'],r['repeat'],f(r.get('greedy_spine',{}).get('area')),
            f(r.get('area')),f(100*r['reduction']) if 'reduction' in r else '-',f(100*maximum) if maximum is not None else '-',
            qualified(r),
            f(r.get('whole_seconds'))))
    note=('Research-kernel mode uses screened conditional divisor construction and targeted multi-step compensation. Physical refinement runs after UCT; its cost is included and it is not counted as tree visits.'
          if p.get('research_kernel') else 'Greedy-spine preservation guarantees search-eligible area only when its measured path completes.')
    lines+=['',note+' Independent final ER failures never trigger a different selection.']
    (folder/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(a):
    folder=Path(a.output).resolve()
    if False:raise ValueError('D: only')
    if folder.exists() and not a.resume:raise ValueError('Output exists; use --resume or a fresh folder')
    if not all(math.isfinite(v) for v in (a.seconds,a.joint_seconds,a.backbone_seconds)) or min(a.seconds,a.repeats,a.joint_seconds,a.backbone_seconds)<1 or not 0<=a.joint_extensions<=4:
        raise ValueError('Invalid finite verification settings')
    if a.resume and not (folder/'protocol.json').exists():raise ValueError('No frozen protocol to resume')
    if a.quality_operators and (not math.isfinite(a.quality_seconds) or a.quality_seconds<4):raise ValueError('Invalid finite quality budget')
    if a.research_kernel and (not a.quality_operators or not math.isfinite(a.research_seconds) or a.research_seconds<1):raise ValueError('Research kernel requires bounded quality operators')
    folder.mkdir(parents=True,exist_ok=True)
    with suite_lock(folder):
        p=read(folder/'protocol.json') if (folder/'protocol.json').exists() else freeze(folder,a)
        check(folder,p);rows=[]
        for er in p['ers']:
            for d in p['designs']:
                for seed in p['seeds']:
                    for repeat in range(p['repeats']):
                        case=folder/('er_'+str(er).replace('.','p'))/d/('seed_'+str(seed))/('repeat_%02d'%repeat)
                        case.mkdir(parents=True,exist_ok=True);receipt=case/'job.json'
                        if receipt.exists():row=read(receipt)
                        else:
                            check(folder,p);attempt=1
                            while (case/('attempt_%03d'%attempt)).exists():attempt+=1
                            out=case/('attempt_%03d'%attempt)
                            print('[RESTORE JOB]',json.dumps(dict(design=d,er=er,seed=seed,repeat=repeat)),flush=True)
                            row=dict(design=d,er=er,seed=seed,repeat=repeat,output=str(out))
                            row.update(execute(folder,p,d,er,seed,out));save(receipt,row)
                        rows.append(row);summarize(folder,p,rows)
                        print('[RESTORE RESULT]',json.dumps({k:row.get(k) for k in ('design','er','seed','repeat','area','reduction','strict_pass','restored_native_kernel','conditional_research_kernel','observed_max_children','error')}),flush=True)
        passed=sum(qualified(row) for row in rows)
        print('[RESTORE COMPLETE]',json.dumps(dict(jobs=len(rows),qualified=passed,
            all_passed=passed==len(rows),summary=str(folder/'SUMMARY.md'))),flush=True)
        return 0 if passed==len(rows) else 2


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--baseline',default=str(DEFAULT))
    p.add_argument('--designs',nargs='+',choices=DESIGNS,default=['c880'])
    p.add_argument('--ers',nargs='+',type=float,choices=(.005,.05),default=[.05])
    p.add_argument('--seeds',nargs='+',type=int,default=[20261027]);p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--seconds',type=float,default=180.);p.add_argument('--joint-extensions',type=int,default=2)
    p.add_argument('--joint-seconds',type=float,default=12.);p.add_argument('--backbone-seconds',type=float,default=60.)
    p.add_argument('--quality-operators',action='store_true',help='Include the previously omitted physical research operators')
    p.add_argument('--quality-seconds',type=float,default=225.)
    p.add_argument('--research-kernel',action='store_true')
    p.add_argument('--research-seconds',type=float,default=120.)
    p.add_argument('--resume',action='store_true');return p


if __name__=='__main__':raise SystemExit(run(parser().parse_args()))
