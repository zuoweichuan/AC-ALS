"""26serial GLM Agent cases; frozen source, accurate ResubALS baselines, ABC timing."""
from portable_tools import tool_command
import runtime
import argparse,csv,json,os,re,shutil,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
from numeric_suite import save,read,sha,suite_lock,stop_worker
from bacs_data import LIBRARY_PATH,DESIGNS as BACS,load,preflight as bacs_preflight,matrix
from bacs_numeric import number,number_numpy
from simulate import pack,trace,outputs,validate_numpy
from logic import Library
from numeric_exact import ExactBackend
from hybrid_identity import structural_id
from prepare_resub_iscas import DEFAULT,DESIGNS as ISCAS,verify_bundle
from comparison_model_provider import MODEL,PRICE

ROOT=Path(__file__).resolve().parent


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default=str(ROOT/'runs/glm53flash_agent_iscas_bacs_v2'))
    p.add_argument('--er-seconds',type=float,default=540.)
    p.add_argument('--nmed-seconds',type=float,default=360.)
    p.add_argument('--seed',type=int,default=20261027)
    p.add_argument('--decisions',type=int,default=32)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--max-jobs',type=int,default=0,help='0runs all pending cases; useful for staged validation')
    p.add_argument('--max-attempts',type=int,default=2,choices=(1,2))
    p.add_argument('--api-wait-minutes',type=float,default=120.,help='Wait for GLM service readiness without running failed circuit cases')
    p.add_argument('--probe-interval-seconds',type=float,default=300.)
    return p


def freeze(folder,a):
    source=folder/'source';source.mkdir();(source/'assets').mkdir();(folder/'inputs').mkdir()
    names=[p for p in ROOT.iterdir() if p.is_file() and (p.suffix in ('.py','.cc','.cpp') or p.name in
        ('numeric_agent_knowledge.json','er_agent_knowledge.json','er_agent_knowledge_v2.json','hybrid_knowledge.json'))]
    hashes={}
    for p in names:shutil.copy2(p,source/p.name);hashes[p.name]=sha(source/p.name)
    shutil.copy2(LIBRARY_PATH,source/'assets/stdcells.lib')
    from portable_assets import load_assets,contained
    asset_folder,asset_manifest=load_assets()
    knowledge=contained(asset_folder,asset_manifest['knowledge'])
    shutil.copy2(knowledge,folder/'inputs/bacs_knowledge.json')
    bundle=verify_bundle(DEFAULT);lib=Library(LIBRARY_PATH);inputs={};dependencies={str(DEFAULT/'baseline.json'):sha(DEFAULT/'baseline.json')}
    import er_resub_data as er
    er.configure(DEFAULT)
    for d in ISCAS:
        c,order,audit=er.preflight(d,lib);path=DEFAULT/d/'accurate.v'
        inputs[d]=dict(path=str(path),sha256=sha(path),area=lib.area(c.units),audit=audit)
    old=dict(inputs={name:dict(row,path=str(contained(asset_folder,row['path'])))
                     for name,row in asset_manifest['bacs_inputs'].items()})
    for d in BACS:
        original,order,audit=bacs_preflight(d,lib)
        prior=old['inputs'][d];path=Path(prior['path'])
        if sha(path)!=prior['sha256']:raise ValueError('Prior accurate BACS baseline changed: '+d)
        dest=folder/'inputs'/(d+'.v');shutil.copy2(path,dest);c=load(dest,lib)
        if c.inputs!=original.inputs or c.outputs!=original.outputs:raise ValueError('Baseline interface differs')
        for split in (0,1,2,100):
            x=pack(matrix(d,c.inputs,split),c.inputs)
            if not (number(outputs(c,trace(c,x,10000)),order,10000)==number(outputs(original,trace(original,x,10000)),order,10000)).all():
                raise ValueError('Accurate BACS baseline differs from original function')
        inputs[d]=dict(path=str(dest),sha256=sha(dest),area=lib.area(c.units),audit=audit,order=list(order))
    abc=subprocess.run(tool_command('sha256sum', '/usr/local/bin/abc', distro='Ubuntu'),capture_output=True,timeout=20)
    if abc.returncode:raise RuntimeError('ABC unavailable')
    # Interleave benchmarks to expose entry issues before expensive large designs.
    jobs=[]
    for level in range(2):
        for i in range(max(len(ISCAS),len(BACS))):
            if i<len(ISCAS):jobs.append(dict(benchmark='ISCAS85',design=ISCAS[i],metric='er',bound=(.005,.05)[level],seconds=a.er_seconds))
            if i<len(BACS):jobs.append(dict(benchmark='BACS',design=BACS[i],metric='nmed',bound=(.0059,.0294)[level],seconds=a.nmed_seconds))
    p=dict(schema=1,model=MODEL,pricing=PRICE,seed=a.seed,decisions=a.decisions,max_attempts=a.max_attempts,
        api_wait_minutes=a.api_wait_minutes,probe_interval_seconds=a.probe_interval_seconds,
        code_sha256=hashes,library_sha256=sha(source/'assets/stdcells.lib'),abc_sha256=abc.stdout.decode().split()[0],
        dependencies=dependencies,baseline=str(DEFAULT),inputs=inputs,jobs=jobs,
        knowledge_sha256=sha(folder/'inputs/bacs_knowledge.json'),
        no_mcts_or_default_fallback=True,approximate_warm_start=False,search_vectors_per_group=10000,
        er_stages=dict(research=120,physical=280,experience=120),
        nmed_tool_seconds=90,delay='ABC stime ps on area-selected graph, no resynthesis',
        reporting='retain every final exceedance and technical failure; never select with final checks',
        nmed_migration_note='Latest NMED tools newly exposed to Agent; not a model-only historical BACS ablation')
    save(folder/'protocol.json',p);return p


def check(folder,p):
    if p['model']!=MODEL:raise ValueError('Wrong model protocol')
    for n,h in p['code_sha256'].items():
        if sha(folder/'source'/n)!=h:raise ValueError('Frozen source changed: '+n)
    if sha(folder/'source/assets/stdcells.lib')!=p['library_sha256']:raise ValueError('Frozen library changed')
    for v in p['inputs'].values():
        if sha(v['path'])!=v['sha256']:raise ValueError('Accurate baseline changed')
    if sha(folder/'inputs/bacs_knowledge.json')!=p['knowledge_sha256']:raise ValueError('Frozen knowledge changed')
    verify_bundle(p['baseline'])


def timing(folder,c,lib):
    directory=folder/'abc_timing';directory.mkdir(exist_ok=True);key=structural_id(c);cache=directory/(key+'.json')
    if cache.exists():return read(cache)
    v=directory/(key+'.v');c.write(v)
    command='read_lib -w %s; read_verilog -m %s; topo; stime'%(
        ExactBackend.quote(ExactBackend.linux(folder/'source/assets/stdcells.lib')),
        ExactBackend.quote(ExactBackend.linux(v)))
    tick=time.perf_counter();r=subprocess.run(tool_command('/usr/local/bin/abc', '-c', command, distro='Ubuntu'),capture_output=True,timeout=30)
    log=(r.stdout+r.stderr).decode('utf-8',errors='replace');(directory/(key+'.log')).write_text(log,encoding='utf-8')
    delay=re.findall(r'Delay\s*=\s*([0-9.eE+\-]+)\s*ps',re.sub(r'\x1b\[[0-9;]*m','',log))
    if r.returncode or not delay:raise RuntimeError('No valid ABC stime physical delay')
    value=dict(delay_ps=float(delay[-1]),physical_area=lib.area(c.units),seconds=time.perf_counter()-tick,
        graph=key,sha256=sha(v),command=command,no_resynthesis=True)
    save(cache,value);return value


def command(folder,p,job,out):
    kind=job['metric'];cmd=[sys.executable,'-u','-B',str(folder/'source/model_benchmark_worker.py'),kind,
        '--design',job['design'],'--output',str(out),'--controller','agent','--model',MODEL,
        '--seconds',str(job['seconds']),'--decisions',str(p['decisions']),'--seed',str(p['seed'])]
    if kind=='er':cmd+=['--er',str(job['bound']),'--baseline',p['baseline']]
    else:cmd+=['--metric','nmed','--bound',str(job['bound']),'--accurate-netlist',p['inputs'][job['design']]['path'],
        '--knowledge',str(folder/'inputs/bacs_knowledge.json'),'--tool-seconds',str(p['nmed_tool_seconds'])]
    return cmd


def execute(folder,p,job,out):
    cmd=command(folder,p,job,out);save(out.parent/(out.name+'_command.json'),cmd)
    env=dict(os.environ,NETLIST_FRONTIER_ROOT=str(ROOT),BACS_PROJECT_ROOT=str(ROOT),
        BACS_LIBRARY_PATH=str(folder/'source/assets/stdcells.lib'),PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8')
    log=out.parent/(out.name+'.log');tick=time.perf_counter();last=tick
    with log.open('w',encoding='utf-8') as f:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None:
                now=time.perf_counter()
                if now-tick>job['seconds']+120:stop_worker(proc);break
                if now-last>=25:
                    state=read(out/'state.json') if (out/'state.json').exists() else {}
                    print('[GLM RUNNING]',json.dumps(dict(design=job['design'],bound=job['bound'],
                        elapsed=round(now-tick),api_calls=state.get('api_calls'),log=str(log))),flush=True);last=now
                time.sleep(.3)
        except BaseException:
            if proc.poll() is None:stop_worker(proc)
            raise
    row=dict(job,model=MODEL,output=str(out),returncode=proc.returncode,process_seconds=time.perf_counter()-tick,log=str(log))
    if (out/'report.json').exists():row.update(read(out/'report.json'))
    else:row['error']='missing_report_or_timeout'
    if proc.returncode not in (0,2) or row.get('provider_error') or not row.get('model_control_verified'):
        row.setdefault('error','agent_not_complete_or_not_verified')
    return row


def wait_for_service(folder,p,job):
    """Do not spend circuit search budgets producing model-less initial graphs."""
    from comparison_model_provider import choose
    directory=folder/'health_checks';directory.mkdir(exist_ok=True);path=directory/'history.json'
    history=read(path) if path.exists() else [];started=time.perf_counter()
    while True:
        index=len(history);tick=time.perf_counter()
        obs=dict(session=str(directory),design=job['design'],decision_id=index,
            decision_menu=[dict(id='a00',kind='finish')],purpose='API health check only; no circuit search')
        try:
            d,receipt=choose(obs,30.,system='Return JSON only: {"choice_id":"a00","reason":"OK"}. API compatibility test.',max_tokens=128)
            if d.get('choice_id')!='a00':raise RuntimeError('Invalid health-check decision')
            history.append(dict(index=index,design=job['design'],ok=True,receipt=receipt,elapsed_seconds=time.perf_counter()-tick))
            save(path,history);print('[GLM SERVICE READY]',job['design'],flush=True)
            return True,time.perf_counter()-started
        except RuntimeError as exc:
            history.append(dict(index=index,design=job['design'],ok=False,error=str(exc),elapsed_seconds=time.perf_counter()-tick))
            save(path,history)
            status=dict(state='waiting_for_glm_service',pending=job,model=MODEL,error=str(exc),
                waited_seconds=time.perf_counter()-started,completed=False,
                next_probe_seconds=p['probe_interval_seconds'])
            save(folder/'launcher_status.json',status)
            print('[GLM SERVICE WAIT]',json.dumps(status),flush=True)
            if 'cap reached' in str(exc) or time.perf_counter()-started>=60*p['api_wait_minutes']:
                return False,time.perf_counter()-started
            until=time.perf_counter()+min(p['probe_interval_seconds'],60*p['api_wait_minutes']-(time.perf_counter()-started))
            while time.perf_counter()<until:time.sleep(min(10.,until-time.perf_counter()))


def enrich(folder,p,row,lib):
    if 'selected_checkpoint' not in row:return row
    tick=time.perf_counter();run=Path(row['output'])
    selected=run/('best.v' if row['strict_pass'] else 'failed_final.v')
    if not selected.exists():
        candidates=[run/'selected.v',run/'selected_before_checks.v',run/'best.v',run/'failed_final.v']
        selected=next((v for v in candidates if v.exists()),None)
    if selected is None:row['timing_error']='selected graph missing';return row
    if sha(selected)!=row['netlist_sha256']:raise ValueError('Selected graph hash differs')
    c=load(selected,lib);baseline=load(p['inputs'][row['design']]['path'],lib)
    if abs(lib.area(c.units)-row['area'])>1e-8:raise ValueError('Physical area differs')
    # Timing is only for the already frozen area winner; it never reselects.
    current=timing(folder,c,lib);initial=timing(folder,baseline,lib)
    row.update(selected_netlist=str(selected),baseline_area=lib.area(baseline.units),area_verified=True,
        delay_ps=current['delay_ps'],initial_delay_ps=initial['delay_ps'],delay_reduction=1-current['delay_ps']/initial['delay_ps'],
        timing_note='ABC stime on same area-selected graph, not an independently delay-optimized result')
    # Common extra held-out checks are reported without modifying the winner.
    if row['metric']=='nmed':
        import numpy as np
        from bacs_numeric import Metric
        order=tuple(p['inputs'][row['design']]['order']);metric=Metric('nmed',row['bound'],order)
        extra=[]
        for split in (4100,4101):
            x=matrix(row['design'],c.inputs,split);z=pack(x,c.inputs)
            y=number(outputs(c,trace(c,z,10000)),order,10000)
            ref=number(outputs(baseline,trace(baseline,z,10000)),order,10000)
            if not np.array_equal(y,number_numpy(validate_numpy(c,x),c.outputs,order)):
                raise ValueError('Final independent simulator disagreement')
            total,value=metric.exact(y,ref);extra.append(dict(group='additional_confirmation',split=split,
                count=10000,loss=total,metric=value,passed=total<=metric.limit))
        row['additional_checks']=extra
    row['all_final_max_error']=max(c['metric'] for c in row.get('checks',[])+row.get('additional_checks',[]))
    row['all_final_pass']=all(c['passed'] for c in row.get('checks',[])+row.get('additional_checks',[]))
    row['timing_and_additional_validation_seconds']=time.perf_counter()-tick
    return row


def summarize(folder,p,rows):
    fields=['benchmark','design','metric','bound','model','area','baseline_area','reduction','delay_ps',
        'initial_delay_ps','delay_reduction','all_final_max_error','all_final_pass','model_control_verified',
        'case_total_seconds','case_total_excluding_llm','api_responses','api_cost_upper_cny','error','selected_netlist']
    with (folder/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    groups=[]
    lines=['GLM-5.3-Flash Agent; ResubALS accurate baselines; raw Nangate physical area; ABC stime ps.',
        'Winner selected by train/development area before final checks. All overruns/errors retained.',
        'BACS latest-tool Agent is newly migrated; comparisons to historical MCTS are not a model-only ablation.','',
        'Benchmark Circuit Bound Area AreaRed% Delay(ps) DelayRed% MaxError Pass Total(s) NoLLM(s) API Error']
    for r in rows:
        fmt=lambda v:'%.5g'%v if isinstance(v,(int,float)) else str(v or '-')
        vals=[r.get('benchmark'),r.get('design'),r.get('bound'),r.get('area'),100*r['reduction'] if 'reduction' in r else None,
            r.get('delay_ps'),100*r['delay_reduction'] if 'delay_reduction' in r else None,r.get('all_final_max_error'),
            r.get('all_final_pass'),r.get('case_total_seconds'),r.get('case_total_excluding_llm'),r.get('api_responses'),r.get('error')]
        lines.append(' '.join(fmt(v) for v in vals))
    for benchmark,metric,bounds in [('ISCAS85','er',(.005,.05)),('BACS','nmed',(.0059,.0294))]:
        for bound in bounds:
            group=[r for r in rows if r['benchmark']==benchmark and r['bound']==bound]
            n=7 if benchmark=='ISCAS85' else 6;physical=[r for r in group if 'delay_reduction' in r]
            complete=len(group)==n
            groups.append(dict(benchmark=benchmark,metric=metric,bound=bound,expected=n,completed=len(group),
                verified=sum(bool(r.get('model_control_verified') and not r.get('error')) for r in group),
                constraints_passed=sum(bool(r.get('all_final_pass')) for r in group),
                mean_area_reduction_all_results=sum(r['reduction'] for r in physical)/n if len(physical)==n else None,
                mean_delay_reduction_all_results=sum(r['delay_reduction'] for r in physical)/n if len(physical)==n else None))
    lines+=['',json.dumps(groups,ensure_ascii=False)]
    (folder/'SUMMARY.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(folder/'results.json',dict(model=MODEL,completed=len(rows),total=len(p['jobs']),jobs=rows,groups=groups))
    save(folder/'progress.json',dict(model=MODEL,completed=len(rows),total=len(p['jobs']),last=rows[-1] if rows else None))


def run(a):
    folder=Path(a.output).resolve()
    if False:raise ValueError('D: only')
    if a.er_seconds<520 or a.nmed_seconds<90:raise ValueError('Keep latest ER stage allowances and numeric tools')
    if folder.exists() and not a.resume:raise ValueError('Output exists; use --resume')
    folder.mkdir(parents=True,exist_ok=True)
    with suite_lock(folder):
        p=read(folder/'protocol.json') if (folder/'protocol.json').exists() else freeze(folder,a)
        check(folder,p)
        if a.prepare_only:print('[GLM PREPARED]',len(p['jobs']),'jobs',flush=True);return 0
        from hybrid_secrets import load_key
        load_key();lib=Library(folder/'source/assets/stdcells.lib');rows=[];started_jobs=0
        for number,job in enumerate(p['jobs'],1):
            check(folder,p);name=job['design']+'_'+job['metric']+'_'+str(job['bound']).replace('.','p')
            case=folder/'cases'/name;case.mkdir(parents=True,exist_ok=True);receipt=case/'job.json'
            if receipt.exists():row=read(receipt);print('[GLM REUSE]',number,name,flush=True)
            else:
                if a.max_jobs and started_jobs>=a.max_jobs:break
                attempts=[];row=None;service_wait=0.
                for attempt in range(1,p['max_attempts']+1):
                    ready,wait=wait_for_service(folder,p,job);service_wait+=wait
                    if not ready:
                        print('[GLM PAUSED AT PENDING CASE] Service still unavailable; resume same command later.',flush=True)
                        return 4
                    existing=sorted(case.glob('attempt_*'));index=max((int(v.name.split('_')[-1]) for v in existing if v.is_dir()),default=0)+1
                    out=case/('attempt_%03d'%index)
                    print('[GLM CASE START]',number,'/26',name,'attempt',index,flush=True)
                    row=execute(folder,p,job,out);attempts.append(row)
                    save(case/('attempt_%03d_result.json'%index),row)
                    # No retry merely because final independent constraints fail.
                    if not row.get('error') or attempt==p['max_attempts']:break
                    if 'cap reached' in str(row.get('provider_error')):break
                    print('[GLM TECHNICAL RETRY]',name,row.get('provider_error',row.get('error')),flush=True)
                    time.sleep(3.)
                try:row=enrich(folder,p,row,lib)
                except (ValueError,RuntimeError,subprocess.TimeoutExpired) as exc:row['postprocess_error']=str(exc);row.setdefault('error','postprocess_failed')
                row['attempts']=[{k:r.get(k) for k in ('output','status','provider_error','error','whole_seconds','api_seconds','api_cost_upper_cny','process_seconds')} for r in attempts]
                row['case_total_seconds']=sum(r['process_seconds'] for r in attempts)+row.get('timing_and_additional_validation_seconds',0.)
                row['case_total_excluding_llm']=row['case_total_seconds']-sum(r.get('llm_call_overhead_seconds',0.) for r in attempts)
                row['case_api_cost_upper_cny']=sum(r.get('api_cost_upper_cny',0.) for r in attempts)
                row['service_readiness_wait_seconds']=service_wait
                save(receipt,row);started_jobs+=1
            rows.append(row);summarize(folder,p,rows)
            print('[GLM CASE RESULT]',json.dumps({k:row.get(k) for k in ('design','bound','area','delay_ps','reduction','delay_reduction','all_final_pass','api_responses','error')}),flush=True)
        complete=len(rows)==len(p['jobs']);save(folder/'launcher_status.json',dict(completed=complete,jobs=len(rows),total=len(p['jobs']),model=MODEL))
        print('[GLM SUITE STATUS]',len(rows),'/26',str(folder/'SUMMARY.txt'),flush=True)
        return 0 if complete and all(not r.get('error') for r in rows) else 2


if __name__=='__main__':raise SystemExit(run(parser().parse_args()))
