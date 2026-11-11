"""Frozen serial ISCAS85 ER experience migration: MCTS and DeepSeek Agent."""
import argparse,getpass,json,math,os,subprocess,sys,time
from pathlib import Path
from verify_restored_mcts import freeze as common_freeze,check as common_check
from numeric_suite import suite_lock,stop_worker,save,read
from prepare_resub_iscas import DEFAULT,DESIGNS
from er_experience_tools import add_arguments,LESSONS
from bacs_data import LIBRARY_PATH
from logic import Library
import er_resub_data as data

ROOT=Path(__file__).resolve().parent
ENTRY='er_experience_search.py'


def qualified(row):
    basic=bool(row.get('returncode')==0 and row.get('strict_pass') and row.get('er_experience_migration')
               and not row.get('provider_error') and not row.get('error'))
    if row.get('arm')=='agent':
        return bool(basic and row.get('model_control_verified') and row.get('valid_model_decisions',0)>0
                    and row.get('executed_agent_tools',0)>0 and row.get('api_responses',0)>0)
    return basic


def freeze(folder,a):
    a.joint_extensions=2;a.joint_seconds=12.;a.backbone_seconds=60.;a.quality_operators=True;a.research_kernel=True
    p=common_freeze(folder,a)
    p.update(entry=ENTRY,experience_migration=True,arms=a.arms,expansions=a.expansions,decisions=a.decisions,
       api_cap_cny=a.api_cap_cny,experience_seconds=a.experience_seconds,experience_roots=a.experience_roots,
       experience_shortlist=a.experience_shortlist,experience_rounds=a.experience_rounds,
       experience_generation_seconds=a.experience_generation_seconds,experience_joint_fraction=a.experience_joint_fraction,
       experience_wide_fraction=a.experience_wide_fraction,knowledge=LESSONS,
       historical_approximate_import=False,no_default_fallback=True,shared_stage_wall_limit=True)
    data.configure(p['baseline']);lib=Library(LIBRARY_PATH);p['input_audits']={}
    for design in p['designs']:
        _,_,audit=data.preflight(design,lib);p['input_audits'][design]=audit
        print('[ER EXPERIENCE INPUT CHECK]',json.dumps(dict(design=design,area=audit['baseline_area'],baseline='author accurate initial')),flush=True)
    save(folder/'protocol.json',p);return p


def command(folder,p,d,er,arm,seed,out):
    cmd=[sys.executable,'-u','-B',str(folder/'source'/ENTRY),'--design',d,'--er',str(er),'--controller',arm,
       '--seed',str(seed),'--baseline',p['baseline'],'--output',str(out)]
    for flag in ('seconds','research_seconds','quality_seconds','experience_seconds','experience_roots',
                 'experience_shortlist','experience_rounds','experience_generation_seconds','experience_joint_fraction',
                 'experience_wide_fraction','decisions','expansions','api_cap_cny'):
        cmd+=['--'+flag.replace('_','-'),str(p[flag])]
    return cmd


def execute(folder,p,d,er,arm,seed,out):
    cmd=command(folder,p,d,er,arm,seed,out);log=out.parent/(out.name+'.log')
    save(out.parent/(out.name+'_command.json'),cmd)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',NETLIST_FRONTIER_ROOT=str(ROOT),
             BACS_PROJECT_ROOT=str(ROOT),BACS_LIBRARY_PATH=str(folder/'source/assets/stdcells.lib'))
    started=time.perf_counter();last=started
    with log.open('w',encoding='utf-8') as stream:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None:
                elapsed=time.perf_counter()-started
                if elapsed>p['seconds']+90:
                    stop_worker(proc);return dict(error='outer_timeout',process_seconds=elapsed,log=str(log))
                if time.perf_counter()-last>=20:
                    state=read(out/'state.json') if (out/'state.json').exists() else {}
                    print('[ER EXPERIENCE RUNNING]',json.dumps(dict(design=d,er=er,arm=arm,seconds=round(elapsed),
                       api_calls=state.get('api_calls',0),valid_decisions=state.get('valid_model_decisions',0),log=str(log))),flush=True)
                    last=time.perf_counter()
                time.sleep(.2)
        except BaseException:
            if proc.poll() is None:stop_worker(proc)
            raise
    result=dict(returncode=proc.returncode,process_seconds=time.perf_counter()-started,log=str(log))
    if (out/'report.json').exists():result.update(read(out/'report.json'))
    else:result['error']='missing_report'
    if result.get('provider_error'):result['error']='agent_controller_failed'
    if not result.get('er_experience_migration'):result.setdefault('error','experience_migration_report_missing')
    if proc.returncode not in (0,2):result.setdefault('error','worker_returncode_'+str(proc.returncode))
    return result


def summarize(folder,p,rows):
    groups=[]
    for er in p['ers']:
        for arm in p['arms']:
            group=[r for r in rows if r['er']==er and r['arm']==arm]
            expected=len(p['designs'])*len(p['seeds'])*p['repeats']
            valid=len(group)==expected and all(qualified(r) for r in group)
            groups.append(dict(er=er,arm=arm,completed=len(group),expected=expected,qualified=sum(qualified(r) for r in group),
               qualified_mean_reduction=sum(r['reduction'] for r in group)/expected if valid else None))
    save(folder/'summary.json',dict(protocol=p,jobs=rows,groups=groups))
    lines=['# ISCAS85 ER: current-graph recollection and joint compensation','',
       '|Circuit|ER|Arm|Area|Reduction %|Max final ER %|Whole s|Without LLM s|API / valid|Experience calls|Qualified|',
       '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        fmt=lambda v:'%.3f'%v if isinstance(v,(int,float)) else '-'
        maximum=max((c['metric'] for c in r.get('checks',[])),default=None)
        lines.append('|%s|%s|%s|%s|%s|%s|%s|%s|%s / %s|%s|%s|'%(
          r['design'],r['er'],r['arm'],fmt(r.get('area')),fmt(100*r['reduction']) if 'reduction' in r else '-',
          fmt(100*maximum) if maximum is not None else '-',fmt(r.get('whole_seconds')),
          fmt(r.get('whole_seconds_excluding_llm')),r.get('api_calls',0),r.get('valid_model_decisions',0),
          r.get('experience_calls',0),qualified(r)))
    lines+=['','All arms start from the same accurate ResubALS initial netlist; no historical approximate solution import.',
      'Formal ER is any-output mismatch. Four independent10k groups are read after frozen winner selection.',
      'MCTS uses four children; physical and experience refinement are separately recorded, not fake tree visits.',
      'Agent must explicitly select experience tools and stopping; finish never invokes hidden search. API failures stop its job without MCTS fallback.',
      'Full propagations, root responses, function scores, repair/joint-plan counters, API fees and complete timing are retained in summary.json.',
      'Only complete-qualified groups have an arithmetic mean; technical/validation failures are retained.']
    (folder/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def ensure_key(prompt):
    from hybrid_secrets import load_key,store_key
    try:load_key()
    except RuntimeError:
        if not prompt:raise
        store_key(getpass.getpass('SiliconFlow API Key (hidden, encrypted on D:): ').strip());load_key()
    print('[Agent KEY OK] credential loaded once; actual model responses and decisions are logged per job',flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--baseline',default=str(DEFAULT))
    p.add_argument('--designs',nargs='+',choices=DESIGNS,default=list(DESIGNS))
    p.add_argument('--ers',nargs='+',type=float,choices=(.005,.05),default=[.005,.05])
    p.add_argument('--arms',nargs='+',choices=('mcts','agent'),default=['mcts','agent'])
    p.add_argument('--seeds',nargs='+',type=int,default=[20261027]);p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--seconds',type=float,default=540.);p.add_argument('--research-seconds',type=float,default=120.)
    p.add_argument('--quality-seconds',type=float,default=280.);p.add_argument('--decisions',type=int,default=32)
    p.add_argument('--expansions',type=int,default=24);p.add_argument('--api-cap-cny',type=float,default=10.)
    p.add_argument('--prompt-key',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--retry-failed',action='store_true');p.add_argument('--prepare-only',action='store_true')
    return add_arguments(p)


def run(a):
    folder=Path(a.output).resolve()
    if False:raise ValueError('D: only')
    values=[a.seconds,a.research_seconds,a.quality_seconds,a.experience_seconds,a.experience_roots,
            a.experience_shortlist,a.experience_rounds,a.experience_generation_seconds,a.repeats,a.expansions,a.decisions,a.api_cap_cny]
    if any(not math.isfinite(v) or v<=0 for v in values) or a.seconds>900 or a.api_cap_cny>10:
        raise ValueError('Invalid bounded resource limits')
    if a.experience_roots>256 or a.experience_shortlist>256 or not 0<a.experience_joint_fraction<=1 or not 0<a.experience_wide_fraction<=1:
        raise ValueError('Invalid bounded shortlist or working margin')
    if a.research_seconds+a.quality_seconds+a.experience_seconds>a.seconds:
        raise ValueError('Stage allowances must fit one shared wall-clock limit')
    if folder.exists() and not a.resume:raise ValueError('Use a fresh output or --resume')
    if a.resume and not (folder/'protocol.json').exists():raise ValueError('No frozen suite protocol')
    folder.mkdir(parents=True,exist_ok=True)
    with suite_lock(folder):
        p=read(folder/'protocol.json') if (folder/'protocol.json').exists() else freeze(folder,a)
        if p.get('entry')!=ENTRY or not p.get('experience_migration'):raise ValueError('Wrong frozen suite version')
        common_check(folder,p)
        if a.prepare_only:return 0
        if 'agent' in p['arms']:
            ensure_key(a.prompt_key)
            for d in p['designs']:
                ledger=ROOT/'hybrid_billing'/d/'ledger.json';spent=read(ledger)['spent_upper_cny'] if ledger.exists() else 0.
                print('[Agent BUDGET]',json.dumps(dict(design=d,cumulative_cap=p['api_cap_cny'],spent=spent,
                    remaining=max(0.,p['api_cap_cny']-spent))),flush=True)
        rows=[];total=len(p['designs'])*len(p['ers'])*len(p['arms'])*len(p['seeds'])*p['repeats']
        for er in p['ers']:
            for d in p['designs']:
                for seed in p['seeds']:
                    for repeat in range(p['repeats']):
                        for arm in p['arms']:
                            common_check(folder,p)
                            case=folder/('er_'+str(er).replace('.','p'))/d/('seed_'+str(seed))/('repeat_%02d'%repeat)/arm
                            case.mkdir(parents=True,exist_ok=True);receipt=case/'job.json'
                            reuse=receipt.exists() and not (a.retry_failed and read(receipt).get('error'))
                            if reuse:row=read(receipt);print('[ER EXPERIENCE REUSE]',str(case),flush=True)
                            else:
                                attempt=1
                                while (case/('attempt_%03d'%attempt)).exists():attempt+=1
                                out=case/('attempt_%03d'%attempt)
                                print('[ER EXPERIENCE START %d/%d]'%(len(rows)+1,total),dict(design=d,er=er,arm=arm),flush=True)
                                row=dict(design=d,er=er,seed=seed,repeat=repeat,arm=arm,output=str(out))
                                row.update(execute(folder,p,d,er,arm,seed,out));save(receipt,row)
                            rows.append(row);save(folder/'progress.json',dict(completed=len(rows),total=total,last=row));summarize(folder,p,rows)
                            print('[ER EXPERIENCE RESULT]',json.dumps({k:row.get(k) for k in ('design','er','arm','area','reduction','strict_pass','experience_calls',
                              'api_calls','valid_model_decisions','model_control_verified','error')}),flush=True)
        print('[ER EXPERIENCE COMPLETE]',dict(jobs=len(rows),qualified=sum(qualified(r) for r in rows),summary=str(folder/'SUMMARY.md')),flush=True)
        return 0 if all(qualified(r) for r in rows) else 2


if __name__=='__main__':raise SystemExit(run(parser().parse_args()))
