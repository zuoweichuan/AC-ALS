"""Updated ISCAS85 strict-vector-ER batch: two error bounds, cold starts."""
import runtime
import argparse,getpass,json,os,shutil,time
from pathlib import Path
from bacs_data import LIBRARY_PATH
from bacs_frontier import SOURCES
from numeric_agentic import EXTRA
from numeric_suite import suite_lock,worker,extract,check_frozen,sha,read,save
import er_data
from logic import Library

ROOT=Path(__file__).resolve().parent
DESIGNS=('c880','c1355','c1908','c2670','c3540','c5315','c7552')
OLD=ROOT/'hybrid_knowledge.json'
LEARNED=ROOT/'er_agent_knowledge.json'


def prepare(folder,a):
    for n in ('source','inputs','knowledge'):(folder/n).mkdir()
    names=list(dict.fromkeys(SOURCES+EXTRA+list(er_data.source_files)+['numeric_suite.py','er_suite.py']))
    code={}
    for n in names:shutil.copy2(ROOT/n,folder/'source'/n);code[n]=sha(ROOT/n)
    for name,source in (('old',OLD),('learned',LEARNED)):
        k=read(source);k['metric_guard']='Strict vector ER: any mismatching output bit makes that input wrong. Output bits have equal status; do not use NMED, Hamming averages or low-word quantization.'
        save(folder/'knowledge'/(name+'.json'),k)
    lib=Library(LIBRARY_PATH);inputs={};dependencies={str(LIBRARY_PATH):sha(LIBRARY_PATH)}
    for d in a.designs:
        c,order,audit=er_data.preflight(d,lib)
        dest=folder/'inputs'/(d+'.v');shutil.copy2(er_data.source_path(d),dest)
        inputs[d]=dict(path=str(dest),sha256=sha(dest),area=lib.area(c.units),reference=audit)
        dependencies[str(er_data.source_path(d))]=sha(er_data.source_path(d))
        print('[ER INPUT CHECK] '+json.dumps(dict(design=d,outputs=len(order),area=lib.area(c.units),
            independent_simulators_agree=True)),flush=True)
    p=dict(schema=1,designs=a.designs,ers=a.ers,arms=a.arms,seeds=a.seeds,seconds=a.seconds,
        expansions=a.expansions,decisions=a.decisions,code_sha256=code,dependencies=dependencies,inputs=inputs,
        knowledge_sha256={n:sha(folder/'knowledge'/(n+'.json')) for n in ('old','learned')},
        metric='strict_vector_er',vectors_per_group=10000,input_splits=[0,1,2,100],
        integer_error_limits={str(er):int(er*10000) for er in a.ers},
        initialized_from='original accurate netlist for every arm; no approximate warm start',
        selection='minimum real area among train+development eligible checkpoints; final/confirmation cannot select or replace winner',
        numeric_word_quantization=False,mcts_max_children=4,max_cached_mcts_graphs=4,
        screened_lac_only=True,shared_api_cap_per_design_cny=5,no_silent_numeric_fallback=True,
        execution='source and dependency hashes checked before each worker; do not edit code mid-suite')
    save(folder/'suite_protocol.json',p);return p


def summarize(folder,p,rows):
    aggregates=[]
    for seed in p['seeds']:
        for er in p['ers']:
            for arm in p['arms']:
                group=[r for r in rows if r['seed']==seed and r['er']==er and r['arm']==arm]
                valid=[r for r in group if r.get('strict_pass') and not r.get('error') and not r.get('provider_error')]
                aggregates.append(dict(seed=seed,er=er,arm=arm,completed=len(group),expected=len(p['designs']),qualified=len(valid),
                    complete_qualified_mean_reduction=sum(r['reduction'] for r in valid)/len(p['designs']) if len(valid)==len(p['designs']) else None))
    save(folder/'suite_report.json',dict(protocol=p,jobs=rows,aggregates=aggregates,
        no_final_based_reselection=True,no_best_of_seeds_selection=True,
        caveat='Finite sampled ER, one seed by default; updated tools do not guarantee better area on every circuit.'))
    lines=['# ISCAS85 / strict vector ER','',
        'Any wrong output bit counts as one wrong input. Each group10,000; ER0.005=50 wrong vectors, ER0.05=500.',
        'Each arm starts from the accurate original. No inherited approximate winner or numerical fallback.','',
        '|Circuit|ER bound|Arm|Seed|Area|Reduction %|Total process s|Excl LLM s|Full propagations|Function scores|API responses/calls|Qualified|Status|',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|']
    for r in rows:
        fmt=lambda key,scale=1:'%.3f'%(r[key]*scale) if isinstance(r.get(key),(float,int)) else '-'
        qualified=r.get('strict_pass') and not r.get('error') and not r.get('provider_error');s=r.get('stats',{})
        lines.append('|%s|%g|%s|%d|%s|%s|%s|%s|%s|%s|%s/%s|%s|%s|'%(r['design'],r['er'],r['arm'],r['seed'],
            fmt('area'),fmt('reduction',100),fmt('process_seconds'),fmt('process_excluding_llm_seconds'),s.get('full_simulations','-'),
            s.get('function_error_evaluations','-'),r.get('api_responses','-'),r.get('api_calls','-'),
            'yes' if qualified else 'no',r.get('error',r.get('status','unknown'))))
    lines+=['','Full propagation, local root response, function scoring and exact mapping are separate counters. See JSON for full checks, times and fees.',
        'A full-group qualified mean is null until all cases in that group pass. Technical failures and validation failures are retained.']
    (folder/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def command(folder,p,d,er,arm,seed,out):
    cmd=[os.sys.executable,'-u','-B',str(ROOT/'er_search.py'),'--design',d,'--er',str(er),
        '--controller','mcts' if arm=='mcts' else 'agent','--accurate-netlist',p['inputs'][d]['path'],
        '--seed',str(seed),'--seconds',str(p['seconds']),'--expansions',str(p['expansions']),
        '--decisions',str(p['decisions']),'--output',str(out)]
    if arm!='mcts':cmd+=['--knowledge',str(folder/'knowledge'/('old.json' if arm=='agent_old' else 'learned.json'))]
    return cmd


def run(a):
    folder=Path(a.output).resolve()
    if False:raise ValueError('Keep experiment files on D:')
    if folder.exists() and not a.resume:raise ValueError('Output exists; use --resume or a new output')
    folder.mkdir(parents=True,exist_ok=True)
    with suite_lock(folder):
        if (folder/'suite_protocol.json').exists():p=read(folder/'suite_protocol.json')
        else:
            if a.resume:raise ValueError('No complete protocol; use a fresh output')
            p=prepare(folder,a)
        check_frozen(folder,p);total=len(p['designs'])*len(p['ers'])*len(p['arms'])*len(p['seeds'])
        print('[ER SUITE READY] '+json.dumps(dict(jobs=total,ers=p['ers'],seconds_per_job=p['seconds'],output=str(folder))),flush=True)
        if a.prepare_only:return
        if any(arm!='mcts' for arm in p['arms']):
            from hybrid_secrets import load_key,store_key
            try:load_key()
            except RuntimeError:
                if not a.prompt_key:raise
                store_key(getpass.getpass('SiliconFlow API Key (hidden, saved on D with DPAPI): ').strip());load_key()
            print('[AGENT CHECK] Credential loaded; actual API response counts are printed per job. No probe request.',flush=True)
            for d in p['designs']:
                ledger=ROOT/'hybrid_billing'/d/'ledger.json'
                spent=read(ledger)['spent_upper_cny'] if ledger.exists() else 0.
                print('[AGENT BUDGET] '+json.dumps(dict(design=d,spent_upper_cny=spent,remaining_cny=max(0.,5-spent),cap_cny=5)),flush=True)
        rows=[]
        for er in p['ers']:
            for d in p['designs']:
                for seed in p['seeds']:
                    for arm in p['arms']:
                        check_frozen(folder,p)
                        parent=folder/('er_'+str(er).replace('.','p'))/d/('seed_'+str(seed))/arm
                        parent.mkdir(parents=True,exist_ok=True);receipt=parent/'job.json'
                        if receipt.exists() and not (read(receipt).get('error') and a.retry_failed):
                            row=read(receipt);print('[ER SUITE REUSE] '+str(parent),flush=True)
                        else:
                            attempt=1
                            while (parent/('attempt_%03d'%attempt)).exists():attempt+=1
                            out=parent/('attempt_%03d'%attempt)
                            row=dict(design=d,er=er,metric='er',seed=seed,arm=arm,path=str(out))
                            print('[ER SUITE START %d/%d] '%(len(rows)+1,total)+json.dumps(row),flush=True)
                            execution=worker(command(folder,p,d,er,arm,seed,out),out,p['seconds']);row.update(execution)
                            if (out/'report.json').exists():
                                r=read(out/'report.json');row.update(extract(r))
                                row['final_er']=r['checks'][0]['metric'];row['confirmation_er']=r['checks'][1]['metric']
                                row['process_excluding_llm_seconds']=execution['process_seconds']-r['llm_call_overhead_seconds']
                                if r.get('provider_error'):row['error']='agent_controller_failed'
                                if execution.get('returncode',0) not in (0,2):row.setdefault('error','worker_failed')
                            else:row.setdefault('error','worker_missing_report')
                            save(receipt,row)
                        rows.append(row);save(folder/'progress.json',dict(completed=len(rows),total=total,last=row));summarize(folder,p,rows)
                        print('[ER SUITE RESULT] '+json.dumps({k:v for k,v in row.items() if k not in ('stats','checks','component_seconds')},ensure_ascii=False),flush=True)
        print('[ER SUITE COMPLETE] '+str(folder/'SUMMARY.md'),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--resume',action='store_true');p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--prompt-key',action='store_true');p.add_argument('--retry-failed',action='store_true')
    p.add_argument('--designs',nargs='+',choices=DESIGNS,default=list(DESIGNS))
    p.add_argument('--ers',nargs='+',type=float,choices=(.005,.05),default=[.005,.05])
    p.add_argument('--arms',nargs='+',choices=('mcts','agent_old','agent_learned'),default=['mcts','agent_learned'])
    p.add_argument('--seeds',nargs='+',type=int,default=[20261027]);p.add_argument('--seconds',type=float,default=180)
    p.add_argument('--expansions',type=int,default=64);p.add_argument('--decisions',type=int,default=128)
    return p


if __name__=='__main__':
    p=parser();a=p.parse_args()
    if min(a.seconds,a.expansions,a.decisions)<=0:p.error('Positive limits required')
    run(a)
