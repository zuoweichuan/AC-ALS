"""Cold-start ER batch using only the ResubALS author's accurate baseline."""
from pathlib import Path
import getpass
import json
import shutil

import er_resub_data as data
import er_suite as previous
from er_suite_v2 import parser as previous_parser, command as previous_command
from bacs_frontier import SOURCES
from numeric_agentic import EXTRA
from bacs_data import LIBRARY_PATH
from logic import Library
from numeric_suite import read,save,suite_lock,worker,extract
from prepare_resub_iscas import DEFAULT,sha,verify_bundle

ROOT=Path(__file__).resolve().parent
KIND='resubals_author_accurate_initial'


def prepare(folder,a):
    baseline=Path(a.baseline).resolve();manifest=data.configure(baseline)
    for name in ('source','inputs','knowledge'):(folder/name).mkdir()
    names=list(dict.fromkeys(SOURCES+EXTRA+list(data.source_files)+['numeric_suite.py']))
    code={}
    for name in names:
        shutil.copy2(ROOT/name,folder/'source'/name);code[name]=sha(ROOT/name)
    for label,name in (('old','hybrid_knowledge.json'),('learned','er_agent_knowledge_v2.json')):
        knowledge=read(ROOT/name)
        knowledge['metric_guard']='Strict vector ER: any wrong output bit makes an input wrong. Common accurate start is the author ResubALS initialization; no approximate trajectory is inherited.'
        save(folder/'knowledge'/(label+'.json'),knowledge)
    lib=Library(LIBRARY_PATH);inputs={}
    for design in a.designs:
        circuit,order,audit=data.preflight(design,lib)
        dest=folder/'inputs'/(design+'.v');shutil.copy2(data.source_path(design),dest)
        row=manifest['designs'][design]
        inputs[design]=dict(path=str(dest),sha256=sha(dest),area=lib.area(circuit.units),
            source=str(data.source_path(design)),source_sha256=row['sha256'],reference=audit)
        print('[RESUB INPUT CHECK] '+json.dumps(dict(design=design,area=row['area'],gates=row['gates'],cec_proofs=2)),flush=True)
    protocol=dict(schema=1,baseline_kind=KIND,baseline=str(baseline),baseline_manifest_sha256=sha(baseline/'baseline.json'),
        designs=a.designs,ers=a.ers,arms=a.arms,seeds=a.seeds,seconds=a.seconds,
        decisions=a.decisions,expansions=a.expansions,er_tools_version=2,
        development_fraction=a.development_fraction,development_confidence=a.development_confidence,
        bit_repair_roots=a.bit_repair_roots,repair_pairs=a.repair_pairs,repair_debt_units=a.repair_debt_units,
        refresh_scans=a.refresh_scans,code_sha256=code,inputs=inputs,
        dependencies={str(LIBRARY_PATH):sha(LIBRARY_PATH)},
        knowledge_sha256={label:sha(folder/'knowledge'/(label+'.json')) for label in ('old','learned')},
        metric='strict_vector_er',vectors_per_group=10000,input_splits=[0,1,2,100],
        untouched_confirmation_splits=[2000,2001],integer_error_limits={str(er):int(er*10000) for er in a.ers},
        reduction_denominator='raw Liberty area of ResubALS author accurate initial mapping',
        initialized_from='author accurate BLIF via original ALSMan::Eval(initial=true); no author approximate result',
        selection='training formal limit and declared development guard; final confirmations never select winner',
        mcts_max_children=4,max_cached_mcts_graphs=4,screened_lac_only=True,
        shared_api_cap_per_design_cny=5,no_silent_numeric_fallback=True,
        caveat='Matching accurate mapped start does not make our internal candidate graph/search implementation identical to the paper.')
    save(folder/'suite_protocol.json',protocol);return protocol


def check_frozen(folder,p):
    if p.get('baseline_kind')!=KIND:raise ValueError('Not a ResubALS baseline suite; use a fresh output')
    for n,h in p['code_sha256'].items():
        if sha(ROOT/n)!=h or sha(folder/'source'/n)!=h:raise RuntimeError('Frozen suite code changed: '+n)
    if sha(Path(p['baseline'])/'baseline.json')!=p['baseline_manifest_sha256']:
        raise RuntimeError('Baseline manifest changed')
    verify_bundle(p['baseline'])
    for path,h in p['dependencies'].items():
        if sha(path)!=h:raise RuntimeError('Frozen dependency changed: '+path)
    for d,row in p['inputs'].items():
        if sha(row['path'])!=row['sha256'] or row['sha256']!=row['source_sha256']:
            raise RuntimeError('Frozen common baseline changed: '+d)
    for n,h in p['knowledge_sha256'].items():
        if sha(folder/'knowledge'/(n+'.json'))!=h:raise RuntimeError('Frozen knowledge changed')


def command(folder,p,d,er,arm,seed,out):
    cmd=previous_command(folder,p,d,er,arm,seed,out)
    cmd[3]=str(ROOT/'er_search_resub.py')
    return cmd+['--baseline',p['baseline']]


def summarize(folder,p,rows):
    previous.summarize(folder,p,rows)
    path=folder/'SUMMARY.md';text=path.read_text(encoding='utf-8')
    text=text.replace('# ISCAS85 / strict vector ER','# ISCAS85 / strict vector ER / ResubALS accurate baseline')
    text=text.replace('Each arm starts from the accurate original.',
        'Each arm starts from the same ResubALS author accurate initial mapping. Reduction uses its raw Liberty area.')
    path.write_text(text,encoding='utf-8')


def run(a):
    folder=Path(a.output).resolve()
    if False:raise ValueError('Keep all experiment files on D:')
    if folder.exists() and not a.resume:raise ValueError('Use a fresh output, or --resume for the same frozen suite')
    folder.mkdir(parents=True,exist_ok=True)
    with suite_lock(folder):
        path=folder/'suite_protocol.json'
        if a.resume and not path.exists():raise ValueError('No frozen protocol; use a fresh output')
        p=read(path) if path.exists() else prepare(folder,a)
        check_frozen(folder,p)
        total=len(p['designs'])*len(p['ers'])*len(p['arms'])*len(p['seeds'])
        print('[RESUB ER SUITE READY] '+json.dumps(dict(jobs=total,baseline=KIND,ers=p['ers'],seconds_per_job=p['seconds'])),flush=True)
        if a.prepare_only:return
        if any(arm!='mcts' for arm in p['arms']):
            from hybrid_secrets import load_key,store_key
            try:load_key()
            except RuntimeError:
                if not a.prompt_key:raise
                store_key(getpass.getpass('SiliconFlow API Key (hidden, DPAPI on D): ').strip());load_key()
            print('[AGENT CHECK] Credential loaded; API responses/calls/cost recorded per job. No probe request.',flush=True)
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
                            row=read(receipt);print('[RESUB ER REUSE] '+str(parent),flush=True)
                        else:
                            attempt=1
                            while (parent/('attempt_%03d'%attempt)).exists():attempt+=1
                            out=parent/('attempt_%03d'%attempt)
                            row=dict(design=d,er=er,metric='er',seed=seed,arm=arm,path=str(out),baseline_kind=KIND)
                            print('[RESUB ER START %d/%d] '%(len(rows)+1,total)+json.dumps(row),flush=True)
                            execution=worker(command(folder,p,d,er,arm,seed,out),out,p['seconds']);row.update(execution)
                            if (out/'report.json').exists():
                                r=read(out/'report.json');row.update(extract(r))
                                row.update(final_er=r['checks'][0]['metric'],confirmation_er=r['checks'][1]['metric'],
                                    process_excluding_llm_seconds=execution['process_seconds']-r['llm_call_overhead_seconds'],
                                    baseline_manifest_sha256=r['baseline_audit']['baseline_manifest_sha256'],
                                    reduction_vs_author_initial=r['reduction_vs_author_initial'],original_dc_area=r['original_dc_area'])
                                if r.get('baseline_kind')!=KIND:raise ValueError('Worker reported a different baseline')
                                if r.get('provider_error'):row['error']='agent_controller_failed'
                                if execution.get('returncode',0) not in (0,2):row.setdefault('error','worker_failed')
                            else:row.setdefault('error','worker_missing_report')
                            save(receipt,row)
                        rows.append(row);save(folder/'progress.json',dict(completed=len(rows),total=total,last=row))
                        summarize(folder,p,rows)
                        print('[RESUB ER RESULT] '+json.dumps({k:v for k,v in row.items() if k not in ('stats','checks','component_seconds')}),flush=True)
        print('[RESUB ER COMPLETE] '+str(folder/'SUMMARY.md'),flush=True)


def parser():
    p=previous_parser();p.description=__doc__;p.add_argument('--baseline',default=str(DEFAULT))
    return p


if __name__=='__main__':
    p=parser();a=p.parse_args()
    if not 0<a.development_fraction<=1 or min(a.seconds,a.expansions,a.decisions,a.bit_repair_roots,a.repair_pairs,a.repair_debt_units)<1:
        p.error('Invalid declared ER limits')
    run(a)
