"""Isolated, resumable strict-ER v2 batch. No legacy/frozen source mutations."""
from pathlib import Path
import getpass
import json
import os
import shutil

import er_data_v2
import er_suite as old
from numeric_suite import read,save,sha,check_frozen,suite_lock,worker,extract

ROOT=Path(__file__).resolve().parent


def prepare(folder,a):
    p=old.prepare(folder,a)
    for name in er_data_v2.source_files:
        shutil.copy2(ROOT/name,folder/'source'/name);p['code_sha256'][name]=sha(ROOT/name)
    k=read(ROOT/'er_agent_knowledge_v2.json')
    save(folder/'knowledge/learned.json',k);p['knowledge_sha256']['learned']=sha(folder/'knowledge/learned.json')
    p.update(er_tools_version=2,development_fraction=a.development_fraction,
        development_confidence=a.development_confidence,
        bit_repair_roots=a.bit_repair_roots,repair_pairs=a.repair_pairs,
        repair_debt_units=a.repair_debt_units,refresh_scans=a.refresh_scans,
        selection='train formal bound + declared development fraction; min real area selected before final/confirmation')
    save(folder/'suite_protocol.json',p);return p


def command(folder,p,d,er,arm,seed,out):
    cmd=old.command(folder,p,d,er,arm,seed,out)
    cmd[3]=str(ROOT/'er_search_v2.py')
    for key in ('development_fraction','development_confidence','bit_repair_roots','repair_pairs','repair_debt_units','refresh_scans'):
        cmd+=['--'+key.replace('_','-'),str(p[key])]
    return cmd


def run(a):
    folder=Path(a.output).resolve()
    if False:raise ValueError('Keep all files on D:')
    if folder.exists() and not a.resume:raise ValueError('Use a new output, or --resume for identical frozen v2')
    folder.mkdir(parents=True,exist_ok=True)
    with suite_lock(folder):
        p=read(folder/'suite_protocol.json') if (folder/'suite_protocol.json').exists() else prepare(folder,a)
        if p.get('er_tools_version')!=2:raise ValueError('Not an ER v2 suite; do not mix versions')
        check_frozen(folder,p)
        total=len(p['designs'])*len(p['ers'])*len(p['arms'])*len(p['seeds'])
        print('[ER V2 SUITE READY] '+json.dumps(dict(jobs=total,seconds_per_job=p['seconds'],development_fraction=p['development_fraction'])),flush=True)
        if a.prepare_only:return
        if any(arm!='mcts' for arm in p['arms']):
            from hybrid_secrets import load_key,store_key
            try:load_key()
            except RuntimeError:
                if not a.prompt_key:raise
                store_key(getpass.getpass('SiliconFlow API Key (hidden, DPAPI on D): ').strip());load_key()
            print('[AGENT CHECK] Credential loaded. Actual API responses/calls/cost follow each job; no probe or fallback.',flush=True)
            for d in p['designs']:
                f=ROOT/'hybrid_billing'/d/'ledger.json'
                spent=read(f)['spent_upper_cny'] if f.exists() else 0
                print('[AGENT BUDGET] '+json.dumps(dict(design=d,spent_upper_cny=spent,remaining_cny=max(0.,5-spent))),flush=True)
        rows=[]
        for er in p['ers']:
            for d in p['designs']:
                for seed in p['seeds']:
                    for arm in p['arms']:
                        check_frozen(folder,p)
                        parent=folder/('er_'+str(er).replace('.','p'))/d/('seed_'+str(seed))/arm
                        parent.mkdir(parents=True,exist_ok=True);receipt=parent/'job.json'
                        if receipt.exists() and not (read(receipt).get('error') and a.retry_failed):
                            row=read(receipt);print('[ER V2 REUSE] '+str(parent),flush=True)
                        else:
                            attempt=1
                            while (parent/('attempt_%03d'%attempt)).exists():attempt+=1
                            out=parent/('attempt_%03d'%attempt)
                            row=dict(design=d,er=er,metric='er',seed=seed,arm=arm,path=str(out))
                            print('[ER V2 START %d/%d] '%(len(rows)+1,total)+json.dumps(row),flush=True)
                            execution=worker(command(folder,p,d,er,arm,seed,out),out,p['seconds']);row.update(execution)
                            if (out/'report.json').exists():
                                r=read(out/'report.json');row.update(extract(r))
                                row.update(final_er=r['checks'][0]['metric'],confirmation_er=r['checks'][1]['metric'],
                                    process_excluding_llm_seconds=execution['process_seconds']-r['llm_call_overhead_seconds'])
                                if r.get('provider_error'):row['error']='agent_controller_failed'
                                if execution.get('returncode',0) not in (0,2):row.setdefault('error','worker_failed')
                            else:row.setdefault('error','worker_missing_report')
                            save(receipt,row)
                        rows.append(row);save(folder/'progress.json',dict(completed=len(rows),total=total,last=row))
                        old.summarize(folder,p,rows)
                        print('[ER V2 RESULT] '+json.dumps({k:v for k,v in row.items() if k not in ('stats','checks','component_seconds')}),flush=True)
        print('[ER V2 COMPLETE] '+str(folder/'SUMMARY.md'),flush=True)


def parser():
    p=old.parser();p.description=__doc__
    p.add_argument('--development-fraction',type=float,default=.9)
    p.add_argument('--development-confidence',type=float,default=.95)
    p.add_argument('--bit-repair-roots',type=int,default=16)
    p.add_argument('--repair-pairs',type=int,default=6)
    p.add_argument('--repair-debt-units',type=int,default=32)
    p.add_argument('--refresh-scans',type=int,default=2)
    return p


if __name__=='__main__':
    p=parser();a=p.parse_args()
    if not 0<a.development_fraction<=1 or min(a.seconds,a.expansions,a.decisions,a.bit_repair_roots,a.repair_pairs,a.repair_debt_units)<1:
        p.error('Invalid declared ER limits')
    run(a)
