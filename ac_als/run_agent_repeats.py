"""Frozen four-configuration DeepSeek experiment, mean +/- sample standard deviation.

Each seed starts from the identical accurate baseline and frozen knowledge.
No cross-seed learning, best-of-seeds selection or automatic failed-run retries.
"""
import runtime
import argparse
import csv
import getpass
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import run_glm_benchmarks as common
from numeric_suite import read, save, sha, suite_lock, stop_worker
from repeat_agent_worker import DEEPSEEK, GLM, AUTHORIZED_CAP

ROOT = Path(__file__).resolve().parent
ISCAS = ('c880', 'c1908', 'c2670', 'c3540', 'c5315', 'c7552')
SEEDS = (20261101, 20261102, 20261103, 20261104, 20261105)
FIELDS = ('area', 'area_reduction_pct', 'delay_ps', 'delay_reduction_pct',
          'case_total_seconds', 'case_total_excluding_llm', 'llm_call_overhead_seconds',
          'full_simulations', 'root_flip_simulations', 'function_error_evaluations',
          'api_responses', 'api_cost_upper_cny', 'all_final_max_error')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default=str(ROOT / 'runs/deepseek_four_configs_5seeds_v1'))
    p.add_argument('--model', choices=(DEEPSEEK, GLM), default=DEEPSEEK)
    p.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    p.add_argument('--benchmark', choices=('all', 'iscas85', 'bacs'), default='all')
    p.add_argument('--er-seconds', type=float, default=540.)
    p.add_argument('--nmed-seconds', type=float, default=360.)
    p.add_argument('--decisions', type=int, default=32)
    p.add_argument('--prompt-key', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--summarize-only', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--max-jobs', type=int, default=0)
    return p


def freeze(folder, args):
    # Reuse the established accurate-baseline validation and source snapshot,
    # not its GLM launcher or model worker.
    common.MODEL = args.model
    args.seed = args.seeds[0]
    args.max_attempts = 1
    args.api_wait_minutes = 0.
    args.probe_interval_seconds = 0.
    p = common.freeze(folder, args)
    templates = [j for j in p['jobs'] if
                 (j['benchmark'] != 'ISCAS85' or j['design'] in ISCAS) and
                 (args.benchmark == 'all' or j['benchmark'].lower() == args.benchmark)]
    p.update(entry='run_agent_repeats.py', model=args.model, seeds=args.seeds,
             jobs=[dict(j, seed=seed) for seed in args.seeds for j in templates],
             templates=templates, api_cap_cny=AUTHORIZED_CAP, pricing='original batch_provider accounting',
             budget_authorization='User2026-11-01 authorized cumulative30CNY per circuit; all historical spending retained',
             statistics='mean +/- sample standard deviation, ddof=1; n>=2',
             sample_policy='fixed 10000-vector groups across seeds; vary search RNG, not final test groups',
             knowledge_policy='identical frozen knowledge per independent cold-start run; no writeback',
             excluded_designs=['c1355'], initial_netlists='same ResubALS accurate starts as previous ER/NMED paper comparison',
             selection='one search-selected minimum-area graph per seed; no best-of-seeds selection',
             benchmark_average='arithmetic mean across circuits within each seed, then mean/std across seeds',
             no_auto_retry=True)
    if args.model == GLM:
        match_reference(folder, p)
    save(folder / 'protocol.json', p)
    return p


def match_reference(folder, p):
    """Preserve the completed DeepSeek experiment's algorithms and knowledge."""
    import shutil
    reference = ROOT / 'runs/deepseek_four_configs_5seeds_v1'
    old = read(reference / 'protocol.json')
    for field in ('seeds', 'templates', 'jobs', 'decisions', 'nmed_tool_seconds', 'api_cap_cny'):
        if p[field] != old[field]:
            raise ValueError('GLM comparison must match the DeepSeek protocol: ' + field)
    for design, row in p['inputs'].items():
        if row['sha256'] != old['inputs'][design]['sha256']:
            raise ValueError('Baseline differs from DeepSeek: ' + design)
    adapters = {'run_agent_repeats.py', 'repeat_agent_worker.py', 'comparison_model_provider.py'}
    for name, digest in old['code_sha256'].items():
        if name in adapters:
            continue
        source = reference / 'source' / name
        if sha(source) != digest:
            raise ValueError('DeepSeek reference source changed: ' + name)
        shutil.copy2(source, folder / 'source' / name)
        p['code_sha256'][name] = digest
    knowledge = reference / 'inputs/bacs_knowledge.json'
    if sha(knowledge) != old['knowledge_sha256']:
        raise ValueError('DeepSeek reference knowledge changed')
    shutil.copy2(knowledge, folder / 'inputs/bacs_knowledge.json')
    p.update(knowledge_sha256=sha(knowledge), matched_reference=str(reference),
             matched_reference_protocol_sha256=sha(reference / 'protocol.json'),
             model_adapter_differences='GLM uses1024 tokens including low reasoning; DeepSeek500 thinking off; same35s HTTP timeout and no retry',
             pricing=common.PRICE)


def check(folder, p):
    if p.get('entry') != 'run_agent_repeats.py' or p['model'] not in (DEEPSEEK, GLM):
        raise ValueError('Wrong repeated-experiment protocol')
    if p.get('api_cap_cny') != AUTHORIZED_CAP:
        raise ValueError('This frozen protocol predates the30CNY authorization; use a fresh --output')
    common.MODEL = p['model']
    common.check(folder, p)


def command(folder, p, job, out):
    cmd = common.command(folder, dict(p, seed=job['seed']), job, out)
    cmd[3] = str(folder / 'source/repeat_agent_worker.py')
    cmd[cmd.index('--model') + 1] = p.get('model', DEEPSEEK)
    return cmd + ['--api-cap-cny', str(p['api_cap_cny'])]


def moments(values):
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                std=statistics.stdev(values) if len(values) > 1 else None)


def normalized(row):
    r = dict(row)
    if isinstance(r.get('reduction'), (int, float)):
        r['area_reduction_pct'] = 100 * r['reduction']
    if isinstance(r.get('delay_reduction'), (int, float)):
        r['delay_reduction_pct'] = 100 * r['delay_reduction']
    for field in ('full_simulations', 'root_flip_simulations', 'function_error_evaluations'):
        if field in r.get('stats', {}):
            r[field] = r['stats'][field]
    r['technical_ok'] = bool(not r.get('error') and not r.get('postprocess_error') and
                             not r.get('provider_error') and r.get('model_control_verified') and
                             r.get('valid_model_decisions', 0) > 0 and
                             r.get('executed_agent_tools', 0) > 0 and
                             r.get('api_responses', 0) > 0 and r.get('returncode') in (0, 2))
    return r


def summarize(folder, p, raw):
    rows = [normalized(r) for r in raw]
    groups = []
    lines = [p.get('model', DEEPSEEK) + ' independent cold-start runs; mean +/- sample SD (ddof=1).',
             'Area/DelayRed are percentage points. Delay = ABC stime of the SAME area-selected graph.',
             'Final exceedances retained in statistics and counted; technical failures retained in RAW, not disguised as valid replicates.',
             'Function scores and root-flip responses are NOT complete Monte Carlo simulations.', '',
             'Circuit Metric Bound N/Expected AreaRed% DelayRed% Whole_s NoLLM_s FullSim FunctionScores OverLimit TechFail']
    def fmt(m):
        if not m['n']:
            return 'NA'
        return '%.4f+/-%.4f' % (m['mean'], m['std']) if m['std'] is not None else '%.4f+/-NA' % m['mean']
    for template in p['templates']:
        subset = [r for r in rows if all(r[k] == template[k] for k in ('benchmark', 'design', 'metric', 'bound'))]
        usable = [r for r in subset if r['technical_ok']]
        g = {k: template[k] for k in ('benchmark', 'design', 'metric', 'bound')}
        g.update(expected=len(p['seeds']), attempted=len(subset), valid=len(usable),
                 over_limit=sum(r.get('all_final_pass') is False for r in usable),
                 technical_failures=len(subset) - len(usable),
                 complete=len(usable) == len(p['seeds']),
                 statistics={f: moments([r[f] for r in usable if isinstance(r.get(f), (int, float)) and math.isfinite(r[f])]) for f in FIELDS})
        groups.append(g)
        fields = ('area_reduction_pct', 'delay_reduction_pct', 'case_total_seconds',
                  'case_total_excluding_llm', 'full_simulations', 'function_error_evaluations')
        lines.append('%s %s %s %d/%d %s %d %d' % (g['design'], g['metric'], g['bound'], g['valid'],
                     g['expected'], ' '.join(fmt(g['statistics'][f]) for f in fields), g['over_limit'], g['technical_failures']))
    averages = []
    for benchmark, metric, bounds in (('ISCAS85', 'er', (.005, .05)), ('BACS', 'nmed', (.0059, .0294))):
        for bound in bounds:
            designs = {t['design'] for t in p['templates'] if t['benchmark'] == benchmark and t['bound'] == bound}
            if not designs:
                continue
            seed_values = []
            for seed in p['seeds']:
                subset = [r for r in rows if r['benchmark'] == benchmark and r['bound'] == bound and r['seed'] == seed and r['technical_ok']]
                if {r['design'] for r in subset} != designs:
                    continue
                v = dict(seed=seed)
                for f in FIELDS:
                    if all(isinstance(r.get(f), (int, float)) for r in subset):
                        v[f] = statistics.mean(r[f] for r in subset)
                seed_values.append(v)
            averages.append(dict(benchmark=benchmark, metric=metric, bound=bound,
                complete_seeds=len(seed_values), expected_seeds=len(p['seeds']),
                statistics={f: moments([v[f] for v in seed_values if f in v]) for f in FIELDS}))
    lines += ['', 'Benchmark arithmetic mean: each seed first averages all circuits; SD then measured across independent seeds.']
    for g in averages:
        lines.append('%s %s %s seeds=%d/%d AreaRed%%=%s DelayRed%%=%s' % (g['benchmark'], g['metric'], g['bound'],
            g['complete_seeds'], g['expected_seeds'], fmt(g['statistics']['area_reduction_pct']), fmt(g['statistics']['delay_reduction_pct'])))
    save(folder / 'results.json', dict(model=p.get('model', DEEPSEEK), jobs=rows, groups=groups, benchmark_averages=averages))
    (folder / 'SUMMARY.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    # Plain text is the primary paper table. JSON retains all component counters.
    with (folder / 'RAW.csv').open('w', encoding='utf-8-sig', newline='') as f:
        names = ['benchmark', 'design', 'metric', 'bound', 'seed', 'technical_ok', 'all_final_pass', *FIELDS, 'error', 'output']
        writer = csv.DictWriter(f, fieldnames=names, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def budget(design):
    path = ROOT / 'hybrid_billing' / design / 'ledger.json'
    r = read(path) if path.exists() else {}
    return dict(design=design, spent=r.get('spent_upper_cny', 0.), cumulative_cap=AUTHORIZED_CAP,
                recorded_ledger_cap=r.get('cap_cny'),
                remaining=max(0., AUTHORIZED_CAP - r.get('spent_upper_cny', 0.)))


def execute(folder, p, job, out):
    cmd = command(folder, p, job, out)
    save(out.parent / 'command.json', cmd)
    env = dict(os.environ, NETLIST_FRONTIER_ROOT=str(ROOT), BACS_PROJECT_ROOT=str(ROOT),
               BACS_LIBRARY_PATH=str(folder / 'source/assets/stdcells.lib'), PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
    log = out.parent / 'worker.log'
    tick = last = time.perf_counter()
    with log.open('w', encoding='utf-8') as stream:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None:
                now = time.perf_counter()
                if now - tick > job['seconds'] + 120:
                    stop_worker(proc)
                    break
                if now - last >= 25:
                    print('[REPEAT RUNNING]', job['design'], job['metric'], job['bound'], job['seed'],
                          round(now - tick), 's; log=', str(log), flush=True)
                    last = now
                time.sleep(.3)
        except BaseException:
            if proc.poll() is None:
                stop_worker(proc)
            raise
    r = dict(job, model=p['model'], output=str(out), returncode=proc.returncode, process_seconds=time.perf_counter() - tick, log=str(log))
    if (out / 'report.json').exists():
        r.update(read(out / 'report.json'))
    else:
        r['error'] = 'missing_report_or_timeout'
    if proc.returncode not in (0, 2) or r.get('provider_error') or not r.get('model_control_verified'):
        r.setdefault('error', 'agent_not_complete_or_not_verified')
    try:
        r = common.enrich(folder, p, r, common.Library(folder / 'source/assets/stdcells.lib'))
    except (ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        r.update(postprocess_error=str(exc), error='postprocess_failed')
    r['case_total_seconds'] = r['process_seconds'] + r.get('timing_and_additional_validation_seconds', 0.)
    r['case_total_excluding_llm'] = r['case_total_seconds'] - r.get('llm_call_overhead_seconds', 0.)
    return r


def run(args):
    if args.model == GLM and args.output == parser().get_default('output'):
        args.output = str(ROOT / 'runs/glm53flash_four_configs_5seeds_v1')
    folder = Path(args.output).resolve()
    if False:
        raise ValueError('D: outputs only')
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        raise ValueError('At least two distinct nonnegative search seeds required')
    if not (520 <= args.er_seconds <= 900 and 90 <= args.nmed_seconds <= 900) or not 1 <= args.decisions <= 96 or args.max_jobs < 0:
        raise ValueError('Invalid bounded per-run limits')
    if folder.exists() and not (args.resume or args.summarize_only):
        raise ValueError('Use a fresh output or --resume')
    if (args.resume or args.summarize_only) and not (folder / 'protocol.json').exists():
        raise ValueError('No frozen protocol to resume')
    folder.mkdir(parents=True, exist_ok=True)
    with suite_lock(folder):
        p = read(folder / 'protocol.json') if (folder / 'protocol.json').exists() else freeze(folder, args)
        check(folder, p)
        receipts = [folder / 'cases' / ('%s_%s_%s_seed_%d' % (j['design'], j['metric'], str(j['bound']).replace('.', 'p'), j['seed'])) / 'job.json' for j in p['jobs']]
        rows = [read(path) for path in receipts if path.exists()]
        summarize(folder, p, rows)
        print('[REPEAT READY]', len(p['jobs']), 'independent runs;', len(p['seeds']), 'seeds; output=', folder, flush=True)
        for d in dict.fromkeys(j['design'] for j in p['jobs']):
            print('[BUDGET]', json.dumps(budget(d)), flush=True)
        if args.prepare_only or args.summarize_only:
            return 0
        from er_experience_suite import ensure_key
        ensure_key(args.prompt_key)
        started = 0
        for job, receipt in zip(p['jobs'], receipts):
            if receipt.exists():
                continue
            if args.max_jobs and started >= args.max_jobs:
                break
            check(folder, p)
            # Do not start a virtually model-less run under an exhausted cap.
            # Provider still applies exact context-dependent reservation checks.
            b = budget(job['design'])
            if b['remaining'] < .15:
                save(folder / 'status.json', dict(state='awaiting_budget', pending=job, budget=b, completed=False))
                print('[REPEAT PAUSED: cumulative API budget]', json.dumps(b), 'No search started for pending run.', flush=True)
                return 5
            receipt.parent.mkdir(parents=True, exist_ok=True)
            out = receipt.parent / 'run'
            if out.exists():
                save(folder / 'status.json', dict(state='interrupted_run_requires_review', pending=job, output=str(out), completed=False))
                raise RuntimeError('Interrupted run retained; no automatic retry or overwrite: ' + str(out))
            print('[REPEAT START]', job, flush=True)
            row = execute(folder, p, job, out)
            save(receipt, row)
            rows.append(row)
            started += 1
            summarize(folder, p, rows)
            print('[REPEAT RESULT]', json.dumps({k: row.get(k) for k in ('design', 'seed', 'bound', 'area', 'model_control_verified', 'all_final_pass', 'error')}), flush=True)
        complete = len(rows) == len(p['jobs'])
        save(folder / 'status.json', dict(state='complete' if complete else 'staged_stop', completed=complete, finished=len(rows), total=len(p['jobs'])))
        return 0 if complete and all(normalized(r)['technical_ok'] for r in rows) else 2


if __name__ == '__main__':
    raise SystemExit(run(parser().parse_args()))
