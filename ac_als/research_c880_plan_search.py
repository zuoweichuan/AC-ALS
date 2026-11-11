"""Bounded plan-space search with contextual recollection; old search untouched.

Actions are accumulated/removed/replaced on an immutable circuit. Proposals
receive exact JOINT sampled response (current-plan cofactors or propagation).
Only the frozen final action set is
materialized, once. This entry is a deterministic research controller (API0),
not a claim that DeepSeek has made its decisions.
"""
import runtime
import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
import numpy as np
from logic import Library, Plan
from netlist import ZERO, ONE
from bacs_data import LIBRARY_PATH, load
from bacs_numeric import cleanup, save
from cutmap import CutMapper
from screen import FanoutCones
from simulate import pack, trace, outputs, validate_numpy
from prepare_resub_iscas import DEFAULT, sha, verify_bundle
from research_c880_local_exact import cuts
from research_c880_budget_care_sat import regions_for
from research_plan_overlay import Overlay, Action, identity
import er_data


def losses(state):
    return tuple(b.bit_count() for b in state['bad'])


def fitted(names, values, forced, mask):
    """One response-fitted truth, not enumeration of Boolean functions."""
    regions = [regions_for(names, v, mask) for v in values]
    truth = 0; loss = [0, 0]; changes = []
    for row in range(1 << len(names)):
        costs = [[(regions[k][row] & forced[k][bit]).bit_count() for k in (0, 1)] for bit in (0, 1)]
        bit = int(sum(costs[1]) < sum(costs[0]))
        truth |= bit << row
        for k in (0, 1): loss[k] += costs[bit][k]
        changes.append((sum(costs[1-bit])-sum(costs[bit]), row))
    return truth, tuple(loss), changes


def context_actions(overlay, context, mapper, cones, args, deadline, stats):
    state = context['state']; current = {a.target: a for a in state['actions']}
    values = state['values']; mask = overlay.mask; base = overlay.base
    actions = []; report = []
    # The area ordering is only a screening prior. Response scores below are
    # recomputed from the proposed circuit at each new context.
    roots = sorted(context['responses'], key=lambda n: (
        -context['responses'][n]['repair_overlap'], -cones.potential(n), n))
    for target in roots:
        if time.perf_counter() > deadline: break
        response = context['responses'][target]; forced = response['forced']; cache = {}
        def fit(ns):
            if ns not in cache:
                cache[ns] = fitted(ns, values, forced, mask)
                stats['support_fits'] += 1
            return cache[ns]
        def rank(ns):
            _, loss, _ = fit(ns)
            return (sum(max(0, e-args.cap) for e in loss), sum(loss), len(ns), ns)
        # Exclude the dynamic transitive fanout, not just the original graph.
        forbidden = overlay.forbidden(context, target)
        available = [n for n in (*base.inputs, *base.producer) if n not in forbidden]
        ranked = sorted(available, key=lambda n: rank((n,)))
        supports = [(n,) for n in ranked[:4]]
        # Bounded structural cuts + a conditional feature beam. No near-neighbor
        # quota and no product of every possible support or truth table.
        boundary = [ns for ns in cuts(base, target, width=3, beam=2)
                    if ns and len(ns) <= 3 and not (set(ns) & forbidden)]
        boundary = sorted(set(boundary), key=lambda ns: (-cones.saved_with_inputs(target, ns), len(ns), ns))[:2]
        supports += boundary
        beam = [(n,) for n in ranked[:3]]
        for width in (2, 3):
            proposed = {tuple(sorted((*ns, n))) for ns in beam for n in ranked[:args.divisors] if n not in ns}
            beam = sorted(proposed, key=rank)[:2]
            supports += beam
        for ns in boundary:
            if len(ns) < 3:
                added = {tuple(sorted((*ns, n))) for n in ranked[:args.divisors] if n not in ns}
                supports += sorted(added, key=rank)[:2]
        # Low-response single roots and constants are cheap screened candidates.
        local = [Action(target, (), Plan('leaf', leaf=-2)), Action(target, (), Plan('leaf', leaf=-1))]
        for ns in dict.fromkeys(supports):
            truth, loss, changes = fit(ns)
            functions = {truth}
            # At most two rare-row relaxations; never enumerate truth assignments.
            if changes:
                for _, row in sorted(changes)[:2]: functions.add(truth ^ (1 << row))
            for fun in sorted(functions):
                p = mapper.map(len(ns), fun)
                local.append(Action(target, ns, p))
        # Predict only this root edit in the CURRENT full plan. Exact joint score
        # is still performed for shortlisted complete proposals below.
        scored = []
        for a in dict.fromkeys(local):
            if current.get(target) == a: continue
            prospective = {**current, target: a}
            try: units = overlay.area(tuple(prospective.values()))
            except ValueError:
                stats['cycle_rejections'] += 1; continue
            from simulate import plan_value
            bad = []
            for k in (0, 1):
                word = plan_value(a.plan, a.names, base.lib, values[k], mask)
                bad.append(((word ^ mask) & forced[k][0]) | (word & forced[k][1]))
            err = tuple(b.bit_count() for b in bad)
            stats['cheap_single_edit_scores'] += 1
            if max(err) > math.ceil(args.cap*args.near_ratio): continue
            if units > base.units + args.area_debt: continue
            key = (sum(max(0, e-args.cap) for e in err), units, sum(err))
            scored.append((key, a, units, err))
        # Keep both area and recovery choices per root, not only min error.
        keep = sorted(scored, key=lambda r:r[0])[:2]
        keep += sorted(scored, key=lambda r:(r[2], sum(r[3])))[:2]
        keep += sorted(scored, key=lambda r:(sum(r[3]), r[2]))[:1]
        seen = set()
        for _, a, units, err in keep:
            if a in seen: continue
            seen.add(a); actions.append(a)
        report.append(dict(target=target, sensitivity=response['sensitivity'],
            repair_overlap=response['repair_overlap'], new_error_overlap=response['new_error_overlap'],
            supports_fitted=len(cache), retained=len(seen)))
    stats['screened_actions'] += len(actions)
    return actions, report


def choose_frontier(states, args):
    unique = {s['actions']:s for s in states}
    pool = list(unique.values())
    def penalized(s):
        err = losses(s)
        caps=getattr(args,'error_caps',(args.cap,)*len(err))
        return (s['units'] + args.overflow_price*sum(max(0,e-cap) for e,cap in zip(err,caps)),
                max(err), len(s['actions']))
    feasible = sorted((s for s in pool if all(e<=cap for e,cap in zip(losses(s),getattr(args,'error_caps',(args.cap,)*len(s['bad']))))),
                      key=lambda s:(s['units'], sum(losses(s)), len(s['actions'])))
    selected = feasible[:1]
    if pool:
        selected += [min(pool,key=penalized), min(pool,key=lambda s:(s['units'],sum(losses(s)))),
                     min(pool,key=lambda s:(sum(losses(s)),s['units']))]
    seen = set(); result = []
    for s in selected + sorted(pool,key=penalized):
        if s['actions'] in seen: continue
        result.append(s); seen.add(s['actions'])
        if len(result) == args.branches: break
    return result


def run(a):
    run_tick = time.perf_counter()
    if not 0 <= a.cap <= 10000 or a.near_ratio < 1 or not 1 <= a.branches <= 4:
        raise ValueError('Expected cap 0..10000, near_ratio>=1, branches 1..4')
    folder = Path(a.output).resolve()
    if False: raise ValueError('D: only')
    folder.mkdir(parents=True, exist_ok=False)
    for filename in ('research_plan_overlay.py', Path(__file__).name):
        (folder/filename).write_bytes((Path(__file__).parent/filename).read_bytes())
    verify_bundle(DEFAULT, 'c880'); lib = Library(LIBRARY_PATH)
    base = load(a.input, lib); ref = load(DEFAULT/'c880/accurate.v', lib)
    mapper = CutMapper(lib, 4); stats = Counter(); tick = time.perf_counter()
    x = [pack(er_data.matrix('c880', ref.inputs, k), ref.inputs) for k in (0,1)]
    gold = [outputs(ref, trace(ref, p, 10000, stats)) for p in x]
    o = Overlay(base, x, gold, 10000, stats); cones = FanoutCones(base)
    large = sorted(base.producer, key=lambda n:(-cones.potential(n),n))
    save(folder/'protocol.json', dict(vars(a), metric='strict_vector_er', count_per_group=10000,
        input_sha256=sha(a.input), accurate_sha256=sha(DEFAULT/'c880/accurate.v'),
        search_splits=[0,1], cap=a.cap, api_calls=0, controller='bounded deterministic plan frontier',
        final_policy='Freeze actions and output hash before declared diagnostics. No reselection.',
        area_policy='Exact joint live physical area before cleanup; actual cleaned area reported separately.',
        candidate_policy='screened roots/support beam/fitted truth and two rare-row edits; no exhaustive LACs'))
    initial = o.score([]); frontier = [initial]; incumbent = initial
    if max(losses(initial)) > a.cap:
        raise ValueError('Input exceeds search cap; use a declared repair experiment instead')
    visited = {()}; contexts = {}; events = []; round_events = []; stale = 0; refreshes = 0
    frozen_context = None
    deadline = tick + a.seconds
    for iteration in range(a.rounds):
        if time.perf_counter() >= deadline: break
        # Visit a retained alternative when the best state stagnates.
        parent = frontier[iteration % len(frontier)]
        key = parent['actions']
        if key not in contexts or (stale >= a.refresh_stall and a.refresh):
            # Rotate an area exploration quota on stagnation. Always recollect
            # the active roots, so replace/remove decisions receive fresh data.
            offset = (refreshes * max(1,a.roots//4)) % max(1,len(large)) if stale else 0
            quota = a.roots//4
            remaining = [n for n in (large[offset:]+large[:offset]) if n not in large[:a.roots-quota]]
            roots = list(dict.fromkeys([*(p.target for p in key), *large[:a.roots-quota],
                *remaining[:quota]]))[:a.roots+len(key)]
            if a.fixed_roots: roots = large[:a.roots]
            reason = 'initial' if not contexts else ('stagnation' if stale >= a.refresh_stall else 'plan_changed')
            if a.refresh or not contexts:
                context = o.refresh(key, roots, reason, previous=contexts.get(key))
                if frozen_context is None: frozen_context = context
            else:
                # Deliberately frozen root responses form the ablation only.
                # Recompute current values so final joint scoring stays exact.
                old = frozen_context
                context = dict(old, state=o.score(key, all_nodes=True), context_key=key,
                               reason='frozen_response_ablation')
                context.pop('dynamic_reverse',None); context.pop('forbidden_by_root',None)
            contexts[key] = context; refreshes += 1
            if len(contexts) > a.branches+1: contexts.pop(next(iter(contexts)))
            events.append(dict(iteration=iteration, reason=context['reason'], epoch=context['epoch'],
                actions=len(key), errors=list(losses(context['state'])), roots=len(context['responses'])))
        context = contexts[key]
        actions, response_report = context_actions(o, context, mapper, cones, a, deadline, stats)
        candidates = []; current = {v.target:v for v in key}
        # Complete-plan add/replace proposals. Intermediate infeasible plans
        # remain eligible for the bounded frontier, rather than being committed.
        for action in actions:
            if len(current) >= a.max_actions and action.target not in current: continue
            candidates.append(identity(tuple({**current, action.target:action}.values())))
        deletion_order = sorted(current, key=lambda n:(
            -context['responses'].get(n,{}).get('repair_overlap',0),
            -mapper.cost(current[n].plan), n))
        for target in deletion_order[:a.delete_quota]:
            candidates.append(identity(tuple(v for n,v in current.items() if n != target)))
        scored = []
        for plan in dict.fromkeys(candidates):
            if plan in visited: continue
            if time.perf_counter() >= deadline: break
            visited.add(plan)
            try:
                diff = [v for v in plan if current.get(v.target) != v]
                # Frozen-response ablation always uses full propagation. It is
                # never allowed to declare stale response predictions exact.
                if a.refresh and not a.full_plan_evaluation and len(diff) == 1 and len(plan) >= len(key):
                    state = o.propose(context, diff[0])
                else:
                    state = o.score(plan)
            except ValueError:
                stats['cycle_rejections'] += 1; continue
            stats['complete_plan_evaluations'] += 1
            # Dormant edits with no sampled PO or area effect otherwise fill
            # all four branches and starve the actual near-miss alternatives.
            same_output = all(all(state['values'][k][base.resolve(n)] == parent['values'][k][base.resolve(n)]
                                  for n in base.outputs) for k in (0,1))
            if same_output and state['units'] >= parent['units']:
                stats['no_effect_proposals'] += 1; continue
            if max(losses(state)) > math.ceil(a.cap*a.near_ratio): continue
            if state['units'] > base.units+a.area_debt: continue
            scored.append(state)
        feasible = [s for s in scored if max(losses(s)) <= a.cap]
        previous = incumbent
        if feasible:
            winner = min(feasible,key=lambda s:(s['units'],sum(losses(s)),len(s['actions'])))
            if (winner['units'],sum(losses(winner))) < (incumbent['units'],sum(losses(incumbent))): incumbent=winner
        improved = incumbent['actions'] != previous['actions']
        stale = 0 if improved else stale+1
        frontier = choose_frontier([*frontier,*scored,incumbent], a)
        row = dict(iteration=iteration, parent_actions=len(key), parent_errors=list(losses(parent)),
            proposed=len(candidates), evaluated=len(scored), best_area=lib.area(incumbent['units']),
            best_errors=list(losses(incumbent)), best_actions=len(incumbent['actions']),
            frontier=[dict(area=lib.area(s['units']),errors=list(losses(s)),actions=len(s['actions'])) for s in frontier],
            refreshed_epoch=context['epoch'], node_responses=response_report,
            elapsed_seconds=time.perf_counter()-tick)
        round_events.append(row)
        print('[PLAN]', json.dumps({k:v for k,v in row.items() if k not in ('node_responses','frontier')},ensure_ascii=False), flush=True)
        save(folder/'search_progress.json',dict(events=events,rounds=round_events,stats=dict(stats)))
        if stale >= a.stop_stall: break
    search_seconds = time.perf_counter()-tick
    confirmed = o.score(incumbent['actions'])
    if confirmed['bad'] != incumbent['bad'] or confirmed['values'] != incumbent['values']:
        raise AssertionError('Response-derived final plan differs from full joint propagation')
    # Freeze the action plan before the single physical application.
    save(folder/'frozen_plan.json',dict(actions=[dict(target=p.target,names=p.names,plan=repr(p.plan)) for p in incumbent['actions']],
        pre_cleanup_area=lib.area(incumbent['units']), errors=list(losses(incumbent)),
        selected_on_search_groups_only=True))
    physical = o.commit(incumbent['actions']); actual_outputs = []
    for k in (0,1):
        actual = outputs(physical,trace(physical,x[k],10000,stats))
        actual_outputs.append(actual)
        virtual = {n:incumbent['values'][k][base.resolve(n)] for n in base.outputs}
        if actual != virtual: raise AssertionError('Joint virtual/physical output mismatch')
    final = cleanup(physical,mapper)
    for k in (0,1):
        if outputs(final,trace(final,x[k],10000,stats)) != actual_outputs[k]:
            raise AssertionError('Cleanup changed search-group function')
    final.write(folder/'selected.v'); frozen_sha = sha(folder/'selected.v')
    save(folder/'frozen_selection.json',dict(sha256=frozen_sha, area=lib.area(final.units),
        source_plan='frozen_plan.json', final_materializations=stats['final_materializations']))
    validate_tick = time.perf_counter(); checks=[]
    for split in a.splits:
        matrix = er_data.matrix('c880',ref.inputs,split)
        r = validate_numpy(ref,matrix); y = validate_numpy(final,matrix)
        bad = int(np.any(r != y,axis=1).sum())
        checks.append(dict(split=split,count=10000,errors=bad,er=bad/10000,passed=bad<=a.final_limit))
        stats['independent_numpy_circuit_evaluations'] += 2
    complete_calls = sum(stats[k] for k in ('overlay_joint_propagations','overlay_forced_propagations','full_simulations'))
    result=dict(initial_area=lib.area(base.units), area=lib.area(final.units),
        pre_cleanup_area=lib.area(incumbent['units']), reduction_vs_input=1-final.units/base.units,
        reduction_vs_common_accurate=1-final.units/ref.units, search_errors=list(losses(incumbent)),
        search_caps_pass=max(losses(incumbent))<=a.cap, checks=checks,
        strict_pass=all(c['passed'] for c in checks) if checks else None,
        search_seconds=search_seconds, validation_seconds=time.perf_counter()-validate_tick,
        total_seconds=time.perf_counter()-run_tick, setup_seconds=tick-run_tick,
        total_seconds_without_llm=time.perf_counter()-run_tick, api_calls=0, llm_request_seconds=0,
        complete_search_and_physical_propagations=complete_calls,
        complete_search_and_physical_simulated_vectors=10000*complete_calls,
        response_edit_evaluations=stats['exact_response_edit_evaluations'],
        selected_netlist_sha256=frozen_sha, actions=len(incumbent['actions']),
        stats=dict(stats), refresh_events=events, termination='deadline' if time.perf_counter()>=deadline else 'rounds_or_stagnation',
        evaluation_note='Each full/forced overlay propagation uses 10000 vectors. Response edits select measured root cofactors by bit operations without propagation; final plan is checked by full propagation and physical readback. NumPy checks reported separately.')
    save(folder/'report.json', result); print('[PLAN RESULT]',json.dumps(result,ensure_ascii=False),flush=True)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='runs/c880_er05_144p704_localcare_v1/best.v')
    p.add_argument('--output',required=True)
    p.add_argument('--seconds',type=float,default=90)
    p.add_argument('--rounds',type=int,default=24)
    p.add_argument('--roots',type=int,default=24)
    p.add_argument('--divisors',type=int,default=16)
    p.add_argument('--branches',type=int,default=4)
    p.add_argument('--max-actions',type=int,default=32)
    p.add_argument('--delete-quota',type=int,default=8)
    p.add_argument('--cap',type=int,default=450)
    p.add_argument('--final-limit',type=int,default=500)
    p.add_argument('--near-ratio',type=float,default=1.45)
    p.add_argument('--area-debt',type=int,default=32)
    p.add_argument('--overflow-price',type=float,default=0.15)
    p.add_argument('--refresh-stall',type=int,default=2)
    p.add_argument('--stop-stall',type=int,default=8)
    p.add_argument('--refresh',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--full-plan-evaluation',action='store_true',help='Control: propagate every full proposal instead of exact response reuse')
    p.add_argument('--fixed-roots',action='store_true',help='Ablation: keep exactly the initial root pool in every context')
    p.add_argument('--splits',type=int,nargs='*',default=[12000,12001,12002,12003,12004,12005,12006,12007])
    run(p.parse_args())
