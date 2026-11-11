"""Screened standard-cell migration of Meng et al., TCAD 2025, Algorithms 1-3.

Separate from the Agent pipeline. No AIG conversion, API, or imported search
policy. R small patterns generate consistent 0/1/2-divisor functions; M=10000
patterns measure vector ER. Phase 1 uses multiple-choice knapsack with new-error
weights. Phase 2 chooses minimum resulting error, then maximum physical gain.
Every selected modification is propagated jointly before acceptance.

Declared deviations: bounded directed divisor/pair screening instead of all
divisor combinations, physical cell area instead of AIG node count, local exact
normalization instead of ABC compress2rs, and 10000 instead of 102400 patterns.
"""
import runtime
import argparse
from collections import defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from logic import Library
from netlist import Circuit, ZERO, ONE
from frontier import normalize
from screen import FanoutCones
from simulate import pack, trace, outputs, wrong, plan_value, random_inputs, validate_numpy
from cutmap import CutMapper
from paper_netlist_subfunctions import proposals as subfunction_proposals, cell_normalize, cut_normalize

ROOT = Path(__file__).resolve().parent
DESIGNS = ('c880', 'c1908', 'c2670', 'c3540', 'c5315', 'c7552', 'c1355')


def consistent_table(words, target, mask):
    """Return 2-input mapper care/value masks, or None on a conflicting row."""
    bins = [mask]
    for word in words:
        bins = [b & (word ^ mask) for b in bins] + [b & word for b in bins]
    care = values = 0
    for i, b in enumerate(bins):
        zeros, ones = b & (target ^ mask), b & target
        if zeros and ones:
            return None
        if b:
            care |= 1 << i
            if ones:
                values |= 1 << i
    if len(words) == 1:
        care |= care << 2
        values |= values << 2
    return care, values


def used_divisors(names, plan):
    used = set()
    def visit(p):
        if p.name == 'leaf':
            if p.leaf >= 0:
                used.add(p.leaf)
        else:
            for child in p.children:
                visit(child)
    visit(plan)
    return tuple(names[i] for i in sorted(used))


@dataclass
class Candidate:
    target: str
    names: tuple
    truth: int
    plan: object
    gain: int
    bad: int
    error: int
    weight: int
    care: int
    family: str = 'root_resubstitution'

    def replacement(self):
        return self.target, self.names, self.plan

    def record(self):
        return dict(target=self.target, inputs=list(self.names), truth=self.truth,
                    care=self.care, gain_units=self.gain, error_count=self.error,
                    new_wrong_count=self.weight, family=self.family)


def knapsack(candidates, capacity):
    """Exact dual multiple-choice DP over actual integer physical area gains.

    Item weight counts newly erroneous inputs, with no credit for repairing old
    errors. Frozen chains prevent accidentally taking two items from one group.
    Sparse dominance pruning is exact, not a beam truncation.
    """
    groups = defaultdict(list)
    for c in candidates:
        if c.gain > 0 and c.weight <= capacity:
            groups[c.target].append(c)
    states = {0: (0, None)}
    for target in sorted(groups):
        previous = states
        next_states = dict(previous)
        for value, (weight, chain) in previous.items():
            for c in groups[target]:
                new_weight = weight + c.weight
                new_value = value + c.gain
                if new_weight <= capacity and new_weight < next_states.get(new_value, (capacity + 1,))[0]:
                    next_states[new_value] = new_weight, (chain, c)
        states = {}
        lowest = capacity + 1
        for value in sorted(next_states, reverse=True):
            if next_states[value][0] < lowest:
                states[value] = next_states[value]
                lowest = next_states[value][0]
    if not states:
        return []
    chain = states[max(states)][1]
    result = []
    while chain is not None:
        chain, c = chain
        result.append(c)
    return list(reversed(result))


def area_shortlist(measured, limit, per_root):
    """An infeasible single item also cannot fit the batch new-error bound.

    error(candidate) <= error(current) + newly_wrong(candidate). Therefore
    filtering error>limit cannot remove an item usable by either paper phase.
    Filter before allocating top-k physical materializations.
    """
    feasible = [c for c in measured if c.error <= limit]
    cheapest = sorted(feasible, key=lambda c: (c.error, -c.gain, len(c.names), c.names))
    largest = sorted(feasible, key=lambda c: (-c.gain, c.error, len(c.names), c.names))
    selected = cheapest[:max(1,per_root//2)]
    return selected + [c for c in largest if c not in selected][:per_root-len(selected)]


def divisor_pool(circuit, target):
    """Paper pool: TFI plus their lower-level direct fanouts; never TFO."""
    tfi, queue = set(), [target]
    while queue:
        n = queue.pop()
        gate = circuit.producer.get(n)
        if gate is None:
            continue
        for pin in circuit.lib.cells[gate.cell].inputs:
            d = circuit.resolve(gate.pins[pin])
            if d not in tfi and d not in (ZERO, ONE):
                tfi.add(d)
                queue.append(d)
    pool = set(tfi)
    for d in tfi:
        for g in circuit.fanouts.get(d, ()):
            for pin in circuit.lib.cells[g.cell].functions:
                out = g.pins.get(pin)
                if out is not None and circuit.level[out] < circuit.level[target]:
                    pool.add(out)
    pool.discard(target)
    return sorted(pool)


def screened_sets(circuit, target, values, count, pool_limit, pair_limit, rng):
    """Directed, bounded coverage; do not enumerate all circuit LAC pairs."""
    mask = (1 << count) - 1
    target_word = values[target]
    pool = divisor_pool(circuit, target)
    def rank(n):
        mismatches = (values[n] ^ target_word).bit_count()
        return min(mismatches, count - mismatches), -circuit.level[n], n
    ordered = sorted(pool, key=rank)
    # Preserve immediate fanins, correlation leaders, plus rotating TFI coverage.
    gate = circuit.producer[target]
    direct = sorted({circuit.resolve(gate.pins[p]) for p in circuit.lib.cells[gate.cell].inputs} & set(pool))
    selected = list(direct)
    for n in ordered[:max(1, pool_limit * 2 // 3)]:
        if n not in selected:
            selected.append(n)
    rest = [n for n in ordered if n not in selected]
    rng.shuffle(rest)
    selected = (selected + rest)[:pool_limit]
    yield from ((n,) for n in selected)
    seen, pairs = set(), []
    def add(a, b):
        pair = tuple(sorted((a, b)))
        if a != b and pair not in seen and len(pairs) < pair_limit:
            seen.add(pair)
            pairs.append(pair)
    # Native local pairs first, then conditionally complementary divisors.
    for i, a in enumerate(direct):
        for b in direct[i+1:]:
            add(a, b)
    # Author Gen012ResubLACsPro explicitly prioritizes retaining one original
    # fanin and replacing the other; preserve that coverage before random pairs.
    for a in direct:
        for b in selected:
            add(a, b)
    for a in selected:
        options = sorted((b for b in selected if b != a), key=lambda b: (
            -sum(bool(v) for v in (values[a] & values[b], values[a] & (values[b] ^ mask),
                                  (values[a] ^ mask) & values[b], (values[a] ^ mask) & (values[b] ^ mask))),
            rank(b)))
        for b in options[:3]:
            add(a, b)
    # Bounded random coverage, not a scan of all combinations.
    for _ in range(pair_limit * 3):
        if len(selected) < 2 or len(pairs) >= pair_limit:
            break
        a, b = rng.choice(len(selected), 2, replace=False)
        add(selected[int(a)], selected[int(b)])
    yield from pairs


class Search:
    def __init__(self, source, inputs, reference, count, args, folder):
        self.source = source
        self.inputs, self.reference, self.count = inputs, reference, count
        self.args, self.folder = args, folder
        self.stats = defaultdict(int)
        self.times = defaultdict(float)
        self.history = []
        self.started = time.perf_counter()
        self.rng = np.random.default_rng(args.seed)
        self.iteration = 0
        self.mapper = CutMapper(source.lib,4) if args.subfunctions or args.cell_remap or args.cut_remap else None
        self.cleanup = (lambda c: cell_normalize(c,self.mapper)) if args.cell_remap else normalize
        self.current = self.cleanup(source)
        if args.cut_remap:
            self.current = cut_normalize(self.current,self.mapper,self.stats)
        self.checkpoint(self.current, 'initial', [])

    def expired(self):
        return time.perf_counter() - self.started >= self.args.seconds

    def checkpoint(self, circuit, phase, selected, bad=None):
        path = self.folder / ('node_%04d.v' % len(self.history))
        circuit.write(path)
        if bad is None:
            vals = trace(circuit, self.inputs, self.count, self.stats)
            bad = wrong(circuit, vals, self.reference)
        row = dict(index=len(self.history), path=path.name, phase=phase,
                   iteration=self.iteration, area=circuit.lib.area(circuit.units),
                   units=circuit.units, train_error_count=bad.bit_count(),
                   seconds=time.perf_counter()-self.started,
                   actions=[c.record() for c in selected])
        self.history.append(row)
        (self.folder/'history.json').write_text(json.dumps(self.history, indent=2), encoding='utf-8')
        print('[PAPER ACCEPT] ' + json.dumps({k:v for k,v in row.items() if k != 'actions'}), flush=True)

    def candidates(self):
        started = time.perf_counter()
        circuit, count = self.current, self.count
        mask = (1 << count) - 1
        values = trace(circuit, self.inputs, count, self.stats)
        bad = wrong(circuit, values, self.reference)
        generation_inputs = pack(random_inputs(circuit, self.args.patterns, int(self.rng.integers(1 << 31))), circuit.inputs)
        sample = trace(circuit, generation_inputs, self.args.patterns)
        self.stats['generation_simulations'] += 1
        self.stats['generation_gate_evaluations'] += len(circuit.gates)
        smask = (1 << self.args.patterns) - 1
        cones = FanoutCones(circuit)
        candidates = []
        for target in sorted(circuit.producer, key=lambda n: (-cones.potential(n), n)):
            if self.expired():
                break
            self.stats['screened_roots'] += 1
            majority = int(sample[target].bit_count() * 2 > self.args.patterns)
            raw = [((), 15 if majority else 0, circuit.lib.mapper.plans[15 if majority else 0], 15, 'root_resubstitution')]
            for names in screened_sets(circuit, target, sample, self.args.patterns,
                                       self.args.pool, self.args.pairs, self.rng):
                self.stats['screened_divisor_sets'] += 1
                table = consistent_table([sample[n] for n in names], sample[target], smask)
                if table is None:
                    continue
                truth, plan = circuit.lib.mapper.complete(*table)
                raw.append((names, truth, plan, table[0], 'root_resubstitution'))
            self.stats['consistent_functions'] += len(raw)
            if self.args.subfunctions:
                raw.extend((*p,'cell_subfunction') for p in subfunction_proposals(circuit, target,
                    sample, self.args.patterns,self.mapper,divisor_pool(circuit,target),self.stats,consistent_table))
            proposals, seen = [], set()
            for names, truth, plan, care, family in raw:
                # Only used leaves rescue logic in the MFFC; unused divisors do not.
                gain = cones.saved_with_inputs(target, used_divisors(names, plan)) - plan.units
                if gain <= 0:
                    continue
                word = plan_value(plan, names, circuit.lib, values, mask)
                if word in seen:
                    continue
                seen.add(word)
                proposals.append((names, truth, plan, care, gain, word, family))
            if not proposals:
                continue
            flip_started = time.perf_counter()
            flipped = wrong(circuit, trace(circuit, self.inputs, count, self.stats,
                                           base=values, flip=target), self.reference)
            self.times['root_flip_subset_of_generation'] += time.perf_counter()-flip_started
            measured = []
            for names, truth, plan, care, gain, word, family in proposals:
                delta = values[target] ^ word
                cbad = (bad & (delta ^ mask)) | (flipped & delta)
                self.stats['candidate_function_evaluations'] += 1
                measured.append(Candidate(target, names, truth, plan, gain, cbad,
                                          cbad.bit_count(), (cbad & ~bad).bit_count(), care, family))
            # Fixed top-k per root, preserving both min-error and area alternatives.
            selected = area_shortlist(measured,self.args.limit,self.args.per_root)
            self.stats['infeasible_function_skips_before_area'] += sum(c.error > self.args.limit for c in measured)
            area_started = time.perf_counter()
            for c in selected:
                trial = self.cleanup(circuit.materialize([c.replacement()]))
                self.stats['individual_area_materializations'] += 1
                actual_gain = circuit.units - trial.units
                if actual_gain > 0:
                    candidates.append(replace(c, gain=actual_gain))
            self.times['area_materialization_subset_of_generation'] += time.perf_counter()-area_started
        self.times['candidate_generation_and_scoring'] += time.perf_counter() - started
        return candidates, bad

    def apply(self, selected, phase):
        started = time.perf_counter()
        try:
            trial = self.cleanup(self.current.materialize([c.replacement() for c in selected]))
            if self.args.cut_remap:
                trial = cut_normalize(trial,self.mapper,self.stats)
        except ValueError as exc:
            self.stats['invalid_joint_graphs'] += 1
            return False, dict(reason='invalid_joint_graph', error=str(exc))
        values = trace(trial, self.inputs, self.count, self.stats)
        self.stats['joint_verifications' if len(selected) > 1 else 'single_verifications'] += 1
        bad = wrong(trial, values, self.reference)
        self.times['materialization_and_actual_verification'] += time.perf_counter() - started
        if len(selected) == 1 and bad != selected[0].bad:
            raise AssertionError('One-target CPM differs from full vector propagation')
        detail = dict(result_error_count=bad.bit_count(), result_area=trial.lib.area(trial.units),
                      batch_size=len(selected), actions=[c.record() for c in selected])
        if bad.bit_count() > self.args.limit or trial.units >= self.current.units:
            detail['reason'] = 'actual_error_exceeds_bound' if bad.bit_count() > self.args.limit else 'no_actual_area_gain'
            return False, detail
        self.current = trial
        self.checkpoint(trial, phase, selected, bad)
        return True, detail

    def run(self):
        phase, stop = 'batch', 'no_acceptable_screened_candidate'
        events = []
        while not self.expired():
            self.iteration += 1
            candidates, bad = self.candidates()
            event = dict(iteration=self.iteration, phase=phase, candidates=len(candidates),
                         current_error_count=bad.bit_count())
            if phase == 'batch':
                tick = time.perf_counter()
                selected = knapsack(candidates, self.args.limit - bad.bit_count())
                self.times['knapsack'] += time.perf_counter() - tick
                event['upper_error_estimate_count'] = bad.bit_count() + sum(c.weight for c in selected)
            else:
                selected = sorted(candidates, key=lambda c: (c.error, -c.gain, c.target, c.names))[:1]
                if selected and selected[0].error > self.args.limit:
                    selected = []
            if not selected:
                event['reason'] = 'no_feasible_selected_candidate'
                accepted = False
            else:
                accepted, detail = self.apply(selected, phase)
                event.update(detail)
            event['accepted'] = accepted
            events.append(event)
            (self.folder/'iterations.json').write_text(json.dumps(events, indent=2), encoding='utf-8')
            print('[PAPER ROUND] ' + json.dumps({k:v for k,v in event.items() if k != 'actions'}), flush=True)
            if not accepted:
                if phase == 'batch':
                    phase = 'minimum_error_single'
                else:
                    break
        if self.expired():
            stop = 'time_limit'
        return dict(stop_reason=stop, search_seconds=time.perf_counter()-self.started,
                    stats=dict(self.stats), component_seconds=dict(self.times), iterations=len(events))


def main(args):
    started = time.perf_counter()
    folder = Path(args.output).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    (folder/'source').mkdir()
    hashes = {}
    for name in ('paper_resub_netlist.py', 'logic.py', 'netlist.py', 'frontier.py', 'screen.py',
                 'simulate.py', 'cutmap.py', 'weighted_cost.py', 'runtime.py', 'paper_netlist_subfunctions.py'):
        path = ROOT/name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copy2(path, folder/'source'/name)
    lib = Library(ROOT/'assets/stdcells.lib')
    path = Path(args.start) if args.start else ROOT/'assets'/(args.design+'.v')
    source = Circuit.read(path, lib)
    matrices = [np.random.default_rng(np.random.SeedSequence([20261025, DESIGNS.index(args.design), split])).integers(
        0, 2, (10000, len(source.inputs)), dtype=np.uint8) for split in range(3)]
    inputs = [pack(m, source.inputs) for m in matrices]
    references = [outputs(source, trace(source, p, 10000)) for p in inputs]
    args.limit = int(np.floor(args.er * 10000 + 1e-9))
    protocol = dict(vars(args), source=str(path.resolve()), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    code_sha256=hashes, library_sha256=hashlib.sha256((ROOT/'assets/stdcells.lib').read_bytes()).hexdigest(),
                    count_per_evaluation_split=10000, seed_scheme=[20261025, DESIGNS.index(args.design), 'split'],
                    controller='paper_screened_numeric_no_agent', exhaustive_lac=False,
                    deviations=['screened divisor sets', 'physical standard-cell area',
                                'local exact normalization instead of ABC compress2rs', '10000 evaluation vectors'],
                    selection='minimum area among development-eligible checkpoints, before final evaluation')
    (folder/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    search = Search(source, inputs[0], references[0], 10000, args, folder)
    outcome = search.run()
    selection_started = time.perf_counter()
    rows = []
    for row in search.history:
        circuit = Circuit.read(folder/row['path'], lib)
        bad = wrong(circuit, trace(circuit, inputs[1], 10000), references[1])
        rows.append(dict(row, development_error_count=bad.bit_count()))
    development_limit = int(np.floor(args.limit * args.development_fraction + 1e-9))
    eligible = [r for r in rows if r['development_error_count'] <= development_limit]
    selected = min(eligible, key=lambda r: (r['units'], r['index']))
    winner = Circuit.read(folder/selected['path'], lib)
    winner.write(folder/'selected.v')
    final_bad = wrong(winner, trace(winner, inputs[2], 10000), references[2])
    original_numpy = validate_numpy(source, matrices[2])
    final_numpy = validate_numpy(winner, matrices[2])
    expected = sum(bool(x) << i for i, x in enumerate(np.any(original_numpy != final_numpy, axis=1)))
    if final_bad != expected:
        raise AssertionError('Independent NumPy verifier disagrees with bit-parallel final ER')
    strict_pass = final_bad.bit_count() <= args.limit
    winner.write(folder/('best.v' if strict_pass else 'selected_failed_final.v'))
    outcome.update(design=args.design, er_bound=args.er, original_area=lib.area(source.units),
                   area=lib.area(winner.units), reduction=1-winner.units/source.units,
                   train_er=selected['train_error_count']/10000,
                   development_er=selected['development_error_count']/10000,
                   final_er=final_bad.bit_count()/10000, strict_pass=strict_pass,
                   promoted_netlist='best.v' if strict_pass else None,
                   development_limit=development_limit, selected_checkpoint=selected['path'],
                   checkpoints=rows, postselection_full_simulations=len(rows)+1,
                   reference_full_simulations=3, independent_numpy_simulations=2,
                   selection_and_final_seconds=time.perf_counter()-selection_started,
                   whole_seconds=time.perf_counter()-started,
                   caveat='Observed fixed streams; netlist migration, not identical paper starting AIG/protocol.')
    (folder/'report.json').write_text(json.dumps(outcome, indent=2), encoding='utf-8')
    print('[PAPER RESULT] ' + json.dumps({k:v for k,v in outcome.items() if k != 'checkpoints'}), flush=True)
    if not strict_pass:
        print('[PAPER FINAL FAIL] Final vector ER exceeds the bound; no best.v is promoted. '
              'Diagnostic selected_failed_final.v and the failed report are retained.',flush=True)
    return 0 if strict_pass else 2


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--design', choices=DESIGNS, default='c880')
    p.add_argument('--start', help='accurate reference netlist; not an approximate warm start')
    p.add_argument('--output', required=True)
    p.add_argument('--er', type=float, default=.005)
    p.add_argument('--seconds', type=float, default=180)
    p.add_argument('--seed', type=int, default=20261027)
    p.add_argument('--patterns', type=int, default=64)
    p.add_argument('--pool', type=int, default=32)
    p.add_argument('--pairs', type=int, default=96)
    p.add_argument('--per-root', type=int, default=4)
    p.add_argument('--subfunctions', action='store_true', help='explicit netlist-only internal-cell extension')
    p.add_argument('--cell-remap', action='store_true', help='exact standard-cell cofactor remapping up to 4 inputs')
    p.add_argument('--cut-remap', action='store_true', help='bounded exact standard-cell cut refactoring after accepted proposals')
    p.add_argument('--development-fraction', type=float, default=.9)
    args = p.parse_args()
    if not 0 < args.er < 1 or not 0 < args.development_fraction <= 1:
        p.error('ER and development fraction must be positive fractions')
    if min(args.seconds, args.patterns, args.pool, args.pairs, args.per_root) <= 0:
        p.error('All screening and time settings must be positive')
    raise SystemExit(main(args))
