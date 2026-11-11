"""Expose Liberty subfunctions without converting the circuit to an AIG.

A substitution inside one physical cell is recomposed into a truth table over
at most four real boundary nets and mapped back to standard cells. The caller
measures the resulting full physical circuit; intermediate expressions carry
no area credit. This is an explicit netlist extension to the paper migration.
"""
from logic import evaluate
from cutmap import variable
from netlist import ZERO, ONE
from simulate import plan_value
from frontier import normalize, boundary_cuts
from screen import FanoutCones


def cell_normalize(circuit, mapper):
    """Exact catalog remapping also handles 4-input cell -> 3-input cofactors.

    The older shared normalizer only remaps cells with <=2 distinct nets. That
    misses NAND4(1,a,b,c) -> NAND3(a,b,c), among other ordinary cleanups.
    """
    current = normalize(circuit)
    while True:
        selected = []
        for g in current.gates:
            cell = current.lib.cells[g.cell]
            if len(cell.functions) != 1:
                continue
            names = tuple(sorted({current.resolve(g.pins[p]) for p in cell.inputs}-{ZERO,ONE}))
            if not 3 <= len(names) <= 4:
                continue
            mask = (1 << (1 << len(names)))-1
            values = {ZERO:0,ONE:mask,**{name:variable(len(names),i) for i,name in enumerate(names)}}
            truth = evaluate(next(iter(cell.functions.values())),{'0':0,'1':mask,
                **{p:values[current.resolve(g.pins[p])] for p in cell.inputs}},mask)
            plan = mapper.catalog[len(names)].get(truth)
            if plan is not None and plan.units < cell.units:
                selected.append((g.pins[next(iter(cell.functions))],names,plan))
        if not selected:
            return current
        trial = normalize(current.materialize(selected))
        if trial.units >= current.units:
            raise AssertionError('Exact cell cofactor remapping did not reduce physical area')
        current = trial


def cut_normalize(circuit, mapper, stats, passes=3):
    """Bounded exact netlist refactoring; direct library cells, never AIGs."""
    current = cell_normalize(circuit,mapper)
    for _ in range(passes):
        cones = FanoutCones(current)
        selected = []
        for target in current.producer:
            best = None
            for names in boundary_cuts(current,target,width=4):
                mask = (1 << (1 << len(names)))-1
                values = {ZERO:0,ONE:mask,**{n:variable(len(names),i) for i,n in enumerate(names)}}
                def visit(n):
                    n = current.resolve(n)
                    if n in values:
                        return values[n]
                    g = current.producer[n]
                    c = current.lib.cells[g.cell]
                    env = {'0':0,'1':mask,**{p:visit(g.pins[p]) for p in c.inputs}}
                    for pin,fun in c.functions.items():
                        if pin in g.pins and g.pins[pin] not in values:
                            values[g.pins[pin]] = evaluate(fun,env,mask)
                    return values[n]
                truth = visit(target)
                stats['exact_cut_truth_tables'] += 1
                plan = mapper.catalog[len(names)].get(truth)
                if plan is None:
                    continue
                gain = cones.saved_with_inputs(target,names)-plan.units
                if gain > 0 and (best is None or gain > best[0]):
                    best = gain,names,plan
            if best:
                selected.append((best[0],target,best[1],best[2]))
        if not selected:
            break
        selected.sort(key=lambda p:(-p[0],p[1]))
        trial = cell_normalize(current.materialize([(t,n,p) for _,t,n,p in selected[:64]]),mapper)
        stats['exact_cut_materializations'] += 1
        if trial.units >= current.units:
            break
        current = trial
    return current


def locations(tree, path=()):
    if tree[0] == 'var':
        return
    if path:
        yield path, tree
    for i, child in enumerate(tree[1:], 1):
        yield from locations(child, path+(i,))


def substitute(tree, path):
    if not path:
        return ('var', '__paper_subfunction__')
    parts = list(tree)
    parts[path[0]] = substitute(parts[path[0]], path[1:])
    return tuple(parts)


def variables(tree):
    if tree[0] == 'var':
        return {tree[1]}
    return set().union(*(variables(c) for c in tree[1:]))


def proposals(circuit, target, sample, patterns, mapper, pool, stats, table_builder=None):
    gate = circuit.producer[target]
    cell = circuit.lib.cells[gate.cell]
    if len(cell.functions) != 1 or len(cell.inputs) <= 2:
        return
    tree = next(iter(cell.functions.values()))
    mask = (1 << patterns)-1
    signals = {p:circuit.resolve(gate.pins[p]) for p in cell.inputs}
    env = {'0':0, '1':mask, **{p:sample[n] for p,n in signals.items()}}
    seen = set()
    for path, expression in list(locations(tree))[:12]:
        word = evaluate(expression, env, mask)
        rewritten = substitute(tree, path)
        untouched = variables(rewritten)-{'__paper_subfunction__', '0', '1'}
        names_needed = {signals[p] for p in untouched}-{ZERO,ONE}
        majority = int(word.bit_count()*2 > patterns)
        funcs = [((), circuit.lib.mapper.plans[15 if majority else 0])]
        # Bound both one- and two-divisor construction inside the physical cell.
        ranked = sorted(pool, key=lambda n: (min((sample[n]^word).bit_count(),
                        patterns-(sample[n]^word).bit_count()), n))[:16]
        for n in ranked:
            difference = sample[n]^word
            if difference in (0,mask):
                funcs.append(((n,), circuit.lib.mapper.plans[5 if difference else 10]))
        if table_builder is not None:
            pairs = set()
            for offset in (1,2,4):
                for i,a in enumerate(ranked):
                    if len(ranked) < 2 or len(pairs) >= 24:
                        break
                    b = ranked[(i+offset)%len(ranked)]
                    pair = tuple(sorted((a,b)))
                    if a == b or pair in pairs:
                        continue
                    pairs.add(pair)
                    stats['screened_subfunction_pair_sets'] += 1
                    table = table_builder([sample[n] for n in pair],word,mask)
                    if table is not None:
                        _,plan = circuit.lib.mapper.complete(*table)
                        funcs.append((pair,plan))
        for divs, plan in funcs:
            names = tuple(sorted(names_needed | (set(divs)-{ZERO,ONE})))
            if len(names) > 4:
                continue
            stats['screened_subfunction_replacements'] += 1
            n = len(names)
            truth_mask = (1 << (1 << n))-1
            values = {ZERO:0, ONE:truth_mask,
                      **{name:variable(n,i) for i,name in enumerate(names)}}
            replacement = plan_value(plan, divs, circuit.lib, values, truth_mask)
            truth = evaluate(rewritten, {'0':0,'1':truth_mask,
                **{p:values.get(s,0) for p,s in signals.items()},
                '__paper_subfunction__':replacement}, truth_mask)
            if (names,truth) in seen:
                continue
            seen.add((names,truth))
            mapped = mapper.map(n,truth)
            stats['subfunction_truth_tables_mapped'] += 1
            yield names, truth, mapped, (1 << (1 << n))-1
