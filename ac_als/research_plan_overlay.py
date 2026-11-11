"""Immutable multi-LAC overlay: exact sampled joint response, one final commit.

No Circuit is constructed for a proposal. Original cell liveness is counted as
a union, including rescued cones and multioutput instances. This is the exact
PRE-cleanup area of Circuit.materialize, not a sum of individual MFFC gains.
"""
from dataclasses import dataclass
from collections import Counter
from logic import evaluate
from netlist import ZERO, ONE


@dataclass(frozen=True)
class Action:
    target: str
    names: tuple
    plan: object

    def replacement(self):
        return self.target, self.names, self.plan


def identity(actions):
    return tuple(sorted(actions, key=lambda a: a.target))


def used_leaves(plan):
    result = set()
    def visit(p):
        if p.name == 'leaf':
            if p.leaf >= 0: result.add(p.leaf)
        else:
            for child in p.children: visit(child)
    visit(plan)
    return result


class Overlay:
    def __init__(self, circuit, packed_inputs, references, count, counters=None):
        self.base = circuit
        self.inputs = packed_inputs
        self.references = references
        self.count = count
        self.mask = (1 << count) - 1
        self.stats = counters if counters is not None else Counter()
        self.epoch = 0

    def action_map(self, actions):
        result = {}
        for a in actions:
            if a.target in result: raise ValueError('Duplicate plan target')
            if self.base.resolve(a.target) != a.target or a.target not in self.base.producer:
                raise ValueError('Target must be a canonical driven net')
            result[a.target] = a
        return result

    def dependencies(self, net, amap):
        net = self.base.resolve(net)
        if net in (ZERO, ONE) or net in self.base.inputs: return ()
        if net in amap:
            a = amap[net]
            return tuple(self.base.resolve(a.names[i]) for i in sorted(used_leaves(a.plan)))
        g = self.base.producer[net]
        return tuple(self.base.resolve(g.pins[p]) for p in self.base.lib.cells[g.cell].inputs)

    def graph(self, actions):
        amap = self.action_map(actions)
        done = set(); visiting = set(); order = []
        def visit(n):
            n = self.base.resolve(n)
            if n in done: return
            if n in visiting: raise ValueError('Combined LAC dependency cycle: ' + n)
            visiting.add(n)
            for d in self.dependencies(n, amap): visit(d)
            visiting.remove(n); done.add(n); order.append(n)
        # Check dormant actions too: a future edit must not activate a hidden cycle.
        for n in (*self.base.outputs, *amap): visit(n)
        return amap, order

    def _values(self, amap, order, split, force=None, all_nodes=False):
        vals = {**self.inputs[split], ZERO: 0, ONE: self.mask}
        visiting = set()
        def ev(n):
            n = self.base.resolve(n)
            if n in vals: return vals[n]
            if n in visiting: raise ValueError('Combined LAC dependency cycle: ' + n)
            visiting.add(n)
            if force is not None and n == force[0]:
                value = force[1]
            elif n in amap:
                a = amap[n]; memo = {}
                def plan(p):
                    if p.name == 'leaf':
                        return ev(a.names[p.leaf]) if p.leaf >= 0 else (0, self.mask)[p.leaf + 2]
                    if p not in memo:
                        c = self.base.lib.cells[p.name]
                        memo[p] = evaluate(next(iter(c.functions.values())),
                            {'0': 0, '1': self.mask, **dict(zip(c.inputs, map(plan, p.children)))}, self.mask)
                        self.stats['overlay_cell_output_evaluations'] += 1
                    return memo[p]
                value = plan(a.plan)
            else:
                g = self.base.producer[n]; c = self.base.lib.cells[g.cell]
                pin = next(p for p in c.functions if g.pins.get(p) == n)
                value = evaluate(c.functions[pin], {'0': 0, '1': self.mask,
                    **{p: ev(g.pins[p]) for p in c.inputs}}, self.mask)
                self.stats['overlay_cell_output_evaluations'] += 1
            vals[n] = value; visiting.remove(n); return value
        for n in self.base.outputs: ev(n)
        if all_nodes:
            for n in self.base.producer: ev(n)
        bad = 0
        for n in self.base.outputs: bad |= ev(n) ^ self.references[split][n]
        self.stats['overlay_forced_propagations' if force else 'overlay_joint_propagations'] += 1
        return vals, bad

    def area(self, actions):
        amap, _ = self.graph(actions)
        visited = set(); gates = set(); plans = set()
        def pv(a, p):
            if p.name == 'leaf':
                if p.leaf >= 0: visit(a.names[p.leaf])
                return
            key = (a.target, p)
            if key in plans: return
            plans.add(key)
            for child in p.children: pv(a, child)
        def visit(n):
            n = self.base.resolve(n)
            if n in visited: return
            visited.add(n)
            if n in (ZERO, ONE) or n in self.base.inputs: return
            if n in amap: pv(amap[n], amap[n].plan)
            else:
                g = self.base.producer[n]; gates.add(id(g))
                for p in self.base.lib.cells[g.cell].inputs: visit(g.pins[p])
        for n in self.base.outputs: visit(n)
        original = sum(self.base.lib.cells[g.cell].units for g in self.base.gates if id(g) in gates)
        added = sum(self.base.lib.cells[p.name].units for _, p in plans)
        return original + added

    def score(self, actions, all_nodes=False):
        amap, order = self.graph(actions)
        result = [self._values(amap, order, k, all_nodes=all_nodes) for k in range(len(self.inputs))]
        if not all_nodes:
            # Frontier states need output signatures, not megabytes of internal
            # node words per proposal. Refresh alone retains complete traces.
            result = [({self.base.resolve(n):v[self.base.resolve(n)] for n in self.base.outputs},b)
                      for v,b in result]
        return dict(actions=identity(actions), units=self.area(actions),
                    bad=tuple(b for _, b in result), values=tuple(v for v, _ in result))

    def refresh(self, actions, roots, reason, previous=None):
        """Recollect on the full proposed circuit, not the immutable base."""
        reusable = previous is not None and previous['context_key'] == identity(actions)
        state = previous['state'] if reusable else self.score(actions, all_nodes=True)
        amap, order = self.graph(actions); responses = {}
        for root in roots:
            if reusable and root in previous['responses']:
                responses[root] = previous['responses'][root]
                self.stats['unchanged_context_response_reuses'] += 1
                continue
            forced = []; output_deltas = []; sensitive = repaired = introduced = 0
            for k, vals in enumerate(state['values']):
                old = vals[root]
                flipped_values, flip = self._values(amap, order, k, force=(root, old ^ self.mask))
                output_deltas.append({n: vals[self.base.resolve(n)] ^ flipped_values[self.base.resolve(n)]
                                      for n in self.base.outputs})
                bad = state['bad'][k]
                # Force-0/1 errors obtained exactly from the vectorwise flip.
                forced.append(((bad & (old ^ self.mask)) | (flip & old),
                               (bad & old) | (flip & (old ^ self.mask))))
                sensitive += (bad ^ flip).bit_count()
                repaired += (bad & ~flip).bit_count()
                introduced += (flip & ~bad).bit_count()
            responses[root] = dict(forced=forced, sensitivity=sensitive,
                                   repair_overlap=repaired, new_error_overlap=introduced,
                                   output_deltas=output_deltas)
        self.epoch += 1; self.stats['response_refreshes'] += 1
        return dict(epoch=self.epoch, reason=reason, state=state, responses=responses,
                    context_key=identity(actions))

    def propose(self, context, action):
        """Exact single-edit PO response in a JOINT plan context, no propagation.

        The edited function must use signals outside the current dynamic TFO.
        Each output is a Boolean function of the forced root: selecting its
        previously measured 0/1 cofactor is exact, even for reconvergent logic.
        Multiple edits must NOT share this context: refresh after each plan edit
        before using this formula again. This does not XOR independent LAC errors.
        """
        parent = context['state']
        if parent['actions'] != context['context_key']:
            raise ValueError('Stale response context')
        forbidden = self.forbidden(context, action.target)
        if any(self.base.resolve(action.names[i]) in forbidden for i in used_leaves(action.plan)):
            raise ValueError('Response edit depends on its dynamic fanout')
        updated = {a.target:a for a in parent['actions']}; updated[action.target] = action
        actions = identity(updated.values()); self.graph(actions)
        response = context['responses'][action.target]
        from simulate import plan_value
        values = []; bad = []
        for k, original in enumerate(parent['values']):
            aliases = {n:original[self.base.resolve(n)] for n in action.names}
            word = plan_value(action.plan, action.names, self.base.lib, aliases, self.mask)
            delta = original[action.target] ^ word
            result = {self.base.resolve(n): original[self.base.resolve(n)] ^
                      (delta & response['output_deltas'][k][n]) for n in self.base.outputs}
            error = 0
            for n in self.base.outputs: error |= result[self.base.resolve(n)] ^ self.references[k][n]
            values.append(result); bad.append(error)
        self.stats['exact_response_edit_evaluations'] += 1
        return dict(actions=actions, units=self.area(actions), values=tuple(values), bad=tuple(bad))

    def forbidden(self, context, target):
        if context['state']['actions'] != context['context_key']:
            raise ValueError('Stale response context')
        if 'dynamic_reverse' not in context:
            amap, _ = self.graph(context['context_key']); reverse = {}
            for n in self.base.producer:
                for d in self.dependencies(n, amap): reverse.setdefault(d, set()).add(n)
            context['dynamic_reverse'] = reverse
            context['forbidden_by_root'] = {}
        cache = context['forbidden_by_root']
        if target not in cache:
            result = {target}; queue = [target]
            while queue:
                for n in context['dynamic_reverse'].get(queue.pop(), ()):
                    if n not in result: result.add(n); queue.append(n)
            cache[target] = result
        return cache[target]

    def commit(self, actions):
        self.graph(actions)
        self.stats['final_materializations'] += 1
        physical = self.base.materialize([a.replacement() for a in identity(actions)])
        expected = self.area(actions)
        if physical.units != expected:
            raise AssertionError(('Overlay physical area mismatch', expected, physical.units))
        return physical
