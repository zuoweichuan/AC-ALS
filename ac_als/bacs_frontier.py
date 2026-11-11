"""Fast numerical netlist search with bounded candidates and exact remapping.

Independent entry: the frozen BACS and Agent protocols are not modified.
Approximate choices belong to this implementation; ABC only proves and maps the
selected function. Default input groups contain 10,000 vectors each.
"""
import runtime
import argparse
from collections import defaultdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from bacs_numeric import (NumericSearch, Metric, Move, bits, number, cleanup,
                          learned_sets, save)
from bacs_data import (ROOT, DESIGNS, LIBRARY_PATH, source_path, load, matrix, preflight)
from logic import Library, evaluate
from cutmap import variable
from netlist import Circuit, Gate, ZERO, ONE
from simulate import trace, outputs, pack, validate_numpy, plan_value
from paper_resub_netlist import divisor_pool, consistent_table, used_divisors
from frontier import boundary_cuts
from paper_netlist_subfunctions import proposals as subfunctions
from screen import FanoutCones
from hybrid_identity import structural_id
from numeric_exact import ExactBackend
from numeric_logic_view import expand_cells


def implication_sets(circuit, target, values, count, pool_limit, pair_limit, stats):
    """Signature implication indexing, with bounded pair construction.

    No scan over all circuit pairs. Native fanins and each implication class
    receive separate quotas so the first class cannot consume the entire pool.
    """
    mask = (1 << count) - 1
    t = values[target]
    pool = divisor_pool(circuit, target)
    def rank(n):
        d = (values[n] ^ t).bit_count()
        return min(d, count-d), -circuit.level[n], n
    ordered = sorted(pool, key=rank)
    g = circuit.producer[target]
    direct = sorted({circuit.resolve(g.pins[p]) for p in circuit.lib.cells[g.cell].inputs} & set(pool))
    selected = list(dict.fromkeys(direct + ordered))[:pool_limit]
    seen = set()
    for n in selected:
        yield (n,)
    pairs = []
    def add(a, b):
        key = tuple(sorted((a, b)))
        if a != b and key not in seen and len(pairs) < pair_limit:
            seen.add(key)
            pairs.append(key)
    native_quota = max(1, pair_limit // 4)
    # Round-robin across retained native inputs. The old nested loop exhausted
    # the entire quota at the first fanin, silently dropping the other anchors.
    for b in selected:
        for a in direct:
            add(a, b)
            if len(pairs) >= native_quota:
                break
        if len(pairs) >= native_quota:
            break
    # Examine a wider cheap signature pool than the expensive function pool.
    indexed = list(dict.fromkeys(direct + ordered))[:max(pool_limit, 256)]
    classes = []
    for kind in ('and', 'or'):
        for polarity in (False, True):
            group = []
            for n in indexed:
                v = values[n] ^ (mask if polarity else 0)
                if ((t & (v ^ mask)) == 0 if kind == 'and' else ((t ^ mask) & v) == 0):
                    group.append(n)
            classes.append(group)
    quota = max(1, (pair_limit-native_quota)//8)
    for i, j in ((0,0),(0,1),(1,1),(2,2),(2,3),(3,3)):
        left, right = classes[i], classes[j]
        if not right:
            continue
        start = len(pairs)
        # Diagonal sweeps cover distinct partners without constructing a product.
        for offset in range(4):
            for index, a in enumerate(left):
                add(a, right[(index+offset+(1 if i == j else 0)) % len(right)])
                if len(pairs)-start >= quota:
                    break
            if len(pairs)-start >= quota:
                break
    # Include native/near-native XOR-style functions too; implication is a filter,
    # not a claim that all useful replacements are AND/OR.
    for offset in range(1, 5):
        for i, a in enumerate(selected):
            if len(pairs) >= pair_limit:
                break
            add(a, selected[(i+offset) % len(selected)])
    stats['signature_divisors_indexed'] += len(indexed)
    stats['signature_pair_trials'] += len(pairs)
    yield from pairs


def response_truth(code,loss0,loss1,base_loss,width,penalty=1):
    """One analytic output-loss-minimizing truth, with optional new-error penalty."""
    a=np.bincount(code,weights=loss0+(penalty-1)*np.maximum(loss0-base_loss,0),minlength=1<<width)
    b=np.bincount(code,weights=loss1+(penalty-1)*np.maximum(loss1-base_loss,0),minlength=1<<width)
    return sum(int(y<x)<<i for i,(x,y) in enumerate(zip(a,b)))


def truth_supports(circuit,target,cones,raw_names,maximum):
    """Reserve structural supports before spending the quota on signal matches."""
    gate=circuit.producer[target]
    direct=tuple(sorted({circuit.resolve(gate.pins[p]) for p in circuit.lib.cells[gate.cell].inputs}-{ZERO,ONE}))
    removed=cones.deletions(target)
    boundary=set()
    for g in removed.values():
        for n in cones.dependencies[id(g)]:
            producer=circuit.producer.get(n)
            if n not in (ZERO,ONE) and (producer is None or id(producer) not in removed):
                boundary.add(n)
    mffc=tuple(sorted(boundary))
    cuts=boundary_cuts(circuit,target,width=4)
    choices=[direct,mffc,*reversed(cuts[-2:]),*raw_names]
    return list(dict.fromkeys(s for s in choices if 0<len(s)<=4))[:maximum]


class VirtualArea:
    """Deletion/rescue area, plus the residual cost of a physical multiout cell.

    Estimates exclude downstream simplification and joint sharing. Acceptance
    always uses a materialized circuit, never a sum of these estimated gains.
    """
    def __init__(self, circuit, mapper):
        self.circuit, self.mapper = circuit, mapper
        self.cones = FanoutCones(circuit)
        self.partial = {}
        live = {circuit.resolve(n) for n in circuit.outputs} | set(circuit.fanouts)
        for target, g in circuit.producer.items():
            cell = circuit.lib.cells[g.cell]
            survivors = [p for p in cell.functions if p in g.pins and g.pins[p] != target and g.pins[p] in live]
            credit = 0
            if len(cell.functions) > 1 and len(survivors) == 1:
                names = tuple(sorted({circuit.resolve(g.pins[p]) for p in cell.inputs}-{ZERO,ONE}))
                if len(names) <= 4:
                    mask = (1 << (1 << len(names))) - 1
                    vals = {ZERO:0, ONE:mask, **{n:variable(len(names),i) for i,n in enumerate(names)}}
                    truth = evaluate(cell.functions[survivors[0]],
                                     {'0':0,'1':mask,**{p:vals[circuit.resolve(g.pins[p])] for p in cell.inputs}}, mask)
                    credit = max(0, cell.units-mapper.cost(mapper.map(len(names), truth)))
            self.partial[target] = credit

    def gain(self, target, names, plan):
        names = used_divisors(names, plan)
        return (self.cones.saved_with_inputs(target, names) + self.partial[target]
                - self.mapper.cost(plan))


def pareto_choices(items, maximum=8):
    """Preserve error, positive error increment and deletion gain tradeoffs."""
    unique = {}
    for m in items:
        key = (m.names, m.truth)
        if key not in unique or m.gain > unique[key].gain:
            unique[key] = m
    front = []
    for m in sorted(unique.values(), key=lambda x:(-x.gain,x.error,x.weight,x.names,x.truth)):
        # Equal scalar costs do not imply equal compensation/joint-search
        # opportunities. Only identical output responses can be safely merged.
        if not any(p.signature==m.signature and p.gain >= m.gain and p.error <= m.error and p.weight <= m.weight for p in front):
            front.append(m)
    if len(front) <= maximum:
        return front
    queues = [sorted(front,key=lambda m:(-m.gain,m.error)),
              sorted(front,key=lambda m:(m.error,-m.gain)),
              sorted(front,key=lambda m:(m.weight,-m.gain)),
              sorted(front,key=lambda m:(-m.gain/max(1,m.weight),m.error))]
    chosen = []
    for i in range(maximum):
        for queue in queues:
            if len(chosen) >= maximum:
                break
            if i < len(queue) and queue[i] not in chosen:
                chosen.append(queue[i])
    return chosen


def bounded_batch(candidates,capacity,seconds,stats):
    """Sparse multiple-choice DP with an explicit wall bound, not LAC enumeration.

    On timeout return the best valid partial selection; no unprocessed item is
    counted. Every chosen target occurs once, even when a group is interrupted.
    """
    deadline=time.perf_counter()+seconds;groups=defaultdict(list)
    for m in candidates:
        if m.gain>0 and m.weight<=capacity:
            groups[m.target].append(m)
    states={0:(0,None)}
    # Keep the established stable group order. Reordering equally optimal DP
    # choices can change subsequent nonlinear circuit interactions materially.
    for target in sorted(groups):
        previous=states;updated=dict(previous);capped=False;checks=0
        for gain,(weight,chain) in previous.items():
            for m in groups[target]:
                g,w=gain+m.gain,weight+m.weight
                if w<=capacity and w<updated.get(g,(capacity+1,None))[0]:
                    updated[g]=(w,(chain,m))
                checks+=1
                if checks%256==0 and time.perf_counter()>=deadline:
                    capped=True
                    break
            if capped:
                break
        states={};lowest=capacity+1
        for g in sorted(updated,reverse=True):
            if updated[g][0]<lowest:
                states[g]=updated[g];lowest=updated[g][0]
        if capped or time.perf_counter()>=deadline:
            stats['knapsack_time_capped']+=1
            break
    chain=states[max(states)][1];result=[]
    while chain is not None:
        chain,m=chain;result.append(m)
    return list(reversed(result))


def constant_outputs(circuit, replacements):
    """Change PO drivers without altering a PO signal's internal fanouts."""
    if set(replacements) & set(circuit.inputs):
        raise ValueError('Cannot detach an inout/input port')
    used = set(circuit.producer) | set(circuit.aliases) | set(circuit.inputs) | set(circuit.outputs)
    rename = {}
    for n in replacements:
        fresh = '__po_internal_%d' % len(rename)
        while fresh in used:
            fresh += '_'
        rename[n] = fresh
        used.add(fresh)
    r = lambda n: rename.get(n,n)
    aliases = {r(n):r(v) for n,v in circuit.aliases.items()}
    aliases.update({n:ONE if v else ZERO for n,v in replacements.items()})
    return Circuit(circuit.lib,circuit.inputs,circuit.outputs,
                   [Gate(g.name,g.cell,{p:r(n) for p,n in g.pins.items()}) for g in circuit.gates],
                   aliases,circuit.name)


class FrontierSearch(NumericSearch):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.formal_limit=self.metric.limit
        self.metric.limit=int(Decimal(self.formal_limit)*Decimal(str(self.args.training_fraction)))
        self.metric.cap=self.metric.limit+1
        if self.args.logic_view:
            self.current=expand_cells(self.current)
            self.lowest_units=self.current.units
            self.seen.add(structural_id(self.current))
            self.checkpoint('exact_expanded_search_view',[])
        self.backend = None if self.args.no_exact else ExactBackend(self.folder/'exact', LIBRARY_PATH,
            distro=self.args.wsl_distro, executable=self.args.abc, timeout=self.args.map_seconds,keep_sop=self.args.sop_view)
        self.mapping_calls = 0
        self.quantization_seen = set()
        self.word_area_gain=0
        self.near_events=[]
        self.refine('initial_exact')
        if self.backend and not any(r.get('cec_equivalent') for r in self.backend.rows) and not self.expired():
            raise RuntimeError('ABC exact backend failed its initial proof; inspect exact/mapping.json or explicitly use --no-exact')

    def remaining(self):
        return max(0.,self.args.seconds-(time.perf_counter()-self.started))

    def refine(self,phase):
        if self.backend is None or self.expired():
            return
        tick=time.perf_counter()
        trial=self.backend.optimize(self.current,self.remaining,variants=self.args.map_variants)
        self.times['exact_mapping'] += time.perf_counter()-tick
        self.stats['mapping_calls'] = len(self.backend.rows)
        self.stats['mapping_cec_proofs'] = sum(r.get('cec_equivalent',False) for r in self.backend.rows)
        self.stats['working_cec_proofs'] = sum(r.get('working_cec_equivalent',False) for r in self.backend.rows)
        if trial.units < self.current.units:
            working=self.current
            self.current=trial
            self.seen.add(structural_id(trial))
            if not self.args.logic_view and not self.args.sop_view:
                self.lowest_units=min(self.lowest_units,trial.units)
            self.checkpoint(phase,[])
            if self.args.logic_view or self.args.sop_view:
                self.current=working
        if self.args.sop_view and self.backend.working is not None:
            self.current=self.backend.working
            self.lowest_units=self.current.units
            self.seen.add(structural_id(self.current))
            self.checkpoint('exact_sop_search_view',[])

    def scored(self,*args,**kwargs):
        # The numerical response is shared across functions at this root. A
        # bounded near-miss pool is retained for *joint* compensation only.
        target,names,truth,plan,values,t,base_y,flip_y,base_loss,flip_loss,family = args[:11]
        penalty=args[11] if len(args)>11 else kwargs.get('penalty',1)
        word=plan_value(plan,names,self.source.lib,values,(1<<self.count)-1)
        delta=bits(word^t,self.count).astype(bool)
        total=int(base_loss[~delta].sum()+flip_loss[delta].sum())
        self.stats['function_error_evaluations']+=1
        ceiling=int(self.metric.limit*(self.args.near_ratio if self.args.compensate else 1))
        if total>ceiling:
            return None
        positive=int(np.maximum(flip_loss[delta]-base_loss[delta],0).sum())
        signature=hashlib.blake2b(np.where(delta,flip_y,base_y).tobytes(),digest_size=16).hexdigest()
        return Move(target,names,truth,plan,0,total,positive,signature,family,penalty)

    def materialized(self,circuit,items,base_total,allow_debt=False):
        # This stage deliberately allocates no candidate Circuit objects.
        good=[]
        for m in items:
            m.gain=self.area_model.gain(m.target,m.names,m.plan)
            if m.gain>0 or (allow_debt and m.gain>=-16 and m.error<base_total):
                good.append(m)
        chosen=pareto_choices(good,self.args.per_root)
        self.stats['virtual_area_scores']+=len(items)
        self.stats['pareto_candidates']+=len(chosen)
        return chosen

    def candidates(self,repair_only=False):
        started=time.perf_counter();circuit=self.current;mask=(1<<self.count)-1
        values=trace(circuit,self.inputs[0],self.count,self.stats)
        base_y=number(outputs(circuit,values),self.order,self.count)
        base_loss=self.metric.loss(base_y,self.reference[0]);base_total=int(base_loss.sum())
        self.area_model=VirtualArea(circuit,self.mapper)
        small=pack(self.rng.integers(0,2,(64,len(circuit.inputs)),dtype=np.uint8),circuit.inputs)
        sample=trace(circuit,small,64);self.stats['generation_64_simulations']+=1
        result=[];repair_contexts=[]
        targets=sorted(circuit.producer,key=lambda n:(-self.area_model.cones.potential(n),n))
        if repair_only:
            targets=targets[:self.args.repair_roots]
        for target in targets:
            if self.expired():
                break
            self.stats['screened_roots']+=1
            raw=[]
            if not repair_only:
                raw=[((),0,circuit.lib.mapper.plans[0],'constant'),
                     ((),15,circuit.lib.mapper.plans[15],'constant')]
                for names in implication_sets(circuit,target,sample,64,self.args.pool,self.args.pairs,self.stats):
                    self.stats['screened_divisor_sets']+=1
                    table=consistent_table([sample[n] for n in names],sample[target],(1<<64)-1)
                    if table is not None:
                        truth,plan=circuit.lib.mapper.complete(*table)
                        raw.append((names,truth,plan,'implication_resub'))
                if self.args.subfunctions:
                    raw.extend((n,t,p,'cell_subfunction') for n,t,p,_ in subfunctions(circuit,target,sample,64,
                        self.mapper,divisor_pool(circuit,target),self.stats,consistent_table))
            raw2=[];seen=set()
            for names,truth,plan,family in raw:
                if self.area_model.gain(target,names,plan)<=0:
                    continue
                key=(names,truth)
                if key not in seen:
                    seen.add(key);raw2.append((names,truth,plan,family))
            if not raw2 and not self.args.compensate and not self.args.response_truths:
                continue
            tick=time.perf_counter()
            flipped=trace(circuit,self.inputs[0],self.count,self.stats,base=values,flip=target)
            flip_y=number(outputs(circuit,flipped),self.order,self.count)
            flip_loss=self.metric.loss(flip_y,self.reference[0]);t=values[target]
            self.times['root_response_subset']+=time.perf_counter()-tick
            if self.args.compensate and base_total:
                possible=int(np.maximum(base_loss-flip_loss,0).sum())
                if possible:
                    repair_contexts.append((possible,target,t,flip_y,flip_loss))
                    repair_contexts=sorted(repair_contexts,key=lambda r:(-r[0],r[1]))[:self.args.repair_contexts]
            measured=[]
            for names,truth,plan,family in raw2:
                m=self.scored(target,names,truth,plan,values,t,base_y,flip_y,base_loss,flip_loss,family)
                if m is not None:
                    measured.append(m)
            if not repair_only and self.args.response_truths:
                # Learn output-loss-aware functions at compression roots too,
                # not only at the few error-repair roots. No new Monte Carlo
                # propagation: all TT rows reuse this root's two responses.
                supports=truth_supports(circuit,target,self.area_model.cones,
                                        [names for names,_,_,_ in raw2 if names],self.args.response_truths)
                decoded={};tbits=bits(t,self.count).astype(bool)
                loss0=np.where(tbits,flip_loss,base_loss);loss1=np.where(tbits,base_loss,flip_loss)
                for names in supports:
                    code=np.zeros(self.count,dtype=np.uint8)
                    for i,n in enumerate(names):
                        if n not in decoded:
                            decoded[n]=bits(values[n],self.count)
                        code |= decoded[n]<<i
                    for penalty in (1,4):
                        truth=response_truth(code,loss0,loss1,base_loss,len(names),penalty)
                        plan=self.mapper.map(len(names),truth)
                        m=self.scored(target,names,truth,plan,values,t,base_y,flip_y,base_loss,flip_loss,'output_loss_aware_truth',penalty)
                        self.stats['response_truths_constructed']+=1
                        if m is not None:
                            measured.append(m)
            result+=self.materialized(circuit,measured,base_total)
        if self.args.compensate:
            for _,target,t,flip_y,flip_loss in repair_contexts:
                if self.expired():
                    break
                tbits=bits(t,self.count).astype(bool)
                loss0=np.where(tbits,flip_loss,base_loss);loss1=np.where(tbits,base_loss,flip_loss)
                measured=[]
                for names,code in learned_sets(circuit,target,values,loss0,loss1,self.count,self.stats):
                    for penalty in (1,2,4):
                        truth=response_truth(code,loss0,loss1,base_loss,len(names),penalty)
                        plan=self.mapper.map(len(names),truth)
                        m=self.scored(target,names,truth,plan,values,t,base_y,flip_y,base_loss,flip_loss,'numeric_protected_truth',penalty)
                        self.stats['compensation_truths_constructed']+=1
                        if m is not None:
                            measured.append(m)
                result+=self.materialized(circuit,measured,base_total,True)
        unique={(m.target,m.names,m.truth):m for m in result}
        self.times['candidate_generation_and_scoring']+=time.perf_counter()-started
        return list(unique.values()),base_total

    def apply(self,selected,phase):
        tick=time.perf_counter()
        try:
            trial=cleanup(self.current.materialize([m.replacement() for m in selected]),self.mapper)
        except ValueError as exc:
            return False,dict(reason='invalid_joint_graph',error=str(exc))
        self.stats['area_materializations']+=1
        identity=structural_id(trial)
        if identity in self.seen:
            self.stats['duplicate_propagations_avoided']+=1
            return False,dict(reason='known_physical_circuit')
        y=number(outputs(trial,trace(trial,self.inputs[0],self.count,self.stats)),self.order,self.count)
        total=int(self.metric.loss(y,self.reference[0]).sum())
        self.stats['joint_verifications' if len(selected)>1 else 'single_verifications']+=1
        if len(selected)==1 and hashlib.blake2b(y.tobytes(),digest_size=16).hexdigest()!=selected[0].signature:
            raise AssertionError('Shared numerical response differs from actual candidate outputs')
        self.times['selected_actual_verification']+=time.perf_counter()-tick
        detail=dict(error_sum=total,metric=total/self.count/self.metric.denominator,
                    area=self.source.lib.area(trial.units),proposed_gain_units=sum(m.gain for m in selected),
                    actual_gain_units=self.current.units-trial.units,actions=[m.record() for m in selected])
        if total>self.metric.limit:
            return False,dict(detail,reason='actual_metric_exceeds_bound')
        if phase!='compensation' and trial.units>=self.current.units:
            return False,dict(detail,reason='no_actual_area_gain')
        if phase=='compensation':
            debt_ceiling=(self.current.units+16) if self.args.sop_view else min(self.source.units,self.lowest_units+16)
            if trial.units>debt_ceiling:
                return False,dict(detail,reason='cumulative_area_debt')
        self.current=trial;self.lowest_units=min(self.lowest_units,trial.units);self.seen.add(identity)
        self.checkpoint(phase,selected)
        return True,detail

    def quantize(self):
        """Learn one low-word constant by L1 median or L2 mean, not LAC enumeration."""
        if self.args.no_word or self.expired():
            return False
        self.word_area_gain=0
        started=time.perf_counter();circuit=self.current
        base=number(outputs(circuit,trace(circuit,self.inputs[0],self.count,self.stats)),self.order,self.count)
        base_total=int(self.metric.loss(base,self.reference[0]).sum())
        choices=[]
        for width in range(1,len(self.order)):
            mask=(1<<width)-1;high=base & ~mask;residual=self.reference[0]-high
            raw=int(np.median(residual)) if self.metric.kind=='nmed' else int(round(float(np.mean(residual))))
            constant=max(0,min(mask,raw));predicted=high+constant
            total=int(self.metric.loss(predicted,self.reference[0]).sum())
            self.stats['word_function_scores']+=1
            if total<=self.metric.limit:
                choices.append((width,constant,total,predicted))
        for width,constant,total,predicted in sorted(choices,reverse=True,key=lambda r:r[0])[:3]:
            identity=(structural_id(circuit),width,constant)
            if identity in self.quantization_seen:
                continue
            self.quantization_seen.add(identity)
            trial=cleanup(constant_outputs(circuit,{n:(constant>>i)&1 for i,n in enumerate(self.order[:width])}),self.mapper)
            self.stats['area_materializations']+=1
            # Retune already-constant low bits for zero area debt. The resulting
            # error headroom can enable subsequent compression of the high word.
            if trial.units>circuit.units or (trial.units==circuit.units and total>=base_total):
                continue
            gain=circuit.units-trial.units
            if total>base_total and gain/max(1,circuit.units)<self.args.word_efficiency*(total-base_total)/max(1,self.metric.limit):
                self.stats['word_poor_economics_rejected']+=1
                continue
            actual=number(outputs(trial,trace(trial,self.inputs[0],self.count,self.stats)),self.order,self.count)
            self.stats['word_joint_verifications']+=1
            if not np.array_equal(actual,predicted):
                raise AssertionError('Detached output quantization changed the retained high word')
            self.current=trial;self.seen.add(structural_id(trial));self.lowest_units=min(self.lowest_units,trial.units)
            self.word_area_gain=gain
            self.checkpoint('numeric_word_quantization',[])
            self.history[-1]['word_action']=dict(width=width,constant=constant,loss=total)
            save(self.folder/'history.json',self.history)
            self.times['word_quantization']+=time.perf_counter()-started
            return True
        self.times['word_quantization']+=time.perf_counter()-started
        return False

    def near_repair(self,candidates):
        if not self.args.compensate or self.expired():
            return False
        near=sorted((m for m in candidates if self.metric.limit<m.error<=self.metric.limit*self.args.near_ratio and m.gain>0),
                    key=lambda m:((m.error-self.metric.limit)/m.gain,-m.gain))[:2]
        original=self.current
        for m in near:
            if self.remaining()<1:
                break
            branch=cleanup(original.materialize([m.replacement()]),self.mapper)
            self.stats['area_materializations']+=1
            if branch.units>=original.units:
                continue
            self.stats['near_miss_branches']+=1
            self.current=branch
            repairs,base_total=self.candidates(repair_only=True)
            event=dict(parent_iteration=self.iteration,action=m.record(),actual_branch_loss=base_total,
                       actual_branch_area=self.source.lib.area(branch.units),repair_trials=[],accepted=False)
            self.near_events.append(event)
            options=sorted((r for r in repairs if r.error<=self.metric.limit),key=lambda r:(-r.gain,r.error))[:3]
            for r in options:
                trial=cleanup(branch.materialize([r.replacement()]),self.mapper)
                self.stats['area_materializations']+=1
                actual=number(outputs(trial,trace(trial,self.inputs[0],self.count,self.stats)),self.order,self.count)
                total=int(self.metric.loss(actual,self.reference[0]).sum())
                self.stats['near_miss_joint_verifications']+=1
                event['repair_trials'].append(dict(action=r.record(),actual_loss=total,
                    actual_area=self.source.lib.area(trial.units)))
                if total<=self.metric.limit and trial.units<original.units:
                    self.current=trial;self.lowest_units=min(self.lowest_units,trial.units)
                    self.seen.add(structural_id(trial));self.checkpoint('near_miss_plus_targeted_compensation',[m,r])
                    self.stats['near_miss_repairs_accepted']+=1
                    event['accepted']=True
                    save(self.folder/'near_misses.json',self.near_events)
                    return True
            self.current=original
            save(self.folder/'near_misses.json',self.near_events)
        return False

    def run(self):
        stop='screened_plateau';stalled=0
        while not self.expired():
            self.iteration+=1
            if self.quantize() and self.word_area_gain>0:
                self.refine('word_exact')
            candidates,base_total=self.candidates()
            positive=[m for m in candidates if m.gain>0 and m.error<=self.metric.limit]
            tick=time.perf_counter()
            batch=bounded_batch(positive,max(0,self.metric.limit-base_total),
                                min(self.args.dp_seconds,self.remaining()),self.stats)
            self.times['knapsack']+=time.perf_counter()-tick
            proposals=[batch] if batch else []
            if len(batch)>2:
                ordered=sorted(batch,key=lambda m:(-m.gain/max(1,m.weight),m.error))
                proposals += [ordered[:max(1,len(ordered)//2)]]
            if self.args.single_policy=='error':
                single_key=lambda m:(m.error,-m.gain)
            elif self.args.single_policy=='efficiency':
                single_key=lambda m:(-m.gain/max(1,m.weight),m.error)
            else:
                single_key=lambda m:(-m.gain,m.error)
            proposals += [[m] for m in sorted(positive,key=single_key)[:3]]
            accepted=False;detail={}
            for selected in proposals:
                if self.expired():
                    break
                accepted,detail=self.apply(selected,'batch' if len(selected)>1 else 'frontier_single')
                if accepted:
                    break
            if not accepted and self.args.compensate and self.repairs<3:
                for m in sorted((m for m in candidates if m.error<base_total and m.error<=self.metric.limit),
                                key=lambda m:(m.error+m.weight,-m.gain))[:3]:
                    accepted,detail=self.apply([m],'compensation')
                    if accepted:
                        self.repairs+=1
                        break
            if not accepted:
                accepted=self.near_repair(candidates)
            if accepted:
                self.refine('accepted_batch_exact')
                stalled=0
            else:
                stalled+=1
            event=dict(iteration=self.iteration,candidates=len(candidates),accepted=accepted,
                       working_area=self.source.lib.area(self.current.units),
                       best_checkpoint_area=min(r['area'] for r in self.history),base_loss=base_total,
                       detail=detail,remaining_seconds=self.remaining())
            self.events.append(event);save(self.folder/'iterations.json',self.events)
            print('[FRONTIER ROUND] '+json.dumps({k:v for k,v in event.items() if k!='detail'}),flush=True)
            if stalled>=2:
                break
        if self.expired():
            stop='time_limit'
        return dict(stop_reason=stop,search_seconds=time.perf_counter()-self.started,stats=dict(self.stats),
                    component_seconds=dict(self.times),iterations=self.iteration,compensation_steps=self.repairs,
                    mapping=self.backend.rows if self.backend else [],controller='numerical_frontier',api_calls=0,
                    search_integer_loss_bound=self.metric.limit,formal_integer_loss_bound=self.formal_limit,
                    component_timing_note='root_response_subset is included in candidate_generation_and_scoring; other named stages are disjoint')


SOURCES=['bacs_frontier.py','numeric_exact.py','numeric_logic_view.py','bacs_numeric.py','bacs_data.py','logic.py','netlist.py',
         'cutmap.py','frontier.py','screen.py','simulate.py','weighted_cost.py','runtime.py',
         'paper_resub_netlist.py','paper_netlist_subfunctions.py','hybrid_identity.py']


def run(args):
    started=time.perf_counter();folder=Path(args.output).resolve();folder.mkdir(parents=True,exist_ok=False)
    (folder/'source').mkdir();hashes={}
    code_root=Path(__file__).resolve().parent
    for name in SOURCES:
        hashes[name]=hashlib.sha256((code_root/name).read_bytes()).hexdigest()
        shutil.copy2(code_root/name,folder/'source'/name)
    lib=Library(LIBRARY_PATH);source,order,audit=preflight(args.design,lib)
    source.write(folder/'accurate.v')
    matrices=[matrix(args.design,source.inputs,i) for i in (0,1,2,100)]
    inputs=[pack(x,source.inputs) for x in matrices]
    refs=[number(outputs(source,trace(source,p,10000)),order,10000) for p in inputs]
    metric=Metric(args.metric,args.bound,order)
    save(folder/'protocol.json',dict(vars(args),code_sha256=hashes,
        source_sha256=hashlib.sha256(source_path(args.design).read_bytes()).hexdigest(),
        library_sha256=hashlib.sha256(LIBRARY_PATH.read_bytes()).hexdigest(),count_per_group=10000,
        input_splits=[0,1,2,100],output_lsb_first=order,integer_bound=metric.limit,audit=audit,
        selection='minimum raw physical area among development-eligible checkpoints, then path; no final filtering',
        exact_mapping='ABC exact-only; completed CEC proof required; approximate search is independent',
        final_is_sampled=True,exhaustive_lacs=False,controller='numerical_frontier',api_calls=0))
    search=FrontierSearch(source,order,inputs,refs,args,folder);result=search.run()
    rows=[];dev_limit=int(Decimal(metric.limit)*Decimal(str(args.development_fraction)))
    for row in search.history:
        c=load(folder/row['path'],lib)
        total,value=metric.exact(number(outputs(c,trace(c,inputs[1],10000)),order,10000),refs[1])
        rows.append(dict(row,development_loss=total,development_value=value))
    selected=min((r for r in rows if r['development_loss']<=dev_limit),key=lambda r:(r['units'],r['path']))
    winner=load(folder/selected['path'],lib);checks=[]
    def independent(c,array):
        columns={n:i for i,n in enumerate(c.outputs)}
        return np.array([sum(int(array[j,columns[n]])<<i for i,n in enumerate(order)) for j in range(len(array))],dtype=np.int64)
    for index,label in ((2,'final'),(3,'confirmation')):
        actual=number(outputs(winner,trace(winner,inputs[index],10000)),order,10000)
        total,value=metric.exact(actual,refs[index])
        alternate=independent(winner,validate_numpy(winner,matrices[index]))
        reference=independent(source,validate_numpy(source,matrices[index]))
        if not np.array_equal(actual,alternate) or not np.array_equal(reference,refs[index]):
            raise AssertionError('Independent packed/NumPy output mismatch')
        checks.append(dict(group=label,split=(2 if index==2 else 100),count=10000,loss=total,
                           metric=value,passed=total<=metric.limit,vector_er=float(np.count_nonzero(actual!=refs[index]))/10000))
    passed=all(c['passed'] for c in checks)
    winner.write(folder/('best.v' if passed else 'failed_final.v'))
    train_total,train_value=metric.exact(number(outputs(winner,trace(winner,inputs[0],10000)),order,10000),refs[0])
    result.update(design=args.design,metric=args.metric,bound=args.bound,original_area=lib.area(source.units),
        area=lib.area(winner.units),reduction=1-winner.units/source.units,training_value=train_value,
        development_value=selected['development_value'],checks=checks,strict_pass=passed,
        selected_checkpoint=selected['path'],all_checkpoints=rows,whole_seconds=time.perf_counter()-started,
        postselection_full_propagations=len(rows)+3,reference_full_propagations=4,independent_numpy_checks=4,
        preflight=audit,netlist_sha256=hashlib.sha256((folder/('best.v' if passed else 'failed_final.v')).read_bytes()).hexdigest())
    save(folder/'report.json',result)
    print('[FRONTIER RESULT] '+json.dumps({k:v for k,v in result.items() if k not in ('all_checkpoints','preflight','mapping')}),flush=True)
    return 0 if passed else 2


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--design',choices=DESIGNS,required=True)
    p.add_argument('--metric',choices=('nmed','mse'),default='nmed')
    p.add_argument('--bound',type=float,required=True);p.add_argument('--output',required=True)
    p.add_argument('--seconds',type=float,default=60);p.add_argument('--seed',type=int,default=20261027)
    p.add_argument('--pool',type=int,default=64);p.add_argument('--pairs',type=int,default=128)
    p.add_argument('--per-root',type=int,default=8);p.add_argument('--subfunctions',action='store_true')
    p.add_argument('--response-truths',type=int,default=8,help='bounded input supports for loss-aware truth construction per compression root; 0 disables')
    p.add_argument('--compensate',action='store_true');p.add_argument('--near-ratio',type=float,default=1.15)
    p.add_argument('--repair-contexts',type=int,default=8);p.add_argument('--repair-roots',type=int,default=64)
    p.add_argument('--no-word',action='store_true');p.add_argument('--no-exact',action='store_true')
    p.add_argument('--word-efficiency',type=float,default=.2,help='minimum fractional area saving per fraction of error budget consumed')
    p.add_argument('--logic-view',action='store_true',help='search an exact decomposed standard-cell DAG; map incumbents without collapsing the search view')
    p.add_argument('--sop-view',action='store_true',help='continue from the exact simplified two-input SOP, keeping physical mapped checkpoints separately')
    p.add_argument('--wsl-distro',default='Ubuntu');p.add_argument('--abc',default='/usr/local/bin/abc')
    p.add_argument('--map-seconds',type=float,default=8);p.add_argument('--map-variants',type=int,choices=(1,2),default=1)
    p.add_argument('--single-policy',choices=('error','area','efficiency'),default='error')
    p.add_argument('--dp-seconds',type=float,default=2.)
    # Equal-bound baseline is the default. Optional margins must be requested
    # explicitly; they otherwise exclude known formally admissible solutions.
    p.add_argument('--training-fraction',type=float,default=1.)
    p.add_argument('--development-fraction',type=float,default=1.)
    return p


if __name__=='__main__':
    p=parser();args=p.parse_args()
    if args.bound<=0 or args.word_efficiency<0 or not 0<=args.response_truths<=16 or min(args.seconds,args.pool,args.pairs,args.per_root,args.map_seconds,args.repair_contexts,args.repair_roots,args.dp_seconds)<=0:
        p.error('Budgets and bound must be positive')
    if not 0<args.development_fraction<=1 or not 0<args.training_fraction<=1 or not 1<=args.near_ratio<=1.5:
        p.error('Invalid development margin or near-miss ratio')
    if (args.logic_view or args.sop_view) and args.no_exact:
        p.error('Logic/SOP views require exact physical remapping')
    if args.logic_view and args.sop_view:
        p.error('Select only one working-view mode')
    raise SystemExit(run(args))
