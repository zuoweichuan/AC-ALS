"""Experimental signed-error plans and conditional numerical output rebuilding.

Screened proposals only. Joint response predictions rank plans; full circuit
simulation and physical mapping decide acceptance. Existing entries are intact.
"""
import runtime
import argparse
import copy
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np

from bacs_data import LIBRARY_PATH, load, matrix, preflight
from bacs_numeric import Metric, bits, cleanup, number, number_numpy, save
from bacs_frontier import SOURCES
from numeric_agentic import Worker, parser as base_parser, sha
from numeric_exact import ExactBackend
from netlist import Circuit, Gate, ZERO, ONE
from simulate import trace, outputs, pack, plan_value, validate_numpy
from hybrid_identity import structural_id
from logic import Library, Plan, evaluate
from cutmap import cof


def signed_plans(worker, maximum=80, depth=10, beam_size=4):
    """Bounded beam over screened LACs, retaining signed output displacements."""
    scan=worker.scan();moves=scan['moves'];c=worker.current;n=worker.count
    selected=[];used=set()
    queues=[sorted(moves,key=lambda m:(-m.gain,m.error)),
            sorted(moves,key=lambda m:(m.error,-m.gain)),
            sorted(moves,key=lambda m:(-m.gain/max(1,m.weight),m.error))]
    for i in range(maximum):
        for queue in queues:
            if i<len(queue) and id(queue[i]) not in used:
                used.add(id(queue[i]));selected.append(queue[i])
                if len(selected)>=maximum:break
        if len(selected)>=maximum:break
    vals=trace(c,worker.inputs[0],n,worker.stats)
    base=number(outputs(c,vals),worker.order,n);reference=worker.reference[0]
    deltas=[];flips={};mask=(1<<n)-1
    for m in selected:
        if worker.remaining()<.5:break
        if m.target not in flips:
            flipped=trace(c,worker.inputs[0],n,worker.stats,base=vals,flip=m.target)
            flips[m.target]=number(outputs(c,flipped),worker.order,n)-base
        changed=bits(plan_value(m.plan,m.names,c.lib,vals,mask)^vals[m.target],n).astype(bool)
        deltas.append(np.where(changed,flips[m.target],0))
    selected=selected[:len(deltas)]
    beam=[((),np.zeros(n,dtype=np.int64),0,int(np.abs(base-reference).sum()))]
    proposals=[];seen_plans=set()
    for level in range(depth):
        if worker.remaining()<.5:break
        expanded=[]
        for ids,delta,gain,_ in beam:
            targets={selected[i].target for i in ids}
            for i,m in enumerate(selected):
                if m.target in targets:continue
                key=tuple(sorted((*ids,i)))
                if key in seen_plans:continue
                seen_plans.add(key)
                new_delta=delta+deltas[i]
                loss=int(np.abs(base+new_delta-reference).sum())
                worker.stats['signed_plan_scores']+=1
                expanded.append((key,new_delta,gain+m.gain,loss))
        if not expanded:break
        # All four paths use actual signed numerical loss, not positive-only
        # increment budgets. A limited debt path may reach complementary repair.
        economic=lambda q: q[2]-2*c.units*max(0,q[3]/worker.metric.limit-1)
        queues=[sorted(expanded,key=lambda q:(-economic(q),q[3],q[0])),
                sorted(expanded,key=lambda q:(q[3],-q[2],q[0])),
                sorted((q for q in expanded if q[3]<=worker.metric.limit),key=lambda q:(-q[2],q[3],q[0]))]
        beam=[];seen=set()
        for rank in range(beam_size):
            for queue in queues:
                if rank<len(queue) and queue[rank][0] not in seen:
                    q=queue[rank];beam.append(q);seen.add(q[0])
                    if len(beam)>=beam_size:break
            if len(beam)>=beam_size:break
        for q in queues[-1][:2]:
            if len(q[0])>1 and q[2]>0:
                proposals.append((q[2],q[3],[selected[i] for i in q[0]]))
    return sorted(proposals,key=lambda q:(-q[0],q[1]))[:12]


def conditional_median(code, target, width, count):
    """One analytic L1 minimizer per observed support row, no TT enumeration."""
    mask=(1<<width)-1
    fallback=int(np.clip(np.median(target),0,mask))
    table=np.full(1<<count,fallback,dtype=np.int64)
    for row in range(1<<count):
        values=target[code==row]
        if len(values):table[row]=int(np.clip(np.median(values),0,mask))
    return table,table[code]


def detached_word(circuit, order, width, names, table, mapper):
    """Replace PO observations while retaining their original internal fanouts."""
    targets=set(order[:width]);used=set(circuit.producer)|set(circuit.aliases)|set(circuit.inputs)|set(circuit.outputs)
    rename={}
    for i,n in enumerate(sorted(targets)):
        fresh='__word_old_'+str(i)
        while fresh in used:fresh+='_'
        used.add(fresh);rename[n]=fresh
    r=lambda n:rename.get(n,n)
    gates=[Gate(g.name,g.cell,{p:r(n) for p,n in g.pins.items()}) for g in circuit.gates]
    aliases={r(n):r(v) for n,v in circuit.aliases.items()};memo={}
    def fresh():
        n='__word_new_'+str(len(used))
        while n in used:n+='_'
        used.add(n);return n
    def emit(p):
        if p.name=='leaf':return r(names[p.leaf]) if p.leaf>=0 else (ZERO,ONE)[p.leaf+2]
        if p in memo:return memo[p]
        cell=circuit.lib.cells[p.name];deps=tuple(emit(child) for child in p.children);net=fresh()
        gates.append(Gate(fresh(),cell.name,{**dict(zip(cell.inputs,deps)),next(iter(cell.functions)):net}))
        memo[p]=net;return net
    @lru_cache(maxsize=128)
    def mapped(n,truth):
        if n<=4:return mapper.map(n,truth)
        # Only the selected truth is decomposed; support rows are never searched.
        rest=tuple(range(n-1));low=mapper.bind(mapped(n-1,cof(truth,n,n-1,0)),rest)
        high=mapper.bind(mapped(n-1,cof(truth,n,n-1,1)),rest);s=Plan('leaf',leaf=n-1)
        return mapper.binary(14,mapper.binary(8,mapper.binary(5,s,s),low),mapper.binary(8,s,high))
    for bit,out in enumerate(order[:width]):
        truth=sum(int((int(v)>>bit)&1)<<row for row,v in enumerate(table))
        aliases[out]=emit(mapped(len(names),truth))
    return cleanup(Circuit(circuit.lib,circuit.inputs,circuit.outputs,gates,aliases,circuit.name),mapper)


def detached_wires(circuit,bindings,mapper):
    """Observe a different net at a PO without changing its internal driver."""
    used=set(circuit.producer)|set(circuit.aliases)|set(circuit.inputs)|set(circuit.outputs);rename={}
    for out in bindings:
        fresh='__observe_old_'+str(len(rename))
        while fresh in used:fresh+='_'
        used.add(fresh);rename[out]=fresh
    r=lambda n:rename.get(n,n)
    aliases={r(k):r(v) for k,v in circuit.aliases.items()}
    aliases.update({out:r(signal) for out,signal in bindings.items()})
    return cleanup(Circuit(circuit.lib,circuit.inputs,circuit.outputs,
        [Gate(g.name,g.cell,{p:r(v) for p,v in g.pins.items()}) for g in circuit.gates],aliases,circuit.name),mapper)


def observation_proposals(worker):
    c=worker.current;n=worker.count;v=trace(c,worker.inputs[0],n,worker.stats)
    base=number(outputs(c,v),worker.order,n);ref=worker.reference[0]
    pool=list(dict.fromkeys([ZERO,ONE,*c.inputs,*c.producer]));decoded={s:bits(v[s],n).astype(np.int64) for s in pool}
    sample=np.arange(0,n,max(1,n//128))[:128];result=[]
    for bit,out in enumerate(worker.order):
        high=base-((base>>bit)&1)*(1<<bit)
        ranked=sorted(pool,key=lambda s:(int(np.abs(high[sample]+decoded[s][sample]*(1<<bit)-ref[sample]).sum()),c.level.get(s,0),s))[:12]
        for signal in dict.fromkeys([ZERO,ONE,*ranked]):
            predicted=high+decoded[signal]*(1<<bit);loss=int(np.abs(predicted-ref).sum())
            worker.stats['detached_observation_scores']+=1
            if loss>worker.metric.limit:continue
            trial=detached_wires(c,{out:signal},worker.mapper);worker.stats['area_materializations']+=1
            if trial.units<c.units:
                result.append((c.units-trial.units,loss,trial,dict(observation={out:signal}),predicted))
    unique={structural_id(q[2]):q for q in result}
    return sorted(unique.values(),key=lambda q:(-q[0],q[1]))[:12]


def wire_repair_proposals(worker):
    """Coordinate-fit observation wires; constant POs remain repairable."""
    c=worker.current;n=worker.count;mask=(1<<n)-1
    vals=trace(c,worker.inputs[0],n,worker.stats);base=number(outputs(c,vals),worker.order,n)
    initial=int(np.abs(base-worker.reference[0]).sum());pred=base.copy();bindings={}
    pool=list(dict.fromkeys([ZERO,ONE,*c.inputs,*c.producer]))
    decoded={s:bits(vals[s],n).astype(np.int64) for s in pool}
    sample=np.arange(0,n,max(1,n//128))[:128]
    for sweep in range(3):
        changed=False
        for bit,out in enumerate(worker.order[:min(10,len(worker.order))]):
            if worker.remaining()<1:break
            weight=1<<bit;high=pred-((pred>>bit)&1)*weight
            ranked=sorted(pool,key=lambda s:(int(np.abs(high[sample]+decoded[s][sample]*weight-worker.reference[0][sample]).sum()),c.level.get(s,0),s))[:12]
            best_loss=int(np.abs(pred-worker.reference[0]).sum());best=None
            for s in ranked:
                candidate=high+decoded[s]*weight;loss=int(np.abs(candidate-worker.reference[0]).sum())
                worker.stats['wire_repair_scores']+=1
                if loss<best_loss:best_loss=loss;best=s,candidate
            if best:
                bindings[out]=best[0];pred=best[1];changed=True
        if not changed:break
    if not bindings or int(np.abs(pred-worker.reference[0]).sum())>=initial:return []
    used=set(c.producer)|set(c.aliases)|set(c.inputs)|set(c.outputs);rename={}
    for out in bindings:
        fresh='__wire_old_'+str(len(rename))
        while fresh in used:fresh+='_'
        used.add(fresh);rename[out]=fresh
    r=lambda n:rename.get(n,n)
    aliases={r(k):r(v) for k,v in c.aliases.items()}
    aliases.update({out:r(signal) for out,signal in bindings.items()})
    trial=cleanup(Circuit(c.lib,c.inputs,c.outputs,[Gate(g.name,g.cell,{p:r(v) for p,v in g.pins.items()}) for g in c.gates],aliases,c.name),worker.mapper)
    return [(c.units-trial.units,int(np.abs(pred-worker.reference[0]).sum()),trial,dict(output_wires=bindings),pred)]


def word_proposals(worker, pool_size=32, support_size=6, repair=False):
    c=worker.current;n=worker.count
    vals=trace(c,worker.inputs[0],n,worker.stats)
    base=number(outputs(c,vals),worker.order,n);reference=worker.reference[0]
    raw=list(dict.fromkeys([*c.inputs,*[c.resolve(p) for p in worker.order],*c.producer]))
    raw=[s for s in raw if s not in (ZERO,ONE)]
    decoded={s:bits(vals[s],n).astype(np.int64) for s in raw}
    proposals=[]
    # Width is an output representation choice; only <=6 greedily selected
    # support signals are considered. No combinations of the full LAC pool.
    base_total=int(np.abs(base-reference).sum())
    for width in range(1,min(4 if repair else 10,len(worker.order))+1):
        if worker.remaining()<1:break
        mask=(1<<width)-1;high=base&~mask;target=reference-high
        singles=[]
        for s in raw:
            table,pred=conditional_median(decoded[s],target,width,1)
            loss=int(np.abs(high+pred-reference).sum())
            singles.append((loss,c.level[s],s))
        for mode in ('all','shallow','inputs'):
            pool=[s for _,_,s in sorted(singles) if mode=='all' or c.level[s]<=(2 if mode=='shallow' else 0)][:pool_size]
            names=();code=np.zeros(n,dtype=np.int64)
            for size in range(1,support_size+1):
                if worker.remaining()<1:break
                best=None
                for s in pool:
                    if s in names:continue
                    candidate_code=code|(decoded[s]<<(size-1))
                    table,pred=conditional_median(candidate_code,target,width,size)
                    loss=int(np.abs(high+pred-reference).sum());worker.stats['conditional_word_scores']+=1
                    trial=None
                    if loss<=worker.metric.limit:
                        trial=detached_word(c,worker.order,width,(*names,s),table,worker.mapper)
                        worker.stats['area_materializations']+=1
                    key=(0,trial.units,loss,s) if trial is not None else (1,loss,c.level[s],s)
                    if best is None or key<best[0]:best=(key,s,candidate_code,table,pred,trial,loss)
                if best is None:break
                _,s,code,table,pred,trial,loss=best;names=(*names,s)
                useful=(trial is not None and ((trial.units<=c.units+16 and loss<base_total) if repair else trial.units<c.units))
                if useful:
                    proposals.append((c.units-trial.units,loss,trial,
                        dict(width=width,inputs=names,table=table.tolist(),support_mode=mode),high+pred))
    unique={structural_id(q[2]):q for q in proposals}
    if repair:return sorted(unique.values(),key=lambda q:(-(base_total-q[1])/max(1,1-q[0]),q[1]))[:8]
    return sorted(unique.values(),key=lambda q:(-q[0],q[1]))[:12]


def input_proposals(worker):
    """Screen global PI cofactoring, jointly retuning detached low-word bias.

    This acts on all fanouts of a selected PI. Ordinary internal-root LACs cannot
    represent that simplification. Only two constants per live PI are tested;
    a measured trajectory selects successive steps, never a Cartesian product.
    """
    from bacs_frontier import constant_outputs
    c=worker.current;n=worker.count;choices=[]
    live={c.resolve(g.pins[p]) for g in c.gates for p in c.lib.cells[g.cell].inputs}
    for signal in c.inputs:
        if signal not in live:continue
        for bit in (0,1):
            if worker.remaining()<1:break
            replace=lambda v:(ZERO,ONE)[bit] if v==signal else v
            trial=cleanup(Circuit(c.lib,c.inputs,c.outputs,
                [Gate(g.name,g.cell,{p:replace(v) for p,v in g.pins.items()}) for g in c.gates],
                {k:replace(v) for k,v in c.aliases.items()},c.name),worker.mapper)
            y=number(outputs(trial,trace(trial,worker.inputs[0],n,worker.stats)),worker.order,n)
            loss=int(np.abs(y-worker.reference[0]).sum());worker.stats['input_cofactor_scores']+=1
            variants=[]
            if loss<=worker.metric.limit:variants.append((trial,y,loss,0,0))
            fits=[]
            for width in range(1,len(worker.order)):
                mask=(1<<width)-1;high=y&~mask
                constant=int(np.clip(np.median(worker.reference[0]-high),0,mask));pred=high+constant
                loss=int(np.abs(pred-worker.reference[0]).sum());worker.stats['input_bias_scores']+=1
                if loss<=worker.metric.limit:fits.append((width,constant,pred,loss))
            for width,constant,pred,loss in sorted(fits,reverse=True,key=lambda q:q[0])[:2]:
                candidate=cleanup(constant_outputs(trial,{out:(constant>>i)&1 for i,out in enumerate(worker.order[:width])}),worker.mapper)
                variants.append((candidate,pred,loss,width,constant))
            for candidate,pred,loss,width,constant in variants:
                if candidate.units<c.units:
                    choices.append((c.units-candidate.units,loss,candidate,
                        dict(clamped_input=signal,assigned_bit=bit,retuned_width=width,retuned_constant=constant),pred))
    unique={structural_id(q[2]):q for q in choices}
    return sorted(unique.values(),key=lambda q:(-q[0],q[1]))[:12]


def linked_input_proposals(worker):
    """Grow <=8-input cofactor plans using signed response cancellation.

    Atomic responses cheaply rank four partners per seed. Only that shortlist
    is constructed; no input-combination Cartesian product is evaluated.
    """
    from bacs_frontier import constant_outputs
    c=worker.current;n=worker.count;ref=worker.reference[0]
    base=number(outputs(c,trace(c,worker.inputs[0],n,worker.stats)),worker.order,n)
    sample=np.arange(0,n,max(1,n//128))[:128]
    live={c.resolve(g.pins[p]) for g in c.gates for p in c.lib.cells[g.cell].inputs}
    def construct(assign):
        r=lambda v:(ZERO,ONE)[assign[v]] if v in assign else v
        return cleanup(Circuit(c.lib,c.inputs,c.outputs,
            [Gate(g.name,g.cell,{p:r(v) for p,v in g.pins.items()}) for g in c.gates],
            {k:r(v) for k,v in c.aliases.items()},c.name),worker.mapper)
    atoms=[]
    for signal in c.inputs:
        if signal not in live:continue
        for bit in (0,1):
            trial=construct({signal:bit})
            y=number(outputs(trial,trace(trial,worker.inputs[0],n,worker.stats)),worker.order,n)
            atoms.append((signal,bit,(y-base)[sample],int(np.abs(y-ref).sum()),trial.units))
    seeds=sorted(range(len(atoms)),key=lambda i:(atoms[i][3],atoms[i][4],i))[:24]
    seen=set();plans=[];candidates=[]
    for i in seeds:
        a=atoms[i]
        partners=sorted((j for j,b in enumerate(atoms) if a[0]!=b[0]),key=lambda j:
            (int(np.abs(base[sample]+a[2]+atoms[j][2]-ref[sample]).sum()),atoms[j][4],j))[:4]
        plans.extend(tuple(sorted((i,j))) for j in partners)
    beam=[]
    for depth in range(2,9):
        measured=[]
        for ids in plans:
            if worker.remaining()<1:break
            if ids in seen:continue
            seen.add(ids);assign={atoms[i][0]:atoms[i][1] for i in ids}
            trial=construct(assign)
            y=number(outputs(trial,trace(trial,worker.inputs[0],n,worker.stats)),worker.order,n)
            worker.stats['linked_input_plan_verifications']+=1
            variants=[];total=int(np.abs(y-ref).sum())
            if total<=worker.metric.limit:variants.append((trial,y,total,0,0))
            fits=[]
            for width in range(1,len(worker.order)):
                mask=(1<<width)-1;high=y&~mask
                constant=int(np.clip(np.median(ref-high),0,mask));pred=high+constant
                loss=int(np.abs(pred-ref).sum());worker.stats['input_bias_scores']+=1
                if loss<=worker.metric.limit:fits.append((width,constant,pred,loss))
            for width,constant,pred,loss in sorted(fits,reverse=True,key=lambda q:q[0])[:2]:
                candidate=cleanup(constant_outputs(trial,{out:(constant>>i)&1 for i,out in enumerate(worker.order[:width])}),worker.mapper)
                variants.append((candidate,pred,loss,width,constant))
            best=min(variants,key=lambda q:(q[0].units,q[2])) if variants else (trial,y,total,0,0)
            candidate,pred,loss,width,constant=best
            measured.append((ids,trial,y,loss,candidate.units))
            if variants and candidate.units<c.units:
                candidates.append((c.units-candidate.units,loss,candidate,
                    dict(clamped_inputs=assign,retuned_width=width,retuned_constant=constant),pred))
        # Different numerical routes survive: minimum area and minimum debt.
        queues=[sorted(measured,key=lambda q:(q[4]+c.units*max(0,q[3]/worker.metric.limit-1),q[3])),
                sorted(measured,key=lambda q:(q[3],q[4]))]
        beam=[];used=set()
        for rank in range(4):
            for queue in queues:
                if rank<len(queue) and queue[rank][0] not in used:
                    beam.append(queue[rank]);used.add(queue[rank][0])
                    if len(beam)>=4:break
            if len(beam)>=4:break
        plans=[]
        for ids,trial,y,_,_ in beam:
            assigned={atoms[i][0] for i in ids}
            partners=sorted((j for j,a in enumerate(atoms) if a[0] not in assigned),key=lambda j:
                (int(np.abs(y[sample]+atoms[j][2]-ref[sample]).sum()),atoms[j][4],j))[:4]
            plans.extend(tuple(sorted((*ids,j))) for j in partners)
    unique={structural_id(q[2]):q for q in candidates}
    return sorted(unique.values(),key=lambda q:(-q[0],q[1]))[:12]


def forced_trace(circuit, inputs, count, forced, stats=None):
    """Exact simultaneous clamps at selected internal nets, including ancestors."""
    mask=(1<<count)-1;values={**inputs,ZERO:0,ONE:mask}
    for g in circuit.gates:
        cell=circuit.lib.cells[g.cell]
        env={'0':0,'1':mask,**{p:values[circuit.resolve(g.pins[p])] for p in cell.inputs}}
        for p,fun in cell.functions.items():
            if p in g.pins:
                out=g.pins[p];values[out]=forced[out]*mask if out in forced else evaluate(fun,env,mask)
    if stats is not None:
        stats['multi_root_simulations']+=1;stats['multi_root_gate_evaluations']+=len(circuit.gates)
    return values


def joint_truth(code, losses, width):
    cost=np.vstack([np.bincount(code,weights=loss,minlength=1<<width) for loss in losses])
    choice=np.argmin(cost,axis=0)
    return choice,int(cost[choice,np.arange(1<<width)].sum())


def economic_truths(code,losses,width,choice,limit,mapper):
    """Screen real-cell patterns against row costs before mapping any circuit.

    Two coordinate orders retain area/error alternatives. This is a bounded
    cell-template projection, not enumeration of all multi-output functions.
    """
    cost=np.vstack([np.bincount(code,weights=loss,minlength=1<<width) for loss in losses])
    rows=np.arange(1<<width);mask=(1<<(1<<width))-1;answers=[choice]
    for order in ((0,1),(1,0)):
        for spend in (1.,.7):
            current=choice.copy();initial=int(cost[current,rows].sum())
            cap=initial+int(max(0,limit-initial)*spend)
            for bit in order:
                original=sum(int((int(v)>>bit)&1)<<r for r,v in enumerate(current))
                # The library catalogue is a cheap pattern index. Only the
                # eight closest feasible area patterns survive to this tool.
                templates=set(mapper.catalog[width]) if width else {0,1}
                templates.add(original)
                low=current&~(1<<bit);high=low|(1<<bit)
                c0=cost[low,rows];c1=cost[high,rows];ranked=[]
                for truth in templates:
                    b=np.array([(truth>>r)&1 for r in rows],dtype=np.int64)
                    total=int(np.where(b,c1,c0).sum())
                    if total<=cap:
                        estimate=mapper.cost(mapper.catalog[width][truth]) if width and truth in mapper.catalog[width] else mapper.cost(mapper.map(width,truth))
                        ranked.append((estimate,total,(truth^original).bit_count(),truth,b))
                if ranked:
                    ranked.sort(key=lambda q:q[:4]);best=ranked[0]
                    current=low|(best[-1]<<bit)
            answers.append(current)
    unique={tuple(map(int,q)):q for q in answers}
    return list(unique.values())


def pair_proposals(worker, maximum=48):
    """Analytic two-root multi-output truth synthesis on screened supports."""
    from screen import FanoutCones
    from frontier import forbidden_fanout
    c=worker.current;n=worker.count;cones=FanoutCones(c)
    vals=trace(c,worker.inputs[0],n,worker.stats);base=number(outputs(c,vals),worker.order,n)
    base_loss=np.abs(base-worker.reference[0]);roots=sorted(c.producer,key=lambda t:(-cones.potential(t),t))
    pairs=[];seen=set()
    for target in roots[:48]:
        gate=c.producer[target];deps={c.resolve(gate.pins[p]) for p in c.lib.cells[gate.cell].inputs}
        neighbors=set(d for d in deps if d in c.producer)
        for dep in deps:
            for other in c.fanouts.get(dep,[]):
                neighbors.update(other.pins[p] for p in c.lib.cells[other.cell].functions if p in other.pins)
        neighbors.update(g.pins[p] for g in c.fanouts.get(target,[]) for p in c.lib.cells[g.cell].functions if p in g.pins)
        for other in sorted(neighbors,key=lambda t:(-cones.potential(t),t))[:6]:
            pair=tuple(sorted((target,other)))
            if target==other or pair in seen:continue
            seen.add(pair);pairs.append(pair)
    # Round-robin candidate cap; not all node pairs or input combinations.
    pairs=pairs[:maximum];result=[];decoded={}
    def decode(s):
        if s not in decoded:decoded[s]=bits(vals[s],n).astype(np.int64)
        return decoded[s]
    for targets in pairs:
        if worker.remaining()<1:break
        losses=[]
        for state in range(4):
            response=forced_trace(c,worker.inputs[0],n,{t:(state>>i)&1 for i,t in enumerate(targets)},worker.stats)
            response=number(outputs(c,response),worker.order,n)
            losses.append(np.abs(response-worker.reference[0]))
        banned=set(targets)
        banned.update(forbidden_fanout(c,targets))
        direct=tuple(sorted({c.resolve(c.producer[t].pins[p]) for t in targets for p in c.lib.cells[c.producer[t].cell].inputs}-banned-{ZERO,ONE}))
        pool=list(dict.fromkeys([*direct,*c.inputs,*roots]))
        pool=[s for s in pool if s not in banned and s not in (ZERO,ONE)]
        ranked=[]
        for s in pool:
            _,loss=joint_truth(decode(s),losses,1);ranked.append((loss,c.level[s],s))
        pool=list(dict.fromkeys([*direct,*[s for _,_,s in sorted(ranked)[:16]]]))[:20]
        supports=[direct] if len(direct)<=4 else []
        names=();code=np.zeros(n,dtype=np.int64)
        for size in range(1,5):
            best=None
            for s in pool:
                if s in names:continue
                cd=code|(decode(s)<<(size-1));_,loss=joint_truth(cd,losses,size)
                worker.stats['joint_support_scores']+=1
                if best is None or (loss,c.level[s],s)<best[0]:best=((loss,c.level[s],s),s,cd)
            if best is None:break
            _,s,code=best;names=(*names,s);supports.append(names)
        supports=list(dict.fromkeys(s for s in supports if len(s)<=4))
        for names in supports:
            code=np.zeros(n,dtype=np.int64)
            for i,s in enumerate(names):code|=decode(s)<<i
            for penalty in (1,4):
                weighted=[loss+(penalty-1)*np.maximum(loss-base_loss,0) for loss in losses]
                choice,_=joint_truth(code,weighted,len(names))
                for variant in economic_truths(code,losses,len(names),choice,worker.metric.limit,worker.mapper):
                    total=int(np.choose(variant[code],losses).sum());worker.stats['joint_truth_scores']+=1
                    if total>worker.metric.limit:continue
                    replacements=[];truths=[]
                    for i,target in enumerate(targets):
                        truth=sum(int((int(state)>>i)&1)<<row for row,state in enumerate(variant))
                        truths.append(truth);replacements.append((target,names,worker.mapper.map(len(names),truth)))
                    try:trial=cleanup(c.materialize(replacements),worker.mapper)
                    except ValueError:continue
                    worker.stats['area_materializations']+=1
                    if trial.units>=c.units:continue
                    result.append((c.units-trial.units,total,trial,
                        dict(targets=targets,inputs=names,truths=truths,penalty=penalty),None))
    unique={structural_id(q[2]):q for q in result}
    return sorted(unique.values(),key=lambda q:(-q[0],q[1]))[:12]


def actual_trials(worker, plans, kind, development, ref_development, events):
    best=None;c=worker.current
    for item in plans:
        if worker.remaining()<1:break
        if kind=='signed':
            try:trial=cleanup(c.materialize([m.replacement() for m in item[2]]),worker.mapper)
            except ValueError:continue
            detail=dict(actions=[m.record() for m in item[2]])
        else:trial=item[2];detail=item[3]
        worker.stats['plan_actual_trials']+=1
        y=number(outputs(trial,trace(trial,worker.inputs[0],worker.count,worker.stats)),worker.order,worker.count)
        loss,value=worker.metric.exact(y,worker.reference[0])
        if kind not in ('signed','pair') and not np.array_equal(y,item[4]):raise AssertionError('Word/cofactor prediction differs')
        if kind=='pair' and loss!=item[1]:raise AssertionError('Joint truth prediction differs')
        row=dict(kind=kind,predicted_gain=item[0],predicted_loss=item[1],training_nmed=value,raw_area=c.lib.area(trial.units),accepted=False,**detail)
        events.append(row)
        if loss>worker.metric.limit:continue
        dev=number(outputs(trial,trace(trial,development,worker.count)),worker.order,worker.count)
        worker.stats['joint_tool_development_evaluations']+=1
        dev_loss,dev_value=worker.metric.exact(dev,ref_development);row['development_nmed']=dev_value
        if dev_loss>worker.metric.limit:continue
        if structural_id(trial) in worker.seen:continue
        if worker.backend:
            trial=worker.backend.optimize(trial,worker.remaining,variants=2)
            worker.stats['mapping_calls']=len(worker.backend.rows)
            worker.stats['mapping_cec_proofs']=sum(r.get('cec_equivalent',False) for r in worker.backend.rows)
        row['mapped_area']=c.lib.area(trial.units)
        if kind=='repair':
            if trial.units>c.units+16 or loss>=worker.loss:continue
            score=(-(worker.loss-loss)/max(1,1+trial.units-c.units),trial.units)
        else:
            if trial.units>=c.units:continue
            score=(trial.units,max(loss,dev_loss))
        if best is None or score<best[4]:
            best=trial,loss,dev_loss,row,score
    if best is None:return False
    trial,loss,dev_loss,row,_=best;row['accepted']=True
    worker.current=trial;worker.loss=loss;worker.lowest_units=min(worker.lowest_units,trial.units);worker.cached=None
    worker.seen.add(structural_id(trial));start=len(worker.history)
    worker.checkpoint('joint_'+kind+'_actual',[]);worker.annotate(start)
    print('[JOINT ACCEPT]',json.dumps(row,ensure_ascii=False),flush=True)
    return True


def run(args):
    folder=Path(args.output).resolve()
    if False:raise ValueError('D: only')
    folder.mkdir(parents=True,exist_ok=False);started=time.perf_counter();lib=Library(LIBRARY_PATH)
    snapshot=folder/'source';snapshot.mkdir();hashes={}
    for name in dict.fromkeys(SOURCES+['numeric_agentic.py','numeric_joint_word.py']):
        path=Path(__file__).resolve().parent/name;shutil.copy2(path,snapshot/name);hashes[name]=sha(path)
    source,order,audit=preflight(args.design,lib)
    baseline=load(args.accurate_netlist,lib) if args.accurate_netlist else source
    start=load(args.start,lib) if args.start else baseline
    if start.inputs!=source.inputs or start.outputs!=source.outputs:raise ValueError('Interface mismatch')
    mats=[matrix(args.design,source.inputs,i) for i in (0,1)]
    inputs=[pack(x,source.inputs) for x in mats]
    refs=[number(outputs(source,trace(source,x,10000)),order,10000) for x in inputs]
    for inp,ref in zip(inputs,refs):
        if not np.array_equal(number(outputs(baseline,trace(baseline,inp,10000)),order,10000),ref):raise ValueError('Declared accurate baseline differs from reference')
    metric=Metric(args.metric,args.bound,order)
    remaining=lambda:max(0,args.seconds-(time.perf_counter()-started))
    (folder/'worker').mkdir()
    w=Worker(start,order,inputs,refs,args,folder/'worker',remaining=remaining,reference_source=baseline)
    events=[];history=[]
    for iteration in range(args.rounds):
        if remaining()<1:break
        before=w.current.units;before_loss=w.loss;before_graph=structural_id(w.current)
        for kind in args.tools:
            if remaining()<1:break
            if kind=='native':
                for step in range(3):
                    if remaining()<1 or (structural_id(w.current),'batch',1.) in w.blocked:break
                    if not w.compress('balanced',1.)['changed']:break
                continue
            plans=signed_plans(w) if kind=='signed' else input_proposals(w) if kind=='input' else linked_input_proposals(w) if kind=='linked' else pair_proposals(w) if kind=='pair' else observation_proposals(w) if kind=='observe' else word_proposals(w,repair=kind=='repair')
            if kind=='repair':plans=wire_repair_proposals(w)+plans
            actual_trials(w,plans,kind,inputs[1],refs[1],events)
        history.append(dict(iteration=iteration,area=lib.area(w.current.units),seconds=time.perf_counter()-started))
        save(folder/'events.json',events);save(folder/'trajectory.json',history)
        print('[JOINT ROUND]',json.dumps(history[-1]),flush=True)
        if structural_id(w.current)==before_graph:break
    # Selection is frozen before held-out matrices/references are constructed.
    eligible=[]
    for checkpoint in w.history:
        if checkpoint['training_loss']>metric.limit:continue
        candidate=load(w.folder/checkpoint['path'],lib)
        y=number(outputs(candidate,trace(candidate,inputs[1],10000)),order,10000)
        w.stats['selection_development_evaluations']+=1
        dev_loss,_=metric.exact(y,refs[1])
        if dev_loss<=metric.limit:eligible.append((candidate.units,max(checkpoint['training_loss'],dev_loss),checkpoint['path'],candidate))
    if not eligible:raise RuntimeError('No training/development eligible checkpoint')
    eligible.sort(key=lambda q:q[:3]);w.current=eligible[0][3]
    w.current.write(folder/'selected_before_checks.v');selected_sha=sha(folder/'selected_before_checks.v')
    checks=[]
    for split in (2,100):
        x=matrix(args.design,source.inputs,split);inp=pack(x,source.inputs)
        y=number(outputs(w.current,trace(w.current,inp,10000)),order,10000)
        ref=number(outputs(source,trace(source,inp,10000)),order,10000)
        if not np.array_equal(y,number_numpy(validate_numpy(w.current,x),w.current.outputs,order)):raise AssertionError('Packed/NumPy mismatch')
        if not np.array_equal(ref,number_numpy(validate_numpy(source,x),source.outputs,order)):raise AssertionError('Reference mismatch')
        loss,value=metric.exact(y,ref);checks.append(dict(split=split,count=10000,nmed=value,passed=loss<=metric.limit))
    passed=all(c['passed'] for c in checks);dest=folder/('best.v' if passed else 'failed_final.v');w.current.write(dest)
    if sha(dest)!=selected_sha:raise AssertionError('Selection changed')
    for name,value in hashes.items():
        if sha(Path(__file__).resolve().parent/name)!=value:raise RuntimeError('Source changed during development experiment: '+name)
    result=dict(design=args.design,bound=args.bound,original_area=lib.area(baseline.units),starting_area=lib.area(start.units),area=lib.area(w.current.units),
        reduction=1-w.current.units/baseline.units,strict_pass=passed,checks=checks,whole_seconds=time.perf_counter()-started,
        stats=dict(w.stats),mapping_calls=len(w.backend.rows),mapping_cec=sum(r.get('cec_equivalent',False) for r in w.backend.rows),
        warm_start=bool(args.start),start_path=args.start,start_sha256=sha(args.start) if args.start else None,
        accurate_baseline=args.accurate_netlist,code_sha256=hashes,netlist_sha256=selected_sha,api_calls=0,exhaustive_lacs=False,
        scope='development experiment: signed-error plan beam and conditional L1 output rebuilding')
    save(folder/'report.json',result);print('[JOINT RESULT]',json.dumps(result,ensure_ascii=False),flush=True)
    return 0 if passed else 2


def parser():
    p=base_parser();p.description=__doc__;p.add_argument('--start',help='explicit approximate starting point; reported as warm start')
    p.add_argument('--tools',nargs='+',choices=('signed','word','input','pair','linked','repair','native','observe'),default=['signed','word'])
    p.add_argument('--rounds',type=int,default=12);p.set_defaults(controller='heuristic',seconds=120.,map_variants=2)
    return p


if __name__=='__main__':raise SystemExit(run(parser().parse_args()))
