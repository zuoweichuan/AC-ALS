"""Bounded cut/divisor learning with exact sampled vector-ER response costs."""
from dataclasses import dataclass
from functools import reduce
from itertools import combinations
from time import perf_counter
from logic import evaluate,Plan
from netlist import ZERO,ONE
from simulate import trace,wrong,outputs,plan_value
from cutmap import variable
from weighted_cost import WeightedCost


def normalize(circuit):
    current=circuit.simplify()
    # Exact functional hashing of individual cells, accounting for pin order,
    # tied inputs and constants. This is not simulation-based equivalence.
    while True:
        seen={};aliases=[]
        for g in current.gates:
            cell=current.lib.cells[g.cell]
            if len(cell.functions)!=1:continue
            names=tuple(sorted({current.resolve(g.pins[p]) for p in cell.inputs}-{ZERO,ONE}))
            if len(names)>6:continue
            mask=(1<<(1<<len(names)))-1
            vals={ZERO:0,ONE:mask,**{n:variable(len(names),i) for i,n in enumerate(names)}}
            truth=evaluate(next(iter(cell.functions.values())),{'0':0,'1':mask,**{p:vals[current.resolve(g.pins[p])] for p in cell.inputs}},mask)
            key=(names,truth);target=g.pins[next(iter(cell.functions))]
            if key in seen:aliases.append((target,(seen[key],),Plan('leaf',leaf=0)))
            else:seen[key]=target
        if not aliases:return current
        trial=current.materialize(aliases).simplify()
        if trial.units>=current.units:return current
        current=trial


@dataclass
class Move:
    target:str
    inputs:tuple
    truth:int
    plan:object
    gain:int
    bad:int
    error:int
    family:str
    def replacement(self):return self.target,self.inputs,self.plan
    def record(self):return dict(target=self.target,inputs=self.inputs,truth=self.truth,gain_units=self.gain,error=self.error,family=self.family)


def partitions(names,values,mask):
    rows=[mask]
    for name in names:
        bits=values[name]
        rows=[r & (bits^mask) for r in rows]+[r & bits for r in rows]
    return rows


def boundary_cuts(circuit,target,width=4):
    boundary={target};result=[]
    for step in range(10):
        proposals=[]
        for n in boundary:
            gate=circuit.producer.get(n)
            if gate is None:continue
            leaves={circuit.resolve(gate.pins[p]) for p in circuit.lib.cells[gate.cell].inputs}-{ZERO,ONE}
            b=(boundary-{n})|leaves
            if 0<len(b)<=width:
                proposals.append((circuit.lib.cells[gate.cell].units,-len(b),n,b))
        if not proposals:break
        boundary=max(proposals,key=lambda r:r[:3])[-1]
        cut=tuple(sorted(boundary))
        if cut not in result:result.append(cut)
    return result


def row_costs(names,values,mask,bad0,bad1):
    parts=partitions(names,values,mask)
    return [(p&bad0).bit_count() for p in parts],[(p&bad1).bit_count() for p in parts]


def forbidden_fanout(circuit,targets):
    forbidden=set(targets);pending=list(targets)
    while pending:
        n=pending.pop()
        for gate in circuit.fanouts.get(n,()):
            for pin in circuit.lib.cells[gate.cell].functions:
                out=gate.pins.get(pin)
                if out is not None and out not in forbidden:forbidden.add(out);pending.append(out)
    return forbidden


def selected_inputs(circuit,target,values,mask,bad0,bad1,pool_size=12,beam_width=3,maximum=4,enhanced=False):
    cuts=boundary_cuts(circuit,target,width=maximum)
    target_bits=values[target];weight=bad0^bad1
    forbidden=forbidden_fanout(circuit,[target])
    available=[n for n in (*circuit.inputs,*circuit.producer) if n not in forbidden]
    def score(n):
        d=((values[n]^target_bits)&weight).bit_count();total=weight.bit_count()
        return min(d,total-d),circuit.level[n],n
    pool=sorted(available,key=score)[:pool_size]
    for n in (cuts[-1] if cuts else ()):
        if n not in pool:pool.append(n)
    pool=pool[:pool_size+4]
    cache={}
    def quality(names):
        if names not in cache:
            a,b=row_costs(names,values,mask,bad0,bad1)
            cache[names]=(sum(min(x,y) for x,y in zip(a,b)),sum(circuit.level[n] for n in names),names)
        return cache[names]
    beam=[()];selected=set(cuts)
    for width in range(1,maximum+1):
        choices={tuple(sorted(s+(n,))) for s in beam for n in pool if n not in s}
        beam=sorted(choices,key=quality)[:beam_width]
        selected.update(beam[:2])
    ordered=sorted(selected,key=lambda s:(quality(s)[0],len(s),quality(s)[1],s))
    # Retain exact cut boundaries as well as information-selected external cones.
    keep=ordered[:4]
    if enhanced:
        # Price the logic actually released when the new cut inputs remain live.
        gone=set(circuit.producer)-set(circuit.materialize([(target,(),circuit.lib.mapper.plans[0])]).producer)
        rescued={}
        def ancestors(n):
            if n in rescued:return rescued[n]
            gate=circuit.producer.get(n)
            if gate is None:return set()
            nodes={n} if n in gone else set()
            for pin in circuit.lib.cells[gate.cell].inputs:nodes|=ancestors(circuit.resolve(gate.pins[pin]))
            rescued[n]=nodes;return nodes
        def potential(names):
            live=set().union(*(ancestors(n) for n in names))
            return sum(circuit.lib.cells[circuit.producer[n].cell].units for n in gone-live)
        area_order=sorted(selected,key=lambda s:(-potential(s),quality(s)[0],len(s),s))
        keep=ordered[:2]
        for names in area_order[:3]:
            if names not in keep:keep.append(names)
    for cut in cuts[-2:]:
        if cut not in keep:keep.append(cut)
    return keep,pool,len(cache)


def candidates(circuit,inputs,count,reference,mapper,counters,*,ceiling=.065,enhanced=False,residual=False,patch=False,
               targets=None,base=None,flip_errors=None,potentials=None,template_cache=None,repair_debt_units=8,shared_seeds=None):
    tick=perf_counter();values=trace(circuit,inputs,count,counters) if base is None else base
    bad=wrong(circuit,values,reference);mask=(1<<count)-1;limit=int(ceiling*count)
    roots=[];zero=circuit.lib.mapper.plans[0]
    for target in (circuit.producer if targets is None else targets):
        potential=(circuit.units-circuit.materialize([(target,(),zero)]).units) if potentials is None or target not in potentials else potentials[target]
        roots.append((potential,circuit.level[target],target))
    roots.sort(reverse=True);found=[];seen=set()
    for potential,_,target in roots:
        if flip_errors is None or target not in flip_errors:
            f=trace(circuit,inputs,count,counters,base=values,flip=target)
            flip=wrong(circuit,f,reference)
        else:flip=flip_errors[target]
        t=values[target]
        bad0=(bad & (t^mask)) | (flip&t)
        bad1=(bad&t) | (flip & (t^mask))
        cuts,pool,scored=selected_inputs(circuit,target,values,mask,bad0,bad1,maximum=mapper.maximum,enhanced=enhanced)
        counters['input_sets_scored']=counters.get('input_sets_scored',0)+scored
        proposals=[((),0,zero,'constant'),((),1,circuit.lib.mapper.plans[15],'constant')]
        for n in pool[:12]:
            proposals.extend([((n,),2,Plan('leaf',leaf=0),'wire'),((n,),1,circuit.lib.mapper.plans[5],'wire')])
        for names in cuts:
            k=len(names);a,b=row_costs(names,values,mask,bad0,bad1)
            if sum(min(x,y) for x,y in zip(a,b))>limit:continue
            optimal=sum((y<x)<<i for i,(x,y) in enumerate(zip(a,b)))
            catalog=mapper.catalog[k]
            if residual:
                from residual_models import residual_models
                models,scored=residual_models(a,b,k,mapper,limit)
                counters['residual_function_costs_scored']=counters.get('residual_function_costs_scored',0)+scored
                for truth,plan in models:proposals.append((names,truth,plan,'residual_two_cell'))
            err=WeightedCost(a,b)
            direct=sorted(catalog,key=lambda truth:(err(truth),catalog[truth].units,truth))[:6]
            if enhanced:
                cheap=sorted((truth for truth in catalog if err(truth)<=limit),key=lambda truth:(catalog[truth].units,err(truth),truth))[:4]
                direct=list(dict.fromkeys(direct+cheap))
                from learn_support import support_models
                models,scored=support_models(a,b,k,mapper,limit)
                counters['support_models_scored']=counters.get('support_models_scored',0)+scored
                for truth,plan in models:proposals.append((names,truth,plan,'support_pruned'))
            learned={optimal}
            cheap_rows=sorted(range(1<<k),key=lambda i:(abs(a[i]-b[i]),i))[:6]
            learned.update(optimal^(1<<i) for i in cheap_rows)
            # A small directed two-row beam, not all truth-table completions.
            learned.update(optimal^(1<<i)^(1<<j) for i,j in list(combinations(cheap_rows[:4],2))[:4])
            for truth in direct:
                if err(truth)<=limit:proposals.append((names,truth,catalog[truth],'cell'))
            for truth in sorted(learned,key=lambda truth:(err(truth),truth))[:8]:
                if err(truth)<=limit:proposals.append((names,truth,mapper.map(k,truth),'learned_cut'))
            counters['local_truth_costs_scored']=counters.get('local_truth_costs_scored',0)+len(err.cache)
        measured=[]
        for names,truth,plan,family in proposals:
            identity=(target,names,truth)
            if identity in seen:continue
            seen.add(identity)
            word=plan_value(plan,names,circuit.lib,values,mask)
            delta=t^word;newbad=(bad&(delta^mask))|(flip&delta);error=newbad.bit_count()
            counters['function_error_evaluations']=counters.get('function_error_evaluations',0)+1
            if error>limit:continue
            # Rank cheap implementations before real graph construction; retain
            # several distinct error sets rather than only the lowest scalar ER.
            proxy=potential-mapper.cost(plan)
            measured.append((error,-proxy,names,truth,plan,family,newbad))
        if patch:
            from signal_patch import signal_patches
            patches=signal_patches(circuit,target,measured,values,mask,bad,flip,mapper,limit,counters)
            for error,cost,names,truth,plan,family,newbad in patches:
                ident=(target,names,truth)
                if ident in seen:continue
                seen.add(ident)
                measured.append((error,cost-potential,names,truth,plan,family,newbad))
                counters['function_error_evaluations']=counters.get('function_error_evaluations',0)+1
        measured.sort(key=lambda x:(x[0],x[1],x[2],x[3]))
        by_area=sorted(measured,key=lambda x:(x[1],x[0],x[2],x[3]))
        shortlist=[];ids=set()
        for row in [*by_area[:5],*measured[:3]]:
            ident=(row[2],row[3])
            if ident not in ids:ids.add(ident);shortlist.append(row)
        if repair_debt_units>8:
            for fraction in (.75,.85,.95):
                rows=[r for r in measured if r[0]<=round(bad.bit_count()*fraction)]
                if rows:
                    row=min(rows,key=lambda r:(r[1],r[0],r[2],r[3]));ident=(row[2],row[3])
                    if ident not in ids:ids.add(ident);shortlist.append(row)
        for error,_,names,truth,plan,family,newbad in shortlist:
            trial=normalize(circuit.materialize([(target,names,plan)]))
            gain=circuit.units-trial.units
            counters['area_materializations']=counters.get('area_materializations',0)+1
            if template_cache is not None and gain>=-8:
                template_cache.add([Move(target,names,truth,plan,gain,newbad,error,family)])
            # Optional current-parent side channel. A neutral individual can
            # participate in a profitable shared-cone deletion; do not change
            # the ordinary single-action acceptance/filtering for old callers.
            if shared_seeds is not None and gain>=-8:
                shared_seeds.append(Move(target,names,truth,plan,gain,newbad,error,family))
            if gain>0 or (gain>=-repair_debt_units and error+5<bad.bit_count()):
                found.append(Move(target,names,truth,plan,gain,newbad,error,family))
    counters['generation_seconds']=counters.get('generation_seconds',0)+perf_counter()-tick
    return found,bad
