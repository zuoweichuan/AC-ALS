"""Detached output-function resynthesis with exact vector-ER union screening."""
import time
from netlist import Circuit,Gate,ZERO,ONE
from bacs_numeric import cleanup
from simulate import trace,outputs,plan_value
from er_repair_tools import development_limit,error_words,error_totals
from hybrid_identity import structural_id
from frontier import boundary_cuts


def patch_output(circuit,target,names,plan):
    used=set(circuit.producer)|set(circuit.inputs)|set(circuit.outputs)|set(circuit.aliases)
    def fresh():
        n='__po_lac_%d'%len(used)
        while n in used:n+='x'
        used.add(n);return n
    old=fresh();r=lambda n:old if n==target else n
    gates=[Gate(g.name,g.cell,{p:r(n) for p,n in g.pins.items()}) for g in circuit.gates]
    aliases={r(n):r(v) for n,v in circuit.aliases.items()};memo={}
    def emit(p):
        if p.name=='leaf':return r(names[p.leaf]) if p.leaf>=0 else (ZERO,ONE)[p.leaf+2]
        children=tuple(emit(k) for k in p.children);key=(p.name,children)
        if key not in memo:
            cell=circuit.lib.cells[p.name];n=fresh()
            gates.append(Gate('po_lac_'+str(len(gates)),p.name,{**dict(zip(cell.inputs,children)),next(iter(cell.functions)):n}));memo[key]=n
        return memo[key]
    aliases[target]=emit(plan)
    return Circuit(circuit.lib,circuit.inputs,circuit.outputs,gates,aliases,circuit.name)


def step(worker,trials=12):
    tick=time.perf_counter();c=worker.current;mask=(1<<worker.count)-1
    ctx=[worker.response_context(k) for k in (0,1)];refs=[worker.er_reference,worker.er_development_reference]
    limits=[worker.metric.limit,development_limit(worker.metric,worker.args)];proposals=[]
    for target in worker.order:
        if worker.remaining()<2:break
        if target in c.inputs:continue
        other=[]
        for k in (0,1):
            bad=0
            for o in worker.order:
                if o!=target:bad |= ctx[k]['out'][o]^refs[k][o]
            other.append(bad)
        def predict(names,plan):
            result=[]
            for k in (0,1):
                word=plan_value(plan,names,c.lib,ctx[k]['values'],mask)
                result.append((other[k]|(word^refs[k][target])).bit_count())
            worker.stats['po_function_union_scores']+=1
            return result
        choices=[]
        for truth in (0,15):
            plan=c.lib.mapper.plans[truth];loss=predict((),plan)
            if all(v<=limit for v,limit in zip(loss,limits)):choices.append(((),truth,plan,loss))
        ranked=[]
        for n in (*c.inputs,*c.producer):
            if n==target:continue
            for truth in (10,5):
                plan=c.lib.mapper.plans[truth];loss=predict((n,),plan)
                ranked.append((sum(loss),n,truth,plan,loss))
        ranked.sort(key=lambda r:r[:3])
        for _,n,t,p,loss in ranked[:4]:
            if all(v<=limit for v,limit in zip(loss,limits)):choices.append(((n,),t,p,loss))
        pool=list(dict.fromkeys(r[1] for r in ranked))[:12]
        root=c.resolve(target);structural=[]
        if root in c.producer:
            g=c.producer[root]
            direct=tuple(sorted({c.resolve(g.pins[p]) for p in c.lib.cells[g.cell].inputs}-{ZERO,ONE}))
            if 0<len(direct)<=4:structural.append(direct)
            structural+=boundary_cuts(c,root,width=4)[-3:]
            pool=list(dict.fromkeys([*direct,*pool]))[:20]
        pairs={}
        for a in pool[:4]:
            for b in pool:
                if a==b:continue
                names=tuple(sorted((a,b)));cost0=[0]*4;cost1=[0]*4
                for k in (0,1):
                    av=ctx[k]['values'][names[0]];bv=ctx[k]['values'][names[1]];care=other[k]^mask;gold=refs[k][target]
                    for row in range(4):
                        region=(av if row&1 else av^mask)&(bv if row&2 else bv^mask)&care
                        cost0[row]+=(region&gold).bit_count();cost1[row]+=(region&(gold^mask)).bit_count()
                truth=care=0
                for row,(x,y) in enumerate(zip(cost0,cost1)):
                    if y<x:truth |= 1<<row
                    if x!=y:care |= 1<<row
                truth,plan=c.lib.mapper.complete(care,truth&care);loss=predict(names,plan)
                pairs[names]=(sum(loss),names,truth,plan,loss)
        for _,names,truth,plan,loss in sorted(pairs.values(),key=lambda r:r[:2])[:6]:
            if all(v<=limit for v,limit in zip(loss,limits)):choices.append((names,truth,plan,loss))
        supports=list(structural)
        seedpairs=sorted(pairs.values(),key=lambda r:r[:2])[:2]
        for _,ns,_,_,_ in seedpairs:
            supports += [tuple(sorted((*ns,n))) for n in pool[:8] if n not in ns][:4]
        for names in dict.fromkeys(s for s in supports if 0<len(s)<=4):
            n=len(names);cost0=[0]*(1<<n);cost1=[0]*(1<<n)
            for k in (0,1):
                regions=[other[k]^mask]
                for name in names:
                    word=ctx[k]['values'][name]
                    regions=[r&(word^mask) for r in regions]+[r&word for r in regions]
                gold=refs[k][target]
                for row,region in enumerate(regions):
                    cost0[row]+=(region&gold).bit_count();cost1[row]+=(region&(gold^mask)).bit_count()
            truth=care=0
            for row,(x,y) in enumerate(zip(cost0,cost1)):
                if y<x:truth |= 1<<row
                if x!=y:care |= 1<<row
            plan=worker.mapper.map(n,truth);plans=[(worker.mapper.cost(plan),truth,plan)]
            for t,p in worker.mapper.catalog[n].items():
                if (t&care)==(truth&care):plans.append((worker.mapper.cost(p),t,p))
            _,truth,plan=min(plans,key=lambda r:r[:2]);loss=predict(names,plan)
            if all(v<=limit for v,limit in zip(loss,limits)):choices.append((names,truth,plan,loss))
        unique=set()
        for names,truth,plan,loss in choices:
            key=(names,truth)
            if key in unique:continue
            unique.add(key)
            try:trial=cleanup(patch_output(c,target,names,plan),worker.mapper)
            except ValueError:continue
            worker.stats['po_screened_materializations']+=1
            if trial.units>=c.units or structural_id(trial) in worker.seen:continue
            proposals.append(dict(target=target,names=names,truth=truth,plan=plan,losses=loss,trial=trial,units=trial.units))
    proposals.sort(key=lambda p:(p['units'],sum(p['losses'])))
    best=None;records=[]
    for p in proposals[:trials]:
        if worker.remaining()<2:break
        mapped=worker.backend.optimize(p['trial'],worker.remaining,variants=worker.args.map_variants)
        losses=[]
        for k in (0,1):
            y=outputs(mapped,trace(mapped,worker.inputs[k],worker.count,worker.stats))
            losses.append(error_totals(error_words(y,refs[k],worker.order))[0])
        worker.stats['po_actual_joint_verifications']+=1
        if losses!=p['losses']:raise AssertionError('Detached-output prediction mismatch')
        record=dict(target=p['target'],inputs=p['names'],truth=p['truth'],wrong_vectors=losses,area=c.lib.area(mapped.units))
        records.append(record)
        if mapped.units>=c.units or structural_id(mapped) in worker.seen:continue
        key=(mapped.units,sum(losses))
        if best is None or key<best[0]:best=(key,mapped,record)
    if best:
        _,worker.current,record=best;worker.loss=record['wrong_vectors'][0];worker.cached=None;worker.bit_cache=None;worker.verified_pairs={}
        worker.lowest_units=min(worker.lowest_units,worker.current.units);worker.seen.add(structural_id(worker.current))
        worker.checkpoint('detached_po_resynthesis',[]);worker.annotate(len(worker.history)-1)
        worker.history[-1]['po_action']=record
    worker.times['po_resynthesis']+=time.perf_counter()-tick
    worker.stats['mapping_calls']=len(worker.backend.rows);worker.stats['mapping_cec_proofs']=sum(r.get('cec_equivalent',False) for r in worker.backend.rows)
    return dict(changed=best is not None,screened=len(proposals),trials=records)
