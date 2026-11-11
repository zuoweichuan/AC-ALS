"""Bounded two-output resynthesis, including shared-error recovery windows."""
from dataclasses import dataclass
from itertools import combinations
from logic import evaluate
from netlist import ZERO,ONE
from simulate import trace,wrong
from frontier import partitions,boundary_cuts,normalize,forbidden_fanout


def forced_trace(circuit,inputs,count,base,forced,counters):
    mask=(1<<count)-1;values=dict(base);values.update(forced);affected=set(forced);evaluated=0
    for gate in circuit.gates:
        cell=circuit.lib.cells[gate.cell];deps={p:circuit.resolve(gate.pins[p]) for p in cell.inputs}
        if not set(deps.values()) & affected:continue
        env={'0':0,'1':mask,**{p:values[n] for p,n in deps.items()}}
        for pin,fun in cell.functions.items():
            if pin not in gate.pins:continue
            net=gate.pins[pin]
            if net not in forced:values[net]=evaluate(fun,env,mask)
            affected.add(net)
        evaluated+=1
    counters['joint_response_simulations']=counters.get('joint_response_simulations',0)+1
    counters['gate_evaluations']=counters.get('gate_evaluations',0)+evaluated
    return values


@dataclass
class JointMove:
    targets:tuple
    inputs:tuple
    truths:tuple
    plans:tuple
    gain:int
    error:int
    bad:int
    def replacements(self):return [(t,self.inputs,p) for t,p in zip(self.targets,self.plans)]
    def record(self):return dict(targets=self.targets,inputs=self.inputs,truths=self.truths,gain_units=self.gain,error=self.error,family='two_output_cut')


def windows(circuit,values,reference,count,quota=12):
    roots=list(circuit.producer);zero=circuit.lib.mapper.plans[0]
    potential={t:circuit.units-circuit.materialize([(t,(),zero)]).units for t in roots}
    top=sorted(roots,key=lambda t:(-potential[t],t))[:20];pairs=set();sample=(1<<min(count,1024))-1
    for target in top:
        peers=[n for n in top if n!=target]
        peers.sort(key=lambda n:(min(((values[n]^values[target])&sample).bit_count(),
                                     ((values[n]^values[target]^sample)&sample).bit_count()),-potential[n],n))
        for peer in peers[:2]:pairs.add(tuple(sorted((target,peer))))
    error={circuit.resolve(o):values[circuit.resolve(o)]^reference[o] for o in circuit.outputs
           if circuit.resolve(o) in circuit.producer}
    residual=sorted(error,key=lambda n:(-error[n].bit_count(),n))[:8]
    residual_pairs=set()
    for n in residual:
        peers=sorted((s for s in residual if s!=n),key=lambda s:(-(error[n]&error[s]).bit_count(),s))
        for s in peers[:2]:residual_pairs.add(tuple(sorted((n,s))))
    def gain(pair):return circuit.units-circuit.materialize([(t,(),zero) for t in pair]).units
    chosen=sorted(pairs,key=lambda p:(-gain(p),p))[:max(1,quota-4)]
    for pair in sorted(residual_pairs,key=lambda p:(-(error[p[0]]&error[p[1]]).bit_count(),-gain(p),p)):
        if pair not in chosen:chosen.append(pair)
        if len(chosen)>=quota:break
    return chosen


def joint_candidates(circuit,inputs,count,reference,mapper,counters,ceiling=.065):
    values=trace(circuit,inputs,count,counters);mask=(1<<count)-1;result=[];max_error=int(ceiling*count)
    for targets in windows(circuit,values,reference,count):
        response=[]
        for action in range(4):
            forced={n:mask if (action>>i)&1 else 0 for i,n in enumerate(targets)}
            response.append(wrong(circuit,forced_trace(circuit,inputs,count,values,forced,counters),reference))
        forbidden=forbidden_fanout(circuit,targets)
        available=[n for n in (*circuit.inputs,*circuit.producer) if n not in forbidden]
        def correlation(n):
            return min(min((values[n]^values[t]).bit_count(),count-(values[n]^values[t]).bit_count()) for t in targets),circuit.level[n],n
        pool=[]
        for target in targets:
            ranked=sorted(available,key=lambda n:(min((values[n]^values[target]).bit_count(),count-(values[n]^values[target]).bit_count()),circuit.level[n],n))[:8]
            for n in ranked:
                if n not in pool:pool.append(n)
        cuts=[boundary_cuts(circuit,t,mapper.maximum) for t in targets];seeds=set()
        for a in (cuts[0][:2]+cuts[0][-2:]):
            for b in (cuts[1][:2]+cuts[1][-2:]):
                names=tuple(sorted(set(a)|set(b)))
                if len(names)<=mapper.maximum and all(n not in forbidden for n in names):seeds.add(names)
                for n in names:
                    if n not in forbidden and n not in pool:pool.append(n)
        pool=pool[:18]
        costs={}
        def rows(names):
            if names not in costs:
                costs[names]=[[ (part&resp).bit_count() for resp in response] for part in partitions(names,values,mask)]
            return costs[names]
        def score(names):return sum(min(row) for row in rows(names)),sum(circuit.level[n] for n in names),names
        beam=[()]
        for width in range(1,mapper.maximum+1):
            choices={tuple(sorted(names+(n,))) for names in beam for n in pool if n not in names}
            beam=sorted(choices,key=score)[:3];seeds.update(beam[:2])
        for names in sorted(seeds,key=lambda ns:(score(ns)[0],len(ns),ns))[:5]:
            table=rows(names);k=len(names)
            if sum(min(row) for row in table)>max_error:continue
            actions=[min(range(4),key=lambda a:(row[a],a)) for row in table]
            optimum=tuple(sum(((a>>o)&1)<<i for i,a in enumerate(actions)) for o in range(2))
            def err(truths):return sum(row[((truths[0]>>i)&1)|(((truths[1]>>i)&1)<<1)] for i,row in enumerate(table))
            catalog=mapper.catalog[k];proposals={optimum}
            for start in (optimum,(0,0),((1<<(1<<k))-1,)*2):
                t=list(start)
                for repeat in range(2):
                    for out in range(2):
                        def loss(truth):
                            q=list(t);q[out]=truth;return err(q)
                        shortlist=sorted(catalog,key=lambda truth:(loss(truth),mapper.cost(catalog[truth]),truth))[:3]
                        for truth in shortlist:
                            q=list(t);q[out]=truth;proposals.add(tuple(q))
                        if shortlist:t[out]=shortlist[0]
                proposals.add(tuple(t))
            ranked=[]
            for truths in proposals:
                error=err(truths)
                if error>max_error:continue
                plans=tuple(mapper.map(k,t) for t in truths)
                ranked.append((error,sum(mapper.cost(p) for p in plans),truths,plans))
            by_area=sorted(ranked,key=lambda r:(r[1],r[0],r[2]))
            picks=[];seen=set()
            for row in by_area[:4]+sorted(ranked,key=lambda r:(r[0],r[1],r[2]))[:2]:
                if row[2] in seen:continue
                seen.add(row[2]);picks.append(row)
            for error,_,truths,plans in picks:
                trial=normalize(circuit.materialize([(t,names,p) for t,p in zip(targets,plans)]));gain=circuit.units-trial.units
                counters['joint_area_materializations']=counters.get('joint_area_materializations',0)+1
                counters['joint_function_evaluations']=counters.get('joint_function_evaluations',0)+1
                if gain>0 or (gain>=-8 and error+5<wrong(circuit,values,reference).bit_count()):
                    # Reconstruct the exact erroneous-input set from responses.
                    pieces=partitions(names,values,mask);newbad=0
                    for i,part in enumerate(pieces):newbad |=part & response[((truths[0]>>i)&1)|(((truths[1]>>i)&1)<<1)]
                    result.append(JointMove(targets,names,truths,plans,gain,error,newbad))
    return result
