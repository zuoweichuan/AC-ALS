"""Bounded exact cut and polarity surgery on an explicitly supplied warm netlist.

No approximation, sampled truth inference, or author solver is used here.
Each local plan is derived from complete small-support truth tables. Global
CEC is required before any smaller mapped circuit is reported as a winner.
"""
import runtime
import argparse,time,json,hashlib
from pathlib import Path
from logic import Library,evaluate
from cutmap import CutMapper,variable
from netlist import ZERO,ONE
from bacs_data import load,LIBRARY_PATH
from bacs_numeric import cleanup,save
from bacs_frontier import VirtualArea
from hybrid_identity import structural_id
from research_c880_hybrid import proof

def gate_truth(c,g,names,invert_inputs=(),invert_output=False):
    n=len(names);mask=(1<<(1<<n))-1
    values={ZERO:0,ONE:mask,**{v:variable(n,i) for i,v in enumerate(names)}}
    inv=set(invert_inputs);cell=c.lib.cells[g.cell]
    pins={}
    for p in cell.inputs:
        net=c.resolve(g.pins[p]);pins[p]=values[net]^(mask if net in inv else 0)
    t=evaluate(next(iter(cell.functions.values())),{'0':0,'1':mask,**pins},mask)
    return t^(mask if invert_output else 0)

def phase_replacements(c,target,mapper):
    return phase_group_replacements(c,(target,),mapper)

def phase_group_replacements(c,targets,mapper):
    targets={c.resolve(n) for n in targets}
    if targets & {c.resolve(n) for n in c.outputs}:return None
    touched={}
    for target in targets:
        if target not in c.producer:return None
        for g in (c.producer[target],*c.fanouts.get(target,())):touched[id(g)]=g
    gates=list(touched.values())
    if any(len(c.lib.cells[h.cell].functions)!=1 for h in gates):return None
    plans=[]
    for h in gates:
        cell=c.lib.cells[h.cell]
        out=h.pins[next(iter(cell.functions))]
        names=tuple(sorted({c.resolve(h.pins[p]) for p in cell.inputs}-{ZERO,ONE}))
        if not names or len(names)>mapper.maximum:return None
        truth=gate_truth(c,h,names,targets,out in targets)
        plan=mapper.map(len(names),truth)
        plans.append((out,names,plan))
    old=sum(c.lib.cells[h.cell].units for h in gates)
    new=sum(mapper.cost(p) for _,_,p in plans)
    return old-new,plans

def cut_truth(c,target,leaves):
    mask=(1<<(1<<len(leaves)))-1
    vals={ZERO:0,ONE:mask,**{n:variable(len(leaves),i) for i,n in enumerate(leaves)}}
    def visit(n):
        n=c.resolve(n)
        if n in vals:return vals[n]
        g=c.producer.get(n)
        if g is None:raise ValueError('Cut does not cover root')
        cell=c.lib.cells[g.cell]
        pin=next(p for p in cell.functions if g.pins.get(p)==n)
        vals[n]=evaluate(cell.functions[pin],{'0':0,'1':mask,**{p:visit(g.pins[p]) for p in cell.inputs}},mask)
        return vals[n]
    return visit(target)

def cuts(c,target,width=5,beam=4):
    states=[frozenset((target,))];seen=set(states);found=set()
    for depth in range(9):
        nexts={}
        for boundary in states:
            for n in sorted(boundary):
                g=c.producer.get(n)
                if g is None:continue
                leaves={c.resolve(g.pins[p]) for p in c.lib.cells[g.cell].inputs}-{ZERO,ONE}
                b=(boundary-{n})|leaves
                if not 0<len(b)<=width or b in seen:continue
                b=frozenset(b);seen.add(b);found.add(tuple(sorted(b)))
                nexts[b]=(sum(c.level[v] for v in b),len(b),tuple(sorted(b)))
        # Two depth-oriented and two support-oriented cut states, not all cuts.
        selected=list(dict.fromkeys([*sorted(nexts,key=lambda b:nexts[b])[:max(1,beam//2)],
            *sorted(nexts,key=lambda b:(len(b),nexts[b]))[:beam]]))[:beam]
        states=selected
        if not states:break
    return sorted(found,key=lambda ns:(len(ns),sum(c.level[v] for v in ns),ns))

def neighbors(c,mapper,stats,remaining,max_trials=32,allow_neutral=False,paired=False):
    proposals=[];area=VirtualArea(c,mapper)
    for target in sorted(c.producer):
        if remaining()<1:break
        row=phase_replacements(c,target,mapper)
        if row is not None:
            gain,plans=row;stats['polarity_truth_plans']+=1
            if gain>=0:proposals.append((gain,'phase',target,plans))
    if paired:
        # Screen topology-connected phase domains near inverter-heavy gates.
        # This is a bounded structural shortlist, not all node pairs/products.
        anchors=sorted(c.producer,key=lambda n:(
            -sum(c.lib.cells[g.cell].name.startswith('INV') for g in (c.producer[n],*c.fanouts.get(n,()))),n))[:24]
        groups=set()
        for n in anchors:
            g=c.producer[n];cell=c.lib.cells[g.cell]
            peers=[c.resolve(g.pins[p]) for p in cell.inputs if c.resolve(g.pins[p]) in c.producer]
            peers += [h.pins[p] for h in c.fanouts.get(n,()) for p in c.lib.cells[h.cell].functions if p in h.pins]
            peers=list(dict.fromkeys(peers))[:4]
            groups.update(tuple(sorted((n,m))) for m in peers if m!=n)
            if len(peers)>=2:groups.add(tuple(sorted((n,*peers[:2]))))
        for group in sorted(groups):
            row=phase_group_replacements(c,group,mapper)
            if row is None:continue
            gain,plans=row;stats['joint_phase_domain_scores']+=1
            if gain>=0:proposals.append((gain,'joint_phase','+'.join(group),plans))
    roots=sorted(c.producer,key=lambda n:(-area.cones.potential(n),n))[:64]
    for target in roots:
        if remaining()<1:break
        for names in cuts(c,target,mapper.maximum):
            if remaining()<1:break
            truth=cut_truth(c,target,names);plan=mapper.map(len(names),truth)
            stats['exact_cut_truths']+=1
            gain=area.gain(target,names,plan)
            if gain>0:proposals.append((gain,'cut',target,[(target,names,plan)]))
    proposals.sort(key=lambda r:(-r[0],r[1],r[2]))
    results=[];seen=set();tried=set()
    for gain,kind,target,plans in proposals:
        if remaining()<1 or len(results)>=max_trials:break
        if target in tried and kind=='cut':continue
        tried.add(target)
        try:trial=cleanup(c.materialize(plans),mapper)
        except ValueError:continue
        stats['actual_local_materializations']+=1
        if trial.units>c.units or (not allow_neutral and trial.units==c.units):continue
        key=structural_id(trial)
        if key in seen:continue
        seen.add(key);results.append((trial,dict(kind=kind,target=target,proxy_gain=gain,actual_gain=c.units-trial.units)))
    return sorted(results,key=lambda r:(r[0].units,r[1]['kind'],r[1]['target']))

def run(a):
    from collections import Counter
    folder=Path(a.output).resolve()
    if False:raise ValueError('D: only')
    folder.mkdir(parents=True,exist_ok=False)
    (folder/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    lib=Library(LIBRARY_PATH);source=load(a.input,lib);source.write(folder/'input.v')
    save(folder/'protocol.json',dict(input=str(Path(a.input).resolve()),input_sha256=hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),
        algorithm='bounded exact cuts + joint polarity surgery',width=a.width,seconds=a.seconds,api_calls=0,approximate_changes=False))
    tick=time.perf_counter();remaining=lambda:max(0,a.seconds-(time.perf_counter()-tick))
    mapper=CutMapper(lib,a.width);stats=Counter();best=source;events=[];visited={structural_id(source)}
    frontier=[source];neutral_steps=0
    for iteration in range(a.rounds):
        if remaining()<2:break
        children=[]
        for parent in frontier[:4]:
            for trial,event in neighbors(parent,mapper,stats,remaining,allow_neutral=True,paired=a.paired):
                key=structural_id(trial)
                if key in visited:continue
                visited.add(key);children.append((trial,event))
        if not children:break
        children.sort(key=lambda r:(r[0].units,r[1]['kind'],r[1]['target']))
        minimum=children[0][0].units
        if minimum<best.units:
            trial,event=children[0]
            passed=proof(source,trial,folder/('cec_%03d'%iteration))
            stats['global_cec_proofs']+=1
            if not passed:raise AssertionError('Exact surgery changed circuit function')
            best=trial;best.write(folder/'best.v');events.append(dict(event,iteration=iteration,units=best.units,area=lib.area(best.units),cec=True))
            print('[EXACT LOCAL ACCEPT] '+json.dumps(events[-1]),flush=True)
            neutral_steps=0
        else:
            neutral_steps+=1
            if neutral_steps>3:break
        frontier=[r[0] for r in children[:4] if r[0].units<=best.units]
        if not frontier:break
    best.write(folder/'best.v')
    result=dict(input=str(Path(a.input).resolve()),initial_area=lib.area(source.units),area=lib.area(best.units),
        baseline_area=198.17,reduction=1-lib.area(best.units)/198.17,seconds=time.perf_counter()-tick,
        events=events,stats=dict(stats),api_calls=0,approximate_changes=False,improved=best.units<source.units)
    save(folder/'report.json',result);print('[EXACT LOCAL RESULT] '+json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seconds',type=float,default=120);p.add_argument('--width',type=int,choices=(4,5,6),default=5)
    p.add_argument('--rounds',type=int,default=12);p.add_argument('--paired',action='store_true');run(p.parse_args())
