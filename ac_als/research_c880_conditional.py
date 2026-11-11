"""Bounded care-conditioned divisor construction, isolated from frozen suites."""
import copy
import hashlib
import time
from pathlib import Path
import numpy as np
import er_search_resub as baseline
from research_c880_views import author_view
from er_repair_tools import ERWorker,development_limit,move_key,overlap_batch
from bacs_frontier import VirtualArea,pareto_choices,bounded_batch
from bacs_numeric import bits,number,save
from frontier import forbidden_fanout
from simulate import trace,outputs,pack
from hybrid_identity import structural_id
from netlist import ZERO,ONE
from numeric_agentic import rollout_profile


def support_summary(names,values,care0,care1,mask):
    regions=[mask]
    for n in names:
        v=values[n];regions=[r&(v^mask) for r in regions]+[r&v for r in regions]
    truth=care=error=0
    for row,region in enumerate(regions):
        a=(region&care1).bit_count();b=(region&care0).bit_count()
        error+=min(a,b)
        if b<a:truth |= 1<<row
        if a!=b:care |= 1<<row
    return error,truth,care


class ConditionalWorker(ERWorker):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if getattr(self.args,'truth_width',4)>4:
            from cutmap import CutMapper
            self.mapper=CutMapper(self.source.lib,self.args.truth_width)

    def candidates(self,repair_only=False):
        moves,total=super().candidates(repair_only)
        if self.expired():return moves,total
        started=time.perf_counter();c=self.current;mask=(1<<self.count)-1
        ctx=self.response_context(0);values=ctx['values'];base_y=number(ctx['out'],self.order,self.count)
        base_loss=self.metric.loss(base_y,self.reference[0]);area=VirtualArea(c,self.mapper)
        targets=sorted(c.producer,key=lambda n:(-area.cones.potential(n),n))[:self.args.conditional_roots]
        extra=[]
        for target in targets:
            if self.expired():break
            flipped=outputs(c,trace(c,self.inputs[0],self.count,self.stats,base=values,flip=target))
            flip_y=number(flipped,self.order,self.count);flip_loss=self.metric.loss(flip_y,self.reference[0])
            tb=bits(values[target],self.count).astype(bool)
            l0=np.where(tb,flip_loss,base_loss);l1=np.where(tb,base_loss,flip_loss)
            def word(x):return int.from_bytes(np.packbits(x,bitorder='little').tobytes(),'little')
            care0=word(l0<l1);care1=word(l1<l0)
            if not care0|care1:continue
            forbidden=forbidden_fanout(c,[target])
            available=[n for n in (*c.inputs,*c.producer) if n not in forbidden and n!=target]
            cache={}
            def summary(names):
                if names not in cache:
                    cache[names]=support_summary(names,values,care0,care1,mask)
                    self.stats['conditional_support_proxies']+=1
                return cache[names]
            ranked=sorted(available,key=lambda n:(summary((n,))[0],-area.cones.saved_with_inputs(target,(n,)),n))
            g=c.producer[target];direct=tuple(sorted({c.resolve(g.pins[p]) for p in c.lib.cells[g.cell].inputs}-{ZERO,ONE}))
            pool=list(dict.fromkeys([*direct,*ranked]))[:96]
            anchors=list(dict.fromkeys([*direct,*ranked[:6]]))[:8]
            supports=[(n,) for n in ranked[:6]]
            for a in anchors:
                pairs=[tuple(sorted((a,b))) for b in pool if a!=b]
                pairs.sort(key=lambda ns:(summary(ns)[0],-area.cones.saved_with_inputs(target,ns),ns))
                supports+=pairs[:2]
            supports=list(dict.fromkeys(supports))
            bestpairs=sorted((ns for ns in supports if len(ns)==2),key=lambda ns:(summary(ns)[0],ns))[:2]
            for ns in bestpairs:
                extended=[tuple(sorted((*ns,n))) for n in ranked[:12] if n not in ns]
                supports+=sorted(extended,key=lambda ss:(summary(ss)[0],-area.cones.saved_with_inputs(target,ss),ss))[:2]
            width=getattr(self.args,'truth_width',4)
            beam=sorted((ss for ss in supports if len(ss)==3),key=lambda ss:(summary(ss)[0],ss))[:2]
            for k in range(4,width+1):
                proposals={tuple(sorted((*ns,n))) for ns in beam for n in ranked[:12] if n not in ns}
                beam=sorted(proposals,key=lambda ss:(summary(ss)[0],-area.cones.saved_with_inputs(target,ss),ss))[:2]
                supports+=beam
            if 0<len(direct)<=width:supports.append(direct)
            measured=[]
            for names in dict.fromkeys(supports):
                _,truth,care=summary(names);n=len(names)
                default=self.mapper.map(n,truth)
                choices=[(self.mapper.cost(default),truth,default)]
                # Physical cell templates only resolve equal-loss don't-cares;
                # no enumeration of Boolean functions or circuit LAC products.
                for t,p in self.mapper.catalog[n].items():
                    if (t&care)==(truth&care):choices.append((self.mapper.cost(p),t,p))
                _,t,plan=min(choices,key=lambda row:(row[0],row[1]))
                m=self.scored(target,names,t,plan,values,values[target],base_y,flip_y,base_loss,flip_loss,'care_conditioned_resub')
                if m is None:continue
                m.gain=area.gain(target,names,plan)
                if m.gain>0 or (m.error<total and m.gain>=-self.args.repair_debt_units):measured.append(m)
            extra+=pareto_choices(measured,8)
        self.stats['conditional_candidates_retained']+=len(extra)
        self.times['conditional_divisor_construction']+=time.perf_counter()-started
        result={move_key([m]):m for m in moves}
        for m in extra:result[move_key([m])]=m
        return list(result.values()),total

    def compress(self,policy,fraction):
        graph=structural_id(self.current);profile=rollout_profile(policy,fraction)
        if (graph,*profile) in self.blocked:raise ValueError('Profile already stalled')
        self.iteration+=1;self.steps+=1;start=len(self.history);before=self.loss
        moves=self.scan()['moves'];base=self.loss;limit=self.metric.limit
        positive=[m for m in moves if m.gain>0 and m.error<=limit]
        queues=[sorted(positive,key=lambda m:(m.error,-m.gain)),sorted(positive,key=lambda m:(-m.gain,m.error)),
                sorted(positive,key=lambda m:(-m.gain/max(1,m.weight),m.error))]
        short={}
        for i in range(self.args.conditional_shortlist//3):
            for q in queues:
                if i<len(q):short.setdefault(move_key([q[i]]),q[i])
        dev=self.development_context();dev_base=dev['bad'].bit_count();dev_limit=development_limit(self.metric,self.args)
        tc=max(0,int((limit-base)*fraction));dc=max(0,int((dev_limit-dev_base)*fraction))
        eligible=[];weights=[];scores={}
        for m in short.values():
            if self.expired():break
            d=self.development_score(m)
            if d['loss']>min(dev_limit,dev_base+dc) or m.error>min(limit,base+tc):continue
            eligible.append(m);scores[move_key([m])]=(self.response_score(m,0),d)
            if m.weight<=tc and d['positive_increment']<=dc:
                n=copy.copy(m);n.weight=max(m.weight,((d['positive_increment']*tc+dc-1)//dc) if dc else 0);weights.append(n)
        proposals=[]
        if policy in ('balanced','budget'):
            group=overlap_batch(eligible,scores,self.response_context(0)['bad'],dev['bad'],tc,dc)
            if group:proposals.append(group)
            batch=bounded_batch(weights,tc,min(self.args.dp_seconds,max(.001,self.remaining())),self.stats)
            if batch:proposals.append(batch)
        rank=(lambda m:(-m.gain,m.error)) if policy=='area' else (lambda m:(m.error,-m.gain))
        proposals.extend([m] for m in sorted(eligible,key=rank)[:8])
        ok=False;detail=dict(reason='no_admissible_proposal')
        for group in proposals:
            if self.expired():break
            ok,detail=self.apply(group,'agent_batch' if len(group)>1 else 'agent_single')
            if ok:break
        if ok:self.refine('conditional_resub_exact')
        changed=len(self.history)>start
        if not changed:self.blocked.add((structural_id(self.current),*profile))
        self.stalls=0 if changed else self.stalls+1
        self.last_result=dict(changed=changed,accepted=ok,before_loss=before,after_loss=self.loss,policy=policy,fraction=fraction,detail=detail)
        self.events.append(dict(self.last_result,iteration=self.iteration,candidates=len(moves),shortlist=len(short)))
        save(self.folder/'iterations.json',self.events);return self.last_result


class ConditionalViews:
    def add(self,circuit,origin,preserve):
        if len(self.active)>=4:raise ValueError('Branch slots full')
        bid='b%03d'%self.next_branch;self.next_branch+=1;folder=self.folder/bid;folder.mkdir()
        options=copy.copy(self.args);options.logic_view=self.args.research_view=='expanded';options.sop_view=False;options.no_exact=False
        w=ConditionalWorker(circuit,self.order,self.inputs,self.refs,options,folder,remaining=self.remaining,
            preserve=preserve or options.logic_view,reference_source=self.source)
        if self.args.research_view=='author':
            w.current=author_view(self.source,folder);w.args.logic_view=True;w.preserve=True;w.loss=0
            w.lowest_units=w.current.units;w.seen.add(structural_id(w.current))
            w.checkpoint('author_accurate_standard_cell_work_view',[]);w.annotate(len(w.history)-1);w.refine('author_initial_map')
        self.workers[bid]=w;self.active.append(bid);self.origin[bid]=dict(origin,research_view=self.args.research_view)
        return bid

class MCTS(ConditionalViews,baseline.MCTS):pass
class Agent(ConditionalViews,baseline.Agent):pass

def run(a):
    oldm,olda=baseline.MCTS,baseline.Agent;baseline.MCTS,baseline.Agent=MCTS,Agent
    try:
        code=baseline.run(a);folder=Path(a.output)
        for name in ('research_c880_conditional.py','research_c880_views.py'):
            source=Path(__file__).parent/name;(folder/'source'/name).write_bytes(source.read_bytes())
        return code
    finally:baseline.MCTS,baseline.Agent=oldm,olda

if __name__=='__main__':
    p=baseline.parser();p.description=__doc__
    p.add_argument('--research-view',choices=('mapped','expanded','author'),default='mapped')
    p.add_argument('--conditional-roots',type=int,default=48);p.add_argument('--conditional-shortlist',type=int,default=96)
    p.add_argument('--truth-width',type=int,choices=(4,5,6),default=4)
    a=p.parse_args()
    if a.design!='c880':p.error('Isolated c880 experiment only')
    raise SystemExit(run(a))
