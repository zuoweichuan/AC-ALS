"""Bounded bit-aware tools for strict-vector ER; old numerical tools unchanged."""
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import copy
import time
from statistics import NormalDist

import numpy as np
from bacs_numeric import Move, bits, cleanup, learned_sets, number, save
from bacs_frontier import VirtualArea, FrontierSearch,bounded_batch
from numeric_agentic import Worker,rollout_profile
from hybrid_identity import structural_id
from frontier import forbidden_fanout
from simulate import trace, outputs, plan_value
from netlist import ZERO, ONE


def error_words(actual, reference, order):
    return tuple(actual[n] ^ reference[n] for n in order)


def error_totals(words):
    bad=0
    for w in words:bad |= w
    return bad.bit_count(),sum(w.bit_count() for w in words)


def reference_words(records, order, count):
    a=np.frombuffer(records.tobytes(),dtype=np.uint8).reshape(count,-1)
    a=np.unpackbits(a,axis=1,bitorder='little')[:,:len(order)]
    return {n:int.from_bytes(np.packbits(a[:,i],bitorder='little').tobytes(),'little') for i,n in enumerate(order)}


@dataclass
class BitRepair(Move):
    hamming_gain:int=0
    output_changes:tuple=()
    def record(self):return dict(super().record(),hamming_gain=self.hamming_gain)


@dataclass
class JointRepair:
    components:tuple
    gain:int
    error:int
    hamming_gain:int
    signature:str
    family:str='er_complementary_repair'
    weight:int=0
    names:tuple=()
    truth:int=0
    @property
    def target(self):return '+'.join(m.target for m in self.components)
    def record(self):
        return dict(family=self.family,total_loss=self.error,gain_units=self.gain,
                    hamming_gain=self.hamming_gain,components=[m.record() for m in self.components])


def flatten(moves):
    return [p for m in moves for p in (m.components if isinstance(m,JointRepair) else (m,))]


def move_key(moves):return tuple((m.target,m.names,m.truth) for m in flatten(moves))


def development_limit(metric,args):
    """Predeclared one-sided Wilson guard, without changing the formal ER."""
    cap=int(metric.limit*args.development_fraction)
    confidence=getattr(args,'development_confidence',0.)
    if not confidence:return cap
    z=NormalDist().inv_cdf(confidence);n=metric.count
    def admissible(k):
        p=k/n
        upper=(p+z*z/(2*n)+z*((p*(1-p)/n+z*z/(4*n*n))**.5))/(1+z*z/n)
        return upper<=metric.bound
    lo,hi=0,cap
    while lo<hi:
        mid=(lo+hi+1)//2
        if admissible(mid):lo=mid
        else:hi=mid-1
    return lo


def overlap_batch(moves,scores,train_base,dev_base,train_capacity,dev_capacity,maximum=12):
    """Input-index union proxy, followed by actual joint propagation elsewhere.

    Joint nonlinear interactions are not assumed additive or independent. This
    bounded greedy proposal only ranks combinations; it is NOT an error bound.
    """
    chosen=[];used=set();train_new=0;dev_new=0
    while len(chosen)<maximum:
        ranked=[]
        for m in moves:
            if any(p.target in used for p in flatten([m])):continue
            t,d=scores[move_key([m])]
            tm=t['bad']&~train_base;dm=d['bad']&~dev_base
            nt=train_new|tm;nd=dev_new|dm
            if nt.bit_count()>train_capacity or nd.bit_count()>dev_capacity:continue
            marginal=(nt^train_new).bit_count()+(nd^dev_new).bit_count()
            ranked.append((m.gain/max(1,marginal),m.gain,-m.error,m,nt,nd))
        if not ranked:break
        _,_,_,m,train_new,dev_new=max(ranked,key=lambda r:r[:3]);chosen.append(m)
        used.update(p.target for p in flatten([m]))
    return chosen


def complementary_shortlist(pool,base_errors,max_pairs=6):
    """At most one promising partner per retained anchor, never a LAC product."""
    base,_=error_totals(base_errors);chosen={}
    for a in pool[:6]:
        peers=[]
        for b in pool:
            if b.target==a.target:continue
            bad,hd=error_totals(tuple(e^da^db for e,da,db in zip(base_errors,a.output_changes,b.output_changes)))
            if bad<base:peers.append((bad,hd,-a.gain-b.gain,b))
        if peers:
            score=min(peers,key=lambda r:r[:3]);b=score[-1]
            key=tuple(sorted(((a.target,a.names,a.truth),(b.target,b.names,b.truth))))
            chosen[key]=(score[:3],a,b)
    return [(a,b) for _,a,b in sorted(chosen.values(),key=lambda r:r[0])[:max_pairs]]


class ERWorker(Worker):
    def __init__(self,*args,**kwargs):
        self.refresh_counts={};self.bit_cache=None;self.verified_pairs={};self.development_view=None;self.training_view=None
        super().__init__(*args,**kwargs)
        self.er_reference=reference_words(self.reference[0],self.order,self.count)
        self.er_development_reference=reference_words(self.reference[1],self.order,self.count)

    def response_context(self,split):
        graph=structural_id(self.current)
        attr='development_view' if split else 'training_view';ctx=getattr(self,attr,None)
        reference=self.er_development_reference if split else self.er_reference
        if ctx is None or ctx['graph']!=graph:
            values=trace(self.current,self.inputs[split],self.count,self.stats)
            self.stats[('development' if split else 'training')+'_preview_full_simulations']+=1
            out=outputs(self.current,values);errors=error_words(out,reference,self.order)
            bad=0
            for e in errors:bad |= e
            ctx=dict(graph=graph,values=values,out=out,errors=errors,bad=bad,flips={},scores={});setattr(self,attr,ctx)
        return ctx

    def development_context(self):return self.response_context(1)
    def development_score(self,m):return self.response_score(m,1)

    def response_score(self,m,split):
        ctx=self.response_context(split);key=move_key([m]);prefix='development' if split else 'training'
        if key in ctx['scores']:return ctx['scores'][key]
        mask=(1<<self.count)-1
        if isinstance(m,JointRepair):
            trial=cleanup(self.current.materialize([p.replacement() for p in m.components]),self.mapper)
            out=outputs(trial,trace(trial,self.inputs[split],self.count,self.stats))
            self.stats[prefix+'_preview_full_simulations']+=1
            errors=error_words(out,self.er_development_reference if split else self.er_reference,self.order)
        else:
            target=m.target
            if target not in ctx['flips']:
                if target in self.current.producer:
                    fv=trace(self.current,self.inputs[split],self.count,self.stats,base=ctx['values'],flip=target)
                    flipped=outputs(self.current,fv);self.stats[prefix+'_preview_root_responses']+=1
                    ctx['flips'][target]=tuple(ctx['out'][o]^flipped[o] for o in self.order)
                else:ctx['flips'][target]=tuple(mask if o==target else 0 for o in self.order)
            word=plan_value(m.plan,m.names,self.current.lib,ctx['values'],mask)
            delta=ctx['values'][self.current.resolve(target)]^word
            errors=tuple(e^(d&delta) for e,d in zip(ctx['errors'],ctx['flips'][target]))
        bad=0
        for e in errors:bad |= e
        result=dict(loss=bad.bit_count(),bad=bad,positive_increment=(bad&(ctx['bad']^mask)).bit_count(),
                    hamming=sum(e.bit_count() for e in errors))
        self.stats[prefix+'_function_scores']+=1
        ctx['scores'][key]=result
        return result

    def compress(self,policy,fraction):
        """Screen a bounded shortlist on shared development responses too.

        Ordinary compression preserves the predeclared development admission
        bound. Explicit near branches remain free to explore outside it.
        """
        graph=structural_id(self.current);profile=rollout_profile(policy,fraction)
        if (graph,*profile) in self.blocked:raise ValueError('This exact profile already stalled')
        self.iteration+=1;self.steps+=1;start=len(self.history);before=self.loss
        moves=self.scan()['moves'];base=self.loss;limit=self.metric.limit
        positive=[m for m in moves if m.gain>0 and m.error<=limit]
        queues=[sorted(positive,key=lambda m:(m.error,-m.gain)),
                sorted(positive,key=lambda m:(-m.gain,m.error)),
                sorted(positive,key=lambda m:(-m.gain/max(1,m.weight),m.error))]
        short={}
        for i in range(12):
            for queue in queues:
                if i<len(queue):short.setdefault(move_key([queue[i]]),queue[i])
        dev=self.development_context();dev_base=dev['bad'].bit_count()
        dev_limit=development_limit(self.metric,self.args)
        tc=max(0,int((limit-base)*fraction));dc=max(0,int((dev_limit-dev_base)*fraction))
        eligible=[];weights=[];scores={}
        for m in short.values():
            if self.expired():break
            d=self.development_score(m)
            if d['loss']>min(dev_limit,dev_base+dc) or m.error>min(limit,base+tc):continue
            eligible.append(m)
            scores[move_key([m])]=(self.response_score(m,0),d)
            if m.weight<=tc and d['positive_increment']<=dc:
                n=copy.copy(m)
                n.weight=max(m.weight,((d['positive_increment']*tc+dc-1)//dc) if dc else 0)
                weights.append(n)
        proposals=[];tick=time.perf_counter()
        if policy in ('balanced','budget'):
            correlated=overlap_batch(eligible,scores,self.response_context(0)['bad'],dev['bad'],tc,dc)
            if correlated:
                proposals.append(correlated);self.stats['er_overlap_batch_proposals']+=1
            batch=bounded_batch(weights,tc,min(self.args.dp_seconds,max(.001,self.remaining())),self.stats)
            if batch:proposals.append(batch)
            if len(batch)>2:proposals.append(sorted(batch,key=lambda m:(-m.gain/max(1,m.weight),m.error))[:len(batch)//2])
        self.times['knapsack']+=time.perf_counter()-tick
        rank=(lambda m:(-m.gain,m.error)) if policy=='area' else (lambda m:(m.error,-m.gain))
        proposals.extend([m] for m in sorted(eligible,key=rank)[:6])
        ok=False;detail=dict(reason='no_train_development_admissible_proposal')
        for selected in proposals:
            if self.expired():break
            ok,detail=self.apply(selected,'agent_batch' if len(selected)>1 else 'agent_single')
            if ok:break
        if ok:self.refine('er_development_guarded_exact')
        changed=len(self.history)>start
        if not changed:self.blocked.add((structural_id(self.current),*profile))
        self.stalls=0 if changed else self.stalls+1
        self.last_result=dict(action='compress',accepted=ok,changed=changed,before_loss=before,
            after_loss=self.loss,policy=policy,fraction=fraction,detail=detail,
            failure_reason=None if ok else detail.get('reason'))
        self.events.append(dict(self.last_result,iteration=self.iteration,candidates=len(moves),
            development_shortlist=len(short),development_admitted=len(eligible)))
        save(self.folder/'iterations.json',self.events)
        return self.last_result

    def refresh(self):
        graph=structural_id(self.current);count=self.refresh_counts.get(graph,0)
        if count>=self.args.refresh_scans:raise ValueError('Bounded signature refreshes exhausted on this graph')
        self.refresh_counts[graph]=count+1;self.cached=None;self.bit_cache=None;self.verified_pairs={}
        self.blocked={p for p in self.blocked if p[0]!=graph}
        self.stats['er_signature_refreshes']+=1
        scan=self.scan()
        return dict(refresh=count+1,screened_candidates=len(scan['moves']),loss=scan['loss'])

    def candidates(self,repair_only=False):
        moves,total=super().candidates(repair_only)
        if total and not self.expired():moves+=self.bit_repairs()
        return moves,total

    def bit_repairs(self):
        graph=structural_id(self.current)
        if self.bit_cache is not None and self.bit_cache[0]==graph:return self.bit_cache[1]
        tick=time.perf_counter();c=self.current;mask=(1<<self.count)-1
        values=trace(c,self.inputs[0],self.count,self.stats);base=outputs(c,values)
        errors=error_words(base,self.er_reference,self.order);base_bad,base_hd=error_totals(errors)
        self.bit_baseline=(graph,base_hd)
        if not base_bad:return []
        base_er=np.zeros(self.count,dtype=np.int64);base_h=np.zeros(self.count,dtype=np.int64)
        for e in errors:
            decoded=bits(e,self.count).astype(np.int64);base_er |= decoded;base_h+=decoded
        area=VirtualArea(c,self.mapper)
        # Error-bearing POs (including constants/aliases) and MFFC roots each
        # receive a quota. A PO alias used internally is not a detachable root.
        raw_inputs={n for g in c.gates for p,n in g.pins.items() if p in c.lib.cells[g.cell].inputs}
        po=[n for n,e in sorted(zip(self.order,errors),key=lambda r:-r[1].bit_count())
            if e and n not in c.inputs and (n in c.producer or n not in raw_inputs)]
        structural=sorted(c.producer,key=lambda n:(-area.cones.potential(n),n))
        quota=self.args.bit_repair_roots
        roots=list(dict.fromkeys([*po[:max(2,quota//2)],*structural[:quota]]))[:quota]
        raw=[];debt=self.args.repair_debt_units
        for target in roots:
            if self.expired():break
            virtual=target not in c.producer
            if virtual:
                change=tuple(mask if o==target else 0 for o in self.order)
                flipped={o:base[o]^d for o,d in zip(self.order,change)}
                self.stats['er_virtual_output_responses']+=1
            else:
                fv=trace(c,self.inputs[0],self.count,self.stats,base=values,flip=target)
                flipped=outputs(c,fv);change=tuple(base[o]^flipped[o] for o in self.order)
            fe=error_words(flipped,self.er_reference,self.order)
            flip_er=np.zeros(self.count,dtype=np.int64);flip_h=np.zeros(self.count,dtype=np.int64)
            for e in fe:
                decoded=bits(e,self.count).astype(np.int64);flip_er |= decoded;flip_h+=decoded
            if not np.any((flip_er<base_er)|(flip_h<base_h)):continue
            t=values[c.resolve(target)];tb=bits(t,self.count).astype(bool)
            # The formal score remains vector ER. Bit recovery is an auxiliary
            # construction/ranking signal, never the acceptance constraint.
            objectives=[(base_er*(len(self.order)+1)+base_h,flip_er*(len(self.order)+1)+flip_h),
                        (base_h+4*len(self.order)*base_er,flip_h+4*len(self.order)*flip_er)]
            unique={}
            for loss,floss in objectives:
                l0=np.where(tb,floss,loss);l1=np.where(tb,loss,floss)
                sets=learned_sets(c,target,values,l0,l1,self.count,self.stats)
                if virtual:
                    forbidden=forbidden_fanout(c,[target]);sets=[(ns,x) for ns,x in sets if not any(n in forbidden for n in ns)]
                from bacs_frontier import response_truth
                for names,code in sets:
                    for penalty in (1,4):
                        truth=response_truth(code,l0,l1,loss,len(names),penalty)
                        plan=self.mapper.map(len(names),truth)
                        word=plan_value(plan,names,c.lib,values,mask);delta=t^word
                        pred=tuple(e^(d&delta) for e,d in zip(errors,change));bad,hd=error_totals(pred)
                        self.stats['function_error_evaluations']+=1;self.stats['er_bit_truths_scored']+=1
                        if bad>base_bad or (bad==base_bad and hd>=base_hd):continue
                        gain=(area.gain(target,names,plan) if not virtual else -self.mapper.cost(plan))
                        if gain < -debt:continue
                        out={o:base[o]^(d&delta) for o,d in zip(self.order,change)}
                        sig=hashlib.blake2b(number(out,self.order,self.count).tobytes(),digest_size=16).hexdigest()
                        m=BitRepair(target,names,truth,plan,gain,bad,max(0,bad-base_bad),sig,
                            'er_partial_bit_repair',penalty,base_hd-hd,tuple(d&delta for d in change))
                        unique[(names,truth)]=m
            best=sorted(unique.values(),key=lambda m:(m.error,-m.hamming_gain,max(0,-m.gain)))
            # Preserve a cheap alternative and a strong partial repair per root.
            keep=best[:2]
            cheap=sorted(best,key=lambda m:(max(0,-m.gain),m.error,-m.hamming_gain))
            if cheap and cheap[0] not in keep:keep.append(cheap[0])
            raw+=keep
        raw.sort(key=lambda m:(m.error,-m.hamming_gain,max(0,-m.gain),m.target))
        # Round-robin across roots prevents one root consuming the pair pool.
        groups=defaultdict(list)
        for m in raw:groups[m.target].append(m)
        pool=[group[0] for group in groups.values()][:12]
        pool += [m for m in raw if m not in pool][:max(0,12-len(pool))]
        joint=[];self.verified_pairs={}
        for pair in complementary_shortlist(pool,errors,self.args.repair_pairs):
            if self.expired():break
            try:trial=cleanup(c.materialize([m.replacement() for m in pair]),self.mapper)
            except ValueError:continue
            self.stats['area_materializations']+=1
            actual=outputs(trial,trace(trial,self.inputs[0],self.count,self.stats))
            total,hd=error_totals(error_words(actual,self.er_reference,self.order))
            self.stats['er_complementary_joint_verifications']+=1
            if total>=base_bad or c.units-trial.units < -debt:continue
            sig=hashlib.blake2b(number(actual,self.order,self.count).tobytes(),digest_size=16).hexdigest()
            m=JointRepair(tuple(pair),c.units-trial.units,total,base_hd-hd,sig)
            self.verified_pairs[(graph,move_key([m]))]=(trial,total,hd,sig)
            joint.append(m)
        result=joint+raw[:24];self.bit_cache=(graph,result)
        self.times['er_bit_repair_subset']+=time.perf_counter()-tick
        return result

    def apply(self,selected,phase):
        graph=structural_id(self.current);key=move_key(selected);fingerprint=(graph,phase,key)
        if fingerprint in self.failed:return False,dict(reason='already_measured_rejection')
        parts=flatten(selected)
        if len({m.target for m in parts})!=len(parts):return False,dict(reason='duplicate_target')
        tick=time.perf_counter();cached=self.verified_pairs.get((graph,key))
        try:
            if cached:trial,total,hd,sig=cached;self.stats['er_verified_response_reuses']+=1
            else:
                trial=cleanup(self.current.materialize([m.replacement() for m in parts]),self.mapper)
                self.stats['area_materializations']+=1
                actual=outputs(trial,trace(trial,self.inputs[0],self.count,self.stats))
                total,hd=error_totals(error_words(actual,self.er_reference,self.order))
                sig=hashlib.blake2b(number(actual,self.order,self.count).tobytes(),digest_size=16).hexdigest()
        except ValueError as exc:return False,dict(reason='invalid_joint_graph',error=str(exc))
        self.stats['joint_verifications' if len(parts)>1 else 'single_verifications']+=1
        if len(selected)==1 and sig!=selected[0].signature:raise AssertionError('ER shared response disagrees with actual child')
        detail=dict(error_sum=total,metric=total/self.count,area=self.source.lib.area(trial.units),
            actual_gain_units=self.current.units-trial.units,hamming_errors=hd,actions=[m.record() for m in selected])
        reason=None
        if structural_id(trial) in self.seen:reason='known_physical_circuit'
        limit=self.metric.limit
        partial=phase=='compensation' and all(getattr(m,'hamming_gain',0)>0 for m in selected)
        ceiling=int(limit*self.args.near_ratio) if phase=='compensation' and self.loss>limit else limit
        if total>ceiling:reason='actual_metric_exceeds_bound'
        if phase!='compensation' and hasattr(self,'er_development_reference'):
            if len(selected)==1:
                dev_loss=self.development_score(selected[0])['loss']
            else:
                dv=outputs(trial,trace(trial,self.inputs[1],self.count,self.stats))
                self.stats['development_preview_full_simulations']+=1
                dev_loss=error_totals(error_words(dv,self.er_development_reference,self.order))[0]
            detail['development_loss']=dev_loss
            if dev_loss>development_limit(self.metric,self.args):reason='development_admission_rejected'
        if phase=='compensation':
            if not (total<self.loss or (partial and total==self.loss)):reason='no_vector_or_partial_bit_recovery'
            if total==self.loss:
                baseline=getattr(self,'bit_baseline',None)
                base_hd=(baseline[1] if baseline and baseline[0]==graph else
                         sum(e.bit_count() for e in self.response_context(0)['errors']))
                if hd>=base_hd:reason='no_actual_partial_bit_recovery'
            if trial.units>min(self.source.units,self.lowest_units+self.args.repair_debt_units):reason='cumulative_area_debt'
        elif trial.units>=self.current.units:reason='no_actual_area_gain'
        self.times['selected_actual_verification']+=time.perf_counter()-tick
        if reason:
            self.failed.add(fingerprint);return False,dict(detail,reason=reason)
        self.current=trial;self.loss=total;self.lowest_units=min(self.lowest_units,trial.units)
        self.seen.add(structural_id(trial));start=len(self.history)
        self.checkpoint(phase,selected);self.annotate(start)
        self.bit_cache=None;self.verified_pairs={}
        return True,detail
