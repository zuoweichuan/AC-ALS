"""Independent Agent + compensation-aware numerical portfolio.

Agent selects representations, bounded rollouts, near-miss branches, explicit
compensation and stopping. Existing search/Agent entry points are unchanged.
"""
import runtime
import argparse
from collections import Counter, defaultdict
import copy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from bacs_frontier import FrontierSearch, bounded_batch, parser as local_parser, SOURCES
from bacs_numeric import Metric, cleanup, number, number_numpy, save
from bacs_data import ROOT, DESIGNS, LIBRARY_PATH, load, matrix, preflight, source_path
from hybrid_identity import structural_id
from logic import Library
from numeric_exact import ExactBackend
from simulate import outputs, pack, trace, validate_numpy

CODE=Path(__file__).resolve().parent
EXTRA=['numeric_agentic.py','numeric_agent_provider.py','numeric_agent_knowledge.json',
       'hybrid_agent.py','hybrid_provider.py','hybrid_secrets.py']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def short_metric(total,metric):return total/metric.count/metric.denominator


def select_checkpoint(rows):
    return min((r for r in rows if r['eligible']),key=lambda r:(r['units'],r['branch'],r['path']))


def rollout_profile(policy,fraction):
    return ('batch' if policy in ('balanced','budget') else policy,round(float(fraction),6))


def compression_choices(moves,base,limit,fraction):
    capacity=max(0,int((limit-base)*fraction))
    positive=[m for m in moves if m.gain>0 and m.error<=limit]
    batch=[m for m in positive if m.weight<=capacity]
    # For ONE measured replacement, net total error already includes cancellation
    # of old errors. Its positive increments alone need not fit remaining room.
    # Only additive batching needs that conservative capacity test.
    singles=[m for m in positive if m.error<=min(limit,base+capacity)]
    return batch,singles,capacity


def decision_reserve(receipts):
    recent=[r['seconds'] for r in receipts[-4:] if r]
    return min(30.,max(12.,1.5*max(recent,default=5.)))


def timing_fields(total,search_wall,llm_overhead,preflight=0.,http=None):
    if min(total,search_wall,llm_overhead,preflight)<0 or llm_overhead>search_wall or search_wall+preflight>total+1e-6:
        raise ValueError('Inconsistent non-overlapping timing intervals')
    if http is not None and not 0<=http<=llm_overhead:raise ValueError('HTTP time must be a subset of LLM overhead')
    return dict(whole_seconds=total,search_and_agent_seconds=search_wall,
        llm_call_overhead_seconds=llm_overhead,llm_http_request_seconds=http,
        whole_seconds_excluding_llm=total-llm_overhead,
        whole_seconds_excluding_llm_http=None if http is None else total-http,
        numerical_search_seconds=search_wall-llm_overhead,
        preflight_seconds=preflight,postselection_seconds=max(0.,total-search_wall-preflight))


class Worker(FrontierSearch):
    def __init__(self,source,order,inputs,refs,args,folder,*,remaining,preserve=False,reference_source=None):
        self.preserve=preserve;self.global_remaining=remaining;self.cached=None;self.failed=set()
        self.loss=None;self.stalls=0;self.steps=0;self.last_result={};self.blocked=set()
        super().__init__(source,order,inputs,refs,args,folder)
        if reference_source is not None:self.source=reference_source
        y=number(outputs(self.current,trace(self.current,self.inputs[0],10000,self.stats)),self.order,10000)
        self.loss=self.metric.exact(y,self.reference[0])[0]
        self.annotate(0)

    def remaining(self):return min(super().remaining(),self.global_remaining())
    def expired(self):return self.remaining()<=0

    def refine(self,phase):
        # Mapping is always a physical checkpoint; adopting its topology is a
        # separate choice. Never compare a logical view cost with mapped gain.
        original=self.current;lowest=self.lowest_units;start=len(self.history)
        super().refine(phase)
        if self.preserve:self.current=original;self.lowest_units=lowest
        if self.loss is not None:self.annotate(start)

    def annotate(self,start):
        for row in self.history[start:]:
            row.update(training_loss=self.loss,training_metric=short_metric(self.loss,self.metric),
                       working_view_preserved=self.preserve)
        save(self.folder/'history.json',self.history)

    def apply(self,selected,phase):
        fingerprint=(structural_id(self.current),phase,tuple((m.target,m.names,m.truth) for m in selected))
        if fingerprint in self.failed:return False,dict(reason='already_measured_rejection')
        if phase=='compensation' and not all(m.error<self.loss for m in selected):
            return False,dict(reason='compensation_requires_measured_error_improvement')
        start=len(self.history);ok,detail=super().apply(selected,phase)
        if ok:self.loss=detail['error_sum'];self.annotate(start)
        else:self.failed.add(fingerprint)
        return ok,detail

    def scan(self):
        identity=structural_id(self.current)
        if self.cached is None or self.cached['hash']!=identity:
            moves,total=self.candidates()
            if total>self.metric.limit and self.metric.kind=='mse':
                # Saturation preserves feasibility screening but not the true
                # size of an infeasible MSE. Never call a large overshoot slight.
                y=number(outputs(self.current,trace(self.current,self.inputs[0],10000,self.stats)),self.order,10000)
                total=self.metric.exact(y,self.reference[0])[0]
            self.loss=total
            self.cached=dict(hash=identity,moves=moves,loss=total)
        return self.cached

    def compress(self,policy,fraction):
        identity=structural_id(self.current)
        profile=rollout_profile(policy,fraction)
        if (identity,*profile) in self.blocked:
            raise ValueError('Identical rollout already stalled on this graph; change fraction, representation, repair, or branch')
        self.iteration+=1;self.steps+=1;old=len(self.history);before=self.loss
        if self.quantize():
            self.loss=self.history[-1]['word_action']['loss'];self.annotate(old)
            if self.word_area_gain>0:self.refine('agent_word_exact')
        scan=self.scan();base=scan['loss']
        allowed,singles,capacity=compression_choices(scan['moves'],base,self.metric.limit,fraction)
        proposals=[];tick=time.perf_counter()
        if policy in ('balanced','budget'):
            batch=bounded_batch(allowed,capacity,min(self.args.dp_seconds,max(.001,self.remaining())),self.stats)
            if batch:proposals.append(batch)
            if len(batch)>2:
                ranked=sorted(batch,key=lambda m:(-m.gain/max(1,m.weight),m.error))
                proposals.append(ranked[:max(1,len(ranked)//2)])
        self.times['knapsack']+=time.perf_counter()-tick
        key=(lambda m:(-m.gain,m.error)) if policy=='area' else (lambda m:(m.error,-m.gain))
        proposals.extend([m] for m in sorted(singles,key=key)[:3])
        ok=False;detail=dict(reason='no_screened_feasible_proposal')
        for selected in proposals:
            if self.expired():break
            ok,detail=self.apply(selected,'agent_batch' if len(selected)>1 else 'agent_single')
            if ok:break
        if ok:self.refine('agent_accepted_exact')
        changed=len(self.history)>old
        if not changed:self.blocked.add((structural_id(self.current),*profile))
        self.stalls=0 if changed else self.stalls+1
        self.last_result=dict(action='compress',accepted=ok,changed=changed,before_loss=before,
            after_loss=self.loss,policy=policy,fraction=fraction,detail=detail,
            failure_reason=None if ok else detail.get('reason'))
        self.events.append(dict(self.last_result,iteration=self.iteration,candidates=len(scan['moves'])))
        save(self.folder/'iterations.json',self.events)
        return self.last_result


def checkpoint_menu(history):
    """Keep structural milestones as well as scalar area/error endpoints."""
    keep={}
    def add(row,category):
        keep.setdefault(row['path'],dict(row,categories=[]))['categories'].append(category)
    for r in sorted(history,key=lambda r:(r['units'],r['path']))[:2]:add(r,'area')
    for r in sorted(history,key=lambda r:(r['training_loss'],r['units'],r['path']))[:2]:add(r,'error')
    for iteration in sorted({r['iteration'] for r in history if r['iteration']>0})[:3]:
        r=min((r for r in history if r['iteration']==iteration),key=lambda r:(r['units'],r['path']))
        add(r,'early_structural_milestone')
    return list(keep.values())


def clone_worker(parent,folder,remaining,started,circuit=None,loss=None,preserve=None):
    """Clone a measured graph without re-normalizing intermediate topology."""
    w=copy.copy(parent);w.folder=Path(folder);w.folder.mkdir()
    w.current=(parent.current if circuit is None else circuit).copy()
    w.loss=parent.loss if loss is None else loss
    if preserve is not None:w.preserve=preserve
    w.history=[];w.events=[];w.stats=defaultdict(int);w.times=defaultdict(float)
    w.started=started;w.global_remaining=remaining;w.rng=copy.deepcopy(parent.rng)
    w.seen=set(parent.seen);w.failed=set(parent.failed);w.blocked=set(parent.blocked)
    w.quantization_seen=set(parent.quantization_seen);w.near_events=[];w.cached=None
    w.steps=0;w.iteration=0;w.stalls=0;w.repairs=0;w.last_result={}
    w.backend=ExactBackend(w.folder/'exact',LIBRARY_PATH,distro=w.args.wsl_distro,
        executable=w.args.abc,timeout=w.args.map_seconds)
    w.checkpoint('measured_parent_snapshot',[]);w.annotate(0)
    return w


def near_repair_rollout(w,m,steps,repair_trials,remaining):
    """Explicitly authorized bounded compound tool; no LAC enumeration."""
    c=cleanup(w.current.materialize([m.replacement()]),w.mapper);w.stats['area_materializations']+=1
    y=number(outputs(c,trace(c,w.inputs[0],10000,w.stats)),w.order,10000)
    total,value=w.metric.exact(y,w.reference[0]);w.stats['near_miss_joint_verifications']+=1
    if hashlib.blake2b(y.tobytes(),digest_size=16).hexdigest()!=m.signature:raise AssertionError('Near child response mismatch')
    detail=dict(actual_near_metric=value,actual_near_area=w.source.lib.area(c.units),repaired=False)
    if c.units>=w.current.units or total>int(w.metric.limit*w.args.near_ratio):
        return dict(detail,reason='actual_window_or_area_rejected')
    w.current=c;w.loss=total;w.lowest_units=min(w.lowest_units,c.units)
    w.checkpoint('explicit_near_infeasible',[m]);w.annotate(len(w.history)-1)
    moves=w.scan()['moves']
    repairs=sorted((r for r in moves if r.error<=w.metric.limit and r.error<total),
        key=lambda r:((max(0,-r.gain)+1)/max(1,total-r.error),r.error))[:repair_trials]
    detail['screened_repairs']=len(repairs)
    for repair in repairs:
        if remaining()<.5:break
        ok,info=w.apply([repair],'compensation')
        if ok:
            detail.update(repaired=True,repair=info);w.repairs+=1;w.refine('explicit_near_repaired_exact')
            detail['steps']=[]
            for _ in range(steps):
                if remaining()<.5 or (structural_id(w.current),'batch',1.) in w.blocked:break
                r=w.compress('balanced',1.)
                detail['steps'].append({k:v for k,v in r.items() if k!='detail'})
                if not r['changed']:break
            break
    return detail


def candidate_context(worker,bid):
    scan=worker.scan();selected={};categories={}
    moves=scan['moves'];limit=worker.metric.limit
    queues=[('area',sorted((m for m in moves if m.gain>0 and m.error<=limit),key=lambda m:(-m.gain,m.error))),
        ('budget',sorted((m for m in moves if m.gain>0 and m.error<=limit),key=lambda m:(-m.gain/max(1,m.weight),m.error))),
        ('repair',sorted((m for m in moves if m.error<worker.loss and m.error<=limit),key=lambda m:(m.error,-m.gain))),
        ('near',sorted((m for m in moves if limit<m.error<=limit*worker.args.near_ratio and m.gain>0),
                       key=lambda m:((m.error-limit)/m.gain,-m.gain)))]
    index={id(m):i for i,m in enumerate(moves)}
    for category,queue in queues:
        categories[category]=[]
        for m in queue:
            if len(categories[category])>=4:break
            # Exclude known single-action rejections on this immutable graph.
            phase='compensation' if category=='repair' else 'agent_single'
            fingerprint=(scan['hash'],phase,((m.target,m.names,m.truth),))
            if fingerprint in worker.failed:continue
            cid='%s:%s:%d'%(bid,scan['hash'][:12],index[id(m)])
            selected[cid]=m;categories[category].append(cid)
    context=[]
    for cid,m in selected.items():
        context.append(dict(id=cid,target=m.target,inputs=list(m.names),family=m.family,
            expected_area_gain=worker.source.lib.area(m.gain),predicted_metric=short_metric(m.error,worker.metric),
            excess_fraction=max(0.,m.error/limit-1),positive_increment_fraction=m.weight/limit,
            feasible=m.error<=limit))
    return context,categories,selected


def validate_decision(d,observation):
    if not isinstance(d,dict):raise ValueError('Decision must be an object')
    action=d.get('action');options=d.get('options',{})
    if not isinstance(options,dict):raise ValueError('options must be an object')
    if action not in ('compress','compensate','near','near_repair','remap','fork','archive','finish'):
        raise ValueError('Unknown action')
    if action=='finish':return action,None,options
    branches={b['id']:b for b in observation['branches']}
    if d.get('branch') not in branches:raise ValueError('Unknown or archived branch')
    b=branches[d['branch']]
    if action=='compress':
        if b.get('compression_screen_exhausted'):raise ValueError('No screened feasible compression; repair, near_repair, remap, fork or switch branch')
        if options.get('policy','balanced') not in ('balanced','budget','area','error'):raise ValueError('Unknown policy')
        steps=options.get('steps',3);fraction=options.get('fraction',.25 if options.get('policy')=='budget' else 1.)
        if isinstance(steps,bool) or not isinstance(steps,int) or not 1<=steps<=6:raise ValueError('steps outside1..6')
        if isinstance(fraction,bool) or not isinstance(fraction,(int,float)) or not .1<=fraction<=1:raise ValueError('fraction outside.1..1')
        profile=rollout_profile(options.get('policy','balanced'),fraction)
        if list(profile) in b.get('blocked_profiles',[]):raise ValueError('That rollout is a measured plateau on this exact graph; choose an unblocked profile or another tool')
    if action in ('near','near_repair','compensate'):
        category='near' if action in ('near','near_repair') else 'repair'
        if options.get('candidate_id') not in b['categories'][category]:raise ValueError('Candidate outside this branch/category/scan')
    if action=='fork' and options.get('checkpoint_id') not in [x['id'] for x in b['checkpoints']]:
        raise ValueError('Checkpoint outside displayed branch history')
    if action=='near_repair':
        for name,default in (('steps',3),('repair_trials',3)):
            value=options.get(name,default)
            if isinstance(value,bool) or not isinstance(value,int) or not 0<=value<=3 or (name=='repair_trials' and value==0):
                raise ValueError('Compound tool limited to3repairs/3rollout steps')
    if action in ('fork','near','near_repair') and len(branches)>=4:raise ValueError('Four branch slots occupied; archive explicitly first')
    if action in ('fork','remap') and not isinstance(options.get('preserve',False),bool):raise ValueError('preserve must be boolean')
    return action,d.get('branch'),options


class Portfolio:
    def __init__(self,args):
        self.whole_started=time.perf_counter()
        self.args=args;self.folder=Path(args.output).resolve();self.folder.mkdir(parents=True,exist_ok=False)
        (self.folder/'source').mkdir();self.lib=Library(LIBRARY_PATH)
        problem=getattr(self,'problem_data',None)
        data_preflight=problem.preflight if problem else preflight
        data_matrix=problem.matrix if problem else matrix
        data_source_path=problem.source_path if problem else source_path
        self.original,self.order,self.audit=data_preflight(args.design,self.lib)
        self.source=load(args.accurate_netlist,self.lib) if args.accurate_netlist else self.original
        if self.source.inputs!=self.original.inputs or self.source.outputs!=self.original.outputs:raise ValueError('Accurate input interface differs')
        self.matrices=[data_matrix(args.design,self.source.inputs,i) for i in (0,1,2,100)]
        self.inputs=[pack(x,self.source.inputs) for x in self.matrices]
        self.refs=[number(outputs(self.original,trace(self.original,p,10000)),self.order,10000) for p in self.inputs]
        # Additional inputs must be accurate, not a hidden optimized warm start.
        for p,ref in zip(self.inputs,self.refs):
            actual=number(outputs(self.source,trace(self.source,p,10000)),self.order,10000)
            if not np.array_equal(actual,ref):raise ValueError('Accurate start differs from original reference')
        hashes={}
        for name in dict.fromkeys(SOURCES+EXTRA+list(getattr(problem,'source_files',()))):
            hashes[name]=sha(CODE/name);shutil.copy2(CODE/name,self.folder/'source'/name)
        knowledge_path=Path(getattr(args,'knowledge',None) or CODE/'numeric_agent_knowledge.json').resolve()
        self.knowledge=json.loads(knowledge_path.read_text(encoding='utf-8'))
        shutil.copy2(knowledge_path,self.folder/'knowledge_snapshot.json')
        self.source.write(self.folder/'accurate.v')
        self.metric=Metric(args.metric,args.bound,self.order)
        self.protocol=dict(vars(args),code_sha256=hashes,source_sha256=sha(args.accurate_netlist or data_source_path(args.design)),
            library_sha256=sha(LIBRARY_PATH),baseline_area=self.lib.area(self.source.units),
            original_dc_area=self.lib.area(self.original.units),count_per_group=10000,input_splits=[0,1,2,100],
            output_lsb_first=self.order,exhaustive_lacs=False,branch_slots=4,formal_integer_loss_bound=self.metric.limit,
            selection='minimum physical area among train and development eligible checkpoints; never filter by final results',
            accurate_start_check=('four fixed 10k groups versus original; independent NumPy agrees with packed original; no cross-structure formal proof claimed' if problem else
                'four fixed 10k packed groups; existing DC-vs-author preflight also independent NumPy; no cross-structure formal proof claimed'),
            alternate_representations='only current-run exact maps with completed source-to-mapped CEC; larger maps may be retained as exploration branches',
            reference=self.audit,knowledge_version=self.knowledge['version'],knowledge_sha256=sha(knowledge_path))
        save(self.folder/'protocol.json',self.protocol)
        self.workers={};self.active=[];self.origin={};self.decisions=[];self.calls=0;self.api_seconds=0.;self.http_seconds=0.;self.error=None
        self.started=time.perf_counter();self.next_branch=0;self.status='running';self.live_choices={};self.scan_seconds=0.
        first=self.add(self.source,dict(kind='accurate_start'),preserve=False)
        # Retain EXACT alternative even when backend would reject its larger
        # physical area. This is an exploration representation, not a winner.
        worker=self.workers[first];seen={structural_id(worker.current)}
        for index,row in enumerate(worker.backend.rows if worker.backend else []):
            if len(self.active)>=args.initial_views:break
            if not row.get('cec_equivalent'):continue
            path=worker.backend.folder/('map_%04d'%index)/'mapped.v'
            from numeric_exact import restore_ports
            c=restore_ports(path,worker.source)
            identity=structural_id(c)
            if identity in seen:continue
            seen.add(identity)
            self.add(c,dict(kind='cec_proven_exact_alternate',parent=first,
                source_graph=structural_id(self.source),proof=str(path.parent/'abc.log'),
                initially_larger_than_baseline=c.units>self.source.units),preserve=True)
        self.persist()

    def remaining(self):return max(0.,self.args.seconds-(time.perf_counter()-self.started))

    def add(self,circuit,origin,preserve):
        if len(self.active)>=4:raise ValueError('Branch slots full')
        bid='b%03d'%self.next_branch;self.next_branch+=1
        folder=self.folder/bid;folder.mkdir()
        options=argparse.Namespace(**vars(self.args))
        options.no_exact=False;options.logic_view=False;options.sop_view=False
        w=Worker(circuit,self.order,self.inputs,self.refs,options,folder,remaining=self.remaining,
                 preserve=preserve,reference_source=self.source)
        self.workers[bid]=w;self.active.append(bid);self.origin[bid]=origin
        return bid

    def persist(self):
        rows=[]
        for bid,w in self.workers.items():
            w.current.write(w.folder/'working.v')
            rows.append(dict(id=bid,active=bid in self.active,origin=self.origin[bid],preserve=w.preserve,
                physical_working_area=self.lib.area(w.current.units),training_loss=w.loss,steps=w.steps,
                stalls=w.stalls,working_sha256=sha(w.folder/'working.v'),last_result=w.last_result,
                stats=dict(w.stats),component_seconds=dict(w.times)))
        save(self.folder/'state.json',dict(status=self.status,elapsed=time.perf_counter()-self.started,
            branches=rows,decisions=self.decisions,api_calls=self.calls,api_seconds=self.api_seconds,http_request_seconds=self.http_seconds,
            provider_error=self.error,automatic_resume=False))

    def clone_branch(self,bid,origin,checkpoint=None,preserve=None):
        if len(self.active)>=4:raise ValueError('Branch slots full')
        parent=self.workers[bid];child='b%03d'%self.next_branch;self.next_branch+=1
        c=None;loss=None
        if checkpoint is not None:
            row=next(r for r in parent.history if r['path']==checkpoint)
            c=load(parent.folder/checkpoint,self.lib);loss=row['training_loss']
        w=clone_worker(parent,self.folder/child,self.remaining,self.started,c,loss,preserve)
        self.workers[child]=w;self.active.append(child);self.origin[child]=origin
        return child,w

    def observation(self):
        branches=[];self.live_choices={}
        for bid in self.active:
            if self.remaining()<.5:break
            w=self.workers[bid];tick=time.perf_counter()
            choices,categories,lookup=candidate_context(w,bid)
            self.scan_seconds+=time.perf_counter()-tick;self.live_choices[bid]=lookup
            checkpoints=[dict(id=bid+':'+r['path'],area=r['area'],metric=short_metric(r['training_loss'],self.metric),categories=r['categories'])
                         for r in checkpoint_menu(w.history)]
            exhausted=w.stalls>0 and not any(m.gain>0 and m.error<=self.metric.limit for m in w.scan()['moves'])
            branches.append(dict(id=bid,origin=self.origin[bid],working_area=self.lib.area(w.current.units),
                best_mapped_checkpoint_area=min(r['area'] for r in w.history),
                mapped_area_headroom=self.lib.area(w.current.units-min(r['units'] for r in w.history)),
                training_metric=short_metric(w.loss,self.metric),budget_fraction=w.loss/self.metric.limit,
                steps=w.steps,stalls=w.stalls,preserve_view=w.preserve,graph_hash=structural_id(w.current),
                choices=choices,categories=categories,screened_candidates=len(w.scan()['moves']),
                checkpoints=checkpoints,compression_screen_exhausted=exhausted,
                last_result={k:v for k,v in w.last_result.items() if k!='detail'}))
            blocked=[list(p[1:]) for p in sorted(w.blocked) if p[0]==structural_id(w.current)]
            branches[-1].update(blocked_profiles=blocked,
                unprobed=w.steps==0,
                suggested_unblocked_rollouts=[dict(policy=policy,fraction=fraction) for policy,fraction in
                    [('balanced',1.),('balanced',.5),('budget',.25),('area',1.),('error',1.)]
                    if list(rollout_profile(policy,fraction)) not in blocked],
                interpretation='working area ranks current graph, mapped checkpoint area ranks its measured function; positive mapped_area_headroom can justify explicit adopt/remap after stagnation')
        obs=dict(session=str(self.folder),decision_id=len(self.decisions),design=self.args.design,
            metric=self.args.metric,formal_bound=self.args.bound,near_ratio=self.args.near_ratio,
            compensation_max_area_debt_units=16,baseline_area=self.lib.area(self.source.units),
            remaining_seconds=self.remaining(),free_branch_slots=4-len(self.active),branches=branches,
            recent_decisions=[{k:v for k,v in d.items() if k in ('action','branch','options','reason','result')} for d in self.decisions[-5:]],
            knowledge=self.knowledge,legal_actions=['compress','compensate','near','near_repair','remap','fork','archive','finish'])
        save(self.folder/'observation.json',obs)
        return obs

    def execute(self,d,obs):
        action,bid,o=validate_decision(d,obs)
        if action=='finish':self.status='agent_finished' if self.args.controller=='agent' else 'heuristic_finished';return dict(stopped=True)
        w=self.workers[bid]
        if action=='archive':self.active.remove(bid);return dict(archived=bid)
        if action=='compress':
            result=[]
            for _ in range(o.get('steps',3)):
                if self.remaining()<.5:break
                r=w.compress(o.get('policy','balanced'),o.get('fraction',.25 if o.get('policy')=='budget' else 1.))
                result.append({k:v for k,v in r.items() if k!='detail'})
                if not r['changed']:break
            return dict(local_steps=result,area=self.lib.area(w.current.units),metric=short_metric(w.loss,self.metric))
        if action=='compensate':
            m=self.live_choices[bid][o['candidate_id']]
            ok,detail=w.apply([m],'compensation')
            if ok:w.repairs+=1;w.refine('agent_compensation_exact')
            w.last_result=dict(action='compensate',accepted=ok,reason=detail.get('reason'),
                area=self.lib.area(w.current.units),metric=short_metric(w.loss,self.metric))
            return w.last_result
        if action=='near_repair':
            m=self.live_choices[bid][o['candidate_id']]
            child,worker=self.clone_branch(bid,dict(kind='agent_compound_near_repair',parent=bid,candidate_id=o['candidate_id']))
            detail=near_repair_rollout(worker,m,o.get('steps',3),o.get('repair_trials',3),self.remaining)
            return dict(branch=child,**detail,working_area=self.lib.area(worker.current.units),metric=short_metric(worker.loss,self.metric))
        if action=='near':
            m=self.live_choices[bid][o['candidate_id']]
            c=cleanup(w.current.materialize([m.replacement()]),w.mapper);w.stats['area_materializations']+=1
            y=number(outputs(c,trace(c,self.inputs[0],10000,w.stats)),self.order,10000)
            loss=self.metric.exact(y,self.refs[0])[0];w.stats['near_miss_joint_verifications']+=1
            if hashlib.blake2b(y.tobytes(),digest_size=16).hexdigest()!=m.signature:
                raise AssertionError('Near-miss shared response differs from actual child')
            if c.units>=w.current.units or loss>int(self.metric.limit*self.args.near_ratio):
                return dict(accepted=False,reason='actual_near_miss_economics_or_window',actual_loss=loss)
            child=self.add(c,dict(kind='agent_near_miss',parent=bid,candidate_id=o['candidate_id'],
                measured_loss=loss,excess_fraction=max(0.,loss/self.metric.limit-1),
                measured_area_gain=self.lib.area(w.current.units-c.units)),preserve=w.preserve)
            return dict(branch=child,metric=short_metric(loss,self.metric),feasible=loss<=self.metric.limit)
        if action=='fork':
            name=o['checkpoint_id'].split(':',1)[1]
            child,_=self.clone_branch(bid,dict(kind='agent_checkpoint_fork',parent=bid,checkpoint=name),name,o.get('preserve',False))
            return dict(branch=child)
        if action=='remap':
            w.preserve=o.get('preserve',False);start=len(w.history);before=structural_id(w.current)
            w.refine('agent_explicit_remap')
            changed=structural_id(w.current)!=before
            w.last_result=dict(action='remap',checkpoints_added=len(w.history)-start,topology_changed=changed,preserve=w.preserve)
            if changed:w.stalls=0
            return w.last_result

    def heuristic(self,obs):
        # Explicit numerical control arm, not an Agent fallback.
        branches=obs['branches'];ready=[b for b in branches if b['stalls']<2]
        if not ready:return dict(action='finish',reason='All numerical branches reached screened plateaus')
        b=min(ready,key=lambda b:(b['steps'],b['best_mapped_checkpoint_area'],b['id']))
        if b['training_metric']>self.args.bound and b['categories']['repair']:
            return dict(action='compensate',branch=b['id'],options=dict(candidate_id=b['categories']['repair'][0]),reason='Measured feasible repair')
        return dict(action='compress',branch=b['id'],options=dict(policy='balanced',steps=4),reason='Explicit numerical balanced exploration')

    def run(self):
        while self.status=='running' and self.active and self.remaining()>.5 and len(self.decisions)<self.args.decisions:
            obs=self.observation()
            if self.remaining()<=.5 or not obs['branches']:break
            if self.args.controller=='agent' and self.remaining()<=decision_reserve(
                    [r['receipt'] for r in self.decisions if r.get('receipt')]):
                self.status='time_limit_reserve';break
            receipt=None;d=dict(action='invalid',reason='No valid decision returned')
            try:
                if self.args.controller=='agent':
                    from numeric_agent_provider import choose
                    self.calls+=1;tick=time.perf_counter()
                    try:
                        d,receipt=choose(obs,self.remaining())
                        self.http_seconds+=receipt.get('http_request_seconds',0.)
                    except RuntimeError as exc:
                        self.http_seconds+=getattr(exc,'request_seconds',0.);raise
                    finally:self.api_seconds+=time.perf_counter()-tick
                else:d=self.heuristic(obs)
                if self.remaining()<=.1:
                    result=dict(skipped=True,reason='budget_expired_after_decision');self.status='time_limit'
                else:result=self.execute(d,obs)
            except ValueError as exc:
                # Corrective evidence is supplied next time; no unchecked action.
                result=dict(rejected=True,reason=str(exc))
            except RuntimeError as exc:
                self.error=str(exc);self.status='time_limit_with_unfinished_request' if self.remaining()<=1 else 'controller_failed';self.persist();break
            record=dict(d,decision_id=len(self.decisions),result=result,receipt=receipt)
            self.decisions.append(record);self.persist()
            print('[NUMERIC AGENT] '+json.dumps(record,ensure_ascii=False),flush=True)
        if self.status=='running':
            self.status='time_limit' if self.remaining()<=.5 else 'decision_limit' if len(self.decisions)>=self.args.decisions else 'no_active_branches'
        self.persist();return self.finish()

    def finish(self):
        # No final/confirmation values are computed before selecting the winner.
        all_rows=[];development_calls=0
        for bid,w in self.workers.items():
            for row in w.history:
                c=load(w.folder/row['path'],self.lib)
                train=row['training_loss']
                if train>self.metric.limit:
                    all_rows.append(dict(row,branch=bid,eligible=False,reason='training_constraint'));continue
                y=number(outputs(c,trace(c,self.inputs[1],10000)),self.order,10000);development_calls+=1
                total,value=self.metric.exact(y,self.refs[1])
                all_rows.append(dict(row,branch=bid,development_loss=total,development_metric=value,eligible=total<=self.metric.limit))
        selected=select_checkpoint(all_rows)
        winner=load(self.workers[selected['branch']].folder/selected['path'],self.lib);checks=[]
        for i,label in ((2,'final'),(3,'confirmation')):
            actual=number(outputs(winner,trace(winner,self.inputs[i],10000)),self.order,10000)
            np_y=validate_numpy(winner,self.matrices[i])
            alternate=number_numpy(np_y,winner.outputs,self.order)
            if not np.array_equal(alternate,actual):raise AssertionError('Independent winner simulation disagrees')
            np_ref=validate_numpy(self.original,self.matrices[i])
            alternate_ref=number_numpy(np_ref,self.original.outputs,self.order)
            if not np.array_equal(alternate_ref,self.refs[i]):raise AssertionError('Independent reference simulation disagrees')
            total,value=self.metric.exact(actual,self.refs[i]);checks.append(dict(group=label,count=10000,
                split=(2 if i==2 else 100),loss=total,metric=value,passed=total<=self.metric.limit))
        passed=all(r['passed'] for r in checks);dest=self.folder/('best.v' if passed else 'failed_final.v');winner.write(dest)
        counters=Counter();times=Counter()
        for w in self.workers.values():counters.update(w.stats);times.update(w.times)
        ledger_path=CODE/'hybrid_billing'/self.args.design/'ledger.json'
        ledger=json.loads(ledger_path.read_text(encoding='utf-8')) if ledger_path.exists() else {}
        requests=[r for r in ledger.get('requests',[]) if r['session']==str(self.folder)]
        result=dict(design=self.args.design,metric=self.args.metric,bound=self.args.bound,
            controller=self.args.controller,status=self.status,provider_error=self.error,api_attempts=self.calls,
            api_calls=len(requests),api_responses=sum(r['status']=='response_received' for r in requests),
            api_seconds=self.api_seconds,api_cost_upper_cny=sum(r.get('estimated_upper_cny',r['reserved_cny']) for r in requests),
            original_area=self.lib.area(self.source.units),area=self.lib.area(winner.units),
            reduction=1-winner.units/self.source.units,selected_checkpoint=selected,checks=checks,strict_pass=passed,
            all_checkpoints=all_rows,stats=dict(counters),component_seconds=dict(times),
            search_seconds=max((r.get('elapsed_seconds',0) for r in self.decisions),default=0),
            search_and_agent_seconds=json.loads((self.folder/'state.json').read_text(encoding='utf-8'))['elapsed'],
            whole_seconds=time.perf_counter()-self.whole_started,preflight_seconds=self.started-self.whole_started,
            development_full_propagations=development_calls,
            final_full_propagations=2,reference_full_propagations=8,independent_numpy_checks=4,
            branches_created=len(self.workers),decisions=len(self.decisions),netlist_sha256=sha(dest),
            no_author_approximate_search_called=True,no_silent_numeric_fallback=True,
            timing_note='candidate_generation includes root_response_subset; observation scan time overlaps it; whole includes Agent and final validation')
        # Search time includes initialization and all numerical/global decisions,
        # but separates provider latency from wall time. Do not sum overlapping scans.
        result['search_seconds']=max(0.,result['search_and_agent_seconds']-self.api_seconds)
        result.update(timing_fields(time.perf_counter()-self.whole_started,result['search_and_agent_seconds'],
            self.api_seconds,self.started-self.whole_started,self.http_seconds))
        result['timing_note']='whole includes preflight, search, model, selection and checks; excluding_llm subtracts entire synchronous call overhead including failed waits; excluding_llm_http subtracts HTTP wait only; numerical_search excludes final/preflight; root_response_subset overlaps candidate_generation'
        save(self.folder/'report.json',result)
        # Store only universal measured facts as a proposal, never hard-code a
        # design's nodes/area/trajectory into global strategy automatically.
        experience=dict(schema=1,status='measured_generic_proposal',source=str(self.folder/'report.json'),
            rules=[dict(topic='representation_diversity',branches=len(self.workers),
                alternate_selected=self.origin[selected['branch']]['kind']!='accurate_start',
                observation='Physical initial area is not sufficient to rank a search representation.'),
                dict(topic='sampled_generalization',strict_pass=passed,
                     observation='Training and development feasibility do not guarantee two held-out groups pass.')],
            contains_benchmark_specific_actions=False,automatically_promoted=False)
        save(self.folder/'experience_proposal.json',experience)
        print('[NUMERIC AGENT RESULT] '+json.dumps({k:v for k,v in result.items() if k not in ('all_checkpoints','component_seconds')},ensure_ascii=False),flush=True)
        return 3 if self.error and self.status=='controller_failed' else 0 if passed else 2


def parser():
    p=local_parser();p.description=__doc__
    p.add_argument('--controller',choices=('agent','heuristic'),default='agent')
    p.add_argument('--accurate-netlist',help='optional shared accurate baseline; never an approximate warm start')
    p.add_argument('--initial-views',type=int,choices=(1,2,3),default=2)
    p.add_argument('--decisions',type=int,default=24)
    p.add_argument('--knowledge',help='explicit general strategy snapshot for controlled experience ablations')
    p.set_defaults(seconds=120.,map_variants=2,compensate=True)
    return p


if __name__=='__main__':
    p=parser();a=p.parse_args()
    if min(a.seconds,a.bound,a.dp_seconds,a.map_seconds,a.decisions,a.pool,a.pairs,a.per_root)<=0:p.error('Positive budgets required')
    if a.logic_view or a.sop_view or a.no_exact:p.error('This entry manages its own representation portfolio; do not set legacy view/no-exact flags')
    if a.training_fraction!=1 or a.development_fraction!=1:p.error('Numerical Agent protocol uses the declared full bound')
    if not 1<=a.near_ratio<=1.5:p.error('near ratio must be1..1.5')
    if a.controller=='agent':
        from hybrid_secrets import load_key
        load_key()  # fail before creating a session or running searches
    raise SystemExit(Portfolio(a).run())
