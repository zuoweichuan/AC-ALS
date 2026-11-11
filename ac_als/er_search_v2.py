"""Strict ER research v2: partial/compound repair and measured global feedback."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import er_data_v2
from er_search import parser as old_parser
from er_repair_tools import ERWorker, JointRepair,development_limit
from bacs_numeric import number, number_numpy, save
from bacs_data import load
from numeric_agentic import Portfolio, short_metric, select_checkpoint, timing_fields, rollout_profile
from numeric_menu_agent import action_key, bind_choice
from numeric_explore import MCTSExplorer, policy_menu
from numeric_exact import ExactBackend
from hybrid_identity import structural_id
from simulate import trace, outputs, validate_numpy,pack

ROOT=Path(__file__).resolve().parent


def repair_rank(m,loss):
    return ((max(0,-m.gain)+1)/max(1,loss-m.error),m.error,-getattr(m,'hamming_gain',0))


def repair_options(worker):
    return [m for m in worker.scan()['moves'] if m.error<worker.loss or
            (m.error==worker.loss and getattr(m,'hamming_gain',0)>0)]


class ERFeedback:
    problem_data=er_data_v2
    def __init__(self,args):
        self.dev_cache={};self.dev_calls=0
        super().__init__(args)
        self.protocol.update(er_tools_version=2,development_fraction=args.development_fraction,
            development_confidence=args.development_confidence,
            development_feedback='cached training+development eligible mapped area available to BOTH controllers during search',
            development_limit=development_limit(self.metric,args),
            additional_untouched_confirmation_splits=[2000,2001],
            partial_bit_repair='auxiliary Hamming ranking; strict vector ER for winner acceptance',
            bit_repair_roots=args.bit_repair_roots,verified_pair_cap=args.repair_pairs,
            repair_debt_units=args.repair_debt_units,max_refreshes_per_graph=args.refresh_scans,
            no_silent_numeric_fallback=True)
        save(self.folder/'protocol.json',self.protocol)

    def add(self,circuit,origin,preserve):
        if len(self.active)>=4:raise ValueError('Branch slots full')
        bid='b%03d'%self.next_branch;self.next_branch+=1
        folder=self.folder/bid;folder.mkdir()
        options=copy.copy(self.args);options.no_exact=False;options.logic_view=False;options.sop_view=False
        worker=ERWorker(circuit,self.order,self.inputs,self.refs,options,folder,remaining=self.remaining,
                        preserve=preserve,reference_source=self.source)
        self.workers[bid]=worker;self.active.append(bid);self.origin[bid]=origin
        return bid

    def clone_branch(self,*args,**kwargs):
        bid,w=super().clone_branch(*args,**kwargs)
        w.refresh_counts=dict(w.refresh_counts);w.bit_cache=None;w.verified_pairs={};w.development_view=None;w.training_view=None
        return bid,w

    def development_rows(self,bid):
        worker=self.workers[bid];rows=[]
        for r in worker.history:
            row=dict(r,branch=bid,eligible=False)
            if r['training_loss']>self.metric.limit:row['reason']='training_constraint'
            else:
                path=worker.folder/r['path'];key=hashlib.sha256(path.read_bytes()).hexdigest()
                if key not in self.dev_cache:
                    c=load(path,self.lib)
                    y=number(outputs(c,trace(c,self.inputs[1],10000)),self.order,10000)
                    self.dev_cache[key]=self.metric.exact(y,self.refs[1]);self.dev_calls+=1
                total,value=self.dev_cache[key]
                row.update(development_loss=total,development_metric=value,
                    eligible=total<=development_limit(self.metric,self.args))
            rows.append(row)
        return rows

    def finish(self):
        all_rows=[row for bid in self.workers for row in self.development_rows(bid)]
        selected=select_checkpoint(all_rows)
        winner=load(self.workers[selected['branch']].folder/selected['path'],self.lib)
        checks=[]
        for i,label in ((2,'final'),(3,'confirmation')):
            actual=number(outputs(winner,trace(winner,self.inputs[i],10000)),self.order,10000)
            alternate=number_numpy(validate_numpy(winner,self.matrices[i]),winner.outputs,self.order)
            reference=number_numpy(validate_numpy(self.original,self.matrices[i]),self.original.outputs,self.order)
            if not np.array_equal(actual,alternate) or not np.array_equal(reference,self.refs[i]):
                raise AssertionError('Independent ER simulators disagree')
            total,value=self.metric.exact(actual,self.refs[i])
            checks.append(dict(group=label,split=2 if i==2 else 100,count=10000,loss=total,metric=value,passed=total<=self.metric.limit))
        # These fresh confirmations are predeclared in the v2 protocol. They
        # are never available to observations, construction or winner selection.
        for split in (2000,2001):
            matrix=er_data_v2.matrix(self.args.design,self.source.inputs,split);inputs=pack(matrix,self.source.inputs)
            actual=number(outputs(winner,trace(winner,inputs,10000)),self.order,10000)
            reference=number(outputs(self.original,trace(self.original,inputs,10000)),self.order,10000)
            if not np.array_equal(actual,number_numpy(validate_numpy(winner,matrix),winner.outputs,self.order)):
                raise AssertionError('Fresh confirmation simulator mismatch')
            total,value=self.metric.exact(actual,reference)
            checks.append(dict(group='fresh_confirmation',split=split,count=10000,loss=total,metric=value,passed=total<=self.metric.limit))
        passed=all(r['passed'] for r in checks);dest=self.folder/('best.v' if passed else 'failed_final.v');winner.write(dest)
        counters=Counter();times=Counter()
        for w in self.workers.values():counters.update(w.stats);times.update(w.times)
        path=getattr(self,'billing_root',ROOT)/'hybrid_billing'/self.args.design/'ledger.json'
        ledger=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        requests=[r for r in ledger.get('requests',[]) if r['session']==str(self.folder)]
        state=json.loads((self.folder/'state.json').read_text(encoding='utf-8'))
        result=dict(design=self.args.design,metric='er',bound=self.args.bound,controller=self.args.controller,
            status=self.status,provider_error=self.error,api_attempts=self.calls,api_calls=len(requests),
            api_responses=sum(r['status']=='response_received' for r in requests),
            api_cost_upper_cny=sum(r.get('estimated_upper_cny',r['reserved_cny']) for r in requests),
            original_area=self.lib.area(self.source.units),area=self.lib.area(winner.units),
            reduction=1-winner.units/self.source.units,strict_pass=passed,checks=checks,
            selected_checkpoint=selected,all_checkpoints=all_rows,stats=dict(counters),component_seconds=dict(times),
            development_full_propagations=self.dev_calls,development_cache_entries=len(self.dev_cache),
            development_feedback_is_search_data=True,final_full_propagations=4,reference_full_propagations=10,
            independent_numpy_checks=6,branches_created=len(self.workers),decisions=len(self.decisions),
            netlist_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),er_tools_version=2,
            no_final_based_reselection=True,no_author_approximate_search_called=True,no_silent_numeric_fallback=True,
            timing_note='ER bit repair subset is separate from ordinary generation; root/full/function counters are distinct; reference/final/development calls are separate')
        result.update(timing_fields(time.perf_counter()-self.whole_started,state['elapsed'],self.api_seconds,
            self.started-self.whole_started,self.http_seconds))
        save(self.folder/'report.json',result)
        save(self.folder/'experience_proposal.json',dict(status='proposal_only',automatically_promoted=False,
            source=str(self.folder/'report.json'),contains_benchmark_specific_actions=False,
            facts=dict(strict_pass=passed,partial_truths=counters.get('er_bit_truths_scored',0),
                       joint_verifications=counters.get('er_complementary_joint_verifications',0))))
        print('[ER V2 RESULT] '+json.dumps({k:v for k,v in result.items() if k not in ('all_checkpoints','component_seconds')},ensure_ascii=False),flush=True)
        return 3 if self.error else 0 if passed else 2


class ERAgent(ERFeedback,Portfolio):
    def __init__(self,args):
        self.used=set();self.bindings={};self.canonical=None
        super().__init__(args)

    def observation(self):
        full=Portfolio.observation(self);self.canonical=full;bindings={};menu=[];branches=[];deferred=[]
        eligible=[]
        for bid in self.workers:
            eligible += [r for r in self.development_rows(bid) if r['eligible']]
        incumbent=min(eligible,key=lambda r:r['units']) if eligible else None
        incumbent_since=min((r['seconds'] for r in eligible if incumbent and r['units']==incumbent['units']),default=0.)
        def add(d,label,graph='global',**evidence):
            worker=self.workers.get(d.get('branch'))
            if worker is not None:graph=graph+':refresh'+str(worker.refresh_counts.get(structural_id(worker.current),0))
            if action_key(d,graph) in self.used:return
            handle='a%02d'%len(menu);bindings[handle]=d
            menu.append(dict(id=handle,kind=d['action'],branch=d.get('branch'),description=label,**evidence))
        for b in full['branches']:
            bid=b['id'];w=self.workers[bid];graph=b['graph_hash'];scan=w.scan();moves=scan['moves']
            valid=[r for r in self.development_rows(bid) if r['eligible']]
            profiles=[('balanced',1.),('balanced',.5),('budget',.25),('area',1.),('error',1.)]
            if any(m.gain>0 and m.error<=self.metric.limit for m in moves):
                for policy,fraction in profiles:
                    if (graph,*rollout_profile(policy,fraction)) in w.blocked:continue
                    add(dict(action='compress',branch=bid,options=dict(policy=policy,fraction=fraction,steps=3)),
                        'Screened compression; fraction controls new error spending, formal constraint is unchanged',graph,
                        policy=policy,fraction=fraction)
            repairs=sorted(repair_options(w),key=lambda m:repair_rank(m,w.loss))
            for index,m in enumerate(repairs[:3]):
                if w.current.units-m.gain>min(w.source.units,w.lowest_units+w.args.repair_debt_units):continue
                cid='er:%s:%d'%(graph[:12],index)
                self.live_choices[bid][cid]=m
                add(dict(action='er_repair',branch=bid,options=dict(candidate_id=cid)),
                    'Measured vector or partial-bit repair; intermediate still above bound cannot win',graph,
                    wrong_vectors=m.error,hamming_recovered=getattr(m,'hamming_gain',0),
                    compound=isinstance(m,JointRepair),estimated_area_cost=w.source.lib.area(-m.gain),
                    development_wrong_vectors=w.development_score(m)['loss'])
            for cid in b['categories']['near'][:2]:
                # A failed infeasible child needs new repair evidence before
                # recursively spending another branch on the same error level.
                # The model may instead refresh its signatures, select repair,
                # revisit a feasible prefix, archive, or stop.
                if w.loss>self.metric.limit and not repairs:continue
                if any(o.get('parent')==bid and o.get('candidate_id')==cid for o in self.origin.values()):continue
                m=self.live_choices[bid][cid]
                if len(self.active)<4:
                    add(dict(action='er_near_repair',branch=bid,options=dict(candidate_id=cid)),
                        'Preserve parent; fork near miss, up to3 vector/partial/compound repairs, then compression',graph,
                        expected_area_gain=w.source.lib.area(m.gain),wrong_vectors=m.error)
                else:deferred.append(dict(branch=bid,kind='near_repair',wrong_vectors=m.error,requires='archive one branch'))
            if b['preserve_view'] and b['mapped_area_headroom']>0:
                add(dict(action='remap',branch=bid,options=dict(preserve=False)),
                    'Adopt exact mapped representation; preserve checkpoints',graph,mapped_area_headroom=b['mapped_area_headroom'])
            for c in b['checkpoints']:
                if 'early_structural_milestone' not in c['categories']:continue
                name=c['id'].split(':',1)[1]
                if structural_id(load(w.folder/name,self.lib))==graph:continue
                if any(o.get('parent')==bid and o.get('checkpoint')==name for o in self.origin.values()):continue
                if len(self.active)<4:
                    add(dict(action='fork',branch=bid,options=dict(checkpoint_id=c['id'],preserve=False)),
                        'Fork earlier topology for a different policy or targeted repair',graph,
                        checkpoint_area=c['area'],wrong_vectors=round(c['metric']*10000))
            refresh=w.refresh_counts.get(graph,0)
            if w.stalls and refresh<self.args.refresh_scans:
                add(dict(action='er_refresh',branch=bid,options={}),
                    'Resample only64 generation patterns; rerank bounded candidates; NOT another10000-vector stream',graph,
                    refresh_used=refresh,refresh_limit=self.args.refresh_scans)
            if len(self.active)>1:
                add(dict(action='archive',branch=bid,options={}),
                    'Free a slot; stored checkpoints remain eligible but this branch stops exploration',graph)
            branches.append(dict(id=bid,working_area=b['working_area'],training_wrong_vectors=w.loss,
                training_feasible=w.loss<=self.metric.limit,
                training_verdict='FEASIBLE' if w.loss<=self.metric.limit else 'INFEASIBLE',
                training_headroom_vectors=self.metric.limit-w.loss,
                current_development_wrong_vectors=w.development_context()['bad'].bit_count(),
                best_development_eligible_area=min((r['area'] for r in valid),default=None),
                best_training_only_area=b['best_mapped_checkpoint_area'],
                development_limit=development_limit(self.metric,self.args),
                stalls=w.stalls,preserve_view=w.preserve,screened_candidates=len(moves),
                blocked_profiles=b['blocked_profiles'],signature_refreshes=refresh))
        add(dict(action='finish',options={}),
            'Stop by global value/cost judgement; local filtered emptiness is not global optimality')
        self.bindings=bindings
        obs=dict(session=full['session'],decision_id=len(self.decisions),design=self.args.design,
            metric='strict_vector_er',formal_bound=self.args.bound,formal_wrong_vector_limit=self.metric.limit,
            remaining_seconds=self.remaining(),free_branch_slots=4-len(self.active),
            global_development_incumbent_area=incumbent['area'] if incumbent else None,
            seconds_without_development_frontier_gain=max(0.,time.perf_counter()-self.started-incumbent_since),
            priority_evidence='Compress accurate or margin-bearing branches before speculative near repair if no smaller development-eligible incumbent exists. Ordinary compression now screens both training and development; untried accurate branches are valuable.',
            decision_menu=menu,branches=branches,deferred_frontier=deferred,knowledge=self.knowledge,
            recent_results=[dict(action=d.get('action'),branch=d.get('branch'),result=d.get('result')) for d in self.decisions[-3:]],
            interpretation='Use integer wrong vectors. Training-only area is not a validated incumbent. Partial-bit recovery is useful but never changes formal ER. Untried profiles and bounded refreshes remain exploration, not evidence of guaranteed gains.')
        save(self.folder/'canonical_observation.json',full);save(self.folder/'observation.json',obs)
        folder=self.folder/'menu_bindings';folder.mkdir(exist_ok=True)
        save(folder/('%04d.json'%len(self.decisions)),dict(bindings=bindings,observation=obs))
        return obs

    def execute(self,d,obs):
        canonical=bind_choice(d,self.bindings);bid=canonical.get('branch');w=self.workers.get(bid)
        graph=(structural_id(w.current)+':refresh'+str(w.refresh_counts.get(structural_id(w.current),0))) if w else 'global'
        self.used.add(action_key(canonical,graph));d.update(canonical)
        action=canonical['action']
        if action=='er_refresh':return w.refresh()
        if action=='er_repair':
            m=self.live_choices[bid][canonical['options']['candidate_id']]
            ok,detail=w.apply([m],'compensation')
            if ok:w.refine('er_bit_repaired_exact')
            w.last_result=dict(accepted=ok,**detail);return w.last_result
        if action=='er_near_repair':
            from bacs_numeric import cleanup
            m=self.live_choices[bid][canonical['options']['candidate_id']]
            child,worker=self.clone_branch(bid,dict(kind='er_near_repair',parent=bid,candidate_id=canonical['options']['candidate_id']))
            c=cleanup(worker.current.materialize([m.replacement()]),worker.mapper)
            actual=number(outputs(c,trace(c,self.inputs[0],10000,worker.stats)),self.order,10000)
            loss,_=self.metric.exact(actual,self.refs[0]);worker.stats['near_miss_joint_verifications']+=1
            if c.units>=worker.current.units or loss>int(self.metric.limit*self.args.near_ratio):return dict(branch=child,accepted=False,reason='actual_near_window_or_area')
            worker.current=c;worker.loss=loss;worker.lowest_units=min(worker.lowest_units,c.units)
            worker.checkpoint('er_near_infeasible',[m]);worker.annotate(len(worker.history)-1)
            outcomes=[]
            for _ in range(3):
                if self.remaining()<.5:break
                repairs=sorted(repair_options(worker),key=lambda r:repair_rank(r,worker.loss))[:3]
                accepted=False
                for repair in repairs:
                    accepted,detail=worker.apply([repair],'compensation');outcomes.append(dict(accepted=accepted,**detail))
                    if accepted:break
                if not accepted:break
                if worker.loss<=self.metric.limit:
                    worker.refine('er_near_repaired_exact')
                    for _ in range(3):
                        if self.remaining()<.5:break
                        if (structural_id(worker.current),'batch',1.) in worker.blocked:break
                        r=worker.compress('balanced',1.)
                        if not r['changed']:break
                    break
            if worker.loss>self.metric.limit:worker.stalls+=1
            return dict(branch=child,repaired=worker.loss<=self.metric.limit,wrong_vectors=worker.loss,repairs=outcomes)
        if action=='archive':
            w.cached=None;w.bit_cache=None;w.verified_pairs={};w.development_view=None;w.training_view=None
        # Binding was generated from canonical branch/checkpoint data. Worker
        # validates the actual graph and formal constraint before accepting.
        if action=='compress':
            out=[];o=canonical['options']
            for _ in range(o['steps']):
                if self.remaining()<.5:break
                r=w.compress(o['policy'],o['fraction']);out.append({k:v for k,v in r.items() if k!='detail'})
                if not r['changed']:break
            return dict(local_steps=out,area=self.lib.area(w.current.units),wrong_vectors=w.loss)
        return Portfolio.execute(self,canonical,self.canonical)


class ERMCTS(ERFeedback,MCTSExplorer):
    def assess(self,bid):
        rows=[r for r in self.development_rows(bid) if r['eligible']]
        area=min((r['area'] for r in rows),default=self.lib.area(self.source.units))
        value=1-area/self.lib.area(self.source.units)
        self.tree[bid].update(best_eligible_area=area,reward=value,
            training_metric=short_metric(self.workers[bid].loss,self.metric))
        self.development_calls=self.dev_calls
        return value

    def menu(self,bid):
        if bid not in self.catalog:
            w=self.touch(bid);base=policy_menu(w)
            repairs=sorted(repair_options(w),key=lambda r:repair_rank(r,w.loss))
            if repairs:
                best=next((m for m in repairs if isinstance(m,JointRepair)),repairs[0])
                base=[dict(kind='repair',move=best),*base]
            # Four children retained; one bounded signature refresh only when
            # stalled and ordinary menus have no remaining productive edge.
            if not any(a['kind'] in ('compress','repair','near_repair') for a in base) and w.refresh_counts.get(structural_id(w.current),0)<self.args.refresh_scans:
                w.refresh();base=policy_menu(w)
            self.catalog[bid]=base[:4];w.cached=None
        return self.catalog[bid]

    def child(self,parent):
        bid,w=super().child(parent);w.refresh_counts=dict(w.refresh_counts);w.bit_cache=None;w.verified_pairs={};w.development_view=None;w.training_view=None
        return bid,w

    def touch(self,bid):
        w=super().touch(bid)
        for other in self.workers.values():
            if other.current is None:other.bit_cache=None;other.verified_pairs={};other.development_view=None;other.training_view=None
        return w


def parser():
    p=old_parser();p.description=__doc__
    p.add_argument('--bit-repair-roots',type=int,default=16)
    p.add_argument('--repair-pairs',type=int,default=6)
    p.add_argument('--repair-debt-units',type=int,default=32)
    p.add_argument('--refresh-scans',type=int,default=2)
    p.add_argument('--development-confidence',type=float,default=.95,
        help='one-sided Wilson development guard;0 disables; formal final ER unchanged')
    p.set_defaults(knowledge=str(ROOT/'er_agent_knowledge_v2.json'),development_fraction=.9)
    return p


def run(a):
    if a.metric!='er' or a.logic_view or a.sop_view or a.no_exact:raise ValueError('Use proven exact standard-cell views')
    if not 0<a.development_fraction<=1 or a.training_fraction!=1:raise ValueError('Invalid declared development margin')
    if a.development_confidence and not .5<a.development_confidence<1:raise ValueError('Invalid development confidence')
    if min(a.seconds,a.decisions,a.expansions,a.bit_repair_roots,a.repair_pairs,a.repair_debt_units)<1 or not 0<=a.refresh_scans<=3:
        raise ValueError('Invalid bounded ER tools')
    if a.bit_repair_roots>64 or a.repair_pairs>12:raise ValueError('Repair screening caps exceeded')
    if a.controller=='agent':
        from hybrid_secrets import load_key
        load_key()
    a.no_word=True
    return (ERAgent if a.controller=='agent' else ERMCTS)(a).run()


if __name__=='__main__':raise SystemExit(run(parser().parse_args()))
