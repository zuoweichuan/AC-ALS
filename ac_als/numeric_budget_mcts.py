"""NMED four-child UCT with branch-local operating budgets and measured repair.

This ports operating-budget diversity to the numerical response tools. ER-only
Hamming/care operators are not substituted for numerical error calculations.
"""
import copy
from decimal import Decimal
import json
import math
from pathlib import Path
import shutil
import time

from bacs_numeric import Metric, save
from hybrid_identity import structural_id
from numeric_agentic import parser as base_parser, near_repair_rollout, sha
from numeric_explore import MCTSExplorer, policy_menu, action_record,uct_score


def freeze_physical_checkpoint(path,lib,folder,selected):
    """Hash exactly the loaded/written circuit used by the final evaluator.

    Loading may drop unused wire declarations. Raw file bytes are retained as
    provenance, but must not be confused with the final serialization hash.
    """
    from bacs_data import load
    normalized=Path(folder)/'selected_before_checks.v'
    load(path,lib).write(normalized)
    frozen=sha(normalized)
    save(Path(folder)/'frozen_selection.json',dict(checkpoint=selected,sha256=frozen,
         checkpoint_file_sha256=sha(path),serialization='same load/write as final evaluator',
         selection='formal NMED training/development only, before final checks'))
    return frozen


def operating_limit(formal_limit, fraction):
    return max(1,int(Decimal(formal_limit)*Decimal(str(fraction))))


def set_operating_budget(worker, formal_metric, limit):
    """Keep the exact integer numerator; do not round through a float NMED."""
    worker.args=copy.copy(worker.args)
    worker.metric=Metric(formal_metric.kind,limit/formal_metric.count/formal_metric.denominator,
                         formal_metric.order,formal_metric.count)
    worker.metric.limit=limit;worker.metric.cap=limit+1
    worker.args.bound=worker.metric.bound
    worker.formal_limit=formal_metric.limit
    worker.cached=None
    if hasattr(worker,'area_model'):del worker.area_model


def budget_edge(moves,loss,limit,fraction,near_ratio):
    cap=operating_limit(limit,fraction)
    if cap>=limit or loss>cap:return None
    near=[m for m in moves if m.gain>0 and cap<m.error<=int(Decimal(cap)*Decimal(str(near_ratio)))]
    if near:
        m=min(near,key=lambda m:(-m.gain/max(1,m.error-cap),m.error,-m.gain))
        return dict(kind='near_repair',move=m,operating_limit=cap,budget_role='conservative_compensation')
    if any(m.gain>0 and m.error<=cap for m in moves):
        return dict(kind='compress',policy='balanced',fraction=1.,operating_limit=cap,
                    budget_role='conservative_compression')
    return None


def numerical_menu(worker,moves):
    """NMED debt/repair tradeoffs after native compression, at most four edges."""
    native=policy_menu(worker)
    limit=worker.metric.limit
    near=[m for m in moves if m.gain>0 and limit<m.error<=int(Decimal(limit)*Decimal(str(worker.args.near_ratio)))]
    # Preserve the native low-error/accurate prefix. Compensation opportunities
    # are introduced only after a real numerical approximation has been made.
    if not near or worker.loss<.6*limit:return native
    efficient=min(near,key=lambda m:((m.error-limit)/m.gain,-m.gain,m.error))
    largest=min(near,key=lambda m:(-m.gain,m.error))
    result=[a for a in native if a['kind']=='compress' and a.get('policy')=='balanced' and a.get('fraction')==1.]
    result.append(dict(kind='near_repair',move=efficient))
    repairs=[m for m in moves if m.error<worker.loss and m.error<=limit]
    if repairs:
        best=min(repairs,key=lambda m:((max(0,-m.gain)+1)/max(1,worker.loss-m.error),m.error))
        result.append(dict(kind='repair',move=best))
    if largest is not efficient:result.append(dict(kind='near_repair',move=largest))
    for a in native:
        if len(result)>=4:break
        if a['kind']=='toggle_view':result.append(a)
    return result[:4]


def release_local_failures(worker):
    """A rejection under one numerator cap is not a proof at a wider cap."""
    prior=getattr(worker,'_formal_rejections_before_local',None)
    if prior is None:return dict(blocked=0,failed=0)
    released=dict(blocked=len(worker.blocked-prior[0]),failed=len(worker.failed-prior[1]))
    worker.blocked=set(prior[0]);worker.failed=set(prior[1])
    del worker._formal_rejections_before_local
    return released


class NumericBudgetMCTS(MCTSExplorer):
    def __init__(self,args):
        self.edge_limit=None;self.edge_deadline=None;self.local_budget_seconds=0.
        super().__init__(args)
        own=Path(__file__).resolve();shutil.copy2(own,self.folder/'source'/own.name)
        self.protocol['code_sha256'][own.name]=sha(own)
        self.protocol.update(nmed_budget_diversity=True,conservative_fraction=args.conservative_fraction,
            formal_bound_unchanged=True,budget_policy=args.budget_policy,
            selection_policy=args.selection_policy,expansion_cap=args.expansions,
            local_edge_seconds=args.local_edge_seconds,local_budget_share=args.local_budget_share,
            operating_budget_policy='native prefix first; bounded optional conservative edge only after measured stall; cap-specific rejection isolation',
            candidate_kernel='existing NMED response fitting, word quantization, measured compensation, exact mapping',
            er_specific_quality_operators=False)
        save(self.folder/'protocol.json',self.protocol)

    def remaining(self):
        value=super().remaining()
        deadline=getattr(self,'edge_deadline',None)
        return min(value,max(0.,deadline-time.perf_counter())) if deadline is not None else value

    def menu(self,bid):
        if bid not in self.catalog:
            w=self.touch(bid);moves=w.scan()['moves']
            native=numerical_menu(w,moves)
            mode=self.args.budget_policy
            stalled=(structural_id(w.current),'batch',1.) in w.blocked or w.stalls>0
            allowed=(mode=='priority' or (mode=='adaptive' and stalled and w.steps>0))
            low=budget_edge(moves,w.loss,self.metric.limit,self.args.conservative_fraction,self.args.near_ratio) if allowed else None
            if low and mode=='priority':menu=[low,*native[:3]]
            elif low:
                # Do not evict a representation switch or economic repair.
                menu=list(native)
                replace=next((i for i,a in reversed(list(enumerate(menu))) if a['kind']=='compress' and
                    (a.get('policy')=='area' or a.get('fraction')!=1.)),None)
                if len(menu)<4:menu.append(low)
                elif replace is not None:menu[replace]=low
            else:menu=native
            self.catalog[bid]=menu[:4]
            w.cached=None
        return self.catalog[bid]

    def select(self):
        if self.args.selection_policy=='uct':return super().select()
        # Standard UCT descent with progressive widening of costly macro
        # actions. The four-edge cap remains; descendants are explored before
        # exhausting four long rollouts from every accurate representation.
        bid='root'
        while True:
            node=self.tree[bid]
            choices=[c for c in node['children'] if not self.tree[c]['terminal']]
            if bid!='root':
                menu=self.menu(bid);pending=node['next_action']<len(menu)
                quota=min(4,max(1,math.ceil(math.sqrt((node['visits']+1)/3.))))
                if pending and (len(node['children'])<quota or not choices):return bid
            if not choices:
                node['terminal']=True
                return None if bid=='root' else self.select()
            bid=max(choices,key=lambda c:(uct_score(self.tree[c],node['visits'],self.args.uct_c),-int(c[1:])))

    def child(self,parent):
        if len(self.tree[parent]['children'])>=4:raise ValueError('MCTS permits at most four children')
        bid,w=super().child(parent)
        cap=self.metric.limit if self.edge_limit is None else self.edge_limit
        if cap<self.metric.limit:
            w._formal_rejections_before_local=(set(w.blocked),set(w.failed))
        set_operating_budget(w,self.metric,cap)
        self.origin[bid].update(operating_limit=cap,formal_limit=self.metric.limit)
        return bid,w

    def expand(self,parent,action):
        self.edge_limit=action.get('operating_limit')
        local=self.edge_limit is not None and self.edge_limit<self.metric.limit
        tick=time.perf_counter()
        if local:
            allowance=min(self.args.local_edge_seconds,
                max(0.,self.args.seconds*self.args.local_budget_share-self.local_budget_seconds),self.remaining())
            if allowance<.5:
                self.coverage.append(dict(iteration=self.expansions,parent=parent,child=None,action=action_record(action),
                    seconds=0.,skipped='local operating-budget resource share exhausted'))
                self.edge_limit=None;return
            self.edge_deadline=tick+allowance
        before_graph=self.tree[parent]['graph']
        parent_metric=self.workers[parent].metric
        parent_args=self.workers[parent].args
        old_workers=set(self.workers)
        try:
            if action['kind']!='near_repair':
                super().expand(parent,action)
            else:
                # Repair relative to this branch's operating budget, even if
                # the compression was feasible under the formal NMED bound.
                start=time.perf_counter();bid,w=self.child(parent)
                detail=near_repair_rollout(w,action['move'],self.args.rollout_steps,3,self.remaining)
                self.archive_graph(bid);graph=structural_id(w.current);self.tree[bid]['graph']=graph
                duplicates=[n for n,t in self.tree.items() if n not in ('root',bid) and t.get('graph')==graph]
                if duplicates:self.tree[bid]['terminal']=True;detail['duplicate_of']=duplicates[0]
                self.assess(bid);reward=self.tree[bid]['reward'];cursor=bid
                while cursor is not None:
                    self.tree[cursor]['visits']+=1;self.tree[cursor]['value_sum']+=reward
                    cursor=self.tree[cursor]['parent']
                record=dict(iteration=self.expansions,parent=parent,child=bid,action=action_record(action),
                    before_eligible_area=self.tree[parent]['best_eligible_area'],after_eligible_area=self.tree[bid]['best_eligible_area'],
                    training_metric=w.loss/self.metric.count/self.metric.denominator,
                    seconds=time.perf_counter()-start,detail=detail)
                self.coverage.append(record);self.expansions+=1
                print('[NMED BUDGET REPAIR]',json.dumps({k:v for k,v in record.items() if k not in ('detail','action')}),flush=True)
            self.coverage[-1].update(operating_limit=self.edge_limit or self.metric.limit,formal_limit=self.metric.limit)
        finally:
            self.edge_deadline=None
            released=dict(blocked=0,failed=0)
            for bid in set(self.workers)-old_workers:
                counts=release_local_failures(self.workers[bid])
                for key in released:released[key]+=counts[key]
                set_operating_budget(self.workers[bid],self.metric,self.metric.limit)
            if local:self.local_budget_seconds+=time.perf_counter()-tick
            if self.coverage:self.coverage[-1]['local_rejections_released']=released
            self.edge_limit=None
        if structural_id(self.touch(parent).current)!=before_graph:
            raise AssertionError('Operating-budget rollout mutated parent graph')
        if self.workers[parent].metric is not parent_metric or self.workers[parent].args is not parent_args:
            raise AssertionError('Operating-budget rollout mutated parent options')

    def finish(self):
        selected_rows=[row for bid in self.workers for row in self._eligible_rows(bid)]
        from numeric_agentic import select_checkpoint
        selected=select_checkpoint(selected_rows)
        path=self.workers[selected['branch']].folder/selected['path']
        # Freeze the selector result before inherited independent checks.
        frozen=freeze_physical_checkpoint(path,self.lib,self.folder,selected)
        code=super().finish();out=self.folder/'report.json'
        r=json.loads(out.read_text(encoding='utf-8'))
        identity=lambda row:(row['branch'],row['path'])
        if r['netlist_sha256']!=frozen or identity(r['selected_checkpoint'])!=identity(selected):
            raise AssertionError('Final checks changed selection')
        r.update(nmed_budget_diversity=True,formal_bound_unchanged=True,
            budget_policy=self.args.budget_policy,selection_policy=self.args.selection_policy,
            local_budget_seconds=self.local_budget_seconds,
            conservative_fraction=self.args.conservative_fraction,
            budget_edges=sum(bool(c['action'].get('budget_role')) for c in self.coverage if c.get('child') is not None),
            conservative_compensation_edges=sum(c['action'].get('budget_role')=='conservative_compensation' for c in self.coverage if c.get('child') is not None),
            observed_max_children=max(len(t['children']) for t in self.tree.values()),
            historical_approximate_import=False,er_specific_quality_operators=False,
            no_final_based_reselection=True)
        save(out,r);return code

    def run(self):
        self.status='running'
        cap=self.args.expansions
        while self.remaining()>.5 and (cap==0 or self.expansions<cap):
            parent=self.select()
            if parent is None:self.status='screened_tree_exhausted';break
            node=self.tree[parent];menu=self.menu(parent)
            if self.remaining()<=.5:break
            action=menu[node['next_action']];node['next_action']+=1
            self.expand(parent,action);self.persist_tree()
        if self.status=='running':self.status='time_limit' if self.remaining()<=.5 else 'expansion_limit'
        self.persist_tree();code=self.finish();path=self.folder/'report.json'
        r=json.loads(path.read_text(encoding='utf-8'))
        r.update(search_algorithm='four-child UCT with bounded macro rollouts and progressive widening' if self.args.selection_policy=='progressive' else 'four-child UCT',
            expansions=self.expansions,expansion_cap=cap,development_reward_propagations=self.development_calls,
            complete_lac_enumeration=False,max_children=4,max_cached_graphs=4,
            tree_exhaustion_is_not_global_optimality=True)
        save(path,r);return code

    def _eligible_rows(self,bid):
        # Reuse the same training/development response cache that defines UCT
        # reward. The inherited selector uses the identical formal bound.
        from bacs_data import load
        from bacs_numeric import number
        from simulate import trace,outputs
        w=self.workers[bid]
        rows=[]
        for checkpoint in w.history:
            row=dict(checkpoint,branch=bid,eligible=False)
            if row['training_loss']<=self.metric.limit:
                path=w.folder/row['path'];key=sha(path)
                if key not in self.development_cache:
                    c=load(path,self.lib)
                    y=number(outputs(c,trace(c,self.inputs[1],10000)),self.order,10000)
                    self.development_cache[key]=self.metric.exact(y,self.refs[1]);self.development_calls+=1
                loss,value=self.development_cache[key]
                row.update(eligible=loss<=self.metric.limit,development_loss=loss,development_metric=value)
            rows.append(row)
        return rows


def parser():
    p=base_parser();p.description=__doc__
    next(a for a in p._actions if a.dest=='controller').choices=('mcts',)
    p.set_defaults(controller='mcts',seconds=180.,decisions=100000)
    p.add_argument('--expansions',type=int,default=0,help='0: use the time/stall limit without an arbitrary expansion-count stop')
    p.add_argument('--uct-c',type=float,default=.05)
    p.add_argument('--rollout-steps',type=int,default=3)
    p.add_argument('--conservative-fraction',type=float,default=.1)
    p.add_argument('--budget-policy',choices=('adaptive','off','priority'),default='adaptive')
    p.add_argument('--selection-policy',choices=('progressive','uct'),default='progressive')
    p.add_argument('--local-edge-seconds',type=float,default=10.)
    p.add_argument('--local-budget-share',type=float,default=.15)
    return p


def run(a):
    if a.metric!='nmed' or a.controller!='mcts':raise ValueError('This port is NMED MCTS only')
    if False:raise ValueError('D: only')
    if any(not math.isfinite(v) or v<=0 for v in (a.seconds,a.bound,a.uct_c,a.dp_seconds,a.map_seconds,a.pool,a.pairs,a.per_root,a.local_edge_seconds)) or a.expansions<0:
        raise ValueError('Invalid finite NMED resources')
    if not math.isfinite(a.conservative_fraction) or not 0<a.conservative_fraction<1:
        raise ValueError('Invalid conservative fraction')
    if not math.isfinite(a.local_budget_share) or not 0<a.local_budget_share<=.3:
        raise ValueError('Invalid local exploration resource share')
    if a.training_fraction!=1 or a.development_fraction!=1 or not 1<=a.near_ratio<=1.5 or not 1<=a.rollout_steps<=6:
        raise ValueError('Use the full formal bound and bounded rollouts')
    if a.logic_view or a.sop_view or a.no_exact:raise ValueError('Use physical exact NMED views')
    return NumericBudgetMCTS(a).run()


if __name__=='__main__':raise SystemExit(run(parser().parse_args()))
