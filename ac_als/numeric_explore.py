"""Four-child UCT MCTS teacher over the SAME screened numerical tools.

No LLM, no LAC enumeration. Immutable search nodes, independent child rollouts,
train/development reward, final sets used only after selection. Four cached
working graphs; all trajectories are archived on D: for later analysis.
"""
import argparse
from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import shutil
import time

from numeric_agentic import Portfolio, Worker, parser as base_parser, sha, short_metric
from bacs_numeric import cleanup, number, save
from bacs_data import LIBRARY_PATH, load
from hybrid_identity import structural_id
from numeric_exact import ExactBackend
from simulate import outputs, trace


def uct_score(child,parent_visits,c):
    if child['visits']==0:return math.inf
    return child['value_sum']/child['visits']+c*math.sqrt(math.log(max(2,parent_visits))/child['visits'])


def policy_menu(worker):
    """At most four semantically different children; candidates already screened."""
    scan=worker.scan();moves=scan['moves'];limit=worker.metric.limit
    repairs=[m for m in moves if m.error<worker.loss and m.error<=limit]
    # Complement pure error-first with area-efficient repairs, not enumeration.
    repair_order=[]
    for queue in (sorted(repairs,key=lambda m:((max(0,-m.gain)+1)/max(1,worker.loss-m.error),m.error)),
                  sorted(repairs,key=lambda m:(m.error,-m.gain)),
                  sorted(repairs,key=lambda m:(-m.gain,m.error))):
        if queue and all(queue[0] is not m for m in repair_order):repair_order.append(queue[0])
    near=sorted((m for m in moves if m.gain>0 and limit<m.error<=limit*worker.args.near_ratio),
                key=lambda m:((m.error-limit)/m.gain,-m.gain))
    menu=[]
    if worker.loss<=limit:
        menu.append(dict(kind='compress',policy='balanced',fraction=1.))
    if worker.loss<=limit and worker.loss<.6*limit:
        menu.append(dict(kind='compress',policy='area',fraction=1.))
        menu.append(dict(kind='compress',policy='balanced',fraction=.5))
        menu.append(dict(kind='toggle_view'))
    else:
        for m in repair_order[:2]:menu.append(dict(kind='repair',move=m))
        if near and worker.loss<=limit:menu.append(dict(kind='near_repair',move=near[0]))
        if len(menu)<4:menu.append(dict(kind='toggle_view'))
        if len(menu)<4 and worker.loss<=limit:menu.append(dict(kind='compress',policy='area',fraction=1.))
    graph=structural_id(worker.current)
    menu=[a for a in menu if a['kind']!='compress' or
          (graph,'batch' if a['policy']=='balanced' else a['policy'],a['fraction']) not in worker.blocked]
    return menu[:4]


def action_record(action):
    return {k:(v.record() if k=='move' else v) for k,v in action.items()}


class MCTSExplorer(Portfolio):
    def __init__(self,args):
        super().__init__(args)
        own=Path(__file__).resolve()
        shutil.copy2(own,self.folder/'source'/own.name)
        self.protocol.update(search_algorithm='standard UCT MCTS',max_children=4,
            max_cached_graphs=4,uct_c=args.uct_c,rollout_steps=args.rollout_steps,
            reward='best train+development eligible physical reduction; no final reward',
            code_sha256=dict(self.protocol['code_sha256'],**{own.name:sha(own)}))
        save(self.folder/'protocol.json',self.protocol)
        self.tree={'root':dict(parent=None,children=list(self.active),visits=0,value_sum=0.,terminal=False)}
        self.catalog={};self.coverage=[];self.cache_order=list(self.active)
        self.shared_mapper=self.workers[self.active[0]].mapper
        self.development_cache={};self.development_calls=0;self.expansions=0
        for bid in self.active:
            w=self.workers[bid];w.mapper=self.shared_mapper
            self.tree[bid]=dict(parent='root',children=[],visits=0,value_sum=0.,terminal=False,
                graph=structural_id(w.current),next_action=0)
            self.archive_graph(bid)
            self.assess(bid)
        self.persist_tree()

    def archive_graph(self,bid):
        self.workers[bid].current.write(self.workers[bid].folder/'working.v')

    def touch(self,bid):
        w=self.workers[bid]
        if w.current is None:w.current=load(w.folder/'working.v',self.lib)
        if bid in self.cache_order:self.cache_order.remove(bid)
        self.cache_order.append(bid)
        while len(self.cache_order)>4:
            old=self.cache_order.pop(0);other=self.workers[old]
            other.current=None;other.cached=None
            if hasattr(other,'area_model'):del other.area_model
        return w

    def assess(self,bid):
        w=self.workers[bid];best_area=None;best_dev=None
        for r in w.history:
            if r['training_loss']>self.metric.limit:continue
            path=w.folder/r['path'];key=sha(path)
            if key not in self.development_cache:
                c=load(path,self.lib)
                y=number(outputs(c,trace(c,self.inputs[1],10000)),self.order,10000)
                total,value=self.metric.exact(y,self.refs[1]);self.development_calls+=1
                self.development_cache[key]=(total,value)
            total,value=self.development_cache[key]
            if total<=self.metric.limit and (best_area is None or r['area']<best_area):best_area,best_dev=r['area'],value
        parent=self.tree[bid]['parent']
        if best_area is None:
            parent_area=self.tree.get(parent,{}).get('best_eligible_area',self.lib.area(self.source.units))
            reward=1-parent_area/self.lib.area(self.source.units)-.02
        else:reward=1-best_area/self.lib.area(self.source.units)
        self.tree[bid].update(best_eligible_area=best_area if best_area is not None else self.lib.area(self.source.units),
                              best_development_metric=best_dev,reward=reward,training_metric=short_metric(w.loss,self.metric))
        return reward

    def menu(self,bid):
        if bid not in self.catalog:
            w=self.touch(bid)
            self.catalog[bid]=policy_menu(w)
            # Only four selected edge proposals stay resident, not the full scan.
            w.cached=None
        return self.catalog[bid]

    def select(self):
        bid='root'
        while True:
            node=self.tree[bid]
            if bid!='root' and node['next_action']<len(self.menu(bid)):return bid
            choices=[c for c in node['children'] if not self.tree[c]['terminal']]
            if not choices:
                node['terminal']=True
                return None if bid=='root' else self.select()
            bid=max(choices,key=lambda c:(uct_score(self.tree[c],node['visits'],self.args.uct_c),-int(c[1:])))

    def child(self,parent):
        p=self.touch(parent);bid='b%03d'%self.next_branch;self.next_branch+=1
        w=copy.copy(p);w.folder=self.folder/bid;w.folder.mkdir()
        w.current=p.current.copy();w.history=[];w.events=[];w.stats=defaultdict(int);w.times=defaultdict(float)
        w.started=self.started;w.global_remaining=self.remaining;w.mapper=self.shared_mapper
        w.rng=copy.deepcopy(p.rng);w.seen=set(p.seen);w.failed=set(p.failed);w.blocked=set(p.blocked)
        w.quantization_seen=set(p.quantization_seen);w.near_events=[];w.cached=None
        w.steps=0;w.iteration=0;w.stalls=0;w.repairs=0;w.last_result={}
        w.backend=ExactBackend(w.folder/'exact',LIBRARY_PATH,distro=self.args.wsl_distro,
            executable=self.args.abc,timeout=self.args.map_seconds)
        w.checkpoint('mcts_parent',[]);w.annotate(0)
        self.workers[bid]=w;self.origin[bid]=dict(kind='mcts_child',parent=parent)
        self.tree[bid]=dict(parent=parent,children=[],visits=0,value_sum=0.,terminal=False,next_action=0)
        self.tree[parent]['children'].append(bid)
        self.touch(bid)
        return bid,w

    def rollout(self,w,steps):
        results=[]
        for _ in range(steps):
            if self.remaining()<.5:break
            profile=('batch',1.)
            if (structural_id(w.current),*profile) in w.blocked:break
            r=w.compress('balanced',1.);results.append({k:v for k,v in r.items() if k!='detail'})
            if not r['changed']:break
        return results

    def expand(self,parent,action):
        before_hash=self.tree[parent]['graph'];parent_area=self.tree[parent]['best_eligible_area']
        bid,w=self.child(parent);tick=time.perf_counter();kind=action['kind'];detail={}
        if kind=='compress':
            steps=[]
            for _ in range(self.args.rollout_steps):
                if self.remaining()<.5:break
                profile=('batch' if action['policy']=='balanced' else action['policy'],action['fraction'])
                if (structural_id(w.current),*profile) in w.blocked:break
                r=w.compress(action['policy'],action['fraction']);steps.append({k:v for k,v in r.items() if k!='detail'})
                if not r['changed']:break
            detail['steps']=steps
        elif kind=='toggle_view':
            w.preserve=not w.preserve;w.refine('mcts_toggle_exact_view')
            detail['steps']=self.rollout(w,self.args.rollout_steps)
        elif kind=='repair':
            ok,info=w.apply([action['move']],'compensation');detail.update(accepted=ok,repair=info)
            if ok:w.repairs+=1;w.refine('mcts_repair_exact');detail['steps']=self.rollout(w,self.args.rollout_steps)
        elif kind=='near_repair':
            m=action['move'];c=cleanup(w.current.materialize([m.replacement()]),w.mapper);w.stats['area_materializations']+=1
            y=number(outputs(c,trace(c,self.inputs[0],10000,w.stats)),self.order,10000)
            total,value=self.metric.exact(y,self.refs[0]);w.stats['near_miss_joint_verifications']+=1
            detail.update(actual_near_metric=value,actual_near_area=self.lib.area(c.units),repaired=False)
            if c.units<w.current.units and total<=int(self.metric.limit*self.args.near_ratio):
                w.current=c;w.loss=total;w.lowest_units=min(w.lowest_units,c.units)
                w.checkpoint('mcts_infeasible_near',[m]);w.annotate(len(w.history)-1)
                repairs=w.scan()['moves'];repairs=sorted((r for r in repairs if r.error<=self.metric.limit and r.error<total),
                    key=lambda r:((max(0,-r.gain)+1)/max(1,total-r.error),r.error))[:3]
                for repair in repairs:
                    if self.remaining()<.5:break
                    ok,info=w.apply([repair],'compensation')
                    if ok:
                        detail.update(repaired=True,repair=info);w.refine('mcts_near_repaired_exact')
                        detail['steps']=self.rollout(w,self.args.rollout_steps);break
        self.archive_graph(bid)
        graph=structural_id(w.current);self.tree[bid]['graph']=graph
        duplicates=[n for n,r in self.tree.items() if n not in ('root',bid) and r.get('graph')==graph]
        if duplicates:self.tree[bid]['terminal']=True;detail['duplicate_of']=duplicates[0]
        self.assess(bid)
        # A rollout on the child must never mutate its parent's state or RNG.
        parent_worker=self.touch(parent)
        if structural_id(parent_worker.current)!=before_hash:raise AssertionError('MCTS parent mutated')
        reward=self.tree[bid]['reward'];cursor=bid
        while cursor is not None:
            self.tree[cursor]['visits']+=1;self.tree[cursor]['value_sum']+=reward
            cursor=self.tree[cursor]['parent']
        record=dict(iteration=self.expansions,parent=parent,child=bid,action=action_record(action),
            before_eligible_area=parent_area,after_eligible_area=self.tree[bid]['best_eligible_area'],
            training_metric=short_metric(w.loss,self.metric),seconds=time.perf_counter()-tick,detail=detail)
        self.coverage.append(record);self.expansions+=1
        label='[Agent TOOL EXPAND]' if self.args.controller=='agent' else '[MCTS EXPLORE]'
        print(label+' '+json.dumps({k:v for k,v in record.items() if k not in ('detail','action')},ensure_ascii=False),flush=True)

    def persist_tree(self):
        # Parent Portfolio.finish consumes the same state elapsed key.
        save(self.folder/'state.json',dict(status=self.status,elapsed=time.perf_counter()-self.started,
            tree=self.tree,expansions=self.expansions,cached_graphs=list(self.cache_order),
            max_children=4,api_calls=0,controller='uct_mcts',coverage=self.coverage,
            development_reward_calls=self.development_calls))

    def run(self):
        self.status='running'
        while self.remaining()>.5 and self.expansions<self.args.expansions:
            parent=self.select()
            if parent is None:self.status='screened_tree_exhausted';break
            node=self.tree[parent];action=self.menu(parent)[node['next_action']];node['next_action']+=1
            self.expand(parent,action);self.persist_tree()
        if self.status=='running':self.status='time_limit' if self.remaining()<=.5 else 'expansion_limit'
        self.persist_tree()
        code=self.finish()
        report=json.loads((self.folder/'report.json').read_text(encoding='utf-8'))
        report.update(search_algorithm='standard UCT MCTS',expansions=self.expansions,
            development_reward_propagations=self.development_calls,complete_lac_enumeration=False,
            max_children=4,max_cached_graphs=4,tree_exhaustion_is_not_global_optimality=True)
        save(self.folder/'report.json',report)
        return code


if __name__=='__main__':
    p=base_parser();p.description=__doc__
    next(a for a in p._actions if a.dest=='controller').choices=('mcts',)
    p.set_defaults(controller='mcts',seconds=180.,decisions=100000)
    p.add_argument('--expansions',type=int,default=64)
    p.add_argument('--uct-c',type=float,default=.05)
    p.add_argument('--rollout-steps',type=int,default=3)
    a=p.parse_args()
    if min(a.seconds,a.expansions,a.uct_c,a.rollout_steps)<=0 or a.rollout_steps>6:p.error('Invalid MCTS budget')
    raise SystemExit(MCTSExplorer(a).run())
