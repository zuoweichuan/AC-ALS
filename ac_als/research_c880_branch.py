"""Bounded near-miss branches with multi-step measured compensation."""
import hashlib,json,time
from pathlib import Path
from collections import defaultdict
import er_search_resub as baseline
from research_c880_conditional import ConditionalViews
from er_search_v2 import repair_options,repair_rank
from bacs_numeric import cleanup,number,save
from hybrid_identity import structural_id
from simulate import trace,outputs
from numeric_explore import action_record


class MCTS(ConditionalViews,baseline.MCTS):
    def menu(self,bid):
        if bid in self.catalog:return self.catalog[bid]
        w=self.touch(bid);moves=w.scan()['moves'];limit=self.metric.limit;graph=structural_id(w.current)
        menu=[]
        near=[m for m in moves if m.gain>0 and limit<m.error<=limit*self.args.near_ratio]
        near.sort(key=lambda m:(-m.gain/max(1,m.error-limit),m.error,-m.gain))
        if near:menu.append(dict(kind='near_repair',move=near[0]))
        if w.loss<=limit and (graph,'batch',1.) not in w.blocked:
            menu.append(dict(kind='compress',policy='balanced',fraction=1.))
        repairs=sorted(repair_options(w),key=lambda m:repair_rank(m,w.loss))
        if repairs:menu.append(dict(kind='repair',move=repairs[0]))
        large=sorted(near,key=lambda m:(-m.gain,m.error))
        if large and (not near or large[0] is not near[0]):menu.append(dict(kind='near_repair',move=large[0]))
        if len(menu)<4 and w.loss<=limit and (graph,'area',1.) not in w.blocked:
            menu.append(dict(kind='compress',policy='area',fraction=1.))
        if len(menu)<4:menu.append(dict(kind='toggle_view'))
        self.catalog[bid]=menu[:4];w.cached=None
        return self.catalog[bid]

    def expand(self,parent,action):
        if action['kind']!='near_repair':return super().expand(parent,action)
        tick=time.perf_counter();before_hash=self.tree[parent]['graph'];before=self.tree[parent]['best_eligible_area']
        bid,w=self.child(parent);m=action['move'];detail=dict(repaired=False)
        try:
            c=cleanup(w.current.materialize([m.replacement()]),w.mapper);w.stats['area_materializations']+=1
            y=number(outputs(c,trace(c,self.inputs[0],10000,w.stats)),self.order,10000)
            loss,value=self.metric.exact(y,self.refs[0]);w.stats['near_miss_joint_verifications']+=1
            detail.update(initial_wrong_vectors=loss,initial_area=self.lib.area(c.units))
            if c.units<w.current.units and loss<=int(self.metric.limit*self.args.near_ratio):
                w.current=c;w.loss=loss;w.lowest_units=min(w.lowest_units,c.units);w.cached=None
                w.checkpoint('targeted_near_infeasible',[m]);w.annotate(len(w.history)-1)
                outcomes=[]
                for _ in range(4):
                    if self.remaining()<1:break
                    repairs=sorted(repair_options(w),key=lambda r:repair_rank(r,w.loss))[:4]
                    accepted=False
                    for repair in repairs:
                        accepted,info=w.apply([repair],'compensation')
                        outcomes.append(dict(accepted=accepted,detail=info))
                        if accepted:break
                    if not accepted:break
                    if w.loss<=self.metric.limit:
                        w.refine('targeted_near_repaired_exact');detail['repaired']=True
                        detail['rollout']=self.rollout(w,self.args.rollout_steps);break
                detail['repairs']=outcomes
        except ValueError as exc:detail['error']=str(exc)
        self.archive_graph(bid);graph=structural_id(w.current);self.tree[bid]['graph']=graph
        duplicates=[n for n,r in self.tree.items() if n not in ('root',bid) and r.get('graph')==graph]
        if duplicates:self.tree[bid]['terminal']=True;detail['duplicate_of']=duplicates[0]
        self.assess(bid)
        if structural_id(self.touch(parent).current)!=before_hash:raise AssertionError('Parent mutated')
        reward=self.tree[bid]['reward'];cursor=bid
        while cursor is not None:
            self.tree[cursor]['visits']+=1;self.tree[cursor]['value_sum']+=reward;cursor=self.tree[cursor]['parent']
        record=dict(iteration=self.expansions,parent=parent,child=bid,action=action_record(action),
            before_eligible_area=before,after_eligible_area=self.tree[bid]['best_eligible_area'],
            training_metric=w.loss/10000,seconds=time.perf_counter()-tick,detail=detail)
        self.coverage.append(record);self.expansions+=1
        print('[TARGETED NEAR] '+json.dumps({k:v for k,v in record.items() if k not in ('detail','action')}),flush=True)


class Agent(ConditionalViews,baseline.Agent):pass

def run(a):
    oldm,olda=baseline.MCTS,baseline.Agent;baseline.MCTS,baseline.Agent=MCTS,Agent
    try:
        code=baseline.run(a);folder=Path(a.output)
        for n in ('research_c880_branch.py','research_c880_conditional.py','research_c880_views.py'):
            source=Path(__file__).parent/n;(folder/'source'/n).write_bytes(source.read_bytes())
        return code
    finally:baseline.MCTS,baseline.Agent=oldm,olda

if __name__=='__main__':
    p=baseline.parser();p.add_argument('--research-view',choices=('mapped','expanded','author'),default='mapped')
    p.add_argument('--conditional-roots',type=int,default=48);p.add_argument('--conditional-shortlist',type=int,default=96)
    a=p.parse_args()
    if a.design!='c880':p.error('Isolated c880 only')
    raise SystemExit(run(a))
