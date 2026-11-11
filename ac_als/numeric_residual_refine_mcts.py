"""Four-child numerical UCT followed by budgeted, protected residual refinement.

All phases share --seconds. Refinement is recorded separately from tree visits.
The common accurate baseline remains the only initial source.
"""
import copy
import json
from pathlib import Path
import shutil
import time
from bacs_data import load
from bacs_numeric import save
from hybrid_identity import structural_id
from numeric_agentic import clone_worker,select_checkpoint,sha
from numeric_joint_word import actual_trials
from numeric_residual_mcts import NumericResidualMCTS,parser as base_parser
from numeric_residual_windows import significant_wire_proposals,residual_window_proposals


class ResidualRefineMCTS(NumericResidualMCTS):
    def __init__(self,args):
        self.search_deadline=None;self.protect_rollout=False
        super().__init__(args)
        self.search_deadline=self.started+args.seconds-args.refine_seconds
        own=Path(__file__).resolve();shutil.copy2(own,self.folder/'source'/own.name)
        self.protocol['code_sha256'][own.name]=sha(own)
        self.protocol.update(refinement_seconds=args.refine_seconds,total_budget_includes_refinement=True,
            protect_fresh_compensation_from_immediate_requantization=True,
            post_search_refinement_is_not_mcts_visits=True)
        save(self.folder/'protocol.json',self.protocol)

    def remaining(self):
        left=super().remaining()
        return min(left,max(0.,self.search_deadline-time.perf_counter())) if self.search_deadline is not None else left

    def expand(self,parent,action):
        self.protect_rollout=action['kind']=='global_residual_repair'
        try:return super().expand(parent,action)
        finally:self.protect_rollout=False

    def rollout(self,w,steps):
        old=w.args
        if self.protect_rollout:w.args=copy.copy(old);w.args.no_word=True
        try:return super().rollout(w,steps)
        finally:w.args=old

    def finish(self):
        selected=select_checkpoint([r for bid in self.workers for r in self._eligible_rows(bid)])
        source=load(self.workers[selected['branch']].folder/selected['path'],self.lib)
        self.search_deadline=None;tree_before=json.dumps(self.tree,sort_keys=True)
        started=time.perf_counter();deadline=started+min(self.args.refine_seconds,self.remaining())
        remaining=lambda:max(0.,min(self.remaining(),deadline-time.perf_counter()))
        w=clone_worker(self.workers[selected['branch']],self.folder/'residual_refinement',remaining,started,
                       source,selected['training_loss'],False)
        w.args=copy.copy(w.args);w.args.no_word=True
        self.workers['residual_refinement']=w
        self.origin['residual_refinement']=dict(kind='post_search_residual_refinement',parent=selected)
        for bid,other in self.workers.items():
            if other is not w:other.current=None;other.cached=None
        self.cache_order=['residual_refinement'];events=[];rounds=[]
        for iteration in range(3):
            if remaining()<2:break
            before=structural_id(w.current)
            for kind in ('wire','native','window','native'):
                if remaining()<2:break
                if kind=='native':
                    for policy in ('balanced','area','balanced'):
                        profile='batch' if policy=='balanced' else policy
                        if remaining()<2:break
                        if (structural_id(w.current),profile,1.) not in w.blocked:w.compress(policy,1.)
                else:
                    # Keep time for mapping and real acceptance after generation.
                    phase_deadline=time.perf_counter()+min(12.,max(.1,remaining()-4))
                    old=w.global_remaining
                    w.global_remaining=lambda:max(0.,min(old(),phase_deadline-time.perf_counter()))
                    try:plans=significant_wire_proposals(w) if kind=='wire' else residual_window_proposals(w,True)
                    finally:w.global_remaining=old
                    actual_trials(w,plans,'repair',self.inputs[1],self.refs[1],events)
            rounds.append(dict(iteration=iteration,area=self.lib.area(w.current.units),loss=w.loss,seconds=time.perf_counter()-started))
            if structural_id(w.current)==before:break
        if json.dumps(self.tree,sort_keys=True)!=tree_before:raise AssertionError('Refinement modified UCT visits')
        detail=dict(parent=selected,events=events,rounds=rounds,seconds=time.perf_counter()-started,
                    initial_area=self.lib.area(source.units),last_area=self.lib.area(w.current.units),protected=True)
        save(self.folder/'post_search_refinement.json',detail)
        self.persist_tree();code=super().finish();p=self.folder/'report.json';r=json.loads(p.read_text(encoding='utf-8'))
        r.update(protected_residual_refinement=detail,total_budget_includes_refinement=True,
            post_search_refinement_is_not_mcts_visits=True)
        save(p,r);return code


def parser():
    p=base_parser();p.add_argument('--refine-seconds',type=float,default=60.)
    p.set_defaults(seconds=300.)
    return p


if __name__=='__main__':
    a=parser().parse_args()
    if a.metric!='nmed' or not 0<a.refine_seconds<a.seconds or False:
        raise ValueError('Use NMED and a positive shared search/refinement budget on D:')
    raise SystemExit(ResidualRefineMCTS(a).run())
