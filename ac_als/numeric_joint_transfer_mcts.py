"""Cold NMED MCTS plus tested conditional joint-plan/near-repair continuation.

Preserve the existing225s native/residual search and75s protected refinement
allowances, then allocate up to60s to the new tool within a shared360s cap.
Every earlier eligible checkpoint remains selectable. No historical graph load.
"""
import copy,json,math,shutil,time
from pathlib import Path
from bacs_data import load
from bacs_numeric import save
from numeric_agentic import clone_worker,select_checkpoint,sha
from hybrid_identity import structural_id
from numeric_residual_refine_mcts import ResidualRefineMCTS,parser as previous_parser
from numeric_residual_mcts import NumericResidualMCTS
from numeric_residual_windows import significant_wire_proposals,residual_window_proposals
from numeric_joint_word import actual_trials
from numeric_joint_plan_transfer import ConditionalNumericWorker,operate


def enhanced_clone(parent,folder,remaining,started,circuit,loss):
    clone=clone_worker(parent,folder,remaining,started,circuit,loss,False)
    enhanced=ConditionalNumericWorker.__new__(ConditionalNumericWorker)
    enhanced.__dict__.update(clone.__dict__)
    enhanced.args=copy.copy(clone.args);enhanced.metric=copy.copy(clone.metric)
    enhanced.cached=None;enhanced.blocked=set();enhanced.failed=set()
    if hasattr(enhanced,'area_model'):del enhanced.area_model
    return enhanced


class JointTransferMCTS(ResidualRefineMCTS):
    def __init__(self,args):
        super().__init__(args)
        self.search_deadline=self.started+args.seconds-args.refine_seconds-args.joint_transfer_seconds
        for name in ('numeric_joint_transfer_mcts.py','numeric_joint_plan_transfer.py'):
            p=Path(__file__).resolve().parent/name;shutil.copy2(p,self.folder/'source'/name)
            self.protocol['code_sha256'][name]=sha(p)
        self.protocol.update(nmed_joint_transfer=True,joint_transfer_seconds=args.joint_transfer_seconds,
            uct_allowance_seconds=args.seconds-args.refine_seconds-args.joint_transfer_seconds,
            joint_transfer_roots=args.joint_transfer_roots,new_phase_within_total_budget=True,
            no_final_dependent_reselection=True,historical_approximate_import=False)
        save(self.folder/'protocol.json',self.protocol)

    def finish(self):
        selected=select_checkpoint([r for bid in self.workers for r in self._eligible_rows(bid)])
        source=load(self.workers[selected['branch']].folder/selected['path'],self.lib)
        self.search_deadline=None;tree_before=json.dumps(self.tree,sort_keys=True)
        started=time.perf_counter();deadline=started+min(self.args.refine_seconds,max(0.,self.remaining()-self.args.joint_transfer_seconds))
        remaining=lambda:max(0.,min(self.remaining(),deadline-time.perf_counter()))
        w=clone_worker(self.workers[selected['branch']],self.folder/'residual_refinement',remaining,started,
                       source,selected['training_loss'],False)
        w.args=copy.copy(w.args);w.args.no_word=True;self.workers['residual_refinement']=w
        self.origin['residual_refinement']=dict(kind='post_search_residual_refinement',parent=selected)
        for other in self.workers.values():
            if other is not w:other.current=None;other.cached=None
        self.cache_order=['residual_refinement'];events=[];rounds=[]
        for iteration in range(3):
            if remaining()<2:break
            before=structural_id(w.current)
            for kind in ('wire','native','window','native'):
                if remaining()<2:break
                if kind=='native':
                    for policy in ('balanced','area','balanced'):
                        key='batch' if policy=='balanced' else policy
                        if remaining()<2:break
                        if (structural_id(w.current),key,1.) not in w.blocked:w.compress(policy,1.)
                else:
                    phase=time.perf_counter()+min(12.,max(.1,remaining()-4));old=w.global_remaining
                    w.global_remaining=lambda:max(0.,min(old(),phase-time.perf_counter()))
                    try:plans=significant_wire_proposals(w) if kind=='wire' else residual_window_proposals(w,True)
                    finally:w.global_remaining=old
                    actual_trials(w,plans,'repair',self.inputs[1],self.refs[1],events)
            rounds.append(dict(iteration=iteration,area=self.lib.area(w.current.units),loss=w.loss,seconds=time.perf_counter()-started))
            if structural_id(w.current)==before:break
        detail=dict(parent=selected,events=events,rounds=rounds,seconds=time.perf_counter()-started,
            initial_area=self.lib.area(source.units),last_area=self.lib.area(w.current.units),protected=True)
        save(self.folder/'post_search_refinement.json',detail)
        # Freeze the prior phase's best before starting the additional tool.
        parent=select_checkpoint([r for bid in self.workers for r in self._eligible_rows(bid)])
        parent_path=self.workers[parent['branch']].folder/parent['path'];parent_sha=sha(parent_path)
        transfer=dict(skipped=True,reason='no_remaining_allowance')
        if self.remaining()>2:
            source=load(parent_path,self.lib)
            new=enhanced_clone(self.workers[parent['branch']],self.folder/'joint_transfer',self.remaining,
                self.started,source,parent['training_loss'])
            new.metric=copy.copy(self.metric);new.formal_limit=self.metric.limit
            new.args.no_word=True;new.args.near_ratio=1.5;new.args.conditional_roots=self.args.joint_transfer_roots
            new.args.targeted_near_repair=True;new.args.generation_seconds=30.;new.args.rounds=6
            self.workers['joint_transfer']=new;self.origin['joint_transfer']=dict(kind='post_search_numeric_joint_transfer',parent=parent)
            for other in self.workers.values():
                if other is not new:other.current=None;other.cached=None
            self.cache_order=['joint_transfer']
            transfer=operate(new,('joint','native'),min(self.args.joint_transfer_seconds,self.remaining()),new.args,self.folder/'joint_transfer_tool')
            transfer.update(parent=parent,initial_area=parent['area'],last_area=self.lib.area(new.current.units))
        if sha(parent_path)!=parent_sha:raise AssertionError('New tool mutated protected parent')
        if json.dumps(self.tree,sort_keys=True)!=tree_before:raise AssertionError('Refinement changed UCT visits')
        save(self.folder/'joint_transfer_report.json',transfer);self.persist_tree()
        # This parent finish performs final selection/checks exactly once.
        code=NumericResidualMCTS.finish(self)
        path=self.folder/'report.json';r=json.loads(path.read_text(encoding='utf-8'))
        if r['area']>parent['area']+1e-8:raise AssertionError('Earlier eligible winner was lost')
        r.update(nmed_joint_transfer=True,joint_transfer=transfer,protected_residual_refinement=detail,
            total_budget_includes_refinement=True,post_search_refinement_is_not_mcts_visits=True,
            prior_phase_best_area=parent['area'],prior_phase_preserved=True)
        save(path,r);return code


def parser():
    p=previous_parser();p.description=__doc__
    p.add_argument('--joint-transfer-seconds',type=float,default=60.)
    p.add_argument('--joint-transfer-roots',type=int,default=160)
    p.set_defaults(seconds=360.,refine_seconds=75.)
    return p


def run(a):
    if a.metric!='nmed' or False:raise ValueError('NMED on D: only')
    if any(not math.isfinite(v) or v<=0 for v in (a.seconds,a.refine_seconds,a.joint_transfer_seconds)):
        raise ValueError('Invalid finite budget')
    if a.refine_seconds+a.joint_transfer_seconds>=a.seconds or not 1<=a.joint_transfer_roots<=160:
        raise ValueError('Reserve positive tree time and bounded conditional roots')
    return JointTransferMCTS(a).run()


if __name__=='__main__':raise SystemExit(run(parser().parse_args()))
