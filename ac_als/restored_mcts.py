"""Original screened ER kernel + protected greedy spine + four-child UCT.

The greedy spine is computed afresh from the accurate baseline in this run;
it is part of the MCTS tree and budget. No historical approximate circuit is
loaded. New joint-plan proposals extend this kernel rather than replace it.
"""
import runtime
import json,math,shutil,time
from pathlib import Path
from bacs_numeric import save
from bacs_data import LIBRARY_PATH,load
from logic import Library
from er_search_resub import MCTS as OriginalMCTS,parser as original_parser
import er_resub_data as data
from hybrid_identity import structural_id
from numeric_agentic import select_checkpoint


class RestoredMCTS(OriginalMCTS):
    def __init__(self,args):
        self.spine_running=False;self.stage_deadline=None;self.joint_attempts=0
        self.joint_seen=set();self.spine=None;self.spine_actions={}
        super().__init__(args)
        # Include the actual new extension sources in the per-run audit.
        names=('restored_mcts.py','restored_mcts_joint.py','research_plan_overlay.py',
               'research_c880_recollect_batches.py','research_c880_plan_search.py','batch_wide.py')
        from prepare_resub_iscas import sha
        for name in names:
            p=Path(__file__).parent/name
            if p.exists():
                shutil.copy2(p,self.folder/'source'/name);self.protocol['code_sha256'][name]=sha(p)
        self.protocol.update(search_algorithm='four-child UCT with protected native greedy spine',
            native_kernel='ERWorker compression / bit and joint repair / exact mapping / exact alternate views',
            shared_kernel_with_greedy=True,greedy_spine_is_current_run_not_fallback=True,
            unchanged_original_training_and_development_guards=True,
            joint_extension_policy='bounded current-topology recollection and joint plan only after native rollout stalls')
        save(self.folder/'protocol.json',self.protocol)

    def remaining(self):
        remaining=super().remaining()
        if self.stage_deadline is not None:remaining=min(remaining,max(0.,self.stage_deadline-time.perf_counter()))
        return remaining

    def child(self,parent):
        if len(self.tree[parent]['children'])>=4:raise ValueError('MCTS permits at most four children')
        return super().child(parent)

    def menu(self,bid):
        menu=super().menu(bid)
        # If a bounded spine scan times out and its catalogue is rebuilt, the
        # already measured spine edge must still count towards four children.
        if bid in self.spine_actions:
            menu=[a for a in menu if (a.get('kind'),a.get('policy'),a.get('fraction'))!=self.spine_actions[bid]]
            self.catalog[bid]=menu
        return menu

    def best_search_checkpoint(self):
        rows=[r for bid in self.workers for r in self.development_rows(bid)]
        return select_checkpoint(rows)

    def greedy_spine(self):
        """Same balanced compression as the default kernel, to measured stall.

        Each macro edge is inserted and backpropagated normally. Its matching
        menu action is consumed, so later UCT expansion retains the other
        original actions without creating a fifth child or replaying the edge.
        """
        parent=self.active[0];tick=time.perf_counter();edges=[];complete=False
        self.spine_running=True
        self.stage_deadline=min(self.started+self.args.seconds,
                                time.perf_counter()+min(self.args.backbone_seconds,self.args.seconds*.5))
        try:
            for _ in range(self.args.backbone_edges):
                if self.remaining()<.5:break
                w=self.touch(parent);graph=structural_id(w.current)
                if (graph,'batch',1.) in w.blocked:complete=self.remaining()>.5;break
                menu=self.menu(parent)
                index=next((i for i,a in enumerate(menu) if a.get('kind')=='compress' and
                            a.get('policy')=='balanced' and a.get('fraction')==1.),None)
                if index is None:complete=self.remaining()>.5;break
                action=menu.pop(index)
                self.expand(parent,action)
                self.spine_actions[parent]=(action.get('kind'),action.get('policy'),action.get('fraction'))
                record=self.coverage[-1];record['protected_greedy_spine']=True
                edges.append(dict(parent=parent,child=record['child'],area=record['after_eligible_area']))
                parent=record['child'];w=self.touch(parent)
                if (structural_id(w.current),'batch',1.) in w.blocked:
                    complete=self.remaining()>.5;break
        finally:
            self.spine_running=False;self.stage_deadline=None
        if not complete:
            # A deadline-truncated candidate scan must not become a claim of
            # convergence or hide the remaining roots during tree expansion.
            self.catalog.clear()
            for w in self.workers.values():w.cached=None
        selected=self.best_search_checkpoint()
        self.spine=dict(completed=complete,seconds=time.perf_counter()-tick,edges=edges,
            selected_checkpoint=selected,area=selected['area'],computed_from_accurate_start=True)
        save(self.folder/'greedy_spine.json',self.spine)
        print('[GREEDY SPINE]',json.dumps(dict(completed=complete,area=selected['area'],seconds=self.spine['seconds'],edges=len(edges))),flush=True)
        self.persist_tree()

    def expand(self,parent,action):
        # Preserve every native operation, including original three-step
        # rollouts, partial-bit repair, CEC remapping and alternate views.
        super().expand(parent,action)
        self.extend_rollout(self.coverage[-1])

    def extend_rollout(self,record):
        bid=record['child'];parent=record['parent'];w=self.touch(bid)
        graph=structural_id(w.current)
        stalled=(graph,'batch',1.) in w.blocked or w.stalls>0
        if self.spine_running or not stalled or self.joint_attempts>=self.args.joint_extensions or self.remaining()<3 or graph in self.joint_seen:
            return
        self.joint_seen.add(graph);self.joint_attempts+=1
        from restored_mcts_joint import extend
        old_reward=self.tree[bid]['reward'];start=time.perf_counter()
        detail=extend(w,self.folder/('joint_%02d'%self.joint_attempts),min(self.args.joint_seconds,self.remaining()-.5),self.args)
        record['detail']['joint_extension']=detail
        record['seconds']+=time.perf_counter()-start
        self.archive_graph(bid);self.tree[bid]['graph']=structural_id(w.current)
        new_reward=self.assess(bid);cursor=bid
        # This is one extended rollout, not a second visit. Correct its
        # original reward in place instead of double-counting simulations.
        while cursor is not None:
            self.tree[cursor]['value_sum']+=new_reward-old_reward;cursor=self.tree[cursor]['parent']
        record['after_eligible_area']=self.tree[bid]['best_eligible_area']
        duplicates=[n for n,r in self.tree.items() if n not in ('root',bid) and r.get('graph')==self.tree[bid]['graph']]
        self.tree[bid]['terminal']=bool(duplicates)
        if not duplicates:record['detail'].pop('duplicate_of',None)
        if structural_id(self.touch(parent).current)!=self.tree[parent]['graph']:raise AssertionError('Extension mutated parent')
        print('[JOINT EXTENSION]',json.dumps(dict(node=bid,area=record['after_eligible_area'],detail=detail)),flush=True)

    def run(self):
        self.greedy_spine()
        # The completed default path is itself a measured stalled rollout.
        # Recollect HERE rather than replacing the earlier greedy trajectory.
        if self.spine['completed'] and self.coverage:self.extend_rollout(self.coverage[-1])
        return super().run()

    def finish(self):
        selected=self.best_search_checkpoint()
        if self.spine and selected['units']>self.spine['selected_checkpoint']['units']:
            raise AssertionError('MCTS lost its protected greedy incumbent')
        from prepare_resub_iscas import sha
        frozen=load(self.workers[selected['branch']].folder/selected['path'],self.lib)
        frozen.write(self.folder/'selected_before_checks.v')
        frozen_hash=sha(self.folder/'selected_before_checks.v')
        save(self.folder/'frozen_selection.json',dict(checkpoint=selected,sha256=frozen_hash,
             selection='original train/development guards only; before independent diagnostics'))
        code=super().finish();path=self.folder/'report.json'
        r=json.loads(path.read_text(encoding='utf-8'))
        if r['netlist_sha256']!=frozen_hash:raise AssertionError('Selection changed after final checks')
        r.update(greedy_spine=self.spine,no_worse_than_greedy_search_area=r['area']<=self.spine['area'],
            greedy_guarantee_complete=bool(self.spine['completed'] and r['area']<=self.spine['area']),
            greedy_comparison_is_current_run=True,joint_extension_attempts=self.joint_attempts,
            observed_max_children=max(len(row['children']) for row in self.tree.values()),
            restored_native_kernel=True,quality_caveat='The protected guarantee concerns search-eligible physical area; independent final ER is still checked and failures are retained.')
        save(path,r);return code


def parser():
    p=original_parser();p.description=__doc__
    p.set_defaults(controller='mcts')
    p.add_argument('--backbone-seconds',type=float,default=60.)
    p.add_argument('--backbone-edges',type=int,default=8)
    p.add_argument('--joint-extensions',type=int,default=2)
    p.add_argument('--joint-seconds',type=float,default=12.)
    return p


def run(a):
    if a.controller!='mcts':raise ValueError('This restoration entry does not change Agent')
    if False:raise ValueError('D: only')
    if not all(math.isfinite(v) for v in (a.seconds,a.expansions,a.backbone_seconds,a.joint_seconds)) or min(a.seconds,a.expansions,a.backbone_edges,a.backbone_seconds,a.joint_seconds)<1 or not 0<=a.joint_extensions<=4:
        raise ValueError('Invalid bounded restoration protocol')
    if a.metric!='er' or a.logic_view or a.sop_view or a.no_exact or a.training_fraction!=1:
        raise ValueError('Use original physical strict-ER kernel')
    tick=time.perf_counter();data.configure(a.baseline)
    data.check_start(a.design,a.accurate_netlist,Library(LIBRARY_PATH))
    a.baseline_precheck_seconds=time.perf_counter()-tick;a.no_word=True
    return RestoredMCTS(a).run()


if __name__=='__main__':raise SystemExit(run(parser().parse_args()))
