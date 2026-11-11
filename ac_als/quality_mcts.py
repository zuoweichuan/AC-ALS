"""Original four-child MCTS plus the previously omitted physical research tools."""
import json,math,time,shutil
from pathlib import Path
from bacs_numeric import save
from restored_mcts import RestoredMCTS,OriginalMCTS,parser as base_parser
from hybrid_identity import structural_id
import er_resub_data as data
from bacs_data import LIBRARY_PATH
from logic import Library


class QualityMCTS(RestoredMCTS):
    def __init__(self,args):
        self.quality_attempts=0
        super().__init__(args)
        from prepare_resub_iscas import sha
        for name in ('quality_mcts.py','quality_operators.py','research_c880_exact_frontier.py',
            'research_c880_observable_exact.py','research_c880_local_care_sat.py',
            'research_c880_mapped_sat.py','research_c880_budget_care_sat.py','research_c880_cut_abc.py','research_c880_loss_cut.py'):
            p=Path(__file__).parent/name;shutil.copy2(p,self.folder/'source'/name)
            self.protocol['code_sha256'][name]=sha(p)
        self.protocol.update(quality_operator_migration=True,
            quality_policy='one bounded research-operator continuation after protected native greedy stall; then original UCT',
            quality_seconds=args.quality_seconds,no_historical_approximate_import=True)
        save(self.folder/'protocol.json',self.protocol)

    def extend_rollout(self,record):
        super().extend_rollout(record)
        if self.spine_running or self.quality_attempts or self.remaining()<4:return
        bid=record['child'];w=self.touch(bid);parent=record['parent']
        # Use the completed greedy/joint trajectory first. Other ordinary
        # rollouts retain the unchanged native kernel and four-child tree.
        if not self.spine:return
        self.quality_attempts+=1;reward=self.tree[bid]['reward'];started=time.perf_counter()
        from quality_operators import extend
        detail=extend(w,self.folder/'quality_operators',min(self.args.quality_seconds,self.remaining()-1),self.args)
        record['detail']['quality_operators']=detail;record['seconds']+=time.perf_counter()-started
        self.archive_graph(bid);self.tree[bid]['graph']=structural_id(w.current)
        new_reward=self.assess(bid);cursor=bid
        while cursor is not None:
            self.tree[cursor]['value_sum']+=new_reward-reward;cursor=self.tree[cursor]['parent']
        record['after_eligible_area']=self.tree[bid]['best_eligible_area']
        if structural_id(self.touch(parent).current)!=self.tree[parent]['graph']:raise AssertionError('Quality extension mutated parent')
        # No second visit: this was the continuation of the same macro rollout.
        self.persist_tree()

    def run(self):
        self.greedy_spine()
        # A larger circuit may hit the spine stage budget before stalling.
        # It must still receive the new tools on its measured current graph.
        if self.coverage:self.extend_rollout(self.coverage[-1])
        return OriginalMCTS.run(self)

    def finish(self):
        code=super().finish();p=self.folder/'report.json';r=json.loads(p.read_text(encoding='utf-8'))
        r.update(quality_operator_migration=True,quality_operator_attempts=self.quality_attempts)
        save(p,r);return code


def parser():
    p=base_parser();p.description=__doc__;p.set_defaults(seconds=360.)
    p.add_argument('--quality-seconds',type=float,default=225.)
    return p


def run(a):
    if a.controller!='mcts' or a.metric!='er' or a.logic_view or a.sop_view or a.no_exact or a.training_fraction!=1:
        raise ValueError('Use original physical strict-ER MCTS')
    if False:raise ValueError('D: only')
    if not all(math.isfinite(v) and v>0 for v in (a.seconds,a.quality_seconds,a.backbone_seconds,a.joint_seconds)):
        raise ValueError('Invalid finite quality budget')
    if not 0<=a.joint_extensions<=4 or a.rollout_steps>6 or a.expansions<1:raise ValueError('Invalid MCTS caps')
    tick=time.perf_counter();data.configure(a.baseline)
    data.check_start(a.design,a.accurate_netlist,Library(LIBRARY_PATH))
    a.baseline_precheck_seconds=time.perf_counter()-tick;a.no_word=True
    return QualityMCTS(a).run()


if __name__=='__main__':raise SystemExit(run(parser().parse_args()))
