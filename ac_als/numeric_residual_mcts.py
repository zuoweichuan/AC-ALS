"""Isolated four-child numerical UCT with scale-aware global compensation.

Keep native accurate-prefix exploration. A bounded residual macro is offered
only after actual approximation; no author approximate solution is imported.
"""
import json
from pathlib import Path
import shutil
import time
from bacs_numeric import save
from hybrid_identity import structural_id
from numeric_agentic import sha
from numeric_budget_mcts import NumericBudgetMCTS,parser as base_parser
from numeric_explore import action_record
from numeric_joint_word import actual_trials
from numeric_residual_windows import significant_wire_proposals


def insert_residual_edge(native,worker):
    if not worker.loss or worker.loss<.45*worker.metric.limit:return native
    result=list(native);edge=dict(kind='global_residual_repair')
    if len(result)<4:result.append(edge)
    else:
        # Native balanced compression and topology changes retain their slots.
        replace=next((i for i,a in reversed(list(enumerate(result)))
            if a['kind']=='repair' or (a['kind']=='compress' and
                (a.get('policy')=='area' or a.get('fraction')!=1.))),None)
        if replace is not None:result[replace]=edge
    return result[:4]


class NumericResidualMCTS(NumericBudgetMCTS):
    def __init__(self,args):
        super().__init__(args)
        for name in ('numeric_residual_mcts.py','numeric_residual_windows.py','numeric_joint_word.py'):
            path=Path(__file__).resolve().parent/name;shutil.copy2(path,self.folder/'source'/name)
            self.protocol['code_sha256'][name]=sha(path)
        self.protocol.update(global_residual_compensation=True,residual_edge_seconds=args.residual_edge_seconds,
            preserve_native_accurate_prefix=True,repair_bit_scope='all numerical bit positions',
            historical_approximate_import=False,max_children=4)
        save(self.folder/'protocol.json',self.protocol)

    def menu(self,bid):
        if bid not in self.catalog:
            native=super().menu(bid);self.catalog[bid]=insert_residual_edge(native,self.touch(bid))
        return self.catalog[bid]

    def expand(self,parent,action):
        if action['kind']!='global_residual_repair':return super().expand(parent,action)
        graph=self.tree[parent]['graph'];tick=time.perf_counter()
        self.edge_deadline=tick+min(self.args.residual_edge_seconds,self.remaining())
        bid,w=self.child(parent);events=[]
        try:
            proposals=significant_wire_proposals(w)
            ok=actual_trials(w,proposals,'repair',self.inputs[1],self.refs[1],events)
            detail=dict(accepted=ok,proposals=len(proposals),events=events)
            if ok:detail['steps']=self.rollout(w,self.args.rollout_steps)
            self.archive_graph(bid);child_graph=structural_id(w.current);self.tree[bid]['graph']=child_graph
            duplicates=[n for n,t in self.tree.items() if n not in ('root',bid) and t.get('graph')==child_graph]
            if duplicates:self.tree[bid]['terminal']=True;detail['duplicate_of']=duplicates[0]
            self.assess(bid);reward=self.tree[bid]['reward'];cursor=bid
            while cursor is not None:
                self.tree[cursor]['visits']+=1;self.tree[cursor]['value_sum']+=reward;cursor=self.tree[cursor]['parent']
            record=dict(iteration=self.expansions,parent=parent,child=bid,action=action_record(action),
                before_eligible_area=self.tree[parent]['best_eligible_area'],after_eligible_area=self.tree[bid]['best_eligible_area'],
                seconds=time.perf_counter()-tick,detail=detail)
            self.coverage.append(record);self.expansions+=1
            print('[RESIDUAL MCTS]',json.dumps({k:v for k,v in record.items() if k not in ('action','detail')}),flush=True)
        finally:self.edge_deadline=None
        if structural_id(self.touch(parent).current)!=graph:raise AssertionError('Residual edge mutated parent')

    def finish(self):
        code=super().finish();path=self.folder/'report.json';r=json.loads(path.read_text(encoding='utf-8'))
        r.update(global_residual_compensation=True,residual_macro_expansions=sum(
            c['action']['kind']=='global_residual_repair' for c in self.coverage),
            historical_approximate_import=False,preserve_native_accurate_prefix=True)
        save(path,r);return code


def parser():
    p=base_parser();p.add_argument('--residual-edge-seconds',type=float,default=10.)
    return p


if __name__=='__main__':
    a=parser().parse_args()
    if a.metric!='nmed' or a.seconds<=0 or not .5<a.residual_edge_seconds<=30:raise ValueError('Bounded NMED only')
    if False:raise ValueError('D: only')
    raise SystemExit(NumericResidualMCTS(a).run())
