"""Screened conditional/near-repair UCT, followed by current-run physical refinement.

This restores the other missing successful prefix: conditional divisor fitting
and multi-step near-miss compensation. Search and refinement share one total
budget. Historical approximate circuits are never loaded. The final refinement
is explicitly recorded separately from tree rollouts, not counted as visits.
"""
import json,math,shutil,time
from pathlib import Path
from bacs_data import load,LIBRARY_PATH
from bacs_numeric import save
from logic import Library
from research_c880_branch import MCTS as ConditionalMCTS
from numeric_agentic import clone_worker,select_checkpoint
from quality_mcts import parser as base_parser
from quality_operators import extend
from prepare_resub_iscas import sha
import er_resub_data as data


class QualityResearchMCTS(ConditionalMCTS):
    def __init__(self,args):
        self.search_deadline=None;self.post_search=None
        super().__init__(args)
        self.search_deadline=min(self.started+args.research_seconds,self.started+args.seconds)
        for name in ('quality_research_mcts.py','quality_operators.py','research_c880_branch.py',
            'research_c880_conditional.py','research_c880_views.py','research_c880_loss_cut.py',
            'research_c880_exact_frontier.py','research_c880_observable_exact.py',
            'research_c880_local_care_sat.py','research_c880_cut_abc.py','research_c880_mapped_sat.py',
            'research_c880_budget_care_sat.py'):
            p=Path(__file__).parent/name;shutil.copy2(p,self.folder/'source'/name)
            self.protocol['code_sha256'][name]=sha(p)
        self.protocol.update(quality_operator_migration=True,
            search_kernel='standard four-child UCT with screened conditional divisors and multi-step targeted compensation',
            research_seconds=args.research_seconds,quality_seconds=args.quality_seconds,
            post_search_refinement=True,historical_approximate_import=False,
            finalization_is_not_tree_visits=True)
        save(self.folder/'protocol.json',self.protocol)

    def remaining(self):
        left=super().remaining()
        return min(left,max(0.,self.search_deadline-time.perf_counter())) if self.search_deadline is not None else left

    def finish(self):
        # No held-out inputs have been read yet. Freeze the best SEARCH parent,
        # refine a detached copy, and preserve every archived tree node.
        parent=select_checkpoint([row for bid in self.workers for row in self.development_rows(bid)])
        source=load(self.workers[parent['branch']].folder/parent['path'],self.lib)
        search_elapsed=time.perf_counter()-self.started;self.search_deadline=None
        original_tree=json.dumps(self.tree,sort_keys=True)
        w=clone_worker(self.workers[parent['branch']],self.folder/'physical_refinement',
                       self.remaining,self.started,source,parent['training_loss'],False)
        w.training_view=None;w.development_view=None;w.bit_cache=None;w.verified_pairs={}
        self.workers['physical_refinement']=w;self.origin['physical_refinement']=dict(kind='post_search_refinement',parent=parent)
        # Archived search nodes remain immutable on disk. Release their
        # name-keyed simulation caches before building fresh refinement data.
        for bid,other in self.workers.items():
            if other is w:continue
            other.current=None;other.cached=None;other.training_view=None;other.development_view=None
            other.bit_cache=None;other.verified_pairs={}
            if hasattr(other,'area_model'):del other.area_model
        self.cache_order=['physical_refinement']
        detail=extend(w,self.folder/'quality_operators',min(self.args.quality_seconds,self.remaining()),self.args)
        if json.dumps(self.tree,sort_keys=True)!=original_tree:raise AssertionError('Finalization changed MCTS tree')
        # Common report timing reads state.elapsed. Include the complete
        # refinement here, rather than reporting only the earlier UCT stage.
        self.persist_tree()
        selected=select_checkpoint([row for bid in self.workers for row in self.development_rows(bid)])
        winner=load(self.workers[selected['branch']].folder/selected['path'],self.lib)
        winner.write(self.folder/'selected_before_checks.v')
        frozen=sha(self.folder/'selected_before_checks.v')
        save(self.folder/'frozen_selection.json',dict(checkpoint=selected,sha256=frozen,
             selection='train/development only, after current-run refinement and before all final checks'))
        code=super().finish();path=self.folder/'report.json';r=json.loads(path.read_text(encoding='utf-8'))
        if r['netlist_sha256']!=frozen:raise AssertionError('Held-out data changed selection')
        r.update(quality_operator_migration=True,conditional_research_kernel=True,
            post_search_parent=parent,post_search_refinement=detail,uct_stage_seconds=search_elapsed,
            observed_max_children=max(len(row['children']) for row in self.tree.values()),
            historical_approximate_import=False,finalization_is_not_tree_visits=True)
        save(path,r);return code


def parser():
    p=base_parser();p.description=__doc__
    p.set_defaults(seconds=420.,quality_seconds=260.,near_ratio=1.5,expansions=24,
                   research_view='mapped',quality_fit_first=True,truth_width=4)
    p.add_argument('--research-seconds',type=float,default=120.)
    p.add_argument('--conditional-roots',type=int,default=48)
    p.add_argument('--conditional-shortlist',type=int,default=96)
    return p


def run(a):
    if a.controller!='mcts' or a.metric!='er' or a.no_exact or a.logic_view or a.sop_view or a.training_fraction!=1:
        raise ValueError('Use physical strict-ER MCTS')
    if False:raise ValueError('D: only')
    if any(not math.isfinite(v) or v<=0 for v in (a.seconds,a.research_seconds,a.quality_seconds,a.expansions,a.conditional_roots,a.conditional_shortlist)):
        raise ValueError('Invalid finite search/refinement budget')
    if not 1<=a.near_ratio<=1.5 or a.rollout_steps>6:raise ValueError('Invalid bounded search')
    tick=time.perf_counter();data.configure(a.baseline);data.check_start(a.design,a.accurate_netlist,Library(LIBRARY_PATH))
    a.baseline_precheck_seconds=time.perf_counter()-tick;a.no_word=True
    return QualityResearchMCTS(a).run()


if __name__=='__main__':raise SystemExit(run(parser().parse_args()))
