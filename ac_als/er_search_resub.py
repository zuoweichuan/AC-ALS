"""ER v2 tools with a mandatory ResubALS-author accurate common baseline."""
import json
import math
from pathlib import Path
import time
import er_resub_data as data
from er_search_v2 import ERAgent, ERMCTS, parser as previous_parser
from bacs_data import LIBRARY_PATH
from bacs_numeric import save
from logic import Library
from prepare_resub_iscas import DEFAULT


class CommonBaseline:
    problem_data=data

    def __init__(self,args):
        super().__init__(args)
        self.protocol.update(baseline_kind='resubals_author_accurate_initial',
            baseline_manifest=self.audit['baseline_manifest'],
            baseline_manifest_sha256=self.audit['baseline_manifest_sha256'],
            original_dc_area=self.audit['dc_area_for_context_only'],
            baseline_area=self.audit['baseline_area'],
            reduction_denominator='raw Liberty area of author accurate initial mapped netlist',
            accurate_start_check='author BLIF and DC each have completed CEC against the frozen common accurate start; split99 packed/NumPy cross-check',
            initialized_from='author accurate BLIF; original ALSMan::Eval(initial=true); bijective port rename only',
            no_author_approximate_warm_start=True)
        save(self.folder/'protocol.json',self.protocol)
        print('[RESUB COMMON START] '+json.dumps(dict(design=args.design,
            area=self.audit['baseline_area'],sha256=self.audit['baseline_sha256'],
            baseline='author accurate initial',formal_equivalence=True)),flush=True)

    def finish(self):
        code=super().finish()
        path=self.folder/'report.json';r=json.loads(path.read_text(encoding='utf-8'))
        r.update(baseline_kind='resubals_author_accurate_initial',baseline_audit=self.audit,
            original_dc_area=self.audit['dc_area_for_context_only'],
            reduction_vs_author_initial=r['reduction'],baseline_precheck_seconds=self.args.baseline_precheck_seconds,
            paper_reproduction_claim=False,
            comparison_note='Same author accurate mapped start and raw Liberty costs; internal candidate graphs and 10000-vector protocol remain our method.')
        save(path,r)
        print('[RESUB BASELINE RESULT] '+json.dumps(dict(design=self.args.design,
            baseline_area=r['original_area'],area=r['area'],reduction=r['reduction'],strict_pass=r['strict_pass'])),flush=True)
        return code


class Agent(CommonBaseline,ERAgent):pass
class MCTS(CommonBaseline,ERMCTS):pass


def parser():
    p=previous_parser();p.description=__doc__
    p.add_argument('--baseline',default=str(DEFAULT),help='Complete verified author-accurate bundle; no DC fallback')
    return p


def run(a):
    if a.metric!='er' or a.logic_view or a.sop_view or a.no_exact:raise ValueError('Use proven standard-cell ER views')
    if a.training_fraction!=1 or not 0<a.development_fraction<=1:raise ValueError('Invalid ER fractions')
    if a.development_confidence and not .5<a.development_confidence<1:raise ValueError('Invalid confidence')
    limits=(a.seconds,a.decisions,a.expansions,a.bit_repair_roots,a.repair_pairs,a.repair_debt_units,
            a.uct_c,a.rollout_steps,a.pool,a.pairs,a.per_root)
    if any(not math.isfinite(v) or v<=0 for v in limits) or not 1<=a.near_ratio<=1.5 or a.rollout_steps>6:
        raise ValueError('Invalid bounded ER limits')
    if not 0<=a.refresh_scans<=3 or a.bit_repair_roots>64 or a.repair_pairs>12:raise ValueError('Repair screening caps exceeded')
    # Baseline rejection precedes credentials, HTTP requests, and output creation.
    tick=time.perf_counter();data.configure(a.baseline)
    row=data.check_start(a.design,a.accurate_netlist,Library(LIBRARY_PATH))
    a.baseline_precheck_seconds=time.perf_counter()-tick
    print('[RESUB BASELINE CHECK PASSED] '+json.dumps(dict(design=a.design,area=row['area'],sha256=row['sha256'],cec_proofs=2)),flush=True)
    if a.controller=='agent':
        from hybrid_secrets import load_key
        load_key()
    a.no_word=True
    return (Agent if a.controller=='agent' else MCTS)(a).run()


if __name__=='__main__':raise SystemExit(run(parser().parse_args()))
