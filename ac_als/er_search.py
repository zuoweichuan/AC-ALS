"""Updated menu Agent / four-child UCT tools on strict ISCAS85 vector ER."""
import math
import er_data
from numeric_agentic import parser as base_parser
from numeric_explore import MCTSExplorer
from numeric_menu_agent import MenuPortfolio


class ERMenuPortfolio(MenuPortfolio):problem_data=er_data
class ERMCTSExplorer(MCTSExplorer):problem_data=er_data


def parser():
    p=base_parser();p.description=__doc__
    next(a for a in p._actions if a.dest=='design').choices=er_data.DESIGNS
    next(a for a in p._actions if a.dest=='metric').choices=('er',)
    next(a for a in p._actions if a.dest=='controller').choices=('agent','mcts')
    next(a for a in p._actions if a.dest=='bound').required=False
    p.add_argument('--er',dest='bound',type=float,choices=(.005,.05),required=True)
    p.add_argument('--expansions',type=int,default=64)
    p.add_argument('--uct-c',type=float,default=.05);p.add_argument('--rollout-steps',type=int,default=3)
    p.set_defaults(metric='er',controller='agent',no_word=True,seconds=180.,decisions=128,
        knowledge=str(er_data.ROOT/'er_agent_knowledge.json'))
    return p


def run(a):
    if a.metric!='er' or a.logic_view or a.sop_view or a.no_exact:
        raise ValueError('ER entry uses its own accurate standard-cell views and proven exact maps')
    if a.training_fraction!=1 or a.development_fraction!=1:raise ValueError('Use full declared ER; no hidden margins')
    limits=(a.seconds,a.decisions,a.expansions,a.uct_c,a.rollout_steps,a.pool,a.pairs,a.per_root)
    if any(not math.isfinite(v) or v<=0 for v in limits) or a.rollout_steps>6 or not 1<=a.near_ratio<=1.5:
        raise ValueError('Invalid explicit tool limits')
    a.no_word=True  # Numeric low-word quantization has no meaning for vector ER.
    if a.controller=='agent':
        from hybrid_secrets import load_key
        load_key()
    portfolio=(ERMenuPortfolio if a.controller=='agent' else ERMCTSExplorer)(a)
    portfolio.protocol.update(metric_definition='P(any output bit differs)',
        vector_errors_allowed=portfolio.metric.limit,count_per_group=10000,
        initialized_from='accurate original netlist, no historical approximate warm start',
        numeric_word_quantization=False)
    from bacs_numeric import save
    save(portfolio.folder/'protocol.json',portfolio.protocol)
    return portfolio.run()


if __name__=='__main__':raise SystemExit(run(parser().parse_args()))
