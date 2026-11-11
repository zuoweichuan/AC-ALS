"""Existing billed provider with the newly explicit ER experience tool menu."""
from batch_provider import choose as request
from quality_agent_provider import SYSTEM as PREVIOUS

SYSTEM=PREVIOUS+'''

Two additional EXPLICIT globally selected tools are available:
experience_wide: recollect a bounded wider shortlist on the selected CURRENT
mapped checkpoint and run ordinary conditional compression. Particularly useful
when older narrow-root scans plateau while measured error headroom remains.
experience_joint: jointly choose compression AND error-reducing LACs; small
area-cost repairs and individually near-infeasible compressions may combine.
At most four plans per beam layer, six actions/plan and twelve actual plans.
Working error reserves on BOTH selection sets are displayed. They do not change
the formal ER bound. Rank using measured headroom, involved functions and history;
input-mask proxies are not exact joint predictions. Whole-graph simulation decides.

These tools use a separate displayed allowance WITHIN the same total wall limit.
You must select them; finish NEVER runs either automatically. Prefer an untested
wide or joint probe on a promising changed graph over repeating a measured
identical-profile plateau. Recommended_profile uses selection headroom only and
is advice, not a command. Choose whether quality justifies the extra computation.
Real joint validation can increase full simulations even when root/function work
falls. Do not promise acceleration merely because it is a new operator.
'''


def choose(observation,remaining,cap=10.):return request(observation,remaining,cap,system=SYSTEM)
