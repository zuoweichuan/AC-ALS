"""Small, deterministic validation portfolio; never select on final-test ER."""

def repair_economics(branch,reference,limit,maximum_debt):
 # Integer physical-area units avoid confusing a hard protocol permission with
 # the much smaller area debt a branch can afford while beating the incumbent.
 headroom=reference['units']-branch['units']
 return dict(reference_node=reference['id'],area_delta_units=branch['units']-reference['units'],
  training_error_delta=branch['train_errors']-reference['train_errors'],
  minimum_repairs_to_training_feasibility=max(0,branch['train_errors']-limit),
  repair_debt_units_to_still_beat_reference=max(0,min(maximum_debt,headroom-1)),
  further_area_units_needed_to_beat_reference=max(0,1-headroom),
  protocol_maximum_debt_units=maximum_debt,
  scope='Soft economic evidence, not an enforced debt cap or a repair guarantee. A larger/lower-error branch needs a measured recompression route.')
def development_choice(records,limit,fraction=1.,*,excluded_nodes=()):
 if not 0<fraction<=1:raise ValueError('Development fraction must be in (0,1]')
 eligible=[r for r in records if r['node'] not in excluded_nodes and r['passed'] and r['er']<=limit*fraction]
 if not eligible:raise RuntimeError('No development-margin incumbent')
 return min(eligible,key=lambda r:(r['area'],r['er'],r['node']))

def validation_portfolio(state,limit,maximum=12):
 # Distinct physical implementations first. A cheap implementation at each
 # training-error margin must survive an area-only top-k dominated by clones.
 rows=sorted(((n,r) for n,r in state['nodes'].items() if r['train_errors']<=limit),
             key=lambda pair:(pair[1]['units'],pair[1]['train_errors'],pair[0]))
 unique={}
 for n,r in rows:unique.setdefault(r['sha256'],(n,r))
 rows=list(unique.values());chosen=[]
 def add(row):
  if row is not None and row[0] not in chosen and len(chosen)<maximum:chosen.append(row[0])
 for fraction in (0,.5,.75,.8,.85,.9,.95,1):
  add(next((r for r in rows if r[1]['train_errors']<=round(limit*fraction)),None))
 for row in rows:add(row)
 return chosen

def frontier_signature(state,limit,original_units):
 # Capture feasible area AND compensation progress at several area levels.
 # Improving one arbitrary child's error count is not global progress.
 nodes=list(state['nodes'].values());sentinel=original_units*2
 area=[min((r['units'] for r in nodes if r['train_errors']<=round(limit*f)),default=sentinel)
       for f in (.75,.85,.9,.95,1)]
 errors=[min((r['train_errors'] for r in nodes if r['units']<=round(original_units*f)),default=10001)
         for f in (.5,.6,.7,.8,.9)]
 return area+errors
