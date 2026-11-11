"""DeepSeek-primary search with explicitly labelled, authorized numerical repair.

The default 15-minute round is soft: material gain or compensation progress can
justify a measured extension. Checkpoints survive every tool invocation.
"""
import runtime
import argparse,json,time
from pathlib import Path
from hybrid_agent import Session,lease,read,save,action_fingerprint
from hybrid_provider import choose
from hybrid_selection import validation_portfolio,frontier_signature

def fallback(session,reason):
 s=session.state;limit=session.limit;options=[]
 verified=[read(p)['area'] for p in (session.folder/'nodes').glob('*_verify1.json') if read(p)['passed']]
 incumbent=min(verified,default=session.manifest['original_area'])
 for node in s['active']:
  record=s['nodes'][node];p=session.folder/'nodes'/(node+'_candidates.json')
  children={r['sha256'] for r in s['nodes'].values() if r['parent']==node}
  if len(children)>=4:
   return dict(action='archive',node=node,options={},reason=reason+'; four distinct child implementations already measured, preserve their checkpoints and move on')
  validation=session.folder/'nodes'/f'{node}_verify1.json'
  needs_repair=record['train_errors']>round(limit*.85) or (validation.exists() and not read(validation)['passed'])
  if p.exists():
   data=read(p)
   for c in data['summary']:
    if c['er']>=record['train_er'] and (needs_repair or c['area_gain']<=0) and not (c['er']==record['train_er'] and c.get('hamming_gain',0)>0):continue
    headroom=max(64,round((incumbent-record['area'])/.266))
    debt=s.get('fallback_debt_units',min(headroom,session.manifest['repair_debt_limit_units']))
    if c['result_area']>session.lib.area(record['units']+debt):continue
    if not c['feasible'] and not needs_repair:continue
    source=next(m for m in data['moves'] if m['id']==c['id'])
    fingerprint=action_fingerprint([source])
    if any(r['parent']==node and r['origin'].get('action_fingerprint')==fingerprint for r in s['nodes'].values()):continue
    if any(r['parent']==node and r['fingerprint']==fingerprint for r in s.get('redundant_actions',[])):continue
    pressure=c['er']*10000
    repair_score=c['result_area']+session.manifest['original_area']*.0001*max(0,pressure-.9*limit)
    options.append(((0 if needs_repair else 1,0 if c['feasible'] else 1,repair_score if needs_repair else c['result_area'],pressure),node,c))
 if options:
  _,node,c=min(options,key=lambda x:x[0]);evict=None
  if len(s['active'])>=4:evict=max((n for n in s['active'] if n!=node),key=lambda n:s['nodes'][n]['area'])
  args=dict(ids=[c['id']],fork=True,allow_near_miss=not c['feasible'])
  if evict:args['evict']=evict
  return dict(action='apply',node=node,options=args,reason=reason+'; select a measured feasible improvement or error-reducing repair')
 # Only screened regions, with a bounded construction width. Distinct measured
 # regions and policies are tools, not an exhaustive LAC candidate enumeration.
 for node in sorted(s['active'],key=lambda n:(s['nodes'][n]['area'],s['nodes'][n]['train_er'])):
  previous=s['nodes'][node]['expanded']
  for mode,width,offset in [('repair',6,0),('repair',6,32),('efficiency',4,0),('area',4,0),('efficiency',6,32),('area',6,32)]:
   if any(e['mode']==mode and e['width']==width and e['offset']==offset for e in previous):continue
   return dict(action='scan',node=node,options=dict(mode=mode,width=width,offset=offset,roots=32),reason=reason+'; measure the next targeted region')
 revisited={d['node'] for d in s['decisions'] if d['action']=='revisit'}
 if len(revisited)<4:
  for node in sorted(validation_portfolio(s,limit),key=lambda n:(s['nodes'][n]['train_errors'],s['nodes'][n]['units'])):
   if node in s['active'] or node in revisited:continue
   children={r['sha256'] for r in s['nodes'].values() if r['parent']==node}
   if len(children)>=4 or len(s['nodes'][node]['expanded'])>=6:continue
   args={}
   if len(s['active'])>=4:args['evict']=max(s['active'],key=lambda n:s['nodes'][n]['area'])
   return dict(action='revisit',node=node,options=args,reason=reason+'; try a preserved low-error prefix after the active screened regions plateau')
 return dict(action='finish',options={},reason=reason+'; this bounded screened-region queue and prefix revisits have no unmeasured tool or feasible improvement')

def finalize(session):
 for node in validation_portfolio(session.state,session.limit):
  if not (session.folder/'nodes'/f'{node}_verify1.json').exists():session.verify(node,1)

def round_attribution(decisions,numeric_mode_entered):
 numeric=sum(d['source']=='numeric_fallback' for d in decisions)
 return dict(numerical_fallback_used=numeric>0,numeric_mode_entered=numeric_mode_entered,
  model_decisions_this_round=sum(d['source']=='siliconflow_agent' for d in decisions),numeric_decisions_this_round=numeric)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',required=True);p.add_argument('--numeric',action='store_true');p.add_argument('--once',action='store_true')
 p.add_argument('--round-seconds',type=float,default=900);p.add_argument('--extension-seconds',type=float,default=900);args=p.parse_args()
 folder=Path(args.session).resolve()
 with lease(folder):
  session=Session(folder);started=time.perf_counter();deadline=args.round_seconds;extended=False;numeric=args.numeric;stalled=0;schema_failures=0;last_error=None
  decision_start=len(session.state['decisions'])
  baseline=session.state['nodes'][session.state['best_train']]['units'] if session.state['best_train'] else session.original.units
  progress=frontier_signature(session.state,session.limit,session.original.units);last_progress=0
  while session.state['status']!='agent_finished':
   elapsed=time.perf_counter()-started
   current=frontier_signature(session.state,session.limit,session.original.units)
   thresholds=[max(2,session.original.units//400)]*5+[max(3,session.limit//100)]*5
   if any(a-b>=t for a,b,t in zip(progress,current,thresholds)):
    progress=current;last_progress=elapsed
   if numeric and elapsed-last_progress>=120:
    save(folder/'plateau_checkpoint.json',dict(reason='no material global frontier progress for 120 seconds',elapsed=elapsed,signature=current))
    break
   if elapsed>=deadline:
    now=session.state['nodes'][session.state['best_train']]['units'] if session.state['best_train'] else baseline
    recent=[r for r in session.state['nodes'].values() if r['parent'] is not None and r['created_at']>=time.time()-300]
    recovery=any(session.state['nodes'][r['parent']]['train_errors']-r['train_errors']>=max(2,session.limit//50)
     and r['area']<session.manifest['original_area']*.9 for r in recent)
    if not extended and args.extension_seconds>0 and (baseline-now>=max(2,session.original.units//200) or recovery):
     deadline+=args.extension_seconds;extended=True
     save(folder/'time_extension.json',dict(reason='measured area gain or compensation progress',elapsed=elapsed,extra_seconds=args.extension_seconds,baseline_units=baseline,current_units=now))
    else:
     save(folder/'round_checkpoint.json',dict(reason='soft_time_boundary_with_no_material_extension',elapsed=elapsed,status='awaiting_agent',best_train=session.state['best_train']));break
   observation=session.write_observation()
   if last_error:observation['last_tool_error']=last_error
   if not numeric:
    try:decision=choose(observation)
    except Exception as exc:
     reason=str(exc);numeric=True;save(folder/'fallback_event.json',dict(reason=reason,time=time.time(),source='numeric_fallback'))
     print('[CONTROLLER SWITCH] '+reason,flush=True)
   if numeric:
    decision=fallback(session,'Authorized numerical continuation after a provider issue or measured plateau' if not args.numeric else 'Codex-authorized targeted numerical experiment')
    decision.update(decision_id=session.state['next_decision'],source='numeric_fallback')
   attempt=str(time.time_ns());save(folder/f'controller_request_{session.state["next_decision"]:05d}_{attempt}.json',decision)
   before=session.state['next_node']
   try:session.execute(decision)
   except Exception as exc:
    last_error=dict(reason=str(exc),decision=decision)
    if hasattr(exc,'measurement'):last_error['measurement']=exc.measurement
    if hasattr(exc,'tool_feedback'):last_error['legal_next_steps']=exc.tool_feedback
    save(folder/f'tool_error_{session.state["next_decision"]:05d}_{attempt}.json',last_error)
    session.commit();print('[TOOL REJECTED] '+str(exc),flush=True);session=Session(folder)
    if numeric:break
    schema_failures+=1
    if schema_failures>=2:numeric=True
    continue
   schema_failures=0;last_error=None
   stalled=stalled+1 if session.state['next_node']==before else 0
   if stalled>=4 and time.perf_counter()-started>30 and not numeric:
    numeric=True;save(folder/'plateau_switch.json',dict(reason='four tools without an accepted state; try targeted numerical repair',time=time.time()))
   if args.once:break
  finalize(session)
  save(folder/'controller_round.json',dict(seconds=time.perf_counter()-started,extended=extended,
   provider='DeepSeek-V4-Flash',status=session.state['status'],best_train=session.state['best_train'],
   **round_attribution(session.state['decisions'][decision_start:],numeric)))
if __name__=='__main__':main()
