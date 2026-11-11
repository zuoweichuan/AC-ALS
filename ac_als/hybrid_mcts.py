"""Standard UCT MCTS over four screened actions, with bounded local rollouts.

Separate numerical diagnostic/controller: selection, expansion, simulation and
backpropagation are explicit. Neither tree visits nor truth scoring are counted
as Monte Carlo circuit propagations. Existing Agent/default entries are intact.
"""
import argparse,math,time
from pathlib import Path
from hybrid_agent import Session,lease,read,save,action_fingerprint
from hybrid_selection import frontier_signature
from hybrid_controller import finalize

class MCTS4:
 def __init__(self,nodes=None,exploration=.7):
  self.nodes=nodes or {};self.exploration=exploration
  if 'root' not in self.nodes:self.add('root',None)
 def add(self,node,parent):
  if node in self.nodes:return
  if parent is not None and len(self.nodes[parent]['children'])>=4:raise ValueError('MCTS has at most four children per node')
  self.nodes[node]=dict(parent=parent,children=[],visits=0,value_sum=0.,actions=None,closed=False)
  if parent is not None:self.nodes[parent]['children'].append(node)
 def select(self,max_depth=8):
  path=['root']
  while path:
   node=path[-1];r=self.nodes[node]
   if r['closed']:
    path.pop();continue
   if node!='root' and len(path)>=max_depth+1:
    r['closed']=True;path.pop();continue
   if node!='root' and (r['actions'] is None or r['actions']):return path
   choices=[n for n in r['children'] if not self.nodes[n]['closed']]
   if not choices:
    r['closed']=True;path.pop();continue
   def uct(n):
    c=self.nodes[n]
    return float('inf') if c['visits']==0 else c['value_sum']/c['visits']+self.exploration*math.sqrt(math.log(max(2,r['visits']))/c['visits'])
   child=max(choices,key=uct)
   if len(path)>=max_depth+1:
    self.nodes[child]['closed']=True;continue
   path.append(child)
  return None
 def backpropagate(self,path,reward):
  if len(path)!=len(set(path)):raise ValueError('A tree path cannot repeat a node')
  for node in path:
   self.nodes[node]['visits']+=1;self.nodes[node]['value_sum']+=reward

def diverse_actions(data,limit,parent_errors,original_area):
 """Four choices from scored LACs, preserving error-budget and mask diversity."""
 moves={m['id']:m for m in data['moves']};rows=[]
 for c in data['summary']:
  if c['area_gain']<=0 and not (round(c['er']*10000)<parent_errors):continue
  if c['er']*10000>1.5*limit:continue
  r=dict(c);r['fingerprint']=action_fingerprint([moves[c['id']]]);r['bad']=int(moves[c['id']]['bad_hex'],16);rows.append(r)
 selected=[]
 def different(r):
  for old in selected:
   if r['fingerprint']==old['fingerprint']:return False
   if set(r['targets'])==set(old['targets']):
    union=(r['bad']|old['bad']).bit_count();distance=(r['bad']^old['bad']).bit_count()/max(1,union)
    if distance<.15 and abs(r['result_area']-old['result_area'])<max(.532,original_area*.01):return False
  return True
 def pick(candidates,key):
  for r in sorted(candidates,key=key):
   if different(r):selected.append(r);return
 feasible=[r for r in rows if r['feasible'] and r['area_gain']>0]
 pick(feasible,lambda r:(r['result_area'],r['er']))
 pick(feasible,lambda r:(-r['area_gain']/(25+r['new_errors']),r['er']))
 margin=min(.75*limit,parent_errors+.2*limit)
 pick([r for r in feasible if r['er']*10000<=margin],lambda r:(r['result_area'],r['er']))
 if len(selected)<3:pick([r for r in rows if r['er']*10000<parent_errors],lambda r:(r['result_area']+.03*original_area*r['er']*10000/max(1,limit),r['er']))
 pick([r for r in rows if not r['feasible'] and r['area_gain']>0],lambda r:(r['excess_errors']/max(.266,r['area_gain']),r['result_area']))
 while len(selected)<4:
  remaining=[r for r in rows if different(r)]
  if not remaining:break
  pick(remaining,lambda r:(-min(((r['bad']^o['bad']).bit_count() for o in selected),default=0),r['result_area'],r['er']))
 return [dict(id=r['id'],fingerprint=r['fingerprint'],targets=r['targets'],er=r['er'],result_area=r['result_area']) for r in selected[:4]]

def run(args):
 folder=Path(args.session).resolve()
 with lease(folder):
  session=Session(folder);path=folder/'mcts_tree.json'
  settings=dict(branches=4,exploration=.7,max_depth=8,roots=16,expansion_width=6,rollout_steps=2,rollout_roots=8,reward_margin_penalty=.02)
  if path.exists():
   checkpoint=read(path)
   if checkpoint['settings']!=settings:raise RuntimeError('Frozen MCTS settings changed')
   tree=MCTS4(checkpoint['nodes']);iterations=checkpoint['iterations']
  else:
   tree=MCTS4();iterations=0
   for node,r in session.state['nodes'].items():
    if r['parent'] is None:tree.add(node,'root')
  started=time.perf_counter();last_progress=0;signature=frontier_signature(session.state,session.limit,session.original.units);reason='time_boundary'
  def execute(action,node=None,options=None,stage='selection'):
   d=dict(decision_id=session.state['next_decision'],source='numeric_fallback',search_algorithm='MCTS4',mcts_iteration=iterations,mcts_stage=stage,
    action=action,options=options or {},reason='Standard four-branch UCT MCTS: '+stage)
   if node is not None:d['node']=node
   session.execute(d);return session.state['decisions'][-1]['result']
  def activate(node):
   if node in session.state['active']:return
   options={}
   if len(session.state['active'])>=4:options['evict']=max(session.state['active'],key=lambda n:session.state['nodes'][n]['area'])
   execute('revisit',node,options,'restore_selected_tree_prefix')
  def reward(node):
   r=session.state['nodes'][node];pressure=r['train_errors']/max(1,session.limit)
   return 1-r['units']/session.original.units-.02*pressure**2 if pressure<=1 else -.01*(pressure-1)
  while time.perf_counter()-started<args.round_seconds:
   selected=tree.select(settings['max_depth'])
   if selected is None:reason='screened_tree_exhausted';break
   node=selected[-1];record=tree.nodes[node];activate(node)
   if record['actions'] is None:
    repairing=session.state['nodes'][node]['train_errors']>session.limit
    execute('scan',node,dict(mode='repair' if repairing else 'efficiency',width=settings['expansion_width'],roots=settings['roots']),'screen_expansion_actions')
    data=read(folder/'nodes'/f'{node}_candidates.json')
    record['actions']=diverse_actions(data,session.limit,session.state['nodes'][node]['train_errors'],session.manifest['original_area'])
   if not record['actions']:
    record['closed']=not record['children'];tree.backpropagate(selected,reward(node));iterations+=1
   else:
    action=record['actions'].pop(0);options=dict(ids=[action['id']],fork=True,allow_near_miss=action['er']>session.state['er_limit'])
    if len(session.state['active'])>=4:options['evict']=max((n for n in session.state['active'] if n!=node),key=lambda n:session.state['nodes'][n]['area'])
    data=read(folder/'nodes'/f'{node}_candidates.json');row=next(r for r in data['moves'] if r['id']==action['id'])
    cached=session.measured_child(node,[row])
    if cached is not None:activate(cached);result=cached
    else:result=execute('apply',node,options,'expansion')
    if isinstance(result,str):
     child=result;tree.add(child,node);selected.append(child);simulation=child
     near=session.state['nodes'][child]['train_errors']>session.limit
     simulation_result=execute('rollout',child,dict(steps=2,width=6 if near else 4,roots=8,mode='repair' if near else 'efficiency',
      policy='error' if near else 'efficiency',area_debt_units=session.manifest['repair_debt_limit_units'] if near else 0,repair_error_target=round(.85*session.limit)),'bounded_simulation_rollout')
     simulation=simulation_result['last_node'];value=reward(simulation)
    else:value=reward(node)
    tree.backpropagate(selected,value);iterations+=1
   save(path,dict(settings=settings,nodes=tree.nodes,iterations=iterations))
   current=frontier_signature(session.state,session.limit,session.original.units)
   thresholds=[max(2,session.original.units//400)]*5+[max(3,session.limit//100)]*5
   if any(a-b>=t for a,b,t in zip(signature,current,thresholds)):signature=current;last_progress=time.perf_counter()-started
   if time.perf_counter()-started-last_progress>=120:reason='no_material_frontier_progress_120s';break
  finalize(session)
  report=dict(algorithm='MCTS4',settings=settings,iterations=iterations,root_visits=tree.nodes['root']['visits'],
   tree_expansions=sum(len(r['children']) for n,r in tree.nodes.items() if n!='root'),
   bounded_rollouts=sum(d.get('mcts_stage')=='bounded_simulation_rollout' for d in session.state['decisions']),
   seconds=time.perf_counter()-started,stop_reason=reason,numerical_fallback_used=True,provider=None,status=session.state['status'],best_train=session.state['best_train'])
  save(folder/'mcts_round.json',report);save(folder/'controller_round.json',report)
  print('[MCTS4 ROUND] '+str(report),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',required=True);p.add_argument('--round-seconds',type=float,default=300)
 p.add_argument('--extension-seconds',type=float,default=0);args=p.parse_args()
 if args.extension_seconds:raise ValueError('Use an explicit new numerical round duration; MCTS diagnostic has no automatic time extension')
 run(args)
