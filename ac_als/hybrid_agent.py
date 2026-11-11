"""Durable Agent decisions over screened numerical tools; no automatic fallback.

The Agent (Codex or a provider adapter) chooses the branch, candidate, search
family, local rollout and stopping action. Workers only execute that decision.
"""
import runtime
import argparse,hashlib,json,os,shutil,subprocess,time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from logic import Library,Plan
from netlist import Circuit,Gate
from cutmap import CutMapper
from frontier import candidates,normalize,Move
from joint import JointMove,joint_candidates
from screen import RootScreen
from candidate_cache import CandidateCache
from simulate import trace,pack,outputs,wrong,validate_numpy
from hybrid_identity import structural_id

CODE_ROOT=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('NETLIST_FRONTIER_ROOT',str(CODE_ROOT))).resolve()
DESIGNS=('c880','c1908','c2670','c3540','c5315','c7552','c1355')
TARGETS={'c880':(.177,.236),'c1355':(.037,.130),'c1908':(.067,.620),
 'c2670':(.327,.368),'c3540':(.029,.220),'c5315':(.021,.075),'c7552':(.183,.192)}

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
 path=Path(path);temporary=path.with_name(path.name+'.tmp')
 temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 for attempt in range(12):
  try:os.replace(temporary,path);return
  except PermissionError:
   if attempt==11:raise
   time.sleep(.02*(attempt+1))
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def matrix(design,inputs,split):
 return np.random.default_rng(np.random.SeedSequence([20261025,DESIGNS.index(design),split])).integers(0,2,(10000,len(inputs)),dtype=np.uint8)
def plan_data(p):return dict(name=p.name,leaf=p.leaf,units=p.units,children=[plan_data(c) for c in p.children])
def plan_read(p):return Plan(p['name'],tuple(plan_read(c) for c in p['children']),leaf=p['leaf'],units=p['units'])
def move_data(m):
 r=m.record();r['bad_hex']=hex(m.bad)
 if hasattr(m,'components'):
  r['components_data']=[move_data(c) for c in m.components];return r
 r['plans']=[plan_data(p) for p in m.plans] if hasattr(m,'plans') else [plan_data(m.plan)]
 if hasattr(m,'extra_gates'):r['extra_gates']=[dict(name=g.name,cell=g.cell,pins=g.pins) for g in m.extra_gates]
 return r
def move_read(r):
 if 'components_data' in r:
  from targeted_repair import CompoundMove
  return CompoundMove(tuple(move_read(c) for c in r['components_data']),r['gain_units'],r['error'],int(r['bad_hex'],16),r.get('hamming_gain',0),r.get('family','targeted_compound_repair'))
 if 'targets' in r:return JointMove(tuple(r['targets']),tuple(r['inputs']),tuple(r['truths']),tuple(plan_read(p) for p in r['plans']),r['gain_units'],r['error'],int(r['bad_hex'],16))
 if 'extra_gates' in r:
  from targeted_repair import RestoredMove
  return RestoredMove(r['target'],tuple(r['inputs']),r['truth'],plan_read(r['plans'][0]),r['gain_units'],int(r['bad_hex'],16),r['error'],r['family'],tuple(Gate(g['name'],g['cell'],g['pins']) for g in r['extra_gates']),r.get('hamming_gain',0))
 return Move(r['target'],tuple(r['inputs']),r['truth'],plan_read(r['plans'][0]),r['gain_units'],int(r['bad_hex'],16),r['error'],r['family'])
def action_fingerprint(rows):
 data=[{k:r[k] for k in ('target','targets','inputs','truth','truths','plans','extra_gates','components_data') if k in r} for r in rows]
 return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()
class RedundantAction(ValueError):pass
class CandidateSelectionRejected(ValueError):
 def __init__(self,reason,feedback):
  self.tool_feedback=feedback
  super().__init__(reason)
class MeasuredActionRejected(ValueError):
 def __init__(self,reason,measurement):
  self.measurement=measurement
  super().__init__(reason+'; measured '+json.dumps(measurement,sort_keys=True))
def live(pid):
 if os.name=='nt':
  p=subprocess.run(['tasklist','/FI',f'PID eq {pid}','/FO','CSV','/NH'],capture_output=True,text=True)
  return f'"{pid}"' in p.stdout
 try:os.kill(pid,0);return True
 except ProcessLookupError:return False
@contextmanager
def lease(folder):
 lock=folder/'worker.lock'
 if lock.exists():
  pid=read(lock)['pid']
  if live(pid):raise RuntimeError(f'An existing worker is active: PID {pid}')
  lock.unlink()
 with lock.open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),started=time.time()),f)
 try:yield
 finally:lock.unlink(missing_ok=True)

def init(args):
 folder=Path(args.output).resolve();folder.mkdir(parents=True,exist_ok=False);(folder/'nodes').mkdir();(folder/'source').mkdir()
 lib=Library(ROOT/'assets/stdcells.lib');original=Circuit.read(ROOT/'assets'/f'{args.design}.v',lib)
 code={p.name:digest(p) for p in CODE_ROOT.glob('*.py')}
 for name in code:shutil.copy2(CODE_ROOT/name,folder/'source'/name)
 manifest=dict(design=args.design,er_limit=args.er,search_samples=10000,validation_samples=10000,branch_slots=4,
  code=code,original_sha256=digest(ROOT/'assets'/f'{args.design}.v'),library_sha256=digest(ROOT/'assets/stdcells.lib'),
  original_area=lib.area(original.units),original_units=original.units,global_numerical_budget=None,
  exhaustive_lacs=False,reference_targets=TARGETS[args.design],agent_primary=getattr(args,'controller','agent')=='agent',search_controller=getattr(args,'controller','agent'),numerical_fallback_authorized=True,
  repair_debt_limit_units=min(384,max(16,original.units//8)),default_soft_round_seconds=900)
 manifest['simulation_counter_schema']='packed_trace_and_separate_numpy_validation_v1'
 knowledge=ROOT/'hybrid_knowledge.json'
 if knowledge.exists():
  shutil.copy2(knowledge,folder/'knowledge_snapshot.json');manifest['knowledge_sha256']=digest(folder/'knowledge_snapshot.json')
 save(folder/'manifest.json',manifest)
 state=dict(version=1,design=args.design,er_limit=args.er,active=[],nodes={},best_train=None,
  next_node=0,next_decision=0,decisions=[],counters={},status='awaiting_agent',observations=0)
 save(folder/'state.json',state)
 session=Session(folder)
 session.add_node(normalize(original),None,dict(kind='original_baseline'))
 for source in args.start:
  c=Circuit.read(source,lib)
  if c.inputs!=original.inputs or c.outputs!=original.outputs:raise ValueError('Warm start interface differs')
  session.add_node(normalize(c),None,dict(kind='measured_warm_start',path=str(Path(source).resolve()),sha256=digest(source)))
 session.commit();session.write_observation()
 print('[HYBRID READY] '+str(folder),flush=True)

class Session:
 def __init__(self,folder):
  self.folder=Path(folder).resolve();self.manifest=read(self.folder/'manifest.json');self.state=read(self.folder/'state.json')
  for name,h in self.manifest['code'].items():
   if not (CODE_ROOT/name).exists() or digest(CODE_ROOT/name)!=h:raise RuntimeError('Worker source differs from its snapshot; use hybrid_launch.py to resume frozen code or create a new version explicitly')
  self.lib=Library(ROOT/'assets/stdcells.lib');self.design=self.manifest['design'];self.limit=int(round(self.state['er_limit']*10000))
  if digest(ROOT/'assets'/f'{self.design}.v')!=self.manifest['original_sha256'] or digest(ROOT/'assets/stdcells.lib')!=self.manifest['library_sha256']:raise RuntimeError('Frozen input changed')
  if self.manifest.get('knowledge_sha256') and digest(self.folder/'knowledge_snapshot.json')!=self.manifest['knowledge_sha256']:raise RuntimeError('Frozen knowledge changed')
  self.original=Circuit.read(ROOT/'assets'/f'{self.design}.v',self.lib)
  self.inputs=pack(matrix(self.design,self.original.inputs,0),self.original.inputs)
  self.counters=self.state['counters'];self.ref=outputs(self.original,trace(self.original,self.inputs,10000,self.counters))
  self.mapper=None;self.cache=CandidateCache();self.response_contexts={}
  if (self.folder/'templates.json').exists():self.cache.add([move_read(x) for x in read(self.folder/'templates.json')])
 def commit(self):save(self.folder/'state.json',self.state)
 def circuit(self,node):return Circuit.read(self.folder/self.state['nodes'][node]['netlist'],self.lib)
 def add_node(self,circuit,parent,origin,*,measured_bad=None):
  if len(self.state['active'])>=4:raise ValueError('Four active branch slots are occupied; the Agent must evict one explicitly')
  # apply() has just propagated this same trial. Reuse that exact result rather
  # than running the identical Monte Carlo simulation again to store the node.
  if measured_bad is None:bad=wrong(circuit,trace(circuit,self.inputs,10000,self.counters),self.ref)
  else:bad=measured_bad
  node=f'n{self.state["next_node"]:05d}';self.state['next_node']+=1
  path=self.folder/'nodes'/(node+'.v');circuit.write(path)
  record=dict(id=node,parent=parent,netlist=str(path.relative_to(self.folder)),sha256=digest(path),structural_id=structural_id(circuit),units=circuit.units,
   area=self.lib.area(circuit.units),train_errors=bad.bit_count(),train_er=bad.bit_count()/10000,origin=origin,expanded=[],created_at=time.time())
  self.state['nodes'][node]=record;self.state['active'].append(node)
  if bad.bit_count()<=self.limit:
   best=self.state['best_train']
   if best is None or (record['units'],record['train_errors'])<(self.state['nodes'][best]['units'],self.state['nodes'][best]['train_errors']):
    self.state['best_train']=node;shutil.copy2(path,self.folder/'best_train.v')
  return node,bad
 def write_observation(self):
  evidence=[]
  for node in self.state['active']:
   r=self.state['nodes'][node];item={k:r[k] for k in ('id','parent','area','train_er','expanded','origin')}
   validation=self.folder/'nodes'/f'{node}_verify1.json'
   if validation.exists():item['development_validation']=read(validation)
   p=self.folder/'nodes'/(node+'_candidates.json')
   if p.exists():
    packdata=read(p)
    if packdata['scan']['parent_sha256']==r['sha256']:
     measured={n['origin'].get('action_fingerprint'):child for child,n in self.state['nodes'].items() if n['parent']==node}
     used={m['id']:measured[action_fingerprint([m])] for m in packdata['moves'] if action_fingerprint([m]) in measured}
     item['scan']=packdata['scan'];item['choices']=[c for c in packdata['summary'] if c['id'] not in used]
     item['known_transitions']=[dict(candidate_id=cid,child=child,area=self.state['nodes'][child]['area'],train_er=self.state['nodes'][child]['train_er'],active=child in self.state['active']) for cid,child in used.items()]
   item['legal_actions']=['scan','rollout','validate','archive']+(['apply'] if item.get('choices') else [])
   item['candidate_scope']=dict(node=node,parent_sha256=r['sha256'],scan_id=item.get('scan',{}).get('scan_id'),
    ready=bool(item.get('choices')),rule='Only IDs displayed for this node and its latest scan are valid. A child requires its own scan or rollout.')
   evidence.append(item)
  knowledge_path=self.folder/'knowledge_snapshot.json' if self.manifest.get('knowledge_sha256') else ROOT/'hybrid_knowledge.json'
  knowledge=read(knowledge_path) if knowledge_path.exists() else {}
  development=[read(p) for p in (self.folder/'nodes').glob('*_verify1.json') if read(p)['passed']]
  from hybrid_selection import repair_economics
  qualified=[r for r in development if r['er']<=self.state['er_limit']*.9]
  reference_node=min(qualified,key=lambda r:(r['area'],r['er'],r['node']))['node'] if qualified else self.state['best_train']
  if reference_node is not None:
   reference=self.state['nodes'][reference_node]
   for item in evidence:
    item['repair_economics']=repair_economics(self.state['nodes'][item['id']],reference,self.limit,self.manifest['repair_debt_limit_units'])
    item['repair_economics']['reference_basis']='development_90pct_margin' if qualified else 'training_feasible'
  from hybrid_selection import validation_portfolio
  archived=[n for n in validation_portfolio(self.state,self.limit) if n not in self.state['active']]
  archived=[{k:self.state['nodes'][n][k] for k in ('id','parent','area','train_er')} for n in archived]
  obs=dict(session=str(self.folder),decision_id=self.state['next_decision'],design=self.design,er_limit=self.state['er_limit'],
   max_repair_debt_units=self.manifest['repair_debt_limit_units'],
   exploration_ceiling_max=min(1.,self.state['er_limit']*2+.005),
   original_area=self.manifest['original_area'],physical_area_per_unit=self.lib.area(1),suggested_margin_errors=round(self.limit*.85),archived_frontier=archived,best_train=self.state['best_train'],best_development=min(development,key=lambda r:r['area']) if development else None,branches=evidence,
   counters=self.counters,recent_decisions=self.state['decisions'][-6:],knowledge=knowledge,
   legal_actions=['scan','apply','rollout','validate','archive','revisit','finish'],
   protocol='Choose a branch and bounded tool; inspect near misses and repairs before stopping. Four active slots. Revisit can restore an archived prefix. A rollout is a local tool, not the global controller. Numerical continuation is explicitly labelled if used.')
  save(self.folder/'observation.json',obs);self.state['observations']+=1;self.commit();return obs
 def scan(self,node,*,width=4,offset=0,roots=32,mode='efficiency',ceiling=None,joint=False,targets=None,restored=True,bundled=True):
  circuit=self.circuit(node);ceiling=ceiling if ceiling is not None else max(self.state['er_limit']*1.5,self.state['er_limit']+.002)
  if not self.state['er_limit']<=ceiling<=min(1.,self.state['er_limit']*2+.005):raise ValueError('Exploration ceiling outside protocol')
  if width not in (4,6) or not 1<=roots<=64 or offset<0:raise ValueError('Unbounded construction request')
  if self.mapper is None or self.mapper.maximum!=width:self.mapper=CutMapper(self.lib,width)
  # Node IDs are immutable, but also bind the cache to the measured file hash.
  # Keep only two contexts so four-branch exploration cannot grow memory without bound.
  identity=self.state['nodes'][node]['sha256'];context=self.response_contexts.pop(identity,None)
  if context:self.counters['response_context_reuses']=self.counters.get('response_context_reuses',0)+1
  cached,_=self.cache.refresh(circuit,self.inputs,10000,self.ref,self.mapper,self.counters,round(ceiling*10000),context=context)
  screen=RootScreen(circuit,self.inputs,10000,self.ref,self.counters,**(self.cache.context or context or {}))
  self.response_contexts[identity]=dict(base=screen.values,flip_errors=screen.responses,potentials=screen.potentials)
  while len(self.response_contexts)>2:self.response_contexts.pop(next(iter(self.response_contexts)))
  if mode=='repair':
   order=sorted(screen.order,key=lambda t:(-(screen.bad&~screen.responses[t]).bit_count()/(25+(screen.responses[t]&~screen.bad).bit_count()),-screen.potentials[t],t))
  elif mode=='area':order=sorted(screen.order,key=lambda t:(-screen.potentials[t],t))
  elif mode=='efficiency':order=screen.order
  else:raise ValueError('Unknown root category')
  if targets is not None:
   if not 1<=len(targets)<=64 or len(set(targets))!=len(targets) or any(t not in circuit.producer for t in targets):raise ValueError('Invalid Agent root selection')
  else:targets=order[offset:offset+roots]
  tick=time.perf_counter()
  shared_seeds=[]
  found,_=candidates(circuit,self.inputs,10000,self.ref,self.mapper,self.counters,ceiling=ceiling,enhanced=True,
   residual=width==6,patch=width==6,targets=targets,base=screen.values,flip_errors=screen.responses,potentials=screen.potentials,template_cache=self.cache,
   repair_debt_units=self.manifest['repair_debt_limit_units'] if mode=='repair' else 8,shared_seeds=shared_seeds if bundled else None)
  found+=cached
  if joint:found+=joint_candidates(circuit,self.inputs,10000,self.ref,self.mapper,self.counters,ceiling=ceiling)
  if restored and mode=='repair':
   from targeted_repair import repair_candidates
   if not hasattr(self,'original_values'):self.original_values=trace(self.original,self.inputs,10000,self.counters)
   found+=repair_candidates(circuit,self.original,self.original_values,screen,self.mapper,self.counters,targets,
    self.manifest['repair_debt_limit_units'],'hr_'+node)
  bundle_trials=[]
  if bundled:
   from coherent_bundle import bundle_candidates
   found+=bundle_candidates(circuit,found,screen,self.inputs,self.ref,self.counters,self.limit,round(ceiling*10000),self.manifest['repair_debt_limit_units'],shared_seeds=shared_seeds,trial_log=bundle_trials)
  self.cache.add([m for m in found if not hasattr(m,'extra_gates')]);save(self.folder/'templates.json',[move_data(m) for m in self.cache.pool.values()])
  # Keep area-rich, low-error and error-efficient alternatives, including repairs.
  orders=[lambda m:(-m.gain,m.error),lambda m:(m.error,-m.gain),
   lambda m:(-m.gain/(25+(m.bad&~screen.bad).bit_count()),m.error),
   lambda m:(max(0,m.error-self.limit)/max(1,m.gain),-m.gain)]
  kept=[];seen=set();scan_id=f'{node}_s{len(self.state["nodes"][node]["expanded"]):03d}'
  for ranking in orders:
   for m in sorted(found,key=ranking)[:16]:
    r=move_data(m);key=json.dumps(m.record(),sort_keys=True)
    if key not in seen:seen.add(key);r['id']=f'{scan_id}_c{len(kept):03d}';kept.append(r)
  summary=[]
  for r in kept:
   m=move_read(r);summary.append(dict(id=r['id'],targets=r.get('targets',[r.get('target')]),inputs=r['inputs'],family=r['family'],
    error_support_signature=hashlib.sha256(m.bad.to_bytes(1250,'little')).hexdigest()[:16],
    area_gain=self.lib.area(m.gain),result_area=self.lib.area(circuit.units-m.gain),er=m.error/10000,
    new_errors=(m.bad&~screen.bad).bit_count(),repaired_errors=(screen.bad&~m.bad).bit_count(),
    excess_errors=max(0,m.error-self.limit),feasible=m.error<=self.limit,hamming_gain=getattr(m,'hamming_gain',0)))
  scan=dict(node=node,scan_id=scan_id,parent_sha256=self.state['nodes'][node]['sha256'],width=width,offset=offset,roots=roots,
   scanned_roots=len(targets),total_roots=len(order),mode=mode,ceiling=ceiling,joint=joint,generation_seconds=time.perf_counter()-tick,
   joint_attempts=bundle_trials,
   root_evidence=[dict(target=t,cell=circuit.producer[t].cell,
    fanins={p:circuit.resolve(circuit.producer[t].pins[p]) for p in circuit.lib.cells[circuit.producer[t].cell].inputs},
    mffc_area=self.lib.area(screen.potentials[t]),level=circuit.level[t],
    flip_repairs=(screen.bad&~screen.responses[t]).bit_count(),flip_new_errors=(screen.responses[t]&~screen.bad).bit_count()) for t in targets])
  save(self.folder/'nodes'/(node+'_candidates.json'),dict(scan=scan,moves=kept,summary=summary))
  self.state['nodes'][node]['expanded'].append(scan);self.commit()
  return summary
 def apply(self,node,ids,*,fork=True,evict=None,allow_near_miss=False):
  path=self.folder/'nodes'/(node+'_candidates.json');parent=self.state['nodes'][node]
  feedback=dict(node=node,next_actions=['scan','rollout'],current_candidate_ids=[])
  if not path.exists():raise CandidateSelectionRejected('This branch has not been scanned; scan it or request a bounded rollout before applying.',feedback)
  data=read(path);feedback['current_candidate_ids']=[m['id'] for m in data['moves']];feedback['scan_id']=data['scan'].get('scan_id')
  if digest(self.folder/parent['netlist'])!=data['scan']['parent_sha256']:raise ValueError('Stale candidate parent')
  if not 1<=len(ids)<=6 or len(set(ids))!=len(ids):raise ValueError('Invalid joint action size')
  chosen=[next((m for m in data['moves'] if m['id']==i),None) for i in ids]
  if any(m is None for m in chosen):raise CandidateSelectionRejected('Candidate IDs do not belong to this branch and latest scan; choose its displayed IDs or rescan.',feedback)
  moves=[move_read(m) for m in chosen];circuit=self.circuit(node)
  records=[m.record() for m in moves]
  fingerprint=action_fingerprint(chosen)
  if any(r['parent']==node and r['origin'].get('action_fingerprint')==fingerprint for r in self.state['nodes'].values()):raise ValueError('This exact action on this parent was already measured')
  replacements=[r for m in moves for r in (m.replacements() if hasattr(m,'replacements') else [m.replacement()])]
  from targeted_repair import augment
  trial=normalize(augment(circuit,moves).materialize(replacements))
  temporary=self.folder/'candidate_identity.v';trial.write(temporary);trial_hash=digest(temporary);temporary.unlink()
  structure=structural_id(trial)
  equivalent=next((n for n,r in self.state['nodes'].items() if r['sha256']==trial_hash or r.get('structural_id')==structure),None)
  if equivalent is not None:
   self.state.setdefault('redundant_actions',[]).append(dict(parent=node,fingerprint=fingerprint,existing_node=equivalent,reason='identical physical netlist'))
   self.counters['known_physical_propagations_avoided']=self.counters.get('known_physical_propagations_avoided',0)+1
   self.commit();raise RedundantAction('Candidate already exists as physical netlist '+equivalent)
  words=trace(trial,self.inputs,10000,self.counters);bad=wrong(trial,words,self.ref)
  if len(moves)==1 and bad!=moves[0].bad:raise AssertionError('Current-parent response prediction mismatch')
  measurement=dict(parent=node,candidate_ids=ids,actual_area=self.lib.area(trial.units),actual_area_gain=self.lib.area(circuit.units-trial.units),
   actual_er=bad.bit_count()/10000,excess_errors=max(0,bad.bit_count()-self.limit),exploration_ceiling=data['scan']['ceiling'])
  if bad.bit_count()>self.limit and not allow_near_miss:raise MeasuredActionRejected('Agent did not authorize a near-miss branch; inspect the measured joint result and decide whether to resubmit with allow_near_miss=true',measurement)
  if bad.bit_count()>round(data['scan']['ceiling']*10000):raise MeasuredActionRejected('Combined action violates exploration ceiling; choose another combination or explicitly rescan within the allowed protocol',measurement)
  if evict is not None and (evict==node or evict not in self.state['active']):raise ValueError('Eviction must identify a different active branch')
  if fork and len(self.state['active'])-(evict is not None)>=4:raise ValueError('Four active branches; Agent must explicitly evict another branch or use fork=false')
  if not fork:self.state['active'].remove(node)
  if evict is not None:
   self.state['active'].remove(evict)
  child,_=self.add_node(trial,node,dict(kind='controller_selected_lac',controller_source=getattr(self,'actor','diagnostic_probe'),action_fingerprint=fingerprint,candidate_ids=ids,
   actions=[m.record() for m in moves],actual_train_er=bad.bit_count()/10000,allow_near_miss=allow_near_miss),measured_bad=bad)
  self.commit();return child
 def measured_child(self,node,rows,*,replace_parent=False):
  fingerprint=action_fingerprint(rows)
  child=next((n for n,r in self.state['nodes'].items() if r['parent']==node and r['origin'].get('action_fingerprint')==fingerprint),None)
  if child is None:return None
  if digest(self.folder/self.state['nodes'][child]['netlist'])!=self.state['nodes'][child]['sha256']:raise RuntimeError('Cached transition netlist changed')
  if replace_parent:
   if node not in self.state['active']:raise ValueError('Cached local transition requires an active parent')
   self.state['active'].remove(node)
   if child not in self.state['active']:self.state['active'].append(child)
  self.counters['known_transition_reuses']=self.counters.get('known_transition_reuses',0)+1
  return child
 def rollout(self,node,*,steps=4,width=4,policy='efficiency',mode='efficiency',roots=32,offset=0,area_debt_units=8,repair_error_target=None,error_fraction=.85,max_error_increment=.15):
  if not 1<=steps<=8 or not 0<=area_debt_units<=self.manifest['repair_debt_limit_units']:raise ValueError('Rollout tool exceeds its local scope')
  if not isinstance(offset,int) or offset<0:raise ValueError('Root region offset must be a nonnegative integer')
  if policy not in ('efficiency','area','error','budget'):raise ValueError('Unknown rollout policy')
  if not 0<error_fraction<=1 or not 0<max_error_increment<=1:raise ValueError('Budget fractions must be in (0,1]')
  repair_error_target=self.limit if repair_error_target is None else repair_error_target
  if not 0<=repair_error_target<=self.limit:raise ValueError('Invalid compensation error target')
  if policy=='budget' and mode=='repair':repair_error_target=min(repair_error_target,round(self.limit*error_fraction))
  if mode=='repair' and self.state['nodes'][node]['train_errors']<=repair_error_target:
   return dict(last_node=node,accepted=[],local_steps_requested=steps,local_stop='repair_target_already_met',policy=policy,
    repair_error_target=repair_error_target)
  accepted=[];initial=self.state['nodes'][node]['units'];attempts=0;duplicate_skips=0;stop='step_limit'
  for _ in range(steps):
   summary=self.scan(node,width=width,roots=roots,offset=offset,mode=mode)
   parent=self.state['nodes'][node];valid=[]
   for c in summary:
    improvement=c['area_gain']>0 and c['feasible']
    if policy=='budget':improvement=improvement and round(c['er']*10000)<=min(round(self.limit*error_fraction),parent['train_errors']+round(self.limit*max_error_increment))
    repair=(c['er']<parent['train_er'] or (c['er']==parent['train_er'] and c.get('hamming_gain',0)>0)) and c['result_area']<=self.lib.area(initial+area_debt_units)
    if (mode=='repair' and repair) or (mode!='repair' and improvement):valid.append(c)
   if not valid:stop='no_feasible_screened_candidate';break
   if policy=='error' or mode=='repair':key=lambda c:(c['er'],c['result_area'],c['id'])
   elif policy in ('area','budget'):key=lambda c:(c['result_area'],c['er'],c['id'])
   elif policy=='efficiency':key=lambda c:(-c['area_gain']/(25+c['new_errors']),c['result_area'],c['er'],c['id'])
   else:raise ValueError('Unknown rollout policy')
   data=read(self.folder/'nodes'/f'{node}_candidates.json');child=None
   # A redundant undo is not evidence that this region has no useful move.
   # Try at most three ranked, already screened alternatives on this parent.
   for chosen in sorted(valid,key=key)[:3]:
    attempts+=1;row=next(r for r in data['moves'] if r['id']==chosen['id'])
    cached=self.measured_child(node,[row],replace_parent=True)
    if cached is not None:child=cached
    else:
     try:child=self.apply(node,[chosen['id']],fork=False,allow_near_miss=mode=='repair')
     except RedundantAction:duplicate_skips+=1;continue
    break
   if child is None:stop='screened_alternatives_redundant';break
   node=child
   accepted.append(node)
   if mode=='repair' and self.state['nodes'][node]['train_errors']<=repair_error_target:stop='repair_target_met';break
  return dict(last_node=node,accepted=accepted,local_steps_requested=steps,local_stop=stop,policy=policy,
   candidate_attempts=attempts,duplicate_skips=duplicate_skips,max_alternatives_per_step=3,
   root_region=dict(offset=offset,maximum_roots=roots,order=mode),
   budget=dict(error_fraction=error_fraction,max_error_increment=max_error_increment) if policy=='budget' else None,
   repair_error_target=repair_error_target if mode=='repair' else None)
 def execute(self,decision):
  if decision.get('decision_id')!=self.state['next_decision']:raise ValueError('Stale Agent decision ID')
  if decision.get('source') not in ('codex_agent','siliconflow_agent','numeric_fallback'):raise ValueError('An identified controller must make global decisions')
  if not decision.get('reason'):raise ValueError('Agent reasoning must be recorded')
  self.actor=decision['source']
  action=decision['action'];node=decision.get('node');options=decision.get('options',{})
  if action in ('scan','apply','rollout','archive') and node not in self.state['active']:raise ValueError('Inactive branch')
  tick=time.perf_counter()
  if action=='scan':result=self.scan(node,**options)
  elif action=='apply':
   try:result=self.apply(node,**options)
   except RedundantAction as exc:result=dict(skipped=True,reason=str(exc))
  elif action=='rollout':result=self.rollout(node,**options)
  elif action=='validate':
   if node not in self.state['nodes']:raise ValueError('Unknown validation node')
   path=self.folder/'nodes'/f'{node}_verify1.json';result=read(path) if path.exists() else self.verify(node,1)
  elif action=='archive':self.state['active'].remove(node);result=dict(archived=node)
  elif action=='revisit':
   if node not in self.state['nodes'] or node in self.state['active']:raise ValueError('Revisit requires an archived node')
   if digest(self.folder/self.state['nodes'][node]['netlist'])!=self.state['nodes'][node]['sha256']:raise ValueError('Archived netlist changed')
   evict=options.get('evict')
   if evict is not None and evict not in self.state['active']:raise ValueError('Invalid revisit eviction')
   if len(self.state['active'])-(evict is not None)>=4:raise ValueError('Revisit needs one free branch slot or an explicit eviction')
   if evict is not None:self.state['active'].remove(evict)
   self.state['active'].append(node);result=dict(revisited=node,evicted=evict)
  elif action=='finish':self.state['status']='agent_finished';result=dict(best_train=self.state['best_train'])
  else:raise ValueError('Unknown global action')
  record=dict(**decision,result=result,seconds=time.perf_counter()-tick,finished_at=time.time())
  self.state['decisions'].append(record);self.state['next_decision']+=1
  if action!='finish':self.state['status']='awaiting_agent'
  save(self.folder/f'decision_{decision["decision_id"]:05d}.json',record);self.commit();self.write_observation()
  print('[AGENT TOOL DONE] '+json.dumps(dict(action=action,decision_id=decision['decision_id'],seconds=record['seconds'],status=self.state['status'])),flush=True)
 def verify(self,node,split=1):
  if split not in (1,2):raise ValueError('Use development split 1 or fresh final split 2')
  circuit=self.circuit(node);x=matrix(self.design,self.original.inputs,split)
  expected=validate_numpy(self.original,x);actual=validate_numpy(circuit,x);bitwords=outputs(circuit,trace(circuit,pack(x,circuit.inputs),10000,self.counters))
  self.counters['numpy_validation_simulations']=self.counters.get('numpy_validation_simulations',0)+2
  self.counters['numpy_validation_input_vectors']=self.counters.get('numpy_validation_input_vectors',0)+20000
  self.counters['numpy_validation_gate_evaluations']=self.counters.get('numpy_validation_gate_evaluations',0)+len(self.original.gates)+len(circuit.gates)
  for j,n in enumerate(circuit.outputs):
   if bitwords[n]!=int.from_bytes(np.packbits(actual[:,j],bitorder='little').tobytes(),'little'):raise AssertionError('Independent simulators disagree')
  er=float(np.any(actual!=expected,axis=1).mean());record=dict(node=node,split=split,samples=10000,area=self.lib.area(circuit.units),
   reduction=1-circuit.units/self.original.units,er=er,passed=er<=self.state['er_limit'],sha256=digest(self.folder/self.state['nodes'][node]['netlist']),independent_simulators_agree=True)
  save(self.folder/'nodes'/f'{node}_verify{split}.json',record);self.commit();self.write_observation();print('[HYBRID VERIFY] '+json.dumps(record),flush=True);return record

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
 q=sub.add_parser('init');q.add_argument('--design',choices=DESIGNS,required=True);q.add_argument('--er',type=float,choices=(.05,.005),required=True);q.add_argument('--start',nargs='*',default=[]);q.add_argument('--output',required=True);q.add_argument('--controller',choices=('agent','mcts','local'),default='agent')
 q=sub.add_parser('act');q.add_argument('--session',required=True);q.add_argument('--decision',required=True)
 q=sub.add_parser('observe');q.add_argument('--session',required=True)
 q=sub.add_parser('verify');q.add_argument('--session',required=True);q.add_argument('--node',required=True);q.add_argument('--split',type=int,default=1)
 args=p.parse_args()
 if args.command=='init':init(args);return
 folder=Path(args.session).resolve()
 with lease(folder):
  session=Session(folder)
  if args.command=='act':session.execute(read(args.decision))
  elif args.command=='observe':print(json.dumps(session.write_observation(),ensure_ascii=False))
  elif args.command=='verify':session.verify(args.node,args.split)
if __name__=='__main__':main()
