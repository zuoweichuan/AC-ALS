"""Input-index-guided bundles from an already screened LAC shortlist.

Proposal categories: overlapping error supports, possible cancellation, and
bounded shared-cone deletion when fresh neutral/costly seeds are supplied.
The latter estimate is deliberately only a proposal heuristic: errors on
different output bits do NOT cancel. Every retained bundle is fully propagated.
No candidate functions are enumerated here. At most six actual bundle tests.
"""
from batching import propose_batch
from targeted_repair import CompoundMove,augment
from frontier import normalize,forbidden_fanout
from screen import FanoutCones
from simulate import trace,wrong
from hybrid_identity import structural_id

def used_inputs(move):
 used=set()
 def visit(plan):
  if plan.name=='leaf':
   if plan.leaf>=0:used.add(move.inputs[plan.leaf])
  else:
   for child in plan.children:visit(child)
 visit(move.plan);return used

def plan_cost(circuit,plan):
 seen=set()
 def visit(p):
  if p.name=='leaf' or p in seen:return
  seen.add(p)
  for child in p.children:visit(child)
 visit(plan);return sum(circuit.lib.cells[p.name].units for p in seen)

def shared_groups(circuit,moves,counters,error_budget=None):
 """At most12 structural ancestors, not all candidate pairs.

 Look back four gate levels from measured targets, group by shared physical
 ancestor, and keep <=4 independent targets which stop using that ancestor.
 With a budget, retain one area-rich group plus the2/3/4-target prefixes of
 one error-guided chain: <=48 cheap group proxies/call, <=96 over two passes.
 Refcount deletion/rescue accounts for new retained inputs. It only ranks;
 normalization and actual joint vector propagation decide the final result.
 """
 cones=FanoutCones(circuit);groups={};ancestors={};dependencies={};supports={};tfo={}
 for m in moves:
  if not hasattr(m,'target') or getattr(m,'extra_gates',()) or m.target not in circuit.producer:continue
  if m.target not in ancestors:
   seen=set();pending=[(circuit.producer[m.target],0)]
   while pending:
    g,depth=pending.pop()
    if id(g) in seen:continue
    seen.add(id(g));dependencies[id(g)]=g
    if depth<4:
     pending.extend((circuit.producer[n],depth+1) for n in cones.dependencies[id(g)] if n in circuit.producer)
   ancestors[m.target]=seen
  key=(m.target,m.inputs,m.truth);supports[key]={circuit.resolve(n) for n in used_inputs(m)}
  for gid in ancestors[m.target]:
   if sum(cones.refs[n] for n in cones.output_nets[gid])>1:groups.setdefault(gid,{})[key]=m
 ranked=sorted((gid for gid,rows in groups.items() if len({m.target for m in rows.values()})>=2),
  key=lambda gid:(-circuit.lib.cells[dependencies[gid].cell].units*(len({m.target for m in groups[gid].values()})-1),dependencies[gid].name))[:12]
 found=[];seen_groups=set()
 for gid in ranked:
  g=dependencies[gid];blocked=forbidden_fanout(circuit,cones.output_nets[gid]);chosen=[];roots=set();sources=set();blocked_targets=set()
  eligible=[]
  for key,m in groups[gid].items():
   if not supports[key]&blocked:eligible.append((key,m))
  for key,m in sorted(eligible,key=lambda row:(-row[1].gain,row[1].error,row[0])):
   used=supports[key]
   if used&blocked or m.target in roots or m.target in blocked_targets:continue
   if m.target not in tfo:tfo[m.target]=forbidden_fanout(circuit,[m.target])
   region=tfo[m.target]
   if region&roots or sources&region or used&blocked_targets:continue
   chosen.append(m);roots.add(m.target);sources|=used;blocked_targets|=region
   if len(chosen)>=4:break
  variants=[chosen] if len(chosen)>=2 else []
  if error_budget is not None:
   # One directed low-error chain, not all subsets/pairs. Keep its2/3/4-root
   # prefixes plus the area-rich alternative above (<=4 proxies/ancestor).
   chain=[];roots=set();sources=set();blocked_targets=set();envelope=0
   while len(chain)<4:
    available=[]
    for key,m in eligible:
     used=supports[key]
     if m.target in roots or m.target in blocked_targets:continue
     if m.target not in tfo:tfo[m.target]=forbidden_fanout(circuit,[m.target])
     region=tfo[m.target]
     if region&roots or sources&region or used&blocked_targets:continue
     union=envelope|m.bad
     available.append(((max(0,union.bit_count()-error_budget),union.bit_count(),-m.gain,key),key,m))
    if not available:break
    _,key,m=min(available,key=lambda row:row[0]);chain.append(m);roots.add(m.target)
    sources|=supports[key];blocked_targets|=tfo[m.target];envelope|=m.bad
    if len(chain)>=2:variants.append(list(chain))
  for chosen in variants:
   identity=tuple(sorted((m.target,m.inputs,m.truth) for m in chosen))
   if identity in seen_groups:continue
   seen_groups.add(identity);roots={m.target for m in chosen};sources=set().union(*(supports[(m.target,m.inputs,m.truth)] for m in chosen))
   refs=cones.refs.copy();removed={};pending=[circuit.producer[t] for t in roots]
   for t in roots:refs[t]=0
   while pending:
    gate=pending.pop();ident=id(gate)
    if ident in removed or any(refs[n]>0 for n in cones.output_nets[ident]):continue
    removed[ident]=gate
    for n in cones.dependencies[ident]:
     refs[n]-=1
     if refs[n]==0 and n in circuit.producer:pending.append(circuit.producer[n])
   rescued=set()
   def rescue(n):
    gate=circuit.producer.get(n)
    if gate is None or id(gate) not in removed or id(gate) in rescued:return
    rescued.add(id(gate))
    for d in cones.dependencies[id(gate)]:rescue(d)
   for n in sources:rescue(n)
   released=sum(circuit.lib.cells[gate.cell].units for ident,gate in removed.items() if ident not in rescued)
   gain=released-sum(plan_cost(circuit,m.plan) for m in chosen);extra=gain-sum(m.gain for m in chosen)
   counters['shared_cone_group_proxies']=counters.get('shared_cone_group_proxies',0)+1
   if gain<=0 or extra<=0:continue
   errors=0
   for m in chosen:errors|=m.bad
   rank=(-gain/(25+errors.bit_count()),-extra,errors.bit_count())
   if error_budget is not None:rank=(max(0,errors.bit_count()-error_budget)/gain,*rank)
   found.append((rank,tuple(chosen)))
 return sorted(found,key=lambda row:row[0])

def bundle_candidates(circuit,moves,screen,inputs,reference,counters,limit,ceiling,debt_units,*,shared_seeds=None,trial_log=None,budgeted_shared=True,deduplicate=True):
 singles=[m for m in moves if hasattr(m,'target')]
 pool=[];seen=set();per_root={}
 shared=[]
 if shared_seeds is not None:
  for _,group in shared_groups(circuit,[*singles,*shared_seeds],counters,error_budget=limit if budgeted_shared else None):
   old={(x.target,x.inputs,x.truth) for x in shared}
   extra=[m for m in group if (m.target,m.inputs,m.truth) not in old]
   if len(shared)+len(extra)<=8:shared.extend(extra)
   if len(shared)>=8:break
 for order in (lambda m:(-m.gain,m.error),lambda m:(m.error,-m.gain)):
  quota=min(24-len(shared),len(pool)+(8 if shared else 12))
  for m in sorted(singles,key=order):
   key=(m.target,m.inputs,m.truth)
   if key in seen or per_root.get(m.target,0)>=2:continue
   seen.add(key);per_root[m.target]=per_root.get(m.target,0)+1;pool.append(m)
   if len(pool)>=quota:break
  if len(pool)>=24:break
 for m in shared:
  key=(m.target,m.inputs,m.truth)
  if key not in seen:seen.add(key);pool.append(m)
 if len(pool)<2:return []
 proposals=[]
 def add(group,rank,category):
  if len(group)<2:return
  key=tuple(sorted((m.target,m.inputs,m.truth) for m in group))
  if not any(p[2]==key for p in proposals):proposals.append((rank,category,key,tuple(group)))
 if shared_seeds is not None:
  for rank,group in shared_groups(circuit,pool,counters,error_budget=limit if budgeted_shared else None):add(group,rank,'shared_cone')
 for seed in pool[:12]:
  peers=sorted((m for m in pool if m.target!=seed.target),key=lambda m:(-(m.gain/max(5,(m.bad&~seed.bad).bit_count())),-m.gain,m.error))
  group=propose_batch(circuit,[seed,*peers],screen.bad,limit,quota=4)
  if group:add(group,(-sum(m.gain for m in group),max(m.error for m in group)),'support_overlap')
  if peers:
   peer=min(peers,key=lambda m:((screen.bad^seed.bad^m.bad).bit_count(),-m.gain))
   estimate=(screen.bad^seed.bad^peer.bad).bit_count()
   if estimate<=ceiling:add([seed,peer],(estimate,-seed.gain-peer.gain),'possible_cancellation')
 selected=[]
 has_shared=any(p[1]=='shared_cone' for p in proposals)
 for category in ('shared_cone','support_overlap','possible_cancellation'):
  selected.extend(sorted((p for p in proposals if p[1]==category),key=lambda p:p[0])[:2 if has_shared else 3])
 result=[];physical_seen=set()
 for _,category,_,components in selected:
  move=CompoundMove(components,0,0,0,family='input_index_'+category)
  try:trial=normalize(augment(circuit,[move]).materialize(move.replacements()))
  except ValueError:continue
  if deduplicate:
   identity=structural_id(trial)
   if identity in physical_seen:
    counters['coherent_bundle_duplicate_structures']=counters.get('coherent_bundle_duplicate_structures',0)+1
    if trial_log is not None:trial_log.append(dict(category=category,targets=[m.target for m in components],retained_candidate=False,rejection='duplicate_physical_structure',propagated=False))
    continue
   physical_seen.add(identity)
  errors=wrong(trial,trace(trial,inputs,10000,counters),reference);gain=circuit.units-trial.units
  counters['coherent_bundle_simulations']=counters.get('coherent_bundle_simulations',0)+1
  if category=='shared_cone':counters['shared_cone_bundle_simulations']=counters.get('shared_cone_bundle_simulations',0)+1
  accepted=errors.bit_count()<=ceiling and gain>=-debt_units and (gain>0 or errors.bit_count()<screen.bad.bit_count())
  if trial_log is not None:trial_log.append(dict(category=category,targets=[m.target for m in components],component_gain_units=[m.gain for m in components],
   actual_gain_units=gain,actual_area=circuit.lib.area(trial.units),actual_er=errors.bit_count()/10000,retained_candidate=accepted,
   rejection=None if accepted else ('beyond_exploration_ceiling' if errors.bit_count()>ceiling else 'no_acceptable_area_error_tradeoff'),propagated=True))
  if accepted:
   move.gain=gain;move.bad=errors;move.error=errors.bit_count();result.append(move)
 counters['coherent_bundle_proposals']=counters.get('coherent_bundle_proposals',0)+len(proposals)
 return result
