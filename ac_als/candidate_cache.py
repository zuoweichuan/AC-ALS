"""Reuse screened LAC definitions; refresh all costs against the current netlist."""
from time import perf_counter
from frontier import Move,forbidden_fanout,normalize
from screen import FanoutCones
from simulate import trace,wrong,plan_value

class CandidateCache:
 def __init__(self):self.pool={};self.context=None
 def add(self,moves):
  for m in moves:
   if not hasattr(m,'target'):continue
   key=(m.target,m.inputs,m.truth)
   self.pool.pop(key,None);self.pool[key]=m
  while len(self.pool)>4096:self.pool.pop(next(iter(self.pool)))
 def refresh(self,circuit,inputs,count,reference,mapper,counters,limit,context=None):
  self.context=None
  if not self.pool:return [],None
  tick=perf_counter();values=context['base'] if context else trace(circuit,inputs,count,counters);bad=wrong(circuit,values,reference)
  mask=(1<<count)-1;responses=context['flip_errors'] if context else {};potentials=context['potentials'] if context else {};cones=FanoutCones(circuit);blocked={};scored=[];seen=set()
  self.context=dict(base=values,flip_errors=responses,potentials=potentials)
  for key,m in list(self.pool.items()):
   target=m.target
   if target not in circuit.producer:self.pool.pop(key,None);continue
   names=tuple(circuit.resolve(n) for n in m.inputs)
   if any(n not in values for n in names):self.pool.pop(key,None);continue
   if target not in blocked:blocked[target]=forbidden_fanout(circuit,[target])
   if set(names)&blocked[target]:self.pool.pop(key,None);continue
   identity=(target,names,m.truth)
   if identity in seen:continue
   seen.add(identity)
   if target not in responses:
    responses[target]=wrong(circuit,trace(circuit,inputs,count,counters,base=values,flip=target),reference)
    potentials[target]=cones.potential(target)
   word=plan_value(m.plan,names,circuit.lib,values,mask);delta=word^values[target]
   errors=(bad&(delta^mask))|(responses[target]&delta);error=errors.bit_count()
   counters['cached_template_scores']=counters.get('cached_template_scores',0)+1
   counters['function_error_evaluations']=counters.get('function_error_evaluations',0)+1
   if error>limit:continue
   support=set()
   def leaves(p):
    if p.name=='leaf':
     if p.leaf>=0:support.add(p.leaf)
    else:
     for child in p.children:leaves(child)
   leaves(m.plan)
   released=cones.saved_with_inputs(target,tuple(names[i] for i in support))
   proxy=max(1,min(4,m.gain),released-mapper.cost(m.plan))
   scored.append((proxy,error,target,names,m.truth,m.plan,errors))
  efficiency=lambda r:(-r[0]/(25+(r[6]&~bad).bit_count()),-r[0],r[1],r[2],r[3])
  byarea=lambda r:(-r[0],r[1],r[2],r[3])
  byerror=lambda r:(r[1],-r[0],r[2],r[3])
  selected=[];used=set();per_target={}
  for row in [*sorted(scored,key=efficiency)[:12],*sorted(scored,key=byarea)[:8],*sorted(scored,key=byerror)[:4]]:
   identity=(row[2],row[3],row[4])
   if identity not in used and per_target.get(row[2],0)<2:
    used.add(identity);selected.append(row);per_target[row[2]]=per_target.get(row[2],0)+1
  found=[]
  for _,error,target,names,truth,plan,errors in selected:
   trial=normalize(circuit.materialize([(target,names,plan)]));gain=circuit.units-trial.units
   counters['cache_area_materializations']=counters.get('cache_area_materializations',0)+1
   if gain>0:found.append(Move(target,names,truth,plan,gain,errors,error,'cached_template'))
  counters['cache_refresh_seconds']=counters.get('cache_refresh_seconds',0)+perf_counter()-tick
  return found,bad
