"""Targeted repair using retained logic and selected original-cone divisors.

CPM scores share each current target's flip response. Original signals need
one shared trace; the limited retained candidates are then physically mapped.
No Monte Carlo propagation is run per candidate function.
"""
from dataclasses import dataclass
from hashlib import sha256
from frontier import Move,normalize,forbidden_fanout
from netlist import Circuit,Gate,ZERO,ONE
from logic import Plan
from simulate import trace,wrong

@dataclass
class RestoredMove(Move):
 extra_gates:tuple=()
 hamming_gain:int=0
 def record(self):return dict(super().record(),hamming_gain=self.hamming_gain)

@dataclass
class CompoundMove:
 components:tuple
 gain:int
 error:int
 bad:int
 hamming_gain:int=0
 family:str='targeted_compound_repair'
 @property
 def extra_gates(self):return tuple(g for m in self.components for g in getattr(m,'extra_gates',()))
 def replacements(self):return [m.replacement() for m in self.components]
 def record(self):return dict(targets=[m.target for m in self.components],inputs=list(dict.fromkeys(n for m in self.components for n in m.inputs)),
  family=self.family,gain_units=self.gain,error=self.error,hamming_gain=self.hamming_gain,components=[m.record() for m in self.components])

def bitplanes(words):
 result=[]
 for word in words:
  carry=word;i=0
  while carry:
   if i==len(result):result.append(0)
   result[i],carry=result[i]^carry,result[i]&carry;i+=1
 return result
def weighted(planes,delta):return sum((1<<i)*(p&delta).bit_count() for i,p in enumerate(planes))

class AugmentedCircuit(Circuit):
 def materialize(self,replacements):
  result=super().materialize(replacements)
  result.outputs=self.real_outputs;result.rebuild();return result

def augment(circuit,moves):
 extra={}
 for move in moves:
  for gate in getattr(move,'extra_gates',()):
   key=tuple(sorted(gate.pins.items()))
   if gate.name in extra and (extra[gate.name].cell,tuple(sorted(extra[gate.name].pins.items())))!=(gate.cell,key):raise ValueError('Conflicting restored cone')
   extra[gate.name]=gate
 if not extra:return circuit
 # Circuit construction immediately prunes unobservable logic. Keep provisional
 # leaves live until replacement, then remove all auxiliary output ports.
 auxiliary=tuple(dict.fromkeys(g.pins[p] for g in extra.values() for p in circuit.lib.cells[g.cell].functions if p in g.pins))
 result=AugmentedCircuit(circuit.lib,circuit.inputs,(*circuit.outputs,*auxiliary),[*circuit.gates,*extra.values()],dict(circuit.aliases),circuit.name)
 result.real_outputs=circuit.outputs;return result

def repair_candidates(circuit,original,original_values,screen,mapper,counters,targets,debt_units,prefix):
 mask=(1<<10000)-1;bad=screen.bad;values=screen.values;raw=[]
 reference={o:original_values[original.resolve(o)] for o in original.outputs}
 error_words=tuple(values[circuit.resolve(o)]^reference[o] for o in circuit.outputs)
 base_planes=bitplanes(error_words);inputs={n:values[n] for n in circuit.inputs}
 original_inputs=set(original.inputs)
 sources=tuple(dict.fromkeys((*original.inputs,*original.producer,ZERO,ONE)))
 raw_gate_inputs={g.pins[p] for g in circuit.gates for p in circuit.lib.cells[g.cell].inputs}
 virtual=[o for o,e in zip(circuit.outputs,error_words) if e and o not in circuit.producer and o not in raw_gate_inputs and o not in circuit.inputs]
 virtual.sort(key=lambda o:-(values[circuit.resolve(o)]^reference[o]).bit_count())
 targets=list(dict.fromkeys([*targets,*virtual[:16]]))
 root_count=0
 for target in targets:
  is_virtual=target not in circuit.producer
  if is_virtual:
   output_flip=tuple(mask if o==target else 0 for o in circuit.outputs)
   flip_errors=tuple(e^d for e,d in zip(error_words,output_flip));flip=0
   for e in flip_errors:flip|=e
   flip_planes=bitplanes(flip_errors);base=circuit.resolve(target);t=values[base]
   counters['virtual_output_responses']=counters.get('virtual_output_responses',0)+1
  else:
   flip=screen.responses[target];gate=circuit.producer[target];cell=circuit.lib.cells[gate.cell]
   if len(cell.functions)!=1:continue
   flipped=trace(circuit,inputs,10000,counters,base=values,flip=target)
   counters['repair_bit_response_simulations']=counters.get('repair_bit_response_simulations',0)+1
   flip_planes=bitplanes(tuple(flipped[circuit.resolve(o)]^reference[o] for o in circuit.outputs))
   output_flip=tuple(values[circuit.resolve(o)]^flipped[circuit.resolve(o)] for o in circuit.outputs)
   t=values[target];base=prefix+'_base_'+target
  root_count+=1;local_prefix=prefix+'_t_'+target
  forbidden=forbidden_fanout(circuit,[target]);equivalent={}
  for net in sorted(values,key=lambda n:(0 if n in original_inputs or n in (ZERO,ONE) else 1,circuit.level.get(n,0),n)):
   if net not in forbidden:equivalent.setdefault(values[net],net)
  matches={source:equivalent[word] for source,word in original_values.items() if word in equivalent}
  counters['sample_equivalent_original_boundaries']=counters.get('sample_equivalent_original_boundaries',0)+len(matches)
  # These boundaries are equal on the training stream, not formally equivalent.
  # Induction over the grafted cone preserves its sampled words. Independent
  # validation is mandatory, and the actual accepted candidate is propagated.
  def renamed(name):
   name=original.resolve(name)
   if name in original_inputs or name in (ZERO,ONE):return name
   return matches.get(name,local_prefix+'_o_'+name)
  def cone(name):
   seen=set()
   def visit(net):
    net=original.resolve(net)
    if net in matches:return
    g=original.producer.get(net)
    if g is None or id(g) in seen:return
    seen.add(id(g))
    for pin in original.lib.cells[g.cell].inputs:visit(g.pins[pin])
   visit(name)
   return tuple(Gate(local_prefix+'_g_'+str(i),g.cell,{p:renamed(n) for p,n in g.pins.items()}) for i,g in enumerate(original.gates) if id(g) in seen)
  cones={}
  clone=() if is_virtual else (Gate(prefix+'_clone_'+target,gate.cell,{p:(base if p in cell.functions else circuit.resolve(n)) for p,n in gate.pins.items()}),)
  scored=[]
  for source in sources:
   word=original_values[source];divisor=renamed(source)
   # Wire or inversion can replace an approximate cut. Small residual wrappers
   # preserve the current root via an auxiliary copy rather than a self loop.
   variants=[(word,(divisor,),2),(word^mask,(divisor,),1),
    (t&word,(base,divisor),8),(t|word,(base,divisor),14),
    (t^word,(base,divisor),6),(t&(word^mask),(base,divisor),2),
    (t|(word^mask),(base,divisor),11)]
   for output,names,truth in variants:
    delta=t^output;errors=(bad&(delta^mask))|(flip&delta);error=errors.bit_count()
    counters['restored_function_scores']=counters.get('restored_function_scores',0)+1
    if error>bad.bit_count():continue
    hd_gain=weighted(base_planes,delta)-weighted(flip_planes,delta)
    if error<bad.bit_count() or hd_gain>0:scored.append((error,len(names),source,names,truth,errors,hd_gain,delta))
  counters['restored_signals_scored']=counters.get('restored_signals_scored',0)+len(sources)
  # Shortlist error-reducing wrappers and wire restorations separately.
  selected={}
  rank=lambda r:(r[0],-r[6],r[1],r[2],r[3],r[4])
  # Equal vector ER can hide very different partial bit repairs. Cheap PI
  # wrappers must not crowd out the complementary repair needed by a pair.
  for rows in (sorted(scored,key=rank)[:4],sorted((r for r in scored if r[1]==1),key=rank)[:3],
               sorted((r for r in scored if r[2] in original_inputs or r[2] in (ZERO,ONE)),key=rank)[:2]):
   for row in rows:selected[row[2:5]]=row
  for error,_,source,names,truth,errors,hd_gain,delta in selected.values():
   if source not in cones:cones[source]=cone(source)
   extra=cones[source]+(clone if base in names else ())
   plan=mapper.map(len(names),truth)
   move=RestoredMove(target,names,truth,plan,0,errors,error,'sample_boundary_cone_compensation',extra,hd_gain)
   move.output_changes=tuple(w&delta for w in output_flip)
   trial=normalize(augment(circuit,[move]).materialize([move.replacement()]));move.gain=circuit.units-trial.units
   counters['repair_area_materializations']=counters.get('repair_area_materializations',0)+1
   if move.gain>=-debt_units:raw.append(move)
 counters['restored_roots_examined']=counters.get('restored_roots_examined',0)+root_count
 raw.sort(key=lambda m:(m.error,-m.hamming_gain,-m.gain,m.target,m.inputs))
 by_root={}
 for m in raw:by_root.setdefault(m.target,[]).append(m)
 diverse=[m for group in by_root.values() for m in group[:2]]
 diverse.sort(key=lambda m:(m.error,-m.hamming_gain,-m.gain));pool=diverse[:12]
 # A shared output-response estimate ranks a small pair pool. It is only a
 # filter; at most six retained pairs undergo actual circuit propagation.
 pairs=[];seen=set()
 for a in pool[:6]:
  peers=[]
  for b in pool:
   if b.target==a.target:continue
   key=tuple(sorted((id(a),id(b))))
   if key in seen:continue
   seen.add(key);estimate=0
   for e,da,db in zip(error_words,a.output_changes,b.output_changes):estimate|=e^da^db
   if estimate.bit_count()<bad.bit_count():peers.append((estimate.bit_count(),-a.gain-b.gain,a,b))
  if peers:pairs.append(min(peers,key=lambda p:p[:2]))
 compound=[]
 for _,_,a,b in sorted(pairs,key=lambda p:p[:2])[:6]:
  move=CompoundMove((a,b),0,0,0)
  try:trial=normalize(augment(circuit,[move]).materialize(move.replacements()))
  except ValueError:continue
  words=trace(trial,inputs,10000,counters);errors=wrong(trial,words,reference);gain=circuit.units-trial.units
  counters['compound_repair_simulations']=counters.get('compound_repair_simulations',0)+1
  if errors.bit_count()<bad.bit_count() and gain>=-debt_units:
   move.gain=gain;move.error=errors.bit_count();move.bad=errors
   move.hamming_gain=sum(e.bit_count() for e in error_words)-sum((words[trial.resolve(o)]^reference[o]).bit_count() for o in trial.outputs)
   compound.append(move)
 return [*compound,*raw[:24]]
