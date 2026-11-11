"""Cheap node screening and exact deletion-only fanout-free cone potential."""
from collections import Counter
from simulate import trace,wrong
from netlist import ZERO,ONE

class FanoutCones:
 def __init__(self,circuit):
  self.circuit=circuit;self.refs=Counter();self.dependencies={};self.output_nets={};self.removed={}
  for gate in circuit.gates:
   cell=circuit.lib.cells[gate.cell]
   deps=tuple(circuit.resolve(gate.pins[p]) for p in cell.inputs)
   self.dependencies[id(gate)]=deps;self.refs.update(deps)
   self.output_nets[id(gate)]=tuple(gate.pins[p] for p in cell.functions if p in gate.pins)
  self.refs.update(circuit.resolve(p) for p in circuit.outputs)
 def deletions(self,target):
  if target in self.removed:return self.removed[target]
  refs=self.refs.copy();refs[target]=0;queue=[self.circuit.producer[target]];removed={}
  while queue:
   gate=queue.pop()
   if id(gate) in removed or any(refs[n]>0 for n in self.output_nets[id(gate)]):continue
   removed[id(gate)]=gate
   for n in self.dependencies[id(gate)]:
    refs[n]-=1
    if refs[n]==0 and n in self.circuit.producer:queue.append(self.circuit.producer[n])
  self.removed[target]=removed;return removed
 def potential(self,target):
  return sum(self.circuit.lib.cells[g.cell].units for g in self.deletions(target).values())
 def saved_with_inputs(self,target,names):
  removed=self.deletions(target);rescued=set()
  def rescue(n):
   n=self.circuit.resolve(n);gate=self.circuit.producer.get(n)
   if gate is None or id(gate) not in removed or id(gate) in rescued:return
   rescued.add(id(gate))
   for d in self.dependencies[id(gate)]:rescue(d)
  for n in names:rescue(n)
  return sum(self.circuit.lib.cells[g.cell].units for k,g in removed.items() if k not in rescued)

class RootScreen:
 def __init__(self,circuit,inputs,count,reference,counters,base=None,flip_errors=None,potentials=None):
  self.values=trace(circuit,inputs,count,counters) if base is None else base;self.bad=wrong(circuit,self.values,reference)
  mask=(1<<count)-1;self.responses=dict(flip_errors or {});self.potentials=dict(potentials or {});cones=FanoutCones(circuit);quality={}
  for target in circuit.producer:
   potential=self.potentials[target] if target in self.potentials else cones.potential(target);self.potentials[target]=potential
   flip=self.responses[target] if target in self.responses else wrong(circuit,trace(circuit,inputs,count,counters,base=self.values,flip=target),reference)
   self.responses[target]=flip;t=self.values[target]
   alternatives=[0,mask];gate=circuit.producer[target]
   for pin in circuit.lib.cells[gate.cell].inputs:
    v=self.values[circuit.resolve(gate.pins[pin])];alternatives.extend((v,v^mask))
   errors=[]
   for word in alternatives:
    delta=t^word;bad=(self.bad&(delta^mask))|(flip&delta)
    errors.append((bad.bit_count(),(bad&~self.bad).bit_count()))
   lowest=min(errors);quality[target]=(-potential/(25+lowest[1]),lowest[0],-potential,target)
  efficient=sorted(circuit.producer,key=lambda t:quality[t]);large=sorted(circuit.producer,key=lambda t:(-self.potentials[t],quality[t]))
  # Four utility-ranked roots followed by one cone-area exploration root.
  self.order=[];used=set();i=j=0
  while len(used)<len(efficient):
   for _ in range(4):
    while i<len(efficient) and efficient[i] in used:i+=1
    if i<len(efficient):n=efficient[i];used.add(n);self.order.append(n);i+=1
   while j<len(large) and large[j] in used:j+=1
   if j<len(large):n=large[j];used.add(n);self.order.append(n);j+=1
  counters['screened_nodes']=counters.get('screened_nodes',0)+len(self.order)
