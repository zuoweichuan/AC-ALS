"""Physical structure identity, independent of generated instance/net names."""
import hashlib,json
from logic import evaluate
from cutmap import variable
from netlist import ZERO,ONE

def structural_id(circuit):
 def hashed(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
 words={ZERO:'constant0',ONE:'constant1',**{n:hashed(('input',n)) for n in circuit.inputs}}
 cells=[]
 for gate in circuit.gates:
  cell=circuit.lib.cells[gate.cell]
  names=sorted(set(words[circuit.resolve(gate.pins[p])] for p in cell.inputs))
  index={n:i for i,n in enumerate(names)};mask=(1<<(1<<len(names)))-1
  env={'0':0,'1':mask,**{p:variable(len(names),index[words[circuit.resolve(gate.pins[p])]]) for p in cell.inputs}}
  functions={p:evaluate(fun,env,mask) for p,fun in cell.functions.items() if p in gate.pins}
  cell_id=hashed((gate.cell,names,functions));cells.append(cell_id)
  for pin in functions:words[gate.pins[pin]]=hashed((cell_id,pin))
 return hashed((circuit.inputs,[(o,words[circuit.resolve(o)]) for o in circuit.outputs],sorted(cells),circuit.units))
