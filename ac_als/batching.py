"""Greedy bounded LAC batching; joint error is always checked by propagation."""
from frontier import forbidden_fanout

def propose_batch(circuit,ordered,bad,limit,quota=6):
 selected=[];blocked=set();sources=set();targets=set();envelope=bad
 for move in ordered:
  roots=(move.target,) if hasattr(move,'target') else move.targets
  if set(roots)&blocked:continue
  tfo=forbidden_fanout(circuit,roots)
  if tfo&targets or set(move.inputs)&blocked or sources&tfo:continue
  predicted=envelope|move.bad
  if predicted.bit_count()>limit:continue
  selected.append(move);blocked|=tfo;sources.update(move.inputs);targets.update(roots);envelope=predicted
  if len(selected)>=quota:break
 return selected if len(selected)>1 else []
