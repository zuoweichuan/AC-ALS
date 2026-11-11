"""Screen cross-cone correction signals against each base LAC's error indices."""
from logic import Plan
from simulate import plan_value
from frontier import forbidden_fanout
from cutmap import variable

def signal_patches(circuit,target,measured,values,mask,bad,flip,mapper,limit,counters):
 blocked=forbidden_fanout(circuit,[target]);survive=circuit.materialize([(target,(),circuit.lib.mapper.plans[0])])
 released=set(circuit.producer)-set(survive.producer)
 sources=[n for n in (*circuit.inputs,*circuit.producer) if n not in blocked and n not in released]
 byarea=sorted((r for r in measured if r[1]<-2 and r[4].name!='leaf'),key=lambda r:(r[1],r[0],r[2],r[3]))
 bases=[];words=set();result=[];t=values[target]
 for row in byarea:
  names,plan=row[2],row[4];word=plan_value(plan,names,circuit.lib,values,mask)
  if word in words:continue
  words.add(word);bases.append((names,plan,word))
  if len(bases)>=6:break
 for names,base,word in bases:
  for op in (8,14,6):
   ranked=[]
   for source in sources:
    q=values[source];candidate=(word&q) if op==8 else ((word|q) if op==14 else (word^q))
    if candidate==word:continue
    delta=t^candidate;newbad=(bad&(delta^mask))|(flip&delta);error=newbad.bit_count()
    counters['residual_signal_candidates_scored']=counters.get('residual_signal_candidates_scored',0)+1
    if error<=limit:ranked.append((error,circuit.level[source],source,newbad))
   for error,_,source,newbad in sorted(ranked)[:2]:
    inputs=names if source in names else (*names,source);q=Plan('leaf',leaf=inputs.index(source))
    plan=mapper.binary(op,base,q);k=len(inputs);truthmask=(1<<(1<<k))-1
    truthvalues={n:variable(k,i) for i,n in enumerate(inputs)}
    truth=plan_value(plan,inputs,circuit.lib,truthvalues,truthmask)
    proxy=mapper.cost(plan)
    result.append((error,proxy,inputs,truth,plan,'cross_cone_residual_patch',newbad))
 return result
