"""Bounded weighted support pruning; learn small functions from shared responses."""
from cutmap import variable

def support_models(a,b,k,mapper,limit):
 """At most three subsets per depth, not all truth functions or input subsets."""
 cache={};scored=0
 def model(active):
  nonlocal scored
  if active in cache:return cache[active]
  n=len(active);aa=[0]*(1<<n);bb=[0]*(1<<n)
  for row in range(1<<k):
   code=sum(((row>>bit)&1)<<j for j,bit in enumerate(active));aa[code]+=a[row];bb[code]+=b[row]
  truth=sum((y<x)<<row for row,(x,y) in enumerate(zip(aa,bb)))
  error=sum(min(x,y) for x,y in zip(aa,bb));scored+=1
  cache[active]=(error,truth);return cache[active]
 beam=[tuple(range(k))];selected=[]
 for width in range(k-1,0,-1):
  choices={tuple(i for i in active if i!=bit) for active in beam for bit in active}
  beam=sorted(choices,key=lambda s:(model(s)[0],s))[:3]
  for active in beam[:2]:
   error,truth=model(active)
   if error>limit:continue
   n=len(active);plan=mapper.bind(mapper.map(n,truth),active)
   full=sum(((truth>>sum(((row>>bit)&1)<<j for j,bit in enumerate(active)))&1)<<row for row in range(1<<k))
   selected.append((full,plan))
 return selected,scored
