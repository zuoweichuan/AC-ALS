"""Targeted two-cell residual construction from cached response costs."""
from weighted_cost import WeightedCost

def residual_models(a,b,k,mapper,limit):
 catalog=mapper.catalog[k];teacher=WeightedCost(a,b);mask=(1<<(1<<k))-1
 bases=sorted((t for t,p in catalog.items() if p.name!='leaf'),key=lambda t:(teacher(t),catalog[t].units,t))[:6]
 result={};scored=len(teacher.cache)
 for g in bases:
  # q is synthesized to repair the residual of an AND/OR/XOR wrapper.
  for op in (8,14,6):
   ca=[];cb=[]
   def bitfun(x,y):return (op>>(x+2*y))&1
   for row in range(1<<k):
    x=(g>>row)&1
    ca.append(b[row] if bitfun(x,0) else a[row]);cb.append(b[row] if bitfun(x,1) else a[row])
   score=WeightedCost(ca,cb)
   h=min(catalog,key=lambda t:(score(t),catalog[t].units,t));scored+=len(score.cache)
   if score(h)>limit:continue
   if op==8:truth=g&h
   elif op==14:truth=g|h
   else:truth=g^h
   network=mapper.binary(op,catalog[g],catalog[h])
   # Compare the constructed shared expression with ordinary exact factoring.
   plan=min((network,mapper.map(k,truth)),key=mapper.cost)
   if truth not in result or mapper.cost(plan)<mapper.cost(result[truth]):result[truth]=plan
 return list(result.items()),scored
