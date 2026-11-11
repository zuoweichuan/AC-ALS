"""Byte lookup for weighted truth-table response costs (no circuit simulation)."""
class WeightedCost:
 def __init__(self,zero,one):
  if len(zero)!=len(one):raise ValueError('Response table length mismatch')
  self.base=sum(zero);self.tables=[];self.cache={}
  delta=[b-a for a,b in zip(zero,one)]
  for offset in range(0,len(delta),8):
   weights=delta[offset:offset+8]+[0]*max(0,8-len(delta[offset:offset+8]));table=[0]*256
   for code in range(1,256):
    low=code&-code;table[code]=table[code^low]+weights[low.bit_length()-1]
   self.tables.append(table)
 def __call__(self,truth):
  if truth in self.cache:return self.cache[truth]
  code=truth;cost=self.base
  for table in self.tables:cost+=table[code&255];code>>=8
  self.cache[truth]=cost;return cost
