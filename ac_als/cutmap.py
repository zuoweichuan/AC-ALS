"""Exact real-cell factoring for selected <=4-input truth tables."""
from functools import lru_cache
from itertools import permutations
from logic import Plan,evaluate


def variable(n,i):return sum(((row>>i)&1)<<row for row in range(1<<n))
def cof(truth,n,i,v):
    return sum(((truth>>((row&((1<<i)-1))|(v<<i)|((row>>i)<<(i+1))))&1)<<row for row in range(1<<(n-1)))


class CutMapper:
    def __init__(self,lib,maximum=4):
        self.lib=lib;self.catalog={};self.stats=dict(mapped=0);self.maximum=maximum
        for n in range(1,maximum+1):
            mask=(1<<(1<<n))-1
            table={0:Plan('leaf',leaf=-2),mask:Plan('leaf',leaf=-1)}
            for i in range(n):table[variable(n,i)]=Plan('leaf',leaf=i)
            for cell in lib.cells.values():
                if len(cell.functions)!=1 or not 1<=len(cell.inputs)<=n:continue
                tree=next(iter(cell.functions.values()))
                for order in permutations(range(n),len(cell.inputs)):
                    truth=evaluate(tree,{'0':0,'1':mask,**dict(zip(cell.inputs,(variable(n,i) for i in order)))},mask)
                    p=Plan(cell.name,tuple(Plan('leaf',leaf=i) for i in order),units=cell.units)
                    if truth not in table or p.units<table[truth].units:table[truth]=p
            self.catalog[n]=table

    @lru_cache(maxsize=4096)
    def cost(self,p):
        seen=set()
        def visit(q):
            if q.name=='leaf' or q in seen:return
            seen.add(q)
            for child in q.children:visit(child)
        visit(p)
        return sum(self.lib.cells[q.name].units for q in seen)

    @lru_cache(maxsize=4096)
    def bind(self,p,positions):
        if p.name=='leaf':return Plan('leaf',leaf=positions[p.leaf]) if p.leaf>=0 else p
        return Plan(p.name,tuple(self.bind(c,positions) for c in p.children),units=p.units)

    def binary(self,truth,a,b):
        p=self.lib.mapper.plans[truth]
        def substitute(q):
            if q.name=='leaf':return (a,b)[q.leaf] if q.leaf>=0 else q
            children=tuple(substitute(c) for c in q.children)
            return Plan(q.name,children,units=self.lib.cells[q.name].units+sum(x.units for x in children))
        return substitute(p)

    @lru_cache(maxsize=2048)
    def map(self,n,truth):
        self.stats['mapped']+=1
        mask=(1<<(1<<n))-1
        if truth==0:return Plan('leaf',leaf=-2)
        if truth==mask:return Plan('leaf',leaf=-1)
        active=tuple(i for i in range(n) if cof(truth,n,i,0)!=cof(truth,n,i,1))
        if len(active)<n:
            compact=sum(((truth>>sum(((row>>j)&1)<<i for j,i in enumerate(active)))&1)<<row for row in range(1<<len(active)))
            return self.bind(self.map(len(active),compact),active)
        options=[]
        if truth in self.catalog[n]:options.append(self.catalog[n][truth])
        if n>4 and options:return options[0]
        if n==1:return self.lib.mapper.plans[5] if truth==1 else Plan('leaf',leaf=0)
        splits=range(n) if n<=4 else sorted(range(n),key=lambda i:(
            (cof(truth,n,i,0)^cof(truth,n,i,1)).bit_count(),i))[:2]
        for i in splits:
            rest=tuple(j for j in range(n) if j!=i)
            lo=cof(truth,n,i,0);hi=cof(truth,n,i,1)
            a=self.bind(self.map(n-1,lo),rest);b=self.bind(self.map(n-1,hi),rest);s=Plan('leaf',leaf=i)
            if lo==hi:options.append(a)
            elif hi==lo^((1<<(1<<(n-1)))-1):options.append(self.binary(6,s,a))
            else:
                if lo==0:options.append(self.binary(8,s,b))
                elif hi==0:options.append(self.binary(8,self.binary(5,s,s),a))
                elif lo==((1<<(1<<(n-1)))-1):options.append(self.binary(14,self.binary(5,s,s),b))
                elif hi==((1<<(1<<(n-1)))-1):options.append(self.binary(14,s,a))
                mux=self.catalog[3].get(228)
                if mux:
                    def sub(q):
                        if q.name=='leaf':return (s,a,b)[q.leaf] if q.leaf>=0 else q
                        children=tuple(sub(c) for c in q.children)
                        return Plan(q.name,children,units=self.lib.cells[q.name].units+sum(c.units for c in children))
                    options.append(sub(mux))
                options.append(self.binary(14,self.binary(8,self.binary(5,s,s),a),self.binary(8,s,b)))
        return min(options,key=lambda p:(self.cost(p),repr(p)))
