"""Exact shared expression DAG made from real <=2-input standard cells.

This is an optional *search representation*. Its area is never reported as the
cost of a mapped incumbent. ABC maps accepted functions for physical comparison.
"""
from netlist import Circuit,Gate,ZERO,ONE
from logic import evaluate
from cutmap import variable
from bacs_data import read_blif


def from_sop(path,source):
    """Import ABC's exact two-input SOP topology as actual standard cells.

    Each local SOP->cell mapping is exhaustively checked on its <=4 truth rows.
    The source-to-SOP equivalence is separately certified by ABC CEC.
    """
    ins,outs,nodes=read_blif(path)
    expected_in=['pi_%d'%i for i in range(len(source.inputs))]
    expected_out=['po_%d'%i for i in range(len(source.outputs))]
    if set(ins)!=set(expected_in) or set(outs)!=set(expected_out):
        raise ValueError('Simplified SOP interface changed')
    values={n:source.inputs[i] for i,n in enumerate(expected_in)}
    gates=[];memo={};used=set(source.inputs)|set(source.outputs)
    def fresh():
        n='__sop_%d'%len(used)
        while n in used:
            n+='_'
        used.add(n)
        return n
    def emit(plan,leaves):
        if plan.name=='leaf':
            return leaves[plan.leaf] if plan.leaf>=0 else (ZERO,ONE)[plan.leaf+2]
        children=tuple(emit(c,leaves) for c in plan.children)
        key=(plan.name,children)
        if key not in memo:
            cell=source.lib.cells[plan.name];n=fresh()
            gates.append(Gate('sop_'+str(len(gates)),cell.name,
                {**dict(zip(cell.inputs,children)),next(iter(cell.functions)):n}))
            memo[key]=n
        return memo[key]
    for names,target,cubes in nodes:
        k=len(names)
        if k>2:
            raise ValueError('Expected two-input SOP after ABC strashing, got '+str(k))
        truth=0;polarity=None
        for cube in cubes:
            pattern,bit=(cube[0],cube[1]) if names else ('',cube[0])
            if polarity is not None and polarity!=bit:
                raise ValueError('Mixed SOP polarity')
            polarity=bit
            for row in range(1<<k):
                if all(b=='-' or int(b)==((row>>i)&1) for i,b in enumerate(pattern)):
                    truth |= 1<<row
        mask=(1<<(1<<k))-1
        if polarity=='0':
            truth ^= mask
        expanded=truth if k==2 else ((truth|(truth<<2)) if k==1 else (15 if truth else 0))
        plan=source.lib.mapper.plans[expanded]
        def check(p):
            if p.name=='leaf':
                return variable(2,p.leaf) if p.leaf>=0 else (0,15)[p.leaf+2]
            cell=source.lib.cells[p.name]
            return evaluate(next(iter(cell.functions.values())),
                {'0':0,'1':15,**dict(zip(cell.inputs,(check(c) for c in p.children)))},15)
        if check(plan)!=expanded:
            raise AssertionError('SOP truth table differs from implementing cells')
        leaves=tuple(values[n] for n in names)+(ZERO,)*(2-k)
        values[target]=emit(plan,leaves)
    return Circuit(source.lib,source.inputs,source.outputs,gates,
                   {n:values['po_%d'%i] for i,n in enumerate(source.outputs)},source.name)


def expand_cells(source):
    gates=[];values={n:n for n in (*source.inputs,ZERO,ONE)};memo={}
    used=set(source.inputs)|set(source.outputs)
    def fresh():
        n='__logic_%d'%len(used)
        while n in used:
            n+='_'
        used.add(n)
        return n
    def emit(plan,leaves):
        if plan.name=='leaf':
            return leaves[plan.leaf] if plan.leaf>=0 else (ZERO,ONE)[plan.leaf+2]
        children=tuple(emit(c,leaves) for c in plan.children)
        key=(plan.name,children)
        if key not in memo:
            c=source.lib.cells[plan.name];n=fresh()
            gates.append(Gate('logic_'+str(len(gates)),c.name,
                              {**dict(zip(c.inputs,children)),next(iter(c.functions)):n}))
            memo[key]=n
        return memo[key]
    def binary(kind,a,b):
        if kind!='not' and a>b:
            a,b=b,a
        return emit(source.lib.mapper.plans[{'and':8,'or':14,'xor':6,'not':5}[kind]],(a,b))
    def expression(tree,env):
        if tree[0]=='var':
            return env[tree[1]]
        if tree[0]=='not':
            a=expression(tree[1],env)
            return binary('not',a,a)
        return binary(tree[0],expression(tree[1],env),expression(tree[2],env))
    for g in source.gates:
        cell=source.lib.cells[g.cell]
        env={'0':ZERO,'1':ONE,**{p:values[source.resolve(g.pins[p])] for p in cell.inputs}}
        for pin,fun in cell.functions.items():
            if pin in g.pins:
                values[g.pins[pin]]=expression(fun,env)
    return Circuit(source.lib,source.inputs,source.outputs,gates,
                   {n:values[source.resolve(n)] for n in source.outputs},source.name).simplify()
