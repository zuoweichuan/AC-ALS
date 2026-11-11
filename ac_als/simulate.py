"""Bit-parallel search simulator and independent NumPy LUT validation."""
import runtime
import numpy as np
from logic import evaluate
from netlist import ZERO,ONE


def random_inputs(circuit,count,seed):
    return np.random.default_rng(seed).integers(0,2,size=(count,len(circuit.inputs)),dtype=np.uint8)


def pack(matrix,inputs):
    return {n:int.from_bytes(np.packbits(matrix[:,i],bitorder='little').tobytes(),'little') for i,n in enumerate(inputs)}


def trace(circuit,inputs,count,counters=None,base=None,flip=None):
    mask=(1<<count)-1
    vals=dict(base) if base is not None else {**inputs,ZERO:0,ONE:mask}
    if flip is not None: vals[flip]^=mask
    affected={flip} if flip is not None else None
    gates=0
    for g in circuit.gates:
        c=circuit.lib.cells[g.cell]
        deps={p:circuit.resolve(g.pins[p]) for p in c.inputs}
        if affected is not None and not (set(deps.values()) & affected):continue
        env={'0':0,'1':mask,**{p:vals[n] for p,n in deps.items()}}
        for p,fun in c.functions.items():
            if p not in g.pins or g.pins[p]==flip:continue
            vals[g.pins[p]]=evaluate(fun,env,mask)
            if affected is not None:affected.add(g.pins[p])
        gates+=1
    if counters is not None:
        key='root_flip_simulations' if flip is not None else 'full_simulations'
        counters[key]=counters.get(key,0)+1
        counters['gate_evaluations']=counters.get('gate_evaluations',0)+gates
    return vals


def wrong(circuit,values,reference):
    result=0
    for n in circuit.outputs:result|=values[circuit.resolve(n)]^reference[n]
    return result


def outputs(circuit,values):return {n:values[circuit.resolve(n)] for n in circuit.outputs}


def plan_value(plan,divisors,lib,values,mask):
    leaves=tuple(values[n] for n in divisors)
    def ev(p):
        if p.name=='leaf':return leaves[p.leaf] if p.leaf>=0 else (0,mask)[p.leaf+2]
        c=lib.cells[p.name]
        return evaluate(next(iter(c.functions.values())),{'0':0,'1':mask,**dict(zip(c.inputs,map(ev,p.children)))},mask)
    return ev(plan)


def validate_numpy(circuit,matrix):
    """Different evaluation path: per-cell indexed scalar truth tables.

    Liberty LUTs use first pin as the most significant input index.
    """
    n=len(matrix);vals={s:matrix[:,i] for i,s in enumerate(circuit.inputs)}
    vals[ZERO]=np.zeros(n,dtype=np.uint8);vals[ONE]=np.ones(n,dtype=np.uint8)
    for g in circuit.gates:
        c=circuit.lib.cells[g.cell];index=np.zeros(n,dtype=np.uint32)
        for p in c.inputs:index=(index<<1)|vals[circuit.resolve(g.pins[p])]
        for p,table in c.tables.items():
            if p not in g.pins:continue
            lut=np.array([(table>>i)&1 for i in range(1<<len(c.inputs))],dtype=np.uint8)
            vals[g.pins[p]]=lut[index]
    return np.column_stack([vals[circuit.resolve(n)] for n in circuit.outputs])
