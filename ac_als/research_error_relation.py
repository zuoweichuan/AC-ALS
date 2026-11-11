"""Whole-input vector-error inclusion miter; reference logic is proof-only.

Prove bad(candidate,accurate) => bad(parent,accurate). This is stronger than
Monte Carlo acceptance and different from requiring identical output functions.
All reference/parent gates belong only to the proof circuit, never the result.
"""
from netlist import Circuit,Gate,ZERO,ONE
from research_c880_hybrid import proof

def inclusion_miter(reference,parent,candidate):
    if any(c.inputs!=reference.inputs or c.outputs!=reference.outputs for c in (parent,candidate)):
        raise ValueError('Error-inclusion interfaces differ')
    lib=reference.lib;gates=[];serial=0
    def fresh():
        nonlocal serial
        serial+=1;return '__relation_%d'%serial
    def import_circuit(c):
        names={n:n for n in reference.inputs};names.update({ZERO:ZERO,ONE:ONE})
        for n in c.producer:names[n]=fresh()
        for g in c.gates:
            gates.append(Gate('relation_'+str(len(gates)),g.cell,{p:names[c.resolve(n)] for p,n in g.pins.items()}))
        return {o:names[c.resolve(o)] for o in reference.outputs}
    def binary(truth,a,b):
        memo={}
        def emit(p):
            if p.name=='leaf':return (a,b)[p.leaf] if p.leaf>=0 else (ZERO,ONE)[p.leaf+2]
            if p in memo:return memo[p]
            cell=lib.cells[p.name];children=tuple(emit(q) for q in p.children);n=fresh()
            gates.append(Gate('relation_'+str(len(gates)),p.name,{**dict(zip(cell.inputs,children)),next(iter(cell.functions)):n}));memo[p]=n;return n
        return emit(lib.mapper.plans[truth])
    gold=import_circuit(reference);old=import_circuit(parent);new=import_circuit(candidate)
    def bad(out):
        signals=[binary(6,out[o],gold[o]) for o in reference.outputs]
        while len(signals)>1:
            signals=[binary(14,signals[i],signals[i+1]) if i+1<len(signals) else signals[i] for i in range(0,len(signals),2)]
        return signals[0]
    oldbad=bad(old);newbad=bad(new)
    introduced=binary(2,newbad,oldbad)  # bit order a=low: a & !b
    miter=Circuit(lib,reference.inputs,('introduced_error',),gates,{'introduced_error':introduced},'error_relation')
    zero=Circuit(lib,reference.inputs,('introduced_error',),[],{'introduced_error':ZERO},'no_introduced_error')
    return miter,zero

def prove_inclusion(reference,parent,candidate,folder):
    miter,zero=inclusion_miter(reference,parent,candidate)
    return proof(miter,zero,folder)
