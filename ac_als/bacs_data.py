"""User DC-mapped BACS inputs and paper-compatible output bit ordering."""
import runtime
from pathlib import Path
import re
import os
import numpy as np
from portable_assets import asset_root
from netlist import Circuit,Gate
from simulate import pack,trace,outputs,validate_numpy

ROOT=Path(os.environ.get('BACS_PROJECT_ROOT',str(Path(__file__).resolve().parent)))
INPUT_ROOT=Path(os.environ.get('BACS_INPUT_ROOT',str(asset_root()/'bacs/reference')))
AUTHOR_ROOT=Path(os.environ.get('BACS_AUTHOR_ROOT',str(asset_root()/'bacs/blif')))
LIBRARY_PATH=Path(os.environ.get('BACS_LIBRARY_PATH',str(asset_root()/'library/stdcells.lib')))
DESIGNS=('abs_diff','adder32','buttfly','mac','mult8','mult16')
NMED_BOUNDS=(.0059,.0294)
MSE_BOUNDS={'abs_diff':33.5,'mac':11.0,'mult8':40.0,'adder32':48.2}
PAPER_NMED_AREA_RATIOS={'abs_diff':(.485,.227),'adder32':(.101,.058),
    'buttfly':(.206,.133),'mac':(.766,.237),'mult8':(.249,.084),'mult16':(.074,.015)}
PAPER_MSE_AREA_RATIOS={'abs_diff':.349,'mac':.626,'mult8':.828,'adder32':.799}


def source_path(design):return INPUT_ROOT/design/(design+'_syn.v')


def load(path,lib):
    c=Circuit.read(path,lib)
    # Circuit.write escapes flattened bit ports; restore the same bit identifier
    # on reload instead of accidentally changing res[0] into a distinct input.
    def canonical(n):
        return n[1:] if re.fullmatch(r'\\[A-Za-z_][\w$]*\[\d+\]',n) else n
    return Circuit(lib,tuple(map(canonical,c.inputs)),tuple(map(canonical,c.outputs)),
        [Gate(g.name,g.cell,{p:canonical(n) for p,n in g.pins.items()}) for g in c.gates],
        {canonical(n):canonical(v) for n,v in c.aliases.items()},c.name)


def read_blif(path):
    text=Path(path).read_text(encoding='utf-8').replace('\\\n',' ')
    inputs=[];out=[];nodes=[];row=None
    for line in text.splitlines():
        parts=line.split('#')[0].split()
        if not parts:continue
        if parts[0]=='.inputs':inputs+=parts[1:]
        elif parts[0]=='.outputs':out+=parts[1:]
        elif parts[0]=='.names':
            row=(parts[1:-1],parts[-1],[]);nodes.append(row)
        elif parts[0] in ('.model','.end'):row=None
        elif parts[0].startswith('.'):raise ValueError('Unsupported BLIF directive '+parts[0])
        elif row is None:raise ValueError('Unattached BLIF cube')
        else:row[2].append(parts)
    return inputs,out,nodes


def blif_outputs(model,inputs,count):
    ports,out,nodes=model
    if set(ports)!=set(inputs):raise ValueError('BACS author/DC input interface mismatch')
    mask=(1<<count)-1;values=dict(inputs)
    for names,target,cubes in nodes:
        value=0;polarity=None
        for cube in cubes:
            pattern,bit=(cube[0],cube[1]) if names else ('',cube[0])
            if polarity is None:polarity=bit
            if polarity!=bit:raise ValueError('Mixed BLIF cover polarity')
            term=mask
            for n,b in zip(names,pattern):
                if b!='-':term&=values[n] if b=='1' else values[n]^mask
            value|=term
        values[target]=value^(mask if polarity=='0' else 0)
    return {n:values[n] for n in out}


def matrix(design,inputs,split):
    return np.random.default_rng(np.random.SeedSequence([20261027,DESIGNS.index(design),split])).integers(
        0,2,(10000,len(inputs)),dtype=np.uint8)


def preflight(design,lib):
    circuit=load(source_path(design),lib)
    model=read_blif(AUTHOR_ROOT/(design+'.blif'))
    if design=='adder32':
        # Same 32-bit unsigned sum, explicit author-to-DC bus port renaming.
        def renamed(n):
            return re.sub(r'^(in1|in2|res)(?=\[)',lambda m:{'in1':'a','in2':'b','res':'r'}[m[1]],n)
        model=([renamed(n) for n in model[0]],[renamed(n) for n in model[1]],
               [([renamed(n) for n in ins],renamed(out),cubes) for ins,out,cubes in model[2]])
    order=tuple(model[1])
    if set(order)!=set(circuit.outputs):raise ValueError('BACS author/DC output interface mismatch')
    x=matrix(design,circuit.inputs,99);packed=pack(x,circuit.inputs)
    actual=outputs(circuit,trace(circuit,packed,10000));expected=blif_outputs(model,packed,10000)
    bad=0
    for n in order:bad|=actual[n]^expected[n]
    np_outputs=validate_numpy(circuit,x)
    for i,n in enumerate(circuit.outputs):
        word=int.from_bytes(np.packbits(np_outputs[:,i],bitorder='little').tobytes(),'little')
        if word!=actual[n]:raise AssertionError('Independent baseline simulator disagreement')
    if bad:raise ValueError('Original DC netlist differs from author BLIF on %d/10000 vectors'%bad.bit_count())
    return circuit,order,dict(design=design,inputs=len(circuit.inputs),outputs=len(order),
        gates=len(circuit.gates),original_area=lib.area(circuit.units),author_dc_wrong_vectors=0,
        checked_vectors=10000,independent_simulators_agree=True,output_lsb_first=list(order),
        numeric_interpretation='unsigned flattened output in author BLIF PO order; not per-port signed error',
        full_propagations=2,independent_numpy_checks=1)
