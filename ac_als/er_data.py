"""ISCAS85 strict vector ER: one wrong output bit makes that input wrong."""
import runtime
from pathlib import Path
import numpy as np
from bacs_data import load
from bacs_numeric import VectorEROrder,number,number_numpy
from simulate import pack,trace,outputs,validate_numpy

ROOT=Path(__file__).resolve().parent
# Preserve previous ER stream identities; c1355 was appended to the original six.
DESIGNS=('c880','c1908','c2670','c3540','c5315','c7552','c1355')
source_files=('er_data.py','er_search.py','numeric_menu_agent.py','numeric_explore.py',
    'hybrid_knowledge.json','er_agent_knowledge.json')

def source_path(design):return ROOT/'assets'/(design+'.v')
def matrix(design,inputs,split):
    return np.random.default_rng(np.random.SeedSequence([20261025,DESIGNS.index(design),split])).integers(
        0,2,(10000,len(inputs)),dtype=np.uint8)

def preflight(design,lib):
    c=load(source_path(design),lib);order=VectorEROrder(c.outputs)
    x=matrix(design,c.inputs,99)
    actual=number(outputs(c,trace(c,pack(x,c.inputs),10000)),order,10000)
    alternate=number_numpy(validate_numpy(c,x),c.outputs,order)
    if not np.array_equal(actual,alternate):raise AssertionError('ISCAS vector-ER simulator disagreement')
    return c,order,dict(design=design,inputs=len(c.inputs),outputs=len(c.outputs),gates=len(c.gates),
        original_area=lib.area(c.units),checked_vectors=10000,independent_simulators_agree=True,
        metric='strict_vector_er',numeric_interpretation='Any output bit mismatch counts once per input; no bit averaging or numeric weighting',
        output_encoding='fixed-width packed bytes, no 64-bit truncation',full_propagations=1,independent_numpy_checks=1)
