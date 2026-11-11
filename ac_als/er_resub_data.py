"""Strict-vector ER on the author's frozen accurate ResubALS initialization."""
from pathlib import Path
import numpy as np
from bacs_data import load, read_blif, blif_outputs, LIBRARY_PATH
from bacs_numeric import VectorEROrder
from simulate import pack,trace,outputs,validate_numpy
from hybrid_identity import structural_id
from prepare_resub_iscas import DEFAULT, verify_bundle, sha, DESIGNS
from er_data import ROOT, matrix
from er_data_v2 import source_files as previous_files

source_files=(*previous_files,'prepare_resub_iscas.py','resub_initial_only.cc',
              'er_resub_data.py','er_search_resub.py','er_suite_resub.py')
_folder=DEFAULT


def configure(folder):
    global _folder
    _folder=Path(folder).resolve()
    return verify_bundle(_folder)


def source_path(design):return _folder/design/'accurate.v'


def preflight(design,lib):
    manifest=verify_bundle(_folder,design);row=manifest['designs'][design]
    circuit=load(source_path(design),lib)
    if sha(LIBRARY_PATH)!=manifest['library_sha256']:
        raise ValueError('Common baseline Liberty changed')
    if circuit.units!=row['units'] or structural_id(circuit)!=row['graph']:
        raise ValueError('Not the author accurate initial structure')
    model=read_blif(_folder/design/'source.blif');mapping=row['port_mapping']
    x=matrix(design,circuit.inputs,99);packed=pack(x,circuit.inputs)
    actual=outputs(circuit,trace(circuit,packed,10000))
    expected=blif_outputs(model,{n:packed[m] for n,m in mapping['inputs'].items()},10000)
    alternate=validate_numpy(circuit,x);bad=0
    for raw,aligned in mapping['outputs'].items():
        bad |= expected[raw]^actual[aligned]
        word=int.from_bytes(np.packbits(alternate[:,circuit.outputs.index(aligned)],bitorder='little').tobytes(),'little')
        if word!=actual[aligned]:raise AssertionError('Accurate baseline independent simulator disagreement')
    if bad:raise ValueError('ResubALS accurate baseline function mismatch')
    audit=dict(design=design,baseline_kind=manifest['baseline_kind'],baseline_manifest=str(_folder/'baseline.json'),
        baseline_manifest_sha256=sha(_folder/'baseline.json'),author_commit=manifest['author_commit'],
        abc_commit=manifest['abc_commit'],author_accurate_blif_sha256=row['source_blif_sha256'],
        initial_mapper=row['initial_mapper'],baseline_sha256=row['sha256'],baseline_area=row['area'],
        dc_area_for_context_only=row['dc_area'],baseline_gates=len(circuit.gates),
        formal_equivalence=row['formal_equivalence'],checked_vectors=10000,
        independent_numpy_agrees=True,approximate_warm_start=False)
    return circuit,VectorEROrder(circuit.outputs),audit


def check_start(design,path,lib):
    """Reject even accurate alternative structures: this entry promises one start."""
    manifest=verify_bundle(_folder,design);row=manifest['designs'][design]
    candidate=load(path or source_path(design),lib)
    if sha(path or source_path(design))!=row['sha256'] or structural_id(candidate)!=row['graph']:
        raise ValueError('Use the frozen ResubALS initial netlist; no DC/alternative/approximate warm start')
    return row
