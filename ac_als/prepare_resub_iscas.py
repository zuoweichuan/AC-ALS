"""Freeze the ResubALS author's accurate ISCAS inputs, with formal proofs.

Uses original compiled ALS/ABC objects only for the initial exact mapper.
Never calls RunMultipleSelection, approximate search, or a model API.
"""
from portable_tools import tool_command
import runtime
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from bacs_data import load, read_blif, blif_outputs, LIBRARY_PATH
from hybrid_identity import structural_id
from logic import Library
from numeric_exact import ExactBackend, write_blif
from simulate import pack, trace, outputs, validate_numpy
import er_data

ROOT = Path(__file__).resolve().parent
AUTHOR = ROOT.parent / 'resubals_paper_baseline'
VENDOR = AUTHOR / 'vendor/ResubALS'
from portable_assets import asset_root
DEFAULT = asset_root() / 'iscas'
DESIGNS = ('c880', 'c1355', 'c1908', 'c2670', 'c3540', 'c5315', 'c7552')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path, value): Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
def linux(path): return ExactBackend.linux(path)


def author_dependencies():
    manifest = read(AUTHOR/'source_manifest.json')
    expected = dict(manifest['original_als_files'])
    expected.update({str(Path(n)): h for n,h in manifest['observed_file_sha256'].items()})
    # Normalize both slash spellings before checking observer-only modifications.
    expected = {str(Path(n.replace('\\','/'))): h for n,h in expected.items()}
    for n,h in expected.items():
        if sha(VENDOR/n) != h: raise ValueError('Unexpected author source change: '+n)
    objects = [AUTHOR/'build/als/CMakeFiles/resubals.out.dir/src'/(n+'.cc.o')
               for n in ('als','error','lac','my_abc','my_util','simulator')]
    archive = AUTHOR/'build/abc/libabc.a'
    dependencies = {str(VENDOR/n): h for n,h in expected.items()}
    for p in [AUTHOR/'source_manifest.json', archive, *objects]: dependencies[str(p)] = sha(p)
    return manifest, objects, archive, dependencies


def build(folder, objects, archive):
    dest=folder/'build';dest.mkdir()
    source=ROOT/'resub_initial_only.cc';shutil.copy2(source,dest/source.name)
    exe=dest/'resub_initial_only'
    args=['/usr/bin/c++','-O3','-DNDEBUG','-std=c++2a','-DLIN64',
          '-DSIZEOF_VOID_P=8','-DSIZEOF_LONG=8','-DSIZEOF_INT=4',
          '-DABC_USE_CUDD=1','-DABC_USE_READLINE','-DABC_USE_PTHREADS',
          '-I'+linux(VENDOR/'abc/src'),'-I'+linux(VENDOR/'als/src'),
          linux(dest/source.name),*[linux(p) for p in objects],linux(archive),
          '-o',linux(exe),'-lm','-ldl','-rdynamic','-lreadline','-ltermcap',
          '-lpthread','-lstdc++','-lrt','-lstdc++fs','-lboost_timer','-lboost_system']
    save(dest/'command.json',args)
    proc=subprocess.run(tool_command(*args, distro='Ubuntu'),capture_output=True,timeout=90)
    (dest/'build.log').write_bytes(proc.stdout+proc.stderr)
    if proc.returncode: raise RuntimeError('Initial-only harness build failed: '+str(dest/'build.log'))
    return exe


def normalized_author_blif(model, mapping, reference, path):
    inputs,outs,nodes=model
    names={n:'pi_%d'%reference.inputs.index(mapping['inputs'][n]) for n in inputs}
    for ins,out,cubes in nodes:
        for n in [*ins,out]:
            if n not in names: names[n]='a_%d'%len(names)
    lines=['.model author_accurate','.inputs '+' '.join('pi_%d'%i for i in range(len(inputs))),
           '.outputs '+' '.join('po_%d'%i for i in range(len(outs)))]
    for ins,out,cubes in nodes:
        lines.append('.names '+' '.join(names[n] for n in [*ins,out]))
        lines.extend(' '.join(cube) for cube in cubes)
    for i,n in enumerate(reference.outputs):
        raw=next(raw for raw,aligned in mapping['outputs'].items() if aligned==n)
        lines += ['.names %s po_%d'%(names[raw],i),'1 1']
    path.write_text('\n'.join(lines+['.end'])+'\n',encoding='utf-8')


def prove(folder):
    q=lambda n:ExactBackend.quote(linux(folder/n))
    command='cec %s %s; cec %s %s'%(q('author.blif'),q('common.blif'),q('dc.blif'),q('common.blif'))
    p=subprocess.run(tool_command('/usr/local/bin/abc', '-c', command, distro='Ubuntu'),capture_output=True,timeout=45)
    log=(p.stdout+p.stderr).decode('utf-8',errors='replace').replace('\x00','')
    (folder/'cec.log').write_text(log,encoding='utf-8')
    if p.returncode or log.count('Networks are equivalent')!=2:
        raise ValueError('Accurate baseline CEC proof missing: '+str(folder/'cec.log'))
    return dict(author_to_common=True,dc_to_common=True,completed_proofs=2)


def prepare(folder):
    folder=Path(folder).resolve()
    if False: raise ValueError('Keep baseline files on D:')
    if folder.exists():
        verify_bundle(folder)
        print('[RESUB BASELINE REUSE] '+str(folder),flush=True);return
    manifest,objects,archive,dependencies=author_dependencies()
    folder.mkdir(parents=True)
    exe=build(folder,objects,archive)
    sys.path.insert(0,str(AUTHOR))
    from author_netlist_io import read_mapped
    lib=Library(LIBRARY_PATH);authorlib=VENDOR/'input/standard-cell/nangate_45nm_typ.lib'
    other=Library(authorlib)
    if set(lib.cells)!=set(other.cells) or any((c.area_decimal,c.inputs,c.tables)!=(other.cells[n].area_decimal,other.cells[n].inputs,other.cells[n].tables) for n,c in lib.cells.items()):
        raise ValueError('Project and author supported combinational libraries differ')
    dependencies.update({str(p):sha(p) for p in (authorlib,LIBRARY_PATH,AUTHOR/'author_netlist_io.py',ROOT/'resub_initial_only.cc',ROOT/'prepare_resub_iscas.py')})
    rows={};started=time.perf_counter()
    for d in DESIGNS:
        print('[RESUB ACCURATE INIT] '+d,flush=True)
        dest=folder/d;dest.mkdir();raw=dest/'author_initial';raw.mkdir()
        blif=VENDOR/'input/benchmark/iscas'/(d+'.blif')
        # Freeze source content too: the bundle remains independently auditable.
        shutil.copy2(blif,dest/'source.blif')
        tick=time.perf_counter()
        proc=subprocess.run(tool_command(linux(exe), linux(blif), linux(authorlib), linux(raw), distro='Ubuntu'),
                            capture_output=True,timeout=45,cwd=folder)
        (dest/'initialize.log').write_bytes(proc.stdout+proc.stderr)
        if proc.returncode or b'[ACCURATE INITIAL ONLY COMPLETE]' not in proc.stdout:
            raise RuntimeError('Author initial mapping failed: '+str(dest/'initialize.log'))
        dcpath=er_data.source_path(d);dc=load(dcpath,lib);model=read_blif(blif)
        common,mapping=read_mapped(raw/'observed_initial.v',lib,dc,model[0],model[1],dest/'accurate.v')
        restored=load(dest/'accurate.v',lib)
        if structural_id(restored)!=structural_id(common) or restored.units!=common.units:
            raise AssertionError('Baseline serialization changed graph')
        checks=[]
        for split in (0,1,2,100):
            x=er_data.matrix(d,common.inputs,split);packed=pack(x,common.inputs)
            expected=blif_outputs(model,{n:packed[m] for n,m in mapping['inputs'].items()},10000)
            ref={mapping['outputs'][n]:v for n,v in expected.items()}
            for circuit,label in ((common,'common'),(dc,'dc')):
                actual=outputs(circuit,trace(circuit,packed,10000));bad=0
                for n in circuit.outputs:bad |= ref[n]^actual[n]
                alternate=validate_numpy(circuit,x)
                for i,n in enumerate(circuit.outputs):
                    if int.from_bytes(np.packbits(alternate[:,i],bitorder='little').tobytes(),'little')!=actual[n]:
                        raise AssertionError('Independent simulator mismatch')
                if bad: raise ValueError('Author accurate functionality mismatch: '+d)
            checks.append(dict(split=split,vectors=10000,wrong_vectors=0,independent_numpy_agrees=True))
        write_blif(common,dest/'common.blif');write_blif(dc,dest/'dc.blif')
        normalized_author_blif(model,mapping,dc,dest/'author.blif')
        proof=prove(dest)
        old=AUTHOR/'runs/iscas85_nmhd5_original_v3'/d/'observed_initial.v'
        oldsame=None
        if old.exists():
            oldc,_=read_mapped(old,lib,dc,model[0],model[1],dest/'historical_check.v')
            oldsame=structural_id(oldc)==structural_id(common)
            (dest/'historical_check.v').unlink();(dest/'historical_check.adapter.v').unlink()
        rows[d]=dict(path=str(dest/'accurate.v'),sha256=sha(dest/'accurate.v'),area=lib.area(common.units),
            gates=len(common.gates),units=common.units,graph=structural_id(common),port_mapping=mapping,
            source_blif=str(blif),source_blif_sha256=sha(blif),dc_reference=str(dcpath),dc_area=lib.area(dc.units),
            source='ResubALS accurate BLIF, original ALSMan::Eval(initial=true), port rename only',
            initial_mapper='st; dch; amap',approximate_search=False,checks=checks,formal_equivalence=proof,
            same_as_historical_initial=oldsame,preparation_seconds=time.perf_counter()-tick)
        for p in (blif,dcpath):dependencies[str(p)]=sha(p)
        print('[RESUB BASELINE VERIFIED] '+json.dumps(dict(design=d,area=rows[d]['area'],gates=rows[d]['gates'],cec_proofs=2,historical_structure_matches=oldsame)),flush=True)
    for path,h in dependencies.items():
        if sha(path)!=h:raise AssertionError('Author dependencies changed during preparation: '+path)
    # Freeze every generated source/proof and executable; no per-run regeneration.
    files={p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file()}
    data=dict(schema=1,baseline_kind='resubals_author_accurate_initial',author_repository=manifest['author_repository'],
        author_commit=manifest['author_commit'],abc_commit=manifest['abc_commit'],designs=rows,
        dependencies=dependencies,files=files,library_sha256=sha(LIBRARY_PATH),author_library_sha256=sha(authorlib),
        combinational_cells_identical=len(lib.cells),approximate_search_called=False,api_calls=0,
        seconds=time.perf_counter()-started,normalization='raw Liberty physical area; not rounded ABC per-cell display',
        limitation='Common mapped accurate start, not a claim of identical internal SOP/AIG search representation, vectors, or paper result.')
    save(folder/'baseline.json',data);verify_bundle(folder)


def verify_bundle(folder=DEFAULT,design=None):
    folder=Path(folder);p=read(folder/'baseline.json')
    if p.get('relocatable'):
        from portable_assets import verify_baseline
        if set(p['designs'])!=set(DESIGNS):raise ValueError('Incomplete accurate initialization bundle')
        return verify_baseline(folder,design)
    if p.get('baseline_kind')!='resubals_author_accurate_initial' or p.get('approximate_search_called') is not False:
        raise ValueError('Not an author accurate initialization bundle')
    if set(p['designs'])!=set(DESIGNS):raise ValueError('Incomplete ResubALS baseline bundle; no DC fallback')
    for name,h in p['files'].items():
        if sha(folder/name)!=h:raise ValueError('Frozen ResubALS baseline changed: '+name)
    for path,h in p['dependencies'].items():
        if sha(path)!=h:raise ValueError('Baseline provenance dependency changed: '+path)
    rows=[p['designs'][design]] if design else p['designs'].values()
    for row in rows:
        if not all(row['formal_equivalence'][k] for k in ('author_to_common','dc_to_common')):
            raise ValueError('Missing accurate baseline CEC')
        if sha(row['path'])!=row['sha256']:raise ValueError('Baseline input changed')
    return p


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default=str(DEFAULT));p.add_argument('--verify-only',action='store_true')
    a=p.parse_args();verify_bundle(a.output) if a.verify_only else prepare(a.output)
