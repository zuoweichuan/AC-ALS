"""Content-checked, relocatable benchmark assets; never download implicitly."""
import hashlib,json,os
from pathlib import Path


def asset_root():
    base=Path(os.environ.get('NETLIST_FRONTIER_ROOT',os.environ.get('BACS_PROJECT_ROOT',str(Path(__file__).resolve().parent))))
    return Path(os.environ.get('AC_ALS_ASSETS',str(base/'benchmark_assets'))).resolve()


def contained(folder,name):
    target=(Path(folder)/name).resolve()
    if not target.is_relative_to(Path(folder).resolve()):raise ValueError('Asset path escapes bundle')
    return target


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_assets():
    folder=asset_root();path=folder/'bundle.json'
    if not path.is_file():raise FileNotFoundError('Install an authorized asset bundle first: '+str(path))
    data=json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema')!=1:raise ValueError('Unsupported asset bundle')
    for name,value in data['files'].items():
        if digest(contained(folder,name))!=value:raise ValueError('Asset content mismatch: '+name)
    return folder,data


def verify_baseline(folder,design=None):
    folder=Path(folder).resolve();data=json.loads((folder/'baseline.json').read_text(encoding='utf-8'))
    if not data.get('relocatable') or data.get('approximate_search_called') is not False:
        raise ValueError('Not a portable accurate initialization bundle')
    for name,value in data['files'].items():
        if digest(contained(folder,name))!=value:raise ValueError('Baseline file changed: '+name)
    for name,row in data['designs'].items():
        if design is not None and name!=design:continue
        if not all(row['formal_equivalence'][key] for key in ('author_to_common','dc_to_common')):
            raise ValueError('Missing original formal baseline proof')
        path=contained(folder,row['path'])
        if digest(path)!=row['sha256']:raise ValueError('Accurate input changed: '+name)
    for row in data['designs'].values():row['path']=str(contained(folder,row['path']))
    return data
