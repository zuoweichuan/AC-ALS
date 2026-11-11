"""Install a separately acquired authorized bundle after verifying all hashes."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ac_als'))
from portable_assets import contained


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bundle',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'ac_als/benchmark_assets');a=p.parse_args()
    source=a.bundle.resolve();dest=a.output.resolve();data=json.loads((source/'bundle.json').read_text(encoding='utf-8'))
    if data.get('schema')!=1:raise ValueError('Unsupported bundle schema')
    for name,expected in data['files'].items():
        if hashlib.sha256(contained(source,name).read_bytes()).hexdigest()!=expected:raise ValueError('Asset checksum mismatch: '+name)
    if dest.exists():raise ValueError('Use a fresh destination; existing assets are not overwritten')
    dest.mkdir(parents=True)
    for name in data['files']:
        target=contained(dest,name);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(contained(source,name),target)
    shutil.copy2(source/'bundle.json',dest/'bundle.json');print('Installed verified asset bundle.')


if __name__=='__main__':main()
