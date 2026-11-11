"""Maintainer-only local export. Third-party assets remain outside Git."""
import argparse,hashlib,json,shutil
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--research-root',type=Path,required=True,help='Original netlist_frontier directory')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();source=a.research_root.resolve();out=a.output.resolve()
    if out.exists():raise ValueError('Use a fresh asset output directory')
    out.mkdir(parents=True)
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    files={}
    def copy(path,name,expected=None):
        if expected and sha(path)!=expected:raise ValueError('Original asset changed: '+name)
        dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest);files[name]=sha(dest)
    protocol=read(source/'runs/deepseek_four_configs_5seeds_v1/protocol.json')
    copy(source/'assets/stdcells.lib','library/stdcells.lib',protocol['library_sha256'])
    copy(source/'runs/deepseek_four_configs_5seeds_v1/inputs/bacs_knowledge.json','knowledge/bacs_learned.json',protocol['knowledge_sha256'])
    original=source/'baselines/resub_iscas_accurate_v1';baseline=read(original/'baseline.json')
    baseline_files={};rows={}
    for name,row in baseline['designs'].items():
        for filename in ('accurate.v','source.blif','author.blif','common.blif','dc.blif','cec.log'):
            path=original/name/filename
            if not path.is_file():raise FileNotFoundError(path)
            relative=name+'/'+filename;copy(path,'iscas/'+relative,baseline['files'][relative]);baseline_files[relative]=files['iscas/'+relative]
        text=(original/name/'cec.log').read_text(encoding='utf-8',errors='replace')
        if text.count('Networks are equivalent')<2:raise ValueError('Missing two baseline proofs: '+name)
        record={k:v for k,v in row.items()if k not in ('path','source_blif','dc_reference')}
        record['path']=name+'/accurate.v';record['source_blif']=name+'/source.blif';rows[name]=record
    portable={k:v for k,v in baseline.items()if k not in ('designs','dependencies','files')}
    portable.update(relocatable=True,designs=rows,files=baseline_files,dependencies={},
                    provenance=dict(original_manifest_sha256=sha(original/'baseline.json'),
                        original_dependency_sha256=sorted(baseline['dependencies'].values()),
                        scope='Archived original CEC proofs, not fresh generation or author-tool execution'))
    (out/'iscas/baseline.json').write_text(json.dumps(portable,indent=2),encoding='utf-8')
    files['iscas/baseline.json']=sha(out/'iscas/baseline.json')
    bacs={}
    for name in ('abs_diff','adder32','buttfly','mac','mult8','mult16'):
        row=protocol['inputs'][name];path=Path(row['path'])
        copy(path,'bacs/accurate/'+name+'.v',row['sha256'])
        copy(source.parent/'compensation_aware_als/designs/BACS/out'/name/(name+'_syn.v'),'bacs/reference/'+name+'/'+name+'_syn.v')
        copy(source/'.tmp/paper_author/ResubALS-master/input/benchmark/bacs'/(name+'.blif'),'bacs/blif/'+name+'.blif')
        bacs[name]=dict(path='bacs/accurate/'+name+'.v',sha256=row['sha256'],area=row['area'])
    data=dict(schema=1,files=files,bacs_inputs=bacs,knowledge='knowledge/bacs_learned.json',
              original_protocol_sha256=sha(source/'runs/deepseek_four_configs_5seeds_v1/protocol.json'),
              redistribution='Local export only; this manifest does not grant third-party redistribution permission')
    (out/'bundle.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps(dict(files=len(files),output=str(out),published=False)))


if __name__=='__main__':main()
