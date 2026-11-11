"""Area-oriented exact representation portfolio with ODC resubstitution.

All actual selected circuits are CEC-equivalent to the supplied warm parent.
No original approximate solver is called, and all added/inverting cells count.
"""
import runtime
import argparse,time,json,random
from pathlib import Path
from collections import defaultdict
from logic import Library
from netlist import Circuit
from bacs_data import load,LIBRARY_PATH
from bacs_numeric import save
from numeric_exact import ExactBackend,restore_ports
from research_c880_hybrid import proof
from hybrid_identity import structural_id
from prepare_resub_iscas import sha

MAPPERS=(
    'amap -F 4 -A 20 -C 2000 -m -s',
    'amap -F 4 -A 20 -C 2000 -x -s',
    'amap -F 4 -A 20 -C 2000 -m -i -s',
    'amap -F 8 -A 40 -C 4000 -s',
)
REWRITES=(
    'strash; dch',
    'strash; resub -K 16 -N 3 -F 4 -z -l; rewrite -z -l; refactor -z -l; resub -K 16 -N 3 -F 8 -z -l; dch',
    'strash; dc2; resub -K 12 -N 3 -F 8 -z -l; resub -K 16 -N 3 -F 16 -z -l; dch -x',
    'strash; '+ExactBackend.COMPRESS+'; dch -x',
    'strash; ifraig; dc2; resub -K 16 -N 3 -F 8 -z; rewrite -z; refactor -z; balance; dch -f',
    'strash',
)

def run(a):
    folder=Path(a.output).resolve();folder.mkdir(parents=True,exist_ok=False);(folder/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    lib=Library(LIBRARY_PATH);source=load(a.input,lib);source.write(folder/'input.v')
    save(folder/'protocol.json',dict(vars(a),source_sha256=sha(a.input),api_calls=0,approximate_changes=False,
        note='mapping flags only propose implementations; raw physical area and completed whole CEC determine acceptance'))
    tick=time.perf_counter();remaining=lambda:max(0,a.seconds-(time.perf_counter()-tick));rng=random.Random(a.seed)
    best=source;frontier=[source];seen={structural_id(source)};events=[];rows=[];proofs=0
    scripts=[r+'; '+m for r in REWRITES for m in MAPPERS]
    for iteration in range(a.rounds):
        if remaining()<2:break
        candidates=[]
        for parent_index,parent in enumerate(frontier[:4]):
            if remaining()<2:break
            # Port-order changes alter deterministic synthesis choices; restore
            # the original interface before any cross-parent equivalence proof.
            if iteration:
                ins=list(parent.inputs);outs=list(parent.outputs);rng.shuffle(ins);rng.shuffle(outs)
                proposed=Circuit(parent.lib,ins,outs,parent.gates,parent.aliases,parent.name)
            else:proposed=parent
            backend=ExactBackend(folder/('round%d_parent%d'%(iteration,parent_index)),LIBRARY_PATH,timeout=10)
            backend.SCRIPTS=tuple(scripts if iteration==0 else scripts[::3])
            backend.optimize(proposed,remaining,variants=len(backend.SCRIPTS))
            for index,row in enumerate(backend.rows):
                rows.append(dict(row,iteration=iteration,parent=parent_index))
                if not row.get('cec_equivalent'):continue
                c=restore_ports(backend.folder/('map_%04d'%index)/'mapped.v',proposed)
                c=Circuit(lib,source.inputs,source.outputs,c.gates,c.aliases,source.name)
                identity=structural_id(c)
                if identity in seen:continue
                seen.add(identity);candidates.append(c)
        if not candidates:break
        candidates.sort(key=lambda c:(c.units,structural_id(c)))
        if candidates[0].units<best.units:
            candidate=candidates[0]
            same=proof(source,candidate,folder/('selected_cec_%d'%proofs));proofs+=1
            if not same:raise AssertionError('Exact portfolio original-interface CEC failed')
            best=candidate;best.write(folder/'best.v');event=dict(iteration=iteration,area=lib.area(best.units),units=best.units,cec=True)
            events.append(event);print('[EXACT FRONTIER ACCEPT] '+json.dumps(event),flush=True)
        frontier=[best,*[c for c in candidates if structural_id(c)!=structural_id(best)][:3]]
    best.write(folder/'best.v');result=dict(initial_area=lib.area(source.units),area=lib.area(best.units),
        events=events,rows=rows,seconds=time.perf_counter()-tick,api_calls=0,approximate_changes=False,
        completed_mapping_proofs=sum(r.get('cec_equivalent',False) for r in rows),selected_global_proofs=proofs)
    save(folder/'report.json',result);print('[EXACT FRONTIER RESULT] '+json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seconds',type=float,default=90);p.add_argument('--rounds',type=int,default=3);p.add_argument('--seed',type=int,default=20261028);run(p.parse_args())
