"""Direct technology-mapped SAT resynthesis, preserving mapped structure.

ABC mfs3 optimizes a mapped physical network, unlike earlier LUT/AIG mfs/mfs2
passes. Every proposal is parsed back, priced with original Liberty precision,
and proved equivalent to the original input before acceptance. No ALS is
delegated and no approximate Monte Carlo acceptance is used here.
"""
from portable_tools import tool_command
import runtime
import argparse,time,json,subprocess
from pathlib import Path
from logic import Library
from bacs_data import load,LIBRARY_PATH
from bacs_numeric import save
from numeric_exact import ExactBackend
from prepare_resub_iscas import sha
from research_c880_hybrid import proof
from hybrid_identity import structural_id

SCRIPTS=(
    'mfs3 -a -e -K 6 -L 1 -H 12 -R 4 -F 100 -V 20 -M 1200 -C 20000',
    'mfs3 -a -r -e -K 6 -L 1 -H 12 -R 4 -F 100 -V 20 -M 1200 -C 20000',
    'mfs3 -a -m -e -K 8 -L 1 -H 20 -R 4 -F 100 -V 30 -M 1600 -C 20000',
    'mfs3 -a -r -m -e -K 8 -L 1 -H 20 -R 4 -F 100 -V 30 -M 1600 -C 20000',
    'mfs3 -a -z -e -K 6 -L 1 -H 20 -R 4 -F 100 -V 20 -M 1600 -C 20000',
    'mfs3 -a -r -z -m -e -K 8 -L 1 -H 30 -R 4 -F 100 -V 30 -M 1600 -C 20000',
)
WIDE_SCRIPTS=tuple('mfs3 -a '+flags+' -K %d -L 1 -H 64 -R 4 -F 1000 -V 100 -M 5000 -C 100000'%width
    for flags,width in (('-e',8),('-e -r',8),('-e -z',8),('-e -z -r',8),('-e -s -z',8),('-e -s -r -z',8),('-e -z',4),('-e -r -z',4)))


def run(a):
    folder=Path(a.output).resolve();folder.mkdir(parents=True,exist_ok=False)
    (folder/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    lib=Library(LIBRARY_PATH);source=load(a.input,lib);source.write(folder/'input.v');best=source
    save(folder/'protocol.json',dict(vars(a),input_sha256=sha(a.input),api_calls=0,approximate_changes=False,
        technology_mapped_input=True,monte_carlo_for_error_acceptance=False,read_mode='read_verilog -m',
        validation='full original-function CEC and raw Liberty area for every proposal'))
    tick=time.perf_counter();rows=[];events=[];frontier=[source];seen={structural_id(source)}
    for iteration in range(a.rounds):
        children=[]
        for parent in frontier[:2]:
            for script in (WIDE_SCRIPTS if a.wide else SCRIPTS):
                remaining=a.seconds-(time.perf_counter()-tick)
                if remaining<2:break
                case=folder/('case_%03d'%len(rows));case.mkdir();parent.write(case/'input.v')
                q=lambda p:ExactBackend.quote(ExactBackend.linux(p))
                command='read_lib -w %s; read_verilog -m %s; %s; write_verilog %s; write_blif %s'%(q(LIBRARY_PATH),q(case/'input.v'),script,q(case/'mapped.v'),q(case/'mapped.blif'))
                row=dict(script=script,iteration=iteration,parent_area=lib.area(parent.units),accepted=False,source_sha256=sha(case/'input.v'));start=time.perf_counter()
                try:
                    proc=subprocess.run(tool_command('env', 'TMPDIR='+ExactBackend.linux(case), 'prlimit', '--core=0:0', '/usr/local/bin/abc', '-c', command, distro='Ubuntu'),capture_output=True,timeout=min(a.case_seconds,remaining))
                    log=(proc.stdout+proc.stderr).decode('utf-8',errors='replace');(case/'abc.log').write_text(log,encoding='utf-8')
                    row['returncode']=proc.returncode
                    if proc.returncode or not (case/'mapped.v').exists():raise ValueError('Mapped SAT pass incomplete; inspect abc.log')
                    child=load(case/'mapped.v',lib);row['area']=lib.area(child.units)
                    if child.inputs!=source.inputs or child.outputs!=source.outputs:raise ValueError('Mapped SAT interface changed')
                    equivalent=proof(source,child,case/'cec');row['cec_equivalent']=equivalent
                    if not equivalent:raise ValueError('Mapped SAT pass failed original whole-circuit CEC')
                    identity=structural_id(child)
                    if identity not in seen:children.append(child);seen.add(identity)
                    if child.units<best.units:
                        best=child;best.write(folder/'best.v');row['accepted']=True
                        events.append(dict(area=lib.area(best.units),script=script,iteration=iteration,case=case.name,cec=True))
                        print('[MAPPED SAT ACCEPT] '+json.dumps(events[-1]),flush=True)
                except (ValueError,OSError,subprocess.TimeoutExpired) as exc:row['error']=str(exc)
                row['seconds']=time.perf_counter()-start;rows.append(row);save(folder/'measurements.json',rows)
        if not children:break
        children.sort(key=lambda c:(c.units,structural_id(c)))
        frontier=[best,*[c for c in children if c.units<=best.units+4 and structural_id(c)!=structural_id(best)][:1]]
        if a.seconds-(time.perf_counter()-tick)<2:break
    best.write(folder/'best.v');result=dict(initial_area=lib.area(source.units),area=lib.area(best.units),events=events,rows=rows,
        seconds=time.perf_counter()-tick,cec_proofs=sum(r.get('cec_equivalent',False) for r in rows),api_calls=0,approximate_changes=False)
    save(folder/'report.json',result);print('[MAPPED SAT RESULT] '+json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seconds',type=float,default=150);p.add_argument('--case-seconds',type=float,default=15)
    p.add_argument('--rounds',type=int,default=4);p.add_argument('--wide',action='store_true');run(p.parse_args())
