"""Bounded large-cut care synthesis for batch proposals, never author search.

ABC only completes/mappings selected local functions. No candidate full netlist
is materialized here. Root effects are recollected from the current physical
circuit; combined actions are later judged by exact current-plan response.
"""
from portable_tools import tool_command
import subprocess,time
from pathlib import Path
from bacs_data import LIBRARY_PATH
from bacs_numeric import save
from netlist import Circuit,ZERO
from cutmap import variable
from simulate import trace,outputs
from numeric_exact import ExactBackend,restore_ports
from research_plan_overlay import Action
from research_c880_local_exact import cuts,cut_truth
from research_c880_local_care_sat import blif_truth,to_plan
from research_c880_budget_care_sat import regions_for,free_rows


def candidates(info,folder,cap,seconds,stats):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    o=info['overlay'];c=o.base;ctx=info['context'];mask=o.mask;rows=[]
    roots=sorted(info['records'],key=lambda r:(-r['mffc_units'],r['root']))[:6]
    for record in roots:
        target=record['root'];possible=[]
        for ns in cuts(c,target,width=8,beam=2):
            if not 4<=len(ns)<=8:continue
            gain=info['cones'].saved_with_inputs(target,ns)
            if gain>=4:possible.append((gain,ns))
        for gain,names in sorted(possible,key=lambda r:(-r[0],len(r[1]),r[1]))[:2]:
            original=cut_truth(c,target,names);n=len(names);regions=[regions_for(names,v,mask) for v in ctx['state']['values']]
            forced=ctx['responses'][target]['forced'];teacher=original;teacher_loss=[0,0]
            for row in range(1<<n):
                bit=(original>>row)&1
                old=[(regions[k][row]&forced[k][bit]).bit_count() for k in (0,1)]
                alternative=[(regions[k][row]&forced[k][1-bit]).bit_count() for k in (0,1)]
                if all(alternative[k]<=old[k] for k in (0,1)) and sum(alternative)<sum(old):
                    teacher^=1<<row;old=alternative
                for k in (0,1):teacher_loss[k]+=old[k]
            risks=[]
            for row in range(1<<n):
                bit=(teacher>>row)&1
                risks.append(tuple((regions[k][row]&forced[k][1-bit]&(forced[k][bit]^mask)).bit_count() for k in (0,1)))
            budgets=[max(0,cap-e) for e in teacher_loss]
            freed,used=free_rows(risks,budgets)
            case=folder/('cut_%02d'%len(rows));case.mkdir()
            blif_truth(case/'input.blif',n,teacher);care=((1<<(1<<n))-1)^freed;blif_truth(case/'care.blif',n,care)
            rows.append(dict(target=target,names=names,teacher=teacher,care=care,case=str(case),new_errors_bound=used))
    if not rows:return []
    q=lambda p:ExactBackend.quote(ExactBackend.linux(p))
    commands=['read_lib -w '+q(LIBRARY_PATH)]
    for r in rows:
        p=Path(r['case'])
        commands.append('read_blif %s; strash; if -K 8 -a -C 64; care_set %s; mfs -a -r -W 0 -F 300 -D 0 -M 1600 -L 8 -C 20000; strash; dch; amap -F 4 -A 20 -C 2000 -m -s; write_verilog %s'%(
            q(p/'input.blif'),q(p/'care.blif'),q(p/'mapped.v')))
    script=folder/'batch.abc';script.write_text('\n'.join(commands)+'\n',encoding='utf-8')
    temporary=folder/'tmp';temporary.mkdir()
    tick=time.perf_counter()
    try:
        p=subprocess.run(tool_command('/usr/bin/env', 'TMPDIR='+ExactBackend.linux(temporary), '/usr/bin/prlimit', '--core=0', '--', '/usr/bin/timeout', '--signal=KILL', str(max(1,round(seconds))), '/usr/local/bin/abc', '-c', 'source '+q(script), distro='Ubuntu'),capture_output=True,timeout=max(1,seconds)+4)
    except (subprocess.TimeoutExpired,OSError):
        stats['wide_backend_unavailable_or_timeout']+=1;return []
    stats['wide_synthesis_milliseconds']+=round(1000*(time.perf_counter()-tick))
    log=(p.stdout+p.stderr).decode('utf-8',errors='replace');(folder/'abc.log').write_text(log,encoding='utf-8')
    if p.returncode:stats['wide_partial_or_timeout_batches']+=1
    result=[]
    for r in rows:
        mapped_path=Path(r['case'])/'mapped.v'
        if not mapped_path.exists():continue
        interface=Circuit(c.lib,r['names'],('__local_result',),[],{'__local_result':ZERO})
        mapped=restore_ports(mapped_path,interface);n=len(r['names'])
        truth=outputs(mapped,trace(mapped,{name:variable(n,i) for i,name in enumerate(r['names'])},1<<n))[mapped.outputs[0]]
        if (truth^r['teacher'])&r['care']:raise AssertionError('Wide synthesis violated retained care rows')
        stats['wide_local_truth_checks']+=1
        result.append(Action(r['target'],tuple(r['names']),to_plan(mapped)))
    save(folder/'candidates.json',rows);stats['wide_actions_generated']+=len(result)
    return result
