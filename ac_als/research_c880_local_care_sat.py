"""Screened local exact synthesis, preserving the rest of the mapped netlist."""
from portable_tools import tool_command
import runtime
import argparse,time,json,subprocess
from pathlib import Path
from collections import defaultdict
from logic import Library,Plan
from netlist import Circuit,ZERO,ONE
from bacs_data import load,LIBRARY_PATH
from bacs_numeric import cleanup,save
from bacs_frontier import VirtualArea
from cutmap import CutMapper,variable
from numeric_exact import ExactBackend,restore_ports
from research_c880_local_exact import cuts,cut_truth
from research_c880_hybrid import proof
from prepare_resub_iscas import sha
from prepare_resub_iscas import DEFAULT
from simulate import trace,outputs,pack
import er_data

def blif_truth(path,n,truth):
    lines=['.model cut_synthesis','.inputs '+' '.join('pi_%d'%i for i in range(n)),'.outputs po_0']
    if truth in (0,(1<<(1<<n))-1):
        lines.append('.names po_0')
        if truth:lines.append('1')
        path.write_text('\n'.join(lines+['.end'])+'\n',encoding='utf-8');return
    lines.append('.names '+' '.join([*('pi_%d'%i for i in range(n)),'po_0']))
    for row in range(1<<n):
        if (truth>>row)&1:lines.append(''.join(str((row>>i)&1) for i in range(n))+' 1')
    path.write_text('\n'.join(lines+['.end'])+'\n',encoding='utf-8')

def to_plan(local):
    index={n:i for i,n in enumerate(local.inputs)};memo={}
    def visit(n):
        n=local.resolve(n)
        if n in index:return Plan('leaf',leaf=index[n])
        if n in (ZERO,ONE):return Plan('leaf',leaf=-2 if n==ZERO else -1)
        if n in memo:return memo[n]
        g=local.producer[n];cell=local.lib.cells[g.cell]
        if len(cell.functions)!=1:raise ValueError('Unexpected multioutput local mapping')
        children=tuple(visit(g.pins[p]) for p in cell.inputs)
        memo[n]=Plan(g.cell,children,units=cell.units+sum(c.units for c in children));return memo[n]
    return visit(local.outputs[0])

def run(a):
    folder=Path(a.output).resolve();folder.mkdir(parents=True,exist_ok=False)
    (folder/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    save(folder/'protocol.json',dict(vars(a),input_sha256=sha(a.input),approximate_changes=a.error_inclusion,
        note='bounded cut beam, batch local synthesis; final full CEC required',api_calls=0))
    lib=Library(LIBRARY_PATH);source=load(a.input,lib);source.write(folder/'input.v');best=source
    mapper=CutMapper(lib,4);stats=defaultdict(int);events=[];tick=time.perf_counter();proof_count=0
    design=getattr(a,'design','c880')
    reference=load(Path(getattr(a,'baseline',DEFAULT))/design/'accurate.v',lib) if a.error_inclusion else None
    sampled_inputs=[pack(er_data.matrix(design,source.inputs,k),source.inputs) for k in (0,1)] if a.sampled_care else []
    allowed=[];mask=(1<<10000)-1
    if a.error_inclusion:
        for p in sampled_inputs:
            old=outputs(source,trace(source,p,10000));gold=outputs(reference,trace(reference,p,10000));bad=0
            for o in source.outputs:bad |= old[o]^gold[o]
            allowed.append(bad)
    for iteration in range(a.rounds):
        if time.perf_counter()-tick>a.seconds-3:break
        work=folder/('round_%d'%iteration);work.mkdir();area=VirtualArea(best,mapper);candidates=[]
        for target in sorted(best.producer,key=lambda n:(-area.cones.potential(n),n))[:80]:
            possible=[]
            for names in cuts(best,target,width=a.width):
                if len(names)<3:continue
                gain=area.cones.saved_with_inputs(target,names)
                if gain<5:continue
                possible.append((gain,names))
            # Reserve both a compact boundary and a larger expanded boundary.
            selected=[]
            for queue in (sorted(possible,key=lambda r:(-r[0],len(r[1]),r[1])),
                          sorted(possible,key=lambda r:(-r[0],-len(r[1]),r[1]))):
                for row in queue[:2]:
                    if row not in selected:selected.append(row)
            for gain,names in selected:candidates.append((gain,target,names))
        candidates=sorted(candidates,key=lambda r:(-r[0],r[1],r[2]))[:a.cuts]
        contexts=[trace(best,p,10000,stats) for p in sampled_inputs];current_outputs=[outputs(best,v) for v in contexts]
        care_cache={}
        def care_table(target,names):
            if target not in care_cache:
                care_cache[target]=[]
                for k,p in enumerate(sampled_inputs):
                    flipped=outputs(best,trace(best,p,10000,stats,base=contexts[k],flip=target));care=0
                    for o in best.outputs:care |= current_outputs[k][o]^flipped[o]
                    if a.error_inclusion:care &= allowed[k]^mask
                    care_cache[target].append(care)
            defined=0
            for k in (0,1):
                regions=[care_cache[target][k]]
                for n in names:
                    v=contexts[k][n];regions=[r&(v^mask) for r in regions]+[r&v for r in regions]
                for row,r in enumerate(regions):
                    if r:defined |= 1<<row
            return defined
        commands=['read_lib -w '+ExactBackend.quote(ExactBackend.linux(LIBRARY_PATH))];rows=[]
        for gain,target,names in candidates:
            original=cut_truth(best,target,names);truths=[original]
            if a.sampled_care:
                defined=care_table(target,names);full=(1<<(1<<len(names)))-1
                truths=[original]
            for truth in truths:
                index=len(rows);case=work/('cut_%03d'%index);case.mkdir();blif_truth(case/'input.blif',len(names),truth)
                q=lambda n:ExactBackend.quote(ExactBackend.linux(case/n))
                defined=care_table(target,names)
                blif_truth(case/'care.blif',len(names),defined)
                commands.append('read_blif %s; strash; if -K 8 -a -C 64; care_set %s; mfs -a -r -W 0 -F 300 -D 0 -M 1600 -L 8 -C 20000; strash; dch; amap -F 4 -A 20 -C 2000 -m -s; write_verilog %s; write_blif %s'%(
                    q('input.blif'),q('care.blif'),q('mapped.v'),q('mapped.blif')))
                rows.append(dict(index=index,target=target,names=names,truth=truth,original_truth=original,potential=gain,care_truth=defined))
        script=work/'batch.abc';script.write_text('\n'.join(commands)+'\n',encoding='utf-8')
        remaining=max(1,a.seconds-(time.perf_counter()-tick))
        try:
            p=subprocess.run(tool_command('/usr/local/bin/abc', '-c', 'source '+ExactBackend.quote(ExactBackend.linux(script)), distro='Ubuntu'),capture_output=True,timeout=remaining)
            log=(p.stdout+p.stderr).decode('utf-8',errors='replace');(work/'abc.log').write_text(log,encoding='utf-8')
            stats['local_synthesis_jobs']+=len(rows)
            stats['local_care_optimizations']+=log.count('external care clauses')
            if p.returncode:raise RuntimeError('Local ABC batch failed; see '+str(work/'abc.log'))
            if log.count('external care clauses')!=len(rows):raise RuntimeError('Incomplete local ABC batch; see '+str(work/'abc.log'))
        except subprocess.TimeoutExpired:
            stats['batch_timeout']+=1;break
        proposals=[]
        for row in rows:
            case=work/('cut_%03d'%row['index'])
            if not (case/'mapped.v').exists():continue
            interface=Circuit(lib,row['names'],('__local_result',),[],{'__local_result':ZERO})
            try:
                mapped=restore_ports(case/'mapped.v',interface)
                tt=outputs(mapped,trace(mapped,{n:variable(len(row['names']),i) for i,n in enumerate(row['names'])},1<<len(row['names'])))[mapped.outputs[0]]
                if (tt^row['original_truth'])&row['care_truth']:raise AssertionError('Local external-care function verification failed')
                stats['exhaustive_local_care_truth_checks']+=1
                plan=to_plan(mapped)
                trial=cleanup(best.materialize([(row['target'],row['names'],plan)]),mapper)
            except ValueError as exc:row['error']=str(exc);continue
            row.update(mapped_local_units=mapped.units,actual_units=trial.units)
            stats['actual_local_embeddings']+=1
            if trial.units<best.units:proposals.append((trial,row))
        save(work/'candidates.json',rows)
        proposals.sort(key=lambda r:(r[0].units,r[1]['index']));accepted=None
        for trial,row in proposals[:12]:
            if time.perf_counter()-tick>a.seconds-1:break
            if a.error_inclusion:
                from research_error_relation import prove_inclusion
                same=prove_inclusion(reference,source,trial,folder/('global_cec_%03d'%proof_count))
            else:same=proof(source,trial,folder/('global_cec_%03d'%proof_count))
            proof_count+=1;stats['global_cec_attempts']+=1
            if same:accepted=(trial,row);break
        if accepted is None:break
        best,row=accepted;best.write(folder/'best.v');events.append(dict(row,area=lib.area(best.units),cec=True))
        print('[CUT ABC ACCEPT] '+json.dumps(events[-1]),flush=True)
    best.write(folder/'best.v');result=dict(initial_area=lib.area(source.units),area=lib.area(best.units),
        events=events,stats=dict(stats),seconds=time.perf_counter()-tick,approximate_changes=a.error_inclusion,api_calls=0)
    save(folder/'report.json',result);print('[CUT ABC RESULT] '+json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seconds',type=float,default=90);p.add_argument('--width',type=int,choices=(6,7,8,9,10),default=8)
    p.add_argument('--cuts',type=int,default=96);p.add_argument('--rounds',type=int,default=4)
    p.add_argument('--sampled-care',action='store_true');p.add_argument('--error-inclusion',action='store_true')
    a=p.parse_args()
    a.sampled_care=True
    if a.error_inclusion and not a.sampled_care:p.error('--error-inclusion requires --sampled-care')
    run(a)
