"""ER joint-plan lessons adapted to numeric NMED, isolated from old flows.

Uses numeric output-word XOR response changes, not ER wrong-vector counts or
additive signed displacements. Multiple-action responses are rankings only;
actual whole-graph train/development loss and physical area decide acceptance.
"""
import runtime
import hashlib,json,shutil,sys,time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from bacs_data import LIBRARY_PATH,load,matrix,preflight
from bacs_numeric import bits,number,number_numpy,Metric,cleanup,save,learned_sets
from bacs_frontier import VirtualArea,pareto_choices,response_truth
from logic import Library
from simulate import trace,outputs,pack,plan_value,validate_numpy
from frontier import forbidden_fanout
from numeric_agentic import Worker,sha,clone_worker
from numeric_residual_windows import parser as base_parser,significant_wire_proposals
from numeric_joint_word import actual_trials
from hybrid_identity import structural_id


@dataclass
class Action:
    move:object
    changes:tuple
    conflicts:frozenset=frozenset()


class ConditionalNumericWorker(Worker):
    """Learn global numeric supports at compression roots, not just repair roots."""
    def candidates(self,repair_only=False):
        found,total=super().candidates(repair_only)
        if self.remaining()<1:return found,total
        tick=time.perf_counter();c=self.current;n=self.count
        values=trace(c,self.inputs[0],n,self.stats);base=number(outputs(c,values),self.order,n)
        losses=np.abs(base-self.reference[0]);area=VirtualArea(c,self.mapper);extra=[]
        roots=sorted(c.producer,key=lambda s:(-area.cones.potential(s),s))[:self.args.conditional_roots]
        for target in roots:
            if self.remaining()<1:break
            fv=trace(c,self.inputs[0],n,self.stats,base=values,flip=target)
            flipped=number(outputs(c,fv),self.order,n);other=np.abs(flipped-self.reference[0])
            target_bits=bits(values[target],n).astype(bool)
            l0=np.where(target_bits,other,losses);l1=np.where(target_bits,losses,other)
            measured=[]
            for names,code in learned_sets(c,target,values,l0,l1,n,self.stats):
                if self.remaining()<.5:break
                for penalty in (1,4):
                    truth=response_truth(code,l0,l1,losses,len(names),penalty);plan=self.mapper.map(len(names),truth)
                    m=self.scored(target,names,truth,plan,values,values[target],base,flipped,losses,other,'global_numeric_condition',penalty)
                    self.stats['numeric_conditional_truths']+=1
                    if m is None:continue
                    m.gain=area.gain(target,names,plan)
                    if m.gain>0 or (m.gain>=-16 and m.error<total):measured.append(m)
            extra+=pareto_choices(measured,8);self.stats['numeric_conditional_roots']+=1
        unique={(m.target,m.names,m.truth):m for m in [*found,*extra]}
        self.stats['numeric_conditional_candidates']+=len(extra)
        self.times['numeric_conditional_construction']+=time.perf_counter()-tick
        return list(unique.values()),total


def numerical_loss(base,change,reference):
    return int(np.abs(np.bitwise_xor(base,change)-reference).sum())


def beam_plans(actions,bases,refs,limit,stats,depth=6,width=4):
    zero=tuple(np.zeros_like(b) for b in bases)
    beam=[(frozenset(),zero,0,tuple(int(np.abs(b-r).sum()) for b,r in zip(bases,refs)))]
    retained={};seen=set()
    def rank(row):
        _,_,gain,loss=row
        excess=sum(max(0,l-limit) for l in loss)
        return excess,-gain,max(loss),sum(loss)
    def queues(rows):
        return [sorted(rows,key=rank),sorted(rows,key=lambda r:(max(r[3]),-r[2],sum(r[3]))),
                sorted(rows,key=lambda r:(-r[2],rank(r)))]
    def choose(rows):
        ordered=queues(rows);picked={}
        for j in range(len(rows)):
            for q in ordered:
                if j<len(q):picked.setdefault(q[j][0],q[j])
                if len(picked)==width:return list(picked.values())
        return list(picked.values())
    for level in range(depth):
        proposals={}
        for ids,change,gain,loss in beam:
            targets={actions[i].move.target for i in ids};children=[]
            for i,a in enumerate(actions):
                if i in ids or ids&a.conflicts or a.move.target in targets:continue
                key=ids|{i}
                if key in seen:continue
                seen.add(key)
                delta=tuple(np.bitwise_xor(p,q) for p,q in zip(change,a.changes))
                scores=tuple(numerical_loss(b,d,r) for b,d,r in zip(bases,delta,refs))
                row=key,delta,gain+a.move.gain,scores
                stats['numeric_joint_plan_proxy_scores']+=1;children.append(row)
            for row in choose(children):proposals[row[0]]=row
        if not proposals:break
        rows=list(proposals.values());beam=choose(rows)
        for row in rows:
            if row[2]>0 and max(row[3])<=limit:retained[row[0]]=row
    return sorted(retained.values(),key=rank)[:12]


def plans(w):
    scan=w.scan();c=w.current;n=w.count;mask=(1<<n)-1
    pool=[m for m in scan['moves'] if m.gain>0 or (m.gain>=-16 and m.error<scan['loss'])]
    queues=[sorted(pool,key=lambda m:(-m.gain,m.error)),sorted(pool,key=lambda m:(m.error,-m.gain)),
            sorted(pool,key=lambda m:(-m.gain/max(1,m.error-scan['loss']),m.error))]
    selected={}
    for rank in range(len(pool)):
        for q in queues:
            if rank<len(q):
                m=q[rank];selected.setdefault((m.target,m.names,m.truth),m)
            if len(selected)==64:break
        if len(selected)==64:break
    values=[trace(c,x,n,w.stats) for x in w.inputs[:2]]
    bases=[number(outputs(c,v),w.order,n) for v in values];flips=[{},{}];actions=[]
    for m in selected.values():
        if w.remaining()<1:break
        changes=[]
        for k in (0,1):
            if m.target not in flips[k]:
                v=trace(c,w.inputs[k],n,w.stats,base=values[k],flip=m.target)
                flips[k][m.target]=bases[k]^number(outputs(c,v),w.order,n)
            replacement=plan_value(m.plan,m.names,c.lib,values[k],mask)
            toggle=bits(replacement^values[k][c.resolve(m.target)],n).astype(bool)
            changes.append(np.where(toggle,flips[k][m.target],0))
            w.stats['numeric_joint_single_response_scores']+=1
        actions.append(Action(m,tuple(changes)))
    fanout={a.move.target:forbidden_fanout(c,[a.move.target]) for a in actions}
    for i,a in enumerate(actions):
        a.conflicts=frozenset(j for j,b in enumerate(actions) if i!=j and
          (b.move.target in fanout[a.move.target] or a.move.target in fanout[b.move.target]
           or set(b.move.names)&fanout[a.move.target] or set(a.move.names)&fanout[b.move.target]))
    proposed=beam_plans(actions,bases,w.reference[:2],w.metric.limit,w.stats)
    result=[]
    for ids,_,gain,loss in proposed:
        if w.remaining()<.5:break
        moves=[actions[i].move for i in sorted(ids)]
        try:trial=cleanup(c.materialize([m.replacement() for m in moves]),w.mapper)
        except (ValueError,KeyError):w.stats['numeric_joint_invalid_materializations']+=1;continue
        w.stats['numeric_joint_materializations']+=1
        result.append((trial,loss,dict(actions=[m.record() for m in moves],proxy_gain_units=gain)))
    return result


def accept(w,proposed,events):
    parent=w.current;best=None;near_count=0
    for trial,predicted,detail in proposed:
        if w.remaining()<.5:break
        losses=[]
        for k in (0,1):
            y=number(outputs(trial,trace(trial,w.inputs[k],w.count,w.stats)),w.order,w.count)
            losses.append(w.metric.exact(y,w.reference[k])[0])
        row=dict(detail,predicted_losses=predicted,actual_losses=losses,
            raw_area=trial.lib.area(trial.units),proxy_exact=tuple(losses)==tuple(predicted),accepted=False)
        events.append(row);w.stats['numeric_joint_actual_plan_checks']+=1
        if (w.args.targeted_near_repair and max(losses)>w.metric.limit and max(losses)<=1.12*w.metric.limit
            and trial.units<parent.units and near_count<2 and w.remaining()>4):
            near_count+=1;end=time.perf_counter()+min(12.,w.remaining()-2)
            branch_folder=w.folder/('near_joint_%04d'%w.stats['numeric_joint_actual_plan_checks'])
            branch=clone_worker(w,branch_folder,lambda:max(0.,min(w.remaining(),end-time.perf_counter())),
                w.started,trial,losses[0],False)
            repairs=significant_wire_proposals(branch);repair_events=[]
            ok=actual_trials(branch,repairs,'repair',branch.inputs[1],branch.reference[1],repair_events)
            save(branch_folder/'repair_events.json',repair_events)
            for key,value in branch.stats.items():
                if not key.startswith('mapping_'):w.stats[key]+=value
            w.stats['numeric_joint_near_mapping_calls']+=len(branch.backend.rows)
            w.stats['numeric_joint_near_mapping_cec']+=sum(q.get('cec_equivalent',False) for q in branch.backend.rows)
            w.stats['numeric_joint_targeted_near_branches']+=1
            row['near_repair']=dict(accepted=ok,folder=str(branch_folder),events=repair_events)
            if ok:
                trial=branch.current;losses=[]
                for k in (0,1):
                    y=number(outputs(trial,trace(trial,w.inputs[k],w.count,w.stats)),w.order,w.count)
                    losses.append(w.metric.exact(y,w.reference[k])[0])
                row['post_repair_losses']=losses;row['post_repair_area']=trial.lib.area(trial.units)
        if max(losses)>w.metric.limit or trial.units>=parent.units or structural_id(trial) in w.seen:continue
        if w.backend and w.remaining()>2:
            trial=w.backend.optimize(trial,w.remaining,variants=1)
            w.stats['mapping_calls']=len(w.backend.rows)
            w.stats['mapping_cec_proofs']=sum(q.get('cec_equivalent',False) for q in w.backend.rows)
        row['mapped_area']=trial.lib.area(trial.units)
        score=trial.units,max(losses),sum(losses)
        if best is None or score<best[0]:best=score,trial,losses,row
    if best is None:return False
    _,trial,losses,row=best;row['accepted']=True;w.current=trial;w.loss=losses[0]
    w.lowest_units=min(w.lowest_units,trial.units);w.seen.add(structural_id(trial));w.cached=None
    if hasattr(w,'area_model'):del w.area_model
    start=len(w.history);w.checkpoint('numeric_joint_plan_actual',[]);w.annotate(start)
    print('[NUMERIC JOINT PLAN]',json.dumps(row),flush=True);return True


def operate(w,tools,seconds,args,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();deadline=started+min(seconds,w.remaining());old=w.global_remaining
    w.global_remaining=lambda:max(0.,min(old(),deadline-time.perf_counter()))
    events=[];rounds=[]
    try:
        for iteration in range(args.rounds):
            identity=structural_id(w.current)
            for kind in tools:
                if w.remaining()<2:break
                print('[NUMERIC TRANSFER TOOL]',dict(iteration=iteration,kind=kind,area=w.current.lib.area(w.current.units),remaining=round(w.remaining(),2)),flush=True)
                if kind=='joint':
                    phase=time.perf_counter()+min(args.generation_seconds,max(.1,w.remaining()-4));prior=w.global_remaining
                    w.global_remaining=lambda:max(0.,min(prior(),phase-time.perf_counter()))
                    try:proposed=plans(w)
                    finally:w.global_remaining=prior
                    accept(w,proposed,events)
                else:
                    for policy in ('balanced','area','balanced'):
                        key='batch' if policy=='balanced' else policy
                        if w.remaining()<2:break
                        if (structural_id(w.current),key,1.) not in w.blocked:w.compress(policy,1.)
            rounds.append(dict(iteration=iteration,area=w.current.lib.area(w.current.units),seconds=time.perf_counter()-started))
            save(folder/'events.json',events);save(folder/'rounds.json',rounds)
            if structural_id(w.current)==identity:break
        return dict(seconds=time.perf_counter()-started,rounds=rounds,events=events)
    finally:w.global_remaining=old


def parser():
    p=base_parser();next(arg for arg in p._actions if arg.dest=='tools').choices=('joint','native')
    p.add_argument('--generation-seconds',type=float,default=30.)
    p.add_argument('--conditional-construction',action='store_true')
    p.add_argument('--conditional-roots',type=int,default=64)
    p.add_argument('--targeted-near-repair',action='store_true')
    p.set_defaults(tools=['joint','native'],rounds=6,seconds=90.,no_word=True,final_splits=[4200,4201])
    return p


def run(a):
    if a.metric!='nmed' or not 0<a.seconds<=300 or set(a.final_splits)&{0,1}:raise ValueError('Bounded NMED and independent final inputs required')
    folder=Path(a.output).resolve()
    if False:raise ValueError('D: only')
    folder.mkdir(parents=True,exist_ok=False);tick=time.perf_counter();lib=Library(LIBRARY_PATH)
    original,order,audit=preflight(a.design,lib);baseline=load(a.accurate_netlist,lib) if a.accurate_netlist else original
    start=load(a.start,lib) if a.start else baseline
    if start.inputs!=original.inputs or start.outputs!=original.outputs:raise ValueError('Different interface')
    inputs=[pack(matrix(a.design,original.inputs,k),original.inputs) for k in (0,1)]
    refs=[number(outputs(original,trace(original,x,10000)),order,10000) for x in inputs]
    for x,ref in zip(inputs,refs):assert np.array_equal(number(outputs(baseline,trace(baseline,x,10000)),order,10000),ref)
    snapshot=folder/'source';snapshot.mkdir();hashes={}
    for module in tuple(sys.modules.values()):
        p=Path(getattr(module,'__file__','') or '.')
        if p.suffix=='.py' and p.resolve().parent==Path(__file__).resolve().parent:
            shutil.copy2(p,snapshot/p.name);hashes[p.name]=sha(p)
    save(folder/'protocol.json',dict(args=vars(a),source_hashes=hashes,baseline_audit=audit,
        warm_start=bool(a.start),start_sha256=sha(a.start) if a.start else None,input_count=10000,
        selection_splits=[0,1],final_splits=a.final_splits,no_exhaustive_lacs=True,api_calls=0))
    remaining=lambda:max(0.,a.seconds-(time.perf_counter()-tick));(folder/'worker').mkdir()
    worker_type=ConditionalNumericWorker if a.conditional_construction else Worker
    w=worker_type(start,order,inputs,refs,a,folder/'worker',remaining=remaining,reference_source=baseline)
    detail=operate(w,a.tools,remaining(),a,folder/'tools');eligible=[]
    for row in w.history:
        if row['training_loss']>w.metric.limit:continue
        c=load(w.folder/row['path'],lib);y=number(outputs(c,trace(c,inputs[1],10000,w.stats)),order,10000)
        loss=w.metric.exact(y,refs[1])[0]
        if loss<=w.metric.limit:eligible.append((c.units,max(row['training_loss'],loss),row['path'],c))
    if not eligible:raise RuntimeError('No feasible checkpoint')
    chosen=min(eligible,key=lambda q:q[:3]);winner=chosen[-1];winner.write(folder/'selected_before_checks.v');digest=sha(folder/'selected_before_checks.v')
    save(folder/'frozen_selection.json',dict(path=chosen[2],sha256=digest,held_out_read=False))
    checks=[]
    for split in a.final_splits:
        x=matrix(a.design,original.inputs,split);inp=pack(x,original.inputs)
        y=number(outputs(winner,trace(winner,inp,10000)),order,10000);ref=number(outputs(original,trace(original,inp,10000)),order,10000)
        assert np.array_equal(y,number_numpy(validate_numpy(winner,x),winner.outputs,order))
        assert np.array_equal(ref,number_numpy(validate_numpy(original,x),original.outputs,order))
        loss,value=w.metric.exact(y,ref);checks.append(dict(split=split,count=10000,nmed=value,passed=loss<=w.metric.limit))
    passed=all(q['passed'] for q in checks);winner.write(folder/('best.v' if passed else 'failed_final.v'))
    for name,h in hashes.items():assert sha(Path(__file__).resolve().parent/name)==h,name
    result=dict(design=a.design,bound=a.bound,baseline_area=lib.area(baseline.units),starting_area=lib.area(start.units),
        area=lib.area(winner.units),reduction=1-winner.units/baseline.units,strict_pass=passed,checks=checks,
        whole_seconds=time.perf_counter()-tick,stats=dict(w.stats),detail=detail,selected_sha256=digest,
        main_mapping_calls=len(w.backend.rows),
        mapping_calls=len(w.backend.rows)+w.stats['numeric_joint_near_mapping_calls'],
        mapping_cec=sum(r.get('cec_equivalent',False) for r in w.backend.rows)+w.stats['numeric_joint_near_mapping_cec'],api_calls=0)
    save(folder/'report.json',result);print('[NUMERIC TRANSFER RESULT]',json.dumps({k:v for k,v in result.items() if k!='detail'}),flush=True)
    return 0 if passed else 2


if __name__=='__main__':raise SystemExit(run(parser().parse_args()))
