"""Isolated bounded joint-LAC plans using vector-error overlap as a proxy.

Single-action output deltas are exact; XOR-combined deltas are only rankings.
Every retained joint plan is materialized and simulated on both selection sets.
No held-out input is consulted until the experiment freezes its winner.
"""
from dataclasses import dataclass
import json
import er_residual_research as experiment
from er_repair_tools import JointRepair, move_key, error_totals, development_limit
from bacs_numeric import cleanup
from hybrid_identity import structural_id
from simulate import plan_value
from frontier import forbidden_fanout


def merged_errors(base, changes):
    result=list(base)
    for change in changes:
        result=[e^d for e,d in zip(result,change)]
    return tuple(result)


@dataclass
class Entry:
    move: object
    train: tuple
    dev: tuple
    conflicts: frozenset=frozenset()


def plan_beam(entries,train,dev,limit,dev_limit,stats,maximum=6,width=4):
    """Four diverse retained plans, at most four partners per plan per depth."""
    empty=(frozenset(),train,dev,0)
    beam=[empty];retained={}
    def ranking(row):
        _,t,d,gain=row;te=error_totals(t)[0];de=error_totals(d)[0]
        excess=max(0,te-limit)+max(0,de-dev_limit)
        return excess,-gain,max(te,de),te+de
    for depth in range(maximum):
        proposals={}
        for ids,t,d,gain in beam:
            targets={entries[i].move.target for i in ids};local=[]
            for i,e in enumerate(entries):
                if i in ids or e.move.target in targets or ids&e.conflicts:continue
                nt=merged_errors(t,[e.train]);nd=merged_errors(d,[e.dev])
                row=ids|{i},nt,nd,gain+e.move.gain
                stats['er_joint_mask_plan_proxy_scores']+=1
                local.append(row)
            # Different selection pressures keep a repair-first path alive.
            queues=[sorted(local,key=ranking),
                    sorted(local,key=lambda r:(-r[3],ranking(r))),
                    sorted(local,key=lambda r:(max(error_totals(r[1])[0],error_totals(r[2])[0]),-r[3]))]
            picked={}
            for rank in range(len(local)):
                for queue in queues:
                    if rank<len(queue):picked.setdefault(queue[rank][0],queue[rank])
                    if len(picked)==width:break
                if len(picked)==width:break
            proposals.update(picked)
        if not proposals:break
        rows=list(proposals.values());queues=[sorted(rows,key=ranking),
            sorted(rows,key=lambda r:(-r[3],ranking(r))),
            sorted(rows,key=lambda r:(max(error_totals(r[1])[0],error_totals(r[2])[0]),-r[3]))]
        chosen={}
        for rank in range(len(rows)):
            for queue in queues:
                if rank<len(queue):chosen.setdefault(queue[rank][0],queue[rank])
                if len(chosen)==width:break
            if len(chosen)==width:break
        beam=list(chosen.values())
        for row in rows:
            if row[3]>0 and ranking(row)[0]==0:retained[row[0]]=row
    return sorted(retained.values(),key=ranking)[:12]


def joint_proposals(w):
    moves=[m for m in w.scan()['moves'] if not isinstance(m,JointRepair)
           and (m.gain>0 or m.error<w.loss) and m.gain>=-w.args.repair_debt_units]
    queues=[sorted(moves,key=lambda m:(-m.gain,m.error)),
            sorted(moves,key=lambda m:(m.error,-m.gain)),
            sorted(moves,key=lambda m:(-m.gain/max(1,m.error-w.loss),m.error))]
    selected={}
    for rank in range(len(moves)):
        for queue in queues:
            if rank<len(queue):selected.setdefault(move_key([queue[rank]]),queue[rank])
            if len(selected)>=64:break
        if len(selected)>=64:break
    entries=[];contexts=[w.response_context(s) for s in (0,1)];mask=(1<<w.count)-1
    for m in selected.values():
        if w.remaining()<1:break
        changes=[]
        for split,ctx in enumerate(contexts):
            w.response_score(m,split)
            delta=plan_value(m.plan,m.names,w.current.lib,ctx['values'],mask)^ctx['values'][w.current.resolve(m.target)]
            changes.append(tuple(d&delta for d in ctx['flips'][m.target]))
        entries.append(Entry(m,*changes))
    # A chain of equivalent roots can otherwise appear to cancel the same
    # error twice. Never combine ancestor roots or divisors changed by a peer.
    fanout={e.move.target:forbidden_fanout(w.current,[e.move.target]) for e in entries}
    for i,e in enumerate(entries):
        conflict=set()
        for j,f in enumerate(entries):
            if i==j:continue
            a=fanout[e.move.target];b=fanout[f.move.target]
            if f.move.target in a or e.move.target in b or set(f.move.names)&a or set(e.move.names)&b:
                conflict.add(j)
        e.conflicts=frozenset(conflict)
    plans=plan_beam(entries,contexts[0]['errors'],contexts[1]['errors'],w.metric.limit,
                    development_limit(w.metric,w.args),w.stats)
    result=[]
    for ids,train,dev,gain in plans:
        if w.remaining()<.5:break
        group=[entries[i].move for i in sorted(ids)]
        try:trial=cleanup(w.current.materialize([m.replacement() for m in group]),w.mapper)
        except (ValueError,KeyError):
            w.stats['er_joint_mask_invalid_materializations']+=1;continue
        w.stats['er_joint_mask_materializations']+=1
        result.append((trial,error_totals(train),dict(kind='joint_error_mask_plan',
            components=[m.record() for m in group],proxy_train=error_totals(train),
            proxy_development=error_totals(dev),proxy_gain_units=gain)))
    return result


def adopt(w,proposals,events):
    parent=w.current;best=None
    for trial,predicted,detail in proposals:
        if w.remaining()<.5:break
        train=experiment.metrics(w,trial,0);dev=experiment.metrics(w,trial,1)
        row=dict(detail,training_errors=train,development_errors=dev,
                 proxy_training_exact=train==predicted,raw_area=trial.lib.area(trial.units),accepted=False)
        events.append(row)
        if train[0]>w.metric.limit or dev[0]>development_limit(w.metric,w.args):continue
        if trial.units>=parent.units or structural_id(trial) in w.seen:continue
        if w.backend and w.remaining()>2:trial=w.backend.optimize(trial,w.remaining,variants=1)
        row['mapped_area']=trial.lib.area(trial.units)
        # Mapping is CEC-checked by the backend, then verify the actual graph too.
        mt=experiment.metrics(w,trial,0);md=experiment.metrics(w,trial,1)
        if mt!=train or md!=dev:raise AssertionError('Mapping changed ER')
        score=trial.units,max(train[0],dev[0]),train[1]+dev[1]
        if best is None or score<best[0]:best=score,trial,train,row
    if best is None:return False
    _,trial,train,row=best;row['accepted']=True;w.current=trial;w.loss=train[0]
    w.lowest_units=min(w.lowest_units,trial.units);w.seen.add(structural_id(trial))
    experiment.reset_caches(w);start=len(w.history);w.checkpoint('er_joint_mask_actual',[]);w.annotate(start)
    print('[ER JOINT PLAN]',json.dumps(row),flush=True);return True


if __name__=='__main__':
    p=experiment.parser();p.set_defaults(tools=['response','native'],repair_generation_seconds=45.,rounds=5)
    a=p.parse_args();experiment.response_repairs=joint_proposals;experiment.adopt_repair=adopt
    raise SystemExit(experiment.run(a))
