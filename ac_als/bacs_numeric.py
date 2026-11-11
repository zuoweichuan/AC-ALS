"""Independent BACS standard-cell ALS under the paper's NMED/MSE constraints.

Integer numerical response costs replace vector-ER masks. A root flip is shared
by all screened functions. Optional protected compensation learns only a few
response-selected truth tables. No AIG optimizer, LLM, exhaustive LAC search,
or external-project search implementation is used.
"""
import runtime
import argparse
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
from math import isqrt
from pathlib import Path
import shutil
import time

import numpy as np
from logic import Library,evaluate
from cutmap import CutMapper,variable
from netlist import Circuit,ZERO,ONE
from frontier import boundary_cuts,forbidden_fanout
from screen import FanoutCones
from simulate import pack,trace,outputs,plan_value,validate_numpy
from paper_resub_netlist import consistent_table,screened_sets,used_divisors
from paper_netlist_subfunctions import cell_normalize,cut_normalize,proposals as subfunctions
from hybrid_identity import structural_id
from bacs_data import ROOT,DESIGNS,source_path,load,matrix,preflight,LIBRARY_PATH

CODE_ROOT=Path(__file__).resolve().parent


def save(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def bits(word,count):
    return np.unpackbits(np.frombuffer(int(word).to_bytes((count+7)//8,'little'),dtype=np.uint8),bitorder='little')[:count]


class VectorEROrder(tuple):
    """Tag bit labels for lossless any-output ER, including >64 output bits."""
    vector_er=True


class SignedOrder(tuple):
    """LSB-first two's-complement word; ordinary BACS tuples remain unsigned."""
    signed_word=True


def signed_decode(value,order):
    if getattr(order,'signed_word',False):
        if not 1<=len(order)<=62:raise ValueError('Signed numerical words require1..62bits')
        value=value-((value>>(len(order)-1))&1)*(1<<len(order))
    return value


def number_numpy(array,outputs_order,order):
    columns={n:i for i,n in enumerate(outputs_order)}
    if getattr(order,'vector_er',False):
        packed=np.ascontiguousarray(np.packbits(array[:,[columns[n] for n in order]],axis=1,bitorder='little'))
        return packed.view(np.dtype('V%d'%packed.shape[1])).reshape(-1)
    value=np.zeros(len(array),dtype=np.int64)
    for i,n in enumerate(order):value |= array[:,columns[n]].astype(np.int64)<<i
    return signed_decode(value,order)


def number(words,order,count):
    if getattr(order,'vector_er',False):
        array=np.column_stack([bits(words[n],count) for n in order])
        return number_numpy(array,order,order)
    value=np.zeros(count,dtype=np.int64)
    for i,n in enumerate(order):value |= bits(words[n],count).astype(np.int64)<<i
    return signed_decode(value,order)


class Metric:
    def __init__(self,kind,bound,order,count=10000):
        self.kind,self.bound,self.order,self.count=kind,float(bound),tuple(order),count
        self.denominator=(1<<len(order))-1 if kind=='nmed' else 1
        self.limit=int(Decimal(str(bound))*self.denominator*count)
        self.cap=self.limit+1
    def loss(self,y,reference):
        if self.kind=='er':return (y!=reference).astype(np.int64)
        d=np.abs(y-reference)
        if self.kind=='nmed':return d
        # Any sample above the entire allowed sum makes a candidate infeasible.
        # Saturate BEFORE squaring to avoid 64-bit overflow, while preserving
        # exact costs for every feasible candidate. Final reporting is unclipped.
        small=d<=isqrt(self.cap);result=np.full(len(d),self.cap,dtype=np.int64)
        result[small]=d[small]*d[small]
        return result
    def exact(self,y,reference):
        if self.kind=='er':
            total=int(np.count_nonzero(y!=reference))
            return total,total/self.count
        d=np.abs(y-reference)
        total=int(d.sum()) if self.kind=='nmed' else sum(int(v)*int(v) for v in d)
        return total,total/self.count/self.denominator


def cleanup(circuit,mapper):
    """Actual physical area; remap a full adder's sole live output as needed."""
    current=cell_normalize(circuit,mapper)
    while True:
        live_outputs={current.resolve(n) for n in current.outputs};chosen=[]
        for g in current.gates:
            cell=current.lib.cells[g.cell]
            if len(cell.functions)<2:continue
            pins=[p for p in cell.functions if p in g.pins and
                  (g.pins[p] in live_outputs or current.fanouts.get(g.pins[p]))]
            if len(pins)!=1:continue
            names=tuple(sorted({current.resolve(g.pins[p]) for p in cell.inputs}-{ZERO,ONE}))
            if len(names)>4:continue
            mask=(1<<(1<<len(names)))-1
            values={ZERO:0,ONE:mask,**{n:variable(len(names),i) for i,n in enumerate(names)}}
            truth=evaluate(cell.functions[pins[0]],{'0':0,'1':mask,
                **{p:values[current.resolve(g.pins[p])] for p in cell.inputs}},mask)
            plan=mapper.map(len(names),truth)
            if mapper.cost(plan)<cell.units:chosen.append((g.pins[pins[0]],names,plan))
        if not chosen:return current
        trial=cell_normalize(current.materialize(chosen),mapper)
        if trial.units>=current.units:raise AssertionError('Sole-output remapping did not save area')
        current=trial


@dataclass
class Move:
    target:str
    names:tuple
    truth:int
    plan:object
    gain:int
    error:int
    weight:int
    signature:str
    family:str='paper_resub'
    penalty:int=1
    def replacement(self):return self.target,self.names,self.plan
    def record(self):
        return dict(target=self.target,inputs=list(self.names),truth=self.truth,gain_units=self.gain,
                    total_loss=self.error,positive_increment=self.weight,family=self.family,penalty=self.penalty)


def knapsack(candidates,capacity):
    groups=defaultdict(list)
    for c in candidates:
        if c.gain>0 and c.weight<=capacity:groups[c.target].append(c)
    states={0:(0,None)}
    for target in sorted(groups):
        previous=states;updated=dict(states)
        for gain,(cost,chain) in previous.items():
            for c in groups[target]:
                g,w=gain+c.gain,cost+c.weight
                if w<=capacity and w<updated.get(g,(capacity+1,))[0]:updated[g]=(w,(chain,c))
        states={};lowest=capacity+1
        for gain in sorted(updated,reverse=True):
            if updated[gain][0]<lowest:
                states[gain]=updated[gain];lowest=updated[gain][0]
    if not states:return []
    chain=states[max(states)][1];result=[]
    while chain is not None:chain,c=chain;result.append(c)
    return list(reversed(result))


def learned_sets(circuit,target,values,loss0,loss1,count,stats):
    forbidden=forbidden_fanout(circuit,[target])
    available=[n for n in (*circuit.inputs,*circuit.producer) if n not in forbidden]
    cache={};decoded={}
    def code(names):
        x=np.zeros(count,dtype=np.uint8)
        for i,n in enumerate(names):
            if n not in decoded:decoded[n]=bits(values[n],count)
            x |= decoded[n]<<i
        return x
    def rank(names):
        if names not in cache:
            x=code(names);n=1<<len(names)
            a=np.bincount(x,weights=loss0,minlength=n);b=np.bincount(x,weights=loss1,minlength=n)
            cache[names]=int(np.minimum(a,b).sum())
        return cache[names],len(names),names
    pool=sorted(available,key=lambda n:rank((n,)))[:10]
    structural=boundary_cuts(circuit,target,width=4)
    for n in (structural[-1] if structural else ()):
        if n not in pool and len(pool)<14:pool.append(n)
    selected=set(structural);beam=[()]
    for _ in range(4):
        options={tuple(sorted((*s,n))) for s in beam for n in pool if n not in s}
        beam=sorted(options,key=rank)[:2];selected.update(beam)
    result=sorted(selected,key=rank)[:4]
    for names in structural[-2:]:
        if names not in result:result.append(names)
    stats['compensation_input_set_proxies']+=len(cache)
    return [(names,code(names)) for names in result[:6]]


class NumericSearch:
    def __init__(self,source,order,inputs,reference,args,folder):
        self.source,self.order,self.inputs,self.reference=source,order,inputs,reference
        self.args,self.folder=args,folder;self.count=10000
        self.metric=Metric(args.metric,args.bound,order);self.mapper=CutMapper(source.lib,4)
        self.stats=defaultdict(int);self.times=defaultdict(float);self.rng=np.random.default_rng(args.seed)
        self.started=time.perf_counter();self.history=[];self.events=[];self.iteration=0;self.repairs=0
        self.current=cleanup(source,self.mapper)
        self.current=cleanup(cut_normalize(self.current,self.mapper,self.stats),self.mapper)
        self.seen={structural_id(self.current)};self.lowest_units=self.current.units
        self.checkpoint('initial',[])
    def expired(self):return time.perf_counter()-self.started>=self.args.seconds
    def checkpoint(self,phase,selected):
        path=self.folder/('node_%04d.v'%len(self.history));self.current.write(path)
        row=dict(path=path.name,units=self.current.units,area=self.source.lib.area(self.current.units),
            phase=phase,iteration=self.iteration,seconds=time.perf_counter()-self.started,
            actions=[m.record() for m in selected])
        self.history.append(row);save(self.folder/'history.json',self.history)
        print('[BACS ACCEPT] '+json.dumps({k:v for k,v in row.items() if k!='actions'}),flush=True)
    def scored(self,target,names,truth,plan,values,t,base_y,flip_y,base_loss,flip_loss,family,penalty=1):
        mask=(1<<self.count)-1;word=plan_value(plan,names,self.source.lib,values,mask)
        delta=bits(word^t,self.count).astype(bool)
        total=int(base_loss[~delta].sum()+flip_loss[delta].sum())
        self.stats['function_error_evaluations']+=1
        if total>self.metric.limit:return None
        positive=int(np.maximum(flip_loss[delta]-base_loss[delta],0).sum())
        predicted=np.where(delta,flip_y,base_y)
        signature=hashlib.blake2b(predicted.tobytes(),digest_size=16).hexdigest()
        return Move(target,names,truth,plan,0,total,positive,signature,family,penalty)
    def materialized(self,circuit,items,base_total,allow_debt=False):
        # Retain min-error and low implementation-cost alternatives without
        # constructing every generated physical candidate.
        items=sorted(items,key=lambda m:(m.error,self.mapper.cost(m.plan),m.names,m.truth))
        chosen=items[:2]
        cheap=sorted(items,key=lambda m:(self.mapper.cost(m.plan),m.error,m.names,m.truth))
        chosen += [m for m in cheap if m not in chosen][:2]
        result=[]
        for m in chosen:
            trial=cleanup(circuit.materialize([m.replacement()]),self.mapper)
            self.stats['area_materializations']+=1;m.gain=circuit.units-trial.units
            if m.gain>0 or (allow_debt and m.gain>=-16 and m.error<base_total):result.append(m)
        return result
    def candidates(self):
        started=time.perf_counter();circuit=self.current;count=self.count;mask=(1<<count)-1
        values=trace(circuit,self.inputs[0],count,self.stats)
        base_y=number(outputs(circuit,values),self.order,count)
        base_loss=self.metric.loss(base_y,self.reference[0]);base_total=int(base_loss.sum())
        small=pack(np.random.default_rng(int(self.rng.integers(1<<31))).integers(0,2,
                    (64,len(circuit.inputs)),dtype=np.uint8),circuit.inputs)
        sample=trace(circuit,small,64);self.stats['generation_64_simulations']+=1
        cones=FanoutCones(circuit);result=[];repair_contexts=[]
        for target in sorted(circuit.producer,key=lambda n:(-cones.potential(n),n)):
            if self.expired():break
            self.stats['screened_roots']+=1
            majority=int(sample[target].bit_count()>32)
            raw=[((),15 if majority else 0,circuit.lib.mapper.plans[15 if majority else 0],'constant')]
            for names in screened_sets(circuit,target,sample,64,self.args.pool,self.args.pairs,self.rng):
                self.stats['screened_divisor_sets']+=1
                table=consistent_table([sample[n] for n in names],sample[target],(1<<64)-1)
                if table is not None:
                    truth,plan=circuit.lib.mapper.complete(*table);raw.append((names,truth,plan,'paper_resub'))
            if self.args.subfunctions:
                from paper_resub_netlist import divisor_pool
                raw.extend((n,t,p,'cell_subfunction') for n,t,p,_ in subfunctions(circuit,target,sample,64,
                    self.mapper,divisor_pool(circuit,target),self.stats,consistent_table))
            raw2=[];seen=set();multi=len(circuit.lib.cells[circuit.producer[target].cell].functions)>1
            for names,truth,plan,family in raw:
                potential=cones.saved_with_inputs(target,used_divisors(names,plan))-self.mapper.cost(plan)
                if potential<=0 and not multi:continue
                key=(names,truth)
                if key not in seen:seen.add(key);raw2.append((names,truth,plan,family))
            if not raw2 and not self.args.compensate:continue
            tick=time.perf_counter()
            flipped=trace(circuit,self.inputs[0],count,self.stats,base=values,flip=target)
            flip_y=number(outputs(circuit,flipped),self.order,count)
            flip_loss=self.metric.loss(flip_y,self.reference[0]);t=values[target]
            self.times['root_response_subset']+=time.perf_counter()-tick
            if self.args.compensate and base_total:
                possible=int(np.maximum(base_loss-flip_loss,0).sum())
                if possible:
                    repair_contexts.append((possible,target,t,flip_y,flip_loss))
                    repair_contexts=sorted(repair_contexts,key=lambda r:(-r[0],r[1]))[:8]
            measured=[]
            for names,truth,plan,family in raw2:
                m=self.scored(target,names,truth,plan,values,t,base_y,flip_y,base_loss,flip_loss,family)
                if m is not None:measured.append(m)
            tick=time.perf_counter();result+=self.materialized(circuit,measured,base_total)
            self.times['area_materialization_subset']+=time.perf_counter()-tick
        if self.args.compensate:
            for _,target,t,flip_y,flip_loss in repair_contexts:
                if self.expired():break
                tbits=bits(t,count).astype(bool)
                loss0=np.where(tbits,flip_loss,base_loss);loss1=np.where(tbits,base_loss,flip_loss)
                measured=[]
                for names,code in learned_sets(circuit,target,values,loss0,loss1,count,self.stats):
                    n=1<<len(names)
                    for penalty in (1,2,4):
                        a=np.bincount(code,weights=loss0+(penalty-1)*np.maximum(loss0-base_loss,0),minlength=n)
                        b=np.bincount(code,weights=loss1+(penalty-1)*np.maximum(loss1-base_loss,0),minlength=n)
                        truth=sum(int(y<x)<<i for i,(x,y) in enumerate(zip(a,b)))
                        plan=self.mapper.map(len(names),truth)
                        m=self.scored(target,names,truth,plan,values,t,base_y,flip_y,base_loss,flip_loss,'numeric_protected_truth',penalty)
                        self.stats['compensation_truths_constructed']+=1
                        if m is not None:measured.append(m)
                tick=time.perf_counter();result+=self.materialized(circuit,measured,base_total,True)
                self.times['area_materialization_subset']+=time.perf_counter()-tick
        unique={}
        for m in result:
            key=(m.target,m.names,m.truth)
            if key not in unique:unique[key]=m
        self.times['candidate_generation_and_scoring']+=time.perf_counter()-started
        return list(unique.values()),base_total
    def apply(self,selected,phase):
        tick=time.perf_counter()
        try:
            trial=cleanup(self.current.materialize([m.replacement() for m in selected]),self.mapper)
            trial=cleanup(cut_normalize(trial,self.mapper,self.stats),self.mapper)
        except ValueError as exc:return False,dict(reason='invalid_graph',error=str(exc))
        identity=structural_id(trial)
        if identity in self.seen:
            self.stats['duplicate_propagations_avoided']+=1
            return False,dict(reason='known_physical_circuit')
        words=trace(trial,self.inputs[0],self.count,self.stats)
        y=number(outputs(trial,words),self.order,self.count);loss=self.metric.loss(y,self.reference[0]);total=int(loss.sum())
        self.stats['joint_verifications' if len(selected)>1 else 'single_verifications']+=1
        if len(selected)==1:
            if hashlib.blake2b(y.tobytes(),digest_size=16).hexdigest()!=selected[0].signature:
                raise AssertionError('Numerical CPM disagrees with actual full output vector')
        self.times['selected_actual_verification']+=time.perf_counter()-tick
        detail=dict(error_sum=total,metric=total/self.count/self.metric.denominator,area=self.source.lib.area(trial.units),
                    actions=[m.record() for m in selected])
        if total>self.metric.limit:return False,dict(detail,reason='actual_metric_exceeds_bound')
        if phase!='compensation' and trial.units>=self.current.units:return False,dict(detail,reason='no_area_gain')
        if phase=='compensation' and trial.units>min(self.source.units,self.lowest_units+16):
            return False,dict(detail,reason='cumulative_area_debt')
        self.current=trial;self.lowest_units=min(self.lowest_units,trial.units);self.seen.add(identity)
        self.checkpoint(phase,selected);return True,detail
    def run(self):
        phase='batch';stop='no_acceptable_screened_candidate'
        while not self.expired():
            self.iteration+=1;candidates,base_total=self.candidates()
            positive=[m for m in candidates if m.gain>0]
            event=dict(iteration=self.iteration,phase=phase,candidates=len(candidates),current_loss=base_total)
            if phase=='batch':
                tick=time.perf_counter();choices=[knapsack(positive,self.metric.limit-base_total)]
                self.times['knapsack']+=time.perf_counter()-tick
            else:choices=[[m] for m in sorted(positive,key=lambda m:(m.error,-m.gain,m.target,m.names))[:3]]
            accepted=False
            for selected in choices:
                if not selected:continue
                accepted,detail=self.apply(selected,phase);event.update(detail)
                if accepted:break
            if not accepted and phase!='batch' and self.args.compensate and self.repairs<3:
                repairs=sorted((m for m in candidates if m.error<base_total),
                    key=lambda m:(m.error+m.weight,-m.gain,m.target,m.names))[:3]
                for m in repairs:
                    accepted,detail=self.apply([m],'compensation');event.update(detail)
                    if accepted:self.repairs+=1;phase='batch';break
            event['accepted']=accepted;self.events.append(event);save(self.folder/'iterations.json',self.events)
            print('[BACS ROUND] '+json.dumps({k:v for k,v in event.items() if k!='actions'}),flush=True)
            if not accepted:
                if phase=='batch':phase='minimum_error_single'
                else:break
        if self.expired():stop='time_limit'
        return dict(stop_reason=stop,search_seconds=time.perf_counter()-self.started,stats=dict(self.stats),
                    component_seconds=dict(self.times),iterations=len(self.events),compensation_steps=self.repairs)


def run(args):
    tick=time.perf_counter();folder=Path(args.output).resolve();folder.mkdir(parents=True,exist_ok=False)
    (folder/'source').mkdir();hashes={}
    names=['bacs_numeric.py','bacs_data.py','logic.py','netlist.py','cutmap.py','frontier.py','screen.py',
        'simulate.py','weighted_cost.py','runtime.py','paper_resub_netlist.py','paper_netlist_subfunctions.py','hybrid_identity.py']
    for name in names:
        hashes[name]=hashlib.sha256((CODE_ROOT/name).read_bytes()).hexdigest();shutil.copy2(CODE_ROOT/name,folder/'source'/name)
    lib=Library(LIBRARY_PATH);source,order,audit=preflight(args.design,lib)
    source.write(folder/'accurate.v');shutil.copy2(source_path(args.design),folder/'accurate_original_dc.v')
    matrices=[matrix(args.design,source.inputs,i) for i in range(3)]
    inputs=[pack(x,source.inputs) for x in matrices]
    refs=[number(outputs(source,trace(source,p,10000)),order,10000) for p in inputs]
    metric=Metric(args.metric,args.bound,order)
    protocol=dict(vars(args),audit=audit,source_sha256=hashlib.sha256(source_path(args.design).read_bytes()).hexdigest(),
        code_sha256=hashes,count_per_group=10000,output_lsb_first=order,unsigned_flattened=True,
        denominator=metric.denominator,integer_loss_bound=metric.limit,
        selection='min area across development-eligible checkpoints, then final; final not used for selection',
        paper_protocol_deviations=['standard-cell search, no AIG/compress2rs','10000 evaluation vectors, fixed input streams',
                                  'bounded screened divisor sets','current Nangate mapping; paper MSE used MCNC'],api_calls=0)
    save(folder/'protocol.json',protocol)
    search=NumericSearch(source,order,inputs,refs,args,folder);result=search.run()
    selected_rows=[];dev_limit=int(Decimal(metric.limit)*Decimal(str(args.development_fraction)))
    for row in search.history:
        circuit=load(folder/row['path'],lib)
        y=number(outputs(circuit,trace(circuit,inputs[1],10000)),order,10000)
        total,value=metric.exact(y,refs[1]);selected_rows.append(dict(row,development_loss=total,development_value=value))
    selected=min((r for r in selected_rows if r['development_loss']<=dev_limit),key=lambda r:(r['units'],r['path']))
    winner=load(folder/selected['path'],lib)
    actual=number(outputs(winner,trace(winner,inputs[2],10000)),order,10000)
    total,value=metric.exact(actual,refs[2])
    # Separate NumPy LUT path, explicit PO order, independent scalar arithmetic.
    np_ref=validate_numpy(source,matrices[2]);np_out=validate_numpy(winner,matrices[2])
    def independent(circuit,array):
        columns={n:i for i,n in enumerate(circuit.outputs)}
        return np.array([sum(int(array[j,columns[n]])<<i for i,n in enumerate(order))
                         for j in range(len(array))],dtype=np.int64)
    np_total,np_value=metric.exact(independent(winner,np_out),independent(source,np_ref))
    if np_total!=total:raise AssertionError('Independent NumPy/scalar final metric mismatch')
    passed=total<=metric.limit;winner.write(folder/('best.v' if passed else 'failed_final.v'))
    train_y=number(outputs(winner,trace(winner,inputs[0],10000)),order,10000)
    train_total,train_value=metric.exact(train_y,refs[0])
    result.update(design=args.design,metric=args.metric,bound=args.bound,compensated=args.compensate,
        original_area=lib.area(source.units),area=lib.area(winner.units),reduction=1-winner.units/source.units,
        training_value=train_value,development_value=selected['development_value'],final_value=value,
        final_vector_er=float(np.count_nonzero(actual!=refs[2]))/10000,strict_pass=passed,
        selected_checkpoint=selected['path'],all_checkpoints=selected_rows,whole_seconds=time.perf_counter()-tick,
        postselection_full_propagations=len(selected_rows)+2,reference_full_propagations=3,independent_numpy_checks=2,
        preflight=audit,api_calls=0)
    save(folder/'report.json',result)
    print('[BACS RESULT] '+json.dumps({k:v for k,v in result.items() if k not in ('all_checkpoints','preflight')}),flush=True)
    return 0 if passed else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--design',choices=DESIGNS,required=True);p.add_argument('--metric',choices=['nmed','mse'],required=True)
    p.add_argument('--bound',type=float,required=True);p.add_argument('--output',required=True)
    p.add_argument('--seconds',type=float,default=45);p.add_argument('--seed',type=int,default=20261027)
    p.add_argument('--pool',type=int,default=24);p.add_argument('--pairs',type=int,default=48)
    p.add_argument('--subfunctions',action='store_true');p.add_argument('--compensate',action='store_true')
    p.add_argument('--development-fraction',type=float,default=1.0)
    args=p.parse_args()
    if args.bound<=0 or min(args.seconds,args.pool,args.pairs)<=0 or not 0<args.development_fraction<=1:
        p.error('Invalid bound/screening/time settings')
    raise SystemExit(run(args))
