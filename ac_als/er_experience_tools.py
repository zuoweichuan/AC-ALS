"""Shared current-graph recollection/joint-plan tools for ER controllers.

No circuit-specific node, historical approximate netlist, target area or final
input is loaded. Operating margins are local to a detached worker, never the
global formal constraint or a parent's mutable Metric/arguments.
"""
import copy
import time
from decimal import Decimal
from pathlib import Path
from bacs_numeric import save
from er_joint_mask_research import joint_proposals,adopt
from er_residual_research import reset_caches,metrics
from er_repair_tools import development_limit
from hybrid_identity import structural_id


PROFILES=('wide','joint')
LESSONS=[
 'Recollect root responses and divisor relationships on the CURRENT mapped graph, not its accurate ancestor.',
 'A narrow shortlist plateau is not convergence: a bounded wider conditional-root probe can expose a different useful LAC path.',
 'Joint compression may include a small area-cost repair and a separately near-infeasible compression. Require actual joint area and both selection sets to pass.',
 'XOR response masks rank plans only. Ancestor/divisor conflicts are excluded, but reconvergence still requires actual whole-netlist simulation.',
 'Reserve error on BOTH selection groups when error is near the boundary. This is a predeclared operating margin, not a changed formal ER or guarantee.',
 'New joint plans may improve quality while requiring more whole-graph validation. Judge measured quality, full propagations, root responses, function scores and time separately.',
 'After refreshed shortlist probes show no acceptable change, avoid replaying the same graph/profile. No final input group may guide checkpoint choice.'
]


def operating_options(args,metric,profile):
    if profile not in PROFILES:raise ValueError('Unknown experience profile')
    options=copy.copy(args);working=copy.copy(metric)
    formal=int(Decimal(str(args.bound))*metric.count)
    options.conditional_roots=args.experience_roots
    options.conditional_shortlist=args.experience_shortlist
    options.no_word=True
    options.training_fraction=args.experience_joint_fraction if profile=='joint' else 1.
    options.development_fraction=1. if profile=='joint' else args.experience_wide_fraction
    working.limit=int(Decimal(formal)*Decimal(str(options.training_fraction)));working.cap=working.limit+1
    return options,working,formal


def recommended_profile(train,development,formal):
    return 'joint' if max(train,development)>=.8*formal else 'wide'


def continue_current(w,profile,seconds,args,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();deadline=started+min(seconds,w.remaining())
    previous_args,previous_metric,previous_remaining=w.args,w.metric,w.global_remaining
    options,working,formal=operating_options(args,w.metric,profile)
    w.args,w.metric=options,working
    w.global_remaining=lambda:max(0.,min(previous_remaining(),deadline-time.perf_counter()))
    # These caches encode topology or the operating cap; neither is reusable.
    reset_caches(w);w.blocked=set();w.failed=set();events=[];rounds=[];before=w.current.units;parent_graph=structural_id(w.current)
    save(folder/'protocol.json',dict(profile=profile,formal_wrong_vectors=formal,
       training_wrong_vectors=working.limit,development_wrong_vectors=development_limit(working,options),
       conditional_roots=options.conditional_roots,conditional_shortlist=options.conditional_shortlist,
       seconds_cap=seconds,input_splits=[0,1],final_inputs_available=False,knowledge=LESSONS,
       max_joint_shortlist=64,max_plan_width=4,max_plan_depth=6,max_actual_plans=12))
    try:
        for iteration in range(args.experience_rounds):
            if w.remaining()<2:break
            identity=structural_id(w.current)
            print('[ER RECOLLECT]',dict(profile=profile,iteration=iteration,area=w.current.lib.area(w.current.units),remaining=round(w.remaining(),2)),flush=True)
            if profile=='joint':
                gen_end=time.perf_counter()+min(args.experience_generation_seconds,max(.1,w.remaining()-4))
                old=w.global_remaining
                w.global_remaining=lambda:max(0.,min(old(),gen_end-time.perf_counter()))
                try:plans=joint_proposals(w)
                finally:w.global_remaining=old
                adopt(w,plans,events)
            for policy in ('balanced','area','balanced'):
                if w.remaining()<2:break
                kind='batch' if policy=='balanced' else policy
                if (structural_id(w.current),kind,1.) not in w.blocked:w.compress(policy,1.)
            rounds.append(dict(iteration=iteration,area=w.current.lib.area(w.current.units),
                training_errors=w.loss,seconds=time.perf_counter()-started))
            save(folder/'events.json',events);save(folder/'rounds.json',rounds)
            if structural_id(w.current)==identity:break
        return_detail=dict(profile=profile,seconds=time.perf_counter()-started,initial_area=w.current.lib.area(before),
            last_area=w.current.lib.area(w.current.units),changed=structural_id(w.current)!=parent_graph,
            training_cap=working.limit,development_cap=development_limit(working,options),formal_cap=formal,
            rounds=rounds,accepted_joint_plans=sum(e.get('accepted',False) for e in events),
            actual_plan_checks=len(events),proxy_disagreements=sum(e.get('proxy_training_exact') is False for e in events))
        save(folder/'report.json',return_detail);return return_detail
    finally:
        # Histories retain measured error counts. Subsequent controller actions
        # use their declared global rules; the formal validator is never narrowed.
        for row in w.history:
            row.setdefault('experience_profile',profile)
        w.args,w.metric,w.global_remaining=previous_args,previous_metric,previous_remaining
        # Feasibility rejections/plateaus at a local400-vector cap do not prove
        # a plateau under the restored global500-vector allowance.
        reset_caches(w);w.blocked=set();w.failed=set()


def add_arguments(p):
    p.add_argument('--experience-seconds',type=float,default=120.)
    p.add_argument('--experience-roots',type=int,default=160)
    p.add_argument('--experience-shortlist',type=int,default=144)
    p.add_argument('--experience-rounds',type=int,default=6)
    p.add_argument('--experience-generation-seconds',type=float,default=45.)
    p.add_argument('--experience-joint-fraction',type=float,default=.8)
    p.add_argument('--experience-wide-fraction',type=float,default=.9)
    return p
