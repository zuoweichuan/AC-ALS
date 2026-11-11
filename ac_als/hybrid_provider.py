"""Optional SiliconFlow adapter for the same observation/decision protocol.

No credentials are stored; absent credentials stop, without numerical fallback.
Use --once to limit a smoke test to one actual model adjudication.
"""
import runtime
import argparse,copy,json,os,time,urllib.request,urllib.error
import hashlib
from pathlib import Path
from hybrid_agent import ROOT,Session,lease,read,save
from hybrid_secrets import load_key

SYSTEM='''You are a global controller for compensation-aware approximate netlist synthesis.
Return one JSON object with action, node when needed, options, reason.
The adapter binds decision_id to this exact serialized request; it is bookkeeping,
not a search choice. At zero free slots, fork=true requires an explicit evict of
another active branch. Alternatively use fork=false to continue this branch;
its ancestor remains stored and can later be revisited.
Legal actions: scan(node, options: width=4|6, mode=efficiency|area|repair, offset=0,
roots<=64, optional joint=true, bundled=true); apply(node, options: ids=[candidate IDs], fork=true,
optional evict=another active node, allow_near_miss=true only when justified);
rollout(node, options: steps<=8, width=4|6, policy=efficiency|area|error|budget,
mode=efficiency|area|repair, roots<=64, offset=0 (nonnegative root-order region), area_debt_units<=384 (respect observation's limit),
optional repair_error_target=integer number of errors <= constraint*10000);
validate(node, options={}) measures development ER on the separate fixed 10k stream;
archive(node); revisit(archived node, options: optional evict=active node);
finish(no node required).
Keep at most four active branches. A local plateau is not global convergence.
Partial output-bit repairs may leave vector ER unchanged until combined with
another repair; hamming_gain describes this opportunity, never the acceptance
constraint. Final feasibility always uses vector ER, not Hamming distance.
Scan distinct regions, preserve feasible incumbents, explore high-gain near misses
and target their error masks for compensation. Temporary area debt needs real error
recovery and a plausible net area gain. Cached scalar costs from other parents are
not additive. Finish only after explaining why remaining measured opportunities
are poor. Do not stop merely because a reference area was reached. Never fabricate
measurements. Candidate IDs belong to the latest scan of that exact branch.
Development validation may already identify a nearly feasible warm start. Use
mode=repair with repair_error_target below the formal training ceiling to restore
some error margin, instead of discarding a valuable near miss or adding only more
area-reducing, error-consuming actions. This development set is not a fresh final
test. You decide global search; a rollout only executes a bounded local tool.'''
SYSTEM+='''\nA sampled candidate close to the training limit can fail on independent inputs.
Prefer an error-margin branch (typically 80-90% of the training error allowance)
alongside high-gain near misses. Inspect measured global progress rather than
spending a full round on equal-function variants. Repeated physical netlists are
not new progress. Compensating constant/alias primary outputs is allowed.
When a warm-start branch is trapped, revisit a lower-error archived prefix and
try a different action family. Four active slots do not mean forgotten ancestors.
Do not continue long repair chains that only undo one another.
Apply only on a branch that has displayed choices from its latest scan. A child
does not inherit its parent's candidate IDs. When a joint action is rejected,
last_tool_error.measurement contains its actually propagated area and ER; assess
that result before authorizing a near miss or choosing a different combination.
input_index_* candidates are small bundles screened by error-input support.
Their listed area and vector ER come from actual joint propagation, not sums of
single-node errors. They can remove shared logic that no one-node action releases.'''
SYSTEM+='''\nEach branch lists its own legal_actions and candidate_scope. An unscanned branch
cannot apply. Candidate IDs encode both the immutable node and scan revision;
neither parent IDs nor IDs from an earlier scan are valid. arrival_history only
describes how this branch was created, not choices available on this branch.
For a budget-preserving trajectory use rollout(policy=budget, steps<=8,
error_fraction=0.85, max_error_increment=0.15). This rescans after every accepted
step, limits total training errors to error_fraction times the formal allowance,
and each step's added errors to max_error_increment times that allowance. You
can set either fraction in (0,1]. It returns control for your global decision.
Combined with mode=repair, budget policy instead uses error-first repair and
stops at the smaller of repair_error_target and error_fraction times the allowance.
If that target is already met, it returns without doing more simulations.
Prefer this bounded tool when pursuing several low-error cuts, rather than
guessing candidate IDs for children that have not yet been scanned.'''
SYSTEM+='''\nknown_transitions are already measured children, not choices to apply again.
Use a listed child directly if active, or revisit it if archived. Choose only
unmeasured IDs from the current branch choices. A scan's joint_attempts include
actual rejected area/ER measurements. A high-gain combination just outside the
current ceiling can justify scan(ceiling=..., mode=repair) up to the observation's
exploration_ceiling_max, then a targeted compensation branch. Explicitly assess
the measured excess and repair evidence; never raise the formal final ER limit.'''
SYSTEM+='''\nrepair_economics compares each branch to a measured incumbent in integer
physical-area units. The protocol maximum area debt is permission, not a
recommended spend. A near miss that saves only a few units cannot retain a net
gain after an expensive error-first repair unless you identify a measured path
to recompress it. Compare training_error_delta before calling a branch lower
error: merely being feasible does not make it lower error than the incumbent.
For a small overshoot, consider first restoring feasibility with a modest debt
and repair_error_target at the formal training allowance; policy=budget with
mode=repair also requires error_fraction=1 if that is your intended target.
Pursue a deeper margin only when its extra area cost is justified. A flip repair
observed on a parent is only a hypothesis for its child; rescan the exact child
before claiming that evidence still applies. Larger lower-error exploration is
allowed, but explain the recompression opportunity and keep it bounded.'''

class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*a,**kw):return None

def compact(observation):
 context=copy.deepcopy(observation)
 context['active_branch_ids']=[b['id'] for b in context['branches']]
 context['free_branch_slots']=4-len(context['branches'])
 for branch in context['branches']:
  origin=branch.pop('origin',{})
  branch['arrival_history']={k:origin[k] for k in ('kind','controller_source','actual_train_er','allow_near_miss') if k in origin}
  if origin.get('actions'):
   branch['arrival_history']['parent_action_summary']=[{k:a[k] for k in ('target','targets','family','gain_units','error') if k in a} for a in origin['actions']]
  ready=bool(branch.get('choices'))
  branch['legal_actions']=['scan','rollout','validate','archive']+(['apply'] if ready else [])
  branch['candidate_scope']=dict(branch.get('candidate_scope',{}),ready=ready,
   rule='Use only IDs in this branch choices from its latest scan. arrival_history contains no executable choices.')
  branch['expanded']=[{k:v for k,v in e.items() if k in ('width','offset','roots','mode','joint','generation_seconds')} for e in branch['expanded'][-4:]]
  if 'scan' in branch:branch['scan']['root_evidence']=branch['scan']['root_evidence'][:8]
  choices=branch.get('choices',[]);selected={};counts={};supports=set();categories={}
  allowance=observation['er_limit'];budget=max(.25*allowance,branch['train_er']+.15*allowance)
  queues=[('area',sorted(choices,key=lambda c:(c['result_area'],c['er'])),5),
   ('efficiency',sorted(choices,key=lambda c:(-max(0,c['area_gain'])/(25+c['new_errors']),c['result_area'])),5),
   ('budget_preserving',sorted((c for c in choices if c['feasible'] and c['area_gain']>0 and c['er']<=budget),key=lambda c:(c['result_area'],c['er'])),4),
   ('repair',sorted(choices,key=lambda c:(c['er'],c['result_area'])),4),
   ('near_miss',sorted((c for c in choices if not c['feasible']),key=lambda c:(c['excess_errors']/max(.001,c['area_gain']),c['result_area'])),4)]
  for round_no in range(5):
   for name,ordered,quota in queues:
    categories.setdefault(name,[])
    if len(categories[name])>=quota:continue
    for c in ordered:
     roots=tuple(c['targets']);sig=(roots,c.get('error_support_signature',c['id']),round(c['result_area']/.266))
     if c['id'] in selected or counts.get(roots,0)>=4 or sig in supports:continue
     selected[c['id']]=c;categories[name].append(c['id']);counts[roots]=counts.get(roots,0)+1;supports.add(sig);break
  if choices:
   # One choice can serve several regimes. Preserve those category labels even
   # when round-robin selection first retained it under another objective.
   categories={name:[c['id'] for c in ordered if c['id'] in selected][:quota] for name,ordered,quota in queues}
   branch['choices']=list(selected.values());branch['candidate_categories']=categories;branch['total_screened_choices']=len(choices)
 for d in context['recent_decisions']:
  d.pop('model_receipt',None);d['reason']=d['reason'][:700]
  if isinstance(d.get('result'),list):d['result']=dict(screened_choices=len(d['result']))
 return context

def bind_request_id(decision,observation):
 # choose() is synchronous under the session's exclusive worker lease. Bind
 # model actions to the observation actually sent, not an unreliable echo of
 # bookkeeping metadata. Session.execute still rejects truly stale requests.
 supplied=decision.get('decision_id')
 if supplied!=observation['decision_id']:
  decision['request_binding']=dict(model_supplied_id=supplied,bound_id=observation['decision_id'],reason='Adapter binds the synchronous response to its originating observation')
 decision['decision_id']=observation['decision_id'];return decision

def choose(observation):
 secret=load_key()
 context=compact(observation)
 payload=dict(model='deepseek-ai/DeepSeek-V4-Flash',temperature=.1,max_tokens=900,stream=False,enable_thinking=False,
  response_format={'type':'json_object'},messages=[dict(role='system',content=SYSTEM),dict(role='user',content=json.dumps(context,ensure_ascii=False))])
 encoded=json.dumps(payload,ensure_ascii=False).encode('utf-8')
 if len(encoded)>180000:raise RuntimeError('Compact Agent context exceeds 180kB; reduce the observation, not the search')
 # Audit exact model input, including the generic knowledge version. No key is
 # part of the payload. Use a unique name for corrective calls with the same ID.
 audit=Path(observation['session'])/'model_inputs';audit.mkdir(exist_ok=True)
 payload_hash=hashlib.sha256(encoded).hexdigest()
 save(audit/f'{observation["decision_id"]:05d}_{time.time_ns()}.json',dict(sha256=payload_hash,payload=payload))
 # Peak, cache-miss prices verified 2026-10-27. Reserve by UTF-8 byte upper
 # bound plus message overhead; never assume a cache hit or off-peak discount.
 billing=ROOT/'hybrid_billing'/observation['design'];billing.mkdir(parents=True,exist_ok=True)
 ledger_path=billing/'ledger.json';reserve=((len(encoded)+1000)*3+900*9)/1000000
 with lease(billing):
  ledger=read(ledger_path) if ledger_path.exists() else dict(cap_cny=5.,spent_upper_cny=0.,requests=[],pricing_source='https://www.siliconflow.cn/pricing',verified='2026-10-27',input_cny_per_million=3.,output_cny_per_million=9.)
  if ledger['spent_upper_cny']+reserve>ledger['cap_cny']:raise RuntimeError('Per-circuit API cost protection reached; checkpoint preserved, no fallback')
  request_index=len(ledger['requests']);ledger['spent_upper_cny']+=reserve
  ledger['requests'].append(dict(session=observation['session'],decision_id=observation['decision_id'],reserved_cny=reserve,status='reserved_or_uncertain'))
  save(ledger_path,ledger)
 request=urllib.request.Request('https://api.siliconflow.cn/v1/chat/completions',encoded,
  headers={'Content-Type':'application/json','Authorization':'Bearer '+secret})
 tick=time.perf_counter()
 try:
  with urllib.request.build_opener(NoRedirect()).open(request,timeout=90) as response:body=response.read(1048577)
  if len(body)>1048576:raise RuntimeError('Model response is oversized')
  completion=json.loads(body);decision=json.loads(completion['choices'][0]['message']['content'])
 except urllib.error.HTTPError as exc:raise RuntimeError(f'Provider HTTP {exc.code}; no automatic retry') from None
 except (OSError,ValueError,KeyError,IndexError):raise RuntimeError('Provider transport/JSON failure; no automatic retry or fallback') from None
 decision=bind_request_id(decision,observation)
 if decision.get('action')=='apply':
  options=decision.setdefault('options',{});branch=next((b for b in observation['branches'] if b['id']==decision.get('node')),None)
  choices={c['id']:c for c in branch.get('choices',[])} if branch else {}
  if any(i in choices and not choices[i]['feasible'] for i in options.get('ids',[])) and 'allow_near_miss' not in options:
   options['allow_near_miss']=True;decision['adapter_normalization']='Inferred near-miss permission from the model explicitly selecting a displayed infeasible candidate'
 usage=completion.get('usage') or {};spent=(usage.get('prompt_tokens',0)*3+usage.get('completion_tokens',0)*9)/1000000
 with lease(billing):
  ledger=read(ledger_path);entry=ledger['requests'][request_index]
  if usage.get('prompt_tokens') is not None and usage.get('completion_tokens') is not None and spent<=reserve:
   ledger['spent_upper_cny']+=spent-reserve;entry['estimated_upper_cny']=spent
  entry.update(status='response_received',response_id=completion.get('id'),usage=usage);save(ledger_path,ledger)
 decision['source']='siliconflow_agent'
 decision['model_receipt']=dict(model=payload['model'],response_id=completion.get('id'),usage=usage,estimated_upper_cny=spent,seconds=time.perf_counter()-tick,input_sha256=payload_hash)
 return decision

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',required=True);p.add_argument('--once',action='store_true');p.add_argument('--round-seconds',type=float,default=900);args=p.parse_args()
 folder=Path(args.session).resolve()
 with lease(folder):
  session=Session(folder)
  round_started=time.perf_counter()
  while session.state['status']!='agent_finished':
   if time.perf_counter()-round_started>=args.round_seconds:
    save(folder/'round_checkpoint.json',dict(reason='soft_round_time_boundary',seconds=time.perf_counter()-round_started,status='awaiting_agent',best_train=session.state['best_train']));break
   observation=session.write_observation();decision=choose(observation)
   save(folder/f'provider_decision_{decision["decision_id"]:05d}.json',decision)
   session.execute(decision)
   if args.once:break
  if session.state['status']=='agent_finished':
   from hybrid_selection import validation_portfolio
   for node in validation_portfolio(session.state,session.limit):
    if not (folder/'nodes'/f'{node}_verify1.json').exists():session.verify(node,1)
if __name__=='__main__':main()
