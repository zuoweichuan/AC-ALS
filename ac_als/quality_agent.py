"""DeepSeek controls the same conditional/compensation/physical tools as MCTS.

All global actions (including refinement and stopping) need a model choice.
No UCT/default fallback, historical approximate start or automatic final polish.
"""
import copy,json,math,os,shutil,time
from pathlib import Path
from bacs_data import load,LIBRARY_PATH
from bacs_numeric import save
from logic import Library
from research_c880_branch import MCTS as CommonTools
from numeric_agentic import clone_worker,select_checkpoint,decision_reserve
from numeric_explore import action_record
from hybrid_identity import structural_id
from er_repair_tools import development_limit
from quality_operators import extend,operator_schedule,operator_signature
from quality_research_mcts import parser as shared_parser
from prepare_resub_iscas import sha
import er_resub_data as data

ROOT=Path(__file__).resolve().parent
KNOWLEDGE=[
 'Migrate conditional construction, compensation paths and physical operators together; a nominally larger space is insufficient.',
 'Equal area can hide different error functions and downstream mapping potential. Preserve a small area/error/representation frontier.',
 'Preserved intermediate topology and toggle_view can expose opportunities even when a later scalar checkpoint dominates.',
 'After promising conditional exploration, reserve time for the full physical package instead of stopping on a native plateau.',
 'MFFC gain is a screening proxy. Adopt only actual physical area and both original train/development guards.',
 'A fresh graph needs fresh cut/response information; a parent plateau is not proof about the child.',
 'Use measured repair economics. Temporary area debt is permission, not a recommended spend.',
 'Finish is a global cost/value judgement, not proof of optimality or a target-area trigger.'
 ,'Initial zero-error exact mapping is NOT measured approximate exploration. Probe conditional construction before spending the full physical budget on an unprobed accurate view.'
 ,'At the first native compression plateau, an untried exact toggle_view is a cheap structural probe worth considering before committing most of the remaining physical budget.'
]


def bind_choice(decision,bindings):
    if not isinstance(decision,dict) or decision.get('choice_id') not in bindings:
        raise ValueError('Unknown current-request choice_id; no numerical action executed')
    reason=decision.get('reason','')
    if not isinstance(reason,str) or len(reason)>2400:raise ValueError('Invalid decision reason')
    return dict(bindings[decision['choice_id']],reason=reason)


def billing_receipt(session,design):
    # Frozen code lives under suite/source, but all actual providers share the
    # project ledger. Never infer zero API usage from a missing snapshot file.
    root=Path(os.environ.get('NETLIST_FRONTIER_ROOT',str(ROOT)))
    path=root/'hybrid_billing'/design/'ledger.json'
    ledger=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    entries=[r for r in ledger.get('requests',[]) if r.get('session')==str(session)]
    return dict(api_calls=len(entries),api_responses=sum(r.get('status')=='response_received' for r in entries),
        api_cost_upper_cny=sum(r.get('estimated_upper_cny',r.get('reserved_cny',0.)) for r in entries),
        shared_billing_ledger=str(path),ledger_total_upper_cny=ledger.get('spent_upper_cny'))


class QualityAgent(CommonTools):
    def __init__(self,args):
        self.local_deadline=None;self.physical_seconds=0.;self.physical_calls=0
        self.research_budget=args.research_seconds;self.physical_budget=args.quality_seconds;self.resource_transfers=[]
        self.operator_coverage={};self.billing_root=Path(os.environ.get('NETLIST_FRONTIER_ROOT',str(ROOT)))
        self.closed=set();self.refined=set();self.valid_model_decisions=0;self.executed_tools=0
        self.bindings={};self.observation_revision=0;self.agent_stop_reason=None
        super().__init__(args)
        for name in ('quality_agent.py','quality_agent_provider.py','batch_provider.py','quality_operators.py',
            'research_c880_conditional.py','research_c880_branch.py','research_c880_views.py',
            'research_c880_loss_cut.py','research_c880_exact_frontier.py','research_c880_observable_exact.py',
            'research_c880_local_care_sat.py','research_c880_mapped_sat.py','research_c880_budget_care_sat.py',
            'research_c880_cut_abc.py'):
            source=ROOT/name;shutil.copy2(source,self.folder/'source'/name)
            self.protocol['code_sha256'][name]=sha(source)
        self.protocol.update(search_algorithm='LLM global selection over immutable screened conditional search forest',
            quality_agent_migration=True,shared_tools_with='quality_research_mcts.py',
            no_uct_selection=True,no_default_fallback=True,no_automatic_final_refinement=True,
            historical_approximate_import=False,research_seconds_excluding_llm=args.research_seconds,
            quality_seconds=args.quality_seconds,api_cap_cny=args.api_cap_cny,
            knowledge=KNOWLEDGE,controller='agent')
        save(self.folder/'protocol.json',self.protocol)
        print('[Agent CHECK PASSED] model='+getattr(args,'model','deepseek-ai/DeepSeek-V4-Flash')+'; credential identified; awaiting real model decisions',flush=True)

    def remaining(self):
        left=super().remaining()
        return min(left,max(0.,self.local_deadline-time.perf_counter())) if self.local_deadline is not None else left

    def research_left(self):
        spent=time.perf_counter()-self.started-self.api_seconds-self.physical_seconds
        return max(0.,self.research_budget-spent)

    def physical_left(self):return max(0.,min(self.physical_budget-self.physical_seconds,self.remaining()-1))

    def transfers_ready(self):
        return not self.resource_transfers or self.resource_transfers[-1].get('tool_epoch')!=(self.expansions,self.physical_calls)

    def physical_options(self,profile):
        options=copy.copy(self.args);options.quality_checkpoint_prefix='agent_quality_'
        filters=dict(care=('budget_care','local_care','local_exact'),local=('local_care','local_exact'),
                     exact=('exact_frontier','observable','mapped_sat','local_exact'),fitted=('fitted_cut',))
        if profile in filters:options.quality_operator_filter=filters[profile]
        options.quality_no_gain_coverage=self.operator_coverage
        return options

    def rows(self):return [r for bid in self.workers for r in self.development_rows(bid)]

    def card(self,bid):
        w=self.touch(bid);valid=[r for r in self.development_rows(bid) if r['eligible']]
        best=select_checkpoint(valid) if valid else None
        return dict(id=bid,working_area=self.lib.area(w.current.units),working_train_errors=w.loss,
            working_development_errors=w.development_context()['bad'].bit_count(),
            best_eligible_area=best['area'] if best else None,preserve_view=w.preserve,
            native_compression_stalled=(structural_id(w.current),'batch',1.) in w.blocked,
            children=len(self.tree[bid]['children']),visits=self.tree[bid]['visits'],
            stalls=w.stalls,unprobed=not self.tree[bid]['children'])

    def observation(self):
        tick=time.perf_counter();bindings={};menu=[];cards=[]
        def add(binding,**evidence):
            handle='a%02d'%len(menu);bindings[handle]=binding
            menu.append(dict(id=handle,kind=binding['kind'],**evidence))
        pending=[bid for bid,row in self.tree.items() if bid!='root' and bid not in self.closed
                 and not row['terminal'] and len(row['children'])<4]
        # Keep accurate unprobed views and measured area-rich frontiers visible.
        ranked=sorted(pending,key=lambda bid:(self.tree[bid].get('best_eligible_area',self.lib.area(self.source.units)),bid))
        first=[bid for bid in self.active if bid in pending and not self.tree[bid]['children']]
        candidates=list(dict.fromkeys([*first,*ranked]))[:8]
        self.local_deadline=time.perf_counter()+min(self.research_left(),self.remaining())
        try:
            for bid in candidates:
                if self.remaining()<.5:break
                card=self.card(bid);cards.append(card);w=self.touch(bid)
                options=self.menu(bid);start=self.tree[bid]['next_action']
                for index,action in enumerate(options[start:],start):
                    m=action.get('move');evidence=dict(parent=bid,action=action['kind'])
                    if m is not None:
                        d=w.development_score(m)
                        evidence.update(target=m.target,inputs=list(m.names),family=m.family,
                            proxy_area_gain=self.lib.area(m.gain),predicted_train_errors=m.error,
                            predicted_development_errors=d['loss'],
                            training_excess_fraction=max(0.,m.error/self.metric.limit-1),
                            actual_child_repair='fresh response screening after model authorization' if action['kind']=='near_repair' else None)
                    else:
                        evidence.update(policy=action.get('policy'),fraction=action.get('fraction'))
                        if action['kind']=='toggle_view':
                            evidence['description']='Cheap exact representation change plus short conditional compression; test this untried view at a native plateau before paying for a full physical package'
                    add(dict(kind='expand',parent=bid,index=index),**evidence)
                if options[start:]:add(dict(kind='close',parent=bid),parent=bid,description='Stop remaining expansions of this node; retain measured descendants and checkpoints')
        finally:self.local_deadline=None
        eligible=[r for r in self.rows() if r['eligible']]
        best=select_checkpoint(eligible);distinct={}
        for row in sorted(eligible,key=lambda r:(r['units'],r['development_loss'],r['training_loss'])):
            key=sha(self.workers[row['branch']].folder/row['path']);distinct.setdefault(key,row)
        quality_parents=[(sha(self.workers[best['branch']].folder/best['path']),best)]
        alternatives=sorted(((key,r) for key,r in distinct.items() if r['units']<=best['units']+32),
                            key=lambda pair:(pair[1]['development_loss']+pair[1]['training_loss'],pair[1]['units']))
        quality_parents=list({key:(key,r) for key,r in [*quality_parents,*alternatives[:2]]}.values())[:3]
        # A full-package opportunity needs at least one actual conditional
        # probe, or evidence that no native expansion is available. Exact
        # initialization alone must not masquerade as a searched function.
        native_available=any(item['kind']=='expand' for item in menu)
        physical_ready=self.expansions>0 or not native_available
        physical_opportunities=0
        if self.physical_left()>=4 and physical_ready:
            for key,row in quality_parents:
                for profile,cap in (('full',self.physical_left()),('care',min(80.,self.physical_left())),
                                    ('local',min(60.,self.physical_left())),('exact',min(65.,self.physical_left())),('fitted',min(45.,self.physical_left()))):
                    if (key,profile) in self.refined:continue
                    if profile=='full' and cap<120:continue
                    plan=operator_schedule(self.physical_options(profile),self.metric.limit,development_limit(self.metric,self.args))
                    uncovered=[name for name,_,_,parameters in plan if operator_signature(name,parameters) not in self.operator_coverage.get(key,[])]
                    if not uncovered:continue
                    physical_opportunities+=1
                    add(dict(kind='refine',checkpoint=row,sha256=key,profile=profile),
                        operation='refine_'+profile,parent=row['branch'],checkpoint=row['path'],area=row['area'],
                        training_errors=row['training_loss'],development_errors=row['development_loss'],
                        seconds_cap=cap,unmeasured_operator_families=list(dict.fromkeys(uncovered)),
                        description='Same physical package as MCTS; cached same-parent/parameter no-gain probes skipped' if profile=='full' else 'Explicit subset of common physical tools')
        if self.transfers_ready() and self.physical_left()<30 and self.research_left()>12 and self.remaining()>15:
            amount=min(60.,self.research_left()-2)
            add(dict(kind='reallocate',seconds=amount,destination='physical'),seconds=amount,
                description='Transfer UNUSED exploration allowance to physical tools; total wall limit stays unchanged')
        if self.transfers_ready() and self.research_left()<10 and self.physical_left()>35 and self.remaining()>15:
            amount=min(40.,max(0.,self.physical_budget-self.physical_seconds)-15)
            add(dict(kind='reallocate',seconds=amount,destination='exploration'),seconds=amount,
                description='Transfer UNUSED physical allowance to conditional branch exploration; total wall limit stays unchanged')
        add(dict(kind='finish'),description='Stop by model judgement; no automatic polish or fallback follows')
        self.bindings=bindings;self.observation_revision+=1
        obs=dict(session=str(self.folder),decision_id=len(self.decisions),revision=self.observation_revision,
            design=self.args.design,metric='strict_vector_ER',formal_bound=self.args.bound,
            search_error_caps=[self.metric.limit,development_limit(self.metric,self.args)],vectors_per_group=10000,
            remaining_wall_seconds=self.remaining(),remaining_exploration_seconds_excluding_llm=self.research_left(),
            remaining_physical_seconds=self.physical_left(),incumbent_area=best['area'],
            physical_opportunity_profiles=physical_opportunities,operator_no_gain_cache_entries=sum(len(v) for v in self.operator_coverage.values()),
            physical_refinement_ready=physical_ready,
            physical_readiness_evidence='a conditional expansion has been measured or no native expansion is available' if physical_ready else 'only exact initialization is measured; conditional probes are available',
            incumbent_training_errors=best['training_loss'],incumbent_development_errors=best['development_loss'],
            max_children=4,max_resident_graphs=4,branches=cards,decision_menu=menu,general_knowledge=KNOWLEDGE,
            recent_results=[{k:d[k] for k in ('choice_id','reason','operation','result') if k in d} for d in self.decisions[-4:]],
            initialization='accurate author baseline; no historical approximate netlist or target area')
        save(self.folder/'observation.json',obs);folder=self.folder/'agent_bindings';folder.mkdir(exist_ok=True)
        save(folder/('%04d.json'%len(self.decisions)),dict(observation=obs,bindings=bindings))
        self.scan_seconds+=time.perf_counter()-tick
        return obs

    def execute_choice(self,binding):
        kind=binding['kind']
        if kind=='finish':self.agent_stop_reason='model_finish';self.status='agent_finished';return dict(stopped=True)
        if kind=='close':self.closed.add(binding['parent']);return dict(closed=binding['parent'])
        if kind=='reallocate':
            amount=binding['seconds']
            if binding['destination']=='physical':
                if amount>self.research_left():raise ValueError('Exploration allowance already spent')
                self.research_budget-=amount;self.physical_budget+=amount
            else:
                if amount>max(0.,self.physical_budget-self.physical_seconds):raise ValueError('Physical allowance already spent')
                self.physical_budget-=amount;self.research_budget+=amount
            self.resource_transfers.append(dict(destination=binding['destination'],seconds=amount,tool_epoch=(self.expansions,self.physical_calls)))
            return dict(destination=binding['destination'],seconds=amount,total_wall_budget_unchanged=True)
        if kind=='expand':
            bid=binding['parent'];index=binding['index'];node=self.tree[bid]
            if bid in self.closed or node['terminal'] or len(node['children'])>=4 or self.research_left()<.5:
                raise ValueError('Expansion no longer legal')
            options=self.menu(bid);next_index=node['next_action']
            if not next_index<=index<len(options):raise ValueError('Stale expansion choice')
            action=options.pop(index);options.insert(next_index,action);node['next_action']+=1
            self.local_deadline=time.perf_counter()+min(self.research_left(),self.remaining())
            try:super().expand(bid,action)
            finally:self.local_deadline=None
            self.executed_tools+=1;r=self.coverage[-1]
            return dict(child=r['child'],action=action_record(action),area=r['after_eligible_area'],
                wrong_vector_metric=r['training_metric'],seconds=r['seconds'],
                repaired=r['detail'].get('repaired'),duplicate_of=r['detail'].get('duplicate_of'))
        if kind=='refine':
            if self.physical_left()<4:raise ValueError('Physical resource budget expired')
            row=binding['checkpoint'];path=self.workers[row['branch']].folder/row['path']
            if sha(path)!=binding['sha256']:raise ValueError('Checkpoint changed after observation')
            key=(binding['sha256'],binding['profile'])
            if key in self.refined:raise ValueError('Physical proposal already measured')
            self.refined.add(key);c=load(path,self.lib);name='p%03d'%self.physical_calls;self.physical_calls+=1
            w=clone_worker(self.workers[row['branch']],self.folder/name,self.remaining,self.started,c,row['training_loss'],False)
            w.training_view=None;w.development_view=None;w.bit_cache=None;w.verified_pairs={}
            self.workers[name]=w;self.origin[name]=dict(kind='agent_selected_physical_tool',parent=row['branch'],checkpoint=row['path'])
            for other in self.workers.values():
                if other is w:continue
                other.current=None;other.cached=None;other.training_view=None;other.development_view=None
                other.bit_cache=None;other.verified_pairs={}
                if hasattr(other,'area_model'):del other.area_model
            self.cache_order=[]
            options=self.physical_options(binding['profile'])
            profile=binding['profile'];cap=self.physical_left()
            if profile=='care':options.quality_operator_filter=('budget_care','local_care','local_exact');cap=min(cap,80.)
            elif profile=='local':options.quality_operator_filter=('local_care','local_exact');cap=min(cap,60.)
            elif profile=='exact':options.quality_operator_filter=('exact_frontier','observable','mapped_sat','local_exact');cap=min(cap,65.)
            elif profile=='fitted':cap=min(cap,45.)
            started=time.perf_counter()
            try:detail=extend(w,self.folder/('quality_'+name),cap,options)
            finally:self.physical_seconds+=time.perf_counter()-started
            # A model-selected physical result is a new current-run exact view.
            # It is a forest root, not a fictitious UCT visit. The Agent can
            # subsequently branch from it using the same four-child tools.
            self.tree[name]=dict(parent=None,children=[],visits=0,value_sum=0.,terminal=not detail['accepted'],
                                 graph=structural_id(w.current),next_action=0)
            self.archive_graph(name);self.touch(name);self.assess(name);self.executed_tools+=1
            return dict(branch=name,profile=profile,area=detail['area'],accepted=detail['accepted'],
                seconds=detail['seconds'],operators=[dict(operator=s['operator'],area=s['proposed_area'],accepted=s['accepted'],cached=s.get('skipped_verified_no_gain',False)) for s in detail['stages']])
        raise ValueError('Unknown bound operation')

    def persist_tree(self):
        super().persist_tree();path=self.folder/'state.json';s=json.loads(path.read_text(encoding='utf-8'))
        s.update(controller='agent',api_calls=self.calls,valid_model_decisions=self.valid_model_decisions,
                 physical_seconds=self.physical_seconds,closed=list(sorted(self.closed)))
        save(path,s);save(self.folder/'agent_decisions.json',self.decisions)

    def run(self):
        while self.remaining()>2 and len(self.decisions)<self.args.decisions and self.status=='running':
            if self.remaining()<=decision_reserve([d.get('receipt') for d in self.decisions]):
                self.status='time_limit_reserve';break
            obs=self.observation()
            if self.remaining()<=decision_reserve([d.get('receipt') for d in self.decisions]):
                self.status='time_limit_reserve';break
            record=dict(decision_id=len(self.decisions),revision=obs['revision'])
            started=time.perf_counter();receipt=None
            try:
                from quality_agent_provider import choose
                self.calls+=1
                try:
                    decision,receipt=choose(obs,self.remaining()-1,self.args.api_cap_cny)
                    self.http_seconds+=receipt.get('http_request_seconds',0.)
                finally:self.api_seconds+=time.perf_counter()-started
                record.update(decision,receipt=receipt)
                binding=bind_choice(decision,self.bindings);self.valid_model_decisions+=1
                record['operation']=binding['kind']
                print('[Agent CHOICE VALID]',json.dumps(dict(id=record['decision_id'],calls=self.calls,
                    valid=self.valid_model_decisions,operation=binding['kind'],reason=binding['reason']),ensure_ascii=False),flush=True)
                self.persist_tree()
                record['result']=self.execute_choice(binding) if self.remaining()>1 else dict(skipped='wall_budget_expired')
            except ValueError as exc:
                record['result']=dict(rejected=True,error=str(exc))
            except RuntimeError as exc:
                self.error=str(exc);self.status='controller_failed';record['result']=dict(controller_error=str(exc))
            self.decisions.append(record);self.persist_tree()
            print('[Agent DECISION]',json.dumps(dict(id=record['decision_id'],valid=self.valid_model_decisions,
                calls=self.calls,operation=record.get('operation'),reason=record.get('reason'),result=record.get('result')),ensure_ascii=False),flush=True)
            if self.error:break
        if self.status=='running':self.status='time_limit' if self.remaining()<=2 else 'decision_limit'
        self.persist_tree();return self.finish()

    def finish(self):
        # Reporting/selection only. No automatic numerical tool is called here.
        selected=select_checkpoint(self.rows());winner=load(self.workers[selected['branch']].folder/selected['path'],self.lib)
        winner.write(self.folder/'selected_before_checks.v');frozen=sha(self.folder/'selected_before_checks.v')
        save(self.folder/'frozen_selection.json',dict(checkpoint=selected,sha256=frozen,selection='both search guards only, before all final checks'))
        self.persist_tree();code=super().finish();p=self.folder/'report.json';r=json.loads(p.read_text(encoding='utf-8'))
        if r['netlist_sha256']!=frozen:raise AssertionError('Final checks changed winner')
        r.update(billing_receipt(self.folder,self.args.design))
        r.update(search_algorithm=getattr(self.args,'model','DeepSeek')+' global selection over shared conditional/repair/physical tools',
            quality_agent_migration=True,valid_model_decisions=self.valid_model_decisions,executed_agent_tools=self.executed_tools,
            physical_operator_calls=self.physical_calls,physical_operator_seconds=self.physical_seconds,
            agent_stop_reason=self.agent_stop_reason,observed_max_children=max(len(row['children']) for row in self.tree.values()),
            no_uct_selection=True,no_default_fallback=True,no_automatic_final_refinement=True,
            model_control_verified=r['api_responses']>0 and self.valid_model_decisions>0 and self.executed_tools>0 and not self.error,
            research_numeric_seconds=max(0.,time.perf_counter()-self.started-self.api_seconds-self.physical_seconds),
            tools_shared_with='quality_research_mcts.py',api_cap_cny=self.args.api_cap_cny)
        r.update(resource_transfers=self.resource_transfers,final_research_allowance=self.research_budget,final_physical_allowance=self.physical_budget)
        r.update(operator_no_gain_cache_entries=sum(len(v) for v in self.operator_coverage.values()))
        save(p,r);print('[Agent FINAL CHECK]',json.dumps({k:r[k] for k in ('area','strict_pass','api_calls','valid_model_decisions','executed_agent_tools','model_control_verified','provider_error')}),flush=True)
        return code if r['model_control_verified'] else 3


def parser():
    p=shared_parser();p.description=__doc__;p.set_defaults(controller='agent',decisions=24,quality_seconds=280.)
    p.add_argument('--api-cap-cny',type=float,default=10.)
    return p


def run(a):
    if a.controller!='agent' or a.metric!='er' or a.no_exact or a.logic_view or a.sop_view or a.training_fraction!=1:
        raise ValueError('Use migrated physical strict-ER Agent')
    if False:raise ValueError('D: only')
    if any(not math.isfinite(v) or v<=0 for v in (a.seconds,a.research_seconds,a.quality_seconds,a.decisions,a.api_cap_cny)) or a.api_cap_cny>10:
        raise ValueError('Invalid bounded Agent protocol')
    from hybrid_secrets import load_key
    load_key();tick=time.perf_counter();data.configure(a.baseline)
    data.check_start(a.design,a.accurate_netlist,Library(LIBRARY_PATH))
    a.baseline_precheck_seconds=time.perf_counter()-tick;a.no_word=True
    return QualityAgent(a).run()


if __name__=='__main__':raise SystemExit(run(parser().parse_args()))
