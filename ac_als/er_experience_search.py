"""Cold ER search with explicit recollection and joint compensation tools."""
import copy,json,math,shutil,time
from pathlib import Path
from bacs_data import load,LIBRARY_PATH
from bacs_numeric import save
from logic import Library
from numeric_agentic import clone_worker,select_checkpoint
from hybrid_identity import structural_id
from er_repair_tools import development_limit
from er_residual_research import reset_caches
from er_experience_tools import continue_current,add_arguments,LESSONS,recommended_profile
from quality_research_mcts import QualityResearchMCTS,ConditionalMCTS
from quality_operators import extend
from quality_agent import QualityAgent
from prepare_resub_iscas import sha
import er_resub_data as data

ROOT=Path(__file__).resolve().parent
NEW_SOURCES=['er_experience_search.py','er_experience_tools.py','er_joint_mask_research.py',
             'er_residual_research.py','er_experience_provider.py']


class ExperienceMixin:
    def __init__(self,args):
        self.experience_seconds=0.;self.experience_calls=0;self.experience_records=[];self.experience_tested=set()
        super().__init__(args)
        for name in NEW_SOURCES:
            shutil.copy2(ROOT/name,self.folder/'source'/name);self.protocol['code_sha256'][name]=sha(ROOT/name)
        self.protocol.update(er_experience_migration=True,current_graph_recollection=True,
            joint_plan_beam_width=4,experience_seconds_cap=args.experience_seconds,
            experience_roots=args.experience_roots,experience_shortlist=args.experience_shortlist,
            experience_joint_fraction=args.experience_joint_fraction,experience_wide_fraction=args.experience_wide_fraction,
            generic_experience=LESSONS,experience_within_total_budget=True)
        save(self.folder/'protocol.json',self.protocol)

    def experience_left(self):return max(0.,min(self.args.experience_seconds-self.experience_seconds,self.remaining()-1))

    def experience(self,row,profile,seconds):
        parent=self.workers[row['branch']];path=parent.folder/row['path'];parent_hash=sha(path)
        key=parent_hash,profile
        if key in self.experience_tested:raise ValueError('Experience profile already measured on this checkpoint')
        self.experience_tested.add(key);name='experience_%03d'%self.experience_calls;self.experience_calls+=1
        current=load(path,self.lib)
        w=clone_worker(parent,self.folder/name,self.remaining,self.started,current,row['training_loss'],False)
        # clone_worker intentionally shares old generic objects. The numerical
        # tool replaces args/Metric locally before changing their operating caps.
        reset_caches(w);w.refresh_counts=dict(getattr(parent,'refresh_counts',{}))
        self.workers[name]=w;self.origin[name]=dict(kind='explicit_er_experience',parent=row,profile=profile)
        for other in self.workers.values():
            if other is not w:
                other.current=None;reset_caches(other)
        self.cache_order=[name]
        started=time.perf_counter()
        try:result=continue_current(w,profile,min(seconds,self.experience_left()),self.args,self.folder/(name+'_tool'))
        finally:self.experience_seconds+=time.perf_counter()-started
        if sha(path)!=parent_hash:raise AssertionError('Experience tool mutated parent')
        self.experience_records.append(dict(branch=name,parent=row,**result));save(self.folder/'experience_tools.json',self.experience_records)
        return name,w,result

    def add_experience_report(self):
        path=self.folder/'report.json';r=json.loads(path.read_text(encoding='utf-8'))
        r.update(er_experience_migration=True,experience_calls=self.experience_calls,
            experience_seconds=self.experience_seconds,experience_tools=self.experience_records,
            experience_within_total_budget=True,historical_approximate_import=False,
            experience_not_counted_as_mcts_visits=True)
        save(path,r)


class ExperienceMCTS(ExperienceMixin,QualityResearchMCTS):
    def finish(self):
        parent=select_checkpoint([r for bid in self.workers for r in self.development_rows(bid)])
        source=load(self.workers[parent['branch']].folder/parent['path'],self.lib)
        elapsed=time.perf_counter()-self.started;self.search_deadline=None;tree_before=json.dumps(self.tree,sort_keys=True)
        w=clone_worker(self.workers[parent['branch']],self.folder/'physical_refinement',self.remaining,self.started,
                       source,parent['training_loss'],False)
        reset_caches(w);self.workers['physical_refinement']=w
        self.origin['physical_refinement']=dict(kind='post_search_refinement',parent=parent)
        for other in self.workers.values():
            if other is not w:other.current=None;reset_caches(other)
        self.cache_order=['physical_refinement']
        reserve=min(self.args.experience_seconds,max(0.,self.remaining()-4))
        physical_cap=min(self.args.quality_seconds,max(0.,self.remaining()-reserve))
        detail=extend(w,self.folder/'quality_operators',physical_cap,self.args)
        # Predeclared adaptive choice uses only current training/development
        # headroom, never circuit names or historical best-area recipes.
        row=select_checkpoint([r for bid in self.workers for r in self.development_rows(bid)])
        first=recommended_profile(row['training_loss'],row['development_loss'],self.metric.limit)
        for profile in (first,'wide' if first=='joint' else 'joint'):
            if self.experience_left()<4:break
            row=select_checkpoint([r for bid in self.workers for r in self.development_rows(bid)])
            self.experience(row,profile,self.experience_left())
        if json.dumps(self.tree,sort_keys=True)!=tree_before:raise AssertionError('Refinement changed tree visits')
        self.persist_tree();winner=select_checkpoint([r for bid in self.workers for r in self.development_rows(bid)])
        c=load(self.workers[winner['branch']].folder/winner['path'],self.lib);c.write(self.folder/'selected_before_checks.v')
        frozen=sha(self.folder/'selected_before_checks.v')
        save(self.folder/'frozen_selection.json',dict(checkpoint=winner,sha256=frozen,final_inputs_read=False))
        code=ConditionalMCTS.finish(self)
        r=json.loads((self.folder/'report.json').read_text(encoding='utf-8'))
        if r['netlist_sha256']!=frozen:raise AssertionError('Final validation changed selection')
        r.update(quality_operator_migration=True,conditional_research_kernel=True,
            uct_stage_seconds=elapsed,post_search_parent=parent,post_search_refinement=detail,
            observed_max_children=max(len(q['children']) for q in self.tree.values()),finalization_is_not_tree_visits=True)
        save(self.folder/'report.json',r);self.add_experience_report();return code


class ExperienceAgent(ExperienceMixin,QualityAgent):
    def research_left(self):
        return max(0.,self.research_budget-(time.perf_counter()-self.started-self.api_seconds-self.physical_seconds-self.experience_seconds))

    def observation(self):
        obs=super().observation();obs['general_knowledge']=list(obs['general_knowledge'])+LESSONS
        obs.update(remaining_experience_seconds=self.experience_left(),experience_tools=[
            {k:r[k] for k in ('profile','seconds','initial_area','last_area','changed','accepted_joint_plans')}
            for r in self.experience_records[-3:]])
        if self.experience_left()>=4 and obs['physical_refinement_ready']:
            rows=[r for r in self.rows() if r['eligible']]
            distinct={}
            for row in sorted(rows,key=lambda r:(r['units'],r['development_loss']+r['training_loss'])):
                h=sha(self.workers[row['branch']].folder/row['path']);distinct.setdefault(h,row)
            for h,row in list(distinct.items())[:2]:
                for profile in ('wide','joint'):
                    if (h,profile) in self.experience_tested:continue
                    handle='a%02d'%len(obs['decision_menu'])
                    binding=dict(kind='experience',checkpoint=row,sha256=h,profile=profile)
                    self.bindings[handle]=binding
                    obs['decision_menu'].append(dict(id=handle,kind='experience',operation='experience_'+profile,
                        parent=row['branch'],checkpoint=row['path'],area=row['area'],
                        training_errors=row['training_loss'],development_errors=row['development_loss'],
                        recommended_profile=recommended_profile(row['training_loss'],row['development_loss'],self.metric.limit),
                        seconds_cap=self.experience_left(),conditional_roots=self.args.experience_roots,
                        description='Recollect a bounded wider current-graph conditional shortlist and recompress' if profile=='wide' else
                        'Joint compression plus targeted repair: four-plan beam, at most six actions, real whole-graph checks, reserve error on BOTH sets',
                        operating_fraction=self.args.experience_joint_fraction if profile=='joint' else self.args.experience_wide_fraction))
        save(self.folder/'observation.json',obs)
        save(self.folder/'agent_bindings'/('%04d.json'%len(self.decisions)),dict(observation=obs,bindings=self.bindings))
        return obs

    def execute_choice(self,binding):
        if binding['kind']!='experience':return super().execute_choice(binding)
        row=binding['checkpoint'];path=self.workers[row['branch']].folder/row['path']
        if sha(path)!=binding['sha256']:raise ValueError('Stale experience checkpoint')
        if self.experience_left()<4:raise ValueError('Experience resource allowance exhausted')
        name,w,result=self.experience(row,binding['profile'],self.experience_left())
        self.tree[name]=dict(parent=None,children=[],visits=0,value_sum=0.,terminal=not result['changed'],
                            graph=structural_id(w.current),next_action=0)
        self.archive_graph(name);self.touch(name);self.assess(name);self.executed_tools+=1
        return dict(branch=name,**result)

    def finish(self):
        code=super().finish();self.add_experience_report();return code


def parser():
    from quality_agent import parser as shared
    p=add_arguments(shared());p.description=__doc__
    p.set_defaults(seconds=540.,research_seconds=120.,quality_seconds=280.,decisions=32)
    return p


def run(a,*,authorized_api_cap=10.):
    if a.metric!='er' or a.no_exact or a.logic_view or a.sop_view or a.training_fraction!=1:
        raise ValueError('Use cold physical strict-vector ER')
    values=[a.seconds,a.research_seconds,a.quality_seconds,a.experience_seconds,a.experience_roots,
            a.experience_shortlist,a.experience_rounds,a.experience_generation_seconds,a.decisions,a.expansions]
    if any(not math.isfinite(v) or v<=0 for v in values) or a.seconds>900 or a.experience_roots>256 or a.experience_shortlist>256:
        raise ValueError('Invalid bounded experience resources')
    if a.research_seconds+a.quality_seconds+a.experience_seconds>a.seconds:
        raise ValueError('Stage allowances must fit one shared wall-clock limit')
    if authorized_api_cap not in (10.,30.) or not 0<a.experience_joint_fraction<=1 or not 0<a.experience_wide_fraction<=1 or not 0<a.api_cap_cny<=authorized_api_cap:
        raise ValueError('Invalid margin or cumulative API cap')
    if False:raise ValueError('D: only')
    tick=time.perf_counter();data.configure(a.baseline);data.check_start(a.design,a.accurate_netlist,Library(LIBRARY_PATH))
    a.baseline_precheck_seconds=time.perf_counter()-tick;a.no_word=True
    if a.controller=='mcts':return ExperienceMCTS(a).run()
    from hybrid_secrets import load_key
    load_key()
    import quality_agent_provider
    from er_experience_provider import choose
    original=quality_agent_provider.choose;quality_agent_provider.choose=choose
    try:return ExperienceAgent(a).run()
    finally:quality_agent_provider.choose=original


if __name__=='__main__':raise SystemExit(run(parser().parse_args()))
