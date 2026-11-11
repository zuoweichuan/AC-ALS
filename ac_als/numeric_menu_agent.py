"""Agent chooses a prevalidated action handle; no numerical fallback.

Action selection remains the LLM's choice. Binding a chosen handle to its exact
branch/candidate/arguments is deterministic bookkeeping, not a second controller.
"""
import json
from pathlib import Path
import shutil

from bacs_numeric import save
from bacs_data import load
from hybrid_identity import structural_id
from numeric_agentic import Portfolio, parser, sha, validate_decision


def action_key(d,graph):
    return json.dumps(dict(graph=graph,action=d['action'],branch=d.get('branch'),options=d.get('options',{})),sort_keys=True)


def bind_choice(choice,bindings):
    if not isinstance(choice,dict) or choice.get('choice_id') not in bindings:
        raise ValueError('Choose exactly one current decision_menu handle')
    return dict(bindings[choice['choice_id']],reason=choice.get('reason',''))


def repair_economics(worker,move):
    """Use the same pre-mapping debt ceiling as apply; actual checks still decide."""
    ceiling=min(worker.source.units,worker.lowest_units+16)
    predicted=worker.current.units-move.gain
    return dict(estimated_area_change=worker.source.lib.area(-move.gain),
        estimated_area_increase=max(0.,worker.source.lib.area(-move.gain)),
        estimated_within_debt=predicted<=ceiling,
        remaining_area_debt=worker.source.lib.area(max(0,ceiling-worker.current.units)))


class MenuPortfolio(Portfolio):
    def __init__(self,args):
        super().__init__(args);self.used=set();self.bindings={};self.canonical=None
        own=Path(__file__).resolve();shutil.copy2(own,self.folder/'source'/own.name)
        self.protocol.update(decision_interface='prevalidated operation handles; actual action choice belongs to LLM',
            code_sha256=dict(self.protocol['code_sha256'],**{own.name:sha(own)}))
        save(self.folder/'protocol.json',self.protocol)

    def observation(self):
        full=super().observation();self.canonical=full;bindings={};menu=[];deferred=[]
        def add(d,label,**evidence):
            branch=next((b for b in full['branches'] if b['id']==d.get('branch')),None)
            graph=branch['graph_hash'] if branch else 'global'
            if action_key(d,graph) in self.used:return
            try:validate_decision(d,full)
            except ValueError:return
            handle='a%02d'%len(menu);bindings[handle]=d
            menu.append(dict(id=handle,kind=d['action'],branch=d.get('branch'),description=label,**evidence))
        for b in full['branches']:
            bid=b['id'];worker=self.workers[bid]
            if not b['compression_screen_exhausted']:
                for policy in ('balanced','area'):
                    add(dict(action='compress',branch=bid,options=dict(policy=policy,fraction=1.,steps=3)),
                        'Three screened compression steps; full formal remaining headroom',policy=policy)
            byid={c['id']:c for c in b['choices']}
            repairs=sorted((byid[c] for c in b['categories']['repair']),
                key=lambda c:((max(0.,-c['expected_area_gain'])+.266)/max(1e-12,b['training_metric']-c['predicted_metric']),c['predicted_metric']))
            repairs=[c for c in repairs if repair_economics(worker,self.live_choices[bid][c['id']])['estimated_within_debt']]
            for c in repairs[:2]:
                economics=repair_economics(worker,self.live_choices[bid][c['id']])
                if not economics['estimated_within_debt']:continue
                add(dict(action='compensate',branch=bid,options=dict(candidate_id=c['id'])),
                    'Reduce error; positive estimated_area_change means PAY more area, negative means save area',
                    predicted_metric=c['predicted_metric'],**economics)
            if full['free_branch_slots']==0 and b['categories']['near']:
                deferred.append(dict(branch=bid,kind='near_repair',requires='archive one active branch first',
                    opportunities=[dict(expected_area_saving=byid[cid]['expected_area_gain'],
                        predicted_metric=byid[cid]['predicted_metric']) for cid in b['categories']['near'][:2]]))
            for cid in b['categories']['near'][:2]:
                if any(o.get('parent')==bid and o.get('candidate_id')==cid for o in self.origin.values()):continue
                c=byid[cid]
                add(dict(action='near_repair',branch=bid,options=dict(candidate_id=cid,steps=3,repair_trials=3)),
                    'Fork this near miss, screen up to3 economical repairs, then up to3 compression steps',
                    expected_area_gain=c['expected_area_gain'],predicted_metric=c['predicted_metric'],excess_fraction=c['excess_fraction'])
            if b['preserve_view'] and b['mapped_area_headroom']>0:
                add(dict(action='remap',branch=bid,options=dict(preserve=False)),
                    'Adopt the exact mapped representation',mapped_area_headroom=b['mapped_area_headroom'])
            for c in b['checkpoints']:
                if 'early_structural_milestone' not in c['categories']:continue
                name=c['id'].split(':',1)[1]
                if structural_id(load(worker.folder/name,self.lib))==b['graph_hash']:continue
                if any(o.get('parent')==bid and o.get('checkpoint')==name for o in self.origin.values()):continue
                add(dict(action='fork',branch=bid,options=dict(checkpoint_id=c['id'],preserve=False)),
                    'Fork an earlier structural milestone; preserve its signals before trying a different repair path',
                    checkpoint_area=c['area'],checkpoint_metric=c['metric'])
            if len(full['branches'])>1:
                add(dict(action='archive',branch=bid,options={}),
                    'Free an active branch slot; its physical checkpoints remain stored')
        add(dict(action='finish',options={}), 'Stop after assessing remaining exploration opportunities')
        self.bindings=bindings
        compact=dict(session=full['session'],decision_id=full['decision_id'],design=full['design'],
            metric=full['metric'],formal_bound=full['formal_bound'],remaining_seconds=full['remaining_seconds'],
            free_branch_slots=full['free_branch_slots'],knowledge=full['knowledge'],decision_menu=menu,
            deferred_frontier=deferred,
            branches=[{k:b[k] for k in ('id','working_area','best_mapped_checkpoint_area','training_metric',
                'steps','stalls','preserve_view','unprobed','compression_screen_exhausted')} for b in full['branches']],
            recent_results=[dict(action=d.get('action'),branch=d.get('branch'),reason=d.get('reason'),
                result=d.get('result')) for d in self.decisions[-3:]])
        save(self.folder/'canonical_observation.json',full);save(self.folder/'observation.json',compact)
        directory=self.folder/'menu_bindings';directory.mkdir(exist_ok=True)
        save(directory/('%04d.json'%len(self.decisions)),dict(bindings=bindings,observation=compact))
        return compact

    def execute(self,d,obs):
        canonical=bind_choice(d,self.bindings)
        branch=canonical.get('branch');worker=self.workers.get(branch)
        graph=structural_id(worker.current) if worker else 'global'
        self.used.add(action_key(canonical,graph));d.update(canonical)
        return super().execute(canonical,self.canonical)


if __name__=='__main__':
    p=parser();p.description=__doc__;p.set_defaults(controller='agent',decisions=128)
    a=p.parse_args()
    if a.controller!='agent':p.error('This entry requires actual model decisions')
    from hybrid_secrets import load_key
    load_key()
    raise SystemExit(MenuPortfolio(a).run())
