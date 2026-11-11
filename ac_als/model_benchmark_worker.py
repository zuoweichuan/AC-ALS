"""Model comparison worker: ER existing Agent, NMED latest-tool Agent migration."""
import json,sys
from pathlib import Path
from numeric_agentic import sha
from bacs_numeric import save


def run():
    if len(sys.argv)<2 or sys.argv[1] not in ('er','nmed'):raise ValueError('Choose ER or NMED worker')
    kind=sys.argv.pop(1)
    from comparison_model_provider import MODEL
    if kind=='er':
        import er_experience_search as entry
        import er_experience_provider
        from comparison_model_provider import choose
        er_experience_provider.request=choose
        p=entry.parser();p.add_argument('--model',default=MODEL);a=p.parse_args()
        if a.model!=MODEL or a.controller!='agent':raise ValueError('This experiment is GLM Agent only')
        code=entry.run(a)
    else:
        import numeric_latest_agent as entry
        a=entry.parser().parse_args()
        if a.model!=MODEL or a.controller!='agent' or a.metric!='nmed':raise ValueError('Use declared GLM NMED Agent')
        from hybrid_secrets import load_key
        load_key();code=entry.LatestNumericAgent(a).run()
    path=Path(a.output)/'report.json';r=json.loads(path.read_text(encoding='utf-8'))
    r.update(agent_model=MODEL,model_comparison=True,worker_kind=kind,
        latest_agent_framework=('er_experience_search.ExperienceAgent' if kind=='er' else 'numeric_latest_agent.LatestNumericAgent'))
    protocol_path=path.parent/'protocol.json';protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    protocol.update(agent_model=MODEL,model_comparison=True)
    for name in ('model_benchmark_worker.py','comparison_model_provider.py'):
        source=Path(__file__).resolve().parent/name
        import shutil
        shutil.copy2(source,path.parent/'source'/name);protocol['code_sha256'][name]=sha(source)
    save(protocol_path,protocol);save(path,r)
    print('[MODEL VERIFIED RESULT]',json.dumps({k:r.get(k) for k in ('design','bound','agent_model','area','strict_pass','api_responses','model_control_verified')},ensure_ascii=False),flush=True)
    return code


if __name__=='__main__':raise SystemExit(run())
