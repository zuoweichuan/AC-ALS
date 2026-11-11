"""Process-local model adapter for independent repeated Agent experiments.

Search behavior is retained. Only this entry opts into the authorized30CNY cap.
No fallback search.
"""
import json
import sys
from pathlib import Path

DEEPSEEK = 'deepseek-ai/DeepSeek-V4-Flash'
GLM = 'zai-org/GLM-5.3-Flash'
AUTHORIZED_CAP = 30.


def deepseek_choice(observation, remaining, cap=AUTHORIZED_CAP, *, system=None, max_tokens=1024):
    # Keep the original DeepSeek request settings (500 output tokens, no thinking).
    from batch_provider import choose
    return choose(observation, remaining, cap, system=system, authorized_cap=AUTHORIZED_CAP)


def glm_choice(observation, remaining, cap=AUTHORIZED_CAP, *, system=None, max_tokens=1024):
    from comparison_model_provider import request_once
    return request_once(observation, remaining, cap, system=system, max_tokens=1024,
                        authorized_cap=AUTHORIZED_CAP, request_timeout=35.)


def run():
    kind = sys.argv.pop(1)
    if kind not in ('er', 'nmed'):
        raise ValueError('Expected er or nmed')
    from hybrid_secrets import load_key
    load_key()
    if kind == 'er':
        import er_experience_search as entry
        parser = entry.parser()
        parser.add_argument('--model', default=DEEPSEEK)
        args = parser.parse_args()
        if args.model not in (DEEPSEEK, GLM) or args.controller != 'agent':
            raise ValueError('Choose a declared Agent model')
        import er_experience_provider
        er_experience_provider.request = deepseek_choice if args.model == DEEPSEEK else glm_choice
        code = entry.run(args, authorized_api_cap=AUTHORIZED_CAP)
    else:
        import numeric_latest_agent as entry
        import comparison_model_provider
        args = entry.parser().parse_args()
        if args.model not in (DEEPSEEK, GLM) or args.controller != 'agent' or args.metric != 'nmed':
            raise ValueError('Choose a declared NMED Agent model')
        # LatestNumericAgent imports choose at run time; this only binds the
        # requested provider in this child process, not any on-disk Agent code.
        comparison_model_provider.choose = deepseek_choice if args.model == DEEPSEEK else glm_choice
        code = entry.LatestNumericAgent(args).run()
    path = Path(args.output) / 'report.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    report.update(agent_model=args.model, repeated_experiment=True,
                  api_cap_cny=args.api_cap_cny,
                  worker_kind=kind, model_comparison=False,
                  provider_settings=('temperature=.1, max_tokens=500, thinking off' if args.model == DEEPSEEK else
                                     'temperature=.1, max_tokens=1024 including reasoning, reasoning_effort=low; no retry;35s HTTP'))
    from numeric_suite import save
    save(path, report)
    protocol_path = path.parent / 'protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    protocol.update(api_cap_cny=args.api_cap_cny,
                    budget_authorization='User2026-11-01: cumulative30CNY per circuit for five independent repeats; original ledger retained')
    save(protocol_path, protocol)
    print('[REPEAT AGENT RESULT]', json.dumps({k: report.get(k) for k in (
        'design', 'area', 'strict_pass', 'agent_model', 'api_responses',
        'valid_model_decisions', 'executed_agent_tools', 'model_control_verified')}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(run())
