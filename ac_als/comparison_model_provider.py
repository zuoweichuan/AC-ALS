"""Explicit model selection; same project ledger, no provider/model fallback."""
import hashlib,json,time,urllib.error,urllib.request
from pathlib import Path
import os
from hybrid_agent import lease,read,save
from hybrid_provider import NoRedirect
from hybrid_secrets import load_key

ROOT=Path(os.environ.get('NETLIST_FRONTIER_ROOT',str(Path(__file__).resolve().parent)))
MODEL='zai-org/GLM-5.3-Flash'
PRICE=dict(input_cny_per_million=.8,output_cny_per_million=2.8,
    cache_hit_cny_per_million=.23,verified_date='2026-10-29',
    source='https://www.siliconflow.cn/news/ok5ja7brf921kgf1644yyedz',
    accounting='upper estimate using all input at uncached rate; actual cached bill can be lower')


class TransportFailure(RuntimeError):
    def __init__(self,message,seconds):super().__init__(message);self.request_seconds=seconds


def request_once(observation,remaining,cap=10.,*,system=None,max_tokens=1024,authorized_cap=10.,request_timeout=90.):
    if authorized_cap not in (10.,30.) or not 0<cap<=authorized_cap:raise ValueError('Cap exceeds explicitly authorized cumulative limit')
    if not 0<request_timeout<=90:raise ValueError('Invalid HTTP timeout')
    if not system:raise ValueError('Explicit circuit-tool instructions required')
    if not 32<=max_tokens<=4096:raise ValueError('Bounded output tokens required')
    payload=dict(model=MODEL,temperature=.1,max_tokens=max_tokens,reasoning_effort='low',
        stream=False,response_format={'type':'json_object'},messages=[dict(role='system',content=system),
            dict(role='user',content=json.dumps(observation,ensure_ascii=False))])
    encoded=json.dumps(payload,ensure_ascii=False).encode()
    if len(encoded)>48000:raise ValueError('Model context exceeds48kB; no API request started')
    if remaining<=1:raise RuntimeError('No request time remaining')
    secret=load_key();folder=Path(observation['session']);audit=folder/'model_inputs';audit.mkdir(exist_ok=True)
    digest=hashlib.sha256(encoded).hexdigest();save(audit/(digest+'.json'),dict(payload=payload,sha256=digest,pricing=PRICE))
    billing=ROOT/'hybrid_billing'/observation['design'];billing.mkdir(parents=True,exist_ok=True)
    ledger_path=billing/'ledger.json'
    reserve=((len(encoded)+1000)*PRICE['input_cny_per_million']+payload['max_tokens']*PRICE['output_cny_per_million'])/1e6
    with lease(billing):
        ledger=read(ledger_path) if ledger_path.exists() else dict(cap_cny=cap,spent_upper_cny=0.,requests=[])
        # Never create a separate model ledger to evade the original per-circuit cap.
        if ledger.get('cap_cny',cap)<cap:
            ledger.setdefault('cap_history',[]).append(dict(previous=ledger['cap_cny'],cap=cap,
                reason=('User2026-11-01 authorized cumulative30yuan per circuit; repeated model comparison'
                        if cap>10 else 'Prior explicit human authorization: cumulative10yuan per circuit')))
            ledger['cap_cny']=cap
        if ledger['spent_upper_cny']+reserve>min(cap,ledger.get('cap_cny',cap)):
            raise RuntimeError('Cumulative circuit API cap reached; no fallback')
        index=len(ledger['requests']);ledger['spent_upper_cny']+=reserve
        ledger['requests'].append(dict(session=str(folder),decision_id=observation['decision_id'],model=MODEL,
            reserved_cny=reserve,status='reserved_or_uncertain',pricing=PRICE));save(ledger_path,ledger)
    req=urllib.request.Request('https://api.siliconflow.cn/v1/chat/completions',encoded,
        headers={'Authorization':'Bearer '+secret,'Content-Type':'application/json'})
    tick=time.perf_counter();response_model=None;usage={};body=b''
    try:
        with urllib.request.build_opener(NoRedirect()).open(req,timeout=min(request_timeout,max(1.,remaining))) as response:
            body=response.read(1048577)
        http_seconds=time.perf_counter()-tick
        if len(body)>1048576:raise ValueError('Oversized response')
        result=json.loads(body);usage=result.get('usage') or {};response_model=result.get('model')
        if response_model and response_model not in (MODEL,MODEL.split('/')[-1]):
            raise ValueError('Provider response names a different model')
        # Bill valid responses even when decision JSON is invalid/truncated.
        with lease(billing):
            ledger=read(ledger_path);entry=ledger['requests'][index]
            spent=(usage.get('prompt_tokens',0)*PRICE['input_cny_per_million']+
                   usage.get('completion_tokens',0)*PRICE['output_cny_per_million'])/1e6
            if all(k in usage for k in ('prompt_tokens','completion_tokens')):
                ledger['spent_upper_cny']+=spent-reserve;entry['estimated_upper_cny']=spent
            entry.update(status='response_received',response_model=response_model,usage=usage,
                response_id=result.get('id'),http_request_seconds=http_seconds);save(ledger_path,ledger)
        message=result['choices'][0]['message'];content=message.get('content')
        decision=json.loads(content)
        if not isinstance(decision,dict):raise ValueError('Expected decision object')
    except urllib.error.HTTPError as exc:
        if exc.code in (429,500,502,503,504):
            raise TransportFailure('GLM provider HTTP%d'%exc.code,time.perf_counter()-tick) from None
        raise RuntimeError('GLM provider HTTP%d; checkpoint preserved; no model fallback'%exc.code) from None
    except OSError as exc:
        raise TransportFailure('GLM '+type(exc).__name__+' during HTTP request',time.perf_counter()-tick) from None
    except (ValueError,KeyError,IndexError,TypeError):
        if body:
            # Provider response contains no credential; retain raw output for diagnostics.
            (audit/(digest+'_response.json')).write_bytes(body)
        raise RuntimeError('GLM transport/decision JSON failure; checkpoint preserved; no model fallback') from None
    receipt=dict(model=MODEL,response_model=response_model,response_id=result.get('id'),usage=usage,
        seconds=time.perf_counter()-tick,http_request_seconds=http_seconds,
        estimated_upper_cny=spent if usage else reserve,input_sha256=digest,
        output_token_cap=payload['max_tokens'],reasoning_effort='low',pricing=PRICE)
    return decision,receipt


def choose(observation,remaining,cap=10.,*,system=None,max_tokens=1024):
    """At most one same-model transport retry, preserving the search prefix."""
    tick=time.perf_counter();failures=[]
    for attempt in range(2):
        left=remaining-(time.perf_counter()-tick)
        try:
            decision,receipt=request_once(observation,left,cap,system=system,max_tokens=max_tokens)
            receipt['http_request_seconds']+=sum(f['seconds'] for f in failures)
            receipt['seconds']=time.perf_counter()-tick;receipt['transport_retries']=failures
            return decision,receipt
        except TransportFailure as exc:
            failures.append(dict(reason=str(exc),seconds=exc.request_seconds))
            if attempt or remaining-(time.perf_counter()-tick)<8:
                raise RuntimeError(str(exc)+'; same-model retry exhausted, checkpoint preserved') from None
            print('[GLM SAME-MODEL RETRY]',str(exc),flush=True);time.sleep(1.)
