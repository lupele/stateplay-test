import json
import sqlite3
import concurrent.futures
import httpx
import pytest
from rl_posttrain.rewards.base import parse_scores,SLOTS
from rl_posttrain.rewards.protocol import payload,group_content
from rl_posttrain.rewards.api_group import RequestStore,APIGroupScorer
from rl_posttrain.io import read_json

@pytest.mark.parametrize("invalid",['{"video_1":true,"video_2":1,"video_3":2,"video_4":3}','{"video_1":NaN,"video_2":1,"video_3":2,"video_4":3}','{"video_1":101,"video_2":1,"video_3":2,"video_4":3}','{"video_1":1}', '```json\n{"video_1":1,"video_2":1,"video_3":2,"video_4":3}\n```','{"video_1":1,"video_1":2,"video_2":1,"video_3":2,"video_4":3}'])
def test_strict_parser(invalid):
    with pytest.raises((ValueError,TypeError)):parse_scores(invalid)

def test_one_request_full_frames_and_mapping(group,config):
    p,key,m=payload(group,config);images=[x for x in p["messages"][0]["content"] if x["type"]=="image_url"]
    assert len(images)==124;assert "PRIVATE_LABEL_SENTINEL" not in json.dumps(p)
    assert list(m)==list(SLOTS)
    parsed=parse_scores('{"video_4":40,"video_2":20,"video_1":10,"video_3":30}')
    assert list(parsed.values())==[10,20,30,40]
    _,mapping=group_content(group,config,["video_4","video_2","video_1","video_3"])
    assert mapping["video_1"]==m["video_4"]

def test_atomic_32_claims_only_one_owner(tmp_path):
    store=RequestStore(tmp_path/"cache.db")
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        rows=list(pool.map(lambda _:store.claim("key","group",1),range(32)))
    assert sum(row[0]=="claimed" for row in rows)==1
    assert sum(row[0]=="inflight" for row in rows)==31

def test_cache_success_no_repeat(group,config,tmp_path):
    calls=[]
    def handle(request):
        calls.append(request);return httpx.Response(200,json={"choices":[{"message":{"content":json.dumps(dict(zip(SLOTS,[10,20,30,40])))}}]})
    scorer=APIGroupScorer(config,tmp_path/"run",1,httpx.MockTransport(handle))
    assert scorer.score(group)==scorer.score(group);assert len(calls)==1

def test_401_fatal_and_response_retained(group,config,tmp_path):
    calls=[]
    def handle(req):calls.append(1);return httpx.Response(401,json={"error":"unauthorized"})
    scorer=APIGroupScorer(config,tmp_path/"run",1,httpx.MockTransport(handle))
    with pytest.raises(RuntimeError):scorer.score(group)
    assert len(calls)==1;assert (group/"teacher/response_raw.json").exists()

def test_unknown_timeout_no_rebilling(group,config,tmp_path):
    calls=[]
    def handle(req):calls.append(1);raise httpx.ReadTimeout("unknown",request=req)
    scorer=APIGroupScorer(config,tmp_path/"run",1,httpx.MockTransport(handle))
    with pytest.raises(RuntimeError):scorer.score(group)
    with pytest.raises(RuntimeError):scorer.score(group)
    assert len(calls)==1;assert (group/"teacher/unknown.json").exists()

def test_429_retry_then_success(group,config,tmp_path):
    calls=[]
    def handle(req):
        calls.append(1)
        if len(calls)==1:return httpx.Response(429,headers={"retry-after":"0"},json={"error":"rate"})
        return httpx.Response(200,json={"choices":[{"message":{"content":json.dumps(dict(zip(SLOTS,[10,20,30,40])))}}]})
    scorer=APIGroupScorer(config,tmp_path/"run",1,httpx.MockTransport(handle));scorer.score(group)
    assert len(calls)==2

@pytest.mark.parametrize('status',[401,413])
def test_html_fatal_response_retained_without_retry(group,config,tmp_path,status):
    calls=[]
    def handle(req):
        calls.append(1)
        return httpx.Response(status,text='<html>gateway rejection</html>',headers={'content-type':'text/html'})
    scorer=APIGroupScorer(config,tmp_path/'run',1,httpx.MockTransport(handle))
    with pytest.raises(RuntimeError,match=f'HTTP {status}'):scorer.score(group)
    assert len(calls)==1
    report=read_json(group/'teacher/response_raw.json')
    assert report['http_status']==status and not report['response_is_json']
    assert report['response_text']=='<html>gateway rejection</html>'
    with scorer.store.connect() as db:
        assert db.execute('select status,error,attempts from requests').fetchone()==('failed',f'fatal_http_{status}',1)

def test_html_gateway_retry_has_durable_status(group,config,tmp_path):
    calls=[]
    def handle(req):
        calls.append(1)
        if len(calls)==1:return httpx.Response(502,text='<html>bad gateway</html>',headers={'retry-after':'0'})
        return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(dict(zip(SLOTS,[10,20,30,40])))}}]})
    scorer=APIGroupScorer(config,tmp_path/'run',1,httpx.MockTransport(handle));scorer.score(group)
    assert len(calls)==2
    assert read_json(group/'teacher/response_attempt_01.json')['http_status']==502
    assert read_json(group/'teacher/retry_attempt_01.json')['reason']=='retryable_http_502'

def test_read_error_is_unknown_not_resubmitted(group,config,tmp_path):
    calls=[]
    def handle(req):calls.append(1);raise httpx.ReadError('stream interrupted',request=req)
    scorer=APIGroupScorer(config,tmp_path/'run',1,httpx.MockTransport(handle))
    with pytest.raises(RuntimeError,match='status unknown'):scorer.score(group)
    with pytest.raises(RuntimeError,match='reconciliation'):scorer.score(group)
    assert len(calls)==1

def test_connection_failures_leave_bounded_attempt_reports(group,config,tmp_path,monkeypatch):
    monkeypatch.setattr('rl_posttrain.rewards.api_group.time.sleep',lambda _:None)
    calls=[]
    def handle(req):calls.append(1);raise httpx.ConnectError('unavailable',request=req)
    scorer=APIGroupScorer(config,tmp_path/'run',1,httpx.MockTransport(handle))
    with pytest.raises(RuntimeError,match='exhausted'):scorer.score(group)
    assert len(calls)==config['reward']['max_retries']+1
    report=read_json(group/f"teacher/retry_attempt_{len(calls):02d}.json")
    assert report['reason']=='ConnectError' and report['retry_exhausted']

def test_formal_report_accepts_retained_html_gateway_responses(group,tmp_path):
    from rl_posttrain.formal_report import build
    from rl_posttrain.io import write_json
    run=tmp_path/'run';(run/'groups').mkdir(parents=True)
    (run/'groups'/group.name).symlink_to(group,target_is_directory=True)
    write_json(run/'plan/cursor.json',{'cursor':0,'api_consumed':0,'local_consumed':0,'inflight':None})
    write_json(run/'plan/epoch_order.json',{'order':['0']})
    write_json(run/'plan/stage_state.json',{'stage':'paused_API_reconciliation'})
    write_json(group/'teacher/response_attempt_01.json',{'http_status':504,'response':None,'response_text':'gateway timeout'})
    report=build(run)
    assert report['saved_provider_responses']==1
    assert report['reported_provider_usage']=={} and not report['formal_completed']

def test_explicit_retry_keeps_history_and_scores_once(group,config,tmp_path):
    from rl_posttrain.io import write_json
    calls=[]
    def handle(req):
        calls.append(1)
        return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(dict(zip(SLOTS,[10,20,30,40])))}}]})
    scorer=APIGroupScorer(config,tmp_path/'run',1,httpx.MockTransport(handle))
    _,key,_=payload(group,config)
    scorer.store.claim(key,group.name,1)
    for _ in range(6):scorer.store.attempt(key)
    scorer.store.finish(key,'unknown',error='user_pause')
    write_json(group/'teacher/response_attempt_01.json',{'historical':True})
    scorer.store.authorize_retry(key,1,'User requested one full-group availability probe')
    assert scorer.score(group)==scorer.score(group)
    assert len(calls)==1
    assert read_json(group/'teacher/response_attempt_01.json')=={'historical':True}
    assert read_json(group/'teacher/response_attempt_07.json')['attempt']==7
    with scorer.store.connect() as db:
        assert db.execute('select status,attempts from requests').fetchone()==('succeeded',7)
        assert db.execute('select baseline_attempts,attempt_limit from retry_authorizations').fetchone()==(6,7)
    with pytest.raises(ValueError):scorer.store.authorize_retry(key,1,'Do not rebill cached success')

def test_single_probe_authorization_does_not_enable_configured_retries(group,config,tmp_path):
    calls=[]
    def handle(req):calls.append(1);raise httpx.ConnectError('offline',request=req)
    scorer=APIGroupScorer(config,tmp_path/'run',1,httpx.MockTransport(handle))
    _,key,_=payload(group,config);scorer.store.claim(key,group.name,1)
    for _ in range(6):scorer.store.attempt(key)
    scorer.store.finish(key,'failed',error='exhausted')
    scorer.store.authorize_retry(key,1,'Single probe only')
    with pytest.raises(RuntimeError,match='exhausted'):scorer.score(group)
    with pytest.raises(RuntimeError,match='Cached failed'):scorer.score(group)
    assert len(calls)==1
    assert read_json(group/'teacher/retry_attempt_07.json')['retry_exhausted']

@pytest.mark.parametrize('status',[504,524])
def test_upstream_timeout_is_unknown_without_rebilling(group,config,tmp_path,status):
    calls=[]
    def handle(req):calls.append(1);return httpx.Response(status,text='Gateway timeout',headers={'cf-ray':'test-ray'})
    scorer=APIGroupScorer(config,tmp_path/'run',1,httpx.MockTransport(handle))
    with pytest.raises(RuntimeError,match='outcome unknown'):scorer.score(group)
    with pytest.raises(RuntimeError,match='reconciliation'):scorer.score(group)
    assert len(calls)==1
    assert read_json(group/'teacher/unknown.json')['reason']==f'upstream_timeout_http_{status}'
    assert read_json(group/'teacher/response_raw.json')['cf_ray']=='test-ray'

def test_stream_scores_and_usage_preserve_canonical_request(group,config,tmp_path,monkeypatch):
    monkeypatch.setenv('STATEPLAY_API_STREAM','1')
    calls=[]
    parts=[{'id':'stream-test','choices':[{'index':0,'delta':{'content':'{"video_1":10,"video_2":20,'},'finish_reason':None}]},
           {'choices':[{'index':0,'delta':{'content':'"video_3":30,"video_4":40}'},'finish_reason':'stop'}]},
           {'choices':[],'usage':{'prompt_tokens':123,'completion_tokens':20,'total_tokens':143}}]
    body=': heartbeat\n\n'+''.join('data: '+json.dumps(p)+'\n\n' for p in parts)+'data: [DONE]\n\n'
    def handle(req):
        wire=json.loads(req.content);calls.append(wire)
        assert wire['stream'] and wire['stream_options']=={'include_usage':True}
        assert len([p for p in wire['messages'][0]['content'] if p['type']=='image_url'])==124
        return httpx.Response(200,text=body,headers={'content-type':'text/event-stream'})
    scorer=APIGroupScorer(config,tmp_path/'run',1,httpx.MockTransport(handle))
    assert scorer.score(group)==dict(zip(SLOTS,[10.,20.,30.,40.]))
    assert len(calls)==1
    assert 'stream' not in read_json(group/'teacher/request_payload.json')
    report=read_json(group/'teacher/response_raw.json')
    assert report['assembled_from_stream'] and report['response']['usage']['total_tokens']==143
    assert (group/'teacher/stream_attempt_01.jsonl').exists()

def test_partial_stream_is_retained_and_never_retried(group,config,tmp_path,monkeypatch):
    monkeypatch.setenv('STATEPLAY_API_STREAM','1')
    calls=[]
    def handle(req):
        calls.append(1)
        return httpx.Response(200,text='data: {"choices":[{"delta":{"content":"partial"}}]}\n\n',headers={'content-type':'text/event-stream'})
    scorer=APIGroupScorer(config,tmp_path/'run',1,httpx.MockTransport(handle))
    with pytest.raises(RuntimeError,match='status unknown'):scorer.score(group)
    with pytest.raises(RuntimeError,match='reconciliation'):scorer.score(group)
    assert len(calls)==1
    assert 'partial' in (group/'teacher/stream_attempt_01.jsonl').read_text()
