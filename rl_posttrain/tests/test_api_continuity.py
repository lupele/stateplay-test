"""Authorized API failures recover inside scoring without restarting CUDA jobs."""
import copy
import json
import httpx
import pytest
from rl_posttrain.io import write_json,read_json
from rl_posttrain.formal import reopen_unscored_gateway_timeouts
from rl_posttrain.rewards import api_group
from rl_posttrain.rewards.transport import SubmissionIncomplete


def prepare(run,group,config,all_errors=True,ceiling=6):
    c=copy.deepcopy(config)
    c['reward'].update(max_retries=5,recover_unscored_gateway_timeouts=True,concurrency_backend='threads',workers=2)
    (run/'groups').mkdir(parents=True)
    (run/'groups'/group.name).symlink_to(group,target_is_directory=True)
    write_json(run/'plan/config.json',c)
    write_json(run/'plan/cursor.json',{'cursor':0,'inflight':{'phase':'generated','condition_ids':['0'],'update_cursor':0}})
    write_json(run/'plan/API_gateway_retry_authorization.json',{'authorized':True,'allow_possible_duplicate_charges':True,
               'retry_all_api_errors':all_errors,'max_total_attempts_per_request':ceiling,
               'endpoint':c['reward']['endpoint'],'model':c['reward']['model']})
    return c


def success():
    return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps({'video_1':10,'video_2':20,'video_3':30,'video_4':40})}}]})


@pytest.mark.parametrize('fault',['stream','read','write','remote','read_error','write_error','fatal_http','missing_scores','decoding','unexpected'])
def test_failure_then_valid_scores_without_world_restart(tmp_path,group,config,monkeypatch,fault):
    run=tmp_path/'run';c=prepare(run,group,config);calls=[]
    def handler(request):
        calls.append(1)
        if len(calls)>1:return success()
        if fault=='stream':raise SubmissionIncomplete('upstream stream failed')
        if fault=='unexpected':raise RuntimeError('previously unseen provider error')
        if fault=='fatal_http':return httpx.Response(401,json={'error':'provider rejection'})
        if fault=='missing_scores':return httpx.Response(200,json={'choices':[{'message':{'content':'{"video_1":10}'}}]})
        error={'read':httpx.ReadTimeout,'write':httpx.WriteTimeout,'remote':httpx.RemoteProtocolError,
               'read_error':httpx.ReadError,'write_error':httpx.WriteError,'decoding':httpx.DecodingError}[fault]
        raise error('transport failure',request=request)
    transport=httpx.MockTransport(handler)
    def score(args):
        values,path,g,limit=args
        return api_group.APIGroupScorer(values,run,limit,transport).score(group)
    monkeypatch.setattr(api_group,'_process_score',score)
    monkeypatch.setattr(api_group.time,'sleep',lambda _:None)
    monkeypatch.setenv('STATEPLAY_API_STREAM','0')
    cursor=(run/'plan/cursor.json').read_bytes()
    result=api_group.score_groups(c,run,[group])
    assert result[0]['video_4']==40 and len(calls)==2
    assert api_group.score_groups(c,run,[group])==result and len(calls)==2
    (group/'teacher/scores.json').unlink()
    assert api_group.score_groups(c,run,[group])==result and len(calls)==2
    assert read_json(group/'teacher/scores.json')['recovered_from_successful_cache']
    assert (run/'plan/cursor.json').read_bytes()==cursor
    store=api_group.RequestStore(run/'requests/cache.sqlite3')
    with store.connect() as db:assert db.execute('select status,attempts from requests').fetchone()==('succeeded',2)


@pytest.mark.parametrize('ceiling',[6,30])
def test_repeated_incomplete_reply_stops_at_total_ceiling_across_relaunches(tmp_path,group,config,monkeypatch,ceiling):
    run=tmp_path/'run';c=prepare(run,group,config,ceiling=ceiling);calls=[]
    def handler(request):
        calls.append(1);raise SubmissionIncomplete('interrupted')
    transport=httpx.MockTransport(handler)
    monkeypatch.setattr(api_group,'_process_score',lambda args:api_group.APIGroupScorer(c,run,None,transport).score(group))
    monkeypatch.setattr(api_group.time,'sleep',lambda _:None)
    monkeypatch.setenv('STATEPLAY_API_STREAM','0')
    with pytest.raises(RuntimeError):api_group.score_groups(c,run,[group])
    assert len(calls)==ceiling
    assert not reopen_unscored_gateway_timeouts(run)
    with pytest.raises(RuntimeError):api_group.score_groups(c,run,[group])
    assert len(calls)==ceiling
    assert not (group/'teacher/scores.json').exists()
    store=api_group.RequestStore(run/'requests/cache.sqlite3')
    with store.connect() as db:assert db.execute('select status,attempts from requests').fetchone()==('unknown',ceiling)


def test_operational_increase_recovers_after_six_without_mutating_training_config(tmp_path,group,config,monkeypatch):
    run=tmp_path/'run';c=prepare(run,group,config,ceiling=30);calls=[]
    original=(run/'plan/config.json').read_bytes()
    def handler(request):
        calls.append(1)
        if len(calls)<=7:raise SubmissionIncomplete('temporary outage')
        return success()
    transport=httpx.MockTransport(handler)
    monkeypatch.setattr(api_group,'_process_score',lambda args:api_group.APIGroupScorer(c,run,None,transport).score(group))
    monkeypatch.setattr(api_group.time,'sleep',lambda _:None)
    monkeypatch.setenv('STATEPLAY_API_STREAM','0')
    assert api_group.score_groups(c,run,[group])[0]['video_4']==40
    assert len(calls)==8
    assert (run/'plan/config.json').read_bytes()==original


def test_legacy_permission_does_not_silently_expand(tmp_path,group,config,monkeypatch):
    run=tmp_path/'run';c=prepare(run,group,config,all_errors=False);calls=[]
    def handler(request):calls.append(1);raise SubmissionIncomplete('interrupted')
    transport=httpx.MockTransport(handler)
    monkeypatch.setattr(api_group,'_process_score',lambda args:api_group.APIGroupScorer(c,run,None,transport).score(group))
    monkeypatch.setenv('STATEPLAY_API_STREAM','0')
    with pytest.raises(RuntimeError):api_group.score_groups(c,run,[group])
    assert len(calls)==1
