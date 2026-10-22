import sqlite3
import httpx
import pytest
from rl_posttrain.io import write_json,read_json,teacher_path,api_cache_path
from rl_posttrain.rewards.protocol import payload
from rl_posttrain.rewards.api_group import RequestStore,APIGroupScorer
from rl_posttrain.formal import api_unsafe


def test_new_provider_preserves_immutable_prior_teacher(group,config):
    _,old_key,_=payload(group,config)
    original=(group/'teacher/request_map.json').read_bytes()
    write_json(group/'teacher/active.json',{'relative_path':'versions/new-provider'})
    config['reward']['endpoint']='https://new-provider.invalid/v1'
    _,new_key,_=payload(group,config)
    assert old_key!=new_key
    assert (group/'teacher/request_map.json').read_bytes()==original
    assert read_json(teacher_path(group)/'request_map.json')['endpoint']==config['reward']['endpoint']


def test_old_unknown_requests_retained_and_cached_new_labels_not_rebilled(group,config,tmp_path):
    run=tmp_path/'run';old=RequestStore(run/'requests/cache.sqlite3')
    old.claim('old','historical');old.attempt('old');old.finish('old','unknown',error='historical_timeout')
    write_json(run/'requests/active.json',{'relative_path':'providers/new/cache.sqlite3','endpoint':config['reward']['endpoint']})
    calls=[]
    def handle(req):
        calls.append(1)
        return httpx.Response(200,json={'choices':[{'message':{'content':'{"video_1":1,"video_2":2,"video_3":3,"video_4":4}'}}]})
    scorer=APIGroupScorer(config,run,1,httpx.MockTransport(handle))
    assert scorer.score(group)==scorer.score(group)
    assert len(calls)==1 and not api_unsafe(run)
    with old.connect() as db:assert db.execute('select status,attempts from requests').fetchall()==[('unknown',1)]
    with pytest.raises(ValueError,match='mismatch'):api_cache_path(run,'https://wrong.invalid/v1')


@pytest.mark.parametrize('relative', ['../escape', '/tmp/escape'])
def test_version_pointers_cannot_escape(group,tmp_path,relative):
    write_json(group/'teacher/active.json',{'relative_path':relative})
    with pytest.raises(ValueError):teacher_path(group)
    write_json(tmp_path/'requests/active.json',{'relative_path':relative,'endpoint':'x'})
    with pytest.raises(ValueError):api_cache_path(tmp_path)
