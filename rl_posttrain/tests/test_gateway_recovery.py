import copy
from pathlib import Path
from rl_posttrain.io import write_json,read_json
from rl_posttrain.rewards.api_group import RequestStore
from rl_posttrain.formal import reopen_unscored_gateway_timeouts,run_command,FormalPause
import pytest


def prepared(tmp_path,group,config):
    run=tmp_path/'run';(run/'groups').mkdir(parents=True)
    (run/'groups'/group.name).symlink_to(group,target_is_directory=True)
    c=copy.deepcopy(config);c['reward'].update(max_retries=5,recover_unscored_gateway_timeouts=True)
    write_json(run/'plan/config.json',c)
    write_json(run/'plan/cursor.json',{'cursor':0,'inflight':{'phase':'generated','condition_ids':['0'],'update_cursor':0}})
    store=RequestStore(run/'requests/cache.sqlite3');store.claim('key',group.name);store.attempt('key')
    store.finish('key','unknown',error='upstream_timeout_http_524')
    write_json(group/'teacher/request_map.json',{'cache_key':'key'})
    write_json(group/'teacher/response_attempt_01.json',{'http_status':524})
    return run,c,store


def authorize(run,config):
    write_json(run/'plan/API_gateway_retry_authorization.json',{'authorized':True,'allow_possible_duplicate_charges':True,'max_total_attempts_per_request':6,'endpoint':config['reward']['endpoint'],'model':config['reward']['model']})


def test_explicit_permission_required_without_changing_unknown(tmp_path,group,config):
    run,c,store=prepared(tmp_path,group,config)
    assert not reopen_unscored_gateway_timeouts(run)
    with store.connect() as db:assert db.execute('select status,attempts from requests').fetchone()==('unknown',1)


def test_authorized_gateway_recovery_preserves_cursor_and_caps_total_six(tmp_path,group,config):
    run,c,store=prepared(tmp_path,group,config);authorize(run,c)
    cursor=(run/'plan/cursor.json').read_bytes()
    for attempt in range(1,6):
        write_json(group/f'teacher/response_attempt_{attempt:02d}.json',{'http_status':524})
        assert reopen_unscored_gateway_timeouts(run)
        assert not reopen_unscored_gateway_timeouts(run)
        assert store.remaining_attempts('key',6)==6-attempt
        assert store.claim('key',group.name)[0]=='claimed'
        assert store.attempt('key')==attempt+1
        store.finish('key','unknown',error='upstream_timeout_http_524')
    assert not reopen_unscored_gateway_timeouts(run)
    assert (run/'plan/cursor.json').read_bytes()==cursor
    with store.connect() as db:
        assert db.execute('select attempts,status from requests').fetchone()==(6,'unknown')
        assert db.execute('select max(attempt_limit) from retry_authorizations').fetchone()==(6,)
    assert len(list((run/'plan/API_gateway_recoveries').glob('*.json')))==5


def test_existing_scores_or_unclassified_transport_unknown_never_reopened(tmp_path,group,config):
    run,c,store=prepared(tmp_path,group,config);authorize(run,c)
    store.finish('key','unknown',error='ReadTimeout')
    assert not reopen_unscored_gateway_timeouts(run)
    store.finish('key','unknown',error='upstream_timeout_http_524')
    write_json(group/'teacher/scores.json',{'scores':{'video_1':1}})
    assert not reopen_unscored_gateway_timeouts(run)


def authorize_read_timeout(run,config,group):
    authorize(run,config)
    path=run/'plan/API_gateway_retry_authorization.json';permission=read_json(path)
    permission['authorized_unknown_errors']=['ReadTimeout'];write_json(path,permission)
    write_json(group/'teacher/unknown.json',{'cache_key':'key','reason':'ReadTimeout','requires_reconciliation':True})


def test_read_timeout_separate_permission_preserves_ledger_and_caps_six(tmp_path,group,config):
    run,c,store=prepared(tmp_path,group,config);authorize_read_timeout(run,c,group)
    cursor=(run/'plan/cursor.json').read_bytes()
    for attempt in range(1,6):
        store.finish('key','unknown',error='ReadTimeout')
        assert reopen_unscored_gateway_timeouts(run)
        assert not reopen_unscored_gateway_timeouts(run)
        assert store.claim('key',group.name)[0]=='claimed'
        assert store.attempt('key')==attempt+1
    store.finish('key','unknown',error='ReadTimeout')
    assert not reopen_unscored_gateway_timeouts(run)
    assert (run/'plan/cursor.json').read_bytes()==cursor
    with store.connect() as db:
        assert db.execute('select attempts,status from requests').fetchone()==(6,'unknown')
        assert db.execute('select max(attempt_limit) from retry_authorizations').fetchone()==(6,)


@pytest.mark.parametrize('evidence',[None,{'cache_key':'wrong','reason':'ReadTimeout'},{'cache_key':'key','reason':'WriteTimeout'},{'cache_key':'key','reason':'ReadTimeout','attempt':2}])
def test_read_timeout_requires_matching_durable_evidence(tmp_path,group,config,evidence):
    run,c,store=prepared(tmp_path,group,config);authorize_read_timeout(run,c,group)
    store.finish('key','unknown',error='ReadTimeout');path=group/'teacher/unknown.json'
    if evidence is None:path.unlink()
    else:write_json(path,evidence)
    assert not reopen_unscored_gateway_timeouts(run)


def test_read_timeout_permission_does_not_authorize_other_errors_or_scored_groups(tmp_path,group,config):
    run,c,store=prepared(tmp_path,group,config);authorize_read_timeout(run,c,group)
    store.finish('key','unknown',error='WriteTimeout')
    assert not reopen_unscored_gateway_timeouts(run)
    store.finish('key','unknown',error='ReadTimeout')
    write_json(group/'teacher/scores.json',{'scores':{'video_1':1}})
    assert not reopen_unscored_gateway_timeouts(run)


def test_other_batch_or_started_optimizer_cannot_reopen(tmp_path,group,config):
    run,c,store=prepared(tmp_path,group,config);authorize(run,c)
    ledger=read_json(run/'plan/cursor.json');ledger['inflight']['condition_ids']=['other']
    write_json(run/'plan/cursor.json',ledger);assert not reopen_unscored_gateway_timeouts(run)
    ledger['inflight'].update(condition_ids=['0'],update_cursor=1)
    write_json(run/'plan/cursor.json',ledger);assert not reopen_unscored_gateway_timeouts(run)


def test_controller_recovers_timeout_without_consuming_worker_crash_budget(tmp_path,group,config,monkeypatch):
    run,c,store=prepared(tmp_path,group,config);authorize(run,c)
    monkeypatch.setattr('rl_posttrain.formal.time.sleep',lambda _:None)
    calls=[]
    def execute(*args,**kwargs):
        calls.append(1)
        if len(calls)==1:return 1
        assert store.claim('key',group.name)[0]=='claimed'
        assert store.attempt('key')==2
        store.finish('key','succeeded',result={'video_1':1,'video_2':2,'video_3':3,'video_4':4})
        return 0
    run_command(['world'],run,'formal_A',list(range(7)),execute)
    assert len(calls)==2
    recovery=read_json(run/'plan/recoveries/formal_A.json')
    assert recovery['failed_launches']==0 and recovery['gateway_timeout_launches']==1
    assert recovery['total_failed_launches']==1


def test_controller_gateway_failures_stop_at_cumulative_request_ceiling(tmp_path,group,config,monkeypatch):
    run,c,store=prepared(tmp_path,group,config);authorize(run,c)
    monkeypatch.setattr('rl_posttrain.formal.time.sleep',lambda _:None)
    calls=[]
    def execute(*args,**kwargs):
        calls.append(1)
        if len(calls)>1:
            assert store.claim('key',group.name)[0]=='claimed'
            attempt=store.attempt('key')
            store.finish('key','unknown',error='upstream_timeout_http_524')
            write_json(group/f'teacher/response_attempt_{attempt:02d}.json',{'http_status':524})
        return 1
    with pytest.raises(FormalPause):run_command(['world'],run,'formal_A',list(range(7)),execute)
    assert len(calls)==6
    with store.connect() as db:assert db.execute('select status,attempts from requests').fetchone()==('unknown',6)


def test_reward_checkpoint_progress_refreshes_consecutive_crash_budget(tmp_path,monkeypatch):
    run=tmp_path/'run';write_json(run/'plan/cursor.json',{'cursor':2000})
    monkeypatch.setattr('rl_posttrain.formal.time.sleep',lambda _:None)
    calls=[]
    def execute(*args,**kwargs):
        calls.append(1)
        if len(calls)==5:return 0
        write_json(run/'reward_model/checkpoints/latest.json',{'checkpoint':'step-'+str(len(calls)),'step':len(calls)})
        return 1
    run_command(['python','-m','rl_posttrain.reward_model.train'],run,'B',list(range(7)),execute)
    assert len(calls)==5
    report=read_json(run/'plan/recoveries/B.json')
    assert report['total_failed_launches']==4 and report['failed_launches']==1 and report['last_reward_step']==4


def test_reward_without_checkpoint_progress_keeps_three_failure_ceiling(tmp_path,monkeypatch):
    run=tmp_path/'run';write_json(run/'plan/cursor.json',{'cursor':2000})
    monkeypatch.setattr('rl_posttrain.formal.time.sleep',lambda _:None)
    calls=[]
    def execute(*args,**kwargs):calls.append(1);return 1
    with pytest.raises(FormalPause):run_command(['python','-m','rl_posttrain.reward_model.train'],run,'B',list(range(7)),execute)
    assert len(calls)==3
