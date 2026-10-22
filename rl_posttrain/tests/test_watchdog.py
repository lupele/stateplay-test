import json
import sys
import pytest
from rl_posttrain.formal import FormalPause, supervise
from rl_posttrain.io import read_json,write_json
from rl_posttrain.watchdog import monitor,record_exit,restart_decision


def setup_run(tmp_path,stage='formal_A'):
    write_json(tmp_path/'plan/cursor.json',{'cursor':224})
    write_json(tmp_path/'plan/stage_state.json',{'stage':stage,'process_state':'running'})
    return tmp_path


def test_no_restart_through_API_gate(tmp_path,monkeypatch):
    run=setup_run(tmp_path,'paused_API_reconciliation')
    monkeypatch.setattr('rl_posttrain.watchdog.api_unsafe',lambda _:True)
    assert restart_decision(run,{})[0]=='awaiting_API_reconciliation'
    monkeypatch.setattr('rl_posttrain.watchdog.api_unsafe',lambda _:False)
    assert restart_decision(run,{})[0]=='start_controller'


def test_quality_and_human_gates_never_restarted(tmp_path,monkeypatch):
    monkeypatch.setattr('rl_posttrain.watchdog.api_unsafe',lambda _:False)
    for stage in ('paused_reward_quality','awaiting_C_initial_review','paused_recovery_limit','paused_preflight','paused_after_B_user_request'):
        setup_run(tmp_path,stage)
        assert restart_decision(tmp_path,{})[0]=='awaiting_existing_gate'


def test_crash_budget_persistent_and_reset_only_on_progress(tmp_path,monkeypatch):
    run=setup_run(tmp_path,'paused_controller_error')
    monkeypatch.setattr('rl_posttrain.watchdog.api_unsafe',lambda _:False)
    record={}
    for failure in range(1,4):
        record=record_exit(record,-9,224)
        write_json(run/'plan/watchdog_budget.json',record)
        record=read_json(run/'plan/watchdog_budget.json')
        assert record['failures_at_cursor']==failure
    assert restart_decision(run,record)[0]=='blocked_controller_crash_limit'
    write_json(run/'plan/cursor.json',{'cursor':252})
    assert restart_decision(run,record)==('start_controller',0)


def test_intentional_pause_does_not_consume_crash_budget():
    record={'failure_cursor':224,'failures_at_cursor':1}
    assert record_exit(record,3,224)['failures_at_cursor']==1
    assert record_exit(record,0,224)['failures_at_cursor']==1


def test_real_child_crash_restarts_once_without_API(tmp_path):
    run=setup_run(tmp_path);(run/'logs').mkdir()
    counter=run/'child_launches.txt'
    program=("from pathlib import Path; import sys; p=Path(sys.argv[1]); "
             "n=int(p.read_text())+1 if p.exists() else 1; p.write_text(str(n)); "
             "sys.exit(2 if n==1 else 0)")
    monitor(run,[sys.executable,'-c',program,str(counter)],poll_seconds=.02)
    assert counter.read_text()=='2'
    heartbeat=read_json(run/'plan/watchdog_state.json')
    assert heartbeat['state']=='completed' and not heartbeat['controller_alive']
    assert heartbeat['API_requests_made_by_watchdog']==0
    assert read_json(run/'plan/watchdog_budget.json')['failures_at_cursor']==1
    events=[json.loads(line) for line in (run/'plan/watchdog_events.jsonl').read_text().splitlines()]
    assert [e['event'] for e in events]==['controller_started','controller_exited','controller_started','controller_exited']


@pytest.mark.parametrize('outcome',['pause','error','success'])
def test_controller_exit_always_clears_false_running(tmp_path,monkeypatch,outcome):
    run=setup_run(tmp_path)
    monkeypatch.setattr('rl_posttrain.formal.wait_owned_workers',lambda _:None)
    def plan(*args):
        if outcome=='pause':raise FormalPause('test gate')
        if outcome=='error':raise ValueError('test crash')
    monkeypatch.setattr('rl_posttrain.formal.run_plan',plan)
    def call():supervise({},run,'config',lambda *a,**k:0,'python','python')
    if outcome=='pause':
        with pytest.raises(SystemExit) as error:call()
        assert error.value.code==3
    elif outcome=='error':
        with pytest.raises(ValueError):call()
    else:call()
    state=read_json(run/'plan/stage_state.json')
    assert state['process_state']=='stopped'
    assert state['formal_conditions_consumed']==224
    assert not read_json(run/'plan/controller_process.json')['active']
