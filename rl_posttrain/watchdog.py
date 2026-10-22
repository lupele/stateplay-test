"""Persistent controller supervision; API reconciliation and quality gates stay intact.

Credentials are inherited in memory. No paid request is made by this module.
An unexpected controller crash may restart at most three times at one cursor;
that budget survives daemon restarts. An intentional gate never counts as a crash.
"""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from .formal import api_unsafe, owned_alive
from .io import ROOT, append_jsonl, now, read_json, run_path, write_json


def snapshot(run):
    ledger=read_json(run/'plan/cursor.json')
    stage=read_json(run/'plan/stage_state.json')
    return ledger,stage


def restart_decision(run,record):
    """Pure decision: never reopen a request, reset a ledger, or clear a gate."""
    ledger,stage=snapshot(run)
    cursor=ledger['cursor']
    failures=record.get('failures_at_cursor',0) if record.get('failure_cursor')==cursor else 0
    if failures>=3:return 'blocked_controller_crash_limit',failures
    if api_unsafe(run):return 'awaiting_API_reconciliation',failures
    name=stage.get('stage','')
    if name.startswith(('paused_','awaiting_')) and name not in ('paused_API_reconciliation','paused_controller_error'):
        return 'awaiting_existing_gate',failures
    return 'start_controller',failures


def record_exit(record,code,cursor):
    result=dict(record,last_exit_code=code,last_exit_at=now())
    if code not in (0,3):
        failures=record.get('failures_at_cursor',0) if record.get('failure_cursor')==cursor else 0
        result.update(failure_cursor=cursor,failures_at_cursor=failures+1)
    return result


def monitor(run,command,poll_seconds=15):
    directory=run/'plan';directory.mkdir(parents=True,exist_ok=True)
    budget_path=directory/'watchdog_budget.json';heartbeat=directory/'watchdog_state.json'
    with (directory/'.watchdog.lock').open('a+') as guard:
        try:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another watchdog owns this run')
        record=read_json(budget_path) if budget_path.exists() else {}
        process=None;complete=False;intentional_gate=False
        tick=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]
        while True:
            ledger,stage=snapshot(run)
            if process is not None:
                code=process.poll()
                if code is not None:
                    record=record_exit(record,code,ledger['cursor']);write_json(budget_path,record)
                    append_jsonl(directory/'watchdog_events.jsonl',{'time':now(),'event':'controller_exited',
                                 'pid':process.pid,'returncode':code,'cursor':ledger['cursor']})
                    process=None;complete=code==0;intentional_gate=code==3
            existing=read_json(directory/'controller_process.json') if (directory/'controller_process.json').exists() else {}
            adopted=owned_alive(existing,run) if existing else False
            if process is not None:state='controller_running'
            elif adopted:state='monitoring_existing_controller'
            elif complete:state='completed'
            else:
                state,failures=restart_decision(run,record)
                # A caller must explicitly request re-evaluation of a deliberate
                # non-API gate. Merely keeping the watchdog alive cannot waive it.
                if intentional_gate and stage.get('stage')!='paused_API_reconciliation':
                    state='awaiting_existing_gate'
                if state=='start_controller':
                    with (run/'logs/controller.log').open('a') as log:
                        process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,
                                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    intentional_gate=False
                    state='controller_starting'
                    append_jsonl(directory/'watchdog_events.jsonl',{'time':now(),'event':'controller_started',
                                 'pid':process.pid,'cursor':ledger['cursor'],'failures_at_cursor':failures})
            value={'pid':os.getpid(),'start_tick':tick,'active':not complete,'heartbeat_at':now(),
                   'state':state,'controller_pid':process.pid if process else existing.get('pid') if adopted else None,
                   'controller_alive':bool(process or adopted),'stage':stage.get('stage'),
                   'conditions_consumed':ledger['cursor'],'last_checkpoint':ledger.get('last_checkpoint'),
                   'max_crash_restarts_at_one_cursor':3,'credentials_saved':False,
                   'API_requests_made_by_watchdog':0,'gates_bypassed':False}
            write_json(heartbeat,value)
            if not (process or adopted) and stage.get('process_state')=='running':
                # Old controllers left this field behind when exiting a gate.
                # Only change it after checking the owned PID and watchdog child.
                stage.update(process_state='stopped',updated_at=now(),formal_conditions_consumed=ledger['cursor'])
                write_json(directory/'stage_state.json',stage)
            if complete:return
            time.sleep(poll_seconds)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True)
    args,remaining=parser.parse_known_args()
    command=[sys.executable,'-u','-m','rl_posttrain.orchestrate','--run-id',args.run_id,*remaining]
    monitor(run_path(args.run_id),command)


if __name__=='__main__':main()
