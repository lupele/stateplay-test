"""One explicitly requested full-group probe, then bounded formal continuation.

Credentials remain in the process environment. Historical requests and attempt
numbers are preserved; a failed probe never triggers another launch or grant.
"""
import argparse
import fcntl
import os
from pathlib import Path
import sqlite3

from .config import load_config
from .io import ROOT, now, read_json, write_json, run_path, append_jsonl, api_cache_path
from .rewards.api_group import APIGroupScorer, RequestStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--attempt-id', required=True)
    parser.add_argument('--authorization-reason', required=True)
    parser.add_argument('--continue-plan', action='store_true')
    parser.add_argument('--group-id', help='Use this same group for a transport repair probe')
    parser.add_argument('--config')
    args = parser.parse_args()
    if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.attempt_id):
        raise ValueError('Invalid attempt id')
    run = run_path(args.run_id)
    config = load_config(args.config or (run/'plan/config.json' if (run/'plan/config.json').exists() else ROOT/'rl_posttrain/configs/api_b_only.yaml'))
    record = run/'plan/API_resume'/f'{args.attempt_id}.json'
    lock_file = (run/'plan/.formal_controller.lock').open('a+')
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if record.exists():
        raise RuntimeError('This probe id already exists; no automatic resubmission')
    from .formal import owned_alive
    if any(owned_alive(read_json(p), run) for p in (run/'plan/processes').glob('*.json')
           if read_json(p).get('pid') != os.getpid()):
        raise RuntimeError('An owned worker is still active')
    if not (os.environ.get('STATEPLAY_GPT_API_KEY') or os.environ.get('OPENAI_API_KEY')):
        raise RuntimeError('Configured credential unavailable before submission')
    store = RequestStore(api_cache_path(run,config['reward']['endpoint']))
    with store.connect() as db:
        if args.group_id:
            row = db.execute("SELECT key,group_id,status,attempts FROM requests WHERE group_id=? AND status IN ('failed','unknown')",(args.group_id,)).fetchone()
            if row is None:raise RuntimeError('Selected repair group is not terminal')
        else:
            row = db.execute("SELECT key,group_id,status,attempts FROM requests WHERE status='failed' ORDER BY group_id LIMIT 1").fetchone()
        if row is None:
            row = db.execute("SELECT key,group_id,status,attempts FROM requests WHERE status='unknown' ORDER BY group_id LIMIT 1").fetchone()
        if row is None:
            raise RuntimeError('No terminal request requires a recovery probe')
        archive = run/'provenance/API_resume'/args.attempt_id
        archive.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(archive/'cache_before_retry.sqlite3') as backup:
            db.backup(backup)
    state = {'time': now(), 'group_id': row[1], 'cache_key': row[0],
             'previous_status': row[2], 'previous_attempts': row[3],
             'authorization_reason': args.authorization_reason,
             'probe_additional_attempt_limit': 1, 'status': 'prepared',
             'streaming_transport': os.environ.get('STATEPLAY_API_STREAM')=='1',
             'continuation_requested': args.continue_plan}
    write_json(record, state)
    store.authorize_retry(row[0], 1, args.authorization_reason)
    state.update(status='submitting', submitted_at=now())
    write_json(record, state)
    print({'probe_group': row[1], 'previous_attempts_preserved': row[3],
           'additional_probe_attempts': 1}, flush=True)
    try:
        scores = APIGroupScorer(config, run).score(run/'groups'/row[1])
    except Exception as error:
        state.update(status='failed_or_unknown', finished_at=now(), error_type=type(error).__name__)
        write_json(record, state)
        from .formal import status
        status(run, 'paused_API_probe_failed', process_state='stopped', probe_report=str(record))
        print({'API_probe_passed': False, 'report': str(record)}, flush=True)
        raise SystemExit(3) from None
    state.update(status='succeeded', scores=scores, finished_at=now())
    write_json(record, state)
    print({'API_probe_passed': True, 'scores': scores}, flush=True)
    if not args.continue_plan:
        return
    # The caller explicitly requested continuation if this full protocol works.
    # Reopen only this same inflight batch, with one configured retry window per
    # group. Nothing resets the epoch cursor, seeds, optimizer, or attempts.
    reserved = read_json(run/'plan/cursor.json')['inflight']
    ids = set(reserved['condition_ids']) if reserved else set()
    with store.connect() as db:
        remaining = db.execute("SELECT key,group_id FROM requests WHERE status IN ('failed','unknown')").fetchall()
    granted = []
    for key, group_id in remaining:
        if read_json(run/'groups'/group_id/'condition.json')['condition_id'] not in ids:
            raise RuntimeError('Retry scope escaped the existing reserved batch')
        store.authorize_retry(key, config['reward']['max_retries']+1, args.authorization_reason)
        granted.append(group_id)
    state['bounded_continuation_groups'] = granted
    state['continuation_additional_attempts_per_group'] = config['reward']['max_retries']+1
    write_json(record, state)
    append_jsonl(run/'plan/formal_events.jsonl', {'time': now(), 'event': 'API_probe_passed_formal_resume',
                 'probe_report': str(record), 'retry_groups': granted, 'ledger_reset': False})
    # The same PID becomes the formal controller with the credential still only
    # in memory. Release the probe lock before the controller acquires it.
    fcntl.flock(lock_file, fcntl.LOCK_UN)
    lock_file.close()
    os.execv('/bin/bash', ['bash', str(ROOT/'rl_posttrain/scripts/run_study_v1.sh')])


if __name__ == '__main__':
    main()
