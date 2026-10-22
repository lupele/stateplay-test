"""Budgeted, restartable formal controller. Human evidence is never manufactured."""
import copy
import fcntl
import math
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import time
import yaml
from .config import load_config
from .io import api_cache_path,ROOT,read_json,write_json,append_jsonl,reward_path,reward_checkpoint,world_checkpoint,digest,sha256,now

class FormalPause(RuntimeError):
    pass

def owned_alive(record,run):
    """PID reuse and other users' jobs never establish ownership."""
    if not record.get('active') or not record.get('start_tick'):return False
    pid=int(record['pid'])
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        argv=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError,ProcessLookupError):return False
    has_id=b'--run-id' in argv and run.name.encode() in argv
    has_run_path=False
    for value in argv:
        if not value or value.startswith(b'-'):continue
        path=Path(os.fsdecode(value))
        if not path.is_absolute():path=ROOT/path
        if run.resolve() in path.resolve().parents:has_run_path=True;break
    return fields[0]!='Z' and fields[19]==record['start_tick'] and (has_id or has_run_path)

def wait_owned_workers(run):
    # An interrupted controller may leave its actual torch worker running.
    # Adopt its progress by waiting, rather than starting a duplicate batch.
    for path in sorted((run/'plan/processes').glob('*.json')):
        record=read_json(path)
        if record['pid']==os.getpid():continue
        if owned_alive(record,run):status(run,'waiting_existing_owned_worker',owned_pid=record['pid'],process_record=str(path))
        while owned_alive(record,run):time.sleep(30)
    path=run/'plan/reward_server_process.json'
    if path.exists():
        record=read_json(path)
        # At this point all owned training workers have exited; only this run's
        # leftover scorer may be released before starting a fresh service.
        if owned_alive(record,run):
            os.kill(record['pid'],signal.SIGTERM)
            for _ in range(30):
                if not owned_alive(record,run):break
                time.sleep(1)
            if owned_alive(record,run):os.kill(record['pid'],signal.SIGKILL)

def status(run,stage,**fields):
    p=run/'plan/stage_state.json';s=read_json(p) if p.exists() else {}
    s.update(stage=stage,updated_at=now(),**fields);write_json(p,s)
    append_jsonl(run/'plan/formal_events.jsonl',{'time':now(),'stage':stage,**fields})

def check_ledger(run,maximum):
    state=read_json(run/'plan/cursor.json');order=read_json(run/'plan/epoch_order.json')['order']
    if len(order)!=len(set(order)):raise ValueError('Formal order contains duplicate conditions')
    if state['cursor']!=state['api_consumed']+state['local_consumed']:raise ValueError('A+C ledger accounting mismatch')
    if not 0<=state['api_consumed']<=maximum or not 0<=state['cursor']<=len(order):raise ValueError('Formal budget exceeded')
    return state,len(order)

def api_unsafe(run):
    path=api_cache_path(run)
    if not path.exists():return False
    with sqlite3.connect(path) as db:
        return bool(db.execute("SELECT count(*) FROM requests WHERE status IN ('failed','unknown')").fetchone()[0])

def reopen_unscored_gateway_timeouts(run,groups=None):
    """Explicitly authorized recovery; unknown billing remains in the audit.

    No successful label is resubmitted, no condition is replaced, and the
    request's cumulative attempt count may never exceed the configured ceiling.
    Legacy authorizations retain their exact scope. A later explicit all-error
    authorization also covers failed/partial API replies, without resetting the
    total attempt ceiling or admitting any label that failed validation.
    """
    config_path=run/'plan/config.json';permission=run/'plan/API_gateway_retry_authorization.json'
    if not config_path.exists() or not permission.exists():return False
    config=read_json(config_path);reward=config['reward'];authorization=read_json(permission)
    if not reward.get('recover_unscored_gateway_timeouts') or not authorization.get('authorized') or not authorization.get('allow_possible_duplicate_charges'):return False
    if authorization['endpoint']!=reward['endpoint'] or authorization['model']!=reward['model']:return False
    all_errors=authorization.get('retry_all_api_errors') is True
    # Operational retry permission is separate from immutable NFT/checkpoint
    # configuration, so a raised ceiling does not change resume compatibility.
    authorized_limit=authorization['max_total_attempts_per_request']
    if not isinstance(authorized_limit,int) or not 1<=authorized_limit<=30:return False
    limit=authorized_limit if all_errors else min(reward['max_retries']+1,authorized_limit)
    path=api_cache_path(run,reward['endpoint'])
    if not path.exists():return False
    with sqlite3.connect(path) as db:
        rows=db.execute("SELECT key,group_id,status,attempts,error FROM requests WHERE status IN ('failed','unknown')").fetchall()
    if not rows:return False
    eligible={'upstream_timeout_http_504','upstream_timeout_http_524'}
    if 'ReadTimeout' in authorization.get('authorized_unknown_errors',[]):eligible.add('ReadTimeout')
    if any(n>=limit or (not all_errors and (s!='unknown' or e not in eligible)) for _,_,s,n,e in rows):return False
    if groups is not None and any(g not in {Path(p).name for p in groups} for _,g,_,_,_ in rows):return False
    ledger=read_json(run/'plan/cursor.json');reserved=ledger.get('inflight')
    if not reserved or reserved['phase'] not in ('generated','scored') or reserved.get('update_cursor',0):return False
    from .io import teacher_path
    from .rewards.api_group import RequestStore
    prepared=[]
    for key,group_id,previous,attempts,error in rows:
        group=run/'groups'/group_id;teacher=teacher_path(group)
        condition=read_json(group/'condition.json')['condition_id']
        if condition not in reserved['condition_ids'] or (teacher/'scores.json').exists():return False
        if read_json(teacher/'request_map.json')['cache_key']!=key:return False
        if all_errors:
            # A transport interruption can occur before any headers or JSON.
            # The committed cache row itself is durable failure evidence. Keep
            # every available provider artifact alongside that snapshot.
            from .io import sha256
            artifacts=[teacher/f'response_attempt_{attempts:02d}.json',teacher/f'retry_attempt_{attempts:02d}.json',
                       teacher/f'stream_headers_attempt_{attempts:02d}.json',teacher/f'stream_attempt_{attempts:02d}.jsonl',teacher/'unknown.json']
            raw={'cache_status':previous,'cache_error':error,'cache_attempts':attempts,
                 'artifacts':[{'path':str(p),'sha256':sha256(p)} for p in artifacts if p.exists()]}
        elif error=='ReadTimeout':
            evidence=teacher/'unknown.json'
            if not evidence.exists():return False
            raw=read_json(evidence)
            if raw.get('cache_key')!=key or raw.get('reason')!='ReadTimeout':return False
            if raw.get('attempt',attempts)!=attempts:return False
        else:
            response=teacher/f'response_attempt_{attempts:02d}.json'
            if not response.exists():return False
            raw=read_json(response)
            if raw.get('http_status')!=int(error.rsplit('_',1)[1]):return False
        prepared.append((key,group_id,previous,attempts,error,raw))
    store=RequestStore(path)
    for key,group_id,previous,attempts,error,raw in prepared:
        reason='User explicitly authorized bounded retry of '+str(error or 'unspecified_API_failure')+', accepting possible duplicate charges; prior billing remains unknown'
        write_json(run/'plan/API_gateway_recoveries'/f'{key}-attempt-{attempts:02d}.json',{
            'time':now(),'group_id':group_id,'cache_key':key,'previous_status':previous,'previous_attempts':attempts,
            'previous_error':error,'original_response':raw,'billing_outcome':'unknown','request_attempt_ceiling':limit,
            'additional_attempts':limit-attempts,'provider_lookup_available':False,
            'successful_labels_reused':True,'ledger_reset':False,'authorization':authorization})
        store.authorize_retry(key,limit-attempts,reason)
    append_jsonl(run/'plan/formal_events.jsonl',{'time':now(),'event':'authorized_API_error_recovery' if all_errors else 'authorized_gateway_timeout_recovery','groups':[g for _,g,_,_,_,_ in prepared],'request_attempt_ceiling':limit,'billing_outcome':'unknown'})
    return True

def run_command(command,run,name,gpus,execute,attempts=3):
    """Torch workers resume committed boundaries. Terminal/unknown API work is not reset."""
    key=run/'plan/recoveries'/(name+'.json')
    recovery=read_json(key) if key.exists() else {'failed_launches':0}
    while recovery['failed_launches']<attempts:
        status(run,name,process_state='running')
        resumed=list(command)
        if 'rl_posttrain.reward_model.train' in resumed:
            latest=reward_path(run)/'checkpoints/latest.json'
            if latest.exists():
                checkpoint=read_json(latest)['checkpoint']
                if '--resume' in resumed:resumed[resumed.index('--resume')+1]=checkpoint
                else:resumed+=['--resume',checkpoint]
        reward_progress_path=reward_path(run)/'checkpoints/latest.json' if 'rl_posttrain.reward_model.train' in resumed else None
        reward_step_before=read_json(reward_progress_path).get('step',0) if reward_progress_path and reward_progress_path.exists() else 0
        code=execute(resumed,run,name,gpus,allow_failure=True)
        if code==0:return
        current_cursor=read_json(run/'plan/cursor.json')['cursor'] if (run/'plan/cursor.json').exists() else 0
        if current_cursor>recovery.get('last_failure_cursor',current_cursor):recovery['failed_launches']=0
        if reward_progress_path:
            reward_step_after=read_json(reward_progress_path).get('step',0) if reward_progress_path.exists() else 0
            if reward_step_after>reward_step_before or reward_step_after>recovery.get('last_reward_step',reward_step_after):
                recovery['failed_launches']=0
            recovery['last_reward_step']=reward_step_after
        recovery['total_failed_launches']=recovery.get('total_failed_launches',recovery['failed_launches'])+1
        recovery['failed_launches']+=1;recovery['last_failure_cursor']=current_cursor;recovery['last_exit_code']=code;write_json(key,recovery)
        initial=run/'reports/C_initial_audit/gate.json'
        if name=='formal_C' and initial.exists():
            gate=read_json(initial)
            if not gate['passed']:
                status(run,'awaiting_C_initial_review' if gate['status']=='awaiting_human_review' else 'paused_C_initial_quality',
                       checkpoint_retained=True,review_path=gate['review_path'],quality_report=str(initial))
                raise FormalPause(gate['status'])
        monitor=run/'metrics/reward_audits.json'
        if name=='formal_C' and monitor.exists() and read_json(monitor).get('paused'):
            status(run,'paused_C_reward_drift',checkpoint_retained=True);raise FormalPause('Reward drift stop condition')
        reopened=reopen_unscored_gateway_timeouts(run)
        if reopened:
            # These attempts have their own per-request ceiling. Preserve all
            # failures in total_failed_launches without consuming the unrelated
            # consecutive worker-crash budget for an authorized API recovery.
            recovery['failed_launches']-=1
            recovery['gateway_timeout_launches']=recovery.get('gateway_timeout_launches',0)+1
            write_json(key,recovery)
            status(run,'recovering_unscored_gateway_timeout',failed_process=name,request_attempts_capped=True)
        if api_unsafe(run):
            status(run,'paused_API_reconciliation',failed_process=name,checkpoint_retained=True)
            raise FormalPause('Terminal or unknown API requests require reconciliation; no automatic double submission')
        if recovery['failed_launches']>=attempts:break
        status(run,'recovering',failed_process=name,restart_number=recovery['failed_launches'])
        time.sleep(10)
    status(run,'paused_recovery_limit',failed_process=name,checkpoint_retained=True)
    raise FormalPause('Bounded recovery launch limit reached: '+name)

def save_config(run,name,c):
    values=copy.deepcopy(c);values.pop('config_hash',None)
    path=run/'plan'/name;text=yaml.safe_dump(values,sort_keys=False)
    if path.exists() and path.read_text()!=text:raise ValueError('Immutable formal configuration changed: '+str(path))
    if not path.exists():path.write_text(text)
    return str(path)

def verify_trial(run,execute,world_python):
    summary=run/'reports/trial_140_summary.json'
    if not summary.exists():
        run_command([world_python,'-m','rl_posttrain.trial_report','--run-id',run.name],run,'formal_140_report',[],execute)
    report=read_json(summary);completion=report['completion']
    reasons=[]
    if completion.get('software_smoke'):reasons.append('Trial contains smoke data')
    if report['groups']<min(140,completion['ledger']['api_consumed']):reasons.append('Trial has incomplete scores')
    if not completion.get('frozen_hash'):reasons.append('Trial frozen-parameter proof missing')
    if completion.get('parameter_delta_from_ref_l2',0)<=0:reasons.append('Trial parameters did not update')
    sync=read_json(run/'reports/api_ddp_sync.json')
    if not sync.get('equal') or sync.get('ranks')!=7:reasons.append('Trial seven-rank synchronization failed')
    if api_unsafe(run):reasons.append('Trial contains unresolved API requests')
    steps=run/'metrics/api_steps.jsonl'
    if not steps.exists():reasons.append('Trial has no real optimizer updates')
    else:
        import json
        for line in steps.read_text().splitlines():
            row=json.loads(line)
            if row.get('smoke') or row['mode']!='B-only':reasons.append('Unexpected trial update mode');break
            for key in ('rank0_fm','rank0_ce'):
                if not math.isfinite(row[key]):reasons.append('Nonfinite '+key)
    gate={'passed':not reasons,'reasons':reasons,'conditions':completion['ledger']['api_consumed'],
          'report':str(summary),'checkpoint':completion['checkpoint'],
          'mechanics_event_review':'Actual video review queue saved; no event accuracy is inferred from scalar rewards'}
    write_json(run/'reports/formal_140_health.json',gate)
    if reasons:status(run,'paused_A_140_health',reasons=reasons);raise FormalPause('First 140 condition health gate failed')

def epoch_manifest(cp,final,c):
    if (cp/'manifest.json').exists():return read_json(cp/'manifest.json')
    h=sha256(cp/'adapter/adapter_model.safetensors');audit=final['audit']
    version=digest({'adapter':h,'base_index':audit['base_index_sha256'],'processor':audit['processor_hash'],
                    'prompt':sha256(c['reward']['prompt']),'protocol':c['reward']['protocol']})[:24]
    m={k:final[k] for k in ('schema','smoke','base_model','protocol','prompt_sha256','config','audit','dataset_hash','steps_per_epoch','world_size','total_steps')}
    m.update(version=version,adapter_sha256=h,steps=int(cp.name.split('-')[-1]),
             training_state_sha256=sha256(cp/'training_state.pt'),quality_claim=False)
    write_json(cp/'manifest.json',m);return m

def reward_training(c,run,config,execute,reward_python):
    """Preserve every attempt; select epoch using validation before opening test."""
    rm=reward_path(run);final=rm/'final/manifest.json';selected=rm/'selected/manifest.json'
    if selected.exists():
        selected_info=read_json(selected)
        if selected_info.get('smoke'):raise ValueError('Development RM cannot be selected')
        if (rm/'selected/selection_complete.json').exists():
            if sha256(rm/'selected/adapter/adapter_model.safetensors')!=selected_info['adapter_sha256']:raise ValueError('Completed selected reward adapter changed')
            return
    args=['--config',config,'--run-id',run.name]
    if not final.exists():
        latest=rm/'checkpoints/latest.json'
        if latest.exists():args+=['--resume',read_json(latest)['checkpoint']]
        command=[reward_python,'-m','torch.distributed.run','--standalone','--nproc_per_node','7','-m','rl_posttrain.reward_model.train',*args]
        run_command(command,run,'formal_B_'+rm.name,list(range(7)),execute)
    m=read_json(final)
    if m.get('smoke'):raise ValueError('Development reward model cannot enter formal selection')
    choices=[]
    for epoch in range(1,c['reward_model']['epochs']+1):
        cp=rm/'checkpoints'/f"step-{epoch*m['steps_per_epoch']:08d}";em=epoch_manifest(cp,m,c)
        metrics=rm/'reports'/em['version']/'validation/metrics.json'
        if not metrics.exists():
            command=[reward_python,'-m','torch.distributed.run','--standalone','--nproc_per_node','7','-m','rl_posttrain.reward_model.evaluate',
                     '--config',config,'--run-id',run.name,'--checkpoint',str(cp),'--split','validation','--check-reload']
            run_command(command,run,f'formal_B_validation_{rm.name}_epoch{epoch}',list(range(7)),execute)
        ev=read_json(metrics)
        # Invalid-output rate has priority; ties prefer the earlier epoch.
        key=(ev.get('json_valid_rate',0)<.99,ev.get('mae',float('inf')),epoch)
        choices.append((key,cp,em,ev))
    _,cp,em,ev=min(choices,key=lambda x:x[0])
    destination=rm/'selected'
    if destination.exists():
        # Resume a partially copied selection without modifying its source checkpoint.
        shutil.copytree(cp,destination,dirs_exist_ok=True)
    elif not destination.exists():shutil.copytree(cp,destination)
    write_json(rm/'reports/selection.json',{'criterion':'validation JSON>=99%, then minimum validation MAE, earlier epoch on tie',
               'chosen_checkpoint':str(cp),'chosen_version':em['version'],'validation_groups':ev['groups'],
               'test_used_for_selection':False,'candidates':[{'checkpoint':str(p),'version':v['version'],'validation_mae':e.get('mae'),'json_valid_rate':e['json_valid_rate']} for _,p,v,e in choices]})
    write_json(destination/'selection_complete.json',{'version':em['version'],'source_checkpoint':str(cp),
               'adapter_sha256':sha256(destination/'adapter/adapter_model.safetensors'),'selected_using':'validation_only'})

def stage_c_control(run):
    """An explicit user hold survives controller/watchdog and manual launches."""
    path=run/'plan/stage_C_control.json'
    if not path.exists():return None
    control=read_json(path)
    return None if control.get('allow_start') is True else control

def pause_before_C(run,**fields):
    control=stage_c_control(run)
    if control is not None:
        status(run,'paused_after_B_user_request',B_completed=True,C_started=False,
               stage_C_start_disabled=True,control_file=str(run/'plan/stage_C_control.json'),**fields)
        raise FormalPause('User requested stop after Stage B; Stage C is disabled')

def wait_for_review(run,path,stage,wait):
    status(run,stage,review_path=str(path),human_review_required=True)
    if not wait:raise FormalPause('Actual human review is pending: '+str(path))
    while not Path(path).exists():time.sleep(30)

def run_plan(c,run,config,execute,world_python,reward_python,auto_extend=False,wait_human=False):
    from .train_world import formal_gate
    from .quality_gate import numeric_reasons,evaluate_gate
    from .review import build_review
    formal_gate(run,'api')
    if c['nft']['mode']!='B-only':raise ValueError('study_v1 is authorized for B-only')
    maximum=min(c['plan']['api_max_conditions'],4000)
    ledger,total=check_ledger(run,maximum)
    controller=run/'plan/formal_controller.json'
    policy=read_json(controller) if controller.exists() else {'A_target':min(c['plan']['api_conditions'],total),
            'A_initial':min(c['plan']['api_conditions'],total),'A_max':maximum,'extension_block':c['plan']['api_extension'],
            'total_conditions':total,'total_dataset_passes':1,'parameter_mode':'B-only',
            'auto_extensions_authorized':auto_extend,'wait_for_actual_human_review':wait_human,'C_started':False}
    if policy['total_conditions']!=total or policy['parameter_mode']!='B-only':raise ValueError('Formal controller identity changed')
    if auto_extend:policy['auto_extensions_authorized']=True
    write_json(controller,policy)
    while True:
        ledger,total=check_ledger(run,maximum);target=policy['A_target']
        values=copy.deepcopy(c);values['plan']['api_conditions']=target
        Aconfig=save_config(run,f'formal_A_budget_{target:06d}_{c["config_hash"][:12]}.yaml',values)
        status(run,'formal_A',A_budget=target,C_budget=total-target,total_conditions=total,C_started=policy['C_started'])
        if ledger['api_consumed']<target:
            if ledger['local_consumed'] or (ledger['inflight'] and ledger['inflight']['stage']=='local'):raise FormalPause('A extension cannot discard a C reservation or rewind consumed conditions')
            if ledger['api_consumed']<min(140,target):
                run_command(['bash','rl_posttrain/scripts/run_api.sh','--run-id',run.name,'--config',Aconfig,
                             '--max-conditions',str(min(140,target)),'--api-group-limit',str(min(140,target))],run,'formal_A_first140',list(range(7)),execute)
            if not (run/'reports/formal_140_health.json').exists():verify_trial(run,execute,world_python)
            elif not read_json(run/'reports/formal_140_health.json')['passed']:raise FormalPause('Stored A health gate has not passed')
            ledger,_=check_ledger(run,maximum)
            if ledger['api_consumed']<target:
                run_command(['bash','rl_posttrain/scripts/run_api.sh','--run-id',run.name,'--config',Aconfig,'--api-group-limit',str(target)],run,'formal_A_to_'+str(target),list(range(7)),execute)
        ledger,_=check_ledger(run,maximum)
        if ledger['api_consumed']!=target:raise ValueError('A completion differs from its saved budget')
        world=world_checkpoint(run,'api')
        if not world.exists() or read_json(world.with_suffix('.json'))['state'].get('smoke'):raise ValueError('Formal A final checkpoint is missing or development-only')
        active=run/'reward_model/active.json';relative='versions/'+f'a-{target:06d}'
        if not active.exists():write_json(active,{'relative_path':relative,'A_conditions':target})
        elif read_json(active)['relative_path']!=relative:
            if policy['C_started']:raise FormalPause('Cannot replace a frozen reward version after C started')
            write_json(active,{'relative_path':relative,'A_conditions':target})
        rm=reward_path(run);rm.mkdir(parents=True,exist_ok=True)
        Bvalues=copy.deepcopy(values);Bvalues['stage']='reward_model'
        Bconfig=save_config(run,f'formal_B_data_{target:06d}_{c["config_hash"][:12]}.yaml',Bvalues)
        reward_training(Bvalues,run,Bconfig,execute,reward_python)
        m=read_json(reward_checkpoint(run)/'manifest.json')
        metrics_path=rm/'reports/evaluation.json'
        if not metrics_path.exists():
            command=[reward_python,'-m','torch.distributed.run','--standalone','--nproc_per_node','7','-m','rl_posttrain.reward_model.evaluate',
                     '--config',Bconfig,'--run-id',run.name,'--split','test','--check-reload','--shortcut-checks']
            run_command(command,run,'formal_B_test_'+rm.name,list(range(7)),execute)
        metrics=read_json(metrics_path)
        if metrics['version']!=m['version']:raise ValueError('RM test/model versions differ')
        numerical=numeric_reasons(metrics)
        review=build_review(run)
        pause_before_C(run,reward_version=m['version'],numeric_quality_reasons=numerical,review_path=str(review))
        if numerical:
            proposal={'previous_A_budget':target,'numeric_reasons':numerical,'A_max':maximum,'approved_block':policy['extension_block']}
            next_target=target+policy['extension_block']
            if not policy.get('auto_extensions_authorized') or next_target>min(maximum,total):
                proposal['unused_API_allowance']=max(0,maximum-target)
                status(run,'paused_reward_quality',proposal=proposal,C_started=False);raise FormalPause('Reward quality failed; remaining authorized budget cannot fit an extension block, or extension not authorized')
            proposal.update(next_A_budget=next_target,additional_conditions=next_target-target,next_C_budget=total-next_target,
                            requires_explicit_budget_approval=False,uses_unconsumed_conditions=True)
            write_json(rm/'reports/expansion.json',proposal)
            policy['A_target']=next_target;write_json(controller,policy)
            status(run,'expanding_A_within_authorized_budget',**proposal)
            continue
        review_gate=evaluate_gate(run,metrics,m['version'],review if review.exists() else None)
        while not review_gate['passed']:
            if review.exists():
                status(run,'paused_invalid_human_review',reasons=review_gate['reasons'],review_path=str(review))
                raise FormalPause('Human evidence did not pass; inspect the reported reasons')
            wait_for_review(run,review,'awaiting_reward_human_review',wait_human)
            review_gate=evaluate_gate(run,metrics,m['version'],review)
        pause_before_C(run,reward_version=m['version'])
        # The formal selected model has now produced actual full-group test inference.
        if not (run/'reports/joint_profile.json').exists():
            sequential=run/'reports/C_timeslice_acceptance.json'
            if not sequential.exists() or not read_json(sequential).get('passed'):
                raise FormalPause('Sequential all-GPU C requires the current four-GPU software acceptance report')
            gpus=c.get('stage_c',{}).get('gpus') or sorted(set(c['topology']['world_gpus']+c['topology']['reward_gpus']))
            write_json(run/'reports/joint_profile.json',{'kind':'sequential_all_GPU_capacity_acceptance',
                'selected_topology':{'strategy':'timeslice','world_gpus':gpus,'reward_gpus':gpus,'conditions_per_batch':28},
                'evidence':str(sequential),'measured_hardware':read_json(sequential).get('physical_GPU_type'),
                'target_hardware_throughput':'measured by the first actual C batch, not inferred from smoke timing'})
        topology=read_json(run/'reports/joint_profile.json')['selected_topology']
        if topology is None:raise FormalPause('No measured joint topology with required headroom')
        # Historical dedicated profiles stay on disk; execution now time-slices
        # the complete selected device list in each of the three phases.
        gpus=c.get('stage_c',{}).get('gpus') or sorted(set(topology['world_gpus']+topology['reward_gpus']))
        topology={**topology,'strategy':'timeslice','world_gpus':gpus,'reward_gpus':gpus}
        local_values=copy.deepcopy(values);local_values['stage']='local';local_values['topology']=topology
        local_values['rollout']['conditions_per_batch']=topology['conditions_per_batch']
        Cconfig=save_config(run,'formal_C_frozen_reward_'+m['version']+'.yaml',local_values)
        pause_before_C(run,reward_version=m['version'])
        policy['C_started']=True;policy['reward_version']=m['version'];write_json(controller,policy)
        status(run,'formal_C',A_budget=target,C_budget=total-target,C_started=True,reward_version=m['version'],
               initial_audit_buffer=56,ordinary_conditions_per_batch=local_values['rollout']['conditions_per_batch'])
        while True:
            try:
                run_command(['bash','rl_posttrain/scripts/run_local.sh','--run-id',run.name,'--config',Cconfig],run,'formal_C',None,execute)
                break
            except FormalPause:
                initial=run/'reports/C_initial_audit/gate.json'
                if not initial.exists() or read_json(initial)['status']!='awaiting_human_review':raise
                review_path=Path(read_json(initial)['review_path'])
                if review_path.exists():
                    status(run,'paused_invalid_C_initial_review',review_path=str(review_path))
                    raise FormalPause('Existing C initial human evidence did not pass; no retry loop')
                wait_for_review(run,review_path,'awaiting_C_initial_review',wait_human)
                # A review pause is an intended gate, not a failed recovery launch.
                recover=run/'plan/recoveries/formal_C.json'
                if recover.exists():
                    r=read_json(recover);r['failed_launches']=max(0,r['failed_launches']-1);write_json(recover,r)
        ledger,total=check_ledger(run,maximum)
        if ledger['cursor']!=total or ledger['inflight']:raise FormalPause('C returned without completing its condition budget')
        exported=run/'world/local/export/StatePlay_NFT.safetensors'
        if not exported.exists():
            run_command([world_python,'-m','rl_posttrain.export_checkpoint','--checkpoint',str(world_checkpoint(run,'local')),'--output',str(exported)],run,'formal_C_export',[],execute)
        run_command([world_python,'-m','rl_posttrain.source_identity','--run-id',run.name,'--check'],run,'formal_source_protection_after',[],execute)
        status(run,'complete',formal_training_completed=True,A_conditions=ledger['api_consumed'],C_conditions=ledger['local_consumed'],
               world_checkpoint=str(world_checkpoint(run,'local')),reward_checkpoint=str(reward_checkpoint(run)),reward_version=m['version'])
        from .formal_report import build
        build(run)
        return

def supervise(c,run,config,execute,world_python,reward_python,auto_extend=False,wait_human=False):
    path=run/'plan/.formal_controller.lock';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another formal controller already owns this run')
        record={'pid':os.getpid(),'started_at':now(),'run_id':run.name,'active':True,
                'start_tick':Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]}
        write_json(run/'plan/controller_process.json',record)
        try:
            wait_owned_workers(run)
            code=execute([world_python,'-m','rl_posttrain.formal_preflight','--run-id',run.name,'--config',config],run,'formal_resource_API_preflight',allow_failure=True)
            if code:
                report=read_json(run/'reports/formal_resource_API_preflight.json')
                status(run,'paused_preflight',failures=report['failures'],formal_conditions_consumed=read_json(run/'plan/cursor.json')['cursor'])
                raise FormalPause('Formal resource/API preflight has not passed')
            run_plan(c,run,config,execute,world_python,reward_python,auto_extend,wait_human)
        except FormalPause as error:
            print('Formal task paused:',str(error),flush=True);raise SystemExit(3)
        except Exception as error:
            status(run,'paused_controller_error',error_type=type(error).__name__)
            raise
        finally:
            record.update(active=False,finished_at=now())
            write_json(run/'plan/controller_process.json',record)
            state=read_json(run/'plan/stage_state.json') if (run/'plan/stage_state.json').exists() else {}
            ledger=read_json(run/'plan/cursor.json') if (run/'plan/cursor.json').exists() else {}
            state.update(process_state='stopped',controller_pid=os.getpid(),updated_at=now(),
                         formal_conditions_consumed=ledger.get('cursor',0))
            write_json(run/'plan/stage_state.json',state)
