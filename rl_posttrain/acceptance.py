"""Build an integration gate from real artifacts; missing evidence cannot pass."""
import argparse
import subprocess
from pathlib import Path
import sqlite3
from .io import ROOT,run_path,read_json,write_json,sha256
from .config import load_config,validate

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--publish-to',help='Publish software evidence without changing the target ledger');p.add_argument('--formal-regression-run');p.add_argument('--refresh-progress',action='store_true');p.add_argument('--formal-config',default=str(ROOT/'rl_posttrain/configs/api_b_only.yaml'));a=p.parse_args();run=run_path(a.run_id)
    checks={};evidence={}
    def get(name,path):
        full=run/path
        if not full.exists():checks[name]=False;return None
        evidence[name]=str(full);return read_json(full)
    modes=get('real_update_modes','reports/real_world_modes_v2.json' if (run/'reports/real_world_modes_v2.json').exists() else 'reports/real_world_modes.json')
    expected={'B-only':{'B'},'Coupling-only':{'Coupling'},'Both':{'B','Coupling'}}
    checks['real_update_modes']=bool(modes and set(modes['modes'])==set(expected) and all(x['frozen_hash_before']==x['frozen_hash_after'] and x['reload_max_abs_error']==0 and len(x['steps'])>=3 and all((x['parameter_delta_l2'][k]>0 and x['steps'][-1]['nft_grad_norm'][k]>0) if k in expected[m] else x['parameter_delta_l2'][k]==0 for k in ('B','Coupling')) for m,x in modes['modes'].items()))
    both_sync=get('Both_7GPU_DDP_sync','reports/both_7gpu_sync_v2.json' if (run/'reports/both_7gpu_sync_v2.json').exists() else 'reports/both_7gpu_sync.json');checks['Both_7GPU_DDP_sync']=bool(both_sync and both_sync['equal'] and both_sync['ranks']==7)
    parity=get('existing_inference_parity','reports/original_entry_parity_v2.json' if (run/'reports/original_entry_parity_v2.json').exists() else 'reports/original_entry_parity.json');checks['existing_inference_parity']=bool(parity and parity['pixel_max_abs_error']==0 and parity['latent_save_reload_decode_exact'])
    world=get('world_original_scale','reports/world_full_profile_v2.json' if (run/'reports/world_full_profile_v2.json').exists() else 'reports/world_full_profile.json');checks['world_original_scale']=bool(world and world['height']==480 and world['width']==832 and world['full_group'])
    api=get('real_API','reports/real_api_acceptance.json');checks['real_API']=bool(api and api['model']=='gpt-6.1-sol' and api['images']==124 and api['candidate_count']==4)
    stress=get('32_process_mock_API','reports/api_32_process_stress.json');checks['32_process_mock_API']=bool(stress and stress['workers']==32 and stress['peak_server_inflight']==32 and stress['paid_API_requests']==0 and stress['cached_repeat_added_requests']==0)
    qwen=get('Qwen_local_full_model','reports/qwen_local_load_v2.json' if (run/'reports/qwen_local_load_v2.json').exists() else 'reports/qwen_local_load.json');checks['Qwen_local_full_model']=bool(qwen and qwen['full_multimodal'] and qwen['local_files_only'] and len(qwen['targets'])==200)
    suffix=get('real_Qwen_suffix_gradient','reports/qwen_suffix_reference_v2.json' if (run/'reports/qwen_suffix_reference_v2.json').exists() else 'reports/qwen_suffix_reference.json');checks['real_Qwen_suffix_gradient']=bool(suffix and abs(suffix['full_loss']-suffix['suffix_loss'])<1e-5 and suffix['max_gradient_abs_error']<2e-4)
    train=get('real_Qwen_LoRA','reports/reward_training_smoke_v2.json' if (run/'reports/reward_training_smoke_v2.json').exists() else 'reports/reward_training_smoke.json');checks['real_Qwen_LoRA']=bool(train and train['steps']>=3 and train['delta_l2']>0 and train['frozen_hash_before']==train['frozen_hash_after'] and all(e['image_count']==124 for e in train['full_group_examples']))
    evaluate=get('RM_save_reload_scores','reports/reward_evaluation_v2.json' if (run/'reports/reward_evaluation_v2.json').exists() else 'reward_model/reports/evaluation.json');checks['RM_save_reload_scores']=bool(evaluate and evaluate['json_valid_rate']==1 and evaluate.get('reload_adapter_tensors',{}).get('exact') and evaluate.get('frozen_buffers_unchanged'))
    local=get('local_C_continuation','reports/local_completion.json');checks['local_C_continuation']=bool(local and local['optimizer_step']>=6 and local['ledger']['local_consumed']>0)
    sync=get('local_DDP_sync','reports/local_ddp_sync.json');checks['local_DDP_sync']=bool(sync and sync['equal'] and sync['ranks']>=2)
    recovery=get('real_recovery','reports/recovery_acceptance.json');checks['real_recovery']=bool(recovery and len(recovery['events'])>=3 and recovery['paid_API_calls']==0)
    c_resume=get('C_transactional_resume','reports/local_resume_comparison.json');checks['C_transactional_resume']=bool(c_resume and c_resume.get('transactional_resume_passed') and c_resume['duplicate_committed_updates']==0)
    rm_resume=get('Qwen_7GPU_resume','reports/reward_resume_acceptance.json');checks['Qwen_7GPU_resume']=bool(rm_resume and rm_resume['passed'] and rm_resume['saved_ranks']==7 and rm_resume['final_step']==3)
    strict_world=get('C_exact_recovery_v2','reports/world_resume_diagnosis_v2.json');checks['C_exact_recovery_v2']=bool(strict_world and strict_world['passed'] and strict_world['bitwise_equal'] and strict_world['optimizer_equal'] and strict_world['policy_bank_equal'])
    strict_reward=get('Qwen_accum4_recovery_v2','reports/reward_accum4_resume_v2.json');checks['Qwen_accum4_recovery_v2']=bool(strict_reward and strict_reward['passed'] and strict_reward['accumulation']==4 and strict_reward['ranks']==7 and strict_reward['bitwise_equal'])
    d_button=get('D_semantics_v2','reports/D_semantics_source_audit_v2.json');checks['D_semantics_v2']=bool(d_button and d_button['both_facing_directions_tested'] and d_button['actual_StatePlay_arxiv']=='2607.26754v1')
    quarantine=get('legacy_rendering_quarantine','reports/legacy_quarantine_v2.json');checks['legacy_rendering_quarantine']=bool(quarantine and quarantine['formal_NFT_excluded'] and quarantine['formal_RM_excluded'] and quarantine['original_group_files_unchanged'])
    rendering=get('current_rendering_v3','reports/current_rendering_v3_protocol_v2.json');checks['current_rendering_v3']=bool(rendering and rendering['passed'] and rendering['images']==124 and rendering['generated_candidates']==4 and rendering['API_requests']==0 and rendering['actual_rendering']['version']=='bf16-no-autocast-deterministic-v3')
    processor=get('Qwen_v2_full_group_processor','reports/Qwen_v2_processor.json');checks['Qwen_v2_full_group_processor']=bool(processor and processor['passed'] and processor['local_files_only'] and processor['metadata']['image_count']==124 and processor['protocol']=='mechanics-reward-v2-24x832-124images')
    leakage=get('source_action_temporal_split','reports/source_leakage_acceptance_v2.json');checks['source_action_temporal_split']=bool(leakage and leakage['passed'] and leakage['training_records']==10000 and leakage['temporal_fingerprints']==5)
    topology=get('joint_topology','reports/joint_profile.json');checks['joint_topology']=bool(topology and topology['selected_topology'])
    protected=get('original_files_protected','reports/source_identity_after.json');before=get('original_file_baseline','reports/source_identity_before.json');checks['original_files_protected']=bool(protected and protected==before)
    original=read_json(run/'reports/original_checkpoint_before.json');after=protected.get('outputs/latent_world/step-4572.safetensors') if protected else None
    checks['original_checkpoint_since_task_start']=bool(after and all(after[k]==original[k] for k in ('sha256','size','mtime_ns')))
    testlog=run/'logs/all_tests_final_code.log';checks['CPU_and_original_regression_tests']=testlog.exists() and 'passed' in testlog.read_text() and not any(s in testlog.read_text() for s in (' FAILED ',' ERROR '))
    count=attempts=0
    if (run/'requests/cache.sqlite3').exists():
        with sqlite3.connect(run/'requests/cache.sqlite3') as db:
            count,attempts=db.execute('SELECT count(*),sum(attempts) FROM requests').fetchone()
            succeeded=db.execute("SELECT count(*) FROM requests WHERE status='succeeded'").fetchone()[0]
        # Authorization limits complete candidate groups, not bounded retries
        # of the same group after confirmed connection failure.
        checks['real_API_budget']=count==1 and succeeded==1 and 1<=attempts<=6
    else:checks['real_API_budget']=False
    if a.formal_regression_run:
        regression=run_path(a.formal_regression_run)
        recovery=read_json(regression/'reports/world_current_code_recovery.json')
        evaluation=read_json(regression/'reward_model/reports/evaluation.json')
        exported=read_json(regression/'reports/export_current_code.json')
        checks['current_7GPU_world_recovery']=bool(recovery['passed'] and all(recovery['comparisons'].values()) and recovery['paid_API_requests']==0 and recovery['formal_conditions_consumed']==0)
        checks['current_distributed_full_Qwen_evaluation']=bool(evaluation['json_valid_rate']==1 and len(evaluation['rank_profiles'])==7 and evaluation['reload_adapter_tensors']['exact'] and evaluation['frozen_buffers_unchanged'] and all(row['images']==124 for row in evaluation['results']))
        checks['current_complete_inference_export']=bool(exported['passed'] and exported['frozen_patch_embedding_present'] and not exported['missing_keys'] and not exported['unexpected_keys'])
        evidence['current_complete_inference_export']=str(regression/'reports/export_current_code.json')
        evidence['current_7GPU_world_recovery']=str(regression/'reports/world_current_code_recovery.json')
        evidence['current_distributed_full_Qwen_evaluation']=str(regression/'reward_model/reports/evaluation.json')
        current_log=run_path(a.publish_to)/'logs/formal_code_regression.log' if a.publish_to else regression/'logs/formal_code_regression.log'
        checks['current_formal_controller_regression']=current_log.exists() and 'passed' in current_log.read_text() and not any(v in current_log.read_text() for v in (' FAILED ',' ERROR ',' failed,'))
        evidence['current_formal_controller_regression']=str(current_log)
        if a.publish_to and load_config(a.formal_config)['reward'].get('concurrency_backend')=='threads':
            threads=run_path(a.publish_to)/'reports/api_32_thread_stress.json'
            thread_report=read_json(threads) if threads.exists() else {}
            checks['current_32_thread_API']=bool(thread_report.get('backend')=='threads' and thread_report.get('peak_server_inflight')==32 and thread_report.get('cached_repeat_added_requests')==0 and thread_report.get('paid_API_requests')==0)
            evidence['current_32_thread_API']=str(threads)
    result={'schema':'rl-integration-gate-v1','passed':all(checks.values()),'software_acceptance_only':True,'formal_reward_quality_passed':False,'run_id':a.run_id,'checks':checks,'evidence':evidence,'server_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_code_hashes':{str(f.relative_to(ROOT)):sha256(f) for f in sorted((ROOT/'rl_posttrain').rglob('*.py'))},'known_formal_limitations':['Only one real teacher condition; RM quality/generalization is not established','Episode IDs absent: source/action/temporal components still require manual episode review','SAM2.1/Grounding DINO/ClipAttackNet extractor weights absent; no Move-Acc or Att-Acc quality numbers are claimed','The immutable real v1 teacher group is quarantined; v2 teacher protocol has not used any additional paid group','Exact D schedule verified only for local collector; released data collector timing unavailable']}
    result['real_API_groups']=count;result['API_client_attempts']=attempts
    result['validated_configs']={str(f.relative_to(ROOT)):validate(load_config(f))['config_hash'] for f in sorted((ROOT/'rl_posttrain/configs').glob('*.yaml'))}
    result['numerical_resume_limitation']=c_resume
    result['numerical_resume_resolution']=strict_world
    result['teacher_v2_real_API_quality_tested']=False
    result['runtime_artifact_hashes']={str(f.relative_to(ROOT)):sha256(f) for f in sorted((ROOT/'rl_posttrain').rglob('*')) if f.is_file() and f.suffix in ('.yaml','.txt','.sh','.in','.lock')}
    result['protected_model_files']=protected
    write_json(run/'reports/integration_gate.json',result);print({'passed':result['passed'],'failed_checks':[k for k,v in checks.items() if not v],'report':str(run/'reports/integration_gate.json')})
    if not result['passed']:raise SystemExit(1)
    if a.publish_to:
        target=run_path(a.publish_to)
        state=read_json(target/'plan/cursor.json');prepared=validate(read_json(target/'plan/config.json'),a.publish_to)
        progressed=bool(state['cursor'] or state['inflight'] or state['api_consumed'] or state['local_consumed'])
        if progressed and not a.refresh_progress:raise ValueError('Refreshing a progressed run requires --refresh-progress and current regression evidence')
        if progressed and (not a.formal_regression_run or not (target/'reports/integration_gate.json').exists()):raise ValueError('Existing progress requires its previous gate and current regression evidence')
        cursor_before=(target/'plan/cursor.json').read_bytes();order_before=(target/'plan/epoch_order.json').read_bytes()
        base=load_config(a.formal_config)
        for name in ('world','data','reward','reward_model'):
            if prepared[name]!=base[name]:raise ValueError('Prepared model/data/protocol differs from accepted configuration: '+name)
        published=dict(result,run_id=a.publish_to,evidence_run_id=a.run_id,evidence_gate_sha256=sha256(run/'reports/integration_gate.json'),prepared_config_hash=prepared['config_hash'])
        previous=target/'reports/integration_gate.json'
        if previous.exists():write_json(target/'provenance/software_gates'/(sha256(previous)+'.json'),read_json(previous))
        write_json(previous,published)
        if cursor_before!=(target/'plan/cursor.json').read_bytes() or order_before!=(target/'plan/epoch_order.json').read_bytes():raise AssertionError('Gate publication changed the immutable training ledger')
        # The same immutable world architecture/scale was measured in the evidence run.
        # RM inference capacity is measured again after the target run's formal B.
        write_json(target/'reports/world_full_profile.json',dict(world,profile_evidence_run_id=a.run_id))
        print({'published_software_gate':str(target/'reports/integration_gate.json'),'formal_training_started':False,'reward_quality_passed':False})
if __name__=='__main__':main()
