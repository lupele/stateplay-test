"""Isolated four-GPU acceptance; never advances study_v1 or calls a paid API."""
import argparse
import copy
import os
import shutil
from pathlib import Path
import torch
import yaml
from .io import ROOT,run_path,read_json,write_json,sha256,digest,atomic_torch,reward_checkpoint,world_checkpoint
from .data.sampler import Ledger

def prepare(run_id):
    if not run_id.startswith('stagec_test_'):raise ValueError('A distinct stagec_test_ run is required')
    run=run_path(run_id)
    if (run/'plan/cursor.json').exists():raise ValueError('Test already exists; resume it instead of resetting')
    source=run_path('study_v1');original=world_checkpoint(source,'api')
    metadata=read_json(original.with_suffix('.json'))
    if sha256(original)!=metadata['sha256']:raise ValueError('Original A checkpoint checksum mismatch')
    payload=torch.load(original,map_location='cpu',weights_only=False);c=copy.deepcopy(payload['config'])
    c['development_smoke']=True;c['rollout'].update(height=128,width=224,requested_frames=9,steps=2,conditions_per_batch=8)
    c['nft']['reward_mode']='standard'
    c['topology']={'strategy':'timeslice','world_gpus':[0,1,2,3],'reward_gpus':[0,1,2,3]}
    c['stage_c']={'generation_batch_size':2,'reward_batch_size':2}
    c['config_hash']=digest({k:v for k,v in c.items() if k!='config_hash'})
    ids=set(read_json(source/'plan/epoch_order.json')['order'][:8])
    catalog=read_json(source/'conditions/catalog.json');catalog['records']=[r for r in catalog['records'] if r['condition_id'] in ids]
    assert len(catalog['records'])==8
    write_json(run/'conditions/catalog.json',catalog);Ledger(run/'plan',catalog['records'],c['plan']['seed'])
    cp=run/'world/api/final/isolated_A_state.pt';payload['config']=c
    payload['state'].update(batch_id=0,update_cursor=0,batch_done=True,smoke=True,stage='api')
    # Optimizer step, optimizer tensors, live/old/ref policy and RNG come from A.
    atomic_torch(cp,payload)
    write_json(cp.with_suffix('.json'),{'sha256':sha256(cp),'schema':payload['schema'],'state':payload['state'],
        'mode':c['nft']['mode'],'config_hash':c['config_hash'],'sft_sha256':payload['sft_sha256']})
    write_json(cp.parent/'current.json',{'checkpoint':str(cp),'optimizer_step':payload['state']['optimizer_step']})
    selected=run/'reward_model/selected';selected.parent.mkdir(parents=True,exist_ok=True)
    selected.symlink_to(reward_checkpoint(source),target_is_directory=True)
    path=run/'plan/test_config.yaml';path.write_text(yaml.safe_dump(c,sort_keys=False))
    write_json(run/'reports/test_scope.json',{'software_smoke':True,'formal_quality_claim':False,
        'source_A_checkpoint':str(original),'source_A_sha256':metadata['sha256'],
        'source_A_optimizer_step':payload['state']['optimizer_step'],'formal_conditions_consumed':0,'paid_API_requests':0,
        'test_conditions':8,'candidates':32,'generation_batch_size':2,'reward_batch_size':2,
        'sampling_for_low_cost_test':c['rollout'],'NFT_difference_for_test':'guard disabled to exercise actual gradients even on deliberately short low-resolution videos',
        'source_train_conditions_already_consumed_in_A':sorted(ids),'test_ledger_isolated':True})
    print(path)

def report(run_id):
    run=run_path(run_id);ledger=read_json(run/'plan/cursor.json');original=read_json(run/'reports/test_scope.json')
    final=world_checkpoint(run,'local');payload=torch.load(final,map_location='cpu',weights_only=False)
    src=torch.load(run/'world/api/final/isolated_A_state.pt',map_location='cpu',weights_only=False)
    groups=list((run/'groups').glob('g-*'));candidates=[x for g in groups for x in read_json(g/'manifest.json')['candidates']]
    scores=[x for g in groups for x in (g/'local_reward').glob('*.json')]
    delta=sum(float((payload['live'][n]-src['live'][n]).square().sum()) for n in src['live'])**.5
    assert ledger['cursor']==ledger['local_consumed']==8 and ledger['api_consumed']==0 and not ledger['inflight']
    assert len(candidates)==32 and all(x['complete'] for x in candidates) and len(scores)==8 and delta>0
    assert all(read_json(p)['request_batch_size']==2 for p in scores)
    assert all(len(read_json(p)['scores'])==4 and read_json(p)['image_count']==124 for p in scores)
    sync=read_json(run/'reports/local_ddp_sync.json');assert sync['ranks']==4 and sync['equal']
    steps=[read_json_line for read_json_line in map(__import__('json').loads,(run/'metrics/local_steps.jsonl').read_text().splitlines())]
    expected=list(range(src['state']['optimizer_step']+1,src['state']['optimizer_step']+5))
    assert [row['optimizer_step'] for row in steps]==expected
    partial_path=next((run/'world/local/checkpoints').glob(f"step-{expected[0]:08d}-batch-000000-cursor-000001.pt"))
    partial=torch.load(partial_path,map_location='cpu',weights_only=False)
    assert partial['state']['update_cursor']==1 and not partial['state']['batch_done']
    assert partial['policies']['version']==src['policies']['version']
    assert all(torch.equal(partial['policies']['old'][k],src['policies']['old'][k]) for k in src['policies']['old'])
    assert payload['policies']['version']==src['policies']['version']+1
    assert all(torch.equal(payload['policies']['ref'][k],src['policies']['ref'][k]) for k in src['policies']['ref'])
    write_json(run/'reports/acceptance.json',{**original,'passed':True,'real_models':True,'physical_GPU_type':'NVIDIA L20, not A100',
        'gpus':[0,1,2,3],'optimizer_step_start':src['state']['optimizer_step'],'optimizer_step_end':payload['state']['optimizer_step'],
        'trainable_parameter_delta_l2':delta,'reference_policy_unchanged':True,'ddp_parameters_equal':True,
        'optimizer_steps':expected,'duplicate_committed_updates':0,'optimizer_boundary_resume_tested':True,
        'old_policy_fixed_across_partial_update':True,'EMA_applied_once_after_full_batch':True,
        'complete_requests':len(scores),'four_scores_per_request':True,'ledger':ledger,
        'final_checkpoint':str(final),'progress':str(run/'progress/overall.json')})
    print(run/'reports/acceptance.json')

def prepare_full_score(run_id):
    """Read-only links to already consumed A assets; no new videos or conditions."""
    if not run_id.startswith('stagec_test_'):raise ValueError('A distinct test run is required')
    run=run_path(run_id);source=run_path('study_v1')
    if (run/'plan/cursor.json').exists():raise ValueError('Existing score test must be resumed')
    payload=torch.load(world_checkpoint(source,'api'),map_location='cpu',weights_only=False);c=payload['config']
    source_groups=sorted((source/'groups').glob('g-*'))[:8]
    records=[read_json(g/'condition.json') for g in source_groups]
    ledger=Ledger(run/'plan',records,c['plan']['seed']);reservation=ledger.reserve(8,'local',payload['policies']['version'])
    by_id={read_json(g/'condition.json')['condition_id']:g for g in source_groups}
    for index,cid in enumerate(reservation['condition_ids']):
        src=by_id[cid];dst=run/'groups'/f'g-{index:07d}-{cid}';dst.mkdir(parents=True)
        for name in ['initial.png','reference.mp4','actions_raw.parquet']:os.link(src/name,dst/name)
        shutil.copytree(src/'frames',dst/'frames',copy_function=os.link)
        m=read_json(src/'manifest.json')
        for candidate in m['candidates']:
            directory=dst/'candidates'/candidate['video_id'];directory.mkdir(parents=True)
            os.link(src/'candidates'/candidate['video_id']/'video.mp4',directory/'video.mp4')
        def rebase(value):
            if isinstance(value,str):return value.replace(str(src),str(dst))
            if isinstance(value,list):return [rebase(v) for v in value]
            if isinstance(value,dict):return {k:rebase(v) for k,v in value.items()}
            return value
        m=rebase(m);m['group_id']=dst.name;m['frame_asset_hash']=digest(m['frame_assets'])
        write_json(dst/'manifest.json',m);write_json(dst/'condition.json',read_json(src/'condition.json'))
    ledger.phase('generated');(run/'reward_model').mkdir();(run/'reward_model/selected').symlink_to(reward_checkpoint(source),target_is_directory=True)
    path=run/'plan/test_config.yaml';path.write_text(yaml.safe_dump(c,sort_keys=False))
    write_json(run/'reports/test_scope.json',{'real_complete_124_image_requests':True,'formal_sampling_assets':True,
        'source_groups':[str(g) for g in source_groups],'formal_conditions_consumed':0,'new_videos_generated':0,
        'API_requests':0,'test_ledger_only':True,'read_only_asset_links':True})
    print(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--prepare-full-score',action='store_true');a=p.parse_args()
    if a.prepare_full_score:prepare_full_score(a.run_id)
    elif a.prepare:prepare(a.run_id)
    else:report(a.run_id)
if __name__=='__main__':main()
