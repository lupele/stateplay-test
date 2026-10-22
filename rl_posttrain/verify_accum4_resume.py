"""Real seven-rank, 124-image accum4 resume versus uninterrupted updates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import torch
from safetensors.torch import load_file
from .io import ROOT,run_path,read_json,write_json
from .distributed import tree_hash

def main():
    p=argparse.ArgumentParser();p.add_argument('--continuous-run',default='final_v2_rm_accum4_continuous');p.add_argument('--resume-run',default='final_v2_rm_accum4_resume');p.add_argument('--wait-pid',type=int);a=p.parse_args()
    continuous=run_path(a.continuous_run);resume=run_path(a.resume_run)
    if a.wait_pid:
        proc=Path('/proc')/str(a.wait_pid)
        if proc.exists() and a.continuous_run not in (proc/'cmdline').read_bytes().decode(errors='replace'):raise ValueError('Wait PID does not belong to the specified development run')
        while proc.exists():time.sleep(1)
    m=read_json(continuous/'reward_model/final/manifest.json');assert m['smoke'] and m['steps']==3
    start=continuous/'reward_model/checkpoints/step-00000001'
    if not (resume/'reward_model/final/manifest.json').exists():
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6';env['TOKENIZERS_PARALLELISM']='false';env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
        with (resume/'logs/resume_accum4.log').open('w') as log:
            subprocess.run([str(ROOT/'outputs/rl_posttrain/_envs/reward/bin/python'),'-m','torch.distributed.run','--standalone','--nproc_per_node=7','-m','rl_posttrain.reward_model.train','--config','rl_posttrain/configs/reward_resume_v1_audit.yaml','--run-id',a.resume_run,'--smoke','--max-optimizer-steps','3','--resume',str(start),'--audit-replay'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    trace_checks=[]
    for rank in range(7):
        def trace(root):return [json.loads(x) for x in (root/f'metrics/replay_rank_{rank}.jsonl').read_text().splitlines() if json.loads(x)['step']>=2]
        x,y=trace(continuous),trace(resume)
        inputs_equal=[{k:v for k,v in z.items() if k not in ('loss','gradient_norm','gradient_hash')} for z in x]==[{k:v for k,v in z.items() if k not in ('loss','gradient_norm','gradient_hash')} for z in y]
        trace_checks.append({'rank':rank,'inputs_equal':inputs_equal,'gradient_hashes_equal':[z.get('gradient_hash') for z in x if 'gradient_hash' in z]==[z.get('gradient_hash') for z in y if 'gradient_hash' in z],'max_loss_delta':max(abs(u['loss']-v['loss']) for u,v in zip(x,y) if 'loss' in u)})
    differences=[]
    for step in (2,3):
        cp=f'reward_model/checkpoints/step-{step:08d}'
        x=load_file(continuous/cp/'adapter/adapter_model.safetensors');y=load_file(resume/cp/'adapter/adapter_model.safetensors')
        delta=max(float((x[k]-y[k]).abs().max()) for k in x)
        sx=torch.load(continuous/cp/'training_state.pt',map_location='cpu',weights_only=False);sy=torch.load(resume/cp/'training_state.pt',map_location='cpu',weights_only=False)
        assert sx['accumulation']==sy['accumulation']==4 and len(sx['rank_rng'])==len(sy['rank_rng'])==7
        differences.append({'step':step,'adapter_max_abs_delta':delta,'optimizer_hash_equal':tree_hash(sx['optimizer'])==tree_hash(sy['optimizer']),'scheduler_equal':sx['scheduler']==sy['scheduler'],'rank_rng_equal':tree_hash(sx['rank_rng'])==tree_hash(sy['rank_rng']),'buffers_equal':tree_hash(sx.get('buffers'))==tree_hash(sy.get('buffers')),'backend_equal':sx.get('backend')==sy.get('backend')})
    kernel_checks=[read_json(continuous/f'reports/kernel_configs_rank_{r}.json')==read_json(resume/f'reports/kernel_configs_rank_{r}.json') for r in range(7)]
    exact_traces=all(t['inputs_equal'] and t['gradient_hashes_equal'] and t['max_loss_delta']==0 for t in trace_checks)
    exact_states=all(d['adapter_max_abs_delta']==0 and d['optimizer_hash_equal'] and d['scheduler_equal'] and d['rank_rng_equal'] and d['buffers_equal'] and d['backend_equal'] for d in differences)
    report={'kind':'real_Qwen_7GPU_full_group_accum4_resume','accumulation':4,'ranks':7,'image_count':124,'start_step':1,'final_step':3,'trace_checks':trace_checks,'checkpoint_checks':differences,'kernel_choices_equal':kernel_checks,'continuous_run':str(continuous),'resume_run':str(resume),'additional_API_groups':0,'passed':exact_traces and exact_states and all(kernel_checks),'bitwise_equal':exact_traces and exact_states}
    write_json(run_path('smoke_v1')/'reports/reward_accum4_resume_v2.json',report);print(report,flush=True)
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':main()
