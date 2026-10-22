"""Real six-rank C replay: compare all draws, anchors, gradients and EMA."""
import argparse
import json
import shutil
from pathlib import Path
import torch
from .io import ROOT,run_path,read_json,write_json
from .config import load_config
from .orchestrate import start_server,execute,torchrun
from .distributed import tree_hash

def main():
    p=argparse.ArgumentParser();p.add_argument('--report-run',default='smoke_v1');a=p.parse_args();source=run_path(a.report_run)
    c=load_config(ROOT/'rl_posttrain/configs/smoke.yaml');cp=source/'world/local/checkpoints/step-00000004-batch-000001-cursor-000001.pt'
    order=read_json(source/'plan/epoch_order.json')['order'];runs=[run_path('final_v2_C_continuous'),run_path('final_v2_C_resume')]
    for r in runs:
        (r/'logs').mkdir(parents=True,exist_ok=True);(r/'plan').mkdir(exist_ok=True);(r/'groups').mkdir(exist_ok=True);(r/'conditions').mkdir(exist_ok=True);(r/'reward_model').mkdir(exist_ok=True)
        for destination,original in [(r/'conditions/catalog.json',source/'conditions/catalog.json'),(r/'conditions/anchors',source/'conditions/anchors'),(r/'reward_model/final',source/'reward_model/final')]:
            if not destination.exists():destination.symlink_to(original,target_is_directory=original.is_dir())
        for group in (source/'groups').glob('g-*'):
            target=r/'groups'/group.name
            if not target.exists():target.symlink_to(group,target_is_directory=True)
        if not (r/'plan/cursor.json').exists():
            shutil.copy2(source/'plan/epoch_order.json',r/'plan/epoch_order.json');shutil.copy2(source/'plan/batch-000001-normalization.json',r/'plan/batch-000001-normalization.json')
            write_json(r/'plan/cursor.json',{'cursor':1,'api_consumed':1,'local_consumed':0,'batch':1,'api_batches':1,'inflight':{'batch':1,'stage':'local','condition_ids':[order[1]],'old_version':1,'phase':'updating','update_cursor':1}})
    server,log=start_server(c,runs[0],runs[0].name,True,[6])
    base=['--config','rl_posttrain/configs/smoke.yaml','--smoke','--max-optimizer-steps','2','--scorer','local','--allow-legacy-rendering','--audit-replay']
    try:
        if not (runs[0]/'world/local/final/world.pt').exists():execute(torchrun('rl_posttrain.train_local',base+['--run-id',runs[0].name,'--resume',str(cp)],list(range(6))),runs[0],'continuous',[0,1,2,3,4,5])
        if not (runs[1]/'world/local/checkpoints/latest.json').exists():
            code=execute(torchrun('rl_posttrain.train_local',base+['--run-id',runs[1].name,'--resume',str(cp),'--interrupt-after-optimizer-step','5'],list(range(6))),runs[1],'interrupted_step5',[0,1,2,3,4,5],allow_failure=True)
            if code==0:raise AssertionError('Expected injected optimizer-boundary interruption')
        if not (runs[1]/'world/local/final/world.pt').exists():
            latest=read_json(runs[1]/'world/local/checkpoints/latest.json')['checkpoint']
            execute(torchrun('rl_posttrain.train_local',base+['--run-id',runs[1].name,'--resume',latest],list(range(6))),runs[1],'resume_step5_to6',[0,1,2,3,4,5])
    finally:server.terminate();server.wait(timeout=30);log.close()
    traces=[]
    for rank in range(6):
        x=[json.loads(s) for s in (runs[0]/f'metrics/replay_rank_{rank}.jsonl').read_text().splitlines()];y=[json.loads(s) for s in (runs[1]/f'metrics/replay_rank_{rank}.jsonl').read_text().splitlines()]
        keys=('optimizer_step','batch','window','old_version','visits','anchor')
        draws_equal=[{k:z[k] for k in keys if k in z} for z in x]==[{k:z[k] for k in keys if k in z} for z in y]
        traces.append({'rank':rank,'samples_timestep_noise_anchor_equal':draws_equal,'gradient_hashes_equal':[z.get('gradient_hash') for z in x]==[z.get('gradient_hash') for z in y],'EMA_equal':[z for z in x if z.get('event')=='ema']==[z for z in y if z.get('event')=='ema']})
    sx=torch.load(runs[0]/'world/local/final/world.pt',map_location='cpu',weights_only=False);sy=torch.load(runs[1]/'world/local/final/world.pt',map_location='cpu',weights_only=False)
    maximum=max(float((sx['live'][k]-sy['live'][k]).abs().max()) for k in sx['live'])
    report={'kind':'real_C_stable_reduction_optimizer_boundary_recovery','ranks':6,'start_step':4,'interrupted_step':5,'final_step':6,'traces':traces,'max_parameter_abs_difference':maximum,'optimizer_equal':tree_hash(sx['optimizer'])==tree_hash(sy['optimizer']),'policy_bank_equal':tree_hash(sx['policies'])==tree_hash(sy['policies']),'cursor_equal':read_json(runs[0]/'plan/cursor.json')['cursor']==read_json(runs[1]/'plan/cursor.json')['cursor']==2,'paid_API_groups':0,'legacy_group_replay':'Explicit software-only --allow-legacy-rendering; formal ingestion remains blocked','passed':maximum<=2e-6 and all(t['samples_timestep_noise_anchor_equal'] and t['EMA_equal'] for t in traces),'bitwise_equal':maximum==0,'source_checkpoint':str(cp),'runs':[str(r) for r in runs]}
    write_json(source/'reports/world_resume_diagnosis_v2.json',report);print(report,flush=True)
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':main()
