"""Unified entry. Preparing a plan does not authorize or start formal training."""
import argparse
from pathlib import Path
import os
import subprocess
import sys
import time
import httpx
from .io import ROOT,run_path,read_json,write_json,reward_path,reward_checkpoint,now
from .config import load_config

WORLD=str(ROOT/"outputs/rl_posttrain/_envs/world/bin/python")
REWARD=str(ROOT/"outputs/rl_posttrain/_envs/reward/bin/python")

def execute(command,run,name,gpus=None,allow_failure=False):
    env=os.environ.copy();env["PYTHONUNBUFFERED"]="1";env["TOKENIZERS_PARALLELISM"]="false";env["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
    if gpus is not None:env["CUDA_VISIBLE_DEVICES"]=",".join(map(str,gpus))
    (run/"logs").mkdir(parents=True,exist_ok=True)
    with (run/"logs"/(name+".log")).open("a") as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        record=run/'plan/processes'/(name+'.json')
        try:tick=Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
        except FileNotFoundError:tick=None
        info={'pid':process.pid,'start_tick':tick,'command':command,'active':True,'started_at':now(),'gpus':gpus}
        write_json(record,info)
        while True:
            try:
                code=process.wait(timeout=15)
                break
            except subprocess.TimeoutExpired:
                info['heartbeat_at']=now();write_json(record,info)
        info.update(active=False,returncode=code,finished_at=now());write_json(record,info)
    if code and not allow_failure:raise RuntimeError(f"{name} failed; see {run/'logs'/(name+'.log')}")
    return code

def torchrun(module,args,gpus):
    return [WORLD,"-m","torch.distributed.run","--standalone","--nproc_per_node",str(len(gpus)),"-m",module,*args]

def start_server(c,run,run_id,smoke,gpus):
    env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=",".join(map(str,gpus));env["TOKENIZERS_PARALLELISM"]="false"
    log=(run/"logs/reward_server.log").open("a")
    import yaml
    serving=run/'plan/reward_serving_config.yaml';serving.parent.mkdir(parents=True,exist_ok=True);serving.write_text(yaml.safe_dump(c,sort_keys=False))
    cmd=[REWARD,"-u","-m","rl_posttrain.reward_model.serve","--config",str(serving),"--run-id",run_id,"--shards",str(len(gpus))]
    if smoke:cmd.append("--smoke")
    server=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    try:tick=Path(f'/proc/{server.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:tick=None
    write_json(run/'plan/reward_server_process.json',{'pid':server.pid,'start_tick':tick,'run_id':run_id,'command':cmd,'active':True})
    deadline=time.monotonic()+300
    while time.monotonic()<deadline:
        if server.poll() is not None:log.close();raise RuntimeError("Reward server failed to load; inspect its log")
        try:
            health=httpx.get(c["reward_model"]["inference_url"]+"/health",timeout=2,trust_env=False).raise_for_status().json()
            if health.get("pid")==server.pid and health["version"]==read_json(reward_checkpoint(run)/"manifest.json")["version"]:return server,log
        except (httpx.HTTPError,ValueError):pass
        time.sleep(1)
    server.terminate();log.close();raise RuntimeError("Reward server did not become ready")

def execute_formal(c,run,run_id,config,auto_extend=False,wait_human=False):
    from .formal import supervise
    if run.name!=run_id:raise ValueError('Formal run identity mismatch')
    return supervise(c,run,config,execute,WORLD,REWARD,auto_extend,wait_human)

def main():
    p=argparse.ArgumentParser();p.add_argument("--run-id",required=True);p.add_argument('--config',default=str(ROOT/'rl_posttrain/configs/api_b_only.yaml'));mode=p.add_mutually_exclusive_group(required=True);mode.add_argument("--development-smoke",action="store_true");mode.add_argument("--prepare-only",action="store_true");mode.add_argument("--execute-plan",action="store_true");p.add_argument("--api-group-limit",type=int);p.add_argument("--auto-extend",action="store_true");p.add_argument("--wait-for-human-review",action="store_true");a=p.parse_args()
    run=run_path(a.run_id);config=a.config;c=load_config(config)
    if a.prepare_only:
        execute([WORLD,"-m","rl_posttrain.preflight","--config",config,"--run-id",a.run_id],run,"preflight")
        prepared=read_json(run/'plan/stage_state.json')
        write_json(run/"reports/execution_plan.json",{"stage_A_conditions":prepared['A_budget'],"stage_B_new_conditions":0,"stage_C_conditions":prepared['C_budget'],"formal_training_started":False,"API_extension_block":280,"API_max_conditions":4000,"formal_commands":["bash rl_posttrain/scripts/run_api.sh --run-id "+a.run_id,"bash rl_posttrain/scripts/train_rm.sh --run-id "+a.run_id,"bash rl_posttrain/scripts/run_local.sh --run-id "+a.run_id]});print("Prepared plan only",run);return
    if a.development_smoke:
        if a.api_group_limit!=1:raise ValueError("Development smoke requires --api-group-limit 1")
        execute([WORLD,"-m","pytest","rl_posttrain/tests","tests/test_latent_state.py","-q"],run,"all_tests")
        if not (run/"reports/world_full_profile.json").exists():execute([WORLD,"-m","rl_posttrain.profile","--config",config,"--run-id",a.run_id,"--component","world","--full-group"],run,"world_full_profile",[0])
        # This reads the same permanent cache if the single controlled request already succeeded.
        api_config=str(ROOT/'rl_posttrain/configs/legacy_api_v1.yaml') if (run/'reports/real_api_acceptance.json').exists() else config
        execute([WORLD,"-m","rl_posttrain.controlled_api","--config",api_config,"--run-id",a.run_id,"--api-group-limit","1"],run,"controlled_api")
        import yaml,copy
        selected_api=load_config(api_config);smoke_values=load_config(ROOT/'rl_posttrain/configs/smoke.yaml')
        for name in ('world','data','reward','reward_model'):smoke_values[name]=copy.deepcopy(selected_api[name])
        smoke_values['nft']['mode']=selected_api['nft']['mode']
        smoke_path=run/'plan/development_smoke_config.yaml';smoke_path.parent.mkdir(parents=True,exist_ok=True);smoke_path.write_text(yaml.safe_dump(smoke_values,sort_keys=False));smoke_config=str(smoke_path)
        if not (run/"world/api/final/world.pt").exists():execute(torchrun("rl_posttrain.train_api",["--config",smoke_config,"--run-id",a.run_id,"--smoke","--max-optimizer-steps","3","--scorer","fake"],list(range(7))),run,"api_smoke",list(range(7)))
        reward_smoke_config=str(ROOT/'rl_posttrain/configs/reward_resume_v1_audit.yaml') if selected_api['reward']['protocol'].startswith('mechanics-reward-v1') else str(ROOT/'rl_posttrain/configs/rm_qwen35.yaml')
        if not (reward_checkpoint(run)/"manifest.json").exists():execute([REWARD,"-m","torch.distributed.run","--standalone","--nproc_per_node","7","-m","rl_posttrain.reward_model.train","--config",reward_smoke_config,"--run-id",a.run_id,"--smoke","--max-optimizer-steps","3"],run,"reward_smoke",list(range(7)))
        if not (run/'reward_model/reports/evaluation.json').exists():execute([REWARD,"-m","rl_posttrain.reward_model.evaluate","--config",reward_smoke_config,"--run-id",a.run_id,"--check-reload","--smoke"],run,"reward_evaluate",[6])
        execute([WORLD,"-m","rl_posttrain.profile","--config",config,"--run-id",a.run_id,"--component","joint","--full-group"],run,"joint_profile")
        if not (run/"world/local/final/world.pt").exists():
            server,log=start_server(smoke_values,run,a.run_id,True,[6])
            try:execute(torchrun("rl_posttrain.train_local",["--config",smoke_config,"--run-id",a.run_id,"--smoke","--max-optimizer-steps","3","--scorer","local"],list(range(6))),run,"local_smoke",list(range(6)))
            finally:server.terminate();server.wait(timeout=30);log.close()
        print("Three-stage development smoke finished. Formal reward quality remains a separate gate.");return
    # This path is reached only by an explicit formal caller, never by prepare/smoke.
    execute_formal(c,run,a.run_id,config,a.auto_extend,a.wait_for_human_review)
if __name__=="__main__":main()
