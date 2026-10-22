"""Create a C runtime configuration from the actual A state, without starting it."""
import argparse
import copy
import torch
import yaml
from .io import run_path,read_json,world_checkpoint,sha256,digest
from .timeslice import gpu_ids,runtime_config

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',default='study_v1');p.add_argument('--gpus',type=gpu_ids,required=True)
    p.add_argument('--generation-batch-size',type=int,default=1);p.add_argument('--reward-batch-size',type=int,default=1)
    p.add_argument('--conditions-per-batch',type=int,default=28);a=p.parse_args()
    if a.conditions_per_batch<1:raise ValueError('conditions-per-batch must be positive')
    run=run_path(a.run_id);cp=world_checkpoint(run,'api')
    if sha256(cp)!=read_json(cp.with_suffix('.json'))['sha256']:raise ValueError('A checkpoint identity mismatch')
    payload=torch.load(cp,map_location='cpu',weights_only=False);c=copy.deepcopy(payload['config'])
    c['stage']='local';c['rollout']['conditions_per_batch']=a.conditions_per_batch
    c=runtime_config(c,a.gpus,a.generation_batch_size,a.reward_batch_size)
    path=run/'plan'/f"formal_C_timeslice_{c['config_hash'][:16]}.yaml"
    content=yaml.safe_dump(c,sort_keys=False)
    if path.exists() and path.read_text()!=content:raise ValueError('Refusing to replace an existing C configuration')
    if not path.exists():path.write_text(content)
    print(path)
    print('Configuration only; formal ledger and C start control unchanged.')

if __name__=='__main__':main()
