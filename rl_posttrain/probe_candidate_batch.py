"""Low-step full-resolution batch 1/2 comparison, outside the training ledger."""
import argparse
import time
from pathlib import Path
import torch
from .io import run_path,read_json,world_checkpoint,write_json
from .adapters.world_model import WorldAdapter
from .adapters.checkpoint import load
from .adapters.parameter_modes import optimizer
from .algorithms.policy_bank import PolicyBank

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-run',default='study_v1');p.add_argument('--output',required=True);a=p.parse_args()
    run=run_path(a.source_run);cp=world_checkpoint(run,'api');payload=torch.load(cp,map_location='cpu',weights_only=False);c=payload['config']
    adapter=WorldAdapter(c);bank=PolicyBank(adapter.dit);opt=optimizer(adapter.dit,c['nft'])
    load(cp,adapter.dit,opt,bank,c)
    group=next(g for g in sorted((run/'groups').glob('g-*')) if (g/'conditioning.safetensors').exists())
    values=adapter.condition_tensors(group);manifest=read_json(group/'manifest.json');seeds=[r['seed'] for r in manifest['candidates'][:2]]
    adapter.scheduler.set_timesteps(2,shift=c['rollout']['sigma_shift']);measurements={};outputs={}
    before={n:p.detach().cpu().clone() for n,p in bank.params.items()}
    with bank.use('old'):
        for batch in (1,2):
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.monotonic();items=[]
            for i in range(0,2,batch):items.append(adapter.sample(values['first'],values['context'],values['actions'],manifest['condition']['latent_frames'],seeds[i:i+batch]))
            torch.cuda.synchronize();output=torch.cat(items);assert torch.isfinite(output).all()
            assert torch.equal(output[:,:,:1],values['first'].expand(2,-1,-1,-1,-1))
            outputs[batch]=output.detach().cpu()
            measurements[str(batch)]={'seconds_for_two_candidates':time.monotonic()-tick,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                'shape':list(output.shape)}
    for n,param in bank.params.items():assert torch.equal(before[n],param.detach().cpu())
    delta=(outputs[1].float()-outputs[2].float()).abs()
    write_json(Path(a.output),{'real_model':True,'height':c['rollout']['height'],'width':c['rollout']['width'],
        'loaded_frames':manifest['condition']['loaded_frames'],'steps_for_controlled_test':2,
        'production_steps_unchanged':c['rollout']['steps'],'same_condition_and_seeds':True,
        'measurements':measurements,'batch2_speedup':measurements['1']['seconds_for_two_candidates']/measurements['2']['seconds_for_two_candidates'],
        'latent_mean_abs_delta':float(delta.mean()),'latent_max_abs_delta':float(delta.max()),
        'FP32_trainable_masters_unchanged':True,'formal_conditions_consumed':0,'API_requests':0,
        'limitation':'Short low-cost compute probe, not a 30-step end-to-end throughput or quality result; batch kernels may change floating-point results.'})
    print(a.output)

if __name__=='__main__':main()
