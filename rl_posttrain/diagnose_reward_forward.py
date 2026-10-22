"""Compare reconstructed full Qwen runtimes without an optimizer or paid API."""
import gc
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from .config import load_config
from .io import run_path,write_json
from .distributed import tensor_hash,tree_hash
from .reward_model.model import load_model,move,vision_features,cached_vision,suffix_loss
from .reward_model.collator import GroupCollator
from .reward_model.replay_dropout import seed_microbatch

def main():
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    torch.set_num_threads(6);torch.cuda.set_device(local);dist.init_process_group('nccl')
    c=load_config('rl_posttrain/configs/reward_resume_v1_audit.yaml')
    root=run_path('smoke_v1');cp=run_path('final_v5_rm_accum4_continuous')/'reward_model/checkpoints/step-00000001'
    state=torch.load(cp/'training_state.pt',map_location='cpu',weights_only=False)
    expected=load_file(cp/'adapter/adapter_model.safetensors')
    group=root/'groups/g-9000000-train-0000000';result={}
    for kind in ('fresh_then_copy','adapter_reload'):
        torch.manual_seed(c['plan']['seed']+rank)
        model,processor,_=load_model(c,f'cuda:{local}',adapter_path=cp/'adapter' if kind=='adapter_reload' else None,training=True)
        wrapped=DistributedDataParallel(model,device_ids=[local],broadcast_buffers=False)
        for _,buffer in sorted(model.named_buffers()):dist.broadcast(buffer,src=0)
        if kind=='fresh_then_copy':set_peft_model_state_dict(model,expected)
        for name,buffer in model.named_buffers():buffer.copy_(state['buffers'][name].to(buffer.device))
        hashes={n:tensor_hash(p) for n,p in model.named_parameters()}
        hashes.update({'buffer:'+n:tensor_hash(b) for n,b in model.named_buffers()})
        settings={n:{'training':m.training,'class':type(m).__name__,'scaling':getattr(m,'scaling',None)} for n,m in model.named_modules()}
        collator=GroupCollator(processor,c);index=rank;epoch=1
        seed_microbatch(model,c['plan']['seed'],epoch,index)
        values,meta=collator(group,training=True,permutation_seed=epoch+index+c['plan']['seed'])
        features=vision_features(model,values,root/'reward_model/feature_cache'/(meta['input_hash']+'.safetensors'))
        inputs=move(values,f'cuda:{local}')
        with torch.no_grad(),wrapped.no_sync(),cached_vision(model,features),torch.autocast('cuda',dtype=torch.bfloat16):loss=float(suffix_loss(wrapped,inputs))
        result[kind]={'parameters_and_buffers':hashes,'module_settings_hash':tree_hash(settings),'loss':loss,'input_hash':meta['input_hash'],'features_hash':tree_hash(features)}
        write_json(root/f'reports/Qwen_forward_reconstruction_rank_{rank}.json',result)
        del inputs,features,values,wrapped,model,processor;gc.collect();torch.cuda.empty_cache();dist.barrier()
    all_hashes=[None]*7;dist.all_gather_object(all_hashes,tree_hash(result['fresh_then_copy']['parameters_and_buffers']))
    result['all_rank_fresh_state_equal']=len(set(all_hashes))==1
    result['state_differences']=[n for n,h in result['fresh_then_copy']['parameters_and_buffers'].items() if h!=result['adapter_reload']['parameters_and_buffers'][n]]
    write_json(root/f'reports/Qwen_forward_reconstruction_rank_{rank}.json',result)
    print(rank,'state differences',result['state_differences'],'losses',result['fresh_then_copy']['loss'],result['adapter_reload']['loss'],flush=True)
    dist.destroy_process_group()

if __name__=='__main__':main()
