"""Check checkpoint tensors and CUDA RNG before/after DDP construction on all ranks."""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from peft import get_peft_model_state_dict
from safetensors.torch import load_file
from .config import load_config
from .reward_model.model import load_model
from .adapters.checkpoint import rng_state,restore_rng
from .distributed import tree_hash
from .io import run_path,write_json

def main():
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);torch.set_num_threads(6);torch.cuda.set_device(local);dist.init_process_group('nccl')
    c=load_config('rl_posttrain/configs/reward_resume_v1_audit.yaml');cp=run_path('final_v3_rm_accum4_continuous')/'reward_model/checkpoints/step-00000001'
    fresh=os.environ.get('RL_DIAG_FRESH')=='1';torch.manual_seed(c['plan']['seed']+rank)
    model,_,audit=load_model(c,f'cuda:{local}',adapter_path=None if fresh else cp/'adapter',training=True)
    expected=load_file(cp/'adapter/adapter_model.safetensors') if not fresh else {k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model,save_embedding_layers=False).items()}
    before=get_peft_model_state_dict(model,save_embedding_layers=False)
    before_delta=max(float((expected[k]-before[k].detach().cpu()).abs().max()) for k in expected)
    small_before={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad and p.numel()<=1024}
    parameter_order=[n for n,p in model.named_parameters()]
    wrapped=DistributedDataParallel(model,device_ids=[local],broadcast_buffers=False)
    after=get_peft_model_state_dict(model,save_embedding_layers=False)
    after_delta=max(float((expected[k]-after[k].detach().cpu()).abs().max()) for k in expected)
    hashes=[None]*7;dist.all_gather_object(hashes,tree_hash(after))
    small_diffs={n:float((p.detach().cpu()-small_before[n]).abs().max()) for n,p in model.named_parameters() if n in small_before}
    state=torch.load(cp/'training_state.pt',map_location='cpu',weights_only=False);restore_rng(state['rank_rng'][rank]);restored=rng_state()
    report={'rank':rank,'fresh':fresh,'post_DDP_adapter_hashes':hashes,'post_DDP_equal':len(set(hashes))==1,'adapter_before_DDP_max_delta':before_delta,'adapter_after_DDP_max_delta':after_delta,'restored_rng_equal':tree_hash(restored)==tree_hash(state['rank_rng'][rank]),'trainable_parameter_order':[n for n,p in model.named_parameters() if p.requires_grad],'full_parameter_order':parameter_order,'frozen_small_params_changed':{n:v for n,v in small_diffs.items() if v},'CUDA_current_device':torch.cuda.current_device()}
    write_json(run_path('smoke_v1')/f'reports/reward_DDP_{"fresh" if fresh else "init"}_rank_{rank}.json',report)
    print({k:v for k,v in report.items() if k not in ('trainable_parameter_order','full_parameter_order')},flush=True);dist.destroy_process_group()
if __name__=='__main__':main()
