import argparse
import copy
import os
from pathlib import Path
import random
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import get_linear_schedule_with_warmup
from .dataset import build_dataset
from .collator import GroupCollator
from .model import load_model,base_model,move,vision_features,cached_vision,suffix_loss,frozen_hash,score
from ..config import load_config
from ..io import run_path,write_json,read_json,atomic_torch,append_jsonl,sha256,digest,reward_path
from ..adapters.checkpoint import rng_state,restore_rng
from ..distributed import gradient_mean,gradients_hash,tensor_hash,tree_hash
from .replay_dropout import seed_microbatch

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--run-id",required=True);p.add_argument("--smoke",action="store_true");p.add_argument("--max-optimizer-steps",type=int);p.add_argument("--resume");p.add_argument("--interrupt-after-step",type=int);p.add_argument("--smoke-accumulation",type=int);p.add_argument('--audit-replay',action='store_true');a=p.parse_args()
    c=load_config(a.config);run=run_path(a.run_id);rm=reward_path(run);data=build_dataset(run,config=c,allow_legacy=a.smoke)
    from ..reproducibility import configure
    backend=configure();backend['reward_dropout']='cursor-seeded-projection-local-v1'
    from .kernel_replay import VERSION,selected_configs
    backend['reward_kernels']=VERSION
    if not a.smoke:
        from ..train_world import formal_gate
        formal_gate(run,"api")
        if c['reward']['protocol']!='mechanics-reward-v2-24x832-124images' or c['data'].get('split_policy')!='source-action-temporal-v2':raise ValueError('Formal RM training requires the current v2 teacher and source leakage protocol')
    if a.smoke and not a.max_optimizer_steps:raise ValueError("Reward smoke needs a step cap")
    groups=data["groups"][:1] if a.smoke else data["splits"]["train"]
    if not groups:raise ValueError("Training split is empty")
    rank=int(os.environ.get("RANK",0));size=int(os.environ.get("WORLD_SIZE",1));local=int(os.environ.get("LOCAL_RANK",0));torch.cuda.set_device(local)
    torch.set_num_threads(6);torch.manual_seed(c["plan"]["seed"]+rank)
    if size>1:dist.init_process_group("nccl")
    model,processor,audit=load_model(c,f"cuda:{local}",adapter_path=Path(a.resume)/"adapter" if a.resume else None,training=True)
    wrapped=DistributedDataParallel(model,device_ids=[local],broadcast_buffers=False) if size>1 else model
    # Derived nonpersistent RoPE buffers are not part of a PEFT adapter.
    # Synchronize them once and preserve them explicitly at optimizer boundaries.
    if size>1:
        for name,buffer in sorted(model.named_buffers()):dist.broadcast(buffer,src=0)
    if a.audit_replay:write_json(run/f'reports/initial_state_rank_{rank}.json',{'adapter_hash':tree_hash({n:p for n,p in model.named_parameters() if p.requires_grad}),'frozen_small_hash':tree_hash({n:p for n,p in model.named_parameters() if not p.requires_grad and p.numel()<=1024}),'parameter_order':[n for n,p in model.named_parameters() if p.requires_grad]})
    params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=c["reward_model"]["learning_rate"],weight_decay=c["reward_model"]["weight_decay"])
    accum=a.smoke_accumulation if a.smoke and a.smoke_accumulation else c["reward_model"]["accumulation"]
    if a.smoke_accumulation and not a.smoke:raise ValueError("Accumulation override is development-only")
    steps_per_epoch=math_ceil(len(groups)/(size*accum))
    total=a.max_optimizer_steps if a.smoke else steps_per_epoch*c["reward_model"]["epochs"]
    scheduler=get_linear_schedule_with_warmup(opt,0 if a.smoke else max(1,int(total*c["reward_model"]["warmup_ratio"])),total)
    step=0
    if a.resume:
        state=torch.load(Path(a.resume)/"training_state.pt",map_location="cpu",weights_only=False)
        if state["dataset_hash"]!=digest(data["groups"]):raise ValueError("Reward dataset changed during resume")
        for section in ('world','reward','reward_model'):
            if state['config'][section]!=c[section]:raise ValueError('Reward resume configuration changed: '+section)
        if state.get('world_size',len(state.get('rank_rng',[])) or size)!=size:raise ValueError('Reward resume topology changed; update schedule must remain identical')
        if state.get('total_steps',total)!=total:raise ValueError('Reward resume total scheduler steps changed')
        if state.get('backend',backend)!=backend:raise ValueError('Reward resume compute backend changed')
        opt.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"]);step=state["step"]
        if 'buffers' in state:
            buffers=dict(model.named_buffers())
            if set(buffers)!=set(state['buffers']):raise ValueError('Reward buffer names changed')
            for name,value in state['buffers'].items():buffers[name].copy_(value.to(buffers[name].device))
        if state.get("accumulation",accum)!=accum:raise ValueError("Reward resume accumulation changed")
        random_states=state.get("rank_rng",[state["rng"]]);restore_rng(random_states[min(rank,len(random_states)-1)])
    collator=GroupCollator(processor,c);before={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    frozen_before=frozen_hash(model)
    torch.cuda.reset_peak_memory_stats();started=time.monotonic();examples=[];global_visits=0
    while step<total:
        opt.zero_grad(set_to_none=True)
        # Deterministic shuffled epochs and permutations, tied to canonical optimizer cursor.
        epoch=step//steps_per_epoch;batch_in_epoch=step%steps_per_epoch
        visit_base=batch_in_epoch*size*accum
        valid_total=min(size*accum,len(groups)-visit_base) if not a.smoke else size*accum
        for micro in range(accum):
            cuda_rng_before=tree_hash(rng_state()['cuda']) if a.audit_replay else None
            index=visit_base+micro*size+rank;order=list(range(len(groups)));random.Random(c["plan"]["seed"]+epoch).shuffle(order)
            valid=a.smoke or (micro*size+rank<valid_total)
            group=Path(groups[order[index%len(groups)]]["group_path"])
            seed_microbatch(model,c['plan']['seed'],epoch,index)
            values,meta=collator(group,training=True,permutation_seed=epoch*len(groups)+index+c["plan"]["seed"])
            cache=run/"reward_model/feature_cache"/(meta["input_hash"]+".safetensors");features=vision_features(model,values,cache)
            values=move(values,f"cuda:{local}")
            context=wrapped.no_sync() if size>1 else __import__('contextlib').nullcontext()
            with context,cached_vision(model,features),torch.autocast("cuda",dtype=torch.bfloat16):
                loss=suffix_loss(wrapped,values);(loss*(size/valid_total if valid else 0)).backward()
            if rank==0:examples.append(meta)
            if a.audit_replay:append_jsonl(run/f'metrics/replay_rank_{rank}.jsonl',{'step':step+1,'micro':micro,'epoch':epoch,'index':index,'valid':valid,'input_hash':meta['input_hash'],'permutation':meta['permutation'],'target_json':meta['target_json'],'input_ids_hash':tensor_hash(values['input_ids']),'CUDA_rng_before':cuda_rng_before,'loss':float(loss.detach())})
        gradient_mean(model)
        norm=float(torch.nn.utils.clip_grad_norm_(params,1.))
        if not math_finite(norm):raise FloatingPointError("Reward gradient nonfinite")
        if a.audit_replay:append_jsonl(run/f'metrics/replay_rank_{rank}.jsonl',{'step':step+1,'gradient_hash':gradients_hash(model),'gradient_norm':norm})
        opt.step();scheduler.step();opt.zero_grad(set_to_none=True);step+=1;global_visits+=valid_total
        if a.audit_replay:
            current=tree_hash({n:p for n,p in model.named_parameters() if p.requires_grad});hashes=[None]*size if size>1 else [current]
            if size>1:dist.all_gather_object(hashes,current)
            write_json(run/f'reports/parameter_sync_step_{step}_rank_{rank}.json',{'hashes':hashes,'equal':len(set(hashes))==1})
            if len(set(hashes))!=1:raise AssertionError('Reward trainable parameters differ across ranks after optimizer update')
        random_states=[None]*size if rank==0 else None
        if size>1:dist.gather_object(rng_state(),random_states,dst=0)
        else:random_states=[rng_state()]
        if rank==0:
            cp=rm/"checkpoints"/f"step-{step:08d}";cp.mkdir(parents=True,exist_ok=True)
            model.save_pretrained(cp/"adapter");processor.save_pretrained(cp/"processor")
            atomic_torch(cp/"training_state.pt",{"optimizer":opt.state_dict(),"scheduler":scheduler.state_dict(),"step":step,"rng":random_states[0],"rank_rng":random_states,'buffers':{n:b.detach().cpu().clone() for n,b in model.named_buffers()},"accumulation":accum,'world_size':size,'total_steps':total,'backend':backend,"dataset_hash":digest(data["groups"]),"smoke":a.smoke,"config":c})
            write_json(rm/"checkpoints/latest.json",{"checkpoint":str(cp),"step":step,"steps_per_epoch":steps_per_epoch,"total_steps":total})
            append_jsonl(run/"metrics/reward_train.jsonl",{"step":step,"loss":float(loss.detach()),"loss_scope":"rank0_last_microbatch","gradient_norm":norm,"input_tokens":meta["prefix_tokens"],"images":meta["image_count"],"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),"smoke":a.smoke})
            print("Reward step",step,"loss",float(loss.detach()),"gradient",norm,flush=True)
        if a.audit_replay:write_json(run/f'reports/kernel_configs_rank_{rank}.json',selected_configs())
        if size>1:dist.barrier()
        if a.interrupt_after_step==step:raise InterruptedError("Injected reward stop at committed optimizer boundary")
    if rank==0:
        final=rm/"final";final.mkdir(parents=True,exist_ok=True)
        model.save_pretrained(final/"adapter");processor.save_pretrained(final/"processor")
        adapter_hash=sha256(final/"adapter/adapter_model.safetensors")
        delta=sum(float((p.detach().cpu()-before[n]).square().sum()) for n,p in model.named_parameters() if p.requires_grad)**.5
        frozen_after=frozen_hash(model)
        assert frozen_before==frozen_after,"Reward frozen base/vision/buffers changed"
        if a.smoke and step>int(state["step"] if a.resume else 0):
            assert frozen_before==frozen_after,"Reward frozen base/vision changed"
            assert delta>0,"Reward LoRA did not update"
        elapsed=time.monotonic()-started
        version=digest({"adapter":adapter_hash,"base_index":audit["base_index_sha256"],"processor":audit["processor_hash"],"prompt":sha256(c["reward"]["prompt"]),"protocol":c["reward"]["protocol"]})[:24]
        info={"schema":"rl-reward-checkpoint-v1","version":version,"smoke":a.smoke,"base_model":c["reward_model"]["base_model"],"adapter_sha256":adapter_hash,"protocol":c["reward"]["protocol"],"prompt_sha256":sha256(c["reward"]["prompt"]),"config":c,"audit":audit,"steps":step,"delta_l2":delta,"frozen_hash_before":frozen_before,"frozen_hash_after":frozen_after,"full_group_examples":examples,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),"gpu_total_bytes":torch.cuda.get_device_properties(local).total_memory,"seconds":elapsed,"groups_per_second":len(examples)/elapsed,"unique_teacher_conditions":data["unique_conditions"],"quality_claim":False,"steps_per_epoch":steps_per_epoch,"world_size":size,"total_steps":total,"dataset_hash":digest(data["groups"])}
        info['groups_per_second']=global_visits/elapsed;info['global_group_visits_this_launch']=global_visits
        info['rank0_logged_visits_per_second']=len(examples)/elapsed
        write_json(final/"manifest.json",info);write_json(rm/"reports/training.json",info)
        if a.smoke:write_json(run/"reports/reward_training_smoke.json",info)
        write_json(rm/"reports/quality_gate.json",{"passed":False,"software_only":a.smoke,"reason":"Formal quality evaluation with independent holdout and manual episode review has not passed","version":version})
    if size>1:dist.destroy_process_group()

def math_ceil(x):return __import__('math').ceil(x)
def math_finite(x):return __import__('math').isfinite(x)
if __name__=="__main__":main()
