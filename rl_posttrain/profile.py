"""Measured memory/throughput profiles; a short-latent test is never a full profile."""
import argparse
import copy
from pathlib import Path
import subprocess
import os
import sys
import time
import torch
from .io import run_path,write_json,read_json,sha256
from .config import load_config

def world_profile(c,run,full_group=False):
    from .adapters.world_model import WorldAdapter
    from .algorithms.policy_bank import PolicyBank
    from .algorithms.nft_loss import forward_noise,nft_loss
    from .data.conditions import catalog,prepare_group,complete_protocol
    torch.set_num_threads(6)
    profile_records=run/"conditions/profile_records.json"
    records=read_json(profile_records) if profile_records.exists() else catalog(c,run)
    adapter=WorldAdapter(c);bank=PolicyBank(adapter.dit)
    group=prepare_group(records[0],c,run,bank.version,9000000)
    torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    values=adapter.generate_group(group,bank) if full_group else adapter.condition_tensors(group)
    if full_group and not read_json(group/"manifest.json").get("frame_assets"):complete_protocol(group,c)
    rollout_seconds=time.monotonic()-started;rollout_peak=torch.cuda.max_memory_allocated();torch.cuda.reset_peak_memory_stats()
    clean=values["real"];sigma=float(adapter.scheduler.sigmas[15 if len(adapter.scheduler.sigmas)>15 else 1]);noise=torch.randn_like(clean)
    xt,_=forward_noise(clean,noise,sigma,values["first"])
    with bank.use("old"),torch.no_grad():old=adapter.forward(xt,sigma,values["context"],values["actions"],values["first"],False)["pred_video"]
    started=time.monotonic()
    with bank.live_graph():
        live=adapter.forward(xt,sigma,values["context"],values["actions"],values["first"],True)["pred_video"]
        loss,_=nft_loss(live,old,xt,clean,sigma,[.8]);loss.backward()
    torch.cuda.synchronize();elapsed=time.monotonic()-started
    result={"kind":"real_implicit_SFT_full_scale" if c["rollout"]["height"]==480 else "real_implicit_SFT_small_scale","full_group":full_group,"group_path":str(group),"height":c["rollout"]["height"],"width":c["rollout"]["width"],"loaded_frames":read_json(group/"condition.json")["loaded_frames"],"latent_shape":list(clean.shape),"steps":c["rollout"]["steps"],"rollout_seconds":rollout_seconds,"rollout_peak_allocated_bytes":rollout_peak,"backward_seconds":elapsed,"backward_peak_allocated_bytes":torch.cuda.max_memory_allocated(),"backward_peak_reserved_bytes":torch.cuda.max_memory_reserved(),"gpu_total_bytes":torch.cuda.get_device_properties(0).total_memory,"weight_update":False}
    write_json(run/"reports/world_full_profile.json",result);print(result,flush=True)

def reward_profile(c,run,group=None,checkpoint=None,steps=0):
    from .reward_model.model import load_model,vision_features,move,cached_vision,suffix_loss,score
    from .reward_model.collator import GroupCollator
    if group is None:
        report=read_json(run/"reports/world_full_profile.json");group=Path(report["group_path"])
    model,processor,audit=load_model(c,training=steps>0,adapter_path=checkpoint)
    collator=GroupCollator(processor,c);values,meta=collator(group,training=steps>0)
    cache=run/"reward_model/feature_cache"/(meta["input_hash"]+".safetensors")
    start=time.monotonic();torch.cuda.reset_peak_memory_stats();features=vision_features(model,values,cache);values=move(values,"cuda:0")
    if steps:
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=c["reward_model"]["learning_rate"])
        from .reward_model.replay_dropout import seed_microbatch
        for visit in range(steps):
            seed_microbatch(model,c['plan']['seed'],0,visit)
            with cached_vision(model,features),torch.autocast("cuda",dtype=torch.bfloat16):loss=suffix_loss(model,values)
            loss.backward();opt.step();opt.zero_grad(set_to_none=True)
        output={"loss":float(loss.detach())}
    else:output={"scores":score(model,processor,values,features,c["reward_model"]["max_new_tokens"])[0]}
    torch.cuda.synchronize();result={"kind":"real_Qwen3.5_9B_full_group","full_multimodal":audit["full_multimodal"],"local_files_only":True,"image_count":meta["image_count"],"input_tokens":meta["prefix_tokens"],"input_hash":meta["input_hash"],"seconds":time.monotonic()-start,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),"gpu_total_bytes":torch.cuda.get_device_properties(0).total_memory,"steps":steps,**output}
    write_json(run/"reports/reward_profile.json",result);print(result,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--run-id",required=True);p.add_argument("--component",choices=["world","reward-model","joint"],default="world");p.add_argument("--full-group",action="store_true");p.add_argument("--mode",default="all");p.add_argument("--group");p.add_argument("--checkpoint");p.add_argument("--steps",type=int,default=0);p.add_argument("--child",action="store_true");a=p.parse_args()
    c=load_config(a.config);run=run_path(a.run_id)
    if a.component=="world":world_profile(c,run,a.full_group)
    elif a.component=="reward-model":reward_profile(c,run,Path(a.group) if a.group else None,a.checkpoint,a.steps)
    else:
        # Joint profile is accepted only with real full-scale evidence from both components.
        world=read_json(run/"reports/world_full_profile.json");reward=read_json(run/"reports/reward_profile.json")
        if not world["full_group"] or world["height"]!=480 or reward["image_count"]!=124:raise RuntimeError("Joint selection requires original-scale world and complete Qwen group profiles")
        world_peak=max(world["rollout_peak_allocated_bytes"],world["backward_peak_allocated_bytes"]);reward_peak=reward["peak_allocated_bytes"]
        fits=world_peak<world["gpu_total_bytes"]*.9 and reward_peak<reward["gpu_total_bytes"]*.9
        selected={"strategy":"dedicated","world_gpus":[0,1,2,3,4,5],"reward_gpus":[6],"conditions_per_batch":24,"effective_video_batch":24} if fits else None
        result={"kind":"measured_component_capacity","selected_topology":selected,"world_peak_bytes":world_peak,"reward_peak_bytes":reward_peak,"headroom_required":.1,"alternatives":{"6+1":"fits" if fits else "does_not_fit","5+2":"requires separate actual two-GPU shard measurement; not inferred","4+3":"requires separate actual three-GPU shard measurement; not inferred"},"joint_acceptance":"pending concurrent C smoke"}
        for shards,label in ((2,"5+2"),(3,"4+3")):
            path=run/"reports"/f"reward_inference_{shards}gpu.json"
            if path.exists():
                measured=read_json(path);result["alternatives"][label]={"measured":True,"fits":measured["fits_10_percent_headroom"],"reward_seconds":measured["seconds"],"memory":measured["memory"]}
        if (run/"reports/local_completion.json").exists():result["joint_acceptance"]="real six-world-plus-one-frozen-RM C smoke passed"
        write_json(run/"reports/joint_profile.json",result)
        if not fits:raise RuntimeError("Complete group does not fit with required headroom; measure multi-GPU sharding")
        print(result)
if __name__=="__main__":main()
