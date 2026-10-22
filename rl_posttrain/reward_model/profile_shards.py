"""Actual full-group frozen-RM inference on explicit 1/2/3 GPU layer maps."""
import argparse
from pathlib import Path
import time
import torch
from .model import load_model,move,vision_features,score
from .collator import GroupCollator
from ..config import load_config
from ..io import run_path,read_json,write_json,reward_path,reward_checkpoint

def layer_map(shards):
    if shards not in (1,2,3):raise ValueError("Profile 1, 2, or 3 actual GPU shards")
    mapping={"model.visual":0,"model.language_model.embed_tokens":0,"model.language_model.rotary_emb":0,"model.language_model.norm":shards-1,"lm_head":shards-1}
    for i in range(32):mapping[f"model.language_model.layers.{i}"]=min(shards-1,i*shards//32)
    return mapping

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",default="rl_posttrain/configs/rm_qwen35.yaml");p.add_argument("--run-id",default="smoke_v1");p.add_argument("--shards",type=int,required=True);a=p.parse_args()
    c=load_config(a.config);run=run_path(a.run_id);checkpoint=reward_checkpoint(run);group=Path(read_json(run/"reports/world_full_profile.json")["group_path"])
    torch.set_num_threads(6);mapping=layer_map(a.shards)
    model,processor,audit=load_model(c,adapter_path=checkpoint/"adapter",device_map=mapping)
    values,meta=GroupCollator(processor,c)(group,training=False)
    features=vision_features(model,values,run/"reward_model/feature_cache"/(meta["input_hash"]+".safetensors"));values=move(values,"cuda:0")
    for i in range(a.shards):torch.cuda.reset_peak_memory_stats(i)
    start=time.monotonic();pred,text=score(model,processor,values,features,c["reward_model"]["max_new_tokens"])
    for i in range(a.shards):torch.cuda.synchronize(i)
    memories=[{"logical_gpu":i,"peak_allocated_bytes":torch.cuda.max_memory_allocated(i),"peak_reserved_bytes":torch.cuda.max_memory_reserved(i),"total_bytes":torch.cuda.get_device_properties(i).total_memory} for i in range(a.shards)]
    result={"kind":"real_full_Qwen_group_explicit_sharding","shards":a.shards,"image_count":124,"input_tokens":meta["prefix_tokens"],"device_map":mapping,"scores":pred,"raw_output":text,"seconds":time.monotonic()-start,"memory":memories,"fits_10_percent_headroom":all(m["peak_reserved_bytes"]<m["total_bytes"]*.9 for m in memories),"full_multimodal":True,"local_files_only":True}
    write_json(run/"reports"/f"reward_inference_{a.shards}gpu.json",result);print(result,flush=True)
if __name__=="__main__":main()
