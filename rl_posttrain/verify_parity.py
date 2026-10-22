"""Compare the server's existing latent inference with the new adapter at fixed seed."""
import argparse
import gc
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from stateplay.pipeline import StatePlayPipeline
from .config import load_config
from .io import run_path,read_json,write_json,save_tensors
from .adapters.world_model import WorldAdapter
from safetensors.torch import load_file

def main():
    p=argparse.ArgumentParser();p.add_argument("--run-id",default="smoke_v1");a=p.parse_args()
    torch.set_num_threads(8)
    c=load_config("rl_posttrain/configs/smoke.yaml");run=run_path(a.run_id);group=run/"groups/g-0000000-train-0000000";condition=read_json(group/"condition.json")
    adapter=WorldAdapter(c);values=adapter.condition_tensors(group);seed=20261013
    # Inference parity uses bf16 transition tensors, as in the original entry;
    # FP32 optimizer masters are independently verified by the gradient tests.
    for param in adapter.dit.latent_state_rollout.transition.parameters():param.data=param.data.bfloat16()
    with torch.no_grad():clean=adapter.sample(values["first"],values["context"],values["actions"],condition["latent_frames"],seed)
    save_tensors(run/"reports/parity_clean.safetensors",{"clean":clean})
    with torch.no_grad():expected=adapter.decode(clean);loaded=load_file(run/"reports/parity_clean.safetensors")["clean"].to(adapter.device);reloaded=adapter.decode(loaded)
    assert all(np.array_equal(np.array(x),np.array(y)) for x,y in zip(expected,reloaded))
    del clean,loaded,values;adapter.dit.to("cpu");del adapter;gc.collect();torch.cuda.empty_cache()
    pipe=StatePlayPipeline.from_pretrained(c["world"]["base_model"],c["world"]["checkpoint"]).to("cuda:0")
    assert pipe.uses_latent_state
    result=pipe(image=Image.open(group/"initial.png"),actions_parquet=str(group/"actions_raw.parquet"),prompt=condition["npc_prompt"],num_frames=condition["loaded_frames"],num_inference_steps=c["rollout"]["steps"],cfg_scale=1.,action_cfg_scale=1.,height=c["rollout"]["height"],width=c["rollout"]["width"],seed=seed,sigma_shift=5.,action_hold_window=10)
    actual=result.frames[0];differences=np.concatenate([np.abs(np.array(x).astype(float)-np.array(y).astype(float)).flatten() for x,y in zip(expected,actual)])
    report={"kind":"real_original_latent_entry_parity","frames":len(actual),"seed":seed,"pixel_max_abs_error":float(differences.max()),"pixel_mean_abs_error":float(differences.mean()),"latent_save_reload_decode_exact":True,"existing_entry_architecture":"latent","comparison_precision":"bf16 inference without autocast (training uses FP32 masters)","state_output_is_none":result.state is None}
    write_json(run/"reports/original_entry_parity.json",report)
    if differences.max()>1:raise AssertionError("Existing inference and adapter differ by more than one uint8 pixel")
    print(report,flush=True)
if __name__=="__main__":main()
