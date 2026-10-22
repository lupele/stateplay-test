"""Export a complete merged DiT for the existing latent inference entry."""
import argparse
import re
from pathlib import Path
import torch
from safetensors.torch import load_file
from safetensors import safe_open
from .io import SFT,SFT_SHA,sha256,save_tensors,write_json,protect,ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    out=protect(a.output)
    if (ROOT/"outputs/rl_posttrain").resolve() not in out.parents:raise ValueError("Export must stay in the RL output tree")
    if out.exists():raise ValueError('Refusing to overwrite an existing inference export')
    checkpoint=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    if checkpoint["sft_sha256"]!=SFT_SHA or sha256(SFT)!=SFT_SHA:raise ValueError("Original implicit SFT identity mismatch")
    state=load_file(str(SFT));live=checkpoint["live"];config=checkpoint["config"];merged=[]
    # SFT logger omitted frozen patch embedding. Include its exact Wan base
    # tensors so the export is a complete inference DiT rather than an overlay.
    base=Path(config['world']['base_model'])/'Wan-AI/Wan2.2-TI2V-5B';added=[]
    for shard in sorted(base.glob('diffusion_pytorch_model*.safetensors')):
        with safe_open(shard,framework='pt',device='cpu') as handle:
            for key in ('patch_embedding.weight','patch_embedding.bias'):
                if key not in state and key in handle.keys():state[key]=handle.get_tensor(key);added.append(key)
    if not all(k in state for k in ('patch_embedding.weight','patch_embedding.bias')):raise ValueError('Frozen base patch embedding is unavailable')
    for name,value in live.items():
        if name.startswith("latent_state_rollout.transition."):
            if name not in state:raise ValueError("Unknown transition tensor")
            state[name]=value.to(state[name].dtype)
        elif name.endswith("lora_A"):
            prefix=name[:-len("lora_A")];weight=prefix+"weight";b=live[prefix+"lora_B"]
            if weight not in state:raise ValueError("Unknown coupling projection")
            delta=b.float()@value.float()*(config["nft"]["coupling_alpha"]/config["nft"]["coupling_rank"])
            state[weight]=(state[weight].float()+delta).to(state[weight].dtype);merged.append(weight)
        elif not name.endswith("lora_B"):raise ValueError("Unexpected trainable export tensor")
    save_tensors(out,state)
    write_json(out.with_suffix(".manifest.json"),{"source_sft":str(SFT),"source_sft_sha256":SFT_SHA,"world_checkpoint":str(Path(a.checkpoint).resolve()),"mode":config["nft"]["mode"],"optimizer_step":checkpoint["state"]["optimizer_step"],"merged_coupling_weights":merged,"frozen_base_tensors_added":added,"complete_inference_DiT":True,"sha256":sha256(out),"dtype":"bf16 inference export","attention_backend":"Disable native MultiheadAttention fastpath for matching RL numerics"})
    print("Complete latent DiT export",out)
if __name__=="__main__":main()
