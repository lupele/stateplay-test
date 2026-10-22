"""Input and precision diagnostics for the unmodified existing inference entry."""
import torch
from PIL import Image
from safetensors.torch import load_file
from stateplay.pipeline import StatePlayPipeline
from stateplay.utils.image import preprocess_image
from stateplay.utils.actions import load_actions
from diffsynth.pipelines.stateplay import model_fn_stateplay
from .io import run_path,read_json,write_json
from .config import load_config

def main():
    torch.set_num_threads(8);run=run_path('smoke_v1');g=run/'groups/g-0000000-train-0000000';c=load_config('rl_posttrain/configs/smoke.yaml');condition=read_json(g/'condition.json')
    cache={k:v.to('cuda:0') for k,v in load_file(g/'conditioning.safetensors').items()}
    pipe=StatePlayPipeline.from_pretrained(c['world']['base_model'],c['world']['checkpoint']).to('cuda:0')
    with torch.no_grad():
        context=pipe.encode_prompt(condition['npc_prompt'])
        im=preprocess_image(Image.open(g/'initial.png'),128,224,'cuda:0',torch.bfloat16)
        first=pipe.vae.encode([im],device=pipe.device,tiled=False,tile_size=(30,52),tile_stride=(15,26)).to(pipe.device,torch.bfloat16)
        actions=load_actions(str(g/'actions_raw.parquet'),9,hold_window=10,device='cuda:0',dtype=torch.bfloat16,button_cols=pipe.button_cols)
        inputs={name:{'shape':list(v.shape),'cached_shape':list(cache[name].shape),'max_abs_difference':float((v.float()-cache[name].float()).abs().max())} for name,v in [('first',first),('context',context),('actions',actions)]}
        x=torch.randn((1,48,3,8,14),generator=torch.Generator('cpu').manual_seed(20261013)).to(pipe.device,torch.bfloat16);x[:,:,:1]=first
        kwargs=dict(dit=pipe.dit,latents=x,timestep=torch.tensor([1000.],device=pipe.device,dtype=torch.bfloat16),context=context,keyboard_action=actions,first_frame_latents=first,fuse_vae_embedding_in_latents=True)
        torch.backends.mha.set_fastpath_enabled(True);native=model_fn_stateplay(**kwargs)
        torch.backends.mha.set_fastpath_enabled(False);ordinary=model_fn_stateplay(**kwargs)
        with torch.autocast('cuda',dtype=torch.bfloat16):amp=model_fn_stateplay(**kwargs)
        outputs={label:{k:float((a[k].float()-b[k].float()).abs().max()) for k in ['pred_video','G_seq_low','G_seq']} for label,a,b in [('native_vs_manual',native,ordinary),('manual_vs_amp',ordinary,amp)]}
        write_json(run/'reports/parity_diagnostics.json',{'inputs':inputs,'outputs':outputs});print(inputs,outputs,flush=True)
if __name__=='__main__':main()
