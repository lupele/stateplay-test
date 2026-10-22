"""Reconstruct the actual implicit 5B DiT; reuse the server's latent forward."""
from pathlib import Path
import os
import time
import torch
from torch import nn
from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import load_file
from PIL import Image
from diffsynth.models.stateplay_dit import StatePlayModel
from diffsynth.models.latent_state import GameTransitionTokenizer
from diffsynth.pipelines.stateplay import model_fn_stateplay
from training.latent_state_training import LATENT_MODEL_DEFAULTS, load_transition_tokenizer
from stateplay.models import WanVideoVAE38, WanTextEncoder, WanTokenizer
from stateplay.utils.image import preprocess_image, to_pil_video, save_mp4
from stateplay.utils.scheduler import FlowMatchScheduler
from ..io import read_json, write_json, save_tensors, sha256
from ..data.frames import read_frames
from .parameter_modes import configure

WAN_CONFIG=dict(has_image_input=False,patch_size=(1,2,2),in_dim=48,dim=3072,ffn_dim=14336,freq_dim=256,text_dim=4096,out_dim=48,num_heads=24,num_layers=30,eps=1e-6,seperated_timestep=True,require_clip_embedding=False,require_vae_embedding=False,fuse_vae_embedding_in_latents=True)

def reconstruct(config,device="cuda:0"):
    world=config["world"]; ti2v=Path(world["base_model"])/"Wan-AI/Wan2.2-TI2V-5B"
    state=load_file(world["checkpoint"],device="cpu")
    if not any(k.startswith("latent_state_rollout.transition.") for k in state): raise ValueError("Wrong checkpoint: implicit transition branch is absent")
    if any(k.startswith("state_encoder.") for k in state): raise ValueError("Explicit public model cannot be used as implicit SFT")
    with init_empty_weights():
        dit=StatePlayModel(**WAN_CONFIG,**LATENT_MODEL_DEFAULTS,num_buttons=11,use_latent_state=True,use_explicit_state=False,use_transition_tokenizer=False)
    sources={k:"original_sft" for k in state}
    for shard in sorted(ti2v.glob("diffusion_pytorch_model*.safetensors")):
        with safe_open(shard,framework="pt",device="cpu") as f:
            for key in ("patch_embedding.weight","patch_embedding.bias"):
                if key not in state and key in f.keys(): state[key]=f.get_tensor(key);sources[key]=str(shard)
    dit.load_state_dict(state,strict=True,assign=True);del state
    tok=GameTransitionTokenizer(visual_hidden_dim=3072)
    load_transition_tokenizer(tok,world["tokenizer_checkpoint"])
    dit.transition_tokenizer=tok
    for key in tok.state_dict():sources["transition_tokenizer."+key]="stage1_tokenizer_checkpoint"
    dit=dit.to(device=device,dtype=torch.bfloat16).eval()
    names=configure(dit,config["nft"]["mode"],config["nft"]["coupling_rank"],config["nft"]["coupling_alpha"],config["nft"]["coupling_dropout"])
    return dit,{"missing_keys":[],"unexpected_keys":[],"key_sources":sources,"trainable_names":names,"trainable_count":sum(p.numel() for p in dit.parameters() if p.requires_grad),"num_layers":len(dit.blocks),"gates":[float(b.latent_state_attention.gate) for b in dit.blocks]}

class WorldForward(nn.Module):
    """This wrapper, including all trainable tensors, is the DDP forward boundary."""
    def __init__(self,dit):super().__init__();self.dit=dit
    def forward(self,xt,sigma,context,actions,first,checkpointing=True,inference_precision=False):
        if xt.shape[0]!=1 and not inference_precision:
            raise ValueError("Training still requires per-rank microbatch=1; inference may batch candidates at the same timestep")
        t=torch.tensor([1000*float(sigma)],device=xt.device,dtype=xt.dtype)
        with torch.autocast(device_type=xt.device.type,dtype=torch.bfloat16,enabled=xt.is_cuda and not inference_precision):
            return model_fn_stateplay(dit=self.dit,latents=xt,timestep=t,context=context,keyboard_action=actions,first_frame_latents=first,fuse_vae_embedding_in_latents=True,use_gradient_checkpointing=checkpointing)

class WorldAdapter:
    def __init__(self,config,device="cuda:0",encoders=True):
        from ..reproducibility import configure
        self.compute_backend=configure()
        # Native MultiheadAttention selects a different bf16 kernel under no_grad.
        # Keep one attention path for old/ref/live and sampling; otherwise the
        # numerical policy changes merely when enabling gradient computation.
        torch.backends.mha.set_fastpath_enabled(False)
        self.config=config;self.device=torch.device(device);self.dtype=torch.bfloat16
        self.dit,self.audit=reconstruct(config,self.device);self.forward=WorldForward(self.dit)
        self.audit["training_precision"]="FP32 trainable masters + bf16 autocast"
        self.audit["sampling_precision"]="bf16 inference without autocast, matching existing entry"
        self.vae=None;self.text_encoder=None;self.tokenizer=None
        self.base=Path(config["world"]["base_model"]);self.scheduler=FlowMatchScheduler()
        self.scheduler.set_timesteps(config["rollout"]["steps"],shift=config["rollout"]["sigma_shift"])
    def load_vae(self):
        if self.vae is None:
            with init_empty_weights():self.vae=WanVideoVAE38()
            state=torch.load(self.base/"Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",map_location="cpu",weights_only=True,mmap=True)
            self.vae.load_state_dict({"model."+k:v for k,v in state.items()},strict=True,assign=True)
            self.vae=self.vae.to(dtype=self.dtype).eval().requires_grad_(False)
        return self.vae.to(self.device)
    def load_text(self):
        if self.text_encoder is None:
            with init_empty_weights():self.text_encoder=WanTextEncoder()
            state=torch.load(self.base/"Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",map_location="cpu",weights_only=True,mmap=True)
            self.text_encoder.load_state_dict(state,strict=True,assign=True)
            self.text_encoder=self.text_encoder.to(dtype=self.dtype).eval().requires_grad_(False)
            self.tokenizer=WanTokenizer(str(self.base/"Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"))
        return self.text_encoder.to(self.device)
    @torch.no_grad()
    def condition_tensors(self,root):
        path=root/"conditioning.safetensors"
        if path.exists():return {k:v.to(self.device) for k,v in load_file(path).items()}
        condition=read_json(root/"condition.json");r=self.config["rollout"]
        text=self.load_text();ids,mask=self.tokenizer(condition["npc_prompt"])
        ids,mask=ids.to(self.device),mask.to(self.device);context=text(ids,mask).to(self.dtype)
        for i,length in enumerate(mask.gt(0).sum(1)):context[i,length:]=0
        self.text_encoder.to("cpu");torch.cuda.empty_cache()
        vae=self.load_vae();frames=read_frames(root/"reference.mp4",condition["loaded_frames"])
        first_tensor=preprocess_image(frames[0],r["height"],r["width"],self.device,self.dtype)
        first=vae.encode([first_tensor],device=self.device,tiled=False).to(self.device,self.dtype)
        video=torch.cat([preprocess_image(im,r["height"],r["width"],self.device,self.dtype) for im in frames],dim=1)
        real=vae.encode([video],device=self.device,tiled=False).to(self.device,self.dtype)
        # The initial-image VAE path is separately encoded in the original training pipeline.
        actions=load_file(root/"actions_dense.safetensors")["actions"].to(self.device,self.dtype)
        self.vae.to("cpu");torch.cuda.empty_cache()
        values={"first":first,"real":real,"context":context,"actions":actions}
        save_tensors(path,values);return values
    @torch.no_grad()
    def sample(self,first,context,actions,latent_frames,seed):
        seeds=list(seed) if isinstance(seed,(list,tuple)) else [seed]
        if not seeds:raise ValueError('A candidate batch cannot be empty')
        x=torch.cat([torch.randn((1,48,latent_frames,first.shape[3],first.shape[4]),
                     generator=torch.Generator('cpu').manual_seed(s),dtype=torch.float32) for s in seeds]).to(self.device,self.dtype)
        def expand(value):
            if value is None:return None
            if value.shape[0]!=1:raise ValueError('Candidate batching requires one common condition')
            return value.expand(len(seeds),*value.shape[1:]).contiguous()
        first,context,actions=map(expand,(first,context,actions))
        x[:,:,:1]=first
        # Original SFT inference is bf16 without autocast. Autocast promotes
        # LayerNorm and changes G even at unchanged weights. Retain the exact
        # FP32 master storage while temporarily casting only for rendering.
        masters={n:p.data for n,p in self.dit.named_parameters() if p.is_floating_point() and p.dtype!=self.dtype}
        params=dict(self.dit.named_parameters())
        try:
            for n in masters:params[n].data=masters[n].to(self.dtype)
            for t in self.scheduler.timesteps:
                sigma=float(t)/1000;v=self.forward(x,sigma,context,actions,first,False,True)["pred_video"]
                x=self.scheduler.step(v,t,x);x[:,:,:1]=first
        finally:
            for n,value in masters.items():params[n].data=value
        return x.detach()
    @torch.no_grad()
    def decode(self,latent):
        vae=self.load_vae();video=vae.decode(latent,device=self.device,tiled=False)
        frames=to_pil_video(video);self.vae.to("cpu");torch.cuda.empty_cache();return frames
    def generate_group(self,root,bank,interrupt_after=None,allow_legacy=False,candidate_batch_size=1,progress=None):
        from ..data.provenance import require_current_group
        require_current_group(root,self.config,allow_legacy)
        manifest=read_json(root/"manifest.json");values=self.condition_tensors(root)
        if not 1<=candidate_batch_size<=4:raise ValueError('generation batch size must be between 1 and 4')
        pending=[]
        for candidate in manifest['candidates']:
            directory=root/'candidates'/candidate['video_id']
            if candidate['complete']:
                if sha256(directory/'clean_latents.safetensors')!=candidate['latent_sha256'] or sha256(directory/'video.mp4')!=candidate['video_sha256']:raise RuntimeError('Completed candidate checksum failed')
            else:pending.append(candidate)
        with bank.use("old"),torch.no_grad():
            for offset in range(0,len(pending),candidate_batch_size):
                chunk=pending[offset:offset+candidate_batch_size]
                started=time.monotonic()
                cleans=self.sample(values['first'],values['context'],values['actions'],manifest['condition']['latent_frames'],[c['seed'] for c in chunk])
                torch.cuda.synchronize(self.device)
                denoising_seconds=time.monotonic()-started
                for candidate,clean in zip(chunk,cleans.split(1)):
                    self._save_candidate(root,manifest,candidate,clean,denoising_seconds/len(chunk),len(chunk))
                    if progress:progress()
                    if interrupt_after is not None and sum(c['complete'] for c in manifest['candidates'])>=interrupt_after:
                        raise InterruptedError('Injected interruption after candidate generation')
        return values

    def _save_candidate(self,root,manifest,candidate,clean,denoising_seconds,batch_size):
        started=time.monotonic()
        directory=root/"candidates"/candidate["video_id"];directory.mkdir(parents=True,exist_ok=True)
        save_tensors(directory/"clean_latents.safetensors",{"clean":clean})
        frames=self.decode(clean);pending=directory/"video.pending.mp4"
        save_mp4(frames,pending,fps=manifest["condition"]["fps"]);os.replace(pending,directory/"video.mp4")
        candidate.update(complete=True,latent_sha256=sha256(directory/"clean_latents.safetensors"),video_sha256=sha256(directory/"video.mp4"),generation_seconds=denoising_seconds+time.monotonic()-started,denoising_seconds_amortized=denoising_seconds,generation_batch_size=batch_size,latent_shape=list(clean.shape))
        manifest['producer_backend']=self.compute_backend
        write_json(root/"manifest.json",manifest)
