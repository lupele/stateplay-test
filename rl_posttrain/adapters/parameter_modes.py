import hashlib
import math
import re
import torch
from torch import nn
import torch.nn.functional as F

class CouplingLoRA(nn.Module):
    def __init__(self,base,rank=8,alpha=16,dropout=0.):
        super().__init__();self.base=base;self.scale=alpha/rank;self.dropout=nn.Dropout(dropout)
        self.lora_A=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32))
        self.lora_B=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))
    def forward(self,x):
        base=self.base(x)
        update=F.linear(F.linear(self.dropout(x).to(self.lora_A.dtype),self.lora_A),self.lora_B)
        return base+update.to(base.dtype)*self.scale

def configure(model,mode,rank=8,alpha=16,dropout=0.):
    if mode not in ("B-only","Coupling-only","Both"): raise ValueError(mode)
    for i,block in enumerate(model.blocks):
        for key in ("q","k","v","out"):
            layer=getattr(block.latent_state_attention,key)
            if not isinstance(layer,CouplingLoRA): setattr(block.latent_state_attention,key,CouplingLoRA(layer,rank,alpha,dropout))
    model.requires_grad_(False)
    selected=[]
    for name,param in model.named_parameters():
        transition=name.startswith("latent_state_rollout.transition.") and mode in ("B-only","Both")
        coupling=bool(re.fullmatch(r"blocks\.\d+\.latent_state_attention\.(q|k|v|out)\.lora_[AB]",name)) and mode in ("Coupling-only","Both")
        if transition or coupling:
            param.data=param.data.float();param.requires_grad_(True);selected.append(name)
    if not selected: raise RuntimeError("No trainable tensors")
    if mode in ("Coupling-only","Both") and sum(n.endswith("lora_A") for n in selected)!=len(model.blocks)*4: raise RuntimeError("Coupling target mismatch")
    return selected

def optimizer(model,config):
    groups=[]
    for prefix,lr in (("latent_state_rollout.transition.",config["transition_lr"]),("blocks.",config["coupling_lr"])):
        params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith(prefix)]
        if params: groups.append({"params":params,"lr":lr})
    return torch.optim.AdamW(groups,weight_decay=config["weight_decay"])

def parameter_hash(model,frozen=True):
    h=hashlib.sha256()
    for name,tensor in sorted(list(model.named_parameters())+list(model.named_buffers())):
        if hasattr(tensor,"requires_grad") and tensor.requires_grad==frozen: continue
        h.update(name.encode());h.update(str(tensor.dtype).encode());h.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def trainable_state(model): return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}

def gradient_norms(model):
    result={}
    for label,prefix in (("B","latent_state_rollout.transition."),("Coupling","blocks.")):
        grads=[p.grad.detach().float().square().sum() for n,p in model.named_parameters() if p.requires_grad and n.startswith(prefix) and p.grad is not None]
        result[label]=float(torch.stack(grads).sum().sqrt()) if grads else 0.
    return result
