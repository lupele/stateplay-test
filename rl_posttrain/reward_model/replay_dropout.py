"""Projection-local dropout masks determined by the canonical microbatch cursor."""
import torch
from torch import nn
from ..io import digest

class ReplayDropout(nn.Module):
    def __init__(self,p,name):super().__init__();self.p=p;self.name=name;self.seed=None
    def forward(self,x):
        if not self.training or not self.p:return x
        if self.seed is None:raise RuntimeError('Training dropout needs an explicit microbatch seed')
        generator=torch.Generator(device=x.device).manual_seed(self.seed)
        # A contiguous one-dimensional draw is independent of x's strides and
        # allocator layout. Activation recomputation repeats exactly this mask.
        mask=(torch.rand(x.numel(),device=x.device,dtype=torch.float32,generator=generator)>=self.p).view(x.shape)
        return x*mask.to(x.dtype)/(1-self.p)

def install(model):
    for name,module in list(model.named_modules()):
        if isinstance(module,nn.Dropout) and '.lora_dropout.' in name:
            parent,key=name.rsplit('.',1);model.get_submodule(parent)._modules[key]=ReplayDropout(module.p,name)

def seed_microbatch(model,seed,epoch,index):
    for module in model.modules():
        if isinstance(module,ReplayDropout):module.seed=int(digest([seed,epoch,index,module.name])[:15],16)
