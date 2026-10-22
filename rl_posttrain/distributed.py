"""Stable, name-ordered FP32 gradient mean independent of DDP bucket rebuilds."""
import hashlib
import torch
import torch.distributed as dist

def tensor_hash(value):
    raw=value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()

def gradient_mean(model):
    selected=sorted((n,p) for n,p in model.named_parameters() if p.requires_grad)
    if not selected:return
    flat=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().float().reshape(-1) for _,p in selected])
    if dist.is_initialized():
        dist.all_reduce(flat,op=dist.ReduceOp.SUM);flat.div_(dist.get_world_size())
    offset=0
    for _,p in selected:
        value=flat[offset:offset+p.numel()].view_as(p).to(p.dtype)
        if p.grad is None:p.grad=value.clone()
        else:p.grad.copy_(value)
        offset+=p.numel()

def gradients_hash(model):
    return tensor_hash(torch.cat([p.grad.detach().float().reshape(-1) for n,p in sorted(model.named_parameters()) if p.requires_grad and p.grad is not None]))

def tree_hash(value):
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):h.update(str((str(v.dtype),tuple(v.shape))).encode());h.update(tensor_hash(v).encode())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(str(k).encode());visit(v[k])
        elif isinstance(v,(list,tuple)):
            for x in v:visit(x)
        else:h.update(repr(v).encode())
    visit(value);return h.hexdigest()
