"""Keep the original SDPA math without a B x L x L padding mask.

Only artificial LEFT padding is removed inside each full-attention operation.
All real text/image tokens, RoPE positions, batched projections and hybrid linear
attention remain intact. This scope is inference-only and restores upstream
configuration/functions on every exit, including OOM or malformed output.
"""
from contextlib import contextmanager
import torch
import torch.nn.functional as F

def ragged_sdpa(module,query,key,value,attention_mask,dropout=0.,scaling=None,**kwargs):
    if attention_mask is None or attention_mask.ndim!=2:raise ValueError('Batch SDPA expects a compact 2D left-padding mask')
    rows=[];q_len=query.shape[2];k_len=key.shape[2]
    if q_len not in (1,k_len):raise ValueError('Only complete prefill or one-token KV-cache decode is supported')
    if attention_mask.shape!=(query.shape[0],k_len):raise ValueError('KV cache and padding mask lengths differ')
    for index,mask in enumerate(attention_mask):
        pad=int((mask==0).sum())
        if pad>=k_len or (pad and mask[:pad].any()) or not bool(mask[pad:].all()):
            raise ValueError('Only contiguous artificial left padding is allowed')
        q=query[index:index+1,:,pad:,:] if q_len==k_len else query[index:index+1]
        k=key[index:index+1,:,pad:,:];v=value[index:index+1,:,pad:,:]
        options={'enable_gqa':True} if q.shape[1]!=k.shape[1] else {}
        output=F.scaled_dot_product_attention(q,k,v,dropout_p=dropout,scale=scaling,is_causal=q_len>1,**options)
        if q_len==k_len and pad:output=F.pad(output,(0,0,pad,0))
        rows.append(output.transpose(1,2).contiguous())
    return torch.cat(rows),None

@contextmanager
def compact_batch_attention(model,enabled):
    if not enabled:
        yield;return
    from transformers.models.qwen3_5 import modeling_qwen3_5 as upstream
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    base=model.get_base_model() if hasattr(model,'get_base_model') else model
    language=base.model.language_model
    if model.training:raise ValueError('Compact request batching is inference-only')
    original_mask=upstream.create_causal_mask;original_backend=language.config._attn_implementation
    def compact_mask(*args,**kwargs):
        mask=kwargs.get('attention_mask')
        if mask is None or mask.ndim!=2:raise ValueError('A complete per-request attention mask is required')
        return mask
    ALL_ATTENTION_FUNCTIONS.register('stateplay_ragged_sdpa',ragged_sdpa)
    try:
        upstream.create_causal_mask=compact_mask
        language.config._attn_implementation='stateplay_ragged_sdpa'
        yield
    finally:
        language.config._attn_implementation=original_backend
        upstream.create_causal_mask=original_mask
