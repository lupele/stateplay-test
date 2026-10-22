"""No text-only AutoModel path. Frozen visual features may be cached exactly."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import re
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from transformers import Qwen3_5ForConditionalGeneration,AutoProcessor,AutoTokenizer
from peft import LoraConfig,get_peft_model,PeftModel
from ..io import QWEN,sha256,digest,save_tensors,read_json,write_json

TARGET=re.compile(r"^model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj)|mlp\.(?:gate_proj|up_proj|down_proj))$")

def load_model(config,device="cuda:0",adapter_path=None,training=False,device_map=None):
    from ..reproducibility import configure
    backend=configure()
    from .kernel_replay import install,VERSION
    pinned_kernels=install()
    backend['reward_kernels']=VERSION
    path=config["reward_model"]["base_model"]
    if Path(path).resolve()!=QWEN.resolve():raise ValueError("Only the specified local Qwen3.5-9B is permitted")
    processor=AutoProcessor.from_pretrained(path,local_files_only=True)
    AutoTokenizer.from_pretrained(path,local_files_only=True)
    processor.tokenizer.padding_side="left"
    kwargs=dict(local_files_only=True,dtype=torch.bfloat16,attn_implementation="sdpa")
    if device_map is not None:kwargs["device_map"]=device_map
    model=Qwen3_5ForConditionalGeneration.from_pretrained(path,**kwargs)
    if model.config.model_type!="qwen3_5" or not hasattr(model.model,"visual") or not hasattr(model.model,"language_model"):raise ValueError("Full Qwen multimodal model was not loaded")
    targets=[n for n,m in model.named_modules() if TARGET.fullmatch(n)]
    if len(targets)!=200:raise ValueError(f"Unexpected hybrid Qwen LoRA targets {len(targets)} (expected 24*6+8*7=200)")
    model.requires_grad_(False)
    if adapter_path:
        model=PeftModel.from_pretrained(model,str(adapter_path),is_trainable=training,local_files_only=True)
    else:
        r=config["reward_model"]
        model=get_peft_model(model,LoraConfig(r=r["lora_rank"],lora_alpha=r["lora_alpha"],lora_dropout=r["lora_dropout"],target_modules=targets,bias="none",task_type="CAUSAL_LM"))
    if device_map is None:model=model.to(device)
    # Keep FP32 trainable masters, but let bf16 autocast handle matmuls without
    # PEFT allocating a full FP32 copy of each 43k-token projection input.
    for module in model.modules():
        if hasattr(module,"cast_input_dtype_enabled"):module.cast_input_dtype_enabled=False
    if training:
        from .replay_dropout import install
        install(model)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        model.config.use_cache=False;model.train();base_model(model).model.visual.eval()
        for n,p in model.named_parameters():
            if p.requires_grad:
                if "lora_" not in n or ".visual." in n:raise RuntimeError("Unapproved reward trainable tensor")
                p.data=p.data.float()
    else:model.eval().requires_grad_(False)
    reload_check=verify_adapter_reload(model,adapter_path) if adapter_path else None
    return model,processor,{"model_class":"Qwen3_5ForConditionalGeneration","model_type":model.config.model_type,"full_multimodal":True,"local_files_only":True,"targets":targets,"trainable_names":[n for n,p in model.named_parameters() if p.requires_grad],"trainable_count":sum(p.numel() for p in model.parameters() if p.requires_grad),"processor_hash":digest(processor.to_dict()),"base_index_sha256":sha256(QWEN/"model.safetensors.index.json"),'reload_adapter_tensors':reload_check,'backend':backend,'pinned_FLA_kernels':pinned_kernels}

def base_model(model):return model.get_base_model() if hasattr(model,"get_base_model") else model

def verify_adapter_reload(model,path):
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    expected=load_file(str(Path(path)/"adapter_model.safetensors"))
    actual=get_peft_model_state_dict(model,save_embedding_layers=False)
    if set(expected)!=set(actual):raise AssertionError("Reloaded LoRA tensor names differ")
    for name,value in expected.items():torch.testing.assert_close(actual[name].detach().cpu(),value,rtol=0,atol=0)
    return {"tensor_count":len(expected),"exact":True,"sha256":sha256(Path(path)/"adapter_model.safetensors")}

def move(values,device):
    return {k:(v.to(device,dtype=torch.bfloat16) if k=="pixel_values" else v.to(device)) if isinstance(v,torch.Tensor) else v for k,v in values.items()}

def suffix_loss(model,values):
    inputs={k:v for k,v in values.items() if k!="labels"}
    # Offload non-reentrant checkpoint inputs, not the model. This is needed
    # for complete 124-image groups on 44.4 GiB L20 cards.
    with torch.autograd.graph.save_on_cpu(pin_memory=True):
        outputs=model(**inputs,use_cache=False)
    positions=values["logits_to_keep"]+1
    targets=values["labels"].index_select(1,positions)
    if (targets==-100).any():raise ValueError("Suffix logits include a masked target")
    return F.cross_entropy(outputs.logits.float().reshape(-1,outputs.logits.shape[-1]),targets.flatten())

@torch.no_grad()
def vision_features(model,values,cache_path=None):
    from safetensors.torch import load_file
    underlying=base_model(model).model;grid=values["image_grid_thw"];pixels=values["pixel_values"]
    if cache_path and Path(cache_path).exists():
        cached=load_file(cache_path);return tuple(cached[f"image_{i:03d}"] for i in range(len(grid)))
    # Frozen vision tower processes one image at a time, preserving all 124 inputs.
    features=[];offset=0;vision_device=next(underlying.visual.parameters()).device
    for entry in grid:
        count=int(entry.prod());part=pixels[offset:offset+count].to(vision_device,torch.bfloat16)
        out=underlying.get_image_features(part,entry.unsqueeze(0).to(vision_device))
        features.append(out.pooler_output[0].detach().cpu());offset+=count
    if offset!=len(pixels):raise ValueError("Pixel/grid segmentation mismatch")
    if cache_path:save_tensors(cache_path,{f"image_{i:03d}":v for i,v in enumerate(features)})
    return tuple(features)

@contextmanager
def cached_vision(model,features):
    underlying=base_model(model).model;original=underlying.get_image_features
    def get_image_features(pixel_values,image_grid_thw=None,**kwargs):
        if image_grid_thw is None or len(image_grid_thw)!=len(features):raise ValueError("Frozen image cache/grid count mismatch")
        device=underlying.get_input_embeddings().weight.device
        expected=[int(e.prod())//4 for e in image_grid_thw]
        if expected!=[f.shape[0] for f in features]:raise ValueError("Frozen image cache token mismatch")
        return SimpleNamespace(pooler_output=tuple(f.to(device,torch.bfloat16) for f in features))
    underlying.get_image_features=get_image_features
    try:yield
    finally:underlying.get_image_features=original

def frozen_hash(model):
    h=hashlib.sha256()
    for n,p in model.named_parameters():
        if p.requires_grad:continue
        h.update(n.encode());h.update(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    for n,p in model.named_buffers():
        h.update(("buffer:"+n).encode());h.update(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

@torch.no_grad()
def score(model,processor,values,features,max_new_tokens=256):
    rows,texts=score_batch(model,processor,values,features,max_new_tokens)
    if len(rows)!=1:raise ValueError('Use score_batch for multiple requests')
    return rows[0],texts[0]

@torch.no_grad()
def score_batch(model,processor,values,features,max_new_tokens=256):
    from .batch_attention import compact_batch_attention
    inputs={k:v for k,v in values.items() if k not in ("labels","logits_to_keep")}
    n=inputs["input_ids"].shape[1]
    with compact_batch_attention(model,inputs['input_ids'].shape[0]>1),cached_vision(model,features),torch.autocast("cuda",dtype=torch.bfloat16):
        outputs=model.generate(**inputs,do_sample=False,max_new_tokens=max_new_tokens,use_cache=True,logits_to_keep=1)
    from ..rewards.base import parse_scores
    texts=[processor.tokenizer.decode(row[n:],skip_special_tokens=True).strip() for row in outputs]
    rows=[]
    for text in texts:
        try:rows.append(parse_scores(text))
        except ValueError as error:
            error.raw_output=text
            raise
    return rows,texts
