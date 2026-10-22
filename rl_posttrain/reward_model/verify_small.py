"""Real 9B suffix-loss/gradient reference, separate from full-group acceptance."""
import argparse
import gc
import json
import torch
from PIL import Image
from .model import load_model,move,cached_vision,vision_features,suffix_loss,frozen_hash,base_model
from ..config import load_config
from ..io import run_path,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",default="rl_posttrain/configs/rm_qwen35.yaml");p.add_argument("--run-id",default="smoke_v1");a=p.parse_args()
    c=load_config(a.config);run=run_path(a.run_id);torch.set_num_threads(6);torch.manual_seed(c['plan']['seed'])
    model,processor,audit=load_model(c,training=True);write_json(run/"reports/qwen_local_load.json",audit)
    image=Image.new("RGB",(256,256),(64,128,64));messages=[{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":'Return only {"video_1":50,"video_2":50,"video_3":50,"video_4":50}.'}]}]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    v=dict(processor(text=[text],images=[image],return_tensors="pt"));n=v["input_ids"].shape[1]
    target=processor.tokenizer('{"video_1":50,"video_2":50,"video_3":50,"video_4":50}<|im_end|>',add_special_tokens=False,return_tensors="pt")["input_ids"]
    v["input_ids"]=torch.cat((v["input_ids"],target),1);v["attention_mask"]=torch.ones_like(v["input_ids"])
    if "mm_token_type_ids" in v:v["mm_token_type_ids"]=torch.cat((v["mm_token_type_ids"],torch.zeros_like(target)),1)
    v["labels"]=v["input_ids"].clone();v["labels"][:,:n]=-100;v["logits_to_keep"]=torch.arange(n-1,v["input_ids"].shape[1]-1)
    features=vision_features(model,v);v=move(v,"cuda:0")
    # Dropout off is necessary for two separately evaluated loss/gradient comparisons.
    model.eval();before=frozen_hash(model)
    inputs={k:x for k,x in v.items() if k not in ("labels","logits_to_keep")}
    def compare(autocast=True):
        with cached_vision(model,features),torch.autocast("cuda",dtype=torch.bfloat16,enabled=autocast):suffix=suffix_loss(model,v)
        suffix.backward();actual={k:p.grad.detach().cpu().clone() for k,p in model.named_parameters() if p.requires_grad and p.grad is not None};model.zero_grad(set_to_none=True)
        with cached_vision(model,features),torch.autocast("cuda",dtype=torch.bfloat16,enabled=autocast):
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                out=model(**inputs,use_cache=False,logits_to_keep=0)
            full=torch.nn.functional.cross_entropy(out.logits[:,:-1].float().reshape(-1,out.logits.shape[-1]),v["labels"][:,1:].reshape(-1),ignore_index=-100)
        full.backward();torch.testing.assert_close(full,suffix,rtol=1e-5,atol=1e-5)
        max_error=0.;squared_error=0.;squared_reference=0.;dot=0.;squared_actual=0.
        for k,p in model.named_parameters():
            if k in actual:
                reference=p.grad.detach().cpu().float();value=actual[k].float();difference=reference-value
                max_error=max(max_error,float(difference.abs().max()))
                squared_error+=float(difference.square().sum());squared_reference+=float(reference.square().sum())
                dot+=float((reference*value).sum());squared_actual+=float(value.square().sum())
        model.zero_grad(set_to_none=True)
        return {"full_loss":float(full.detach()),"suffix_loss":float(suffix.detach()),"max_gradient_abs_error":max_error,"relative_gradient_l2":(squared_error/max(squared_reference,1e-30))**.5,"gradient_cosine":dot/max((squared_reference*squared_actual)**.5,1e-30)}
    production=compare()
    # Test exact mask/shift/gradient equivalence using an FP32 language oracle.
    # Production BF16 GEMMs have different row shapes when logits are sliced;
    # record their residual rather than claiming bitwise gradient equivalence.
    write_json(run/'reports/qwen_bf16_geometry_diagnostic.json',production)
    parameter_dtypes={n:p.dtype for n,p in model.named_parameters()}
    buffer_dtypes={n:b.dtype for n,b in model.named_buffers()}
    underlying=base_model(model)
    underlying.model.language_model.float();underlying.lm_head.float()
    torch.backends.cuda.matmul.allow_tf32=False
    try:reference=compare(autocast=False)
    finally:
        with torch.no_grad():
            for n,p in model.named_parameters():p.data=p.data.to(parameter_dtypes[n])
            for n,b in model.named_buffers():b.data=b.data.to(buffer_dtypes[n])
    write_json(run/'reports/qwen_fp32_language_diagnostic.json',reference)
    if reference['max_gradient_abs_error']>2e-4:raise AssertionError("FP32 language suffix reference gradient mismatch: "+str(reference))
    assert before==frozen_hash(model)
    write_json(run/"reports/qwen_suffix_reference.json",{"kind":"real_9B_one_image_reference","full_multimodal":True,"tokens":v["input_ids"].shape[1],**reference,"reference_compute_dtype":"float32_language","production_bf16_comparison":production,"frozen_hash":before,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"complete_group_test":False})
    print("Real Qwen local loading and suffix-gradient reference passed",flush=True)
if __name__=="__main__":main()
