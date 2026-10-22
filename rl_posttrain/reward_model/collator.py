import json
import random
from pathlib import Path
import torch
from PIL import Image
from ..rewards.protocol import group_content
from ..rewards.base import SLOTS,parse_scores
from ..io import read_json,digest,teacher_path

class GroupCollator:
    def __init__(self,processor,config):self.processor=processor;self.config=config
    def __call__(self,group,training=True,permutation_seed=None,parts_override=None):
        order=list(SLOTS)
        if permutation_seed is not None:random.Random(permutation_seed).shuffle(order)
        parts,mapping=group_content(Path(group),self.config,order)
        if parts_override is not None:
            if training:raise ValueError('Shortcut probes are evaluation-only')
            parts=parts_override
        content=[{"type":"text","text":p["text"]} if p["type"]=="text" else {"type":"image","image":p["path"]} for p in parts]
        images=[Image.open(p["path"]).convert("RGB") for p in parts if p["type"]=="image"]
        messages=[{"role":"user","content":content}]
        text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        prefix=self.processor(text=[text],images=images,return_tensors="pt",padding=False)
        values=dict(prefix);n=values["input_ids"].shape[1]
        if values["image_grid_thw"].shape[0]!=124:raise ValueError("Processor did not preserve all images")
        image_id=self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        image_tokens=int((values["input_ids"]==image_id).sum())
        grid_tokens=int(values["image_grid_thw"].prod(1).sum())//4
        if image_tokens!=grid_tokens:raise ValueError("Expanded image tokens disagree with processor grid")
        if "mm_token_type_ids" not in values:values["mm_token_type_ids"]=(values["input_ids"]==image_id).long()
        meta={"image_count":124,"prefix_tokens":n,"image_tokens":image_tokens,"grid":values["image_grid_thw"].tolist(),"input_keys":list(values),"permutation":order,"mapping":mapping,"input_hash":digest({"parts":parts,"grid":values["image_grid_thw"].tolist(),"processor":self.processor.to_dict(),"nonthinking":True})}
        if training:
            teacher=parse_scores(read_json(teacher_path(group)/"scores.json")["scores"])
            answer={new:teacher[old] for new,old in zip(SLOTS,order)}
            json_text=json.dumps(answer,separators=(",",":"))
            suffix=self.processor.tokenizer(json_text+"<|im_end|>",add_special_tokens=False,return_tensors="pt")["input_ids"]
            values["input_ids"]=torch.cat((values["input_ids"],suffix),1)
            values["attention_mask"]=torch.ones_like(values["input_ids"])
            if "mm_token_type_ids" in values:values["mm_token_type_ids"]=torch.cat((values["mm_token_type_ids"],torch.zeros_like(suffix)),1)
            labels=values["input_ids"].clone();labels[:,:n]=-100
            values["labels"]=labels;values["logits_to_keep"]=torch.arange(n-1,values["input_ids"].shape[1]-1)
            meta["answer_tokens"]=suffix.shape[1];meta["target_json"]=json_text
        if values["input_ids"].shape[1]>self.config["reward_model"]["max_length"]:raise ValueError(f"Full group has {values['input_ids'].shape[1]} tokens; truncation is forbidden")
        return values,meta

    def batch(self,groups):
        """Left-pad complete, independent requests; never concatenate their texts."""
        if not groups:raise ValueError('Empty reward request batch')
        items=[self(group,training=False) for group in groups]
        width=max(values['input_ids'].shape[1] for values,_ in items)
        pad=self.processor.tokenizer.pad_token_id
        if pad is None:pad=self.processor.tokenizer.eos_token_id
        values={}
        for key in items[0][0]:
            tensors=[v[key] for v,_ in items]
            if key in ('input_ids','attention_mask','mm_token_type_ids','token_type_ids'):
                fill=pad if key=='input_ids' else 0
                values[key]=torch.cat([torch.nn.functional.pad(t,(width-t.shape[1],0),value=fill) for t in tensors])
            elif key in ('pixel_values','image_grid_thw'):
                values[key]=torch.cat(tensors)
            else:raise ValueError('Unverified reward batch input key: '+key)
        if values['image_grid_thw'].shape[0]!=124*len(groups):raise ValueError('Incomplete reward group batch')
        return values,[meta for _,meta in items],[v for v,_ in items]
