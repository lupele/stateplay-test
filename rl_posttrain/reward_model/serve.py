"""Frozen scorer bound to one run's group assets and one immutable RM version."""
import argparse
import json
import os
from pathlib import Path
from http.server import BaseHTTPRequestHandler,HTTPServer
import torch
from .model import load_model,move,vision_features,score
from .collator import GroupCollator
from ..io import run_path,read_json,sha256,reward_path,reward_checkpoint
from ..config import load_config

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--run-id",required=True);p.add_argument("--checkpoint");p.add_argument("--host",default="127.0.0.1");p.add_argument("--port",type=int,default=8001);p.add_argument("--smoke",action="store_true");p.add_argument("--shards",type=int,default=1);a=p.parse_args()
    if a.host!="127.0.0.1":raise ValueError("Scorer is private to localhost")
    c=load_config(a.config);run=run_path(a.run_id);cp=Path(a.checkpoint) if a.checkpoint else reward_checkpoint(run);manifest=read_json(cp/"manifest.json")
    if manifest["smoke"] and not a.smoke:raise ValueError("Smoke RM cannot serve formal C")
    if manifest["protocol"]!=c["reward"]["protocol"]:raise ValueError("Protocol mismatch")
    if sha256(cp/'adapter/adapter_model.safetensors')!=manifest['adapter_sha256']:raise ValueError('Reward checkpoint identity changed')
    if sha256(c['reward']['prompt'])!=manifest['prompt_sha256']:raise ValueError('Reward checkpoint teacher prompt changed')
    from .profile_shards import layer_map
    model,processor,_=load_model(c,adapter_path=cp/"adapter",device_map=layer_map(a.shards) if a.shards>1 else None);collator=GroupCollator(processor,c);version=manifest["version"]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,status,value):
            data=json.dumps(value).encode();self.send_response(status);self.send_header("Content-Type","application/json");self.send_header("Content-Length",str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):
            if self.path=="/health":self.reply(200,{"status":"ready","pid":os.getpid(),"version":version,"base_model":c["reward_model"]["base_model"],"frozen":True,"full_multimodal":True,"smoke":manifest["smoke"]})
            else:self.reply(404,{"error":"unknown_endpoint"})
        def do_POST(self):
            try:
                if self.path!="/score":return self.reply(404,{"error":"unknown_endpoint"})
                length=int(self.headers.get("Content-Length",0))
                if not 0<length<16384:return self.reply(400,{"error":"invalid_request_length"})
                body=json.loads(self.rfile.read(length));group=Path(body["group_path"]).resolve()
                if body["version"]!=version:return self.reply(409,{"error":"version_mismatch"})
                if (run/"groups").resolve() not in group.parents:return self.reply(403,{"error":"path_outside_run_groups"})
                m=read_json(group/"manifest.json")
                for images in m["frame_assets"].values():
                    for image in images:
                        if group not in Path(image["path"]).resolve().parents:return self.reply(403,{"error":"asset_outside_group"})
                values,meta=collator(group,training=False);features=vision_features(model,values,run/"reward_model/feature_cache"/(meta["input_hash"]+".safetensors"));values=move(values,"cuda:0")
                pred,raw=score(model,processor,values,features,c["reward_model"]["max_new_tokens"])
                self.reply(200,{"version":version,"scores":pred,"raw_output":raw,"input_hash":meta["input_hash"],"candidate_mapping":meta["mapping"],"image_count":124,"full_multimodal":True})
            except (KeyError,ValueError,FileNotFoundError):self.reply(422,{"error":"invalid_full_group_or_scores"})
            except Exception as e:self.reply(500,{"error":type(e).__name__})
    print("Frozen reward scorer ready",version,flush=True);HTTPServer((a.host,a.port),Handler).serve_forever()
if __name__=="__main__":main()
