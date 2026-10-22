import argparse
import os
import subprocess
import json
from pathlib import Path
from .config import load_config,validate
from .io import ROOT,run_path,fingerprint,write_json,sha256,QWEN
from .data.conditions import catalog
from .data.sampler import Ledger

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--run-id",required=True);p.add_argument("--skip-catalog",action="store_true");a=p.parse_args()
    c=validate(load_config(a.config),a.run_id);run=run_path(a.run_id)
    for folder in ("plan","conditions","groups","requests","world/api/checkpoints","world/local/checkpoints","reward_model/reports","metrics","logs","reports"):(run/folder).mkdir(parents=True,exist_ok=True)
    cp=fingerprint(c["world"]["checkpoint"])
    if cp["sha256"]!=c["world"]["checkpoint_sha256"]:raise ValueError("Original SFT hash mismatch")
    before=run/"reports/original_checkpoint_before.json"
    if not before.exists():write_json(before,cp)
    index=json.loads((QWEN/"model.safetensors.index.json").read_text());shards=sorted(set(index["weight_map"].values()))
    for name in shards:
        if not (QWEN/name).is_file():raise FileNotFoundError(name)
    from safetensors import safe_open
    keys=set()
    for name in shards:
        with safe_open(QWEN/name,framework="pt",device="cpu") as h:keys.update(h.keys())
    if set(index["weight_map"])-keys:raise ValueError("Qwen index contains missing tensors")
    if not any("visual." in k for k in keys) or not any("language_model." in k for k in keys):raise ValueError("Full multimodal Qwen weights required")
    gpu=subprocess.check_output(["nvidia-smi","--query-gpu=index,name,memory.total,memory.used","--format=csv,noheader"],text=True)
    write_json(run/"plan/config.json",c)
    if not a.skip_catalog:
        records=catalog(c,run);ledger=Ledger(run/"plan",records,c["plan"]["seed"])
        write_json(run/"plan/stage_state.json",{"stage":"prepared","A_budget":min(c["plan"]["api_conditions"],len(records)),"C_budget":max(0,len(records)-c["plan"]["api_conditions"]),"total_conditions":len(records),"epochs":1})
    report={"sft":cp,"transition_tokenizer":fingerprint(c["world"]["tokenizer_checkpoint"]),"qwen_model_type":json.loads((QWEN/"config.json").read_text())["model_type"],"qwen_shards":shards,"qwen_tensors":len(keys),"qwen_local_files_only":True,"gpus":gpu,"api_credential_configured":bool(os.environ.get("STATEPLAY_GPT_API_KEY") or os.environ.get("OPENAI_API_KEY")),"official_nft_commit":"cbb14f84b8312b620390dfcbe2fab69c1383b104","independent_evaluation":{"mechanics":"stateplay_eval.py","hud":"evaluation/hud_original_weights_20261009/evaluation/hud","move_acc":"missing callable extractor; formal metric blocked","att_acc":"missing callable extractor; formal metric blocked"}}
    write_json(run/"reports/preflight.json",report);print(json.dumps({"preflight":"passed","run":str(run),"qwen_full_multimodal":True,"api_credential_configured":report["api_credential_configured"]}))
if __name__=="__main__":main()
