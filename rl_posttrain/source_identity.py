"""Immutable model/data/source identities, without reading credential stores."""
import argparse
from pathlib import Path
import subprocess
from .io import ROOT,QWEN,SFT,SFT_SHA,run_path,fingerprint,write_json,read_json,sha256

def main():
    p=argparse.ArgumentParser();p.add_argument("--run-id",default="smoke_v1");p.add_argument("--check",action="store_true");a=p.parse_args();run=run_path(a.run_id)
    base=ROOT/"base_model/Wan-AI/Wan2.2-TI2V-5B"
    paths=[SFT,ROOT/"outputs/tokenizer/transition_tokenizer.pt",base/"Wan2.2_VAE.pth",base/"models_t5_umt5-xxl-enc-bf16.pth",*sorted(base.glob("diffusion_pytorch_model*.safetensors")),*sorted(QWEN.glob("model*.safetensors")),QWEN/"config.json",QWEN/"tokenizer.json",QWEN/"preprocessor_config.json",QWEN/"chat_template.jinja"]
    result={str(path.relative_to(ROOT)):fingerprint(path) for path in paths}
    if result[str(SFT.relative_to(ROOT))]["sha256"]!=SFT_SHA:raise AssertionError("Original SFT identity changed")
    before=run/"reports/source_identity_before.json"
    if a.check:
        previous=read_json(before)
        if result!=previous:raise AssertionError("Source model file hash/size/mtime changed")
        write_json(run/"reports/source_identity_after.json",result)
    elif before.exists():
        if read_json(before)!=result:raise AssertionError("Existing source baseline disagrees")
    else:write_json(before,result)
    sources=["stateplay/pipeline.py","training/train.py","training/latent_state_training.py","diffsynth/models/latent_state.py","diffsynth/models/stateplay_dit.py","diffsynth/pipelines/stateplay.py","diffsynth/diffusion/flow_match.py","stateplay_eval.py","scripts/generate_proxy_benchmark.py"]
    write_json(run/"reports/source_code_mapping.json",{"server_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"files":{s:sha256(ROOT/s) for s in sources},"reference_stateplay_commit":"1b3f5256372552469b4d33fa38bab8491ee1769b","official_diffusion_nft_commit":"cbb14f84b8312b620390dfcbe2fab69c1383b104","official_checkout":"/tmp/stateplay_rl_research/DiffusionNFT",'actual_StatePlay_arxiv':'2607.26754v1','official_StatePlay_audit_commit':'8355dc956422da4e70f5062be40f511f91e37a66',"differences":["Actual pipeline already supports implicit checkpoint; do not use explicit public checkpoint","Original inference rounds frames up; RL uses training loaded-frame policy 100->97","Original evaluator is root stateplay_eval.py; HUD evaluator is under evaluation/hud_original_weights_20261009/evaluation/hud","Source episode IDs are missing; source/action/temporal components still require manual episode review","Parity diagnostics found autocast LayerNorm promotion changes G; sampler now uses original bf16 inference without autocast and restores FP32 masters afterward","The numeric 2605.15256v1.pdf is ReactiveGWM; the named Lin PDF is the actual StatePlay paper","Exact D schedule is verified only for local collector; released training uses high-level Super Art attempt semantics"]})
    print("Source identities",len(paths),"verified; SFT protected",flush=True)
if __name__=="__main__":main()
