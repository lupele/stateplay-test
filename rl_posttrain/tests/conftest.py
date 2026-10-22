import copy
import io
import pytest
from PIL import Image
from rl_posttrain.config import load_config
from rl_posttrain.io import ROOT,write_json,sha256,digest

@pytest.fixture
def config():
    c=load_config(ROOT/"rl_posttrain/configs/api_b_only.yaml")
    c["reward"].update(rpm=100000,tpm=1000000000,max_retries=2,retry_budget_seconds=30,timeout_seconds=2)
    return c

def fake_group(tmp_path,config,index=0):
    import pandas as pd
    from rl_posttrain.data.conditions import BUTTONS
    root=tmp_path/f"g-{index:06d}";root.mkdir(parents=True,exist_ok=True)
    asset=root/"frame.jpg";Image.new("RGB",(64,32),(128,64,32)).save(asset)
    frame={"path":str(asset),"sha256":sha256(asset),"width":64,"height":32,"mode":"RGB","timestamp":0.,"index":0}
    assets={k:[frame.copy() for _ in range(24)] for k in ("reference_video","video_1","video_2","video_3","video_4")}
    for k in ("initial","macro_success","result_win","result_lose"):assets[k]=[frame.copy()]
    pd.DataFrame({k:[0]*100 for k in BUTTONS}).to_parquet(root/"actions_raw.parquet")
    condition={"condition_id":str(index),"npc_prompt":"NPC: Idle","action_rows":100,"fps":20,"button_order":BUTTONS,"action_sha256":digest(index),"mechanism_class":"PRIVATE_LABEL_SENTINEL","cluster_id":"test-cluster"}
    candidates=[{"slot":f"video_{i}","video_id":f"id-{index}-{i}","complete":True} for i in range(1,5)]
    write_json(root/"condition.json",condition)
    write_json(root/"manifest.json",{"schema":"rl-group-v1","condition_id":str(index),"condition":condition,"candidates":candidates,"frame_assets":assets,"frame_asset_hash":digest(assets),"protocol":config["reward"]["protocol"],"prompt_sha256":sha256(config["reward"]["prompt"])})
    return root

@pytest.fixture
def group(tmp_path,config):return fake_group(tmp_path,config)
