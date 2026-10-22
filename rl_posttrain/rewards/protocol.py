"""Shared canonical group content for API and Qwen, including exact frame bytes."""
import base64
from pathlib import Path
import json
from ..io import read_json,digest,sha256,atomic_bytes,write_json,teacher_path
from .base import SLOTS

def group_content(root,config,order=None):
    m=read_json(root/"manifest.json");c=m["condition"];assets=m["frame_assets"]
    order=list(order or SLOTS)
    if sorted(order)!=sorted(SLOTS):raise ValueError("Candidate permutation is not bijective")
    if m["protocol"]!=config["reward"]["protocol"] or m["prompt_sha256"]!=sha256(config["reward"]["prompt"]):raise ValueError("Group protocol/prompt changed")
    mapping={new:next(x["video_id"] for x in m["candidates"] if x["slot"]==old) for new,old in zip(SLOTS,order)}
    prompt=Path(config["reward"]["prompt"]).read_text()
    # Persist prompt text even for a frozen local scorer. No class/HUD label enters it.
    teacher=teacher_path(root);teacher.mkdir(parents=True,exist_ok=True)
    saved=teacher/'prompt.txt'
    if saved.exists():
        if saved.read_text()!=prompt:raise ValueError('Permanent teacher prompt must not be overwritten')
    else:atomic_bytes(saved,prompt.encode())
    parts=[{"type":"text","text":prompt}]
    def image(asset):
        if sha256(asset["path"])!=asset["sha256"]:raise ValueError("Frame checksum changed")
        parts.append({"type":"image","path":asset["path"],"sha256":asset["sha256"]})
    parts.append({"type":"text","text":"Initial observation:"});image(assets["initial"][0])
    for name,label in (("macro_success","Static example of a super-art visual effect, not evidence that any candidate executed it."),("result_win","Static example of a WIN result screen."),("result_lose","Static example of a LOSE result screen.")):
        parts.append({"type":"text","text":label});image(assets[name][0])
    import pandas as pd
    buttons=c["button_order"];df=pd.read_parquet(root/"actions_raw.parquet")[buttons].iloc[:c["action_rows"]]
    hold=config["rollout"]["action_hold_window"]
    timeline=[{"start_seconds":i/c["fps"],"end_seconds":min(i+hold,len(df))/c["fps"],"pressed":[k for k in buttons if df.iloc[i][k]==1]} for i in range(0,len(df),hold)]
    label='Player action timeline (semantics follow the versioned instructions; execution is judged from visible evidence): ' if config['reward']['protocol'].startswith('mechanics-reward-v2') else 'Player action timeline (button symbols only, unknown mapping is not guessed): '
    parts.append({"type":"text","text":"NPC conditioning: "+c["npc_prompt"]+"\n"+label+json.dumps(timeline)})
    for label,source in [("reference_video","reference_video")]+list(zip(SLOTS,order)):
        parts.append({"type":"text","text":label+": ordered 24 frames"})
        if len(assets[source])!=24:raise ValueError("Missing frames")
        for item in assets[source]:
            parts.append({"type":"text","text":f"{label} t={item['timestamp']:.4f}s"});image(item)
    if sum(p["type"]=="image" for p in parts)!=124:raise ValueError("Full protocol requires exactly 124 images")
    return parts,mapping

def payload(root,config):
    parts,mapping=group_content(root,config)
    content=[]
    for part in parts:
        if part["type"]=="text":content.append(part)
        else:
            data=base64.b64encode(Path(part["path"]).read_bytes()).decode()
            content.append({"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+data,"detail":"high"}})
    from .base import SCORE_SCHEMA
    r=config["reward"]
    result={"model":r["model"],"messages":[{"role":"user","content":content}],"response_format":{"type":"json_schema","json_schema":{"name":"group_mechanics_scores","strict":True,"schema":SCORE_SCHEMA}},"max_completion_tokens":r["max_completion_tokens"],"reasoning_effort":r["reasoning_effort"]}
    key=digest({"parts":parts,"mapping":mapping,"model":r["model"],"endpoint":r["endpoint"],"protocol":r["protocol"],"prompt_hash":sha256(r["prompt"]),"condition":read_json(root/"condition.json")["action_sha256"]})
    def permanent(path,value):
        if path.exists():
            if read_json(path)!=value:raise ValueError('Permanent teacher request must not be overwritten; use a new version/run')
        else:write_json(path,value)
    teacher=teacher_path(root)
    permanent(teacher/"request_map.json",{"slots":mapping,"cache_key":key,"image_count":124,"content":parts,"protocol":r["protocol"],"model":r["model"],"endpoint":r["endpoint"]})
    permanent(teacher/"request_payload.json",result)
    return result,key,mapping
