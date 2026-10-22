import copy
import os
from pathlib import Path
import yaml
from .io import ROOT, QWEN, SFT, SFT_SHA, digest, run_path

def load_config(path):
    p = Path(path)
    c = yaml.safe_load(p.read_text())
    if "extends" in c:
        base = load_config(p.parent / c.pop("extends"))
        def merge(a, b):
            for k, v in b.items():
                if isinstance(v, dict) and isinstance(a.get(k), dict): merge(a[k], v)
                else: a[k] = v
            return a
        c = merge(copy.deepcopy(base), c)
    # Public configurations contain relative paths and no provider credentials.
    paths = {
        "world": ("checkpoint", "tokenizer_checkpoint", "base_model"),
        "data": ("root", "metadata"),
        "reward": ("prompt",),
        "reward_model": ("base_model", "python"),
    }
    for section, keys in paths.items():
        for key in keys:
            if key in c.get(section, {}):
                value = Path(c[section][key]).expanduser()
                c[section][key] = str(value if value.is_absolute() else ROOT / value)
    for key in ("exclusion_archives", "exclusion_roots"):
        if key in c.get("data", {}):
            c["data"][key] = [str(Path(value) if Path(value).is_absolute() else ROOT / value)
                              for value in c["data"][key]]
    endpoint = os.environ.get("STATEPLAY_GPT_API_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if endpoint and "reward" in c:
        c["reward"]["endpoint"] = endpoint.rstrip("/")
    c["config_hash"] = digest({k: v for k, v in c.items() if k != "config_hash"})
    return c

def validate(c, run_id=None):
    for name in ("world", "data", "rollout", "nft", "reward", "reward_model", "plan", "topology"):
        if name not in c: raise ValueError(f"Missing required config section: {name}")
    if Path(c["world"]["checkpoint"]).resolve() != SFT.resolve(): raise ValueError("This study must start from step-4572 implicit SFT")
    if c["world"]["checkpoint_sha256"] != SFT_SHA: raise ValueError("SFT identity mismatch")
    if Path(c["reward_model"]["base_model"]).resolve() != QWEN.resolve(): raise ValueError("Qwen3.5-9B local full multimodal path is mandatory")
    if not c['reward_model'].get('local_files_only'):raise ValueError('Local-only Qwen loading is required')
    if not Path(c['reward']['prompt']).is_file():raise FileNotFoundError(c['reward']['prompt'])
    for key in ("checkpoint", "tokenizer_checkpoint", "base_model"):
        if not Path(c["world"][key]).exists(): raise FileNotFoundError(c["world"][key])
    for key in ("root", "metadata"):
        if not Path(c["data"][key]).exists(): raise FileNotFoundError(c["data"][key])
    if c["rollout"]["group_size"] != 4 or c["reward"]["frames_per_video"] != 24: raise ValueError("Protocol requires 4 candidates and 24 frames per video")
    if c["reward"]["max_side"] != 832 or c["reward"]["model"] != "gpt-6.1-sol": raise ValueError("Protocol/model change requires an explicit version and relabeling")
    if c["nft"]["mode"] not in ("B-only", "Coupling-only", "Both"): raise ValueError("Unknown update mode")
    if c["nft"]["beta"] <= 0 or c["nft"]["adv_clip"] <= 0: raise ValueError("Invalid NFT coefficients")
    if c["rollout"]["cfg_scale"] != 1 or c["rollout"]["action_cfg_scale"] != 1: raise ValueError("This NFT protocol uses conditional CFG=1")
    if c["plan"]["epochs"] != 1: raise ValueError("A+C share one pass; B consumes no new conditions")
    if not 0<c['plan']['api_conditions']<=c['plan']['api_max_conditions']:raise ValueError('API condition budget exceeds the approved plan maximum')
    if c['reward']['protocol'].startswith('mechanics-reward-v2') and c['data'].get('split_policy')!='source-action-temporal-v2':raise ValueError('Formal v2 requires the strengthened source leakage audit')
    if set(c["topology"]["world_gpus"]) & set(c["topology"]["reward_gpus"]) and c["topology"]["strategy"] != "timeslice": raise ValueError("World/RM GPU lists overlap")
    if run_id: run_path(run_id)
    if c["reward_model"]["max_length"] < 65536: raise ValueError("Full group must not be silently truncated")
    return c
