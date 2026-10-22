"""Canonical optimizer-boundary checkpoint. Old/live/ref and each rank's RNG survive."""
from pathlib import Path
import random
import numpy as np
import torch
from ..io import atomic_torch, write_json, read_json, digest, protect, sha256
from .parameter_modes import trainable_state

def rng_state():return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
def restore_rng(state):
    random.setstate(state["python"]);np.random.set_state(state["numpy"]);torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        # A runs on seven visible devices; dedicated C may expose only six.
        # Restore corresponding surviving device streams without initializing
        # a nonexistent logical GPU or resetting the optimizer/cursor.
        torch.cuda.set_rng_state_all(state["cuda"][:torch.cuda.device_count()])

def save(path,model,optimizer,bank,state,config,rank_rng=None):
    protect(path)
    payload={"schema":"rl-world-checkpoint-v1","live":trainable_state(model),"policies":bank.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":None,"scaler":None,"state":state,"rank_rng":rank_rng or [rng_state()],"config":config,"sft_sha256":config["world"]["checkpoint_sha256"],"tokenizer_sha256":sha256(config["world"]["tokenizer_checkpoint"])}
    atomic_torch(path,payload)
    write_json(Path(path).with_suffix(".json"),{"sha256":sha256(path),"schema":payload["schema"],"state":state,"mode":config["nft"]["mode"],"config_hash":config["config_hash"],"sft_sha256":payload["sft_sha256"]})

def load(path,model,optimizer,bank,config,rank=0,restore_random=True):
    metadata=read_json(Path(path).with_suffix(".json"))
    if sha256(path)!=metadata["sha256"]:raise ValueError("Checkpoint checksum mismatch")
    state=torch.load(path,map_location="cpu",weights_only=False)
    if 'backend' in state['state']:
        from ..reproducibility import describe
        if state['state']['backend']!=describe():raise ValueError('World resume compute backend changed')
    if state["sft_sha256"]!=config["world"]["checkpoint_sha256"] or state["config"]["nft"]["mode"]!=config["nft"]["mode"]:raise ValueError("Checkpoint base/mode mismatch")
    if state["tokenizer_sha256"]!=sha256(config["world"]["tokenizer_checkpoint"]):raise ValueError("Tokenizer changed")
    # Stage/topology may change; algorithm, sampling and frame protocol must remain identical.
    for section in ("world","nft","reward"):
        if state["config"][section]!=config[section]:raise ValueError(f"Resume changed {section}")
    if state['config']['data']!=config['data']:raise ValueError('Resume changed training sources or leakage policy')
    if state["config"]["rollout"]!={**config["rollout"],"conditions_per_batch":state["config"]["rollout"]["conditions_per_batch"]}:raise ValueError("Resume changed rollout protocol")
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    if set(params)!=set(state["live"]):raise ValueError("Trainable names changed")
    with torch.no_grad():
        for n,p in params.items():p.copy_(state["live"][n].to(p.device))
    bank.load_state_dict(state["policies"]);optimizer.load_state_dict(state["optimizer"])
    if restore_random:
        # Topology changes derive deterministic RNG for newly introduced ranks and log it.
        if rank<len(state["rank_rng"]):restore_rng(state["rank_rng"][rank])
        else:torch.manual_seed(config["plan"]["seed"]+rank+state["state"]["optimizer_step"]*100003)
        if torch.cuda.is_available():state["state"]["rng_topology_mapping"]={"saved_visible_devices":len(state["rank_rng"][min(rank,len(state["rank_rng"])-1)]["cuda"]),"current_visible_devices":torch.cuda.device_count(),"kept_logical_devices":list(range(torch.cuda.device_count()))}
    return state["state"]
