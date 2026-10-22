from pathlib import Path
import httpx
from ..io import write_json,read_json,sha256,digest
from .protocol import group_content
from .base import GroupScorer,parse_scores

def score_journal(group,version):
    return group.parent.parent/'reward_model/score_journal'/version/(group.name+'.json')

def save_scores(group,version,result):
    # Keep the exact raw response in two independent atomic files on shared storage.
    write_json(score_journal(group,version),result)
    write_json(group/'local_reward'/(version+'.json'),result)

def cached_scores(group,config,version):
    parts,mapping=group_content(group,config)
    content_hash=digest({'parts':parts,'mapping':mapping})
    path=group/'local_reward'/(version+'.json')
    try:value=read_json(path);repair=False
    except (FileNotFoundError,OSError):
        journal=score_journal(group,version)
        if not journal.exists():return None
        value=read_json(journal);repair=True
    if value.get('version')!=version or value.get('content_hash')!=content_hash or value.get('candidate_mapping')!=mapping:
        raise ValueError('Cached local reward input or model identity differs')
    scores=parse_scores(value['scores'])
    if repair:write_json(path,value)
    return scores

class CachedGroupScorer:
    """An update phase consumes durable scores without retaining a GPU server."""
    def __init__(self,config,version):self.config=config;self.version=version
    def score(self,group):
        value=cached_scores(group,self.config,self.version)
        if value is None:raise RuntimeError('Complete scoring phase before starting the update phase')
        return value

class QwenGroupScorer(GroupScorer):
    def __init__(self,config,version=None):
        self.config=config;self.url=config["reward_model"]["inference_url"];self.version=version
        health=httpx.get(self.url+"/health",timeout=30,trust_env=False).raise_for_status().json()
        if version and health["version"]!=version:raise ValueError("Reward version changed")
        if health.get('base_model')!=config['reward_model']['base_model'] or not health.get('frozen') or not health.get('full_multimodal'):raise ValueError('Scorer must be the specified frozen full local Qwen model')
        self.version=health["version"];self.model_id="Qwen3.5-9B-LoRA:"+self.version
    def score(self,group):
        parts,mapping=group_content(group,self.config);content_hash=digest({'parts':parts,'mapping':mapping})
        p=group/"local_reward"/(self.version+".json")
        if p.exists():
            cached=read_json(p)
            if cached.get('content_hash')!=content_hash or cached.get('candidate_mapping')!=mapping:raise ValueError('Cached local reward input identity differs or is unversioned')
            return parse_scores(cached["scores"])
        result=httpx.post(self.url+"/score",json={"group_path":str(group.resolve()),"version":self.version},timeout=1800,trust_env=False).raise_for_status().json()
        if result["version"]!=self.version:raise RuntimeError("Scorer version changed mid-batch")
        if result.get('candidate_mapping')!=mapping:raise ValueError('Local scorer candidate mapping differs')
        scores=parse_scores(result["scores"]);write_json(p,{**result,"scores":scores,"scorer":self.model_id,'content_hash':content_hash})
        return scores
