"""One frozen full multimodal Qwen replica per selected GPU, then exit and release VRAM."""
import argparse
import os
import time
import json
from pathlib import Path
import torch
from .model import load_model,move,vision_features,score_batch
from .collator import GroupCollator
from ..config import load_config
from ..io import run_path,read_json,write_json,reward_checkpoint,sha256,digest
from ..rewards.protocol import group_content
from ..rewards.qwen_group import cached_scores,save_scores
from ..phase_progress import PhaseProgress

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.batch_size<1:raise ValueError('reward batch size must be positive')
    c=load_config(a.config);run=run_path(a.run_id);reservation=read_json(run/'plan/cursor.json')['inflight']
    if reservation is None:raise RuntimeError('No in-flight conditions to score')
    rank=int(os.environ.get('RANK',0));size=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    torch.cuda.set_device(local);torch.set_num_threads(4)
    cp=reward_checkpoint(run);manifest=read_json(cp/'manifest.json');version=manifest['version']
    if manifest.get('smoke') and not a.smoke:raise ValueError('Smoke reward model cannot score formal C')
    if manifest['base_model']!=c['reward_model']['base_model'] or manifest['protocol']!=c['reward']['protocol']:raise ValueError('Reward model/protocol mismatch')
    if sha256(cp/'adapter/adapter_model.safetensors')!=manifest['adapter_sha256'] or sha256(c['reward']['prompt'])!=manifest['prompt_sha256']:raise ValueError('Reward weights or prompt changed')
    groups=[run/'groups'/f"g-{reservation['global_condition_offset']+i:07d}-{cid}" for i,cid in enumerate(reservation['condition_ids'])]
    # Round-robin groups keeps every GPU active even for a short final batch.
    own=groups[rank::size];pending=[g for g in own if cached_scores(g,c,version) is None]
    progress=PhaseProgress(run,reservation['batch'],'score',rank,len(own),len(own)-len(pending))
    if pending:
        progress.write('loading')
        model,processor,audit=load_model(c,device=f'cuda:{local}',adapter_path=cp/'adapter')
        collator=GroupCollator(processor,c);torch.cuda.reset_peak_memory_stats()
        progress.start_work()
        for start in range(0,len(pending),a.batch_size):
            chunk=pending[start:start+a.batch_size];tick=time.monotonic()
            values,metas,individual=collator.batch(chunk);features=[]
            for value,meta in zip(individual,metas):
                features.extend(vision_features(model,value,run/'reward_model/feature_cache'/(meta['input_hash']+'.safetensors')))
            preds,texts=score_batch(model,processor,move(values,f'cuda:{local}'),tuple(features),c['reward_model']['max_new_tokens'])
            torch.cuda.synchronize(local);seconds=time.monotonic()-tick
            for g,meta,pred,text in zip(chunk,metas,preds,texts):
                parts,mapping=group_content(g,c)
                save_scores(g,version,{'version':version,'scores':pred,'raw_output':text,
                    'input_hash':meta['input_hash'],'candidate_mapping':mapping,'image_count':124,
                    'content_hash':digest({'parts':parts,'mapping':mapping}),'scorer':'Qwen3.5-9B-LoRA:'+version,
                    'full_multimodal':True,'request_batch_size':len(chunk),'batch_seconds':seconds})
                progress.advance()
        memory={'rank':rank,'gpu':local,'full_multimodal':audit['full_multimodal'],'local_files_only':True,
             'batch_size':a.batch_size,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
             'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'gpu_total_bytes':torch.cuda.get_device_properties(local).total_memory}
        progress.extra={'memory':memory}
        print('Reward phase memory '+json.dumps(memory),flush=True)
        try:write_json(run/'progress'/f"batch-{reservation['batch']:06d}"/f'score-memory-rank-{rank}.json',memory)
        except OSError:pass  # Also present in progress snapshot and worker log.
    progress.finish()

if __name__=='__main__':main()
