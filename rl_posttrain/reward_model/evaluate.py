"""Resumable distributed validation/test; only the test issues a quality report."""
import argparse
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from .dataset import build_dataset
from .collator import GroupCollator
from .model import load_model,move,vision_features,score,verify_adapter_reload
from .metrics import summarize,canonical_identity,load_cached_canonical
from ..config import load_config
from ..io import read_json,write_json,run_path,reward_path,sha256,digest,reward_checkpoint

def evaluate_item(item, model, processor, collator, c, run, version):
    group=Path(item['group_path']);values,meta=collator(group,training=False)
    result={'group':str(group),'group_id':item['group_id'],'teacher':item['scores'],
            'mechanism_class':item['mechanism_class'],'version':version,
            'input_hash':meta['input_hash'],'images':meta['image_count'],'tokens':meta['prefix_tokens']}
    features=vision_features(model,values,run/'reward_model/feature_cache'/(meta['input_hash']+'.safetensors'))
    try:
        pred,text=score(model,processor,move(values,'cuda:'+str(torch.cuda.current_device())),features,c['reward_model']['max_new_tokens'])
        result.update(scores=pred,raw_output=text)
    except ValueError as error:
        result.update(error=type(error).__name__,raw_output=getattr(error,'raw_output',None));return result
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--checkpoint');p.add_argument('--check-reload',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--human-review')
    p.add_argument('--split',choices=['validation','test'],default='test')
    p.add_argument('--shortcut-checks',action='store_true')
    p.add_argument('--resume');p.add_argument('--max-optimizer-steps',type=int);a=p.parse_args()
    rank=int(os.environ.get('RANK',0));size=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    torch.cuda.set_device(local);torch.set_num_threads(6)
    if size>1:dist.init_process_group('gloo',timeout=__import__('datetime').timedelta(hours=6))
    c=load_config(a.config);run=run_path(a.run_id);rm=reward_path(run)
    cp=Path(a.checkpoint) if a.checkpoint else reward_checkpoint(run);m=read_json(cp/'manifest.json')
    if m['protocol']!=c['reward']['protocol']:raise ValueError('Reward protocol mismatch')
    if sha256(cp/'adapter/adapter_model.safetensors')!=m['adapter_sha256'] or sha256(c['reward']['prompt'])!=m['prompt_sha256']:raise ValueError('Reward adapter or prompt identity changed')
    data=build_dataset(run,config=c,allow_legacy=bool(a.smoke or m['smoke']))
    if m.get('dataset_hash') and m['dataset_hash']!=digest(data['groups']):raise ValueError('Evaluation teacher dataset changed after training')
    groups=data['groups'][:1] if a.smoke or m['smoke'] else data['splits'][a.split]
    if not groups:raise ValueError('Independent reward '+a.split+' split is empty')
    model,processor,audit=load_model(c,device=f'cuda:{local}',adapter_path=cp/'adapter',training=False)
    collator=GroupCollator(processor,c);reload_audit=verify_adapter_reload(model,cp/'adapter') if a.check_reload else None
    buffers={n:b.detach().cpu().clone() for n,b in model.named_buffers()}
    reports=rm/'reports'/m['version']/a.split;reports.mkdir(parents=True,exist_ok=True)
    torch.cuda.reset_peak_memory_stats();started=time.monotonic();results=[]
    for item in groups[rank::size]:
        cache=reports/'canonical_groups'/(item['group_id']+'.json')
        identity=canonical_identity(m['version'],item)
        result=load_cached_canonical(reports,item,m['version'])
        if result is None:
            result=evaluate_item(item,model,processor,collator,c,run,m['version'])
            result['evaluation_identity']=identity;write_json(cache,result)
        results.append(result);print('Reward',a.split,rank,'group',item['group_id'],'valid',bool('scores' in result),flush=True)
    for name,buffer in model.named_buffers():torch.testing.assert_close(buffer.detach().cpu(),buffers[name],rtol=0,atol=0)
    memory={'rank':rank,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'gpu_total_bytes':torch.cuda.get_device_properties(local).total_memory,'seconds':time.monotonic()-started}
    gathered=[None]*size if rank==0 else None
    if size>1:dist.gather_object((results,memory),gathered,dst=0)
    else:gathered=[(results,memory)]
    if rank==0:
        combined=sorted((r for rows,_ in gathered for r in rows),key=lambda r:r['group_id'])
        metrics=summarize(combined,m['version'],bool(a.smoke or m['smoke']))
        metrics.update(memory);metrics['rank_profiles']=[v for _,v in gathered]
        metrics['peak_allocated_bytes']=max(v['peak_allocated_bytes'] for _,v in gathered)
        metrics['peak_reserved_bytes']=max(v['peak_reserved_bytes'] for _,v in gathered)
        metrics.update(split=a.split,dataset_hash=digest(data['groups']),reload_adapter_tensors=reload_audit,
                       frozen_buffers_unchanged=True,full_multimodal=True,local_files_only=True)
        write_json(reports/'metrics.json',metrics)
        write_json(rm/'reports'/(a.split+'.json'),metrics)
        if a.split=='test':
            if a.shortcut_checks:
                from .shortcuts import probe
                probe(run,c,combined,model,processor,collator,f'cuda:{local}',m['version'])
            write_json(rm/'reports/evaluation.json',metrics)
            write_json(run/'reports/reward_profile.json',{**metrics,'kind':'real_Qwen3.5_9B_full_group','image_count':124})
            from ..quality_gate import evaluate_gate
            gate=evaluate_gate(run,metrics,m['version'],a.human_review)
            print({'test_groups':metrics['groups'],'mae':metrics.get('mae'),'quality_gate_passed':gate['passed']},flush=True)
        else:print({'validation_groups':metrics['groups'],'mae':metrics.get('mae')},flush=True)
    if size>1:dist.destroy_process_group()
if __name__=='__main__':main()
