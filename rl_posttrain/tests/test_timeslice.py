"""CPU control/shape tests. Real four-GPU acceptance is recorded separately."""
import argparse
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from rl_posttrain.timeslice import gpu_ids,runtime_config,command,phases_needed
from rl_posttrain.config import load_config
from rl_posttrain.io import ROOT
from rl_posttrain.train_world import schedule
from rl_posttrain.data.sampler import Ledger
from rl_posttrain.phase_progress import PhaseProgress,snapshot
from rl_posttrain.reward_model.collator import GroupCollator
from rl_posttrain.adapters.world_model import WorldAdapter
from stateplay.utils.scheduler import FlowMatchScheduler

def test_gpu_selection_and_independent_one_pass_budget():
    assert gpu_ids('3,1,5,0')==[3,1,5,0]
    for bad in ['0,0','-1,2','','a,1']:
        with pytest.raises(argparse.ArgumentTypeError):gpu_ids(bad)
    base=load_config(ROOT/'rl_posttrain/configs/local_b_only.yaml')
    c=runtime_config(base,[0,2,3,5],2,2)
    assert c['rollout']==base['rollout'] and c['plan']==base['plan'] and c['nft']==base['nft']
    assert c['topology']['world_gpus']==c['topology']['reward_gpus']==[0,2,3,5]
    assert c['rollout']['conditions_per_batch']==28
    for phase in ('generate','score','update'):
        cmd=command(phase,'config','test',[0,2,3,5],'version')
        assert cmd[cmd.index('--nproc_per_node')+1]=='4'

def test_four_gpu_ledger_visits_once_including_tail(tmp_path):
    records=[{'condition_id':str(i),'mechanism_class':'normal'} for i in range(101)]
    ledger=Ledger(tmp_path,records);seen=[]
    while batch:=ledger.reserve(28,'local',0):
        seen.extend(batch['condition_ids']);ledger.commit('checkpoint-path-in-control-test')
    assert len(seen)==len(set(seen))==101
    assert ledger.state()['local_consumed']==101 and ledger.state()['cursor']==101

def test_training_schedule_unchanged_by_inference_batch(config):
    samples=list(range(96));before=schedule(samples,config,4,82)
    config['stage_c']={'generation_batch_size':2,'reward_batch_size':2}
    assert schedule(samples,config,4,82)==before
    assert sum(x['valid'] for x in before)==96*config['nft']['inner_epochs']

def test_progress_cache_is_not_counted_as_new_throughput(tmp_path):
    p=PhaseProgress(tmp_path,3,'generate',0,8,4);p.advance(2);p.finish()
    s=snapshot(tmp_path,3,'generate')
    assert s['completed']==6 and s['new_completed']==2 and s['total']==8

def test_observer_file_failure_does_not_abort_training(tmp_path):
    p=PhaseProgress(tmp_path,0,'score',0,2)
    with patch('rl_posttrain.phase_progress.write_json',side_effect=OSError('shared storage telemetry')):
        p.advance()
    assert p.done==1
    with patch('rl_posttrain.phase_progress.read_json',side_effect=FileNotFoundError):
        assert snapshot(tmp_path,0,'score')['total']==0

def test_exact_score_journal_repairs_missing_primary(tmp_path,config):
    from rl_posttrain.tests.conftest import fake_group
    from rl_posttrain.rewards.qwen_group import cached_scores,save_scores
    from rl_posttrain.rewards.protocol import group_content
    from rl_posttrain.io import digest,read_json
    group=fake_group(tmp_path/'groups',config);parts,mapping=group_content(group,config)
    result={'version':'fixture','scores':{'video_1':10,'video_2':20,'video_3':30,'video_4':40},
        'raw_output':'exact original response','candidate_mapping':mapping,'content_hash':digest({'parts':parts,'mapping':mapping})}
    save_scores(group,'fixture',result);(group/'local_reward/fixture.json').unlink()
    assert cached_scores(group,config,'fixture')==result['scores']
    assert read_json(group/'local_reward/fixture.json')==result

def test_candidate_batch_seed_and_master_storage():
    a=WorldAdapter.__new__(WorldAdapter);a.device=torch.device('cpu');a.dtype=torch.bfloat16
    a.dit=torch.nn.Linear(2,2);a.scheduler=FlowMatchScheduler();a.scheduler.set_timesteps(2)
    original={n:p.detach().clone() for n,p in a.dit.named_parameters()};ptrs={n:p.data_ptr() for n,p in a.dit.named_parameters()}
    a.forward=lambda x,*args:{'pred_video':torch.zeros_like(x)}
    first=torch.zeros(1,48,1,2,2,dtype=torch.bfloat16);context=torch.zeros(1,3,4);actions=torch.zeros(1,9,11)
    batch=a.sample(first,context,actions,3,[11,29])
    separate=torch.cat([a.sample(first,context,actions,3,s) for s in [11,29]])
    assert torch.equal(batch,separate) and not torch.equal(batch[0,:,1:],batch[1,:,1:])
    for n,p in a.dit.named_parameters():assert torch.equal(p,original[n]) and p.data_ptr()==ptrs[n]

def test_reward_batch_left_padding_and_image_boundaries():
    values=[{'input_ids':torch.tensor([[1,2,3]]),'attention_mask':torch.ones(1,3,dtype=torch.long),
        'mm_token_type_ids':torch.tensor([[0,1,1]]),'pixel_values':torch.ones(124,2),
        'image_grid_thw':torch.ones(124,3,dtype=torch.long)},
        {'input_ids':torch.tensor([[4,5]]),'attention_mask':torch.ones(1,2,dtype=torch.long),
        'mm_token_type_ids':torch.tensor([[1,1]]),'pixel_values':torch.full((124,2),2.),
        'image_grid_thw':torch.ones(124,3,dtype=torch.long)}]
    class Fixture(GroupCollator):
        def __call__(self,g,training=False):return values[g],{'group':g}
    c=Fixture(SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0)),{})
    merged,metas,individual=c.batch([0,1])
    assert merged['input_ids'].tolist()==[[1,2,3],[0,4,5]]
    assert merged['attention_mask'].tolist()==[[1,1,1],[0,1,1]]
    assert merged['mm_token_type_ids'].tolist()==[[0,1,1],[0,1,1]]
    assert merged['image_grid_thw'].shape[0]==248
    assert torch.equal(merged['pixel_values'][:124],values[0]['pixel_values'])
    assert torch.equal(merged['pixel_values'][124:],values[1]['pixel_values'])

def test_resume_skips_finished_phases_and_keeps_partial_update():
    assert phases_needed(None)==['generate','score','update']
    assert phases_needed({'phase':'generated'})==['score','update']
    assert phases_needed({'phase':'updating','update_cursor':1})==['update']

def test_phase_score_parser_keeps_all_batch_outputs():
    import transformers
    if not hasattr(transformers,'Qwen3_5ForConditionalGeneration'):pytest.skip('Runs in the separate REWARD environment')
    from rl_posttrain.reward_model.model import score_batch
    outputs=torch.tensor([[1,2,3,11],[4,5,6,22]])
    model=SimpleNamespace(generate=lambda **kwargs:outputs)
    processor=SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda row,**kwargs:'{"video_1":'+str(int(row[0]))+',"video_2":20,"video_3":30,"video_4":40}'))
    with patch('rl_posttrain.reward_model.model.cached_vision',return_value=__import__('contextlib').nullcontext()),patch('rl_posttrain.reward_model.batch_attention.compact_batch_attention',return_value=__import__('contextlib').nullcontext()):
        rows,texts=score_batch(model,processor,{'input_ids':outputs[:,:3]},(),10)
    assert rows[0]['video_1']==11 and rows[1]['video_1']==22

@pytest.mark.parametrize('decode',[False,True])
def test_compact_request_attention_matches_dense_mask(decode):
    from rl_posttrain.reward_model.batch_attention import ragged_sdpa
    torch.manual_seed(41);keys=torch.randn(2,2,7,4);values=torch.randn(2,2,7,4)
    queries=torch.randn(2,2,1 if decode else 7,4);mask=torch.tensor([[0,0,1,1,1,1,1],[1,1,1,1,1,1,1]])
    causal=torch.ones(1 if decode else 7,7,dtype=torch.bool)
    if not decode:causal=causal.tril()
    dense=causal[None,None]&mask[:,None,None,:].bool()
    expected=torch.nn.functional.scaled_dot_product_attention(queries,keys,values,attn_mask=dense,scale=.3)
    actual,_=ragged_sdpa(None,queries,keys,values,mask,scaling=.3)
    for index,pad in enumerate([2,0]):
        torch.testing.assert_close(actual[index,0 if decode else pad:],expected.transpose(1,2)[index,0 if decode else pad:])
