"""Mock dispatch/accounting tests; no GPU model or paid API is started here."""
from pathlib import Path
from unittest.mock import patch
import copy
import pytest
from rl_posttrain.config import load_config
from rl_posttrain.io import ROOT,write_json,read_json,reward_path,sha256
from rl_posttrain.formal import run_plan,FormalPause,run_command,check_ledger,reward_training,epoch_manifest,owned_alive

def metrics(version='v1',mae=5):
 return {'version':version,'smoke':False,'json_valid_rate':1,'mae':mae,'spearman_global':.9,
         'pair_agreement':.9,
         'per_class':{k:{'mae':5} for k in ('normal','macro_success','macro_fail','result_win','result_lose')}}

def prepared(tmp_path,mae=5):
 c=load_config(ROOT/'rl_posttrain/configs/api_b_only.yaml')
 write_json(tmp_path/'plan/epoch_order.json',{'order':list(range(10000))})
 write_json(tmp_path/'plan/cursor.json',{'cursor':2000,'api_consumed':2000,'local_consumed':0,'inflight':None,'batch':72})
 world=tmp_path/'world/api/final/world.pt';world.parent.mkdir(parents=True);world.touch()
 write_json(world.with_suffix('.json'),{'state':{'smoke':False}})
 write_json(tmp_path/'reward_model/active.json',{'relative_path':'versions/a-002000','A_conditions':2000})
 rm=reward_path(tmp_path)
 write_json(rm/'selected/manifest.json',{'smoke':False,'version':'v1'})
 write_json(rm/'reports/evaluation.json',metrics(mae=mae))
 write_json(tmp_path/'reports/joint_profile.json',{'selected_topology':{'strategy':'dedicated','world_gpus':[0,1,2,3,4,5],'reward_gpus':[6],'conditions_per_batch':24}})
 return c

def test_manual_pause_never_repeats_A_B_or_spends_extension(tmp_path):
 c=prepared(tmp_path)
 with patch('rl_posttrain.train_world.formal_gate'),patch('rl_posttrain.formal.reward_training'),patch('rl_posttrain.review.build_review',return_value=tmp_path/'review.json'),patch('rl_posttrain.quality_gate.evaluate_gate',return_value={'passed':False,'reasons':['human review pending']}):
  calls=[]
  with pytest.raises(FormalPause,match='Actual human review'):
   run_plan(c,tmp_path,'config',lambda *a,**kw:calls.append(a) or 0,'world','reward',auto_extend=True)
  assert not calls
 assert read_json(tmp_path/'plan/stage_state.json')['stage']=='awaiting_reward_human_review'
 assert read_json(tmp_path/'plan/cursor.json')['cursor']==2000

def test_after_review_resume_only_C_export_protection(tmp_path):
 c=prepared(tmp_path);names=[]
 def execute(command,run,name,*a,**kw):
  names.append(name)
  if name=='formal_C':write_json(run/'plan/cursor.json',{'cursor':10000,'api_consumed':2000,'local_consumed':8000,'inflight':None,'batch':406})
  return 0
 with patch('rl_posttrain.train_world.formal_gate'),patch('rl_posttrain.formal.reward_training'),patch('rl_posttrain.review.build_review',return_value=tmp_path/'review.json'),patch('rl_posttrain.quality_gate.evaluate_gate',return_value={'passed':True}):
  run_plan(c,tmp_path,'config',execute,'world','reward',auto_extend=True)
 assert names==['formal_C','formal_C_export','formal_source_protection_after']
 assert read_json(tmp_path/'plan/stage_state.json')['stage']=='complete'

def test_wrong_evaluation_version_does_not_launch_C(tmp_path):
 c=prepared(tmp_path);write_json(reward_path(tmp_path)/'reports/evaluation.json',metrics('other'))
 with patch('rl_posttrain.train_world.formal_gate'),patch('rl_posttrain.formal.reward_training'):
  with pytest.raises(ValueError,match='versions differ'):run_plan(c,tmp_path,'config',lambda *a,**kw:pytest.fail('Training launched'),'world','reward')

@pytest.mark.parametrize('cap,expected_blocks,final_cursor',[(4000,7,3960),(2840,3,2840),(2280,1,2280)])
def test_authorized_extensions_use_new_conditions_and_stop_at_cap(tmp_path,cap,expected_blocks,final_cursor):
 c=prepared(tmp_path,mae=20);c['plan']['api_max_conditions']=cap;blocks=[]
 def train(values,run,*a):
  rm=reward_path(run);write_json(rm/'selected/manifest.json',{'smoke':False,'version':'v1'})
  write_json(rm/'reports/evaluation.json',metrics(mae=20))
 def execute(command,run,name,*a,**kw):
  assert name.startswith('formal_A_to_')
  target=int(name.split('_')[-1]);s=read_json(run/'plan/cursor.json')
  blocks.append(target-s['api_consumed']);s.update(api_consumed=target,cursor=target);write_json(run/'plan/cursor.json',s);return 0
 write_json(tmp_path/'reports/formal_140_health.json',{'passed':True})
 with patch('rl_posttrain.train_world.formal_gate'),patch('rl_posttrain.formal.reward_training',side_effect=train),patch('rl_posttrain.review.build_review',return_value=tmp_path/'review.json'):
  with pytest.raises(FormalPause,match='remaining authorized budget'):
   run_plan(c,tmp_path,'config',execute,'world','reward',auto_extend=True)
 assert blocks==[280]*expected_blocks
 assert read_json(tmp_path/'plan/cursor.json')['cursor']==final_cursor
 assert read_json(tmp_path/'plan/stage_state.json')['C_budget']==10000-final_cursor
 assert len(list((tmp_path/'reward_model/versions').iterdir()))==expected_blocks+1
 assert read_json(tmp_path/'plan/stage_state.json')['proposal']['unused_API_allowance']==cap-final_cursor

def test_reward_crash_retry_uses_newest_committed_optimizer(tmp_path):
 seen=[]
 def execute(command,run,*a,**kw):
  seen.append(command)
  if len(seen)==1:
   write_json(reward_path(run)/'checkpoints/latest.json',{'checkpoint':'newest-step'})
   return 1
  return 0
 with patch('rl_posttrain.formal.time.sleep'):
  run_command(['reward','-m','rl_posttrain.reward_model.train'],tmp_path,'formal_B',list(range(7)),execute)
 assert '--resume' not in seen[0]
 assert seen[1][-2:]==['--resume','newest-step']

def test_inconsistent_budget_is_rejected(tmp_path):
 prepared(tmp_path);s=read_json(tmp_path/'plan/cursor.json');s['cursor']=1999;write_json(tmp_path/'plan/cursor.json',s)
 with pytest.raises(ValueError,match='accounting mismatch'):check_ledger(tmp_path,4000)

def test_epoch_selection_uses_validation_and_preserves_both_checkpoints(tmp_path):
 c=load_config(ROOT/'rl_posttrain/configs/api_b_only.yaml');rm=reward_path(tmp_path)
 final={'schema':'rl-reward-checkpoint-v1','smoke':False,'base_model':c['reward_model']['base_model'],
        'protocol':c['reward']['protocol'],'prompt_sha256':sha256(c['reward']['prompt']),'config':c,
        'audit':{'base_index_sha256':'fixture-base','processor_hash':'fixture-processor'},
        'dataset_hash':'fixture-groups','steps_per_epoch':1,'world_size':7,'total_steps':2}
 write_json(rm/'final/manifest.json',final)
 snapshots=[]
 for i in (1,2):
  cp=rm/'checkpoints'/f'step-{i:08d}';(cp/'adapter').mkdir(parents=True);(cp/'processor').mkdir()
  (cp/'adapter/adapter_model.safetensors').write_bytes(bytes([i]));(cp/'training_state.pt').write_bytes(b'fixture-state')
  em=epoch_manifest(cp,final,c)
  write_json(rm/'reports'/em['version']/'validation/metrics.json',{'groups':200,'json_valid_rate':1,'mae':5 if i==1 else 9})
  snapshots.append(sha256(cp/'adapter/adapter_model.safetensors'))
 reward_training(c,tmp_path,'config',lambda *a,**kw:pytest.fail('Validation results were already cached'),'reward')
 chosen=read_json(rm/'selected/manifest.json')
 assert chosen['steps']==1
 assert read_json(rm/'reports/selection.json')['test_used_for_selection'] is False
 assert (rm/'selected/selection_complete.json').exists()
 assert snapshots==[sha256(rm/'checkpoints'/f'step-{i:08d}/adapter/adapter_model.safetensors') for i in (1,2)]
 reward_training(c,tmp_path,'config',lambda *a,**kw:pytest.fail('Completed B repeated'),'reward')

def test_worker_ownership_checks_run_and_pid_start_time(tmp_path):
 import subprocess,sys
 process=subprocess.Popen([sys.executable,'-c','import time; time.sleep(10)','--run-id',tmp_path.name])
 try:
  fields=Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()
  record={'pid':process.pid,'start_tick':fields[19],'active':True}
  assert owned_alive(record,tmp_path)
  assert not owned_alive({**record,'start_tick':'wrong-start-time'},tmp_path)
  assert not owned_alive(record,tmp_path/'other-run')
 finally:
  process.terminate();process.wait(timeout=5)


def test_budget_revision_preserves_previous_reward_config(tmp_path):
 from rl_posttrain.formal import save_config
 c=prepared(tmp_path)
 previous=copy.deepcopy(c);previous['plan']['api_max_conditions']=2840;previous['stage']='reward_model'
 original=Path(save_config(tmp_path,'formal_B_data_002000.yaml',previous));original_bytes=original.read_bytes()
 c['plan']['api_max_conditions']=2280
 with patch('rl_posttrain.train_world.formal_gate'),patch('rl_posttrain.formal.reward_training'),patch('rl_posttrain.review.build_review',return_value=tmp_path/'review.json'),patch('rl_posttrain.quality_gate.evaluate_gate',return_value={'passed':False,'reasons':['human review pending']}):
  with pytest.raises(FormalPause,match='Actual human review'):
   run_plan(c,tmp_path,'config',lambda *args,**kwargs:pytest.fail('Existing completed training repeated'),'world','reward',auto_extend=True)
 assert original.read_bytes()==original_bytes
 amended=list((tmp_path/'plan').glob('formal_B_data_002000_*.yaml'))
 assert len(amended)==1
 assert load_config(amended[0])['plan']['api_max_conditions']==2280


@pytest.mark.parametrize('mae',[5,20])
def test_user_hold_finishes_B_reports_without_C_or_A_extension(tmp_path,mae):
 c=prepared(tmp_path,mae=mae)
 write_json(tmp_path/'plan/stage_C_control.json',{'allow_start':False,'reason':'user revised plan pending'})
 with patch('rl_posttrain.train_world.formal_gate'),patch('rl_posttrain.formal.reward_training') as train,patch('rl_posttrain.review.build_review',return_value=tmp_path/'review.json') as review,patch('rl_posttrain.quality_gate.evaluate_gate',side_effect=AssertionError('User hold bypassed')):
  with pytest.raises(FormalPause,match='User requested stop after Stage B'):
   run_plan(c,tmp_path,'config',lambda *args,**kwargs:pytest.fail('C or A extension launched during user hold'),'world','reward',auto_extend=True,wait_human=True)
  train.assert_called_once();review.assert_called_once()
 state=read_json(tmp_path/'plan/stage_state.json')
 assert state['stage']=='paused_after_B_user_request' and state['B_completed'] and not state['C_started']
 assert read_json(tmp_path/'plan/cursor.json')['cursor']==2000
 assert not (tmp_path/'reports/formal_completion.json').exists()
