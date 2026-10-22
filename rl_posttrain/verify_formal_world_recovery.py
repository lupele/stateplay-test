"""Real seven-GPU small-resolution A recovery; zero API and zero formal conditions."""
import argparse
import copy
from pathlib import Path
import shutil
import torch
import yaml
from .io import ROOT,run_path,read_json,write_json,sha256,world_checkpoint
from .config import load_config
from .orchestrate import execute,torchrun
from .distributed import tree_hash
from .data.sampler import Ledger

def main():
 p=argparse.ArgumentParser();p.add_argument('--report-run',default='formal_software_acceptance_v1');a=p.parse_args()
 records=read_json(run_path('study_v1')/'conditions/catalog.json')
 values=load_config(ROOT/'rl_posttrain/configs/smoke.yaml');formal=load_config(ROOT/'rl_posttrain/configs/api_b_only.yaml')
 for section in ('world','data','reward','reward_model'):values[section]=copy.deepcopy(formal[section])
 continuous=run_path('formal_A_recovery_continuous');resumed=run_path('formal_A_recovery_resumed')
 for run in (continuous,resumed):
  (run/'plan').mkdir(parents=True,exist_ok=True);(run/'conditions').mkdir(exist_ok=True)
  if not (run/'conditions/catalog.json').exists():write_json(run/'conditions/catalog.json',records)
  config=run/'plan/development_config.yaml'
  text=yaml.safe_dump(values,sort_keys=False)
  if config.exists() and config.read_text()!=text:raise ValueError('Existing recovery development config changed')
  if not config.exists():config.write_text(text)
 if not world_checkpoint(continuous,'api').exists():
  execute(torchrun('rl_posttrain.train_api',['--config',str(continuous/'plan/development_config.yaml'),'--run-id',continuous.name,'--smoke','--max-optimizer-steps','3','--scorer','fake'],list(range(7))),continuous,'world_continuous',list(range(7)))
 checkpoints=sorted((continuous/'world/api/checkpoints').glob('step-00000001-*.pt'))
 start=next(path for path in checkpoints if not read_json(path.with_suffix('.json'))['state']['batch_done'])
 if not (resumed/'plan/cursor.json').exists():
  for folder in ('groups','conditions/anchors'):
   if (continuous/folder).exists():shutil.copytree(continuous/folder,resumed/folder,dirs_exist_ok=True)
  for path in (continuous/'plan').glob('batch-*-normalization.json'):shutil.copy2(path,resumed/'plan'/path.name)
  ledger=Ledger(resumed/'plan',records['records'],values['plan']['seed']);reservation=ledger.reserve(1,'api',0);ledger.phase('updating',update_cursor=1)
  copied=resumed/'world/api/checkpoints'/start.name;copied.parent.mkdir(parents=True,exist_ok=True)
  shutil.copy2(start,copied);shutil.copy2(start.with_suffix('.json'),copied.with_suffix('.json'))
  write_json(copied.parent/'latest.json',{'checkpoint':str(copied),'optimizer_step':1})
 if not world_checkpoint(resumed,'api').exists():
  execute(torchrun('rl_posttrain.train_api',['--config',str(resumed/'plan/development_config.yaml'),'--run-id',resumed.name,'--smoke','--max-optimizer-steps','3','--scorer','fake'],list(range(7))),resumed,'world_resumed',list(range(7)))
 left=torch.load(world_checkpoint(continuous,'api'),map_location='cpu',weights_only=False)
 right=torch.load(world_checkpoint(resumed,'api'),map_location='cpu',weights_only=False)
 comparisons={section:tree_hash(left[section])==tree_hash(right[section]) for section in ('live','policies','optimizer')}
 comparisons['cursor']=read_json(continuous/'plan/cursor.json')['cursor']==read_json(resumed/'plan/cursor.json')['cursor']==1
 comparisons['optimizer_step']=left['state']['optimizer_step']==right['state']['optimizer_step']==3
 comparisons['ranks']=len(left['rank_rng'])==len(right['rank_rng'])==7
 comparisons['all_rank_rng']=tree_hash(left['rank_rng'])==tree_hash(right['rank_rng'])
 report={'kind':'real_7GPU_implicit_world_small_resolution_recovery','paid_API_requests':0,'formal_conditions_consumed':0,
         'comparisons':comparisons,'passed':all(v for k,v in comparisons.items() if k not in ('all_rank_rng',)),
         'continuous_checkpoint':str(world_checkpoint(continuous,'api')),'resumed_checkpoint':str(world_checkpoint(resumed,'api')),
         'resume_optimizer_step':1,'final_optimizer_step':3,
         'sampling_resolution':[128,224],'original_resolution_proof':'Retained prior actual 480x832 modes/profile evidence; this is checkpoint/controller regression only'}
 write_json(run_path(a.report_run)/'reports/world_current_code_recovery.json',report);print(report,flush=True)
 if not report['passed']:raise SystemExit(1)
if __name__=='__main__':main()
