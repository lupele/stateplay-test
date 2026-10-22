"""Protocol/control tests with fixture scores, distinct from real-model acceptance."""
from pathlib import Path
import pytest
from rl_posttrain.io import write_json,read_json,reward_path,publish_link,world_checkpoint
from rl_posttrain.rewards.initial_audit import assess
from rl_posttrain.reward_model.metrics import summarize
from rl_posttrain.quality_gate import numeric_reasons

def test_reward_version_pointer_stays_inside_run(tmp_path):
 write_json(tmp_path/'reward_model/active.json',{'relative_path':'../../outside'})
 with pytest.raises(ValueError):reward_path(tmp_path)

def test_world_version_does_not_overwrite_legacy_checkpoint(tmp_path):
 final=tmp_path/'world/api/final';final.mkdir(parents=True)
 legacy=final/'world.pt';legacy.write_bytes(b'original-existing-world')
 new=final/'versions/new/world.pt';new.parent.mkdir(parents=True);new.write_bytes(b'new-world')
 write_json(final/'current.json',{'checkpoint':str(new)})
 assert world_checkpoint(tmp_path,'api')==new
 assert legacy.read_bytes()==b'original-existing-world'
 with pytest.raises(ValueError):publish_link(legacy,new)

def test_invalid_json_counts_against_rate_without_imputed_scores():
 rows=[{'error':'parse_failed','group_id':'one'}]
 m=summarize(rows,'actual-adapter')
 assert m['json_valid_rate']==0 and 'mae' not in m and 'permutation_test_passed' not in m
 assert numeric_reasons(m)

def test_first56_requires_numeric_and_real_manual_evidence(tmp_path):
 classes=['normal','macro_success','macro_fail','result_win','result_lose'];groups=[];points=[]
 for i in range(56):
  group=tmp_path/'groups'/f'g-{i}';groups.append(group)
  candidates=[{'video_id':f'candidate-{j}'} for j in range(4)]
  write_json(group/'manifest.json',{'group_id':group.name,'candidates':candidates,'condition':{'mechanism_class':classes[i%5]}})
  points.append({f'video_{j+1}':float(20+j*20+i%5) for j in range(4)})
 config={'reward_model':{'audit_initial_conditions':56},'rollout':{'conditions_per_batch':24}}
 gate=assess(tmp_path,config,groups,points,points,'v1')
 assert gate['status']=='awaiting_human_review' and gate['C_optimizer_updates_before_gate']==0
 assert gate['ordinary_conditions_per_batch']==24 and gate['audit_buffer_conditions']==56
 write_json(tmp_path/'reports/C_initial_audit/human_review.json',{'version':'v1','reviewer':'mock test reviewer, not actual acceptance',
     'reviewed_candidate_ids':[g.name+'/candidate-0' for g in groups],'severe_mechanics_violations':[]})
 assert assess(tmp_path,config,groups,points,points,'v1')['passed']
 bad=[{k:100-v for k,v in row.items()} for row in points]
 assert assess(tmp_path,config,groups,points,bad,'v1')['status']=='failed_numeric_quality'
