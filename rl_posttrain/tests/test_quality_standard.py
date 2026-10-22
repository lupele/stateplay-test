"""Software-only acceptance tests; no real review, GPU or API is substituted."""
import copy
from pathlib import Path
import pytest
from rl_posttrain.io import write_json,read_json,digest,sha256
from rl_posttrain.quality_gate import numeric_reasons,evaluate_gate,QUALITY_STANDARD
from rl_posttrain.reward_model.metrics import summarize,load_cached_canonical,canonical_identity
from rl_posttrain.rewards.initial_audit import assess


def acceptable(mae=15):
 return {'version':'fixture-model','smoke':False,'json_valid_rate':1,'mae':mae,
         'spearman_global':.8,'pair_agreement':.8,
         'per_class':{k:{'mae':15} for k in ('normal','macro_success','macro_fail','result_win','result_lose')}}


@pytest.mark.parametrize('mae,expected_failure',[(12.56,False),(15,False),(15.0001,True)])
def test_overall_mae_uses_inclusive_fifteen(mae,expected_failure):
 reasons=numeric_reasons(acceptable(mae))
 assert bool(reasons)==expected_failure
 if expected_failure:assert reasons==['MAE exceeds 15 points']


def test_old_order_diagnostics_do_not_affect_current_numeric_standard():
 m=acceptable();m.update(permutation_test_passed=False,permutation_max_abs_deltas=[60])
 assert numeric_reasons(m)==[]


def test_manual_review_is_still_required_under_new_standard(tmp_path):
 report=evaluate_gate(tmp_path,acceptable(),'fixture-model')
 assert not report['passed'] and report['numeric_passed']
 assert report['quality_standard']['mae_max']==15
 assert any('Human review' in reason for reason in report['reasons'])
 assert not any('permutation' in reason.lower() or 'MAE exceeds' in reason for reason in report['reasons'])


def test_real_candidate_ids_and_version_matched_review_are_not_waived(tmp_path):
 groups=[];ids=[]
 for i in range(25):
  group=tmp_path/'groups'/f'g-{i}';candidates=[{'video_id':f'v-{j}'} for j in range(4)]
  write_json(group/'manifest.json',{'group_id':group.name,'candidates':candidates})
  groups.append({'group_path':str(group)});ids.extend(group.name+'/'+r['video_id'] for r in candidates)
 write_json(tmp_path/'reward_model/splits/manifest.json',{'groups':groups})
 audit=tmp_path/'conditions/leakage_index.json';write_json(audit,{'software_fixture':True})
 review=tmp_path/'review.json';write_json(review,{'version':'fixture-model','reviewed_candidate_ids':ids,
  'source_clusters_reviewed':True,'source_audit_sha256':sha256(audit),
  'reference_and_action_swap_checks_passed':True,'severe_mechanics_violations':[]})
 report=evaluate_gate(tmp_path,acceptable(),'fixture-model',review)
 assert report['passed'] and report['numeric_passed']


def test_current_summary_has_only_canonical_predictions():
 row={'group_id':'g-0','teacher':dict(video_1=10,video_2=30,video_3=50,video_4=70),
      'scores':dict(video_1=22,video_2=42,video_3=62,video_4=82),'mechanism_class':'normal',
      'permutation_max_abs_delta':60,'permutation_scores':{'video_1':100},'permutation_raw_output':'legacy'}
 report=summarize([row],'fixture')
 assert report['mae']==12
 assert report['quality_standard']==QUALITY_STANDARD
 assert not any(k.startswith('permutation') for k in report)
 assert not any(k.startswith('permutation') for k in report['results'][0])
 assert row['permutation_max_abs_delta']==60  # Historical evidence stays intact.


@pytest.mark.parametrize('previous_flag',[False,True])
def test_verified_legacy_canonical_reply_is_reused_without_touching_original(tmp_path,previous_flag):
 item={'group_id':'g-0','scores':{'video_1':10}}
 legacy=tmp_path/'groups/g-0.json'
 write_json(legacy,{'scores':{'video_1':10},'permutation_max_abs_delta':60,
  'evaluation_identity':digest({'version':'v1','group':item,'permutation':previous_flag})})
 before=sha256(legacy);result=load_cached_canonical(tmp_path,item,'v1')
 assert result['evaluation_identity']==canonical_identity('v1',item)
 assert 'permutation_max_abs_delta' not in result
 assert sha256(legacy)==before
 assert load_cached_canonical(tmp_path,item,'v1')==result
 changed=copy.deepcopy(item);changed['scores']['video_1']=20
 with pytest.raises(ValueError,match='identity changed'):load_cached_canonical(tmp_path,changed,'v1')


@pytest.mark.parametrize('error,expected_status',[(12,'awaiting_human_review'),(15,'awaiting_human_review'),(15.01,'failed_numeric_quality')])
def test_C_initial_audit_uses_same_overall_mae_standard(tmp_path,error,expected_status):
 classes=['normal','macro_success','macro_fail','result_win','result_lose'];groups=[];teacher=[];student=[]
 for i in range(56):
  g=tmp_path/'groups'/f'g-{i}';groups.append(g)
  write_json(g/'manifest.json',{'group_id':g.name,'candidates':[{'video_id':f'v-{j}'} for j in range(4)],'condition':{'mechanism_class':classes[i%5]}})
  t={f'video_{j+1}':float(10+20*j) for j in range(4)};teacher.append(t);student.append({k:v+error for k,v in t.items()})
 report=assess(tmp_path,{'reward_model':{'audit_initial_conditions':56},'rollout':{'conditions_per_batch':24}},groups,teacher,student,'fixture')
 assert report['status']==expected_status
 assert report['quality_standard']['mae_max']==15
 assert not report['passed']  # No real manual evidence in these software fixtures.
