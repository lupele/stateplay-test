import pytest
import httpx
from PIL import Image
from rl_posttrain.io import write_json,read_json,digest
from rl_posttrain.train_world import formal_gate
from rl_posttrain.rewards.protocol import group_content
from rl_posttrain.rewards.qwen_group import QwenGroupScorer
from rl_posttrain.rewards.api_group import APIGroupScorer
from rl_posttrain.reward_model.dataset import build_dataset

def test_actual_A_gate_does_not_depend_on_student_quality(tmp_path):
    write_json(tmp_path/'reports/integration_gate.json',{'passed':True,'source_code_hashes':{}})
    write_json(tmp_path/'reward_model/reports/quality_gate.json',{'passed':False})
    formal_gate(tmp_path,'api')
    with pytest.raises(RuntimeError,match='reward quality'):formal_gate(tmp_path,'local')

def test_formal_launch_rejects_changed_protected_model(tmp_path,monkeypatch):
    import rl_posttrain.io
    monkeypatch.setattr(rl_posttrain.io,'ROOT',tmp_path)
    model=tmp_path/'source.safetensors';model.write_bytes(b'original')
    stat=model.stat()
    write_json(tmp_path/'reports/integration_gate.json',{'passed':True,'protected_model_files':{'source.safetensors':{'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}}})
    formal_gate(tmp_path,'api')
    model.write_bytes(b'replacement')
    with pytest.raises(RuntimeError,match='source model file changed'):formal_gate(tmp_path,'api')

def test_missing_key_is_pre_submission_pending_not_permanent_failure(group,config,tmp_path,monkeypatch):
    monkeypatch.delenv('STATEPLAY_GPT_API_KEY',raising=False);monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    scorer=APIGroupScorer(config,tmp_path/'run',1)
    with pytest.raises(RuntimeError,match='no network submission'):scorer.score(group)
    with scorer.store.connect() as db:assert db.execute('select status,attempts from requests').fetchall()==[('pending',0)]

def test_local_score_cache_checks_actual_frame_bytes(group,config,monkeypatch):
    health={'version':'v','base_model':config['reward_model']['base_model'],'frozen':True,'full_multimodal':True}
    class Response:
        def raise_for_status(self):return self
        def json(self):return health
    monkeypatch.setattr(httpx,'get',lambda *a,**k:Response())
    scorer=QwenGroupScorer(config);parts,mapping=group_content(group,config)
    write_json(group/'local_reward/v.json',{'scores':dict(video_1=10,video_2=20,video_3=30,video_4=40),'content_hash':digest({'parts':parts,'mapping':mapping}),'candidate_mapping':mapping})
    assert scorer.score(group)['video_4']==40
    Image.new('RGB',(64,32),(0,0,0)).save(group/'frame.jpg')
    with pytest.raises(ValueError,match='checksum changed'):scorer.score(group)

def test_unversioned_teacher_is_not_formal_RM_data(group,config,tmp_path):
    run=tmp_path/'run';(run/'groups').mkdir(parents=True)
    (run/'groups'/group.name).symlink_to(group,target_is_directory=True)
    write_json(group/'teacher/scores.json',{'scorer':'gpt-6.1-sol','scores':dict(video_1=10,video_2=20,video_3=30,video_4=40)})
    with pytest.raises(ValueError,match='legacy rendering'):build_dataset(run,config=config)
    assert build_dataset(run,config=config,allow_legacy=True)['unique_conditions']==1


def test_persistent_user_hold_blocks_direct_formal_C_but_allows_A_B(tmp_path):
 write_json(tmp_path/'reports/integration_gate.json',{'passed':True})
 write_json(tmp_path/'plan/stage_C_control.json',{'allow_start':False})
 formal_gate(tmp_path,'api')
 with pytest.raises(RuntimeError,match='Stage C is disabled'):formal_gate(tmp_path,'local')
 # Fresh reads honor an explicit later release; the independent quality gate remains.
 write_json(tmp_path/'plan/stage_C_control.json',{'allow_start':True})
 with pytest.raises(RuntimeError,match='reward quality'):formal_gate(tmp_path,'local')
