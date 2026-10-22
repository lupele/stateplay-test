"""Stored fixture labels: no API request or actual model quality claim."""
import json
import pytest
from rl_posttrain.io import write_json,sha256
from rl_posttrain.reward_model.validate_assets import validate_assets

def fixture(root):
 root.mkdir();files={'reference.mp4':b'reference','actions_raw.parquet':b'actions','teacher/prompt.txt':b'prompt','frames/one.jpg':b'image'}
 candidates=[]
 for i in range(4):
  name=f'v{i}';files[f'candidates/{name}/video.mp4']=bytes([i]);files[f'candidates/{name}/clean_latents.safetensors']=bytes([i+4])
  candidates.append({'slot':f'video_{i+1}','video_id':name,'seed':i})
 for name,data in files.items():
  p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
 for c in candidates:
  c.update(video_sha256=sha256(root/'candidates'/c['video_id']/'video.mp4'),latent_sha256=sha256(root/'candidates'/c['video_id']/'clean_latents.safetensors'))
 scores={f'video_{i+1}':i*20 for i in range(4)};mapping={c['slot']:c['video_id'] for c in candidates}
 s={'scores':scores,'mapping':mapping,'cache_key':'fixture-request'}
 m={'candidates':candidates,'frame_assets':{'one':[{'path':str(root/'frames/one.jpg'),'sha256':sha256(root/'frames/one.jpg')}]},
    'condition':{'video_sha256':sha256(root/'reference.mp4'),'action_sha256':sha256(root/'actions_raw.parquet')},'prompt_sha256':sha256(root/'teacher/prompt.txt')}
 write_json(root/'manifest.json',m);write_json(root/'teacher/scores.json',s)
 write_json(root/'teacher/request_map.json',{'slots':mapping,'cache_key':'fixture-request'})
 write_json(root/'teacher/response_raw.json',{'http_status':200,'response':{'choices':[{'message':{'content':json.dumps(scores)}}]}})
 return m,s,{'reward':{'protocol':'fixture'}}

def test_changed_video_cannot_reuse_validation_cache(tmp_path):
 root=tmp_path/'group';m,s,c=fixture(root);validate_assets(root,m,s,c)
 (root/'candidates/v0/video.mp4').write_bytes(b'corrupted')
 with pytest.raises(ValueError,match='checksum changed'):validate_assets(root,m,s,c)

def test_changed_labels_cannot_be_accepted_without_original_response(tmp_path):
 root=tmp_path/'group';m,s,c=fixture(root);validate_assets(root,m,s,c)
 s['scores']['video_1']=99;write_json(root/'teacher/scores.json',s)
 with pytest.raises(ValueError,match='original raw response'):validate_assets(root,m,s,c)


def original_prompt_requests(root,text='prompt',sent=None):
 p=root/'teacher/request_map.json';request=json.loads(p.read_text())
 request['content']=[{'type':'text','text':text}];write_json(p,request)
 write_json(root/'teacher/request_payload.json',{'messages':[{'role':'user','content':[{'type':'text','text':text if sent is None else sent}]}]})


def test_missing_prompt_restored_only_from_matching_original_requests(tmp_path):
 root=tmp_path/'group';m,s,c=fixture(root);original_prompt_requests(root)
 expected=(root/'teacher/prompt.txt').read_bytes();(root/'teacher/prompt.txt').unlink()
 validate_assets(root,m,s,c)
 assert (root/'teacher/prompt.txt').read_bytes()==expected
 report=json.loads((root/'provenance/recovered_teacher_prompt.json').read_text())
 assert report['new_API_requests']==0 and not report['existing_file_overwritten']


@pytest.mark.parametrize('text,sent',[('changed','changed'),('prompt','different')])
def test_missing_prompt_cannot_be_invented_or_reconstructed_from_wrong_request(tmp_path,text,sent):
 root=tmp_path/'group';m,s,c=fixture(root);original_prompt_requests(root,text,sent)
 (root/'teacher/prompt.txt').unlink()
 with pytest.raises(ValueError):validate_assets(root,m,s,c)
 assert not (root/'teacher/prompt.txt').exists()


def test_existing_corrupt_prompt_is_preserved_and_rejected(tmp_path):
 root=tmp_path/'group';m,s,c=fixture(root);original_prompt_requests(root)
 (root/'teacher/prompt.txt').write_bytes(b'corruption')
 with pytest.raises(ValueError,match='checksum changed'):validate_assets(root,m,s,c)
 assert (root/'teacher/prompt.txt').read_bytes()==b'corruption'


def test_transient_shared_storage_error_retries_then_validates(tmp_path,monkeypatch):
 import errno
 from rl_posttrain.reward_model import validate_assets as module
 root=tmp_path/'group';m,s,c=fixture(root);original=module._validate_assets;calls=[]
 def intermittent(*args):
  calls.append(1)
  if len(calls)<3:raise OSError(errno.ESTALE,'temporary stale file handle')
  return original(*args)
 monkeypatch.setattr(module,'_validate_assets',intermittent)
 monkeypatch.setattr(module.time,'sleep',lambda _:None)
 validate_assets(root,m,s,c)
 assert len(calls)==3
 assert json.loads((root/'provenance/reward_dataset_validation.json').read_text())['passed']


def test_corrupted_assets_are_never_retried_into_acceptance(tmp_path,monkeypatch):
 from rl_posttrain.reward_model import validate_assets as module
 root=tmp_path/'group';m,s,c=fixture(root);original=module._validate_assets;calls=[]
 (root/'reference.mp4').write_bytes(b'corruption')
 def count(*args):calls.append(1);return original(*args)
 monkeypatch.setattr(module,'_validate_assets',count)
 with pytest.raises(ValueError,match='checksum changed'):validate_assets(root,m,s,c)
 assert len(calls)==1
