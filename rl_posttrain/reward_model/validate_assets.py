"""Verify real labels and assets once per immutable file-stat/content identity."""
import hashlib
import errno
import time
from ..io import read_json,write_json,sha256,digest,lock,teacher_path,atomic_bytes,append_jsonl,now
from ..rewards.base import parse_scores

def recover_missing_prompt(root,manifest):
    """Restore only a missing sidecar from the immutable original API request.

    Never infer a prompt from current code or replace an existing file. Both
    saved request representations must agree and match the original manifest.
    """
    teacher=teacher_path(root);saved=teacher/'prompt.txt'
    if saved.exists():return
    with lock(root/'provenance/.reward_prompt_recovery.lock'):
        if saved.exists():return
        mapping=read_json(teacher/'request_map.json')
        payload=read_json(teacher/'request_payload.json')
        first=mapping['content'][0];sent=payload['messages'][0]['content'][0]
        if first.get('type')!='text' or sent.get('type')!='text' or first.get('text')!=sent.get('text'):
            raise ValueError('Missing teacher prompt cannot be recovered from disagreeing original requests')
        text=first['text'];encoded=text.encode('utf-8')
        if hashlib.sha256(encoded).hexdigest()!=manifest['prompt_sha256']:
            raise ValueError('Recovered teacher prompt does not match original manifest checksum')
        atomic_bytes(saved,encoded)
        write_json(root/'provenance/recovered_teacher_prompt.json',{'recovered':True,
                   'source':'matching original request_map.json and request_payload.json text',
                   'prompt_sha256':manifest['prompt_sha256'],'request_map_sha256':sha256(teacher/'request_map.json'),
                   'request_payload_sha256':sha256(teacher/'request_payload.json'),'existing_file_overwritten':False,
                   'new_API_requests':0})

def validate_assets(root,manifest,scores,config):
    # Shared storage can briefly report missing/stale handles. Retry reads,
    # never checksum or label mismatches, and never substitute an absent asset.
    for attempt in range(5):
        try:return _validate_assets(root,manifest,scores,config)
        except OSError as error:
            if error.errno not in (errno.ENOENT,errno.ESTALE,errno.EIO,errno.ETIMEDOUT) or attempt==4:raise
            delay=min(16,2**(attempt+1))
            append_jsonl(root/'provenance/reward_asset_read_retries.jsonl',{'time':now(),'attempt':attempt+1,
                         'errno':error.errno,'exception':type(error).__name__,'delay_seconds':delay})
            time.sleep(delay)

def _validate_assets(root,manifest,scores,config):
    recover_missing_prompt(root,manifest)
    expected={}
    for values in manifest['frame_assets'].values():
        for item in values:expected[item['path']]=item['sha256']
    for item in manifest['candidates']:
        expected[str(root/'candidates'/item['video_id']/'video.mp4')]=item['video_sha256']
        expected[str(root/'candidates'/item['video_id']/'clean_latents.safetensors')]=item['latent_sha256']
    expected[str(root/'reference.mp4')]=manifest['condition']['video_sha256']
    expected[str(root/'actions_raw.parquet')]=manifest['condition']['action_sha256']
    expected[str(teacher_path(root)/'prompt.txt')]=manifest['prompt_sha256']
    from pathlib import Path
    files=[]
    for name in sorted(expected):
        s=Path(name).stat();files.append((name,s.st_size,s.st_mtime_ns,s.st_ctime_ns))
    response=teacher_path(root)/'response_raw.json'
    identity=digest({'manifest':sha256(root/'manifest.json'),'scores':sha256(teacher_path(root)/'scores.json'),
                     'response':sha256(response),'files':files,'protocol':config['reward']['protocol']})
    cached=root/'provenance/reward_dataset_validation.json'
    with lock(root/'provenance/.reward_dataset_validation.lock'):
        if cached.exists() and read_json(cached).get('identity')==identity:return
        slots={c['slot']:c['video_id'] for c in manifest['candidates']}
        if scores.get('mapping')!=slots:raise ValueError('Teacher slot mapping disagrees with candidate identity')
        if len(set(slots.values()))!=4 or len({c['seed'] for c in manifest['candidates']})!=4:raise ValueError('Non-unique four-candidate identities/seeds')
        raw=read_json(response)
        if raw['http_status']!=200:raise ValueError('Teacher raw response was not successful')
        parsed=parse_scores(raw['response']['choices'][0]['message']['content'])
        if parsed!=parse_scores(scores['scores']):raise ValueError('Teacher labels differ from the saved original raw response')
        request=read_json(teacher_path(root)/'request_map.json')
        if request['slots']!=slots or request['cache_key']!=scores['cache_key']:raise ValueError('Original teacher request mapping/cache identity changed')
        for name,h in expected.items():
            if sha256(name)!=h:raise ValueError('Permanent reward dataset asset checksum changed: '+name)
        write_json(cached,{'passed':True,'identity':identity,'validated_assets':len(expected),'real_original_response_verified':True})
