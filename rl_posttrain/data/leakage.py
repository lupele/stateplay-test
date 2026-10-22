"""Conservative source/action/temporal duplicate components, never episode proof."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile
import cv2
import numpy as np
import pandas as pd
from ..io import read_json,write_json,digest,sha256

BUTTONS=['UP','DOWN','LEFT','RIGHT','Y','X','Z','A','B','C','D']

def phash(frame):
    gray=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(32,32)).astype(np.float32)
    dct=cv2.dct(gray)[:8,:8].flatten()[1:]
    return sum(int(v>np.median(dct))<<i for i,v in enumerate(dct))

def fingerprint_pair(video,action):
    raw=pd.read_parquet(action)[BUTTONS].to_numpy()
    if not np.isin(raw,[0,1]).all():raise ValueError('Non-binary source actions')
    raw=raw.astype(np.uint8)
    decisions=raw[::10];tokens=[sum(int(v)<<i for i,v in enumerate(row)) for row in decisions]
    options=[cv2.CAP_PROP_N_THREADS,1] if hasattr(cv2,'CAP_PROP_N_THREADS') else []
    cap=cv2.VideoCapture(str(video),cv2.CAP_FFMPEG,options);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));hashes=[]
    if count<5:cap.release();raise ValueError('Temporal leakage audit needs at least five frames')
    targets=set(np.linspace(0,count-1,5).round().astype(int))
    for i in range(count):
        if not cap.grab():raise ValueError('Unreadable temporal fingerprint')
        if i in targets:
            ok,frame=cap.retrieve()
            if not ok:raise ValueError('Unreadable temporal fingerprint')
            hashes.append(phash(frame))
    cap.release()
    return {'temporal_phash':hashes,'decision_tokens':tokens,'action_bytes_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),'D_events':int(raw[:,-1].sum()),'D_combined_events':int(((raw[:,-1]>0)&(raw[:,:-1].sum(1)>0)).sum())}

def close(a,b,all_frames=False):
    pairs=[(int(x)^int(y)).bit_count() for x,y in zip(a['temporal_phash'],b['temporal_phash'])]
    return all(v<=3 for v in pairs) if all_frames else sum(v<=3 for v in pairs)>=2

def components(features):
    """Merge known source episodes, informative action overlap and temporal near copies."""
    parent={key:key for key in features};edges=[]
    def find(key):
        while parent[key]!=key:parent[key]=parent[parent[key]];key=parent[key]
        return key
    def merge(a,b,reason):
        x,y=find(a),find(b)
        if x!=y:parent[max(x,y)]=min(x,y);edges.append({'a':a,'b':b,'reason':reason})
    sources={};windows={};near_actions={}
    for key,f in sorted(features.items()):
        source=f.get('episode_id') or f.get('source_recording')
        if source:
            if source in sources:merge(key,sources[source],'source_episode_or_recording')
            else:sources[source]=key
        tokens=f['decision_tokens']
        for start in range(max(0,len(tokens)-4)):
            window=tokens[start:start+5]
            if len(set(window))<3 or sum(bool(t&~15) for t in window)<2:continue
            signature=tuple(window)
            for other in windows.get(signature,[]):
                if close(f,features[other]):merge(key,other,'50_frame_action_overlap_and_temporal_similarity')
            windows.setdefault(signature,[]).append(key)
        if len(set(tokens))>=3:
            # A leave-one-decision-out signature finds >=90% identical commands.
            for missing in range(len(tokens)):
                signature=(len(tokens),missing,tuple(tokens[:missing]+tokens[missing+1:]))
                for other in near_actions.get(signature,[]):
                    other_tokens=features[other]['decision_tokens']
                    agreement=sum(a==b for a,b in zip(tokens,other_tokens))/len(tokens)
                    if agreement>=.9 and close(f,features[other],True):merge(key,other,'near_identical_actions_and_all_temporal_frames')
                near_actions.setdefault(signature,[]).append(key)
    groups={}
    for key in features:groups.setdefault(find(key),[]).append(key)
    mapping={key:'source-action-temporal-'+digest(sorted(members))[:20] for members in groups.values() for key in members}
    return mapping,edges

def build_index(records,config,run,workers=16):
    cv2.setNumThreads(1)
    path=run/'conditions/leakage_index.json';features_path=run/'conditions/leakage_features.json'
    identity={'records':[(r['condition_id'],r['video_sha256'],r['action_sha256']) for r in records],'archives':[(p,sha256(p)) for p in config['data'].get('exclusion_archives',[])]}
    provenance=[(r['condition_id'],r.get('episode_id'),r.get('source_recording')) for r in records if r.get('episode_id') or r.get('source_recording')]
    if provenance:identity['provenance']=provenance
    if path.exists():
        result=read_json(path)
        if result['input_hash']!=digest(identity):raise ValueError('Leakage index inputs changed; create a new run')
        return result
    features=read_json(features_path) if features_path.exists() else {}
    def one(r):
        result=fingerprint_pair(r['video'],r['action']);result.update(episode_id=r.get('episode_id'),source_recording=r.get('source_recording'),video_sha256=r['video_sha256'],source_action_sha256=r['action_sha256']);return r['condition_id'],result
    pending=[r for r in records if features.get(r['condition_id'],{}).get('video_sha256')!=r['video_sha256'] or features.get(r['condition_id'],{}).get('source_action_sha256')!=r['action_sha256']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for n,(key,value) in enumerate(pool.map(one,pending),1):
            features[key]=value
            if n%500==0:write_json(features_path,features);print('Leakage fingerprints',len(features),'/',len(records),flush=True)
    write_json(features_path,features)
    # Reference/action pairs from supplied holdout archives are compared in the same index.
    with tempfile.TemporaryDirectory(prefix='stateplay-leakage-') as temp:
        for ai,archive in enumerate(config['data'].get('exclusion_archives',[])):
            with zipfile.ZipFile(archive) as z:
                for video in (n for n in z.namelist() if n.endswith('/video.mp4')):
                    action=str(Path(video).with_name('actions.parquet'))
                    if action not in z.namelist():raise ValueError('Holdout lacks paired actions: '+video)
                    key='holdout-'+str(ai)+'-'+digest(video)[:16]
                    vp=Path(temp)/'video.mp4';ap=Path(temp)/'actions.parquet'
                    vp.write_bytes(z.read(video));ap.write_bytes(z.read(action));features[key]=fingerprint_pair(vp,ap)
    mapping,edges=components(features)
    holdout_clusters={v for k,v in mapping.items() if k.startswith('holdout-')}
    excluded=[r['condition_id'] for r in records if mapping[r['condition_id']] in holdout_clusters]
    result={'schema':'rl-source-leakage-v2','input_hash':digest(identity),'source_clusters':mapping,'edges':edges,'holdout_overlap_excluded':excluded,'training_records':len(records),'temporal_fingerprints':5,'minimum_informative_action_overlap_frames':50,'source_episode_ids_available':sum(bool(r.get('episode_id')) for r in records),'cluster_sizes':sorted(__import__('collections').Counter(mapping[r['condition_id']] for r in records).values(),reverse=True),'D_audit':{'events':sum(features[r['condition_id']]['D_events'] for r in records),'combined_events':sum(features[r['condition_id']]['D_combined_events'] for r in records)},'limitation':'Conservative duplicate components improve leakage checks but cannot prove episode independence without source episode IDs. Formal RM quality still needs manual source review.'}
    write_json(path,result);return result
