from pathlib import Path
import copy
import pytest
from rl_posttrain.data.provenance import require_current_group,rendering_identity
from rl_posttrain.data.leakage import components
from rl_posttrain.io import write_json

def test_legacy_rendering_requires_explicit_development_opt_in(tmp_path):
    c={'world':{'checkpoint_sha256':'sft'},'rollout':{'steps':30},'reward':{'protocol':'v2'}}
    m={'protocol':'v2'};write_json(tmp_path/'manifest.json',m)
    with pytest.raises(ValueError,match='quarantined'):require_current_group(tmp_path,c)
    assert require_current_group(tmp_path,c,True) is False
    m['rendering']=rendering_identity(c);write_json(tmp_path/'manifest.json',m)
    assert require_current_group(tmp_path,c)
    changed=copy.deepcopy(c);changed['rollout']['steps']=20
    with pytest.raises(ValueError,match='configuration changed'):require_current_group(tmp_path,changed)

def feature(tokens,phash=1,episode=None):return {'decision_tokens':tokens,'temporal_phash':[phash]*5,'episode_id':episode}

def test_known_source_and_informative_overlap_share_split_component():
    a=[16,1,32,4,0,64,8,16,4,0]
    f={'a':feature(a),'b':feature([128,2,16,1,32,4,0,256,1,2]),'c':feature([4]*10,episode='same'),'d':feature([8]*10,episode='same'),'holdout-0':feature(a)}
    mapping,edges=components(f)
    assert mapping['a']==mapping['b']==mapping['holdout-0']
    assert mapping['c']==mapping['d']
    assert {e['reason'] for e in edges}>={'source_episode_or_recording','50_frame_action_overlap_and_temporal_similarity'}

def test_generic_idle_or_walk_is_not_claimed_to_be_same_episode():
    mapping,edges=components({'a':feature([0]*10),'b':feature([0]*10),'c':feature([4]*10),'d':feature([4]*10)})
    assert len(set(mapping.values()))==4 and not edges

def test_temporally_different_same_actions_do_not_merge():
    actions=[16,1,32,4,0,64,8,16,4,0]
    mapping,edges=components({'a':feature(actions,0),'b':feature(actions,(1<<63)-1)})
    assert mapping['a']!=mapping['b'] and not edges

def test_near_action_threshold_is_ninety_percent_for_short_clips():
    mapping,_=components({'a':feature([1,2,4,8,16]),'b':feature([1,2,4,8,32])})
    assert mapping['a']!=mapping['b']
    mapping,_=components({'a':feature([1,2,4,8,0,1,2,4,8,16]),'b':feature([1,2,4,8,0,1,2,4,8,32])})
    assert mapping['a']==mapping['b']

def test_recording_provenance_groups_different_actions():
    a=feature([1]*10);b=feature([2]*10)
    a['source_recording']=b['source_recording']='actual-source-recording'
    mapping,_=components({'a':a,'b':b})
    assert mapping['a']==mapping['b']

def test_fractional_actions_cannot_be_rounded_into_binary(monkeypatch):
    import pandas as pd
    from rl_posttrain.data.leakage import fingerprint_pair,BUTTONS
    frame=pd.DataFrame({k:[0.] for k in BUTTONS});frame.loc[0,'Y']=.5
    monkeypatch.setattr(pd,'read_parquet',lambda _:frame)
    with pytest.raises(ValueError,match='Non-binary'):fingerprint_pair('unused','unused')
