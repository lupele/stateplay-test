"""Rendering eligibility is explicit; historical groups remain immutable."""
from ..io import read_json,digest

RENDERING_VERSION='bf16-no-autocast-deterministic-v3'

def rendering_identity(config):
    keys=('height','width','requested_frames','steps','sigma_shift','cfg_scale','action_cfg_scale','action_hold_window')
    sampling={k:config['rollout'][k] for k in keys if k in config['rollout']}
    return {'version':RENDERING_VERSION,'precision':'bf16-without-autocast','compute_backend':'deterministic-v1','sampling':sampling,'source_sft':config['world']['checkpoint_sha256']}

def require_current_group(root,config,allow_legacy=False):
    manifest=read_json(root/'manifest.json')
    expected=rendering_identity(config)
    good=manifest.get('rendering',{}).get('version')==RENDERING_VERSION
    if good and manifest['rendering']!=expected:raise ValueError('Rendering sampling configuration changed')
    if not good and not allow_legacy:raise ValueError('Legacy or unversioned rendering group is quarantined: '+str(root))
    if manifest['protocol']!=config['reward']['protocol']:raise ValueError('Teacher protocol differs from this run')
    return good
