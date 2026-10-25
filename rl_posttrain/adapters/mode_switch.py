"""Explicit A -> C trainable-set migration, preserving the full A policy.

Frozen B values must survive reconstruction, checkpoints and inference export.
Changing requires_grad alone would silently fall back to the original SFT B.
"""
import copy
from pathlib import Path
import torch
from ..io import sha256

B_PREFIX='latent_state_rollout.transition.'

def source_A(config):
    init=config['mode_switch']
    path=Path(init['source_A_checkpoint'])
    if sha256(path)!=init['source_A_sha256']:raise ValueError('Immutable source A checkpoint changed')
    source=torch.load(path,map_location='cpu',weights_only=False)
    if source['config']['nft']['mode']!='B-only' or source['state'].get('stage')!='api':
        raise ValueError('Expected the formal B-only A checkpoint, never a C checkpoint')
    if not source['state'].get('batch_done') or source['state'].get('update_cursor',0):
        raise ValueError('A mode switch requires a completed optimizer/batch boundary')
    if not source['live'] or any(not n.startswith(B_PREFIX) for n in source['live']):
        raise ValueError('Unexpected trainable tensors in source A checkpoint')
    if source['sft_sha256']!=config['world']['checkpoint_sha256']:
        raise ValueError('Source A/SFT identities differ')
    return source

def apply_A_live(model,config):
    source=source_A(config)
    params=dict(model.named_parameters())
    expected={n for n in params if n.startswith(B_PREFIX)}
    if expected!=set(source['live']):raise ValueError('Incomplete A transition overlay')
    with torch.no_grad():
        for n,value in source['live'].items():
            if params[n].shape!=value.shape:raise ValueError('A transition shape differs')
            params[n].data=value.to(device=params[n].device,dtype=value.dtype).clone()
    model._rl_frozen_override_names=tuple(n for n in source['live'] if not params[n].requires_grad)
    model._rl_frozen_override_reference={n:source['live'][n].clone() for n in model._rl_frozen_override_names}

def migrate_payload(source,config,lora):
    """CPU operation; existing A RNG, EMA, ref and B optimizer moments survive.

    New Coupling parameters start at zero functional delta and have fresh Adam
    moments. B-only/Both retain the original B optimizer moments by parameter ID.
    """
    mode=config['nft']['mode']
    if mode not in ('B-only','Coupling-only','Both'):raise ValueError(mode)
    train_B=mode in ('B-only','Both');train_C=mode in ('Coupling-only','Both')
    live={**(copy.deepcopy(source['live']) if train_B else {}),**(copy.deepcopy(lora) if train_C else {})}
    policies={'old':{**(copy.deepcopy(source['policies']['old']) if train_B else {}),
                     **(copy.deepcopy(lora) if train_C else {})},
              'ref':{**(copy.deepcopy(source['policies']['ref']) if train_B else {}),
                     **(copy.deepcopy(lora) if train_C else {})},
              'version':source['policies']['version'],
              'old_frozen':{} if train_B else copy.deepcopy(source['policies']['old']),
              'ref_frozen':{} if train_B else copy.deepcopy(source['policies']['ref'])}
    original_optimizer=source['optimizer']
    if len(original_optimizer['param_groups'])!=1:raise ValueError('Expected one original B optimizer group')
    template=original_optimizer['param_groups'][0]
    if len(template['params'])!=len(source['live']):raise ValueError('Original B optimizer tensor count differs')
    optimizer=copy.deepcopy(original_optimizer) if train_B else {'state':{},'param_groups':[]}
    if train_C:
        start=max((p for g in optimizer['param_groups'] for p in g['params']),default=-1)+1
        group=copy.deepcopy(template)
        group.update(params=list(range(start,start+len(lora))),lr=config['nft']['coupling_lr'])
        optimizer['param_groups'].append(group)
    payload=copy.deepcopy(source)
    payload.update(live=live,policies=policies,optimizer=optimizer,config=copy.deepcopy(config),
                   frozen_overrides={} if train_B else copy.deepcopy(source['live']),
                   mode_switch=copy.deepcopy(config['mode_switch']))
    payload['state'].update(stage='local',mode_switch_from='B-only',mode_switch_to=mode,
                            coupling_optimizer_steps=0,mode_switch_A_step=source['state']['optimizer_step'])
    return payload
