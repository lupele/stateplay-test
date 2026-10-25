"""Select C modules once at the completed A boundary; never reset the ledger."""
import copy
import math
from pathlib import Path
import re
import torch
import yaml
from safetensors import safe_open
from .io import read_json,write_json,sha256,digest,atomic_torch,world_checkpoint,lock,shared_exists
from .config import load_config
from .adapters.mode_switch import migrate_payload,source_A

MODES=('B-only','Coupling-only','Both')


def initialize_lora(config):
    values={};rank=config['nft']['coupling_rank']
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['plan']['seed']+739391)
        with safe_open(config['world']['checkpoint'],framework='pt',device='cpu') as handle:
            for name in sorted(handle.keys()):
                if not re.fullmatch(r'blocks\.\d+\.latent_state_attention\.(q|k|v|out)\.weight',name):continue
                out_dim,in_dim=handle.get_slice(name).get_shape();prefix=name[:-len('weight')]
                a=torch.empty(rank,in_dim,dtype=torch.float32)
                torch.nn.init.kaiming_uniform_(a,a=math.sqrt(5))
                values[prefix+'lora_A']=a
                values[prefix+'lora_B']=torch.zeros(out_dim,rank,dtype=torch.float32)
    if len(values)!=240:raise ValueError('Expected the complete 30-layer Coupling LoRA targets')
    return values


def _same(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(_same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(_same(x,y) for x,y in zip(a,b))
    if hasattr(a,'shape'):
        import numpy as np
        return isinstance(b,type(a)) and np.array_equal(a,b)
    return a==b


def prepare_mode(config,run,mode='B-only'):
    """New modules get fresh Adam moments; original A old/ref/RNG remain exact.

    A resumed C branch must explicitly use its original mode. Changes after a
    reservation or update are refused; make a separate branch for a new study.
    """
    if mode not in MODES:raise ValueError('Unknown C update mode: '+str(mode))
    run=Path(run)
    requested_conditions=config['rollout']['conditions_per_batch']
    if requested_conditions<1:raise ValueError('conditions_per_batch must be positive')
    with lock(run/'plan/.module_switch.lock'):
        pending=run/'plan/C_module_selection.json'
        selection=read_json(pending) if shared_exists(pending,retry_missing=False) else None
        if selection:
            if selection['target_mode']!=mode:raise ValueError('Existing C mode differs; progress preserved. Pass --update-mode '+selection['target_mode'])
            config=yaml.safe_load(Path(selection['config']).read_text())
            if config['config_hash']!=digest({k:v for k,v in config.items() if k!='config_hash'}):
                raise ValueError('Stored module configuration checksum changed')
        latest=run/'world/local/checkpoints/latest.json'
        pointer=read_json(latest) if shared_exists(latest,retry_missing=False) else None
        if pointer:
            path=Path(pointer['checkpoint']);metadata=read_json(path.with_suffix('.json'))
            if metadata.get('mode')!=mode:raise ValueError('Existing C checkpoint uses another mode; no reset or automatic migration')
            if config['nft']['mode']!=mode:raise ValueError('Existing C configuration uses another mode; progress preserved')
            if selection and selection.get('status')!='complete':
                write_json(pending,dict(selection,status='complete'))
            # The saved selection fixes model/algorithm identity, not the
            # runtime rollout buffer size. Preserve the requested batch count
            # without rewriting the selection or checkpoint. In-flight ledger
            # reservations retain their original size until committed.
            config=copy.deepcopy(config)
            config['rollout']['conditions_per_batch']=requested_conditions
            config['config_hash']=digest({k:v for k,v in config.items() if k!='config_hash'})
            return config
        if config['nft']['mode']==mode and mode=='B-only':
            # Legacy/default B-only resumes A directly, including its optimizer.
            return config
        ledger=read_json(run/'plan/cursor.json')
        if ledger['local_consumed'] or ledger['inflight'] is not None or ledger['cursor']!=ledger['api_consumed']:
            raise ValueError('C has already consumed/reserved conditions; changing modules would mix policies')
        source=world_checkpoint(run,'api');source_hash=sha256(source)
        metadata=read_json(source.with_suffix('.json'))
        if source_hash!=metadata['sha256']:raise ValueError('Source A checkpoint checksum mismatch')
        c=copy.deepcopy(config);c['nft']['mode']=mode;c['stage']='local'
        c['mode_switch']={'schema':'A-to-C-trainable-mode-v1','source_A_checkpoint':str(source),
            'source_A_sha256':source_hash,'source_A_conditions':ledger['api_consumed'],
            'source_mode':'B-only','target_mode':mode,
            'reference_policy':'original A reference, including inherited B parameters',
            'old_policy':'original A EMA, including inherited B parameters',
            'lora_initialization_seed':c['plan']['seed']+739391}
        original=source_A(c)
        if original['state'].get('smoke'):raise ValueError('Formal module migration requires non-smoke A')
        step=original['state']['optimizer_step'];c['mode_switch']['source_A_optimizer_step']=step
        if original['policies']['version']!=ledger['batch']:
            raise ValueError('A policy version and C ledger boundary differ')
        c['config_hash']=digest({k:v for k,v in c.items() if k!='config_hash'})
        config_path=run/'plan'/('C_mode_'+mode+'.yaml')
        body=yaml.safe_dump(c,sort_keys=False)
        if config_path.exists() and config_path.read_text()!=body:raise ValueError('Existing module configuration differs; preserved')
        if not config_path.exists():config_path.write_text(body)
        record={'schema':'C-module-selection-v1','status':'preparing','target_mode':mode,
                'config':str(config_path),'source_A_checkpoint':str(source),'source_A_sha256':source_hash,
                'source_A_step':step,'A_conditions_excluded':ledger['api_consumed'],
                'C_start_cursor':ledger['cursor'],'ledger_reset':False}
        if not selection:write_json(pending,record)
        elif any(selection[k]!=record[k] for k in record if k!='status'):raise ValueError('Interrupted module selection identity changed')
        payload=migrate_payload(original,c,initialize_lora(c))
        initial=run/'world/local/initial'/('from_A_step-'+format(step,'08d')+'-'+mode+'.pt')
        if shared_exists(initial,retry_missing=False):
            if not _same(payload,torch.load(initial,map_location='cpu',weights_only=False)):
                raise ValueError('Existing initial mode checkpoint differs; preserved')
        else:atomic_torch(initial,payload)
        write_json(initial.with_suffix('.json'),{'schema':payload['schema'],'sha256':sha256(initial),
            'state':payload['state'],'mode':mode,'config_hash':c['config_hash']})
        # Commit the new model pointer only after payload/config publication.
        write_json(latest,{'checkpoint':str(initial),'optimizer_step':step})
        write_json(pending,dict(record,status='complete'))
        if read_json(run/'plan/cursor.json')!=ledger:raise RuntimeError('Ledger changed during module preparation')
        return c
