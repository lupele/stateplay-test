import copy
from pathlib import Path
import torch
import pytest
from rl_posttrain.tests.test_mode_switch import prepare
from rl_posttrain.io import write_json,read_json,sha256,atomic_torch,digest
from rl_posttrain import prepare_local_mode as pm


def setup_run(tmp_path,monkeypatch):
    _,_,source,model,_,_,config,_=prepare(tmp_path,'B-only')
    config=copy.deepcopy(config);config.pop('mode_switch');config['config_hash']=digest(config)
    run=tmp_path/'run';cp=run/'world/api/final/world.pt';atomic_torch(cp,source)
    write_json(cp.with_suffix('.json'),{'sha256':sha256(cp),'mode':'B-only'})
    write_json(run/'plan/cursor.json',{'cursor':2,'api_consumed':2,'local_consumed':0,'batch':source['policies']['version'],'inflight':None})
    lora={n:p.detach().cpu().clone() for n,p in model.named_parameters() if 'lora_' in n}
    monkeypatch.setattr(pm,'initialize_lora',lambda c:copy.deepcopy(lora))
    return run,config,source


@pytest.mark.parametrize('mode',pm.MODES)
def test_prepare_defaults_and_modes_preserve_one_pass_and_source(tmp_path,monkeypatch,mode):
    run,c,source=setup_run(tmp_path,monkeypatch)
    ledger=(run/'plan/cursor.json').read_bytes();original=sha256(run/'world/api/final/world.pt')
    result=pm.prepare_mode(c,run,mode)
    assert result['nft']['mode']==mode
    assert (run/'plan/cursor.json').read_bytes()==ledger
    assert sha256(run/'world/api/final/world.pt')==original
    if mode!='B-only':
        cp=Path(read_json(run/'world/local/checkpoints/latest.json')['checkpoint'])
        payload=torch.load(cp,weights_only=False)
        assert pm._same(payload['rank_rng'],source['rank_rng'])
        before=sha256(cp)
        assert pm.prepare_mode(c,run,mode)==result and sha256(cp)==before
        with pytest.raises(ValueError,match='mode differs'):pm.prepare_mode(c,run,'B-only')
        assert sha256(cp)==before


def test_c_progress_cannot_be_reset_for_new_modules(tmp_path,monkeypatch):
    run,c,_=setup_run(tmp_path,monkeypatch)
    p=run/'plan/cursor.json';s=read_json(p);s.update(cursor=3,local_consumed=1);write_json(p,s)
    before=p.read_bytes()
    with pytest.raises(ValueError,match='consumed/reserved'):pm.prepare_mode(c,run,'Coupling-only')
    assert p.read_bytes()==before and not (run/'world/local/checkpoints/latest.json').exists()


def test_interrupted_preparation_reuses_checkpoint_without_reset(tmp_path,monkeypatch):
    run,c,_=setup_run(tmp_path,monkeypatch)
    ledger=(run/'plan/cursor.json').read_bytes();original=pm.write_json
    def crash(path,value):
        if Path(path).name=='latest.json':raise OSError('injected crash before pointer commit')
        return original(path,value)
    monkeypatch.setattr(pm,'write_json',crash)
    with pytest.raises(OSError,match='injected'):pm.prepare_mode(c,run,'Both')
    cp=next((run/'world/local/initial').glob('*.pt'));before=sha256(cp)
    monkeypatch.setattr(pm,'write_json',original)
    assert pm.prepare_mode(c,run,'Both')['nft']['mode']=='Both'
    assert sha256(cp)==before and (run/'plan/cursor.json').read_bytes()==ledger


@pytest.mark.parametrize('mode',['Coupling-only','Both'])
@pytest.mark.parametrize('new_batch',[28,32,64])
def test_resume_runtime_batch_override_keeps_saved_policy_and_ledger(tmp_path,monkeypatch,mode,new_batch):
    from rl_posttrain.adapters.parameter_modes import configure,optimizer
    from rl_posttrain.adapters.mode_switch import apply_A_live
    from rl_posttrain.adapters.checkpoint import load
    from rl_posttrain.algorithms.policy_bank import PolicyBank
    from rl_posttrain.tests.test_mode_switch import Toy
    run,c,_=setup_run(tmp_path,monkeypatch)
    chosen=pm.prepare_mode(c,run,mode)
    pointer=run/'world/local/checkpoints/latest.json';cp=Path(read_json(pointer)['checkpoint'])
    cp_sha=sha256(cp);ledger=run/'plan/cursor.json';ledger_before=ledger.read_bytes()
    selection=run/'plan/C_module_selection.json';selection_before=selection.read_bytes()
    saved_config=Path(read_json(selection)['config']);config_before=saved_config.read_bytes()
    request=copy.deepcopy(c);request['rollout']['conditions_per_batch']=new_batch
    monkeypatch.setattr(pm,'initialize_lora',lambda _:(_ for _ in ()).throw(AssertionError('Existing parameters reinitialized')))
    resumed=pm.prepare_mode(request,run,mode)
    assert resumed['rollout']['conditions_per_batch']==new_batch
    assert resumed['config_hash']==digest({k:v for k,v in resumed.items() if k!='config_hash'})
    assert {k:v for k,v in resumed.items() if k not in ('rollout','config_hash')}=={k:v for k,v in chosen.items() if k not in ('rollout','config_hash')}
    assert ledger.read_bytes()==ledger_before and selection.read_bytes()==selection_before
    assert saved_config.read_bytes()==config_before and sha256(cp)==cp_sha
    model=Toy();configure(model,mode,rank=2);apply_A_live(model,resumed)
    opt=optimizer(model,resumed['nft']);bank=PolicyBank(model)
    state=load(cp,model,opt,bank,resumed,restore_random=False)
    assert state['optimizer_step']==7


@pytest.mark.parametrize('conditions,expected',[ (28,[64,48]*2),(32,[64]*4),(64,[64]*8),(8,[32]*2)])
def test_sixteen_rank_schedule_visits_every_video_once_per_inner_epoch(conditions,expected):
    from rl_posttrain.train_world import schedule
    c={'nft':{'accumulation':4,'inner_epochs':2,'full_schedule':False,'time_draws_per_visit':1},'plan':{'seed':3},'rollout':{'steps':30}}
    samples=list(range(conditions*4));windows=schedule(samples,c,16,batch=9)
    assert [w['valid'] for w in windows]==expected
    for inner in range(2):
        visits=[v[0] for w in windows for v in w['visits'] if v is not None and v[1]==inner]
        assert sorted(visits)==samples
