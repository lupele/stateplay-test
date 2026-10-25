import copy
import torch
from torch import nn
import pytest
from rl_posttrain.adapters.parameter_modes import configure,optimizer,trainable_state
from rl_posttrain.adapters.mode_switch import migrate_payload,apply_A_live,B_PREFIX
from rl_posttrain.adapters.checkpoint import save,load,rng_state
from rl_posttrain.algorithms.policy_bank import PolicyBank
from rl_posttrain.io import atomic_torch,write_json,sha256,digest

class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_state_rollout=nn.Module()
        self.latent_state_rollout.transition=nn.Linear(4,4)
        self.blocks=nn.ModuleList([nn.Module()])
        self.blocks[0].latent_state_attention=nn.Module()
        for key in ('q','k','v','out'):
            setattr(self.blocks[0].latent_state_attention,key,nn.Linear(4,4))
    def forward(self,x):
        x=self.latent_state_rollout.transition(x)
        return sum(getattr(self.blocks[0].latent_state_attention,k)(x) for k in ('q','k','v','out'))

def prepare(tmp_path,mode):
    torch.manual_seed(17)
    source_model=Toy();configure(source_model,'B-only',rank=2)
    config={'world':{'checkpoint_sha256':'fixture','tokenizer_checkpoint':str(tmp_path/'tokenizer')},
            'nft':{'mode':'B-only','transition_lr':.001,'coupling_lr':.002,'weight_decay':.01},
            'data':{'root':'fixture'},'rollout':{'conditions_per_batch':28},'reward':{},'plan':{'seed':3}}
    (tmp_path/'tokenizer').write_bytes(b'fixture')
    source_opt=optimizer(source_model,config['nft']);source_bank=PolicyBank(source_model)
    source_model(torch.ones(2,4)).square().mean().backward();source_opt.step();source_opt.zero_grad()
    source_bank.update_old(7)
    source={'schema':'rl-world-checkpoint-v1','live':trainable_state(source_model),
            'policies':copy.deepcopy(source_bank.state_dict()),'optimizer':source_opt.state_dict(),
            'config':copy.deepcopy(config),'state':{'stage':'api','optimizer_step':7,'batch_done':True,'update_cursor':0},
            'rank_rng':[rng_state()],'sft_sha256':'fixture','tokenizer_sha256':sha256(tmp_path/'tokenizer')}
    path=tmp_path/'A.pt';atomic_torch(path,source)
    target=copy.deepcopy(source_model);configure(target,mode,rank=2)
    lora={n:p.detach().clone() for n,p in target.named_parameters() if 'lora_' in n}
    config['nft']['mode']=mode
    config['mode_switch']={'source_A_checkpoint':str(path),'source_A_sha256':sha256(path),'target_mode':mode}
    config['config_hash']=digest(config)
    apply_A_live(target,config)
    payload=migrate_payload(source,config,lora)
    path=tmp_path/'migrated.pt';atomic_torch(path,payload)
    write_json(path.with_suffix('.json'),{'sha256':sha256(path)})
    opt=optimizer(target,config['nft']);bank=PolicyBank(target)
    state=load(path,target,opt,bank,config,restore_random=False)
    return source_model,source_bank,source,target,opt,bank,config,state

@pytest.mark.parametrize('mode',['B-only','Coupling-only','Both'])
def test_real_optimizer_module_switch_and_full_policy_preservation(tmp_path,mode):
    source_model,source_bank,source,target,opt,bank,c,state=prepare(tmp_path,mode)
    x=torch.arange(8,dtype=torch.float32).reshape(2,4)/8
    assert state['optimizer_step']==7
    torch.testing.assert_close(target(x),source_model(x),rtol=0,atol=0)
    for policy in ('old','ref'):
        with source_bank.use(policy),bank.use(policy):
            torch.testing.assert_close(target(x),source_model(x),rtol=0,atol=0)
    if mode=='Coupling-only':
        assert not opt.state
        assert set(bank.old_frozen)==set(source['live'])
    else:
        assert len(opt.state)==len(source['optimizer']['state'])
    before={n:p.detach().clone() for n,p in target.named_parameters()}
    refs=copy.deepcopy(bank.ref_frozen)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        target(x).square().mean().backward()
        assert all(p.grad is None for p in target.parameters() if not p.requires_grad)
        opt.step()
    changed={n for n,p in target.named_parameters() if not torch.equal(p,before[n])}
    assert bool(any(n.startswith(B_PREFIX) for n in changed))==(mode!='Coupling-only')
    assert bool(any('lora_' in n for n in changed))==(mode!='B-only')
    assert all(p.requires_grad or torch.equal(p,before[n]) for n,p in target.named_parameters())
    old_frozen=copy.deepcopy(bank.old_frozen);eta=bank.update_old(8)
    for n,t in old_frozen.items():
        expected=t*eta+dict(target.named_parameters())[n].detach().cpu()*(1-eta)
        torch.testing.assert_close(bank.old_frozen[n],expected,rtol=0,atol=1e-7)
        torch.testing.assert_close(bank.ref_frozen[n],refs[n],rtol=0,atol=0)
    expected=target(x).detach().clone()
    cp=tmp_path/'roundtrip.pt';save(cp,target,opt,bank,state,c)
    saved=torch.load(cp,weights_only=False)
    if mode=='Coupling-only':
        assert set(saved['frozen_overrides'])==set(source['live'])
        assert all(torch.equal(saved['frozen_overrides'][n],source['live'][n]) for n in source['live'])
    with torch.no_grad():
        for n,p in target.named_parameters():
            if p.requires_grad or n.startswith(B_PREFIX):p.add_(1.)
    load(cp,target,opt,bank,c,restore_random=False)
    torch.testing.assert_close(target(x),expected,rtol=0,atol=0)
    before_B={n:p.detach().clone() for n,p in target.named_parameters() if n.startswith(B_PREFIX)}
    with pytest.raises(ValueError,match='mode mismatch'):
        wrong=copy.deepcopy(c);wrong['nft']['mode']='Both' if mode!='Both' else 'B-only'
        load(cp,target,opt,bank,wrong,restore_random=False)
    assert all(torch.equal(dict(target.named_parameters())[n],v) for n,v in before_B.items())

def test_new_cli_mode_option_is_available():
    from rl_posttrain.timeslice import parser
    a=parser().parse_args(['--run-id','fixture','--config','fixture','--update-mode','Coupling-only'])
    assert a.update_mode=='Coupling-only'
    a=parser().parse_args(['--run-id','fixture','--config','fixture','--train-mode','Both'])
    assert a.update_mode=='Both'

def test_policy_rejects_trainable_frozen_alias(tmp_path):
    _,_,_,model,_,bank,_,_=prepare(tmp_path,'Both')
    state=bank.state_dict();name=next(iter(bank.params))
    state['old_frozen']={name:bank.params[name].detach().clone()}
    state['ref_frozen']=copy.deepcopy(state['old_frozen'])
    with pytest.raises(ValueError,match='trainable/missing'):
        bank.load_state_dict(state)
