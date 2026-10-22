import torch
from rl_posttrain.adapters.world_model import WorldAdapter
from stateplay.utils.scheduler import FlowMatchScheduler

def test_sampler_restores_exact_fp32_master_storage():
    adapter=WorldAdapter.__new__(WorldAdapter);adapter.device=torch.device('cpu');adapter.dtype=torch.bfloat16
    adapter.dit=torch.nn.Linear(2,2);adapter.scheduler=FlowMatchScheduler();adapter.scheduler.set_timesteps(2)
    before={n:p.detach().clone() for n,p in adapter.dit.named_parameters()};ptrs={n:p.data_ptr() for n,p in adapter.dit.named_parameters()}
    def forward(x,*args):
        assert all(p.dtype==torch.bfloat16 for p in adapter.dit.parameters())
        return {"pred_video":torch.zeros_like(x)}
    adapter.forward=forward
    first=torch.zeros(1,48,1,2,2,dtype=torch.bfloat16)
    out=adapter.sample(first,None,None,3,20261013)
    assert torch.equal(out[:,:,:1],first)
    for n,p in adapter.dit.named_parameters():
        assert p.dtype==torch.float32 and p.data_ptr()==ptrs[n];assert torch.equal(p.detach(),before[n])
