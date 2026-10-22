import pytest
import torch
from rl_posttrain.algorithms.nft_loss import forward_noise,nft_loss
from rl_posttrain.algorithms.reward_transform import transform

def official(live,old,xt,x0,sigma,r,beta,clip):
    old=old.detach();s=torch.tensor(sigma).reshape(-1,1,1,1,1)
    positive=beta*live+(1-beta)*old;negative=(1+beta)*old-beta*live
    # Identical official equations, with the StatePlay first-observation mask.
    p=(xt-s*positive-x0)[:,:,1:];n=(xt-s*negative-x0)[:,:,1:]
    with torch.no_grad():
        dp=p.double().abs().mean((1,2,3,4)).clamp_min(.00001)
        dn=n.double().abs().mean((1,2,3,4)).clamp_min(.00001)
    pos=(p.square()/dp.reshape(-1,1,1,1,1)).mean((1,2,3,4))
    neg=(n.square()/dn.reshape(-1,1,1,1,1)).mean((1,2,3,4))
    return clip*(r*pos/beta+(1-r)*neg/beta).mean()

@pytest.mark.parametrize("beta",[1.,.1,.7])
@pytest.mark.parametrize("r",[0.,.5,1.])
def test_official_adaptive_value_and_gradient(beta,r):
    torch.manual_seed(7);x=torch.randn(2,4,3,2,2);noise=torch.randn_like(x);xt,_=forward_noise(x,noise,[.2,.7])
    old=torch.randn_like(x);live=torch.randn_like(x,requires_grad=True)
    result,terms=nft_loss(live,old,xt,x,[.2,.7],[r,r],beta=beta)
    reference=official(live,old,xt,x,[.2,.7],torch.tensor([r,r]),beta,5)
    torch.testing.assert_close(result.double(),reference,rtol=2e-6,atol=1e-6)
    g=torch.autograd.grad(result,live,retain_graph=True)[0];expected=torch.autograd.grad(reference,live)[0]
    torch.testing.assert_close(g,expected,rtol=2e-6,atol=1e-7)
    assert not terms["denominator_positive"].requires_grad
    assert g[:,:,0].abs().sum()==0

@pytest.mark.parametrize("sigma",[0.,.3,1.])
def test_wan_velocity_oracle_and_pinning(sigma):
    x=torch.randn(2,4,3,2,2);eps=torch.randn_like(x);xt,target=forward_noise(x,eps,sigma)
    torch.testing.assert_close((xt-sigma*target)[:,:,1:],x[:,:,1:])
    assert torch.equal(xt[:,:,0],x[:,:,0])
    loss,_=nft_loss(target,target,xt,x,sigma,[1.,0.])
    assert loss<1e-8

@pytest.mark.parametrize("mode",["adaptive","velocity_mse"])
def test_tied_gradient_cancels_at_live_old(mode):
    x=torch.randn(1,2,3,2,2);eps=torch.randn_like(x);xt,_=forward_noise(x,eps,.5)
    live=torch.randn_like(x,requires_grad=True)
    loss,_=nft_loss(live,live.detach(),xt,x,.5,[.5],mode=mode,noise=eps)
    assert loss>0;loss.backward();torch.testing.assert_close(live.grad,torch.zeros_like(live),atol=1e-6,rtol=0)

def test_reward_std_floor_and_gates():
    scores=[[0,25,75,100],[30]*4,[30,30.1,30.2,30.3],[10,20,30,40]]
    guard=transform(scores);standard=transform(scores,reward_mode="standard")
    assert guard["keep"].tolist()==[True,False,False,False]
    assert standard["keep"].all();assert torch.equal(guard["r"][1],torch.full((4,),.5))
    assert 0<=guard["r"].min()<=guard["r"].max()<=1
    expected=torch.tensor(scores,dtype=torch.float64).std(correction=0)
    torch.testing.assert_close(guard["std_global"],expected)
    assert transform([[20,20.01,20.02,20.03]],reward_mode="standard")["Z"].item()==25

def test_first_slot_not_in_denominator():
    x=torch.zeros(1,2,3,2,2);live=torch.ones_like(x,requires_grad=True);xt=torch.ones_like(x);xt[:,:,0]=1e8
    a,_=nft_loss(live,live.detach(),xt,x,.4,[1.])
    xt[:,:,0]=0;b,_=nft_loss(live,live.detach(),xt,x,.4,[1.]);assert a==b
