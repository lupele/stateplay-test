import torch
from types import SimpleNamespace
from rl_posttrain.reward_model.collator import GroupCollator

def test_answer_suffix_shift_loss_and_gradient_without_big_model():
    torch.manual_seed(19);hidden=torch.randn(1,11,6,requires_grad=True);head=torch.nn.Linear(6,20,bias=False)
    ids=torch.randint(20,(1,11));labels=ids.clone();labels[:,:7]=-100
    full=head(hidden);loss=torch.nn.functional.cross_entropy(full[:,:-1].reshape(-1,20),labels[:,1:].reshape(-1),ignore_index=-100)
    positions=torch.arange(6,10);suffix=head(hidden[:,positions]);other=torch.nn.functional.cross_entropy(suffix.reshape(-1,20),ids[:,7:].reshape(-1))
    torch.testing.assert_close(loss,other)
    g=torch.autograd.grad(loss,hidden,retain_graph=True)[0];expected=torch.autograd.grad(other,hidden)[0];torch.testing.assert_close(g,expected)
