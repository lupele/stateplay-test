import torch
from rl_posttrain.reward_model.replay_dropout import ReplayDropout

def test_mask_repeats_with_same_cursor_and_different_input_layout():
    m=ReplayDropout(.05,'projection');m.seed=17845
    x=torch.arange(600,dtype=torch.float32).reshape(20,30)
    y=x.T.contiguous().T
    assert x.stride()!=y.stride()
    state=torch.get_rng_state().clone()
    torch.testing.assert_close(m(x),m(y),rtol=0,atol=0)
    assert torch.equal(state,torch.get_rng_state())
    assert torch.equal(m(x),m(x))
    first=m(x);m.seed+=1;assert not torch.equal(first,m(x))
    m.eval();assert torch.equal(m(x),x)
