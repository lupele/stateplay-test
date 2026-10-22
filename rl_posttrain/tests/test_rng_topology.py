import random
import numpy as np
import torch
from rl_posttrain.adapters.checkpoint import restore_rng

def test_seven_to_six_cuda_rngs(monkeypatch):
    received=[]
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(torch.cuda,"device_count",lambda:6)
    monkeypatch.setattr(torch.cuda,"set_rng_state_all",lambda values:received.extend(values))
    restore_rng({"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),"cuda":list(range(7))})
    assert received==list(range(6))
