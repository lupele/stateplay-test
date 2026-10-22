import pytest
import torch
from rl_posttrain.data.sampler import Ledger,stratified_order
from rl_posttrain.data.frames import frame_indices,loaded_frame_count
from rl_posttrain.data.conditions import bins,action_data
from rl_posttrain.io import SFT,protect
from rl_posttrain.algorithms.policy_bank import PolicyBank
from rl_posttrain.algorithms.anchors import AnchorScheduler
from stateplay.utils.scheduler import FlowMatchScheduler

def test_lengths_actions_and_frame_endpoints():
    assert loaded_frame_count(100)==97;assert loaded_frame_count(101)==101
    assert (97-1)//4+1==25;assert (101-1)//4+1==26
    assert frame_indices(97)[0]==0;assert frame_indices(97)[-1]==96;assert len(frame_indices(97))==24
    assert bins(100,25)[0]=={"start":0,"end_exclusive":4}

def test_ledger_four_phase_resume_and_no_duplicates(tmp_path):
    records=[{"condition_id":str(i),"mechanism_class":"normal" if i<7 else "result_win"} for i in range(10)]
    ledger=Ledger(tmp_path,records,2);first=ledger.reserve(4,"api",0)
    for phase in ("reserved","generated","scored","updating"):
        ledger.phase(phase);restored=Ledger(tmp_path);assert restored.reserve(4,"api",0)["condition_ids"]==first["condition_ids"]
    ledger.commit("checkpoint");rest=ledger.reserve(100,"local",1)["condition_ids"]
    assert not set(first["condition_ids"])&set(rest);assert len(first["condition_ids"]+rest)==10
    assert ledger.order==stratified_order(records,2)

def test_protected_checkpoint():
    with pytest.raises(ValueError):protect(SFT)

def test_policy_switch_guard_and_soft_update():
    model=torch.nn.Linear(2,2);bank=PolicyBank(model)
    with bank.live_graph():
        with pytest.raises(RuntimeError):
            with bank.use("old"):pass
    original={k:v.clone() for k,v in bank.old.items()}
    with torch.no_grad():model.weight.add_(1)
    eta=bank.update_old(10);assert eta==.01
    assert torch.equal(bank.ref["weight"],original["weight"])
    torch.testing.assert_close(bank.old["weight"],.01*original["weight"]+.99*model.weight.detach())

def test_rollout_scheduler_does_not_mutate_anchor():
    anchor=AnchorScheduler();before=anchor.scheduler.timesteps.clone();rollout=FlowMatchScheduler();rollout.set_timesteps(30,shift=5)
    assert len(rollout.timesteps)==30;assert len(anchor.scheduler.timesteps)==1000;assert torch.equal(before,anchor.scheduler.timesteps)
