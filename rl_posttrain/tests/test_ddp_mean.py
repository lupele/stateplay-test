import tempfile
from pathlib import Path
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

def _worker(rank,size,init,path,counts,stable=False):
    torch.set_num_threads(1);dist.init_process_group("gloo",init_method=init,rank=rank,world_size=size)
    model=torch.nn.Linear(1,1,bias=False);model.weight.data.fill_(.4);ddp=DistributedDataParallel(model)
    total=sum(counts)
    if total:
        from contextlib import nullcontext
        with ddp.no_sync() if stable else nullcontext():
            loss=ddp(torch.ones(max(1,counts[rank]),1)).square().sum()*(size/total if counts[rank] else 0)
            loss.backward()
        if stable:
            from rl_posttrain.distributed import gradient_mean
            gradient_mean(model)
        torch.save(model.weight.grad,Path(path)/f"rank-{rank}.pt")
    else:torch.save(None,Path(path)/f"rank-{rank}.pt")
    dist.destroy_process_group()

@pytest.mark.parametrize('stable',[False,True])
@pytest.mark.parametrize("counts",[[2,0],[0,1,2,0,3,0,1],[0,0]])
def test_real_gloo_uneven_mean_and_all_invalid(tmp_path,counts,stable):
    init="file://"+str(tmp_path/"init")
    mp.spawn(_worker,args=(len(counts),init,str(tmp_path),counts,stable),nprocs=len(counts),join=True)
    for i in range(len(counts)):
        gradient=torch.load(tmp_path/f"rank-{i}.pt",weights_only=True)
        if sum(counts):torch.testing.assert_close(gradient,torch.tensor([[.8]]))
        else:assert gradient is None
