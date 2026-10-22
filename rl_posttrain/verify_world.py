"""Real SFT weight tests; small latents for gradients, full scale separately profiled."""
import argparse
import copy
import time
import torch
from safetensors.torch import load_file
from .config import load_config
from .io import run_path,write_json,save_tensors,digest
from .adapters.world_model import WorldAdapter
from .adapters.parameter_modes import configure,optimizer,gradient_norms,trainable_state,parameter_hash
from .algorithms.policy_bank import PolicyBank
from .algorithms.nft_loss import forward_noise,nft_loss
from .algorithms.anchors import AnchorScheduler,anchor_losses
from .adapters.checkpoint import save,load

def main():
    p=argparse.ArgumentParser();p.add_argument("--run-id",default="smoke_v1");p.add_argument("--config",default="rl_posttrain/configs/smoke.yaml");p.add_argument("--group");a=p.parse_args()
    c=load_config(a.config);run=run_path(a.run_id);root=__import__('pathlib').Path(a.group) if a.group else next((run/"groups").glob("g-*"))
    torch.set_num_threads(8);torch.manual_seed(c["plan"]["seed"])
    adapter=WorldAdapter(c);write_json(run/"reports/checkpoint_key_audit.json",adapter.audit)
    values=adapter.condition_tensors(root);x0=values["real"];x0[:,:,:1]=values["first"]
    initial={n:p.detach().cpu().clone() for n,p in adapter.dit.named_parameters() if n.startswith("latent_state_rollout.transition.") or "lora_" in n}
    report={"kind":"real_SFT_full_scale_gradient_test" if tuple(x0.shape[-2:])==(30,52) else "real_SFT_small_resolution_gradient_test","sft":c["world"]["checkpoint"],"latent_shape":list(x0.shape),"modes":{}}
    anchor=AnchorScheduler()
    for mode in ("B-only","Coupling-only","Both"):
        c=copy.deepcopy(c);c["nft"]["mode"]=mode
        c["config_hash"]=digest({k:v for k,v in c.items() if k!="config_hash"})
        params=dict(adapter.dit.named_parameters())
        with torch.no_grad():
            for n,v in initial.items():params[n].copy_(v.to(params[n].device))
        names=configure(adapter.dit,mode);opt=optimizer(adapter.dit,c["nft"]);bank=PolicyBank(adapter.dit)
        frozen_before=parameter_hash(adapter.dit,True);before=trainable_state(adapter.dit)
        old_before={k:v.clone() for k,v in bank.old.items()};reference_before={k:v.clone() for k,v in bank.ref.items()}
        steps=[];torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        for step in range(3):
            noise=torch.randn_like(x0);sigma=float(adapter.scheduler.sigmas[min(1,len(adapter.scheduler.sigmas)-1)])
            xt,_=forward_noise(x0,noise,sigma,values["first"])
            with bank.use("old"),torch.no_grad():old=adapter.forward(xt,sigma,values["context"],values["actions"],values["first"],False)["pred_video"].detach()
            with bank.live_graph():
                live=adapter.forward(xt,sigma,values["context"],values["actions"],values["first"],True)["pred_video"]
                nft,_=nft_loss(live,old,xt,x0,sigma,[.8])
                nft.backward();nft_grads=gradient_norms(adapter.dit);opt.zero_grad(set_to_none=True)
                # Separate graphs establish each component's actual gradient path.
                fm,ce,_=anchor_losses(adapter,adapter.forward,x0,values["first"],values["context"],values["actions"],sigma,1.,torch.randn_like(x0))
                if ce.requires_grad:
                    ce.backward(retain_graph=True);ce_grads=gradient_norms(adapter.dit);opt.zero_grad(set_to_none=True)
                else:ce_grads={"B":0.,"Coupling":0.}
                fm.backward();fm_grads=gradient_norms(adapter.dit);opt.zero_grad(set_to_none=True)
                live=adapter.forward(xt,sigma,values["context"],values["actions"],values["first"],True)["pred_video"]
                n,_=nft_loss(live,old,xt,x0,sigma,[.8])
                fm,ce,_=anchor_losses(adapter,adapter.forward,x0,values["first"],values["context"],values["actions"],sigma,1.,torch.randn_like(x0))
                total=n+.1*fm+.01*ce;total.backward();grads=gradient_norms(adapter.dit)
                torch.nn.utils.clip_grad_norm_([p for p in adapter.dit.parameters() if p.requires_grad],1.);opt.step();opt.zero_grad(set_to_none=True)
            steps.append({"step":step+1,"nft_loss":float(n.detach()),"fm":float(fm.detach()),"ce":float(ce.detach()),"nft_grad_norm":nft_grads,"fm_grad_norm":fm_grads,"ce_grad_norm":ce_grads,"total_grad_norm":grads})
        after=trainable_state(adapter.dit)
        delta={label:sum(float((after[k]-before[k]).float().square().sum()) for k in after if k.startswith(prefix))**.5 for label,prefix in (("B","latent_state_rollout.transition."),("Coupling","blocks."))}
        frozen_after=parameter_hash(adapter.dit,True)
        assert frozen_after==frozen_before,"Frozen parameter/buffer corruption"
        if mode in ("B-only","Both"):assert delta["B"]>0 and any(s["nft_grad_norm"]["B"]>0 for s in steps)
        if mode in ("Coupling-only","Both"):assert delta["Coupling"]>0 and any(s["nft_grad_norm"]["Coupling"]>0 for s in steps)
        assert all(torch.equal(bank.old[k],old_before[k]) and torch.equal(bank.ref[k],reference_before[k]) for k in bank.old)
        eta=bank.update_old(3);checkpoint=run/"world"/"mode_tests"/mode/"step-3.pt"
        save(checkpoint,adapter.dit,opt,bank,{"optimizer_step":3,"batch_done":True},c)
        with torch.no_grad():expected=adapter.forward(xt,sigma,values["context"],values["actions"],values["first"],False)["pred_video"].cpu()
        with torch.no_grad():
            for p in bank.params.values():p.add_(.1)
        load(checkpoint,adapter.dit,opt,bank,c)
        with torch.no_grad():actual=adapter.forward(xt,sigma,values["context"],values["actions"],values["first"],False)["pred_video"].cpu()
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        entry={"trainable_names":names,"trainable_count":sum(p.numel() for p in adapter.dit.parameters() if p.requires_grad),"steps":steps,"parameter_delta_l2":delta,"frozen_hash_before":frozen_before,"frozen_hash_after":frozen_after,"old_fixed_during_batch":True,"reference_immutable":True,"ema_eta":eta,"reload_max_abs_error":float((actual-expected).abs().max()),"checkpoint":str(checkpoint),"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),"seconds":time.monotonic()-start}
        report["modes"][mode]=entry;write_json(run/"reports/real_world_modes.json",report);print(mode,delta,entry["peak_allocated_bytes"],flush=True)
    print("All real SFT update modes passed",flush=True)
if __name__=="__main__":main()
