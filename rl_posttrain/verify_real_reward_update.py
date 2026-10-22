"""Replay the one real teacher group through NFT. No additional API requests."""
from pathlib import Path
import argparse
import time
import torch
from safetensors.torch import load_file
from stateplay.utils.image import save_mp4
from .config import load_config
from .io import teacher_path,run_path,read_json,write_json,save_tensors
from .adapters.world_model import WorldAdapter
from .adapters.parameter_modes import optimizer,trainable_state,gradient_norms,parameter_hash
from .adapters.checkpoint import save
from .algorithms.policy_bank import PolicyBank
from .algorithms.nft_loss import forward_noise,nft_loss
from .algorithms.reward_transform import transform
from .algorithms.anchors import AnchorScheduler,anchor_losses
from .data.conditions import prepare_group,bins
from .rewards.base import SLOTS

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',default='smoke_v1');p.add_argument('--also-101',action='store_true');a=p.parse_args()
    c=load_config('rl_posttrain/configs/api_b_only.yaml');run=run_path(a.run_id);group=Path(read_json(run/'reports/real_api_acceptance.json')['group_path'])
    teacher=read_json(teacher_path(group)/'scores.json');scores=[teacher['scores'][k] for k in SLOTS];reward=transform([scores]);assert reward['keep'].item()
    torch.set_num_threads(6);torch.manual_seed(c['plan']['seed']);adapter=WorldAdapter(c);bank=PolicyBank(adapter.dit);opt=optimizer(adapter.dit,c['nft'])
    values=adapter.condition_tensors(group);m=read_json(group/'manifest.json');before=trainable_state(adapter.dit);frozen=parameter_hash(adapter.dit,True);losses=[];started=time.monotonic()
    for i,candidate in enumerate(m['candidates']):
        clean=load_file(group/'candidates'/candidate['video_id']/'clean_latents.safetensors')['clean'].to(adapter.device)
        sigma=float(adapter.scheduler.sigmas[15]);noise=torch.randn_like(clean);xt,_=forward_noise(clean,noise,sigma,values['first'])
        with bank.use('old'),torch.no_grad():old=adapter.forward(xt,sigma,values['context'],values['actions'],values['first'],False)['pred_video']
        with bank.live_graph():
            live=adapter.forward(xt,sigma,values['context'],values['actions'],values['first'],True)['pred_video']
            loss,_=nft_loss(live,old,xt,clean,sigma,[reward['r'][0,i]]);(loss/4).backward();losses.append(float(loss.detach()))
    catalog=read_json(run/'conditions/catalog.json')['records'];anchor_rng=torch.Generator('cpu').manual_seed(c['plan']['seed']+987654)
    record=catalog[int(torch.randint(len(catalog),(1,),generator=anchor_rng))]
    anchor=prepare_group(record,c,run/'conditions/real_teacher_anchor',bank.version,987654);av=adapter.condition_tensors(anchor);scheduler=AnchorScheduler();sigma,weight=scheduler.draw(anchor_rng)
    noise=torch.randn(av['real'].shape,generator=anchor_rng).to(adapter.device,av['real'].dtype)
    with bank.live_graph():
        fm,ce,_=anchor_losses(adapter,adapter.forward,av['real'],av['first'],av['context'],av['actions'],sigma,weight,noise)
        (.1*fm+.01*ce).backward()
    norms=gradient_norms(adapter.dit);torch.nn.utils.clip_grad_norm_(list(bank.params.values()),1.);opt.step();opt.zero_grad(set_to_none=True)
    after=trainable_state(adapter.dit);delta=sum(float((after[k]-before[k]).square().sum()) for k in after)**.5;assert delta>0 and norms['B']>0
    assert parameter_hash(adapter.dit,True)==frozen
    cp=run/'world/api_controlled_real/checkpoints/step-1.pt';save(cp,adapter.dit,opt,bank,{'optimizer_step':1,'batch_done':True,'software_only':True},c)
    report={'kind':'real_API_scores_to_real_NFT_update','group_path':str(group),'scorer':'gpt-6.1-sol','additional_API_requests':0,'scores':scores,'optimality':reward['r'].tolist(),'nft_losses':losses,'fm':float(fm.detach()),'ce':float(ce.detach()),'grad_norm':norms,'parameter_delta_l2':delta,'frozen_hash':frozen,'checkpoint':str(cp),'seconds':time.monotonic()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
    if a.also_101:
        # Inference-only horizon shape check using the same observed image and 100 actions.
        with bank.use('old'),torch.no_grad():clean=adapter.sample(values['first'],values['context'],values['actions'],26,20261014)
        directory=run/'conditions/frame_101_check';save_tensors(directory/'clean_latents.safetensors',{'clean':clean})
        frames=adapter.decode(clean);save_mp4(frames,directory/'video.mp4',fps=20)
        with torch.no_grad():out=adapter.forward(clean,.5,values['context'],values['actions'],values['first'],False)
        assert len(frames)==101 and out['G_seq_low'].shape==(1,26,4,256) and out['transition_logits'].shape==(1,25,4,512)
        report['frame101']={'video':str(directory/'video.mp4'),'frames':101,'latent_shape':list(clean.shape),'raw_action_rows':values['actions'].shape[1],'bins':bins(values['actions'].shape[1],26),'G_low':list(out['G_seq_low'].shape),'G':list(out['G_seq'].shape),'transition_logits':list(out['transition_logits'].shape),'first_slot_exact':bool(torch.equal(clean[:,:,:1],values['first']))}
    write_json(run/'reports/real_teacher_nft_update.json',report);print(report,flush=True)
if __name__=='__main__':main()
