import torch
import torch.nn.functional as F
from diffsynth.diffusion import FlowMatchScheduler
from .nft_loss import forward_noise

class AnchorScheduler:
    def __init__(self):
        self.scheduler=FlowMatchScheduler("Wan");self.scheduler.set_timesteps(1000,training=True,shift=5)
    def draw(self,generator):
        index=int(torch.randint(1000,(1,),generator=generator))
        return float(self.scheduler.sigmas[index]),float(self.scheduler.linear_timesteps_weights[index])

def anchor_losses(adapter,forward,clean,first,context,actions,sigma,weight,noise):
    xt,target=forward_noise(clean,noise,sigma,first)
    with torch.no_grad():
        tokens,shape=adapter.dit.extract_clean_visual_tokens(clean)
        labels=adapter.dit.transition_tokenizer(tokens,latent_frames=shape["f"])["z_indices"].detach()
    result=forward(xt, sigma, context, actions, first, True)
    fm=F.mse_loss(result["pred_video"][:,:,1:].float(),target[:,:,1:].float())*weight
    ce=F.cross_entropy(result["transition_logits"].flatten(0,2).float(),labels.flatten())
    return fm,ce,result
