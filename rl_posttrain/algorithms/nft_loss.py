import torch

def sigma_view(sigma, x):
    return torch.as_tensor(sigma, device=x.device, dtype=x.dtype).reshape(-1, *([1]*(x.ndim-1)))

def forward_noise(x0, noise, sigma, first=None):
    s = sigma_view(sigma, x0); xt = (1-s)*x0+s*noise
    xt[:, :, :1] = x0[:, :, :1] if first is None else first
    return xt, noise-x0

def nft_loss(live, old, xt, x0, sigma, reward, beta=1., adv_clip=5., mode="adaptive", noise=None, denominator_dtype=torch.float64):
    if beta <= 0: raise ValueError("beta must be positive")
    if x0.shape[2] < 2: raise ValueError("NFT requires future latent slots")
    live, old, xt, x0 = live.float(), old.detach().float(), xt.float(), x0.float()
    s = sigma_view(sigma, x0); pos = (1-beta)*old+beta*live; neg=(1+beta)*old-beta*live
    if mode == "adaptive":
        ep = (xt-s*pos-x0)[:, :, 1:]; en=(xt-s*neg-x0)[:, :, 1:]
        axes=tuple(range(1, ep.ndim))
        dp=ep.detach().to(denominator_dtype).abs().mean(axes).clamp_min(1e-5).float()
        dn=en.detach().to(denominator_dtype).abs().mean(axes).clamp_min(1e-5).float()
        lp=ep.square().mean(axes)/dp; ln=en.square().mean(axes)/dn
    elif mode == "velocity_mse":
        if noise is None: raise ValueError("Velocity-MSE needs epsilon")
        target=noise.float()-x0; ep=(pos-target)[:,:,1:]; en=(neg-target)[:,:,1:]
        axes=tuple(range(1,ep.ndim));lp=ep.square().mean(axes);ln=en.square().mean(axes)
        dp=dn=torch.ones_like(lp)
    else: raise ValueError("Unknown NFT loss")
    r=torch.as_tensor(reward,device=lp.device,dtype=lp.dtype).reshape(-1)
    if r.numel()!=lp.numel() or not torch.isfinite(r).all() or ((r<0)|(r>1)).any(): raise ValueError("Invalid reward broadcasting")
    per_video=adv_clip/beta*(r*lp+(1-r)*ln)
    return per_video.mean(), {"per_video":per_video,"positive":lp.detach(),"negative":ln.detach(),"denominator_positive":dp,"denominator_negative":dn}
