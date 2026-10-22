import torch

def transform(scores, adv_clip=5., std_floor=5., normalization="global_std", reward_mode="mechanism_guard", guard_min_max=50., guard_min_spread=1., fixed_scale=20.):
    scores=torch.as_tensor(scores,dtype=torch.float64)
    if scores.ndim!=2 or scores.shape[1]!=4 or not torch.isfinite(scores).all() or ((scores<0)|(scores>100)).any(): raise ValueError("Scores must be finite [G,4] points in [0,100]")
    means=scores.mean(1,keepdim=True); group_std=scores.std(1,correction=0,keepdim=True); global_std=scores.std(correction=0)
    if normalization=="global_std": std=global_std.expand_as(group_std)
    elif normalization=="group_std": std=group_std
    elif normalization=="fixed_scale": std=torch.full_like(group_std,fixed_scale)
    else: raise ValueError("Invalid normalization")
    z=adv_clip*std.clamp_min(std_floor); r=.5+.5*((scores-means)/z).clamp(-1,1)
    spread=scores.max(1).values-scores.min(1).values
    keep=torch.ones(scores.shape[0],dtype=torch.bool)
    if reward_mode=="mechanism_guard": keep=(spread>=guard_min_spread)&(scores.max(1).values>=guard_min_max)
    elif reward_mode!="standard": raise ValueError("Invalid reward mode")
    return {"r":r.float(),"keep":keep,"means":means,"std_group":group_std,"std_global":global_std,"Z":z,"spread":spread}
