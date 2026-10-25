import numpy as np
from ..io import write_json,read_json,shared_exists

def audit_metrics(teacher,student):
    t=np.asarray(teacher);s=np.asarray(student)
    if t.shape!=s.shape or t.ndim!=2 or t.shape[1]!=4:raise ValueError("Audit shape mismatch")
    pairs=[]
    for tg,sg in zip(t,s):
        for i in range(4):
            for j in range(i):
                if abs(tg[i]-tg[j])>=10:pairs.append(np.sign(tg[i]-tg[j])==np.sign(sg[i]-sg[j]))
    return {"mae":float(np.abs(t-s).mean()),"pair_agreement":float(np.mean(pairs)) if pairs else None,"valid_pairs":len(pairs)}

def record_audit(path,teacher,student):
    state=read_json(path) if shared_exists(path) else {"windows":[],"consecutive_bad":0,"paused":False}
    metrics=audit_metrics(teacher,student)
    bad=metrics["mae"]>15 or (metrics["pair_agreement"] is not None and metrics["pair_agreement"]<.65)
    state["consecutive_bad"]=state["consecutive_bad"]+1 if bad else 0
    state["paused"]=state["consecutive_bad"]>=2;state["windows"].append(metrics);write_json(path,state);return state

def audit_count(state,ledger,config,batch_length):
    initial=config["reward_model"]["audit_initial_conditions"]
    initial_seen=state.get("initial_seen",0)
    if initial_seen<initial:return min(batch_length,initial-initial_seen)
    pending=len(state.get("pending",[]));every=config["reward_model"]["audit_every_batches"]
    if pending or (ledger.get("local_batches",0)>0 and ledger.get("local_batches",0)%every==0):return min(batch_length,config["reward_model"]["audit_groups"]-pending)
    return 0

def append_audit(path,groups,teacher,student,initial_limit=56,window_size=28):
    state=read_json(path) if shared_exists(path) else {"windows":[],"consecutive_bad":0,"paused":False,"pending":[],"initial_seen":0}
    state.setdefault("pending",[]);state.setdefault("initial_seen",0)
    seen=set(state.get("seen_groups",[]))
    for group,t,s in zip(groups,teacher,student):
        if str(group) in seen:continue
        seen.add(str(group))
        state["pending"].append({"group":str(group),"teacher":t,"student":s})
        state["initial_seen"]=min(initial_limit,state["initial_seen"]+1)
    while len(state["pending"])>=window_size:
        window=state["pending"][:window_size];state["pending"]=state["pending"][window_size:]
        m=audit_metrics([g["teacher"] for g in window],[g["student"] for g in window]);m["groups"]=[g["group"] for g in window]
        bad=m["mae"]>15 or (m["pair_agreement"] is not None and m["pair_agreement"]<.65)
        state["consecutive_bad"]=state["consecutive_bad"]+1 if bad else 0;state["windows"].append(m)
        state["paused"]=state["consecutive_bad"]>=2
    state["seen_groups"]=sorted(seen);write_json(path,state);return state
