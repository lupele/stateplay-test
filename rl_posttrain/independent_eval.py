"""Paper Appendix B reducers; require real extractor artifacts, never reward proxies."""
import argparse
import numpy as np
import pandas as pd
from .io import read_json,write_json

def movement(trajectory,key):
    """Normalized SAM2.1/Grounding-DINO x,y,h; 5-frame rolling median."""
    a=np.asarray(trajectory,dtype=float)
    if a.ndim!=2 or a.shape[1]!=3:raise ValueError("Trajectory must be [T,3] x,y,h in normalized coordinates")
    valid=np.isfinite(a).all(1)&(a[:,2]>0)
    if not valid.any():return None
    a=pd.DataFrame(a[valid]).rolling(5,min_periods=1,center=True).median().to_numpy()
    x,y,h=a.T;x0,y0,h0=x[0],y[0],h[0]
    if key=="LEFT":return bool(x[-1]-x0<=-.025)
    if key=="RIGHT":return bool(x[-1]-x0>=.025)
    if key=="UP":return bool(y.min()-y0<=-.030)
    if key=="DOWN":
        mid=slice(len(y)//3,max(len(y)//3+1,2*len(y)//3))
        return bool(np.median(h[mid])<=.85*h0 or np.median(y[mid])-y0>=.010)
    raise ValueError("Move-Acc uses the four single-key movement commands")

def attack(probabilities,keys):
    p=np.asarray(probabilities,dtype=float)
    if p.ndim!=2 or p.shape[1]!=6 or len(keys)!=6 or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError("ClipAttackNet outputs must be [T,6] probabilities")
    if p.max()<=.7:return "noop"
    return keys[np.unravel_index(p.argmax(),p.shape)[1]]

def main():
    p=argparse.ArgumentParser();p.add_argument("--artifacts",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    data=read_json(a.artifacts)
    if data.get("protocol")!="stateplay-paper-appendix-B-single-action-41frames":raise ValueError("Original Move/Att benchmark is separate from the 5-second mechanism benchmark")
    if data.get("movement_extractor")!="SAM2.1+GroundingDINO" or data.get("attack_extractor")!="ClipAttackNet":raise ValueError("Supply authentic frozen extractor outputs and their checkpoint hashes")
    results=[];moves=[];attacks=[]
    for clip in data["clips"]:
        if clip["key"] in ("LEFT","RIGHT","UP","DOWN"):
            hit=movement(clip["trajectory"],clip["key"])
            if hit is not None:moves.append(hit)
        else:
            hit=attack(clip["attack_probabilities"],data["attack_key_order"])==clip["key"];attacks.append(hit)
        results.append({"clip_id":clip["clip_id"],"correct":hit})
    report={"protocol":data["protocol"],"Move-Acc":float(np.mean(moves)) if moves else None,"Att-Acc":float(np.mean(attacks)) if attacks else None,"movement_valid":len(moves),"attack_valid":len(attacks),"results":results,"extractor_checkpoint_hashes":data["extractor_checkpoint_hashes"],"ambiguity":"Paper does not fix centered/causal median or h_mid window. This implementation explicitly uses centered 5-frame median and middle-third median; cross-paper exact numeric reproduction requires authors' extractor/window artifacts."}
    write_json(a.output,report);print(report)
if __name__=="__main__":main()
