import argparse
import collections
from pathlib import Path
from ..io import read_json,write_json,digest,run_path,reward_path,teacher_path
from ..rewards.base import parse_scores

def build_dataset(run,build_splits=True,config=None,allow_legacy=False):
    groups=[]
    for path in sorted((run/"groups").glob("*/manifest.json")):
        root=path.parent;score=teacher_path(root)/"scores.json"
        if not score.exists():continue
        m=read_json(path);s=read_json(score)
        from ..data.provenance import RENDERING_VERSION,require_current_group
        if not allow_legacy and m.get('rendering',{}).get('version')!=RENDERING_VERSION:continue
        if config is not None:require_current_group(root,config,allow_legacy)
        if s["scorer"]!="gpt-6.1-sol":continue
        parse_scores(s["scores"])
        if len(m["candidates"])!=4 or not all(x["complete"] for x in m["candidates"]):raise ValueError("Incomplete teacher group")
        if sum(len(v) for v in m["frame_assets"].values())!=124:raise ValueError("Frame protocol mismatch")
        if not allow_legacy:
            from .validate_assets import validate_assets
            validate_assets(root,m,s,config)
        groups.append({"group_path":str(root.resolve()),"group_id":root.name,"condition_id":m["condition_id"],"cluster_id":m["condition"]["cluster_id"],"mechanism_class":m["condition"]["mechanism_class"],"scores":s["scores"],"frame_asset_hash":m["frame_asset_hash"],"protocol":m["protocol"],"prompt_sha256":m["prompt_sha256"]})
    if not groups:raise ValueError("No eligible complete real API teacher groups; legacy rendering and fake labels are excluded by default")
    if not allow_legacy:
        index_path=run/'conditions/leakage_index.json'
        if not index_path.exists():raise ValueError('Formal RM requires source/action/temporal leakage index')
        index=read_json(index_path);excluded=set(index['holdout_overlap_excluded'])
        for g in groups:
            if g['condition_id'] in excluded:raise ValueError('Teacher group overlaps an independent holdout')
            if g['condition_id'] not in index['source_clusters']:raise ValueError('Teacher source absent from leakage index')
            g['cluster_id']=index['source_clusters'][g['condition_id']]
    clusters={};splits={"train":[],"validation":[],"test":[]}
    for g in groups:
        value=int(digest(g["cluster_id"])[:8],16)%100
        split="train" if value<80 else "validation" if value<90 else "test"
        clusters[g["cluster_id"]]=split;splits[split].append(g)
    # Development may repeat a single actual teacher group, but must never relabel it as a quality set.
    report={"schema":"rl-rm-dataset-v1","groups":groups,"splits":splits,"clusters":clusters,"unique_conditions":len(set(x["condition_id"] for x in groups)),"class_counts":dict(collections.Counter(x["mechanism_class"] for x in groups)),"cluster_limitation":"Legacy development clusters" if allow_legacy else "Source/action/temporal duplicate components; source-episode independence still requires manual review."}
    path=reward_path(run)/"splits/manifest.json"
    if path.exists() and read_json(path)["clusters"]!={k:v for k,v in clusters.items() if k in read_json(path)["clusters"]}:raise ValueError("Immutable split assignment changed")
    if build_splits:write_json(path,report)
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument("--run-id",required=True);p.add_argument("--validate",action="store_true");p.add_argument("--build-splits",action="store_true");a=p.parse_args()
    report=build_dataset(run_path(a.run_id),a.build_splits);print({"groups":len(report["groups"]),"split_counts":{k:len(v) for k,v in report["splits"].items()},"quality_ready":False})
if __name__=="__main__":main()
