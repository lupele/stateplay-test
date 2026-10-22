"""Real-weight candidate/scoring/optimizer interruptions. No paid API calls."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
from .io import ROOT,run_path,read_json,write_json,sha256

def main():
    p=argparse.ArgumentParser();p.add_argument("--report-run-id",default="smoke_v1");a=p.parse_args()
    parent=run_path(a.report_run_id);run_id=a.report_run_id+"_recovery";run=run_path(run_id)
    (run/"conditions").mkdir(parents=True,exist_ok=True);(run/"logs").mkdir(exist_ok=True)
    if not (run/"conditions/catalog.json").exists():shutil.copy2(parent/"conditions/catalog.json",run/"conditions/catalog.json")
    command=[str(ROOT/"outputs/rl_posttrain/_envs/world/bin/python"),"-u","-m","rl_posttrain.train_api","--config","rl_posttrain/configs/smoke.yaml","--run-id",run_id,"--smoke","--max-optimizer-steps","3","--scorer","fake"]
    env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=os.environ.get("CUDA_VISIBLE_DEVICES","0");env["TOKENIZERS_PARALLELISM"]="false"
    def execute(extra,name,expected_failure):
        with (run/"logs"/(name+".log")).open("a") as f:result=subprocess.run(command+extra,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        if bool(result.returncode)!=expected_failure:raise AssertionError(f"Unexpected recovery execution status: {name}")
    report={"kind":"real_SFT_small_resolution_recovery","paid_API_calls":0,"events":[]}
    if not (run/"world/api/final/world.pt").exists():
        execute(["--interrupt-after-candidates","2"],"half_generated",True)
        group=next((run/"groups").glob("g-*"));m=read_json(group/"manifest.json")
        completed={c["video_id"]:{"sha256":sha256(group/"candidates"/c["video_id"]/"video.mp4"),"mtime_ns":(group/"candidates"/c["video_id"]/"video.mp4").stat().st_mtime_ns} for c in m["candidates"] if c["complete"]}
        assert len(completed)==2
        execute(["--interrupt-after-scoring"],"scored_before_update",True)
        assert read_json(run/"plan/cursor.json")["cursor"]==0
        for vid,stamp in completed.items():
            assert sha256(group/"candidates"/vid/"video.mp4")==stamp["sha256"]
            assert (group/"candidates"/vid/"video.mp4").stat().st_mtime_ns==stamp["mtime_ns"]
        execute(["--interrupt-after-optimizer-step","1"],"optimizer_committed",True)
        latest=read_json(run/"world/api/checkpoints/latest.json");assert latest["optimizer_step"]==1
        execute(["--resume",latest["checkpoint"]],"resumed_optimizer",False)
        end=read_json(run/"reports/api_completion.json")
        assert end["optimizer_step"]==3 and end["ledger"]["cursor"]==1 and end["ledger"]["inflight"] is None
        report["events"]=[{"point":"two_candidates_generated","completed_videos_reused_without_mtime_change":True},{"point":"scored_before_update","cursor_unchanged":True},{"point":"optimizer_step_1_committed","final_optimizer_step":3,"cursor":1,"duplicate_committed_updates":0}]
    else:report["events"]=read_json(parent/"reports/recovery_acceptance.json")["events"]
    report["API_partial_batch"]={"kind":"mock","evidence":"test_cache_success_no_repeat and unknown timeout no-rebilling plus 32-process persisted cache restart"}
    report["run_path"]=str(run);write_json(parent/"reports/recovery_acceptance.json",report);print(report,flush=True)
if __name__=="__main__":main()
