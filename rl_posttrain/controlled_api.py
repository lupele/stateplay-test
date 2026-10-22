"""One authorized complete real group; credential may be supplied by hidden stdin."""
import argparse
import json
import os
from pathlib import Path
import sys
from .config import load_config
from .io import run_path,write_json
from .rewards.api_group import APIGroupScorer

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--run-id",required=True);p.add_argument("--group");p.add_argument("--api-group-limit",type=int,default=1);p.add_argument("--credentials-stdin",action="store_true");a=p.parse_args()
    if a.api_group_limit!=1:raise ValueError("Initial controlled acceptance is exactly one complete group")
    if a.credentials_stdin:
        if sys.stdin.isatty():
            import termios
            attributes=termios.tcgetattr(0);attributes[3]&=~termios.ECHO;termios.tcsetattr(0,termios.TCSANOW,attributes)
        print("Ready for configured credential on hidden stdin",flush=True)
        credential=json.loads(sys.stdin.readline())
        os.environ["STATEPLAY_GPT_API_KEY"]=credential["api_key"]
        del credential
    run=run_path(a.run_id);group=Path(a.group) if a.group else Path(__import__('rl_posttrain.io',fromlist=['read_json']).read_json(run/"reports/world_full_profile.json")["group_path"])
    c=load_config(a.config)
    try:
        scores=APIGroupScorer(c,run,1).score(group)
        write_json(run/"reports/real_api_acceptance.json",{"kind":"real_API_one_complete_group","model":c["reward"]["model"],"group_path":str(group),"images":124,"candidate_count":4,"scores":scores,"quality_evaluation":False,"group_limit":1})
        print(json.dumps({"model":c["reward"]["model"],"scores":scores,"images":124,"group_path":str(group)}),flush=True)
    finally:os.environ.pop("STATEPLAY_GPT_API_KEY",None)
if __name__=="__main__":main()
