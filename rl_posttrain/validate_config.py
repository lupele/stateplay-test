import argparse
from pathlib import Path
from .config import load_config,validate
from .io import ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument("--config");p.add_argument("--all",action="store_true");p.add_argument("--run-id");a=p.parse_args()
    paths=sorted((ROOT/"rl_posttrain/configs").glob("*.yaml")) if a.all else [Path(a.config)]
    for path in paths:
        c=validate(load_config(path),a.run_id);print(f"VALID {path.name} {c['config_hash'][:16]}")
if __name__=="__main__":main()
