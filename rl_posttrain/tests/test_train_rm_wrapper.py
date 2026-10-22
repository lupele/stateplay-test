import json
import os
from pathlib import Path
import subprocess

def test_training_and_evaluation_arguments_are_separate(tmp_path):
    fake=tmp_path/'python';log=tmp_path/'calls.jsonl'
    fake.write_text('#!/usr/bin/env python3\nimport json,os,sys\nwith open(os.environ["CALLS"],"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\n')
    fake.chmod(0o755)
    env=dict(os.environ,STATEPLAY_REWARD_PYTHON=str(fake),CALLS=str(log),CUDA_VISIBLE_DEVICES='0,1')
    script=Path(__file__).resolve().parents[1]/'scripts/train_rm.sh'
    subprocess.run(['bash',str(script),'--run-id','test','--config','custom.yaml','--smoke','--smoke-accumulation','4','--audit-replay','--resume','checkpoint','--human-review','review.json'],env=env,check=True)
    train,evaluate=[json.loads(x) for x in log.read_text().splitlines()]
    assert '--audit-replay' in train and '--smoke-accumulation' in train and '--resume' in train
    assert '--human-review' not in train
    assert '--human-review' in evaluate and '--check-reload' in evaluate and '--permutation-check' not in evaluate
    assert '--audit-replay' not in evaluate and '--resume' not in evaluate and '--smoke-accumulation' not in evaluate
