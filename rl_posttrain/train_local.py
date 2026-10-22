import os
import subprocess
import sys
from .train_world import parser,train
from .io import run_path,read_json

if __name__=="__main__":
    p=parser();p.add_argument("--launch-from-profile",action="store_true");p.set_defaults(scorer="local");a=p.parse_args()
    if a.launch_from_profile and not os.environ.get("RL_POSTTRAIN_PROFILE_CHILD"):
        from .timeslice import main
        sys.argv=[x for x in sys.argv if x!='--launch-from-profile'];main()
    else:train(a,"local")
