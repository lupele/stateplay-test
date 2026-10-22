"""Own and stop only the scorer process launched for this local training run."""
from .train_world import parser
from .config import load_config
from .io import run_path,read_json
from .orchestrate import execute,start_server,torchrun

def main():
    from .timeslice import main as timeslice_main
    return timeslice_main()

def legacy_dedicated_main():
    a=parser().parse_args();run=run_path(a.run_id);c=load_config(a.config)
    profile=read_json(run/"reports/joint_profile.json");top=profile["selected_topology"]
    if top is None:raise RuntimeError("No verified joint GPU assignment")
    server,log=start_server(c,run,a.run_id,a.smoke,top["reward_gpus"])
    args=["--config",a.config,"--run-id",a.run_id,"--scorer","local"]
    if a.smoke:args += ["--smoke","--max-optimizer-steps",str(a.max_optimizer_steps or 3)]
    if a.resume:args += ["--resume",a.resume]
    try:execute(torchrun("rl_posttrain.train_local",args,top["world_gpus"]),run,"local_training",top["world_gpus"])
    finally:server.terminate();server.wait(timeout=30);log.close()
if __name__=="__main__":main()
