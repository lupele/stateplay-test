"""Sequential all-GPU C phases with durable accounting and visible throughput."""
import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import yaml
from tqdm.auto import tqdm
from .config import load_config,validate
from .io import ROOT,run_path,read_json,write_json,append_jsonl,digest,lock,now,reward_checkpoint
from .phase_progress import snapshot
from .orchestrate import WORLD,REWARD

def gpu_ids(value):
    try:ids=[int(x.strip()) for x in value.split(',')]
    except (ValueError,AttributeError):raise argparse.ArgumentTypeError('GPU IDs must be comma-separated integers')
    if not ids or len(set(ids))!=len(ids) or min(ids)<0:raise argparse.ArgumentTypeError('GPU IDs must be distinct nonnegative integers')
    return ids

def phases_needed(inflight):
    if not inflight:return ['generate','score','update']
    if inflight['phase'] in ('scored','updating'):return ['update']
    if inflight['phase']=='generated':return ['score','update']
    return ['generate','score','update']

def gpu_inventory(gpus):
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.free',
        '--format=csv,noheader,nounits'],text=True)
    devices={}
    for line in output.splitlines():
        index,name,total,free=[x.strip() for x in line.split(',')]
        devices[int(index)]={'index':int(index),'name':name,'memory_total_MiB':int(total),'memory_free_MiB':int(free)}
    missing=set(gpus)-devices.keys()
    if missing:raise ValueError('Requested GPU IDs are unavailable: '+str(sorted(missing)))
    return [devices[g] for g in gpus]

def runtime_config(c,gpus,generation_batch_size=None,reward_batch_size=None):
    c=copy.deepcopy(c);runtime=c.setdefault('stage_c',{})
    runtime.setdefault('generation_batch_size',1);runtime.setdefault('reward_batch_size',1)
    if generation_batch_size is not None:runtime['generation_batch_size']=generation_batch_size
    if reward_batch_size is not None:runtime['reward_batch_size']=reward_batch_size
    if not 1<=runtime['generation_batch_size']<=4:raise ValueError('generation batch size must be in [1,4]')
    if runtime['reward_batch_size']<1:raise ValueError('reward batch size must be positive')
    c['topology']={'strategy':'timeslice','world_gpus':gpus,'reward_gpus':gpus}
    # Do not scale the condition budget or accumulation with the number of GPUs.
    c['config_hash']=digest({k:v for k,v in c.items() if k!='config_hash'})
    return c

def command(phase,config,run_id,gpus,version,smoke=False,max_steps=None,audit_replay=False):
    python=REWARD if phase=='score' else WORLD
    module='rl_posttrain.reward_model.score_phase' if phase=='score' else 'rl_posttrain.train_local'
    result=[python,'-m','torch.distributed.run','--standalone','--nproc_per_node',str(len(gpus)),'-m',module,
            '--config',str(config),'--run-id',run_id]
    if phase!='score':
        result+=['--phase',phase,'--single-batch','--reward-version',version]
        if audit_replay:result+=['--audit-replay']
    if smoke:
        result+=['--smoke']
        if phase!='score':result+=['--max-optimizer-steps',str(max_steps or 100)]
    return result

class Dashboard:
    def __init__(self,run,total):
        self.run=run;self.total=total;self.started=time.monotonic();self.initial=read_json(run/'plan/cursor.json')['cursor']
        self.overall=tqdm(total=total,initial=self.initial,desc='A+C overall',unit='cond',position=0,dynamic_ncols=True)
        self.bars={p:tqdm(total=0,desc=p,unit=u,position=i+1,dynamic_ncols=True) for i,(p,u) in enumerate([('generate','video'),('score','group'),('update','step')])}
    def begin(self,batch,phase,total):
        bar=self.bars[phase];bar.reset(total=total);bar.set_postfix_str('loading models / releasing previous phase')
        self.phase=phase;self.batch=batch;self.phase_started=time.monotonic();self.wall_started=now()
    def refresh(self):
        s=snapshot(self.run,self.batch,self.phase,self.wall_started);bar=self.bars[self.phase]
        if s['total']:
            bar.total=s['total'];bar.n=s['completed']
            prefix=f"loading {s['ranks_loading']} rank(s); " if s.get('ranks_loading') else ''
            bar.set_postfix_str(prefix+f"work {s['units_per_second']:.3f}/s; phase {time.monotonic()-self.phase_started:.0f}s",refresh=False)
        bar.refresh()
        try:ledger=read_json(self.run/'plan/cursor.json')
        except (OSError,json.JSONDecodeError):return
        cursor=ledger['cursor'];self.overall.n=cursor
        history=self.run/'progress/batches.jsonl'
        rows=[json.loads(x) for x in history.read_text().splitlines()] if history.exists() else []
        measured=sum(x['conditions'] for x in rows);seconds=sum(x['seconds'] for x in rows)
        per=seconds/measured if measured else None
        eta=(self.total-cursor)*per if per is not None else None
        estimated=now_plus(eta) if eta is not None else None
        self.overall.set_postfix_str(f"ETA {int(eta/60)} min incl. model switches" if eta is not None else 'ETA pending first completed batch',refresh=False)
        self.overall.refresh()
        value={'status':'running','phase':self.phase,'batch':self.batch,
            'cursor':cursor,'total_conditions':self.total,'A_consumed':ledger['api_consumed'],
            'remaining_conditions':self.total-cursor,'phase_progress':s,
            'phase_wall_seconds':time.monotonic()-self.phase_started,'session_wall_seconds':time.monotonic()-self.started,
            'measured_seconds_per_condition':per,'eta_seconds':eta,'estimated_finish':estimated,
            'eta_basis':'completed C batches, includes model loading, process switching and checkpoint saves; excludes future manual pauses',
            'updated_at':now()}
        try:write_json(self.run/'progress/overall.json',value)
        except OSError:pass  # Primary ledger, phase journal and model state remain mandatory.
    def close(self,status):
        path=self.run/'progress/overall.json';value=read_json(path) if path.exists() else {}
        cursor=read_json(self.run/'plan/cursor.json')['cursor']
        value.update(status=status,cursor=cursor,total_conditions=self.total,remaining_conditions=self.total-cursor,updated_at=now())
        if cursor==self.total:value['eta_seconds']=0
        write_json(path,value)
        for bar in self.bars.values():bar.close()
        self.overall.close()

def now_plus(seconds):
    return (datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=seconds)).isoformat()

def execute_phase(cmd,run,phase,batch,gpus,dashboard):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),PYTHONUNBUFFERED='1',
        TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    (run/'logs').mkdir(parents=True,exist_ok=True);path=run/'logs'/f'C_batch-{batch:06d}_{phase}.log'
    tick=time.monotonic();process=None
    try:
        with path.open('a') as log:
            process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            start_tick=Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
            record={'pid':process.pid,'start_tick':start_tick,'command':cmd,'gpus':gpus,'active':True,'started_at':now()}
            target=run/'plan/processes'/f'C_timeslice_{phase}.json';write_json(target,record)
            while process.poll() is None:
                dashboard.refresh();time.sleep(2)
            dashboard.refresh();code=process.wait()
            record.update(active=False,returncode=code,finished_at=now());write_json(target,record)
        append_jsonl(run/'progress/phases.jsonl',{'batch':batch,'phase':phase,'seconds':time.monotonic()-tick,
            'returncode':code,'gpus':gpus,'log':str(path),'finished_at':now()})
        if code:raise RuntimeError(f'{phase} failed ({code}); durable state retained: {path}')
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    return time.monotonic()-tick

def run(a):
    run=run_path(a.run_id);original=load_config(a.config)
    if not a.dry_run:
        from .prepare_local_mode import prepare_mode
        original=prepare_mode(original,run,a.update_mode)
    gpus=a.gpus
    if gpus is None and os.environ.get('CUDA_VISIBLE_DEVICES'):gpus=gpu_ids(os.environ['CUDA_VISIBLE_DEVICES'])
    if gpus is None:gpus=sorted(set(original['topology']['world_gpus']+original['topology']['reward_gpus']))
    c=runtime_config(original,gpus,a.generation_batch_size,a.reward_batch_size)
    if a.dry_run:c['nft']['mode']=a.update_mode
    validate(c,a.run_id)
    if a.smoke and a.run_id=='study_v1':raise ValueError('Software smoke must use an isolated run, never study_v1')
    if a.dry_run:
        print(json.dumps({'gpus':gpus,'generation_batch_size':c['stage_c']['generation_batch_size'],
            'reward_batch_size':c['stage_c']['reward_batch_size'],'conditions_per_batch':c['rollout']['conditions_per_batch'],
            'update_mode':c['nft']['mode'],'epochs':c['plan']['epochs'],'A_plus_C_one_pass':True,'formal_training_started':False},indent=2));return
    if not a.smoke:
        from .train_world import formal_gate
        formal_gate(run,'local')
    selected=reward_checkpoint(run);manifest=read_json(selected/'manifest.json');version=manifest['version']
    hardware=gpu_inventory(gpus)
    ledger=read_json(run/'plan/cursor.json');order=read_json(run/'plan/epoch_order.json')['order'];total=len(order)
    if len(set(order))!=total or not 0<=ledger['cursor']<=total or ledger['cursor']!=ledger['api_consumed']+ledger['local_consumed']:
        raise ValueError('One-pass ledger accounting mismatch')
    config=run/'plan'/f"C_timeslice_{c['config_hash'][:16]}.yaml"
    if not config.exists():config.write_text(yaml.safe_dump(c,sort_keys=False))
    write_json(run/'plan/C_execution.json',{'strategy':'timeslice','gpus':gpus,'generation_batch_size':c['stage_c']['generation_batch_size'],
        'reward_batch_size':c['stage_c']['reward_batch_size'],'conditions_per_batch':c['rollout']['conditions_per_batch'],
        'one_pass_total':total,'remaining_conditions':total-ledger['cursor'],'reward_version':version,
        'config':str(config),'smoke':a.smoke,'updated_at':now()})
    execution=read_json(run/'plan/C_execution.json');execution['hardware']=hardware;write_json(run/'plan/C_execution.json',execution)
    dashboard=Dashboard(run,total);status='failed'
    try:
        with lock(run/'plan/.timeslice.lock'):
            while read_json(run/'plan/cursor.json')['cursor']<total:
                before=read_json(run/'plan/cursor.json');batch=before['inflight']['batch'] if before['inflight'] else before['batch']
                for phase in phases_needed(before['inflight']):
                    # A scored or partially updated buffer must never be regenerated.
                    current=read_json(run/'plan/cursor.json');inflight=current['inflight']
                    if inflight and inflight['phase'] in ('scored','updating') and phase in ('generate','score'):continue
                    if inflight and inflight['phase']=='generated' and phase=='generate':continue
                    count=len(inflight['condition_ids']) if inflight else min(c['rollout']['conditions_per_batch'],total-current['cursor'])
                    dashboard.begin(batch,phase,count*4 if phase=='generate' else count if phase=='score' else 0)
                    cmd=command(phase,config,a.run_id,gpus,version,a.smoke,a.max_optimizer_steps,a.audit_replay)
                    if phase=='score':cmd+=['--batch-size',str(c['stage_c']['reward_batch_size'])]
                    if a.resume and phase!='score' and not (run/'world/local/checkpoints/latest.json').exists():cmd+=['--resume',a.resume]
                    execute_phase(cmd,run,phase,batch,gpus,dashboard)
                    if phase=='score':
                        from .data.sampler import Ledger
                        Ledger(run/'plan').phase('scored')
                    if a.stop_after_phase==phase:status='test_stopped_after_'+phase;return
                after=read_json(run/'plan/cursor.json')
                if after['inflight']:
                    status='stopped_at_optimizer_limit';return
                events=[json.loads(x) for x in (run/'progress/phases.jsonl').read_text().splitlines()]
                seconds=sum(e['seconds'] for e in events if e['batch']==batch)
                append_jsonl(run/'progress/batches.jsonl',{'batch':batch,'conditions':after['cursor']-before['cursor'],
                    'seconds':seconds,'cursor':after['cursor'],'finished_at':now()})
                dashboard.refresh()
                if a.smoke:break
            status='complete' if read_json(run/'plan/cursor.json')['cursor']==total else 'smoke_batch_complete'
    except KeyboardInterrupt:
        status='interrupted_state_preserved';raise
    finally:dashboard.close(status)

def parser():
    from .train_world import parser as world_parser
    p=world_parser();p.description=__doc__
    p.add_argument('--gpus',type=gpu_ids,help='Physical GPU IDs, e.g. 0,1,2,3; every phase uses the same list')
    p.add_argument('--generation-batch-size',type=int,help='Simultaneous candidates of one condition per GPU, 1..4')
    p.add_argument('--reward-batch-size',type=int,help='Complete 124-image scoring requests per GPU batch')
    p.add_argument('--update-mode','--train-mode',dest='update_mode',choices=['B-only','Coupling-only','Both'],default='B-only',help='C trainable modules; default B-only; changing existing C progress is refused')
    p.add_argument('--dry-run',action='store_true');p.add_argument('--stop-after-phase',choices=['generate','score','update'])
    return p

def main():
    def stopped(signum,frame):raise KeyboardInterrupt('Controller terminated; preserving the current buffer and checkpoint')
    signal.signal(signal.SIGTERM,stopped)
    run(parser().parse_args())
if __name__=='__main__':main()
