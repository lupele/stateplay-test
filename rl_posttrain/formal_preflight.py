"""Actual resource and unbilled API catalogue checks, with sanitized evidence."""
import argparse
import os
import shutil
import subprocess
import httpx
from .io import run_path,write_json,now
from .config import load_config

def check(config,run):
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
    gpus=[]
    for line in raw.splitlines():
        index,name,total,used=[v.strip() for v in line.split(',')]
        gpus.append({'index':int(index),'name':name,'total_MiB':int(total),'used_MiB':int(used)})
    disk=shutil.disk_usage(run);secret=os.environ.get('STATEPLAY_GPT_API_KEY') or os.environ.get('OPENAI_API_KEY')
    failures=[]
    if len(gpus)!=7 or any(g['total_MiB']<40000 for g in gpus):failures.append('Seven GPUs with measured L20 capacity are required')
    if any(g['used_MiB']>2048 for g in gpus):failures.append('A requested GPU is occupied; no other job is terminated')
    if disk.free<2*(1024**4):failures.append('Less than 2 TiB available for permanent formal artifacts')
    api={'credential_configured':bool(secret),'model':config['reward']['model'],'endpoint':config['reward']['endpoint'],
         'paid_group_requests':0,'checked':False}
    if not secret:failures.append('Configured API credential is unavailable')
    else:
        try:
            with httpx.Client(timeout=30,trust_env=False,proxy=os.environ.get('STATEPLAY_API_PROXY')) as client:
                response=client.get(config['reward']['endpoint'].rstrip('/')+'/models',headers={'Authorization':'Bearer '+secret})
            api['checked']=True;api['http_status']=response.status_code
            if response.status_code==200:
                names={r.get('id') for r in response.json().get('data',[])}
                api['required_model_listed']=config['reward']['model'] in names
                if not api['required_model_listed']:failures.append('Required gpt-6.1-sol is not listed by the configured endpoint')
            else:failures.append('API catalogue HTTP '+str(response.status_code))
        except (httpx.HTTPError,ValueError,KeyError) as error:
            api['error_type']=type(error).__name__;failures.append('API availability check failed')
    report={'time':now(),'passed':not failures,'failures':failures,'gpus':gpus,'disk_free_bytes':disk.free,
            'api':api,'four_candidate_image_protocol_check':'First formal group validates the current v2 full request; it counts in the A budget'}
    write_json(run/'reports/formal_resource_API_preflight.json',report)
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--config',required=True);a=p.parse_args()
    report=check(load_config(a.config),run_path(a.run_id))
    print({'passed':report['passed'],'failures':report['failures'],'paid_API_groups':0},flush=True)
    if not report['passed']:raise SystemExit(2)
if __name__=='__main__':main()
