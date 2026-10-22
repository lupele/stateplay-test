"""Read-only progress display for a detached Stage C controller."""
import argparse
import time
from tqdm.auto import tqdm
from .io import run_path,read_json
from .phase_progress import snapshot

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',default='study_v1');p.add_argument('--once',action='store_true');a=p.parse_args()
    run=run_path(a.run_id);path=run/'progress/overall.json'
    if not path.exists():
        ledger=read_json(run/'plan/cursor.json');total=len(read_json(run/'plan/epoch_order.json')['order'])
        print(f"C has not started: A+C {ledger['cursor']}/{total}; C consumed {ledger['local_consumed']}; remaining {total-ledger['cursor']}")
        return
    overall=read_json(path);bars=[tqdm(total=overall['total_conditions'],desc='A+C overall',unit='cond',position=0,dynamic_ncols=True)]
    for index,(label,unit) in enumerate([('generate','video'),('score','group'),('update','step')]):
        bars.append(tqdm(total=0,desc=label,unit=unit,position=index+1,dynamic_ncols=True))
    try:
        while True:
            value=read_json(path);bars[0].n=value['cursor']
            eta=value.get('eta_seconds');text=f"ETA {eta/3600:.2f}h" if eta is not None else 'ETA pending first completed batch'
            bars[0].set_postfix_str(value['status']+'; '+text)
            for phase,bar in zip(('generate','score','update'),bars[1:]):
                s=snapshot(run,value.get('batch',0),phase);bar.total=s['total'];bar.n=s['completed']
                bar.set_postfix_str(f"new work {s['units_per_second']:.3f}/s")
            if a.once or value['status']=='complete':break
            time.sleep(2)
    finally:
        for bar in bars:bar.close()

if __name__=='__main__':main()
