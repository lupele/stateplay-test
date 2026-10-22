"""Atomic progress snapshots. Counts include cached work; speed counts only new work."""
import time
import json
from .io import read_json,write_json,now

class PhaseProgress:
    def __init__(self,run,batch,phase,rank,total,initial=0):
        self.path=run/'progress'/f'batch-{batch:06d}'/f'{phase}-rank-{rank}.json'
        self.total=total;self.initial=initial;self.done=initial;self.started=time.monotonic()
        self.phase=phase;self.rank=rank;self.batch=batch;self.write()
    def write(self,status='running'):
        elapsed=time.monotonic()-self.started;new=self.done-self.initial
        value={'phase':self.phase,'batch':self.batch,'rank':self.rank,
            'total':self.total,'completed':self.done,'cached_at_start':self.initial,
            'new_completed':new,'work_seconds':elapsed,'units_per_second':new/elapsed if elapsed else 0,
            'status':status,'updated_at':now()}
        value.update(getattr(self,'extra',{}))
        try:write_json(self.path,value)
        except OSError as error:
            if not getattr(self,'warned',False):print('Progress snapshot unavailable; preserving computation and primary artifacts:',type(error).__name__,flush=True)
            self.warned=True
    def advance(self,n=1):self.done+=n;self.write()
    def finish(self):self.write('complete')
    def start_work(self):self.started=time.monotonic();self.write()

def snapshot(run,batch,phase,not_before=None):
    rows=[]
    for path in (run/'progress'/f'batch-{batch:06d}').glob(phase+'-rank-*.json'):
        try:rows.append(read_json(path))
        except (OSError,json.JSONDecodeError):continue  # Observer metadata must not kill GPU workers.
    if not_before:rows=[r for r in rows if r['updated_at']>=not_before]
    elapsed=max((r['work_seconds'] for r in rows),default=0)
    return {'completed':sum(r['completed'] for r in rows),'total':sum(r['total'] for r in rows),
            'new_completed':sum(r['new_completed'] for r in rows),'work_seconds':elapsed,
            'units_per_second':sum(r['new_completed'] for r in rows)/elapsed if elapsed else 0,
            'ranks_loading':sum(r['status']=='loading' for r in rows)}
