"""Actual cumulative formal outcomes, usable while running or after completion."""
import argparse
import collections
import json
import sqlite3
from .io import read_json,write_json,run_path,reward_path,reward_checkpoint,world_checkpoint,now,api_cache_path

def build(run):
    ledger=read_json(run/'plan/cursor.json');state=read_json(run/'plan/stage_state.json')
    videos=groups=0;classes=collections.Counter();teacher_usage=collections.Counter();responses=0
    for p in (run/'groups').glob('*/manifest.json'):
        m=read_json(p);groups+=1;videos+=sum(c['complete'] for c in m['candidates']);classes[m['condition']['mechanism_class']]+=1
        for response in (p.parent/'teacher').rglob('response_attempt_*.json'):
            raw=read_json(response);responses+=1
            response_body=raw.get('response')
            usage=response_body.get('usage') if isinstance(response_body,dict) else None
            for key,value in (usage or {}).items():
                if isinstance(value,(int,float)) and not isinstance(value,bool):teacher_usage[key]+=value
    request_states={};attempts=0;dbpath=api_cache_path(run)
    if dbpath.exists():
        with sqlite3.connect(dbpath) as db:
            request_states=dict(db.execute('SELECT status,count(*) FROM requests GROUP BY status').fetchall())
            attempts=db.execute('SELECT coalesce(sum(attempts),0) FROM requests').fetchone()[0]
    historical_states={};historical_attempts=0
    legacy=run/'requests/cache.sqlite3'
    if dbpath!=legacy and legacy.exists():
        with sqlite3.connect(legacy) as db:
            historical_states=dict(db.execute('SELECT status,count(*) FROM requests GROUP BY status'))
            historical_attempts=db.execute('SELECT coalesce(sum(attempts),0) FROM requests').fetchone()[0]
    stages={}
    for stage in ('api','local'):
        logs=run/'metrics'/(stage+'_steps.jsonl');rows=[json.loads(line) for line in logs.read_text().splitlines()] if logs.exists() else []
        batches=[read_json(p) for p in (run/'plan').glob('batch-*-normalization.json')]
        batches=[v for v in batches if v.get('stage')==stage]
        valid=sum(sum(v['normalization']['keep']) for v in batches)
        seen=sum(len(v['normalization']['keep']) for v in batches)
        stages[stage]={'new_conditions_committed':ledger[stage+'_consumed'],'logged_optimizer_updates':len(rows),
                       'latest_optimizer_step':rows[-1]['optimizer_step'] if rows else None,
                       'kept_groups':valid,'scored_groups':seen,'filtered_group_fraction':1-valid/seen if seen else None,
                       'last_step':rows[-1] if rows else None,
                       'checkpoint':str(world_checkpoint(run,stage)) if world_checkpoint(run,stage).exists() else None}
    rm=reward_path(run);quality=rm/'reports/quality_gate.json';evaluation=rm/'reports/evaluation.json'
    out={'time':now(),'run_id':run.name,'formal_completed':state.get('stage')=='complete','current_stage':state['stage'],
         'ledger':ledger,'planned_conditions':len(read_json(run/'plan/epoch_order.json')['order']),
         'groups_with_manifests':groups,'complete_candidate_videos':videos,'class_counts':dict(classes),
         'API_request_states':request_states,'API_http_submission_attempts':attempts,
         'historical_provider_request_states':historical_states,
         'historical_provider_http_submission_attempts':historical_attempts,
         'API_http_submission_attempts_all_providers':attempts+historical_attempts,
         'saved_provider_responses':responses,'reported_provider_usage':dict(teacher_usage),
         'API_cost_currency':'No price is inferred; raw usage/response assets are retained',
         'stages':stages,'reward_checkpoint':str(reward_checkpoint(run)),
         'reward_quality_gate':read_json(quality) if quality.exists() else None,
         'reward_test_metrics':{k:v for k,v in read_json(evaluation).items() if k!='results'} if evaluation.exists() else None,
         'source_protection':str(run/'reports/source_identity_after.json'),
         'export':str(run/'world/local/export/StatePlay_NFT.safetensors') if (run/'world/local/export/StatePlay_NFT.safetensors').exists() else None,
         'independent_world_quality_metrics':'Missing extractor weights remain documented; no independent mechanics improvement is inferred from training rewards'}
    write_json(run/'reports/formal_results.json',out);return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();out=build(run_path(a.run_id))
    print({k:out[k] for k in ('run_id','formal_completed','current_stage','complete_candidate_videos','API_request_states')})
if __name__=='__main__':main()
