"""Explicit pre-update provider migration; preserve ledger and prior API assets."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sqlite3
from .config import load_config,validate
from .io import ROOT,run_path,read_json,write_json,atomic_bytes,sha256,digest,now,teacher_path,api_cache_path
from .rewards.api_group import RequestStore,APIGroupScorer
from .rewards.protocol import payload


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--config',required=True);p.add_argument('--successful-probe',required=True);a=p.parse_args()
    run=run_path(a.run_id);config=validate(load_config(a.config),a.run_id)
    version='provider-'+config['config_hash'][:12];report_path=run/'reports/provider_migrations'/f'{version}.json'
    with (run/'plan/.formal_controller.lock').open('a+') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from .formal import owned_alive
        if any(owned_alive(read_json(f),run) for f in (run/'plan/processes').glob('*.json')):raise RuntimeError('Owned worker still active')
        if report_path.exists() and read_json(report_path).get('completed'):
            print({'migration_already_completed':True,'report':str(report_path)});return
        old=read_json(run/'plan/config.json')
        for key in set(old)|set(config):
            if key not in ('reward','config_hash') and old.get(key)!=config.get(key):raise ValueError('Provider migration changed training/data configuration: '+key)
        changes={k:(old['reward'].get(k),config['reward'].get(k)) for k in set(old['reward'])|set(config['reward']) if old['reward'].get(k)!=config['reward'].get(k)}
        if set(changes)-{'endpoint','concurrency_backend'}:raise ValueError('Unexpected reward protocol change')
        if config['reward']['model']!='gpt-6.1-sol':raise ValueError('Unexpected teacher model')
        ledger_bytes=(run/'plan/cursor.json').read_bytes();order_bytes=(run/'plan/epoch_order.json').read_bytes();ledger=read_json(run/'plan/cursor.json')
        if ledger['cursor']!=0 or not ledger['inflight'] or ledger['inflight']['phase']!='generated':raise ValueError('This migration is only for the preserved, not-yet-updated first batch')
        probe=Path(a.successful_probe).resolve();result=read_json(probe/'result.json')
        if not result['valid_scores'] or result['endpoint']!=config['reward']['endpoint'] or result['model']!=config['reward']['model']:raise ValueError('Successful probe identity differs')
        source_group=Path(read_json(probe/'origin.json')['source_group']).resolve()
        if source_group.parent!=(run/'groups').resolve():raise ValueError('Probe escaped formal group scope')
        copied_teacher=probe/'groups'/source_group.name/'teacher'
        if sha256(source_group/'manifest.json')!=read_json(probe/'origin.json')['source_manifest_sha256']:raise ValueError('Probe source manifest changed')
        archive=run/'provenance/provider_migrations'/version;archive.mkdir(parents=True,exist_ok=True)
        original_config=archive/'plan_config_before.json'
        if not original_config.exists():atomic_bytes(original_config,(run/'plan/config.json').read_bytes())
        original_state=archive/'stage_state_before.json'
        if not original_state.exists():atomic_bytes(original_state,(run/'plan/stage_state.json').read_bytes())
        old_assets={str(f):sha256(f) for root in (run/'groups').iterdir() for f in (root/'teacher').rglob('*') if f.is_file() and 'versions' not in f.relative_to(root/'teacher').parts and f.name!='active.json'}
        write_json(archive/'original_teacher_hashes.json',old_assets)
        groups=[]
        for manifest in sorted((run/'groups').glob('*/manifest.json')):
            root=manifest.parent
            target=root/'teacher/versions'/version;target.mkdir(parents=True,exist_ok=True)
            write_json(root/'teacher/active.json',{'relative_path':'versions/'+version,'endpoint':config['reward']['endpoint'],'model':config['reward']['model'],'selected_at':now()})
            groups.append(root.name)
        target=teacher_path(source_group)
        for item in copied_teacher.rglob('*'):
            if not item.is_file():continue
            destination=target/item.relative_to(copied_teacher)
            destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists():
                if sha256(destination)!=sha256(item):raise ValueError('Refusing to overwrite imported teacher evidence')
            else:shutil.copy2(item,destination)
        _,key,mapping=payload(source_group,config)
        labels=read_json(target/'scores.json')
        if key!=labels['cache_key'] or mapping!=labels['mapping']:raise ValueError('Imported labels/cache do not match the exact formal request')
        cache_relative='providers/'+version+'/cache.sqlite3'
        active_cache=run/'requests'/cache_relative;store=RequestStore(active_cache)
        with sqlite3.connect(probe/'requests/cache.sqlite3') as db:
            row=db.execute('SELECT key,group_id,status,owner,result,attempts,updated,error FROM requests WHERE key=?',(key,)).fetchone()
        if row is None or row[2]!='succeeded' or row[5]!=1:raise ValueError('Expected one actual successful probe submission')
        with store.connect() as db:
            current=db.execute('SELECT status,result FROM requests WHERE key=?',(key,)).fetchone()
            if current is None:db.execute('INSERT INTO requests(key,group_id,status,owner,result,attempts,updated,error) VALUES(?,?,?,?,?,?,?,?)',row)
            elif current!=('succeeded',row[4]):raise ValueError('Active cache already contains conflicting labels')
        write_json(run/'requests/active.json',{'relative_path':cache_relative,'endpoint':config['reward']['endpoint'],'model':config['reward']['model'],'historical_database':'cache.sqlite3','selected_at':now()})
        write_json(run/'plan/config.json',config)
        from .reward_model.validate_assets import validate_assets
        validate_assets(source_group,read_json(source_group/'manifest.json'),labels,config)
        import httpx
        def forbid_network(request):raise AssertionError('Cached successful group must not submit another request')
        cached=APIGroupScorer(config,run,transport=httpx.MockTransport(forbid_network)).score(source_group)
        if cached!=result['scores']:raise ValueError('Imported cached scores changed')
        if any(sha256(f)!=h for f,h in old_assets.items()):raise AssertionError('Prior teacher evidence changed')
        if (run/'plan/cursor.json').read_bytes()!=ledger_bytes or (run/'plan/epoch_order.json').read_bytes()!=order_bytes:raise AssertionError('Provider migration reset the ledger')
        output={'time':now(),'completed':True,'version':version,'provider_changes':changes,'preserved_teacher_files':len(old_assets),'original_teacher_hashes_unchanged':True,'ledger_and_order_byte_identical':True,'original_condition_cursor':ledger['cursor'],'groups_with_provider_versions':groups,'imported_successful_group':source_group.name,'cached_scores':cached,'cached_repeat_network_requests':0,'active_database':str(active_cache),'historical_database':str(run/'requests/cache.sqlite3'),'successful_probe':str(probe),'real_API_requests_during_migration':0,'concurrency_backend':'threads','workers':32,'formal_training_started':False}
        write_json(report_path,output);write_json(run/'reports/provider_migration_latest.json',output)
        state=read_json(run/'plan/stage_state.json');state.update(stage='provider_migrated_ready',process_state='stopped',updated_at=now(),provider=config['reward']['endpoint'],provider_migration_report=str(report_path));write_json(run/'plan/stage_state.json',state)
        print({k:output[k] for k in ['completed','version','preserved_teacher_files','ledger_and_order_byte_identical','cached_repeat_network_requests','workers']})


if __name__=='__main__':main()
