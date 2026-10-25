"""Bounded software tests: HTTP faults, immutable assets, Git, and resume state."""
import contextlib
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

SPEC=importlib.util.spec_from_file_location('bootstrap',Path(__file__).resolve().parents[2]/'bootstrap_stateplay.py')
b=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(b)


@contextlib.contextmanager
def server(body, *, failures=0, truncate=False, ignore_range=False, bad_range=False):
    calls=[]
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,*args):pass
        def do_GET(self):
            n=len(calls);offset=int(self.headers.get('Range','bytes=0-').split('=')[1].split('-')[0])
            calls.append(dict(offset=offset,range=self.headers.get('Range'),authorization=self.headers.get('Authorization')))
            if n<failures:
                self.send_response(503);self.send_header('Content-Length','0');self.end_headers();return
            if ignore_range:offset=0
            part=body[offset:]
            self.send_response(206 if self.headers.get('Range') and not ignore_range else 200)
            self.send_header('Content-Length',str(len(part)))
            if self.headers.get('Range') and not ignore_range:
                self.send_header('Content-Range',f'bytes {offset+(1 if bad_range else 0)}-{len(body)-1}/{len(body)}')
            self.end_headers()
            if truncate and n==failures:
                self.wfile.write(part[:len(part)//3]);self.wfile.flush()
                self.connection.shutdown(socket.SHUT_RDWR);self.connection.close();return
            try:self.wfile.write(part)
            except (BrokenPipeError,ConnectionResetError):pass
    http=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=http.serve_forever,daemon=True);thread.start()
    try:yield f'http://127.0.0.1:{http.server_port}',calls
    finally:http.shutdown();http.server_close();thread.join()


class DownloadTests(unittest.TestCase):
    body=bytes(range(256))*8192
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.journal=b.Verified(self.root/'verified.sqlite3')
        self.item=dict(label='world',repo='test/model',kind='model',revision='a'*40,
            remote='world.pt',target='renamed project/world.pt',bytes=len(self.body),
            algorithm='sha256',checksum=hashlib.sha256(self.body).hexdigest())
    def tearDown(self):self.journal.close();self.temp.cleanup()
    def loader(self,url,attempts=4):return b.Downloader(self.root,b.Http([url],timeout=3,attempts=attempts,backoff=0),self.journal)
    def partial(self,body):
        p=self.root/self.item['target'];p.parent.mkdir(parents=True,exist_ok=True)
        p.with_name(p.name+'.incomplete').write_bytes(body)
    def test_resume_existing_partial_and_skip_without_network(self):
        self.partial(self.body[:199999])
        with server(self.body) as (url,calls):
            loader=self.loader(url)
            self.assertEqual(loader.fetch(self.item),'downloaded')
            self.assertEqual(calls[0]['offset'],199999)
            self.assertEqual((self.root/self.item['target']).read_bytes(),self.body)
            self.assertEqual(loader.fetch(self.item),'skipped');self.assertEqual(len(calls),1)
    def test_truncated_body_retries_from_received_bytes(self):
        with server(self.body,truncate=True) as (url,calls):
            self.loader(url).fetch(self.item)
            self.assertGreater(len(calls),1);self.assertGreater(calls[1]['offset'],0)
            self.assertEqual((self.root/self.item['target']).read_bytes(),self.body)
    def test_503_is_retried(self):
        with server(self.body,failures=2) as (url,calls):
            self.loader(url).fetch(self.item);self.assertEqual(len(calls),3)
    def test_range_ignored_restarts_instead_of_appending(self):
        self.partial(self.body[:12345])
        with server(self.body,ignore_range=True) as (url,calls):
            self.loader(url).fetch(self.item)
            self.assertEqual((self.root/self.item['target']).read_bytes(),self.body)
    def test_invalid_content_range_is_bounded_and_keeps_partial(self):
        self.partial(self.body[:12345])
        with server(self.body,bad_range=True) as (url,calls):
            with self.assertRaises(RuntimeError):self.loader(url,attempts=2).fetch(self.item)
            self.assertEqual(len(calls),2)
            p=self.root/self.item['target'];self.assertFalse(p.exists())
            self.assertEqual(p.with_name(p.name+'.incomplete').stat().st_size,12345)
    def test_existing_different_file_never_overwritten(self):
        p=self.root/self.item['target'];p.parent.mkdir(parents=True);p.write_bytes(b'original')
        with server(self.body) as (url,calls):
            with self.assertRaises(RuntimeError):self.loader(url).fetch(self.item)
            self.assertEqual(p.read_bytes(),b'original');self.assertFalse(calls)
    def test_corrupt_owned_partial_restarts_after_checksum_failure(self):
        self.partial(b'wrong prefix')
        with server(self.body) as (url,calls):
            self.loader(url).fetch(self.item)
            self.assertEqual(calls[0]['offset'],12);self.assertEqual(calls[1]['offset'],0)
            self.assertEqual((self.root/self.item['target']).read_bytes(),self.body)
    def test_corrupt_complete_file_is_preserved(self):
        p=self.root/self.item['target'];p.parent.mkdir(parents=True)
        p.write_bytes(b'0'*len(self.body))
        with server(self.body) as (url,calls):
            with self.assertRaises(RuntimeError):self.loader(url).fetch(self.item)
            self.assertFalse(calls);self.assertEqual(p.read_bytes(),b'0'*len(self.body))
    def test_git_blob_checksum(self):
        item=dict(self.item,algorithm='git-sha1',checksum=hashlib.sha1(f'blob {len(self.body)}\0'.encode()+self.body).hexdigest())
        with server(self.body) as (url,calls):self.loader(url).fetch(item)
    def test_token_not_sent_to_mirror_or_test_host(self):
        with server(self.body) as (url,calls):
            loader=b.Downloader(self.root,b.Http([url],attempts=1,token='not-a-real-token'),self.journal)
            loader.fetch(self.item);self.assertIsNone(calls[0]['authorization'])
    def test_tampered_stat_requires_rehash(self):
        with server(self.body) as (url,calls):
            loader=self.loader(url);loader.fetch(self.item)
            p=self.root/self.item['target'];p.write_bytes(b'x'*len(self.body))
            with self.assertRaises(RuntimeError):loader.fetch(self.item)
            self.assertEqual(len(calls),1)
    def test_interruption_keeps_partial(self):
        self.partial(self.body[:1234])
        with server(self.body) as (url,calls):
            loader=self.loader(url);loader.stop.set()
            with self.assertRaises(InterruptedError):loader.fetch(self.item)
            self.assertFalse(calls)


class PlanAndPathTests(unittest.TestCase):
    def test_only_minimal_recovery_selected(self):
        self.assertIsNone(b.destination('recovery','groups/g-0000000/candidates/video.mp4'))
        self.assertIsNone(b.destination('recovery','records/study_v1/provenance/secret.txt'))
        self.assertIsNotNone(b.destination('recovery','records/study_v1/plan/cursor.json'))
    def test_exact_dataset_mapping(self):
        self.assertEqual(b.destination('sf3','SF3/clips/clip_0000000/video.mp4'),'data/SF3/train/clips_5s/clips/clip_0000000/video.mp4')
        self.assertIsNone(b.destination('sf3','other_game/clips/video.mp4'))
        self.assertIsNone(b.destination('umt5','diffusion_pytorch_model.safetensors'))
        self.assertIsNone(b.destination('wan','google/umt5-xxl/tokenizer.json'))
    def test_paths_cannot_escape(self):
        for value in ('../outside','/etc/passwd','a/../../b','a\\b'):
            with self.assertRaises(ValueError):b.safe_relative(value)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'project';root.mkdir();(root/'escape').symlink_to(Path(tmp))
            with self.assertRaises(ValueError):b.under(root,'escape/file')
    def test_root_relocation_only_matches_complete_prefix(self):
        self.assertEqual(b.relocate(b.OLD_ROOT+'/outputs/file',Path('/new root')),'/new root/outputs/file')
        self.assertEqual(b.relocate(b.OLD_ROOT+'-other/file',Path('/new root')),b.OLD_ROOT+'-other/file')
    def test_python_path_does_not_resolve_venv_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'venv/bin/python';p.parent.mkdir(parents=True);p.symlink_to('/usr/bin/python3')
            self.assertEqual(str(b.absolute(p)),str(p))
    def test_failed_gate_stays_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'quality_gate.json';b.immutable_json(p,{'passed':False})
            with self.assertRaises(RuntimeError):b.immutable_json(p,{'passed':True})
            self.assertFalse(b.read_json(p)['passed'])
    def test_resume_identity_does_not_reset_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'cursor.json';b.write_json(p,{'cursor':2308,'local_consumed':28})
            before=p.read_bytes()
            with self.assertRaises(RuntimeError):b.immutable_json(p,{'cursor':2280,'local_consumed':0})
            self.assertEqual(p.read_bytes(),before)
    def test_plan_tampering_rejected(self):
        core=dict(schema='stateplay-bootstrap-download-v1',github_commit=b.GIT_COMMIT,repositories=[],files=[])
        plan=dict(core,plan_hash=b.digest(core),total_bytes=0);b.validate_plan(plan)
        plan['files'].append({'target':'bad'})
        with self.assertRaises(ValueError):b.validate_plan(plan)
    def test_positive_gpu_ids_unique(self):
        self.assertEqual(b.parse_gpus('4,5,6,7'),[4,5,6,7])
        for x in ('1,1','-1,2','x'):
            with self.assertRaises(Exception):b.parse_gpus(x)
    def test_external_holdout_archives_relocated(self):
        config={'data':{'exclusion_archives':['/private/StatePlay_test_100_train_ratio_v2_wins.zip']},
                'reward_model':{'python':'old'},'reward':{'endpoint':'https://service.example.invalid/v1'}}
        result=b.deployment_config(config,Path('/new root'),'/my env/bin/python')
        self.assertEqual(result['data']['exclusion_archives'],['/new root/data/test_archives/StatePlay_test_100_train_ratio_v2_wins.zip'])
        self.assertEqual(result['reward_model']['python'],'/my env/bin/python')
        self.assertEqual(config['reward_model']['python'],'old')
    def test_internal_orchestrator_python_entries_use_configured_env(self):
        import sys
        with tempfile.TemporaryDirectory(prefix='python entries ') as tmp:
            root=Path(tmp);request={'world_python':sys.executable,'reward_python':sys.executable}
            b.install_python_entries(root,request)
            for name in ('world','reward'):
                entry=root/'outputs/rl_posttrain/_envs'/name/'bin/python'
                result=subprocess.check_output([str(entry),'-c','import sys;print(sys.prefix)'],text=True).strip()
                self.assertEqual(result,sys.prefix)
            b.install_python_entries(root,request)
    def test_cross_host_redirect_strips_token(self):
        import urllib.request
        request=urllib.request.Request('https://huggingface.co/model',headers={'Authorization':'Bearer not-a-real-token'})
        redirected=b.SafeRedirect().redirect_request(request,None,302,'Found',{},'https://cdn.example.invalid/model')
        self.assertIsNone(redirected.get_header('Authorization'))


class TorchMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:import torch,numpy
        except ImportError:raise unittest.SkipTest('torch/numpy not available in this interpreter')
    def test_checkpoint_roundtrip_preserves_all_numeric_state(self):
        import torch,numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source.pt';target=Path(tmp)/'new/model.pt'
            state=dict(live={'w':torch.randn(3,4)},policies={'old':torch.randn(3,4),'ref':torch.zeros(3,4)},
                optimizer={'state':{0:{'exp_avg':torch.randn(3,4),'step':torch.tensor(652)}}},
                rank_rng=[{'torch':torch.get_rng_state(),'numpy':np.random.get_state()}],
                state={'optimizer_step':652},config={'root':b.OLD_ROOT,'config_hash':'old'})
            torch.save(state,source);before=b.file_hash(source)
            config={'root':'/renamed project','config_hash':'new'}
            original,loaded=b.save_checkpoint(source,target,config)
            for key in state:
                if key!='config':self.assertTrue(b.exact(original[key],loaded[key]))
            self.assertEqual(before,b.file_hash(source));self.assertEqual(loaded['config'],config)
            b.save_checkpoint(source,target,config)
            with self.assertRaises(RuntimeError):b.save_checkpoint(source,target,{'root':'/wrong'})


if __name__=='__main__':unittest.main(verbosity=2)


class CLauncherCompatibilityTests(unittest.TestCase):
    def test_default_and_explicit_module_forwarding(self):
        for mode in ('B-only','Coupling-only','Both'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);state=root/b.STATE;run=root/b.RUN
                request=dict(world_python='/env/world/python',reward_python='/env/reward/python',
                    gpus=[0,1,2,3],generation_batch_size=1,reward_batch_size=1)
                b.write_json(state/'PREPARED.json',dict(request_identity=request,config=str(run/'plan/C.yaml')))
                b.write_json(run/'plan/stage_C_control.json',dict(allow_start=True))
                argv=['--project-root',str(root),'--start-c','--gpus','4,5,6,7']
                if mode!='B-only':argv+=['--update-mode',mode]
                args=b.parser().parse_args(argv)
                with patch.object(b,'check_workspace'),patch.object(b.subprocess,'call',return_value=0) as call:
                    self.assertEqual(b.start_c(root,args,[]),0)
                command=call.call_args.args[0]
                self.assertEqual(command[command.index('--update-mode')+1],mode)
                self.assertEqual(command[command.index('--gpus')+1],'4,5,6,7')

    def test_cached_bootstrap_is_backed_up_and_unknown_edits_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cached=root/b.STATE/'bootstrap_stateplay.py'
            old=b'known legacy bootstrap\n';b.write_bytes(cached,old)
            request=dict(world_python='/env/world/python',reward_python='/env/reward/python',gpus=[0,1,2,3])
            known=hashlib.sha256(old).hexdigest()
            with patch.object(b,'LEGACY_BOOTSTRAP_SHA256',{known}):b.launch_files(root,request)
            self.assertEqual((root/b.STATE/'history'/('bootstrap_'+known+'.py')).read_bytes(),old)
            self.assertEqual(cached.read_bytes(),Path(b.__file__).read_bytes())
            cached.write_text('unknown local changes')
            with self.assertRaisesRegex(RuntimeError,'unknown local edits'):b.launch_files(root,request)
            self.assertEqual(cached.read_text(),'unknown local changes')

    def test_clean_git_upgrade_preserves_untracked_training_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);origin=folder/'origin';origin.mkdir()
            def git_at(root,*args):
                return subprocess.run(['git','-C',str(root),*args],check=True,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout.strip()
            git_at(origin,'init','-q');(origin/'rl_posttrain').mkdir()
            file=origin/'rl_posttrain/version.txt';file.write_text('old')
            git_at(origin,'add','.');git_at(origin,'-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','old')
            previous=git_at(origin,'rev-parse','HEAD')
            checkout=folder/'checkout';subprocess.run(['git','clone','-q',str(origin),str(checkout)],check=True)
            ledger=checkout/'outputs/cursor.json';ledger.parent.mkdir();ledger.write_text('{"cursor":3708}')
            file.write_text('new');git_at(origin,'add','.');git_at(origin,'-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','new')
            current=git_at(origin,'rev-parse','HEAD')
            with patch.object(b,'GIT_URL',str(origin)),patch.object(b,'GIT_COMMIT',current),patch.object(b,'COMPATIBLE_GIT_COMMITS',{previous,current}):
                b.clone(checkout,2,0)
                self.assertEqual(git_at(checkout,'rev-parse','HEAD'),current)
                self.assertEqual(ledger.read_text(),'{"cursor":3708}')
                (checkout/'rl_posttrain/version.txt').write_text('local edit')
                with self.assertRaisesRegex(RuntimeError,'Tracked checkout files changed'):b.clone(checkout,2,0)
                self.assertEqual((checkout/'rl_posttrain/version.txt').read_text(),'local edit')
