#!/usr/bin/env python3
"""Clone StatePlay and prepare pinned HF assets for implicit SFT and Stage C.

The downloader uses only Python's standard library and git. Preparation invokes
the user's WORLD Python for trusted torch checkpoints. No training is started.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import copy
import csv
import datetime as dt
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

GIT_URL = 'https://github.com/lupele/stateplay-test.git'
GIT_COMMIT = '9da20eb500907a8a17fde5ca23189ab25a693f4e'
COMPATIBLE_GIT_COMMITS = {'f0f3a13732ca5172fb7e6ffa046a769b037b6804', GIT_COMMIT}
LEGACY_BOOTSTRAP_SHA256 = {'1c7fcc59f46262641ad7f7828bf45c5aea700098e2c41dffa70ad8e9653d8966', '54d8b3988916a3939d1ef879622666dcb2df09c37e38eeccc1ff97e4b2dcabb9', '67e0f08fdd3e1ff230e622a9d39fd538b491e0ce35d24011eb003c29642f3e46'}
STATE = '.stateplay_bootstrap'
RUN = 'outputs/rl_posttrain/study_v1'
OLD_ROOT = '/private/StatePlay-main'
MODEL_VERSION = 'conditions-00002280-step-00000652'
RM_VERSION = '28befdbd2782031fe8d67975'
SFT_SHA = '53eff9e5e59e92bb3cc7810e3ff36f9ee490be4a122a3a8ff96a4de0a03b4c5d'
TOK_SHA = '3714f9338f4166b343690ce68f90fa6a273e3ab3775d58b390cc91c0c467b85c'
WORLD_SHA = '56328f45f3ee19237b63add2eb666dfcaa8aaebf505d29aa400a78519aea28ac'
ADAPTER_SHA = '81078f22d9cbdddde35f1c818d53bdf079316d88f30ca7ad63dc99e0c0e605e6'

# All artifacts are public HF resources. Revisions are intentionally immutable.
SPECS = (
    ('tokenizer', 'lupele/transition_tokenizer.pt', 'model', '672ad0d748b3574b66e3a635b40a8087034448f3'),
    ('world', 'lupele/rl_model', 'model', '20f6cfd00584712a8b8d433d6bc3ec3ddb259bcb'),
    ('reward', 'lupele/rewardmodel', 'model', '6a0c0875b3e3807604336a116b2dc8a76c284588'),
    ('sft', 'lupele/ourmodel', 'model', '8e905718a5498aba3dd1071ab0eeedf559904e39'),
    ('wan', 'Wan-AI/Wan2.2-TI2V-5B', 'model', '921dbaf3f1674a56f47e83fb80a34bac8a8f203e'),
    ('umt5', 'Wan-AI/Wan2.1-T2V-1.3B', 'model', '37ec512624d61f7aa208f7ea8140a131f93afc9a'),
    ('qwen', 'Qwen/Qwen3.5-9B', 'model', 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'),
    ('sf3', 'onepiece1999/StatePlay-Dataset', 'dataset', 'b258f9709865ec4bb7af917aa0a800cd8989428a'),
    ('test100', 'lupele/test100', 'dataset', 'd2df174ebfef78d96777ed1f7819a00ba3d5d9db'),
    ('test200', 'lupele/test200', 'dataset', '7ade1245f0c081c08d8648719a37a95531a4d326'),
    ('recovery', 'lupele/StatePlay-Reward-2280', 'dataset', '172f4cf3f0ea0c6d67a311b12452e45137a4ef09'),
)
RECOVERY_FILES = (
    'metadata/source_audit/catalog.json',
    'metadata/source_audit/leakage_index.json',
    'metadata/source_audit/leakage_features.json',
    'records/study_v1/plan/config.json',
    'records/study_v1/plan/cursor.json',
    'records/study_v1/plan/epoch_order.json',
    'records/study_v1/plan/stage_state.json',
    'records/study_v1/plan/stage_C_control.json',
    'records/study_v1/reports/integration_gate.json',
    'records/study_v1/reports/source_identity_before.json',
    'records/study_v1/reports/source_identity_after.json',
    'records/study_v1/reward_model/versions/a-002280/reports/quality_gate.json',
    'records/study_v1/reward_model/versions/a-002280/reports/selection.json',
    'records/study_v1/reward_model/versions/a-002280/reports/test.json',
    'records/study_v1/reward_model/versions/a-002280/reports/validation.json',
    'records/study_v1/reward_model/versions/a-002280/reports/shortcut_checks.json',
    'records/study_v1/reward_model/versions/a-002280/splits/manifest.json',
)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def absolute(value):
    # Do not resolve a venv's bin/python symlink: that discards the environment.
    return Path(os.path.abspath(os.path.expanduser(str(value))))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def file_hash(path, algorithm='sha256'):
    h = hashlib.new('sha1' if algorithm == 'git-sha1' else algorithm)
    if algorithm == 'git-sha1':
        h.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    write_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode())


def write_bytes(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name+'.pending')
    if pending.is_symlink():
        raise RuntimeError('Unexpected pending symlink: '+str(pending))
    with pending.open('wb') as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def read_json(path):
    return json.loads(path.read_text())


def safe_relative(value):
    p = PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or not p.parts or '\\' in value:
        raise ValueError('Unsafe relative path')
    return p.as_posix()


def under(root, relative):
    p = root / safe_relative(relative)
    # Follow parents to reject a download redirected outside the chosen project.
    if root.resolve() not in p.parent.resolve().parents and p.parent.resolve() != root.resolve():
        raise ValueError('Target parent escapes project root: '+relative)
    return p


@contextlib.contextmanager
def process_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another bootstrap process is using this project') from None
        yield


def hf_token():
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if token:
        return token.strip()
    # Only the standard HF login cache; never inspect shell/Codex history.
    location = Path(os.environ.get('HF_TOKEN_PATH', str(
        Path(os.environ.get('HF_HOME', str(Path.home()/'.cache/huggingface'))) / 'token')))
    return location.read_text().strip() if location.is_file() else None


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected and urllib.parse.urlsplit(req.full_url).netloc != urllib.parse.urlsplit(newurl).netloc:
            redirected.remove_header('Authorization')
        return redirected


class Http:
    def __init__(self, endpoints, timeout=120, attempts=10, backoff=3, token=None):
        self.endpoints = list(dict.fromkeys(e.rstrip('/') for e in endpoints))
        for endpoint in self.endpoints:
            u = urllib.parse.urlsplit(endpoint)
            if u.username or u.password or u.query or u.fragment or not u.netloc:
                raise ValueError('Endpoint must be an origin URL without credentials')
            if u.scheme != 'https' and u.hostname not in ('127.0.0.1', 'localhost'):
                raise ValueError('Use HTTPS for remote endpoints')
        self.timeout, self.attempts, self.backoff, self.token = timeout, attempts, backoff, token
        self.local = threading.local()

    def open(self, url, headers=None):
        headers = dict(headers or {}, **{'Accept-Encoding': 'identity', 'User-Agent': 'StatePlay-bootstrap/1'})
        if self.token and urllib.parse.urlsplit(url).hostname == 'huggingface.co':
            headers['Authorization'] = 'Bearer '+self.token
        if not hasattr(self.local, 'opener'):
            self.local.opener = urllib.request.build_opener(SafeRedirect())
        return self.local.opener.open(urllib.request.Request(url, headers=headers), timeout=self.timeout)

    def pause(self, attempt, error):
        delay = min(30, self.backoff * (2 ** min(attempt, 5)))
        if isinstance(error, urllib.error.HTTPError):
            header = error.headers.get('Retry-After', '')
            if header.isdigit():delay = max(delay, min(120, int(header)))
        if delay:time.sleep(delay + random.uniform(0, min(delay/4, 1)))

    def metadata(self, repo, kind, revision):
        plural = 'datasets' if kind == 'dataset' else 'models'
        suffix = f'/api/{plural}/{repo}/revision/{revision}?blobs=true'
        for attempt in range(self.attempts):
            try:
                endpoint = self.endpoints[attempt % len(self.endpoints)]
                with self.open(endpoint+suffix) as response:
                    value = json.load(response)
                if value.get('sha') != revision:
                    raise ValueError('Repository revision mismatch')
                return value
            except (OSError, ValueError, http.client.HTTPException) as error:
                if attempt+1 == self.attempts:raise RuntimeError('Cannot fetch pinned metadata: '+repo) from None
                self.pause(attempt, error)


def destination(label, remote):
    """Whitelist exact assets. In particular never select the 44 GB reward videos."""
    remote = safe_relative(remote)
    if remote == '.gitattributes':return None
    if label == 'tokenizer':
        return 'outputs/tokenizer/'+remote if remote == 'transition_tokenizer.pt' else STATE+'/sources/tokenizer/'+remote
    if label == 'sft':
        return 'outputs/latent_world/'+remote if remote == 'step-4572.safetensors' else STATE+'/sources/sft/'+remote
    if label in ('world', 'reward'):
        return STATE+'/sources/'+label+'/'+remote
    if label == 'wan':
        wanted = {'Wan2.2_VAE.pth', 'models_t5_umt5-xxl-enc-bf16.pth', 'config.json', 'configuration.json',
                  'diffusion_pytorch_model.safetensors.index.json',
                  *(f'diffusion_pytorch_model-{i:05d}-of-00003.safetensors' for i in (1,2,3))}
        return 'base_model/Wan-AI/Wan2.2-TI2V-5B/'+remote if remote in wanted else None
    if label == 'umt5':
        return 'base_model/Wan-AI/Wan2.1-T2V-1.3B/'+remote if remote.startswith('google/umt5-xxl/') else None
    if label == 'qwen':
        return 'base_model/Qwen/Qwen3.5-9B/'+remote
    if label == 'sf3':
        if remote == 'SF3/metadata_state_polish.csv' or re.fullmatch(r'SF3/clips/clip_\d+/(video\.mp4|actions\.parquet)', remote):
            return 'data/SF3/train/clips_5s/'+remote[len('SF3/'):]
        return None
    if label in ('test100', 'test200'):
        expected = 'StatePlay_test_100_train_ratio_v2_wins.zip' if label == 'test100' else 'StatePlay_test_200_train_ratio.zip'
        return 'data/test_archives/'+remote if remote == expected else None
    if label == 'recovery':
        return STATE+'/sources/recovery/'+remote if remote in RECOVERY_FILES else None
    raise ValueError('Unknown resource group')


def make_plan(http):
    files, repositories = [], []
    for position,(label, repo, kind, revision) in enumerate(SPECS,1):
        print(f'读取固定版本清单 [{position}/{len(SPECS)}] {repo}',flush=True)
        info = http.metadata(repo, kind, revision)
        chosen = []
        for item in info['siblings']:
            target = destination(label, item['rfilename'])
            if target is None:continue
            lfs = item.get('lfs')
            algorithm, checksum = ('sha256', lfs['sha256']) if lfs else ('git-sha1', item.get('blobId'))
            if not checksum or item.get('size') is None:raise ValueError('Missing remote checksum/size')
            entry = dict(label=label, repo=repo, kind=kind, revision=revision, remote=item['rfilename'],
                         target=safe_relative(target), bytes=item['size'], algorithm=algorithm, checksum=checksum)
            files.append(entry);chosen.append(entry)
        if not chosen:raise ValueError('Empty resource selection: '+repo)
        if label == 'recovery' and {x['remote'] for x in chosen} != set(RECOVERY_FILES):
            raise ValueError('Missing required recovery metadata')
        if label == 'sf3' and len(chosen) != 20001:raise ValueError('Expected 10000 SF3 videos/actions and CSV')
        repositories.append(dict(label=label, repo=repo, revision=revision, files=len(chosen), bytes=sum(x['bytes'] for x in chosen)))
    if len({x['target'] for x in files}) != len(files):raise ValueError('Conflicting download targets')
    files.sort(key=lambda x:(x['bytes'], x['repo'], x['remote']))
    core = dict(schema='stateplay-bootstrap-download-v1', github_commit=GIT_COMMIT, repositories=repositories, files=files)
    return dict(core, plan_hash=digest(core), total_bytes=sum(x['bytes'] for x in files), reward_training_videos_included=False)


def validate_plan(plan):
    core = {key:plan[key] for key in ('schema', 'github_commit', 'repositories', 'files')}
    if plan.get('plan_hash') != digest(core):raise ValueError('Download plan checksum changed')
    if plan['github_commit'] not in COMPATIBLE_GIT_COMMITS:raise ValueError('Unsupported checkout revision')
    identities={label:(repo,kind,revision) for label,repo,kind,revision in SPECS}
    for item in plan['files']:
        if destination(item['label'], item['remote']) != item['target']:raise ValueError('Unapproved asset in plan')
        if identities[item['label']] != (item['repo'],item['kind'],item['revision']):raise ValueError('Pinned asset identity changed')
    if plan['total_bytes']!=sum(x['bytes'] for x in plan['files']):raise ValueError('Plan byte count changed')
    if len({x['target'] for x in plan['files']})!=len(plan['files']):raise ValueError('Duplicate plan target')
    return plan


class Verified:
    """Small restartable stat/checksum journal; files remain independently verifiable."""
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute('PRAGMA synchronous=NORMAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS verified (path TEXT PRIMARY KEY, identity TEXT, size INTEGER, mtime INTEGER)')
        self.lock = threading.Lock()
        self.last_commit = time.monotonic();self.pending = 0

    def matches(self, item, path):
        if not path.is_file():return False
        st = path.stat()
        if st.st_size != item['bytes']:return False
        identity = item['algorithm']+':'+item['checksum']
        with self.lock:
            row = self.db.execute('SELECT identity,size,mtime FROM verified WHERE path=?', (item['target'],)).fetchone()
        return row == (identity, st.st_size, st.st_mtime_ns)

    def remember(self, item, path):
        st = path.stat()
        with self.lock:
            self.db.execute('INSERT OR REPLACE INTO verified VALUES(?,?,?,?)',
                            (item['target'], item['algorithm']+':'+item['checksum'], st.st_size, st.st_mtime_ns))
            self.pending += 1
            if self.pending >= 32 or time.monotonic()-self.last_commit >= 1:
                self.db.commit();self.pending=0;self.last_commit=time.monotonic()

    def close(self):
        with self.lock:self.db.commit();self.db.close()


class Downloader:
    def __init__(self, root, http, journal, stop=None):
        self.root, self.http, self.journal = root, http, journal
        self.stop = stop or threading.Event()
        self.lock = threading.Lock()
        self.done_files = self.done_bytes = self.received = self.skipped = 0

    def note_done(self, item, skipped):
        with self.lock:
            self.done_files += 1;self.done_bytes += item['bytes'];self.skipped += int(skipped)

    def fetch(self, item):
        path = under(self.root, item['target'])
        if path.is_file():
            if self.journal.matches(item, path) or (path.stat().st_size == item['bytes'] and file_hash(path,item['algorithm']) == item['checksum']):
                self.journal.remember(item,path);self.note_done(item,True);return 'skipped'
            # Never replace/move an existing model or source dataset file.
            raise RuntimeError('Existing file differs; preserved without overwrite: '+item['target'])
        if path.is_symlink():raise RuntimeError('Dangling target symlink: '+item['target'])
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name+'.incomplete')
        if partial.is_symlink():raise RuntimeError('Unexpected partial symlink')
        prefix = '/datasets/' if item['kind']=='dataset' else '/'
        suffix = prefix+item['repo']+'/resolve/'+item['revision']+'/'+urllib.parse.quote(item['remote'],safe='/')+'?download=true'
        for attempt in range(self.http.attempts):
            if self.stop.is_set():raise InterruptedError('Download interrupted')
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                if offset > item['bytes']:
                    raise ValueError('Partial file is larger than pinned asset')
                if offset < item['bytes']:
                    endpoint = self.http.endpoints[attempt % len(self.http.endpoints)]
                    # A fresh resolve request on each retry refreshes expiring CDN URLs.
                    headers = {'Range':f'bytes={offset}-'} if offset else {}
                    url=endpoint+suffix+(f'&stateplay_retry={time.time_ns()}' if attempt else '')
                    with self.http.open(url, headers) as response:
                        status = response.getcode()
                        if response.headers.get('Content-Encoding','identity') != 'identity':
                            raise ValueError('Unexpected compressed response')
                        if status == 206:
                            cr = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range',''))
                            if not cr or int(cr[1]) != offset or int(cr[3]) != item['bytes']:
                                raise ValueError('Invalid resume Content-Range')
                        elif status == 200:
                            # Server ignored Range: restart this owned partial, never append a full body.
                            offset = 0
                        else:raise ValueError('Unexpected download response status')
                        with partial.open('ab' if offset else 'wb') as stream:
                            while not self.stop.is_set():
                                # read1 retains bytes received before a truncated HTTP response.
                                block = response.read1(1 << 20)
                                if not block:break
                                if stream.tell()+len(block) > item['bytes']:raise ValueError('Response exceeds pinned size')
                                stream.write(block)
                                with self.lock:self.received += len(block)
                            stream.flush();os.fsync(stream.fileno())
                    if self.stop.is_set():raise InterruptedError('Download interrupted')
                if not partial.exists() and item['bytes'] == 0:partial.touch()
                if partial.stat().st_size != item['bytes']:raise OSError('Incomplete response; partial kept')
                if file_hash(partial,item['algorithm']) != item['checksum']:
                    # Only the downloader-owned incomplete file may be reset.
                    partial.unlink()
                    raise ValueError('Checksum mismatch; incomplete file reset')
                if path.exists():raise RuntimeError('Target appeared during download; preserved')
                os.replace(partial,path)
                self.journal.remember(item,path);self.note_done(item,False)
                return 'downloaded'
            except (OSError, ValueError, http.client.HTTPException) as error:
                if self.stop.is_set():raise InterruptedError('Download interrupted') from None
                if attempt+1 == self.http.attempts:
                    raise RuntimeError(f"Download exhausted {self.http.attempts} attempts: {item['target']} ({type(error).__name__})") from None
                with self.lock:
                    print(f"重试 {attempt+1}/{self.http.attempts}: {item['target']} ({type(error).__name__})",flush=True)
                self.http.pause(attempt,error)


def git(root, *args, check=True):
    env=os.environ.copy();env['GIT_TERMINAL_PROMPT']='0'
    return subprocess.run(['git','-c','http.lowSpeedLimit=1024','-c','http.lowSpeedTime=60','-C',str(root),*args],
        env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=check,timeout=300)


def clone(root, attempts, backoff):
    if root.exists() and not (root/'.git').is_dir() and any(root.iterdir()):
        raise RuntimeError('Nonempty target is not a Git checkout; choose a new project directory')
    root.mkdir(parents=True,exist_ok=True)
    if not (root/'.git').exists():git(root,'init','-q')
    with process_lock(root/'.git/stateplay-bootstrap.lock'):
        finish_clone(root,attempts,backoff)


def finish_clone(root, attempts, backoff):
    remote=git(root,'remote','get-url','origin',check=False)
    if remote.returncode:git(root,'remote','add','origin',GIT_URL)
    elif remote.stdout.strip().removesuffix('.git') != GIT_URL.removesuffix('.git'):
        raise RuntimeError('Existing origin is a different repository; preserved')
    head=git(root,'rev-parse','--verify','HEAD',check=False)
    if head.returncode == 0:
        if git(root,'diff','HEAD','--quiet',check=False).returncode:
            raise RuntimeError('Tracked checkout files changed; preserved. Commit/copy your edits before upgrading.')
        current=head.stdout.strip()
        if current==GIT_COMMIT:return
        # A shallow HEAD may omit its pinned code parent. Fetch the object
        # without moving HEAD, then compare the complete source tree except
        # this delivery script. A bootstrap-only release is compatible even
        # when shallow history prevents merge-base from seeing ancestry.
        present=git(root,'cat-file','-e',GIT_COMMIT+'^{commit}',check=False)
        if present.returncode:
            for attempt in range(attempts):
                fetched=git(root,'fetch','--depth=1','origin',GIT_COMMIT,check=False)
                if fetched.returncode==0:break
                if attempt+1==attempts:raise RuntimeError('Pinned code lookup failed after bounded retries')
                time.sleep(min(30,backoff*2**min(attempt,5)))
        same=git(root,'diff',GIT_COMMIT,'HEAD','--quiet','--','.',':(exclude)bootstrap_stateplay.py',check=False)
        if same.returncode==0:return
        if current not in COMPATIBLE_GIT_COMMITS:
            raise RuntimeError('Existing checkout is not a recognized compatible release; preserved')
    # Interrupted first fetch/check-out can safely continue without a hard reset.
    for attempt in range(attempts):
        try:
            fetched=git(root,'fetch','--depth='+('64' if head.returncode==0 else '1'),'origin',GIT_COMMIT,check=False)
            if fetched.returncode == 0:break
        except subprocess.TimeoutExpired:pass
        if attempt+1==attempts:raise RuntimeError('Git fetch failed after bounded retries')
        time.sleep(min(30,backoff*(2**min(attempt,5))))
    if head.returncode==0:git(root,'merge','--ff-only',GIT_COMMIT)
    else:git(root,'checkout','--detach',GIT_COMMIT)


def download_all(root, plan, http, workers):
    journal=Verified(root/STATE/'verified.sqlite3');stop=threading.Event()
    loader=Downloader(root,http,journal,stop)
    total=len(plan['files']);started=time.monotonic();finished=threading.Event()
    def progress():
        while not finished.wait(10):
            with loader.lock:
                print(f"下载/校验 [{loader.done_files}/{total}] {loader.done_bytes/1e9:.2f}/{plan['total_bytes']/1e9:.2f} GB；"
                      f"跳过 {loader.skipped} 文件；本次接收 {loader.received/1e9:.2f} GB；{(time.monotonic()-started)/60:.1f} 分钟",flush=True)
    thread=threading.Thread(target=progress,daemon=True);thread.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(loader.fetch,item) for item in plan['files']]
            try:
                for future in concurrent.futures.as_completed(futures):future.result()
            except BaseException:
                stop.set()
                for future in futures:future.cancel()
                raise
    finally:finished.set();thread.join();journal.close()
    result=dict(completed_at=now(),files=loader.done_files,verified_bytes=loader.done_bytes,
                skipped=loader.skipped,received_bytes=loader.received,plan_hash=plan['plan_hash'])
    write_json(root/STATE/'downloads_complete.json',result)
    print(f"全部 {total} 文件下载/校验完成，共 {plan['total_bytes']/1e9:.2f} GB。",flush=True)


def relocate(value, root):
    if isinstance(value,str):
        return str(root)+value[len(OLD_ROOT):] if value==OLD_ROOT or value.startswith(OLD_ROOT+'/') else value
    if isinstance(value,dict):return {k:relocate(v,root) for k,v in value.items()}
    if isinstance(value,list):return [relocate(v,root) for v in value]
    if isinstance(value,tuple):return tuple(relocate(v,root) for v in value)
    return value


def deployment_config(source,root,reward_python,endpoint=None):
    config=relocate(copy.deepcopy(source),root)
    # The original run's holdout ZIPs lived beside, rather than inside, its project.
    archives=[]
    for path in config['data'].get('exclusion_archives',[]):
        name=Path(path).name
        if name not in ('StatePlay_test_100_train_ratio_v2_wins.zip','StatePlay_test_200_train_ratio.zip'):
            raise RuntimeError('Unexpected exclusion archive; refusing to guess its destination')
        archives.append(str(root/'data/test_archives'/name))
    config['data']['exclusion_archives']=archives
    config['reward_model']['python']=reward_python
    if endpoint:config['reward']['endpoint']=endpoint.rstrip('/')
    config['config_hash']=digest({k:v for k,v in config.items() if k!='config_hash'})
    return config


def exact(a,b):
    import torch
    import numpy as np
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a) is type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b


def immutable_json(path,value):
    if path.exists():
        if read_json(path)!=value:raise RuntimeError('Existing prepared artifact differs; preserved: '+str(path))
    else:write_json(path,value)


def install_file(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if file_hash(source)!=file_hash(target):raise RuntimeError('Existing artifact differs: '+str(target))
        return
    if target.is_symlink():raise RuntimeError('Unexpected target symlink')
    try:os.link(source,target)
    except OSError:shutil.copyfile(source,target)


def install_python_entries(root,request):
    # The published orchestrator internally uses these fixed paths, regardless
    # of its outer shell's environment variables. Adapt with new shims only.
    for name in ('world','reward'):
        target=root/'outputs/rl_posttrain/_envs'/name/'bin/python'
        actual=absolute(request[name+'_python'])
        if target.absolute()==actual:
            if not os.access(target,os.X_OK):raise RuntimeError('Configured Python is missing')
            continue
        body='#!/usr/bin/env bash\nexec '+shlex.quote(str(actual))+' "$@"\n'
        if target.exists():
            try:matches=target.read_bytes()==body.encode()
            except OSError:matches=False
            if not matches:raise RuntimeError('Existing internal Python entry differs; preserved: '+str(target))
        else:
            if target.is_symlink():raise RuntimeError('Dangling internal Python entry')
            write_bytes(target,body.encode());target.chmod(0o755)


def save_checkpoint(source,target,config):
    import torch
    original=torch.load(source,map_location='cpu',weights_only=False)
    payload=dict(original,config=config)
    if not target.exists():
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_name(target.name+'.pending')
        torch.save(payload,temporary);os.replace(temporary,target)
    loaded=torch.load(target,map_location='cpu',weights_only=False)
    if not exact(payload,loaded):raise RuntimeError('Migrated checkpoint changed tensors/state or configuration')
    return original,loaded


def verify_local_assets(root):
    """Check indexes/CSV without allocating any CUDA/model weights."""
    for relative,expected in (('outputs/latent_world/step-4572.safetensors',SFT_SHA),
                              ('outputs/tokenizer/transition_tokenizer.pt',TOK_SHA)):
        if file_hash(root/relative)!=expected:raise RuntimeError('Protected checkpoint mismatch: '+relative)
    data=root/'data/SF3/train/clips_5s'
    csv_path=data/'metadata_state_polish.csv'
    if file_hash(csv_path)!='a31e932122985f4fc127f1814bf65742de586adafac15b6e2b0707285e19f15b':raise RuntimeError('SF3 CSV identity mismatch')
    with csv_path.open(newline='',encoding='utf-8-sig') as stream:rows=list(csv.DictReader(stream))
    if len(rows)!=10000:raise RuntimeError('Expected 10000 SF3 training conditions')
    for row in rows:
        for key in ('video','action'):
            if not under(data,row[key]).is_file():raise FileNotFoundError(row[key])
    qwen=root/'base_model/Qwen/Qwen3.5-9B'
    config=read_json(qwen/'config.json');index=read_json(qwen/'model.safetensors.index.json')
    if config.get('model_type')!='qwen3_5' or not config.get('vision_config') or not config.get('text_config'):
        raise RuntimeError('Expected complete Qwen3.5 multimodal configuration')
    weights=index['weight_map']
    if not any(k.startswith('model.visual.') for k in weights) or not any(k.startswith('model.language_model.') for k in weights):
        raise RuntimeError('Qwen vision/language branches missing')
    for shard in set(weights.values()):
        if not under(qwen,shard).is_file():raise FileNotFoundError(shard)
    if len(set(weights.values()))!=4:raise RuntimeError('Wrong Qwen checkpoint sharding')
    for name in ('preprocessor_config.json','video_preprocessor_config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja'):
        if not (qwen/name).is_file():raise FileNotFoundError(qwen/name)
    return dict(training_conditions=len(rows),qwen_full_vision_language_index=True,qwen_weight_shards=4)


def probe_reward_python(root,python):
    probe='''import json,sys
import transformers
from transformers import AutoConfig,AutoProcessor,AutoTokenizer
base=sys.argv[1]
c=AutoConfig.from_pretrained(base,local_files_only=True)
assert c.model_type=='qwen3_5' and c.vision_config is not None and c.text_config is not None
p=AutoProcessor.from_pretrained(base,local_files_only=True)
t=AutoTokenizer.from_pretrained(base,local_files_only=True)
assert getattr(transformers,'Qwen3_5ForConditionalGeneration',None) is not None
print(json.dumps(dict(local_files_only=True,full_multimodal_config=True,model_class='Qwen3_5ForConditionalGeneration',processor=type(p).__name__,tokenizer=type(t).__name__)))
'''
    env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
    result=subprocess.run([python,'-B','-c',probe,str(root/'base_model/Qwen/Qwen3.5-9B')],
        cwd=root,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if result.returncode:
        # Keep diagnostic output local; never include environment values in it.
        write_bytes(root/STATE/'reward_processor_probe.log',(result.stdout+result.stderr).encode())
        raise RuntimeError('REWARD Python 无法离线加载本地 Qwen processor/config；配置环境后 --prepare-only 重试，见 reward_processor_probe.log')
    data=json.loads(result.stdout.strip().splitlines()[-1])
    write_json(root/STATE/'reward_processor_probe.json',data)
    return data


def prepare_worker(request):
    import torch
    import yaml
    root=absolute(request['root']);state=root/STATE;run=root/RUN
    sources=state/'sources';recovery=sources/'recovery'
    plan=validate_plan(read_json(state/'download_plan.json'))
    completed=read_json(state/'downloads_complete.json')
    if completed['plan_hash']!=plan['plan_hash']:raise RuntimeError('Download completion belongs to another plan')
    if completed.get('files')!=len(plan['files']) or completed.get('verified_bytes')!=plan['total_bytes']:
        raise RuntimeError('Not all planned files have completed downloading/verification')
    already=state/'PREPARED.json'
    identity={key:request[key] for key in ('root','world_python','reward_python','gpus','conditions_per_batch','generation_batch_size','reward_batch_size')}
    if already.exists():
        previous=read_json(already)
        if previous['request_identity']!=identity:raise RuntimeError('Prepared workspace parameters differ; existing progress is preserved. Use launcher arguments to tune batches/GPUs.')
        install_python_entries(root,request)
        launch_files(root,request)
        print('工作区已经准备完成；保留现有 cursor、checkpoint 和全部训练成果。',flush=True)
        return
    marker=state/'initializing.json'
    if not marker.exists():
        if run.exists() and any(run.iterdir()):raise RuntimeError('Existing study_v1 is not owned by this bootstrap; preserved')
        write_json(marker,dict(request_identity=identity,created_at=now()))
    elif read_json(marker)['request_identity']!=identity:
        raise RuntimeError('Interrupted preparation has different parameters; preserved')
    print('核验 SFT、tokenizer、训练 CSV 和完整 Qwen 分片索引……',flush=True)
    asset_checks=verify_local_assets(root)
    asset_checks['reward_python_local_processor']=probe_reward_python(root,request['reward_python'])
    world_source=sources/'world/world.pt'
    if file_hash(world_source)!=WORLD_SHA:raise RuntimeError('Published formal world checkpoint identity mismatch')
    public=torch.load(world_source,map_location='cpu',weights_only=False)
    if public.get('schema')!='rl-world-checkpoint-v1' or public['state']['optimizer_step']!=652 or public['state']['smoke']:
        raise RuntimeError('Expected formal Stage A step 652')
    if not public['state']['batch_done'] or public['state']['update_cursor']:
        raise RuntimeError('Stage A is not at a complete batch boundary')
    if public['config']['nft']['mode']!='B-only' or public['sft_sha256']!=SFT_SHA or public['tokenizer_sha256']!=TOK_SHA:
        raise RuntimeError('Wrong Stage A model/tokenizer/mode')
    endpoint=os.environ.get('STATEPLAY_GPT_API_BASE_URL') or os.environ.get('OPENAI_BASE_URL')
    if endpoint:
        parsed=urllib.parse.urlsplit(endpoint)
        if parsed.scheme not in ('http','https') or not parsed.netloc or parsed.username or parsed.password or parsed.query:
            raise RuntimeError('API base URL must contain no credentials or query')
    config=deployment_config(public['config'],root,request['reward_python'],endpoint)
    final=run/'world/api/final/versions'/MODEL_VERSION/'world.pt'
    _,migrated=save_checkpoint(world_source,final,config)
    sidecar=read_json(sources/'world/world.json')
    sidecar.update(sha256=file_hash(final),config_hash=config['config_hash'])
    immutable_json(final.with_suffix('.json'),sidecar)
    immutable_json(run/'world/api/final/current.json',dict(checkpoint=str(final),optimizer_step=652,conditions_cursor=2280))
    # Source order and leakage audit stay byte-identical; only catalog paths relocate.
    source_plan=recovery/'records/study_v1/plan'
    order_record=read_json(source_plan/'epoch_order.json');order=order_record['order']
    cursor=read_json(source_plan/'cursor.json')
    if len(order)!=10000 or len(set(order))!=10000 or digest(order)!=order_record['order_hash']:
        raise RuntimeError('Original one-pass order checksum mismatch')
    if (cursor['cursor'],cursor['api_consumed'],cursor['local_consumed'],cursor['inflight'])!=(2280,2280,0,None):
        raise RuntimeError('Expected the completed 2280-condition Stage A boundary')
    cursor['last_checkpoint']=str(final)
    immutable_json(run/'plan/epoch_order.json',order_record)
    immutable_json(run/'plan/cursor.json',cursor)
    immutable_json(run/'plan/config.json',config)
    for name in ('stage_C_control.json','stage_state.json'):
        immutable_json(run/'plan'/name,relocate(read_json(source_plan/name),root))
    catalog=relocate(read_json(recovery/'metadata/source_audit/catalog.json'),root)
    records=catalog['records'];by_id={r['condition_id']:r for r in records}
    if len(by_id)!=10000 or set(by_id)!=set(order):raise RuntimeError('Catalog/order membership mismatch')
    for item in records:
        for name in ('video','action'):
            p=Path(item[name])
            if root.resolve() not in p.resolve().parents or not p.is_file():raise RuntimeError('Relocated source missing')
    immutable_json(run/'conditions/catalog.json',catalog)
    for name in ('leakage_index.json','leakage_features.json'):
        install_file(recovery/'metadata/source_audit'/name,run/'conditions'/name)
    # Preserve all published recovery evidence under provenance, never mark it as
    # current source/hardware acceptance or change a failed quality gate to pass.
    for relative in RECOVERY_FILES:
        install_file(recovery/relative,run/'provenance/huggingface'/relative)
    reward_root=run/'reward_model/versions/a-002280'
    selected=reward_root/'selected';source_selected=sources/'reward'
    if file_hash(source_selected/'adapter/adapter_model.safetensors')!=ADAPTER_SHA:
        raise RuntimeError('Wrong formal reward adapter')
    manifest=relocate(read_json(source_selected/'manifest.json'),root)
    if manifest['version']!=RM_VERSION or manifest['smoke'] or manifest['steps']!=132:
        raise RuntimeError('Expected formal non-smoke selected reward model')
    # Keep each RM algorithm/data setting; relocate paths and deployment endpoints.
    manifest['config']=deployment_config(manifest['config'],root,request['reward_python'],endpoint)
    rm_state=source_selected/'training_state.pt'
    if file_hash(rm_state)!=read_json(source_selected/'manifest.json')['training_state_sha256']:
        raise RuntimeError('Published RM training state checksum mismatch')
    original_rm=torch.load(rm_state,map_location='cpu',weights_only=False)
    if original_rm.get('step')!=132 or original_rm.get('smoke'):raise RuntimeError('Wrong selected RM optimizer step')
    migrated_rm_config=deployment_config(original_rm['config'],root,request['reward_python'],endpoint)
    save_checkpoint(rm_state,selected/'training_state.pt',migrated_rm_config)
    manifest['training_state_sha256']=file_hash(selected/'training_state.pt')
    for path in source_selected.rglob('*'):
        if not path.is_file():continue
        relative=path.relative_to(source_selected)
        if relative.as_posix() in ('manifest.json','training_state.pt','SHA256SUMS','PUBLICATION.json','README.md'):continue
        if relative.as_posix()=='adapter/adapter_config.json':
            immutable_json(selected/relative,relocate(read_json(path),root))
        elif relative.as_posix()=='selection_complete.json':
            immutable_json(selected/relative,relocate(read_json(path),root))
        else:install_file(path,selected/relative)
    immutable_json(selected/'manifest.json',manifest)
    immutable_json(run/'reward_model/active.json',dict(relative_path='versions/a-002280'))
    for relative in RECOVERY_FILES:
        prefix='records/study_v1/reward_model/versions/a-002280/'
        if relative.startswith(prefix):
            immutable_json(reward_root/relative[len(prefix):],relocate(read_json(recovery/relative),root))
    # This is an honest pending gate, not an invented acceptance result.
    immutable_json(run/'reports/integration_gate.json',dict(
        schema='rl-integration-gate-v1',passed=False,software_acceptance_only=True,
        reasons=['Fresh checkout/hardware acceptance must be generated on this machine'],
        historical_evidence=str(run/'provenance/huggingface/records/study_v1/reports/integration_gate.json')))
    c=copy.deepcopy(config);c['stage']='local'
    c['plan'].update(api_conditions=2280,api_max_conditions=2280)
    c['rollout']['conditions_per_batch']=request['conditions_per_batch']
    c['topology']=dict(strategy='timeslice',world_gpus=request['gpus'],reward_gpus=request['gpus'])
    c['stage_c']=dict(generation_batch_size=request['generation_batch_size'],reward_batch_size=request['reward_batch_size'])
    c['config_hash']=digest({k:v for k,v in c.items() if k!='config_hash'})
    c_path=run/'plan/bootstrap_C.yaml'
    c_body=yaml.safe_dump(c,sort_keys=False).encode()
    if c_path.exists() and c_path.read_bytes()!=c_body:raise RuntimeError('Existing C config differs; preserved')
    if not c_path.exists():write_bytes(c_path,c_body)
    sys.path.insert(0,str(root))
    from rl_posttrain.config import validate
    validate(c,'study_v1')
    preserved=[k for k in public if k!='config']
    if not all(exact(public[k],migrated[k]) for k in preserved):raise RuntimeError('World training state changed')
    report=dict(schema='stateplay-bootstrap-prepared-v1',request_identity=identity,created_at=now(),
        github_commit=GIT_COMMIT,download_plan_hash=plan['plan_hash'],asset_checks=asset_checks,
        source_world_sha256=WORLD_SHA,installed_world_sha256=file_hash(final),
        model_state_preserved_exactly=preserved,reward_version=RM_VERSION,adapter_sha256=ADAPTER_SHA,
        formal_training_started=False,cursor=2280,remaining_conditions=7720,
        config=str(c_path),api_endpoint_from_environment=bool(endpoint),
        saved_world_torch_version=public['state']['backend']['torch_version'],
        current_world_torch_version=torch.__version__,
        formal_quality_gate_passed=read_json(reward_root/'reports/quality_gate.json').get('passed',False))
    install_python_entries(root,request)
    launch_files(root,request)
    write_json(already,report)
    print('C 工作区已恢复：已消费 2280，剩余 7720；模型、优化器和 RNG 数值完全保留。',flush=True)
    print('正式 C 仍须通过真实软件/硬件验收与人工质量审核；详情见 CHECK_REPORT.json。',flush=True)


def launch_files(root, request):
    q=shlex.quote;folder=root/STATE/'launch';folder.mkdir(parents=True,exist_ok=True)
    script=root/STATE/'bootstrap_stateplay.py'
    body=Path(__file__).absolute().read_bytes()
    if script.exists() and script.read_bytes()!=body:
        previous=script.read_bytes();previous_sha=hashlib.sha256(previous).hexdigest()
        if previous_sha not in LEGACY_BOOTSTRAP_SHA256:
            raise RuntimeError('Installed bootstrap has unknown local edits; preserved')
        backup=root/STATE/'history'/('bootstrap_'+previous_sha+'.py')
        if not backup.exists():write_bytes(backup,previous)
        write_bytes(script,body)
    if not script.exists():write_bytes(script,body)
    world=q(request['world_python']);reward=q(request['reward_python'])
    gpus=q(','.join(map(str,request['gpus'])))
    common=f'''#!/usr/bin/env bash
set -euo pipefail
cd {q(str(root))}
export STATEPLAY_WORLD_PYTHON={world}
export STATEPLAY_REWARD_PYTHON={reward}
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
'''
    sft=common+f'''export PATH={q(str(Path(request['world_python']).parent))}:"$PATH"
export CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES:-{','.join(map(str,request['gpus']))}}}"
export TRAIN_MODE=latent_state_world_model
export BATCH_SIZE="${{BATCH_SIZE:-1}}"
export MAX_TRAIN_STEPS="${{MAX_TRAIN_STEPS:-40000}}"
export SAVE_STEPS="${{SAVE_STEPS:-10000}}"
export STATEPLAY_BASE_MODEL="$PWD/base_model"
export STATEPLAY_DATA_ROOT="$PWD/data/SF3/train/clips_5s"
export STATEPLAY_METADATA="$STATEPLAY_DATA_ROOT/metadata_state_polish.csv"
export STATEPLAY_OUTPUT="${{STATEPLAY_OUTPUT:-$PWD/outputs/latent_world_fresh_$(date +%Y%m%d_%H%M%S)_$$}}"
{world} - "$PWD" "$STATEPLAY_OUTPUT" "$@" <<'PY'
import sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();out=sys.argv[2];extra=sys.argv[3:]
for i,value in enumerate(extra):
    if value=='--output_path':out=extra[i+1]
    elif value.startswith('--output_path='):out=value.split('=',1)[1]
out=Path(out).expanduser().resolve()
for protected in ('outputs/latent_world','outputs/tokenizer','outputs/rl_posttrain'):
    guard=(root/protected).resolve()
    if out==guard or guard in out.parents:
        raise SystemExit('拒绝写入受保护的原模型或 RL 输出目录。')
PY
exec bash scripts/train.sh --transition_tokenizer_ckpt "$PWD/outputs/tokenizer/transition_tokenizer.pt" "$@"
'''
    c=common+f'''# Running this launcher is an explicit request to start C after all genuine gates pass.
exec {world} {q(str(script))} --project-root {q(str(root))} --start-c "$@"
'''
    check=common+f'''exec {world} {q(str(script))} --project-root {q(str(root))} --check "$@"
'''
    for name,body in (('start_sft.sh',sft),('start_c.sh',c),('check.sh',check)):
        target=folder/name
        if target.exists() and target.read_text()!=body:raise RuntimeError('Existing launcher differs; preserved')
        if not target.exists():write_bytes(target,body.encode());target.chmod(0o755)


def check_workspace(root, *, require_c=False):
    state=root/STATE;run=root/RUN;blockers=[]
    if not (state/'PREPARED.json').is_file():raise RuntimeError('Workspace not prepared; run --prepare-only with configured environments')
    info=read_json(state/'PREPARED.json')
    cursor=read_json(run/'plan/cursor.json');order=read_json(run/'plan/epoch_order.json')
    if digest(order['order'])!=order['order_hash'] or len(set(order['order']))!=10000:raise RuntimeError('One-pass order changed')
    if cursor['cursor']!=cursor['api_consumed']+cursor['local_consumed'] or cursor['api_consumed']!=2280:
        raise RuntimeError('Budget ledger mismatch; refusing to reset')
    for name,message in (
        (run/'reports/integration_gate.json','本机软件验收尚未通过'),
        (run/'reward_model/versions/a-002280/reports/quality_gate.json','Reward Model 人工/独立质量审核尚未通过')):
        if not name.exists() or not read_json(name).get('passed'):blockers.append(message)
    measured=run/'reports/C_timeslice_acceptance.json'
    joint=run/'reports/joint_profile.json'
    if not (measured.exists() and read_json(measured).get('passed')) and not (joint.exists() and read_json(joint).get('selected_topology')):
        blockers.append('本机 C 显卡拓扑验收尚未完成')
    key=os.environ.get('STATEPLAY_GPT_API_KEY') or os.environ.get('OPENAI_API_KEY')
    endpoint=os.environ.get('STATEPLAY_GPT_API_BASE_URL') or os.environ.get('OPENAI_BASE_URL')
    if not key or not endpoint:blockers.append('C 首批 56 条及周期审计需要本机 API 环境变量')
    prepared_config=read_json(run/'plan/config.json')
    if endpoint and endpoint.rstrip('/')!=prepared_config['reward']['endpoint']:
        blockers.append('API 地址与已迁移 checkpoint 配置不同；需按项目 provider 迁移协议处理，不能直接修改配置')
    for key in ('world_python','reward_python'):
        if not os.access(info['request_identity'][key],os.X_OK):blockers.append(key+' 路径不可执行')
    for name in ('world','reward'):
        if not os.access(root/'outputs/rl_posttrain/_envs'/name/'bin/python',os.X_OK):
            blockers.append(name+' 内部 Python 入口缺失')
    if info['current_world_torch_version']!=info['saved_world_torch_version']:
        blockers.append('WORLD torch 版本与原 A checkpoint 的恢复后端不同')
    result=dict(checked_at=now(),assets_prepared=True,formal_c_ready=not blockers,
        cursor=cursor['cursor'],api_consumed=cursor['api_consumed'],local_consumed=cursor['local_consumed'],
        remaining_conditions=10000-cursor['cursor'],blockers=blockers,
        sft_launcher=str(state/'launch/start_sft.sh'),c_launcher=str(state/'launch/start_c.sh'),
        historical_C_hold_preserved=not read_json(run/'plan/stage_C_control.json').get('allow_start',False))
    write_json(state/'CHECK_REPORT.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    if require_c and blockers:raise RuntimeError('正式 C 启动条件未满足；检查报告中的真实门槛未被跳过')
    return result


def start_c(root, args, extra):
    state=root/STATE;run=root/RUN
    check_workspace(root,require_c=True)
    info=read_json(state/'PREPARED.json');request=info['request_identity']
    world=request['world_python'];reward=request['reward_python']
    # Delegate to the repository's own full gate before changing the historical hold.
    control_path=run/'plan/stage_C_control.json';old=read_json(control_path)
    if not old.get('allow_start'):
        install_file(control_path,run/'provenance/bootstrap/historical_stage_C_control.json')
        write_json(control_path,dict(old,allow_start=True,reason='User explicitly invoked start_c.sh',resumed_at=now()))
    env=os.environ.copy();env.update(STATEPLAY_WORLD_PYTHON=world,STATEPLAY_REWARD_PYTHON=reward)
    config_path=info['config']
    if args.conditions_per_batch is not None:
        import yaml
        config=yaml.safe_load(Path(config_path).read_text())
        config['rollout']['conditions_per_batch']=args.conditions_per_batch
        config['config_hash']=digest({k:v for k,v in config.items() if k!='config_hash'})
        path=run/'plan'/('bootstrap_C_batch_'+str(args.conditions_per_batch)+'.yaml')
        body=yaml.safe_dump(config,sort_keys=False).encode()
        if path.exists() and path.read_bytes()!=body:raise RuntimeError('Existing runtime batch config differs')
        if not path.exists():write_bytes(path,body)
        config_path=str(path)
    command=[world,'-m','rl_posttrain.launch_local','--run-id','study_v1','--config',config_path,
             '--update-mode',args.update_mode,
             '--gpus',','.join(map(str,args.gpus or request['gpus'])),
             '--generation-batch-size',str(args.generation_batch_size or request['generation_batch_size']),
             '--reward-batch-size',str(args.reward_batch_size or request['reward_batch_size']),*extra]
    return subprocess.call(command,cwd=root,env=env)


def parse_gpus(value):
    try:ids=[int(s) for s in value.split(',')]
    except ValueError:raise argparse.ArgumentTypeError('显卡编号使用逗号分隔') from None
    if not ids or min(ids)<0 or len(set(ids))!=len(ids):raise argparse.ArgumentTypeError('显卡编号必须非负且不能重复')
    return ids


def prepare(root,args):
    request=dict(root=str(root),world_python=str(absolute(args.world_python or sys.executable)),
        reward_python=str(absolute(args.reward_python or sys.executable)),gpus=args.gpus or [0,1,2,3],
        conditions_per_batch=args.conditions_per_batch or 28,generation_batch_size=args.generation_batch_size or 1,
        reward_batch_size=args.reward_batch_size or 1)
    for key in ('world_python','reward_python'):
        if not os.access(request[key],os.X_OK):raise RuntimeError('Python 不可执行: '+request[key])
    path=root/STATE/'prepare_request.json';write_json(path,request)
    print('开始在 WORLD Python 中准备路径和恢复状态；不占用 GPU，不启动训练。',flush=True)
    result=subprocess.run([request['world_python'],'-B',str(Path(__file__).absolute()),'--_prepare-request',str(path)],cwd=root)
    if result.returncode:raise RuntimeError('准备失败；下载内容保留。配置环境后用 --prepare-only 重试')
    check_workspace(root)


def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root',type=absolute,help='项目根目录，可自定义名称')
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--plan',action='store_true',help='只查询并展示下载清单，不克隆/下载资产')
    modes.add_argument('--download-only',action='store_true',help='只完成 Git 与资源下载，无需 torch 环境')
    modes.add_argument('--prepare-only',action='store_true',help='使用已有下载结果准备 C/SFT 工作区')
    modes.add_argument('--check',action='store_true',help='检查工作区和正式 C 未满足项')
    modes.add_argument('--start-c',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--workers',type=int,default=8,help='并发下载文件数，默认8')
    p.add_argument('--retries',type=int,default=10,help='每个文件/元数据的总尝试次数，默认10')
    p.add_argument('--timeout',type=float,default=120,help='每次网络读取超时秒数')
    p.add_argument('--backoff',type=float,default=3,help='重试退避基数秒数，默认3')
    p.add_argument('--endpoint',default='https://huggingface.co',help='可指定 https://hf-mirror.com，失败自动回退官方')
    p.add_argument('--world-python',help='默认使用执行脚本的当前 Python')
    p.add_argument('--reward-python',help='默认使用执行脚本的当前 Python')
    p.add_argument('--gpus',type=parse_gpus,help='默认0,1,2,3；例如4,5,6,7')
    p.add_argument('--conditions-per-batch',type=int,help='默认28；调整全局组数，不改变一遍数据的总预算')
    p.add_argument('--generation-batch-size',type=int,choices=(1,2,3,4))
    p.add_argument('--reward-batch-size',type=int)
    p.add_argument('--update-mode','--train-mode',choices=('B-only','Coupling-only','Both'),default='B-only',help='C 阶段训练模块，默认 B-only；仅能在 A 完成且 C 尚未开始时选择')
    p.add_argument('--_prepare-request',type=Path,help=argparse.SUPPRESS)
    return p


def main(argv=None):
    p=parser();args,extra=p.parse_known_args(argv)
    if extra and not args.start_c:p.error('unrecognized arguments: '+' '.join(extra))
    if args._prepare_request:
        prepare_worker(read_json(args._prepare_request));return 0
    if args.project_root is None:p.error('--project-root is required')
    if min(args.workers,args.retries,args.timeout)<=0 or args.conditions_per_batch is not None and args.conditions_per_batch<1 or args.backoff<0 or args.reward_batch_size is not None and args.reward_batch_size<1:
        p.error('workers/retries/timeout/batch 必须为正数，backoff 不能为负数')
    root=args.project_root
    if args.check:check_workspace(root);return 0
    if args.start_c:return start_c(root,args,extra)
    http=Http([args.endpoint,'https://huggingface.co'],args.timeout,args.retries,args.backoff,hf_token())
    if args.plan:
        plan=make_plan(http)
        for repo in sorted(plan['repositories'],key=lambda x:x['bytes']):
            print(f"{repo['bytes']/1e9:8.3f} GB  {repo['files']:5d} 文件  {repo['repo']}")
        print(f"合计 {len(plan['files'])} 文件，{plan['total_bytes']/1e9:.3f} GB；不含完整44.18GB奖励训练集。")
        return 0
    if args.prepare_only:
        clone(root,args.retries,args.backoff)
        with process_lock(root/STATE/'bootstrap.lock'):prepare(root,args)
        return 0
    # Git is intentionally first. Do not put a state folder into an un-cloned target.
    print('克隆/恢复 Git 仓库（固定已验证 commit，不修改已有工作区）……',flush=True)
    clone(root,args.retries,args.backoff)
    with process_lock(root/STATE/'bootstrap.lock'):
        manifest=root/STATE/'download_plan.json'
        if manifest.exists():plan=validate_plan(read_json(manifest))
        else:
            plan=make_plan(http);write_json(manifest,plan)
        print(f"下载计划：{len(plan['files'])} 文件，{plan['total_bytes']/1e9:.2f} GB，按文件大小排序。",flush=True)
        download_all(root,plan,http,args.workers)
        if not args.download_only:prepare(root,args)
        else:print('资源下载完成。环境配置后使用同一脚本 --prepare-only。',flush=True)
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:
        print('\n已停止；完整文件与 .incomplete 均保留，重复原命令即可恢复。',file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        # Exception messages produced by this script contain no credentials;
        # external transport errors are reduced to their class before reporting.
        if isinstance(error,(RuntimeError,ValueError,FileNotFoundError)):
            print('错误：'+str(error),file=sys.stderr)
        else:print('错误类型：'+type(error).__name__+'；已保留下载/恢复状态。',file=sys.stderr)
        raise SystemExit(1)
