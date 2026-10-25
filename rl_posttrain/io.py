"""Durable artifacts and path protection. No credentials are serialized here."""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import errno

ROOT = Path(__file__).resolve().parents[1]
SFT = ROOT / "outputs/latent_world/step-4572.safetensors"
SFT_SHA = "53eff9e5e59e92bb3cc7810e3ff36f9ee490be4a122a3a8ff96a4de0a03b4c5d"
QWEN = ROOT / "base_model/Qwen/Qwen3.5-9B"

def retry_storage(operation):
    """Bounded retry of lookup/transport failures; validation errors stay terminal."""
    for attempt in range(6):
        try:return operation()
        except OSError as error:
            if error.errno not in (errno.ENOENT,errno.EIO,errno.ESTALE,errno.ETIMEDOUT) or attempt==5:
                raise
            time.sleep(.2*2**attempt)

def resolve_shared(path):
    return retry_storage(lambda:Path(path).resolve())

def shared_exists(path,file_only=False,retry_missing=True):
    # Path.exists silently converts transient ENOENT to False. Probe explicitly
    # before interpreting absence of a committed pointer or quality report.
    import stat
    def probe():
        mode=Path(path).stat().st_mode
        return stat.S_ISREG(mode) if file_only else True
    def checked():
        try:return probe()
        except FileNotFoundError:
            if retry_missing:raise
            return False
    try:return retry_storage(checked)
    except FileNotFoundError:return False

def sha256(path):
    def read(p):
        h=hashlib.sha256()
        with p.open("rb") as f:
            for block in iter(lambda:f.read(16 << 20),b""):h.update(block)
        return h.hexdigest()
    return _read_shared(path,read)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()

def protect(path):
    p = resolve_shared(path)
    source = resolve_shared(SFT)
    if p == source or source in p.parents:
        raise ValueError("Original SFT checkpoint is strictly read-only")
    return p

def run_path(run_id):
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id):
        raise ValueError("run_id must contain only letters, digits, _ and -")
    return ROOT / "outputs/rl_posttrain" / run_id

def reward_path(run):
    """A mutable pointer selects an immutable, separately resumable RM attempt."""
    root = resolve_shared(Path(run) / "reward_model")
    pointer = root / "active.json"
    if not shared_exists(pointer,retry_missing=(root/'versions').is_dir()):
        return root
    relative = Path(read_json(pointer)["relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Reward version path must stay inside this run")
    selected = resolve_shared(root / relative)
    if resolve_shared(root) not in selected.parents:
        raise ValueError("Reward version path escaped its run")
    return selected

def teacher_path(group):
    """Select provider-versioned labels without moving historical teacher assets."""
    root=resolve_shared(Path(group)/'teacher')
    pointer=root/'active.json'
    if not shared_exists(pointer,retry_missing=False):return root
    relative=Path(read_json(pointer)['relative_path'])
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Teacher version escaped its group')
    selected=resolve_shared(root/relative)
    if resolve_shared(root) not in selected.parents:raise ValueError('Teacher version escaped its group')
    return selected

def api_cache_path(run,endpoint=None):
    """Retain earlier provider submissions in their original database."""
    root=resolve_shared(Path(run)/'requests');pointer=root/'active.json'
    if not shared_exists(pointer,retry_missing=False):return root/'cache.sqlite3'
    record=read_json(pointer)
    if endpoint is not None and record['endpoint']!=endpoint:raise ValueError('API cache/provider configuration mismatch')
    relative=Path(record['relative_path'])
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('API cache escaped its run')
    selected=resolve_shared(root/relative)
    if resolve_shared(root) not in selected.parents:raise ValueError('API cache escaped its run')
    return selected

def reward_checkpoint(run):
    root=reward_path(run)
    selected=root/'selected'
    return selected if shared_exists(selected/'manifest.json',retry_missing=selected.is_dir()) else root/'final'

def world_checkpoint(run,stage):
    root=Path(run)/'world'/stage/'final'
    pointer=root/'current.json'
    if not shared_exists(pointer,retry_missing=(root/'versions').is_dir()):return root/'world.pt'
    selected=resolve_shared(read_json(pointer)['checkpoint'])
    if resolve_shared(Path(run)/'world'/stage) not in selected.parents:
        raise ValueError('World final checkpoint pointer escaped this run/stage')
    return selected

def publish_link(path, target):
    """Advance a pointer without rewriting the checkpoint it previously selected."""
    path = Path(path)
    protect(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not path.is_symlink():
        raise ValueError("Refusing to replace an existing artifact with a pointer: " + str(path))
    temporary = path.with_name(path.name + ".pointer-pending")
    if temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(os.path.relpath(Path(target).resolve(), path.parent.resolve()))
    os.replace(temporary, path)

@contextlib.contextmanager
def lock(path):
    p = protect(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield
        fcntl.flock(f, fcntl.LOCK_UN)

def _atomic_bytes_once(path, data):
    p = protect(path); p.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=p.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(name, p)
        dfd = os.open(p.parent, os.O_DIRECTORY)
        try: os.fsync(dfd)
        finally: os.close(dfd)
    finally:
        if os.path.exists(name): os.unlink(name)

def atomic_bytes(path,data):
    # Replacing the same JSON bytes is idempotent. Never retry append_jsonl,
    # ledger commits or optimizer steps here.
    return retry_storage(lambda:_atomic_bytes_once(path,data))

def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())

def _read_shared(path,reader):
    # Resolution itself may fail on an NFS alias; it belongs inside the retry.
    return retry_storage(lambda:reader(Path(path).resolve()))

def read_json(path):
    return json.loads(_read_shared(path,lambda p:p.read_text()))

def load_tensors(path,*args,**kwargs):
    from safetensors.torch import load_file
    return _read_shared(path,lambda p:load_file(p,*args,**kwargs))

def append_jsonl(path, value):
    p = protect(path); p.parent.mkdir(parents=True, exist_ok=True)
    with lock(p.with_suffix(p.suffix + ".lock")):
        with p.open("a") as f:
            f.write(json.dumps(value, allow_nan=False) + "\n"); f.flush(); os.fsync(f.fileno())

def atomic_torch(path, value):
    import torch
    p = protect(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".pending")
    torch.save(value, tmp)
    with tmp.open("rb") as f: os.fsync(f.fileno())
    os.replace(tmp, p)

def save_tensors(path, values):
    from safetensors.torch import save
    atomic_bytes(path, save({k: v.detach().cpu().contiguous() for k, v in values.items()}))

def fingerprint(path, hash_file=True):
    p = Path(path); s = p.stat()
    result = {"path": str(p.resolve()), "size": s.st_size, "mtime_ns": s.st_mtime_ns}
    if hash_file: result["sha256"] = sha256(p)
    return result

def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
