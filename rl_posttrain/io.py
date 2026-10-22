"""Durable artifacts and path protection. No credentials are serialized here."""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
SFT = ROOT / "outputs/latent_world/step-4572.safetensors"
SFT_SHA = "53eff9e5e59e92bb3cc7810e3ff36f9ee490be4a122a3a8ff96a4de0a03b4c5d"
QWEN = ROOT / "base_model/Qwen/Qwen3.5-9B"

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()

def protect(path):
    p = Path(path).resolve()
    if p == SFT.resolve() or SFT.resolve() in p.parents:
        raise ValueError("Original SFT checkpoint is strictly read-only")
    return p

def run_path(run_id):
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id):
        raise ValueError("run_id must contain only letters, digits, _ and -")
    return ROOT / "outputs/rl_posttrain" / run_id

def reward_path(run):
    """A mutable pointer selects an immutable, separately resumable RM attempt."""
    root = Path(run) / "reward_model"
    pointer = root / "active.json"
    if not pointer.exists():
        return root
    relative = Path(read_json(pointer)["relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Reward version path must stay inside this run")
    selected = (root / relative).resolve()
    if root.resolve() not in selected.parents:
        raise ValueError("Reward version path escaped its run")
    return selected

def teacher_path(group):
    """Select provider-versioned labels without moving historical teacher assets."""
    root=Path(group)/'teacher'
    pointer=root/'active.json'
    if not pointer.exists():return root
    relative=Path(read_json(pointer)['relative_path'])
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Teacher version escaped its group')
    selected=(root/relative).resolve()
    if root.resolve() not in selected.parents:raise ValueError('Teacher version escaped its group')
    return selected

def api_cache_path(run,endpoint=None):
    """Retain earlier provider submissions in their original database."""
    root=Path(run)/'requests';pointer=root/'active.json'
    if not pointer.exists():return root/'cache.sqlite3'
    record=read_json(pointer)
    if endpoint is not None and record['endpoint']!=endpoint:raise ValueError('API cache/provider configuration mismatch')
    relative=Path(record['relative_path'])
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('API cache escaped its run')
    selected=(root/relative).resolve()
    if root.resolve() not in selected.parents:raise ValueError('API cache escaped its run')
    return selected

def reward_checkpoint(run):
    root=reward_path(run)
    selected=root/'selected'
    return selected if (selected/'manifest.json').exists() else root/'final'

def world_checkpoint(run,stage):
    root=Path(run)/'world'/stage/'final'
    pointer=root/'current.json'
    if not pointer.exists():return root/'world.pt'
    selected=Path(read_json(pointer)['checkpoint']).resolve()
    if (Path(run)/'world'/stage).resolve() not in selected.parents:
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

def atomic_bytes(path, data):
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

def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())

def read_json(path):
    return json.loads(Path(path).read_text())

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
