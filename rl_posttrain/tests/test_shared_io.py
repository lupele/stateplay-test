from pathlib import Path
import errno
import pytest
import torch
from rl_posttrain.io import read_json,write_json,save_tensors,load_tensors,_read_shared

def test_transient_existing_shared_json_read_retries(tmp_path,monkeypatch):
    p=tmp_path/'value.json';write_json(p,{'value':3})
    original=Path.read_text;calls=[]
    def flaky(path,*args,**kwargs):
        calls.append(path)
        if len(calls)==1:raise FileNotFoundError(errno.ENOENT,'injected transient lookup',str(path))
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',flaky)
    monkeypatch.setattr('rl_posttrain.io.time.sleep',lambda _:None)
    assert read_json(p)=={'value':3} and len(calls)==2

def test_aliased_tensor_paths_are_read_canonically(tmp_path):
    real=tmp_path/'real';real.mkdir();alias=tmp_path/'alias';alias.symlink_to(real,target_is_directory=True)
    save_tensors(real/'values.safetensors',{'x':torch.arange(4)})
    assert torch.equal(load_tensors(alias/'values.safetensors')['x'],torch.arange(4))
    seen=[]
    assert _read_shared(alias/'values.safetensors',lambda p:seen.append(p) or 7)==7
    assert seen==[(real/'values.safetensors').resolve()]

def test_shared_io_retry_is_bounded_and_preserves_failure(tmp_path,monkeypatch):
    p=tmp_path/'exists';p.write_text('exists')
    calls=[]
    def broken(path):
        calls.append(path);raise OSError(errno.EIO,'injected persistent I/O failure')
    monkeypatch.setattr('rl_posttrain.io.time.sleep',lambda _:None)
    with pytest.raises(OSError,match='persistent'):_read_shared(p,broken)
    assert len(calls)==6

@pytest.mark.parametrize('code',[errno.ENOENT,errno.EIO,errno.ESTALE,errno.ETIMEDOUT])
def test_resolution_failure_is_inside_read_retry(tmp_path,monkeypatch,code):
    p=tmp_path/'value.json';p.write_text('{"ok": true}')
    original=Path.resolve;calls=[]
    def flaky(path,*a,**kw):
        calls.append(path)
        if len(calls)==1:raise OSError(code,'injected resolution failure',str(path))
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'resolve',flaky)
    monkeypatch.setattr('rl_posttrain.io.time.sleep',lambda _:None)
    assert read_json(p)=={'ok':True} and len(calls)==2


def test_shared_exists_retries_false_negative(tmp_path,monkeypatch):
    from rl_posttrain.io import shared_exists
    p=tmp_path/'committed.json';p.write_text('{}')
    original=Path.stat;calls=[]
    def flaky(path,*a,**kw):
        if path==p:
            calls.append(path)
            if len(calls)<=2:raise FileNotFoundError(errno.ENOENT,'transient lookup')
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'stat',flaky)
    monkeypatch.setattr('rl_posttrain.io.time.sleep',lambda _:None)
    assert shared_exists(p,file_only=True) and len(calls)==3


def test_shared_exists_missing_still_rejects(tmp_path,monkeypatch):
    from rl_posttrain.io import shared_exists
    monkeypatch.setattr('rl_posttrain.io.time.sleep',lambda _:None)
    assert shared_exists(tmp_path/'missing') is False


def test_atomic_json_retries_replace_without_partial_output(tmp_path,monkeypatch):
    import os
    p=tmp_path/'state.json';write_json(p,{'before':1})
    original=os.replace;calls=[]
    def flaky(src,dst):
        calls.append(dst)
        if len(calls)==1:raise OSError(errno.ESTALE,'transient rename')
        return original(src,dst)
    monkeypatch.setattr(os,'replace',flaky)
    monkeypatch.setattr('rl_posttrain.io.time.sleep',lambda _:None)
    write_json(p,{'after':2})
    assert read_json(p)=={'after':2} and len(calls)==2
    assert not list(tmp_path.glob('.pending-*'))


def test_protected_sft_stays_protected_after_transient_lookup(tmp_path,monkeypatch):
    import rl_posttrain.io as io
    p=tmp_path/'sft';p.write_text('immutable')
    alias=tmp_path/'alias';alias.symlink_to(p)
    original=Path.resolve;calls=[]
    def flaky(path,*a,**kw):
        if path==alias:
            calls.append(path)
            if len(calls)==1:raise FileNotFoundError(errno.ENOENT,'transient SFT alias')
        return original(path,*a,**kw)
    monkeypatch.setattr(io,'SFT',p)
    monkeypatch.setattr(Path,'resolve',flaky)
    monkeypatch.setattr(io.time,'sleep',lambda _:None)
    with pytest.raises(ValueError,match='strictly read-only'):write_json(alias,{'bad':1})
    assert p.read_text()=='immutable'


def test_permission_errors_are_not_retried(tmp_path,monkeypatch):
    calls=[]
    def denied(path):
        calls.append(path);raise PermissionError(errno.EACCES,'denied')
    monkeypatch.setattr('rl_posttrain.io.time.sleep',lambda _:None)
    with pytest.raises(PermissionError):_read_shared(tmp_path,denied)
    assert len(calls)==1


def test_committed_reward_pointer_false_absence_cannot_select_legacy(tmp_path,monkeypatch):
    from rl_posttrain.io import reward_path
    root=tmp_path/'reward_model';version=root/'versions/v1';version.mkdir(parents=True)
    pointer=root/'active.json';write_json(pointer,{'relative_path':'versions/v1'})
    original=Path.stat;calls=[]
    def flaky(path,*a,**kw):
        if path==pointer:
            calls.append(path)
            if len(calls)==1:raise FileNotFoundError(errno.ENOENT,'transient active pointer')
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'stat',flaky)
    monkeypatch.setattr('rl_posttrain.io.time.sleep',lambda _:None)
    assert reward_path(tmp_path)==version and len(calls)>=2


def test_optional_legacy_pointer_does_not_add_absence_backoff(tmp_path,monkeypatch):
    from rl_posttrain.io import teacher_path
    group=tmp_path/'group';(group/'teacher').mkdir(parents=True)
    sleeps=[];monkeypatch.setattr('rl_posttrain.io.time.sleep',sleeps.append)
    assert teacher_path(group)==group/'teacher'
    assert sleeps==[]
