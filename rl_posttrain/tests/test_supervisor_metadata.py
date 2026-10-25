"""No GPU/API: a failed heartbeat must not orphan its still-running child."""
import errno
from unittest.mock import Mock
import subprocess
from rl_posttrain import orchestrate


def test_metadata_outage_keeps_waiting_for_owned_child(tmp_path,monkeypatch):
    process=Mock(pid=99999999)
    process.wait.side_effect=[subprocess.TimeoutExpired('test',15),0]
    monkeypatch.setattr(orchestrate.subprocess,'Popen',lambda *a,**kw:process)
    writes=[]
    def failed_record(*args,**kwargs):
        writes.append(1)
        raise OSError(errno.ESTALE,'injected metadata outage')
    monkeypatch.setattr(orchestrate,'write_json',failed_record)
    assert orchestrate.execute(['fake-worker'],tmp_path,'isolated_test',allow_failure=True)==0
    assert process.wait.call_count==2 and len(writes)==3
    process.terminate.assert_not_called()
    process.kill.assert_not_called()
