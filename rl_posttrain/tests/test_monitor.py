from rl_posttrain.rewards.monitor import audit_count,append_audit,record_audit

def test_initial_exact_56_and_two_bad_windows(tmp_path,config):
    path=tmp_path/"audit.json";state={};counts=[]
    for consumed in (0,24,48):
        n=audit_count(state,{"local_consumed":consumed,"local_batches":consumed//24},config,24);counts.append(n)
        t=[[0,25,75,100]]*n;s=[[100,75,25,0]]*n
        state=append_audit(path,[f"g{i+consumed}" for i in range(n)],t,s)
    assert counts==[24,24,8];assert state["initial_seen"]==56;assert len(state["windows"])==2;assert state["paused"]

def test_periodic_window_spans_24_plus_4(tmp_path,config):
    state={"initial_seen":56,"pending":[]};assert audit_count(state,{"local_batches":100},config,24)==24
    state["pending"]=[{}]*24;assert audit_count(state,{"local_batches":101},config,24)==4
