"""Durable rank-zero API scoring, transactional claim, shared rate and group budgets."""
import concurrent.futures
import multiprocessing
import os
from pathlib import Path
import random
import sqlite3
import time
import uuid
from contextlib import contextmanager
import httpx
from ..io import write_json,read_json,now,lock,teacher_path,api_cache_path
from .base import GroupScorer,parse_scores
from .protocol import payload
from .transport import submit,SubmissionIncomplete

class RequestStore:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with lock(self.path.with_suffix(".init.lock")),self.connect() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("CREATE TABLE IF NOT EXISTS requests (key TEXT PRIMARY KEY, group_id TEXT, status TEXT, owner TEXT, result TEXT, attempts INTEGER DEFAULT 0, updated REAL, error TEXT)")
            c.execute("CREATE TABLE IF NOT EXISTS rate (stamp REAL, tokens INTEGER)")
            c.execute("CREATE TABLE IF NOT EXISTS retry_authorizations (id INTEGER PRIMARY KEY, key TEXT, baseline_attempts INTEGER, attempt_limit INTEGER, reason TEXT, granted REAL)")
    @contextmanager
    def connect(self):
        connection=sqlite3.connect(self.path,timeout=60,isolation_level=None)
        try:yield connection
        finally:connection.close()
    def claim(self,key,group_id,group_limit=None):
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            row=c.execute("SELECT status,result,attempts FROM requests WHERE key=?",(key,)).fetchone()
            if row:
                authorization=c.execute("SELECT attempt_limit FROM retry_authorizations WHERE key=? ORDER BY id DESC LIMIT 1",(key,)).fetchone()
                if row[0]=="pending" and (row[2]==0 or (authorization and row[2]<authorization[0])):
                    owner=uuid.uuid4().hex;c.execute("UPDATE requests SET status='inflight',owner=?,updated=? WHERE key=?",(owner,time.time(),key));c.execute("COMMIT");return "claimed",owner
                c.execute("COMMIT");return row[0],row[1]
            count=c.execute("SELECT count(*) FROM requests").fetchone()[0]
            if group_limit is not None and count>=group_limit:c.execute("ROLLBACK");raise RuntimeError("Authorized real API group limit reached")
            owner=uuid.uuid4().hex
            c.execute("INSERT INTO requests (key,group_id,status,owner,updated) VALUES (?,?, 'inflight',?,?)",(key,group_id,owner,time.time()));c.execute("COMMIT")
            return "claimed",owner
    def finish(self,key,status,result=None,error=None):
        import json
        with self.connect() as c:c.execute("UPDATE requests SET status=?,result=?,error=?,updated=? WHERE key=?",(status,json.dumps(result) if result is not None else None,error,time.time(),key))
    def attempt(self,key):
        with self.connect() as c:
            row=c.execute("UPDATE requests SET attempts=attempts+1,updated=? WHERE key=? RETURNING attempts",(time.time(),key)).fetchone()
            return row[0]
    def authorize_retry(self,key,additional_attempts,reason):
        """Explicit operator authorization; history and cumulative attempts survive.

        Automatic recovery may call this only under an explicit durable user
        authorization; all callers preserve the cumulative attempt ceiling.
        """
        if not isinstance(additional_attempts,int) or not 1<=additional_attempts<=30 or not reason:
            raise ValueError('Retry authorization requires a reason and 1..30 additional attempts')
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            row=c.execute('SELECT status,attempts FROM requests WHERE key=?',(key,)).fetchone()
            if not row or row[0] not in ('failed','unknown'):
                c.execute('ROLLBACK');raise ValueError('Only terminal failed/unknown requests can be explicitly reopened')
            c.execute('INSERT INTO retry_authorizations(key,baseline_attempts,attempt_limit,reason,granted) VALUES(?,?,?,?,?)',(key,row[1],row[1]+additional_attempts,reason,time.time()))
            c.execute("UPDATE requests SET status='pending',owner=NULL,error='explicit_bounded_retry_authorized',updated=? WHERE key=?",(time.time(),key))
            c.execute('COMMIT')
    def remaining_attempts(self,key,configured_max):
        with self.connect() as c:
            attempts=c.execute('SELECT attempts FROM requests WHERE key=?',(key,)).fetchone()[0]
            authorization=c.execute('SELECT attempt_limit FROM retry_authorizations WHERE key=? ORDER BY id DESC LIMIT 1',(key,)).fetchone()
        return max(0,min(configured_max,(authorization[0] if authorization else configured_max)-attempts))
    def recover_abandoned(self):
        # Only a coordinator holding the run lock may call this after workers died.
        with self.connect() as c:c.execute("UPDATE requests SET status='unknown',error='worker_abandoned_submission_state_unknown' WHERE status='inflight'")
    def rate_wait(self,rpm,tpm,tokens):
        if tokens>tpm:raise ValueError("Full group estimate exceeds configured TPM; increase capacity, do not drop images")
        while True:
            with self.connect() as c:
                c.execute("BEGIN IMMEDIATE");t=time.time();c.execute("DELETE FROM rate WHERE stamp<?",(t-60,))
                count,total,oldest=c.execute("SELECT count(*),coalesce(sum(tokens),0),min(stamp) FROM rate").fetchone()
                if count<rpm and total+tokens<=tpm:
                    c.execute("INSERT INTO rate VALUES (?,?)",(t,tokens));c.execute("COMMIT");return
                c.execute("COMMIT")
            time.sleep(min(1,max(.05,60-(t-oldest))))

class APIGroupScorer(GroupScorer):
    def __init__(self,config,run,group_limit=None,transport=None):
        self.config=config;self.run=run;self.limit=group_limit;self.transport=transport
        self.store=RequestStore(api_cache_path(run,config['reward']['endpoint']));self.model_id=config["reward"]["model"]
    def score(self,group):
        import json
        request,key,mapping=payload(group,self.config);teacher=teacher_path(group);state,value=self.store.claim(key,group.name,self.limit)
        if state=="succeeded":
            scores=parse_scores(json.loads(value))
            # Repair the crash window between committing a successful cache
            # row and publishing its score file. No provider call is needed.
            if not (teacher/'scores.json').exists():
                write_json(teacher/'scores.json',{'scores':scores,'mapping':mapping,'cache_key':key,
                           'scorer':self.model_id,'usage':None,'recovered_from_successful_cache':True})
            return scores
        if state in ("unknown","inflight"):raise RuntimeError(f"Request {state}; operator/provider reconciliation required before resubmission")
        if state=="failed":raise RuntimeError("Cached failed request; inspect durable report, no automatic duplicate submission")
        secret=os.environ.get("STATEPLAY_GPT_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not secret and self.transport is None:
            self.store.finish(key,"pending",error="pre_submission_missing_configured_credential");raise RuntimeError("Configured API credential unavailable; no network submission occurred")
        r=self.config["reward"];start=time.monotonic();budget=r["retry_budget_seconds"]
        allowed_attempts=self.store.remaining_attempts(key,r["max_retries"]+1)
        if not allowed_attempts:
            self.store.finish(key,'failed',error='authorized_attempt_limit_reached')
            raise RuntimeError('Authorized teacher request attempt limit reached')
        byte_count=len(json.dumps(request).encode());tokens=124*2200+len(json.dumps(request["messages"][0]["content"][:1]))//3+r["max_completion_tokens"]
        write_json(teacher/"capacity.json",{"payload_bytes":byte_count,"images":124,"estimated_token_rate_reservation":tokens,"protocol":"chat_completions-json_schema","idempotency_supported":r["idempotency_supported"]})
        # The server's ALL_PROXY uses socks5h, unsupported by its httpx 0.27.
        # Explicit proxy configuration is optional; direct HTTPS avoids implicit incompatible env parsing.
        proxy=os.environ.get("STATEPLAY_API_PROXY")
        try:client=httpx.Client(timeout=r["timeout_seconds"],transport=self.transport,trust_env=False,proxy=proxy)
        except (ValueError,ImportError):
            self.store.finish(key,"pending",error="pre_submission_client_configuration_error")
            raise RuntimeError("API client configuration failed before any network submission") from None
        with client:
            for attempt in range(allowed_attempts):
                self.store.rate_wait(r["rpm"],r["tpm"],tokens);submission_number=self.store.attempt(key)
                headers={"Authorization":"Bearer "+(secret or "fake"),"Content-Type":"application/json"}
                if r["idempotency_supported"]:headers["Idempotency-Key"]=key
                response=None
                try:
                    response,raw,response_is_json,transport_metadata=submit(client,r["endpoint"].rstrip("/")+"/chat/completions",headers,request,group,submission_number,streaming=os.environ.get('STATEPLAY_API_STREAM')=='1',max_elapsed=r['timeout_seconds'])
                    # Gateways can return HTML for 413/502/504. Persist the response
                    # before JSON parsing so status and diagnostics are never lost.
                    report={"timestamp":now(),"http_status":response.status_code,"provider_request_id":response.headers.get("x-request-id"),"response":raw,"response_is_json":response_is_json,"content_type":response.headers.get("content-type"),"latency_seconds":time.monotonic()-start,"attempt":submission_number}
                    report.update(transport_metadata)
                    report['cf_ray']=response.headers.get('cf-ray')
                    if not response_is_json:
                        body=response.text
                        report["response_text"]=body.replace(secret,"[REDACTED]") if secret else body
                    write_json(teacher/f"response_attempt_{submission_number:02d}.json",report)
                    write_json(teacher/"response_raw.json",report)
                    if response.status_code in (408,504,524):
                        reason=f'upstream_timeout_http_{response.status_code}'
                        self.store.finish(key,'unknown',error=reason)
                        write_json(teacher/'unknown.json',{'cache_key':key,'reason':reason,'provider_id':report['provider_request_id'],'cf_ray':report['cf_ray'],'requires_reconciliation':True,'attempt':submission_number})
                        raise RuntimeError('Upstream timed out; submission outcome unknown, no automatic retry')
                    if response.status_code in (401,403,404) or (400<=response.status_code<500 and response.status_code!=429):
                        self.store.finish(key,"failed",error=f"fatal_http_{response.status_code}");raise RuntimeError(f"Teacher endpoint rejected full group: HTTP {response.status_code}")
                    if response.status_code>=400:raise ValueError(f"retryable_http_{response.status_code}")
                    if not response_is_json:raise ValueError("non_json_success_response")
                    scores=parse_scores(raw["choices"][0]["message"]["content"])
                    self.store.finish(key,"succeeded",scores)
                    write_json(teacher/"scores.json",{"scores":scores,"mapping":mapping,"cache_key":key,"scorer":self.model_id,"usage":raw.get("usage"),"latency_seconds":time.monotonic()-start})
                    return scores
                except (httpx.ReadTimeout,httpx.WriteTimeout,httpx.RemoteProtocolError,httpx.ReadError,httpx.WriteError,SubmissionIncomplete) as e:
                    # Submission may have completed. Never silently rebill an unknown request.
                    self.store.finish(key,"unknown",error=type(e).__name__)
                    write_json(teacher/"unknown.json",{"cache_key":key,"reason":type(e).__name__,"provider_id":None,"requires_reconciliation":True})
                    raise RuntimeError("Teacher submission status unknown; automatic retry paused") from None
                except (httpx.ConnectError,httpx.ConnectTimeout,ValueError,TypeError,KeyError,IndexError) as e:
                    reason=type(e).__name__ if isinstance(e,httpx.RequestError) else (f"retryable_http_{response.status_code}" if response is not None and response.status_code>=400 else "invalid_or_retryable_response")
                    write_json(teacher/f"retry_attempt_{submission_number:02d}.json",{"timestamp":now(),"cache_key":key,"attempt":submission_number,"reason":reason,"exception_type":type(e).__name__,"http_status":response.status_code if response is not None else None,"latency_seconds":time.monotonic()-start,"retry_exhausted":attempt>=allowed_attempts-1 or time.monotonic()-start>=budget})
                    if attempt>=allowed_attempts-1 or time.monotonic()-start>=budget:
                        self.store.finish(key,"failed",error=reason);raise RuntimeError("Teacher retries exhausted; group retained") from None
                    retry_after=response.headers.get("retry-after") if response is not None else None
                    try:delay=float(retry_after) if retry_after else min(60,2**attempt)+random.random()
                    except ValueError:delay=min(60,2**attempt)
                    time.sleep(min(delay,max(0,budget-(time.monotonic()-start))))
                except Exception as e:
                    # New transport/provider exception classes must not leave
                    # the claim permanently inflight. Preserve terminal states
                    # already recorded by the classified handlers above.
                    with self.store.connect() as db:
                        current=db.execute('SELECT status FROM requests WHERE key=?',(key,)).fetchone()[0]
                    if current!='inflight':raise
                    self.store.finish(key,'unknown',error=type(e).__name__)
                    write_json(teacher/'unknown.json',{'cache_key':key,'reason':type(e).__name__,
                               'provider_id':None,'attempt':submission_number,'requires_reconciliation':True})
                    raise RuntimeError('Unclassified API failure retained for bounded authorized retry') from None

def _process_score(args):
    config,run,group,limit=args
    return APIGroupScorer(config,Path(run),limit).score(Path(group))

def _score_groups_once(config,run,groups,limit=None):
    workers=config["reward"]["workers"]
    RequestStore(api_cache_path(run,config['reward']['endpoint']))
    backend=config['reward'].get('concurrency_backend','processes')
    if backend=='threads':
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(_process_score,[(config,str(run),str(g),limit) for g in groups]))
    if backend!='processes':raise ValueError('Unknown API concurrency backend')
    # Spawn keeps the CPU request workers separate from the initialized CUDA runtime.
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(_process_score,[(config,str(run),str(g),limit) for g in groups]))

def score_groups(config,run,groups,limit=None):
    """Retry authorized API faults after all request threads have settled.

    Keep the world model resident while recovering transient scoring failures;
    cached successful labels never issue another paid request. A finite durable
    attempt ceiling, rather than the exception's spelling, controls recovery.
    """
    run=Path(run);groups=list(groups)
    while True:
        try:return _score_groups_once(config,run,groups,limit)
        except Exception:
            permission=run/'plan/API_gateway_retry_authorization.json'
            if not permission.exists() or read_json(permission).get('retry_all_api_errors') is not True:raise
            from ..formal import reopen_unscored_gateway_timeouts
            if not reopen_unscored_gateway_timeouts(run,groups=groups):raise
            store=RequestStore(api_cache_path(run,config['reward']['endpoint']))
            with store.connect() as db:
                attempt=db.execute("SELECT coalesce(max(attempts),0) FROM requests WHERE status='pending'").fetchone()[0]
            delay=min(60,2**min(attempt,6))
            from ..io import append_jsonl
            append_jsonl(run/'plan/formal_events.jsonl',{'time':now(),'event':'API_retry_without_world_reload',
                         'backoff_seconds':delay,'cumulative_attempts_preserved':True,'successful_cache_reused':True})
            time.sleep(delay)
