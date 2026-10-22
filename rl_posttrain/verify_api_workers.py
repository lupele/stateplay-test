"""32 spawned CPU processes against a local fake teacher. No paid API requests."""
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import threading
import time
from pathlib import Path
from .config import load_config
from .io import run_path,write_json,sha256,digest
from .rewards.api_group import score_groups
from .rewards.base import SLOTS

def main():
    p=argparse.ArgumentParser();p.add_argument("--run-id",default="smoke_v1");p.add_argument('--backend',choices=['processes','threads'],default='processes');a=p.parse_args()
    c=load_config("rl_posttrain/configs/api_b_only.yaml");run=run_path(a.run_id);root=run/"reports/fake_api_stress"/str(time.time_ns())
    from rl_posttrain.tests.conftest import fake_group
    groups=[fake_group(root,c,i) for i in range(32)]
    mutex=threading.Lock();active=0;peak=0;requests=0
    barrier=threading.Barrier(32)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            nonlocal active,peak,requests
            body=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            images=sum(p["type"]=="image_url" for p in body["messages"][0]["content"])
            if images!=124:raise ValueError("Incorrect fake stress payload")
            with mutex:active+=1;peak=max(peak,active);requests+=1
            if a.backend=='threads':barrier.wait(timeout=60)
            else:time.sleep(2)
            data=json.dumps({"choices":[{"message":{"content":json.dumps(dict(zip(SLOTS,[25,50,75,100])))}}]}).encode()
            self.send_response(200);self.send_header("Content-Type","application/json");self.send_header("Content-Length",str(len(data)));self.end_headers();self.wfile.write(data)
            with mutex:active-=1
    class Server(ThreadingHTTPServer):request_queue_size=128
    server=Server(("127.0.0.1",0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    c["reward"].update(endpoint=f"http://127.0.0.1:{server.server_port}",workers=32,concurrency_backend=a.backend,rpm=100000,tpm=1000000000,timeout_seconds=60)
    previous=os.environ.get("STATEPLAY_GPT_API_KEY");os.environ["STATEPLAY_GPT_API_KEY"]="software-test-placeholder"
    start=time.monotonic()
    try:
        values=score_groups(c,root,groups,32)
        # Repeat from persisted SQLite cache; zero server calls are added.
        again=score_groups(c,root,groups,32)
        assert values==again and requests==32
        report={"kind":"mock_local_API_server","workers":32,"backend":"threads" if a.backend=='threads' else "spawned_processes","requests":requests,"peak_server_inflight":peak,"groups":32,"images_per_request":124,"seconds":time.monotonic()-start,"cached_repeat_added_requests":0,"paid_API_requests":0}
        write_json(run/("reports/api_32_thread_stress.json" if a.backend=='threads' else "reports/api_32_process_stress.json"),report);print(report,flush=True)
    finally:
        server.shutdown()
        if previous is None:os.environ.pop("STATEPLAY_GPT_API_KEY",None)
        else:os.environ["STATEPLAY_GPT_API_KEY"]=previous
if __name__=="__main__":main()
