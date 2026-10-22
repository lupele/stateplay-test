import collections
import random
from ..io import lock, read_json, write_json, digest

def stratified_order(records, seed):
    """Proportional deficit interleaving; every real record appears exactly once."""
    rng = random.Random(seed); bins = collections.defaultdict(list)
    for record in records: bins[record["mechanism_class"]].append(record["condition_id"])
    for items in bins.values(): rng.shuffle(items)
    sizes = {k: len(v) for k, v in bins.items()}; used = collections.Counter(); total = sum(sizes.values())
    order = []
    for position in range(total):
        key = max((k for k in bins if bins[k]), key=lambda k: ((position + 1) * sizes[k] / total - used[k], k))
        order.append(bins[key].pop()); used[key] += 1
    return order

class Ledger:
    """Reservation remains in-flight until its optimizer checkpoint is committed."""
    def __init__(self, root, records=None, seed=20261013):
        self.root = root; root.mkdir(parents=True, exist_ok=True)
        self.path = root / "cursor.json"; self.order_path = root / "epoch_order.json"
        with lock(root / ".ledger.lock"):
            if not self.order_path.exists():
                if records is None: raise FileNotFoundError("Initialize plan with eligible records")
                order = stratified_order(records, seed)
                write_json(self.order_path, {"schema": "rl-plan-v1", "seed": seed, "order": order, "order_hash": digest(order)})
                write_json(self.path, {"cursor": 0, "inflight": None, "api_consumed": 0, "local_consumed": 0, "batch": 0})
        self.order = read_json(self.order_path)["order"]
    def state(self): return read_json(self.path)
    def reserve(self, n, stage, old_version):
        with lock(self.root / ".ledger.lock"):
            s = self.state()
            if s["inflight"]:
                if s["inflight"]["stage"] != stage: raise RuntimeError("Another stage has an unfinished batch")
                return s["inflight"]
            ids = self.order[s["cursor"]:s["cursor"] + n]
            if not ids: return None
            s["inflight"] = {"batch": s["batch"], "stage": stage, "condition_ids": ids, "old_version": old_version, "phase": "reserved", "update_cursor": 0, "global_condition_offset": s["cursor"]}
            write_json(self.path, s); return s["inflight"]
    def phase(self, phase, **fields):
        with lock(self.root / ".ledger.lock"):
            s = self.state()
            if not s["inflight"]: raise RuntimeError("No reservation")
            s["inflight"].update(phase=phase, **fields); write_json(self.path, s)
    def commit(self, checkpoint):
        with lock(self.root / ".ledger.lock"):
            s = self.state(); b = s["inflight"]
            if not b: return
            s["cursor"] += len(b["condition_ids"]); s[b["stage"] + "_consumed"] += len(b["condition_ids"])
            s[b["stage"]+"_batches"]=s.get(b["stage"]+"_batches",0)+b.get('rollout_batch_count',1)
            s["batch"] += 1; s["last_checkpoint"] = str(checkpoint); s["inflight"] = None; write_json(self.path, s)
