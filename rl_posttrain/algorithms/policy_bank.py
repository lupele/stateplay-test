from contextlib import contextmanager
import torch

class PolicyBank:
    """One shared frozen backbone; only trainable tensors have old/ref snapshots."""
    def __init__(self, model):
        self.params={n:p for n,p in model.named_parameters() if p.requires_grad}
        self.old={n:p.detach().cpu().clone() for n,p in self.params.items()}
        self.ref={n:p.clone() for n,p in self.old.items()}; self.version=0; self.graph_active=False
    @contextmanager
    def use(self, policy):
        if self.graph_active: raise RuntimeError("Cannot switch policy before checkpoint recomputation/backward finishes")
        values=self.old if policy=="old" else self.ref if policy=="ref" else None
        if values is None: raise ValueError(policy)
        current={n:p.detach().clone() for n,p in self.params.items()}
        with torch.no_grad():
            for n,p in self.params.items(): p.copy_(values[n].to(p.device))
        try: yield
        finally:
            with torch.no_grad():
                for n,p in self.params.items(): p.copy_(current[n])
    @contextmanager
    def live_graph(self):
        if self.graph_active: raise RuntimeError("Nested live graph")
        self.graph_active=True
        try: yield
        finally: self.graph_active=False
    @torch.no_grad()
    def update_old(self, optimizer_step):
        if self.graph_active: raise RuntimeError("Live backward not finished")
        eta=min(.001*optimizer_step,.5)
        for n,p in self.params.items(): self.old[n].mul_(eta).add_(p.detach().cpu(),alpha=1-eta)
        self.version+=1; return eta
    def state_dict(self): return {"old":self.old,"ref":self.ref,"version":self.version}
    def load_state_dict(self,state):
        if set(state["old"])!=set(self.params): raise ValueError("Policy trainable names changed")
        self.old=state["old"];self.ref=state["ref"];self.version=state["version"]
