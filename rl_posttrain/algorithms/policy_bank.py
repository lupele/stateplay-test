from contextlib import contextmanager
import torch

class PolicyBank:
    """One shared frozen backbone; only trainable tensors have old/ref snapshots."""
    def __init__(self, model):
        self.all_params=dict(model.named_parameters())
        self.params={n:p for n,p in model.named_parameters() if p.requires_grad}
        self.old={n:p.detach().cpu().clone() for n,p in self.params.items()}
        self.ref={n:p.clone() for n,p in self.old.items()}; self.version=0; self.graph_active=False
        self.old_frozen={}; self.ref_frozen={}
    @contextmanager
    def use(self, policy):
        if self.graph_active: raise RuntimeError("Cannot switch policy before checkpoint recomputation/backward finishes")
        values=self.old if policy=="old" else self.ref if policy=="ref" else None
        if values is None: raise ValueError(policy)
        frozen=self.old_frozen if policy=='old' else self.ref_frozen
        values={**values,**frozen}
        params={n:self.all_params[n] for n in values}
        current={n:p.detach().clone() for n,p in params.items()}
        with torch.no_grad():
            for n,p in params.items(): p.copy_(values[n].to(p.device))
        try: yield
        finally:
            with torch.no_grad():
                for n,p in params.items(): p.copy_(current[n])
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
        # B can be frozen after A while its inherited EMA still differs from live.
        # Preserve the same full-policy EMA; the original reference stays fixed.
        for n,tensor in self.old_frozen.items():
            tensor.mul_(eta).add_(self.all_params[n].detach().cpu(),alpha=1-eta)
        self.version+=1; return eta
    def state_dict(self): return {"old":self.old,"ref":self.ref,"version":self.version,
                                  'old_frozen':self.old_frozen,'ref_frozen':self.ref_frozen}
    def load_state_dict(self,state):
        if set(state["old"])!=set(self.params) or set(state['ref'])!=set(self.params): raise ValueError("Policy trainable names changed")
        old_frozen=state.get('old_frozen',{});ref_frozen=state.get('ref_frozen',{})
        if set(old_frozen)!=set(ref_frozen):raise ValueError('Frozen old/ref policy names differ')
        for n in old_frozen:
            if n not in self.all_params or self.all_params[n].requires_grad:
                raise ValueError('Policy frozen override targets a trainable/missing parameter')
            if old_frozen[n].shape!=self.all_params[n].shape or ref_frozen[n].shape!=self.all_params[n].shape:
                raise ValueError('Frozen policy shape mismatch')
        self.old=state["old"];self.ref=state["ref"];self.version=state["version"]
        self.old_frozen=old_frozen;self.ref_frozen=ref_frozen
