"""Choose FLA kernel configurations by a stable order rather than wall time.

PyTorch's deterministic flag does not constrain third-party Triton autotuning.
Keep the upstream pruning, compile/resource checks and buffer restoration hooks;
exercise each candidate once and select the first valid upstream configuration.
No installed library file is edited.
"""
import sys
from types import MethodType

VERSION='fla-stable-valid-config-order-v1'

def pin(tuner):
    if getattr(tuner,'_stateplay_stable_tuning',False):return False
    original=tuner._bench
    def bench(self,*args,config,**kwargs):
        # Identity, not equality: upstream Config objects are the dictionary keys.
        score=next(i for i,c in enumerate(self.configs) if c is config)
        previous=self.do_bench
        def exercise(kernel_call,quantiles):
            kernel_call()
            return [float(score)]*len(quantiles)
        self.do_bench=exercise
        try:return original(*args,config=config,**kwargs)
        finally:self.do_bench=previous
    tuner._bench=MethodType(bench,tuner)
    tuner.cache.clear()
    tuner._stateplay_stable_tuning=True
    return True

def install():
    from triton.runtime.autotuner import Autotuner
    seen=set();count=0
    for name,module in list(sys.modules.items()):
        if not name.startswith('fla.') or module is None:continue
        for value in list(vars(module).values()):
            while hasattr(value,'fn') and id(value) not in seen:
                seen.add(id(value))
                if isinstance(value,Autotuner):
                    count+=int(pin(value))
                value=value.fn
    return count

def selected_configs():
    from triton.runtime.autotuner import Autotuner
    seen=set();result={}
    for name,module in list(sys.modules.items()):
        if not name.startswith('fla.') or module is None:continue
        for value in list(vars(module).values()):
            while hasattr(value,'fn') and id(value) not in seen:
                seen.add(id(value))
                if isinstance(value,Autotuner) and value.cache:
                    key=value.base_fn.__module__+'.'+value.base_fn.__name__
                    result[key]=[{'key':str(k),'config':str(v)} for k,v in sorted(value.cache.items(),key=lambda x:str(x[0]))]
                value=value.fn
    return result
