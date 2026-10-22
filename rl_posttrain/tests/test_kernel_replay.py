from rl_posttrain.reward_model.kernel_replay import pin

def test_kernel_choice_preserves_checks_and_ignores_timing():
    class Tuner:
        def __init__(self):
            self.configs=[object(),object(),object()];self.cache={'old':'choice'}
            self.calls=[];self.do_bench=lambda call,quantiles:[99.,50.,2.]
        def _bench(self,*,config):
            def call():self.calls.append(config)
            if config is self.configs[0]:return [float('inf')]*3
            return self.do_bench(call,quantiles=(.5,.2,.8))
    tuner=Tuner();old=tuner.do_bench
    assert pin(tuner) and not pin(tuner) and tuner.cache=={}
    scores={c:tuner._bench(config=c) for c in tuner.configs}
    assert min(scores,key=scores.get) is tuner.configs[1]
    assert tuner.calls==tuner.configs[1:]
    assert tuner.do_bench is old
