import numpy as np
from rl_posttrain.independent_eval import movement,attack

def test_paper_movement_rules():
    t=np.linspace(0,1,41);left=np.stack((.5-.04*t,.5+t*0,.3+t*0),1)
    assert movement(left,"LEFT");assert not movement(left,"RIGHT")
    jump=np.stack((.5+t*0,.5-.05*np.sin(t*np.pi),.3+t*0),1);assert movement(jump,"UP")

def test_attack_threshold_is_strict():
    keys=list("ABCXYZ");p=np.zeros((41,6));p[10,3]=.7;assert attack(p,keys)=="noop"
    p[10,3]=.71;assert attack(p,keys)=="X"
