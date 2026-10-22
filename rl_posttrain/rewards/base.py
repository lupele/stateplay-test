from abc import ABC,abstractmethod
import json
import math

SLOTS=tuple(f"video_{i}" for i in range(1,5))
SCORE_SCHEMA={"type":"object","properties":{k:{"type":"number","minimum":0,"maximum":100} for k in SLOTS},"required":list(SLOTS),"additionalProperties":False}

def parse_scores(value):
    if isinstance(value,str):
        def bad_constant(x):raise ValueError("Non-finite JSON number")
        def pairs(items):
            result={}
            for key,v in items:
                if key in result:raise ValueError("Duplicate JSON key")
                result[key]=v
            return result
        value=json.loads(value,parse_constant=bad_constant,object_pairs_hook=pairs)
    if not isinstance(value,dict) or set(value)!=set(SLOTS):raise ValueError("Scorer must return exactly video_1..video_4")
    for score in value.values():
        if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=100:raise ValueError("Scores must be finite numbers in [0,100]")
    return {slot:float(value[slot]) for slot in SLOTS}

class GroupScorer(ABC):
    @abstractmethod
    def score(self,group):pass

class FakeGroupScorer(GroupScorer):
    """Software tests only; deliberately labeled and never used for RM quality."""
    model_id="fake-software-test"
    def score(self,group):return dict(zip(SLOTS,[25.,50.,75.,100.]))
