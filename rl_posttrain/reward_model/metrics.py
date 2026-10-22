"""Held-out four-score metrics; an invalid prediction is never assigned zero."""
import numpy as np
from scipy.stats import spearmanr
from ..rewards.base import SLOTS
from ..rewards.monitor import audit_metrics

def canonical_result(result):
    """Keep canonical four-score evidence; retain old diagnostics in old files."""
    return {k:v for k,v in result.items() if not k.startswith('permutation')}

def canonical_identity(version,item):
    from ..io import digest
    return digest({'version':version,'group':item,'evaluation_schema':'canonical-four-score-v2'})

def load_cached_canonical(reports,item,version):
    """Reuse validated canonical replies without rerunning legacy extra inputs."""
    from ..io import read_json,write_json,digest
    current=reports/'canonical_groups'/(item['group_id']+'.json')
    identity=canonical_identity(version,item)
    if current.exists():
        result=read_json(current)
        if result.get('evaluation_identity')!=identity:raise ValueError('Cached reward evaluation identity changed')
        return canonical_result(result)
    previous=reports/'groups'/(item['group_id']+'.json')
    if not previous.exists():return None
    result=read_json(previous)
    previous_identities={digest({'version':version,'group':item,'permutation':flag}) for flag in (False,True)}
    if result.get('evaluation_identity') not in previous_identities:raise ValueError('Cached reward evaluation identity changed')
    result=canonical_result(result);result['evaluation_identity']=identity
    write_json(current,result)
    return result

def summarize(results, version, smoke=False):
    from ..quality_gate import QUALITY_STANDARD
    results=[canonical_result(row) for row in results]
    valid=[r for r in results if 'scores' in r]
    out={'quality_standard':dict(QUALITY_STANDARD),'version':version,'smoke':smoke,'groups':len(results),
         'failed':len(results)-len(valid),'json_valid_rate':len(valid)/len(results) if results else 0,
         'results':results}
    if not valid:return out
    t=np.asarray([[r['teacher'][k] for k in SLOTS] for r in valid],dtype=float)
    s=np.asarray([[r['scores'][k] for k in SLOTS] for r in valid],dtype=float)
    classes=np.asarray([r['mechanism_class'] for r in valid])
    out.update(audit_metrics(t,s))
    out['rmse']=float(np.sqrt(np.mean((t-s)**2)))
    corr=spearmanr(t.ravel(),s.ravel()).statistic
    out['spearman_global']=float(corr) if np.isfinite(corr) else None
    out['top_choice_accuracy']=float(np.mean(np.argmax(t,1)==np.argmax(s,1)))
    out['per_class']={cls:{'mae':float(np.abs(t[classes==cls]-s[classes==cls]).mean()),
                                'bias':float((s[classes==cls]-t[classes==cls]).mean()),
                                'groups':int(sum(classes==cls))} for cls in sorted(set(classes))}
    correlations=[spearmanr(a,b).statistic for a,b in zip(t,s)]
    finite=[float(v) for v in correlations if np.isfinite(v)]
    out['spearman_per_group']=float(np.mean(finite)) if finite else None
    baseline=np.full_like(t,t.mean())
    out['constant_mean_baseline_mae']=float(np.abs(t-baseline).mean())
    out['mean_baseline_estimated_on']='reported holdout only; never used for tuning or training'
    rng=np.random.default_rng(20261013)
    errors=np.abs(t-s).mean(1)
    boot=np.asarray([errors[rng.integers(len(errors),size=len(errors))].mean() for _ in range(1000)])
    out['mae_95pct_group_bootstrap_ci']=np.quantile(boot,[.025,.975]).tolist()
    return out
