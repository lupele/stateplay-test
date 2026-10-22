"""Formal RM quality and manually reviewed evidence, separate from software acceptance."""
from pathlib import Path
from .io import read_json,write_json,digest,sha256,reward_path

QUALITY_STANDARD = {
    'schema': 'stateplay-rm-quality-v2',
    'json_valid_rate_min': .99,
    'mae_max': 15,
    'spearman_global_min': .70,
    'pair_agreement_min': .75,
    'per_class_mae_max': 15,
    'manual_candidates_min': 100,
}

def numeric_reasons(metrics):
    """Quality failures that additional teacher data can address; no manual claims."""
    reasons=[]
    if metrics.get('smoke'):reasons.append('Smoke dataset cannot establish reward generalization')
    if metrics.get('json_valid_rate',0)<.99:reasons.append('JSON validity below 99%')
    if metrics.get('mae',float('inf'))>QUALITY_STANDARD['mae_max']:reasons.append('MAE exceeds 15 points')
    if metrics.get('spearman_global') is None or metrics['spearman_global']<.7:reasons.append('Global Spearman below 0.7 or undefined')
    if metrics.get('pair_agreement') is None or metrics['pair_agreement']<.75:reasons.append('Valid-pair agreement below 75% or undefined')
    for cls in ('normal','macro_success','macro_fail','result_win','result_lose'):
        if cls not in metrics.get('per_class',{}) or metrics['per_class'][cls]['mae']>15:reasons.append('Class MAE missing or exceeds 15: '+cls)
    return reasons

def evaluate_gate(run,metrics,version,review_path=None):
    numeric=numeric_reasons(metrics)
    reasons=list(numeric)
    review=read_json(review_path) if review_path else {}
    if review.get("version")!=version:reasons.append("Human review is missing or belongs to a different model version")
    if len(set(review.get("reviewed_candidate_ids",[])))<100:reasons.append("At least 100 distinct representative manual candidate checks are required")
    split_path=reward_path(run)/'splits/manifest.json'
    candidates={}
    if split_path.exists():
        for group in read_json(split_path)['groups']:
            m=read_json(Path(group['group_path'])/'manifest.json')
            for c in m['candidates']:
                canonical=m['group_id']+'/'+c['video_id'];candidates[c['video_id']]=canonical;candidates[canonical]=canonical
    actual_ids={candidates[x] for x in review.get('reviewed_candidate_ids',[]) if x in candidates}
    if len(actual_ids)<100:reasons.append('Human review must identify >=100 actual distinct dataset candidates')
    if not review.get("source_clusters_reviewed"):reasons.append("Episode/cluster leakage review is required")
    audit=run/'conditions/leakage_index.json'
    if not audit.exists() or review.get('source_audit_sha256')!=sha256(audit):reasons.append('Human source review must match the current source/action/temporal audit')
    if not review.get("reference_and_action_swap_checks_passed"):reasons.append("Reference/action shortcut checks require review")
    if review.get("severe_mechanics_violations"):reasons.append("Severe reviewed mechanism violation")
    report={"passed":not reasons,"numeric_passed":not numeric,"quality_standard":dict(QUALITY_STANDARD),"software_only":bool(metrics.get("smoke")),"version":version,"reasons":reasons,"review_path":str(review_path) if review_path else None}
    write_json(reward_path(run)/"reports/quality_gate.json",report);return report
