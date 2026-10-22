"""Gate the saved first 56 C conditions before any local-reward NFT update."""
from pathlib import Path
from ..io import read_json,write_json,reward_checkpoint,teacher_path
from ..reward_model.metrics import summarize
from .base import SLOTS
from ..quality_gate import QUALITY_STANDARD

def assess(run,config,groups,teacher,student,reward_version):
    rows=[];review=[]
    for group,t,s in zip(groups,teacher,student):
        m=read_json(group/'manifest.json')
        rows.append({'group':str(group),'group_id':m['group_id'],'teacher':t,'scores':s,
                     'mechanism_class':m['condition']['mechanism_class']})
        for candidate in m['candidates']:
            review.append({'candidate_id':m['group_id']+'/'+candidate['video_id'],
                           'video':str(group/'candidates'/candidate['video_id']/'video.mp4'),
                           'reference':str(group/'reference.mp4'),
                           'context':str(teacher_path(group)/'request_map.json')})
    metrics=summarize(rows,reward_version)
    reasons=[]
    if len(rows)!=config['reward_model']['audit_initial_conditions']:reasons.append('Initial audit buffer is incomplete')
    if metrics.get('json_valid_rate',0)<.99:reasons.append('Initial audit JSON validity below 99%')
    if metrics.get('mae',float('inf'))>QUALITY_STANDARD['mae_max']:reasons.append('Initial audit MAE exceeds 15')
    if metrics.get('spearman_global') is None or metrics['spearman_global']<.7:reasons.append('Initial audit Spearman below 0.7')
    if metrics.get('pair_agreement') is None or metrics['pair_agreement']<.75:reasons.append('Initial audit pair agreement below 75%')
    for cls in ('normal','macro_success','macro_fail','result_win','result_lose'):
        if cls not in metrics.get('per_class',{}) or metrics['per_class'][cls]['mae']>15:reasons.append('Initial audit class MAE missing/exceeds 15: '+cls)
    root=run/'reports/C_initial_audit';root.mkdir(parents=True,exist_ok=True)
    write_json(root/'review_packet.json',{'version':reward_version,'status':'pending_real_manual_review',
                                        'candidates':review,'required_output':str(root/'human_review.json')})
    review_path=root/'human_review.json';human=read_json(review_path) if review_path.exists() else {}
    actual={r['candidate_id'] for r in review}
    reviewed={x.split('/')[0] for x in human.get('reviewed_candidate_ids',[]) if x in actual}
    manual_passed=(human.get('version')==reward_version and reviewed=={r['group_id'] for r in rows}
                   and bool(human.get('reviewer')) and not human.get('severe_mechanics_violations'))
    status='failed_numeric_quality' if reasons else 'passed' if manual_passed else 'awaiting_human_review'
    report={'quality_standard':dict(QUALITY_STANDARD),'status':status,'passed':not reasons and manual_passed,'numeric_reasons':reasons,
            'metrics':metrics,'reward_version':reward_version,'manual_review_passed':manual_passed,
            'review_path':str(review_path),'ordinary_conditions_per_batch':config['rollout']['conditions_per_batch'],
            'audit_buffer_conditions':len(rows),'C_optimizer_updates_before_gate':0,
            'group_ids':[r['group_id'] for r in rows],'frozen_model':str(reward_checkpoint(run))}
    write_json(root/'gate.json',report);return report
