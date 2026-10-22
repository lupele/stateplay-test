"""Descriptive A trial summary plus an explicit manual mechanics review queue."""
import argparse
from collections import Counter
import numpy as np
from .io import run_path,read_json,write_json,teacher_path

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();run=run_path(a.run_id)
    groups=[];review=[];scores=[];kept=0
    for path in sorted((run/'plan').glob('batch-*-normalization.json')):
        d=read_json(path);points=[[v['video_'+str(i)] for i in range(1,5)] for v in d['scores']];scores.extend(points);kept+=sum(d['normalization']['keep'])
    for path in sorted((run/'groups').glob('*/manifest.json')):
        m=read_json(path);t=teacher_path(path.parent)/'scores.json'
        if not t.exists():continue
        groups.append({'group_id':m['group_id'],'condition_id':m['condition_id'],'class':m['condition']['mechanism_class']})
        for c in m['candidates']:review.append({'group_id':m['group_id'],'video_id':c['video_id'],'video':str(path.parent/'candidates'/c['video_id']/'video.mp4'),'reference':str(path.parent/'reference.mp4'),'action_timeline':str(teacher_path(path.parent)/'request_map.json'),'review_fields':['visible_D_attempt_time','actual_super_execution','meter_sufficiency_and_consumption','hit_block_whiff_interruption','terminal_result','severe_mechanics_violation']})
    if not scores:raise ValueError('Trial has no actual scores yet; this tool does not invent a result')
    array=np.asarray(scores);report={'kind':'actual_API_A_trial_descriptive_report','groups':len(scores),'class_counts':dict(Counter(g['class'] for g in groups)),'mean':float(array.mean()),'std_population':float(array.std()),'quantiles':np.quantile(array,[0,.1,.25,.5,.75,.9,1]).tolist(),'kept_groups':int(kept),'filtered_fraction':1-kept/len(scores),'completion':read_json(run/'reports/api_completion.json'),'mechanics_review':'Pending actual independent/manual review; scalar API scores are not mechanics-event measurements','manual_review_queue':review}
    write_json(run/'reports/trial_140_summary.json',report);print({'groups':len(scores),'filtered_fraction':report['filtered_fraction'],'review_candidates':len(review)})
if __name__=='__main__':main()
