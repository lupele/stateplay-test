"""Permanent real review materials. No checkbox is checked by the program."""
import collections
import html
import json
import os
from pathlib import Path
from .io import teacher_path,read_json,write_json,sha256,reward_path,reward_checkpoint

def build_review(run):
    rm=reward_path(run);manifest=read_json(reward_checkpoint(run)/'manifest.json')
    evaluation=read_json(rm/'reports/evaluation.json')
    buckets=collections.defaultdict(list)
    for result in evaluation['results']:
        if 'scores' not in result:continue
        group=Path(result['group']);m=read_json(group/'manifest.json')
        for c in m['candidates']:
            score=result['scores'][c['slot']]
            item={'candidate_id':m['group_id']+'/'+c['video_id'],'group_id':m['group_id'],
                  'slot':c['slot'],'class':m['condition']['mechanism_class'],
                  'teacher_score':result['teacher'][c['slot']],'student_score':score,
                  'video':str(group/'candidates'/c['video_id']/'video.mp4'),
                  'reference':str(group/'reference.mp4'),'context':str(teacher_path(group)/'request_map.json'),
                  'condition':str(group/'condition.json')}
            buckets[(item['class'],int(score)//25)].append(item)
    rows=[]
    while buckets and len(rows)<100:
        for key in sorted(list(buckets)):
            rows.append(buckets[key].pop(0))
            if not buckets[key]:del buckets[key]
            if len(rows)==100:break
    root=rm/'review';root.mkdir(parents=True,exist_ok=True)
    packet={'status':'pending_real_human_review','version':manifest['version'],
            'source_audit_sha256':sha256(run/'conditions/leakage_index.json'),
            'source_audit':str(run/'conditions/leakage_index.json'),
            'episode_independence_proven':False,'candidates':rows,
            'shortcut_checks':str(rm/'reports/shortcut_checks.json'),
            'required_output':str(root/'human_review.json')}
    write_json(root/'packet.json',packet)
    content=['<!doctype html><meta charset="utf-8"><title>StatePlay actual RM review</title>',
             '<style>body{font:16px sans-serif;margin:20px}video{width:416px}td{padding:10px;vertical-align:top}textarea{width:100%}</style>',
             '<h1>StatePlay: actual human review</h1>',
             '<p>Review each candidate with its actual reference and action timeline. Record severe mechanical errors. ',
             'Episode IDs are absent: source clustering cannot prove episode independence. Do not certify an unreviewed claim.</p>',
             '<p><a href="../reports/shortcut_checks.json">Actual reference/action swap outputs</a> · ',
             '<a href="../../../../conditions/leakage_index.json">Source/action/temporal audit</a></p>',
             '<table><tr><th>Candidate and reference</th><th>Recorded evidence</th></tr>']
    for i,row in enumerate(rows):
        video=html.escape(os.path.relpath(row['video'],root),quote=True)
        reference=html.escape(os.path.relpath(row['reference'],root),quote=True)
        context=html.escape(os.path.relpath(row['context'],root),quote=True)
        content.append(f'<tr><td>{html.escape(row["candidate_id"])}<br>{html.escape(row["class"])}; teacher={row["teacher_score"]}; student={row["student_score"]}<br><video controls preload="none" src="{video}"></video><br><video controls preload="none" src="{reference}"></video><br><a href="{context}">Action timeline and exact teacher input</a></td><td><label><input type="checkbox" id="reviewed{i}">Actually reviewed</label><br><label><input type="checkbox" id="severe{i}">Severe mechanics violation</label><br><textarea id="note{i}" placeholder="D attempt / execution / meter / hit-block-whiff / result evidence"></textarea></td></tr>')
    content += ['</table><p><label><input type="checkbox" id="source">I reviewed the source clusters and the limitation from missing episode IDs.</label></p>',
                '<p><label><input type="checkbox" id="swaps">I reviewed the actual reference/action swap probes and found no material shortcut.</label></p>',
                '<p>Reviewer: <input id="reviewer"></p><button onclick="downloadReview()">Download human_review.json</button>']
    encoded=json.dumps(packet,ensure_ascii=False).replace('<','\\u003c')
    content.append('<script>const packet='+encoded+';function downloadReview(){const ids=[],bad=[],notes={};packet.candidates.forEach((r,i)=>{if(document.getElementById("reviewed"+i).checked){ids.push(r.candidate_id);notes[r.candidate_id]=document.getElementById("note"+i).value;if(document.getElementById("severe"+i).checked)bad.push(r.candidate_id)}});const output={version:packet.version,source_audit_sha256:packet.source_audit_sha256,reviewed_candidate_ids:ids,severe_mechanics_violations:bad,source_clusters_reviewed:document.getElementById("source").checked,reference_and_action_swap_checks_passed:document.getElementById("swaps").checked,reviewer:document.getElementById("reviewer").value,notes,reviewed_at:new Date().toISOString()};const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(output,null,2)],{type:"application/json"}));a.download="human_review.json";a.click()}</script>')
    (root/'index.html').write_text('\n'.join(content))
    return root/'human_review.json'
