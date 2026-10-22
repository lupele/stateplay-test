"""Real frozen-model probes for human inspection; no invented swap labels."""
from pathlib import Path
from ..io import read_json,write_json,digest,reward_path
from ..rewards.protocol import group_content
from .model import vision_features,move,score

def swap_parts(original, alternative, kind):
    parts=[dict(p) for p in original]
    if kind=='reference':
        start=next(i for i,p in enumerate(parts) if p.get('text')=='reference_video: ordered 24 frames')
        end=next(i for i,p in enumerate(parts) if p.get('text')=='video_1: ordered 24 frames')
        a=next(i for i,p in enumerate(alternative) if p.get('text')=='reference_video: ordered 24 frames')
        b=next(i for i,p in enumerate(alternative) if p.get('text')=='video_1: ordered 24 frames')
        parts[start:end]=[dict(p) for p in alternative[a:b]]
    elif kind=='action':
        i=next(i for i,p in enumerate(parts) if p.get('text','').startswith('NPC conditioning: '))
        other=next(p['text'] for p in alternative if p.get('text','').startswith('NPC conditioning: '))
        parts[i]['text']=parts[i]['text'].split('\n',1)[0]+'\n'+other.split('\n',1)[1]
    else:raise ValueError('Unknown shortcut probe')
    return parts

def probe(run,c,results,model,processor,collator,device,version):
    path=reward_path(run)/'reports/shortcut_checks.json'
    if path.exists():
        saved=read_json(path)
        if saved['version']!=version:raise ValueError('Shortcut probe model changed')
        return saved
    valid=[r for r in results if 'scores' in r];selected={}
    for r in valid:selected.setdefault(r['mechanism_class'],r)
    probes=[]
    for cls,r in sorted(selected.items()):
        group=Path(r['group']);parts,_=group_content(group,c)
        partner=next((q for q in valid if q['group']!=r['group'] and q['mechanism_class']!=cls),None)
        if partner is None:continue
        alternative,_=group_content(Path(partner['group']),c)
        for kind in ('reference','action'):
            changed=swap_parts(parts,alternative,kind)
            values,meta=collator(group,training=False,parts_override=changed)
            cache=run/'reward_model/feature_cache'/(meta['input_hash']+'.safetensors')
            features=vision_features(model,values,cache)
            record={'kind':kind,'group':str(group),'substitution_source':partner['group'],
                    'canonical_scores':r['scores'],'input_hash':meta['input_hash'],'exact_input':changed}
            try:
                scores,raw=score(model,processor,move(values,device),features,c['reward_model']['max_new_tokens'])
                record.update(swapped_scores=scores,raw_output=raw)
            except ValueError as error:record.update(error=type(error).__name__,raw_output=getattr(error,'raw_output',None))
            probes.append(record)
            write_json(path.with_name('shortcut_probes_partial.json'),{'version':version,'probes':probes})
    out={'version':version,'probes':probes,'actual_model_inference':True,
         'passed':None,'status':'requires_actual_human_interpretation',
         'note':'A changed reference or action is not assigned a fabricated teacher score; inspect for mechanical shortcuts.'}
    write_json(path,out);return out
