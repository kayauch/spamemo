"""Fuse separately inferred architecture/objects, anchor walls to observed planes."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from spatial_memory.generation_evidence import vertical_anchors, anchor_wall_heights, transform_points

p=argparse.ArgumentParser();p.add_argument('run',type=Path)
p.add_argument('--architecture',required=True);p.add_argument('--objects',required=True)
p.add_argument('--extra-objects');p.add_argument('--extra-category',nargs='*',default=[])
p.add_argument('--architecture-check')
p.add_argument('--allow-assumed-ceiling',action='store_true')
p.add_argument('--output',default='prediction-fused-001');a=p.parse_args();run=a.run.resolve()
for name in (a.architecture,a.objects,a.output,a.extra_objects,a.architecture_check):
    if name is None:continue
    if not name.replace('-','').isalnum():raise ValueError('Invalid artifact name')
out=run/a.output
if out.exists():raise FileExistsError(out)
arch_path=run/a.architecture/'layout.json';objects_path=run/a.objects/'layout.json'
architecture=json.loads(arch_path.read_text());objects=json.loads(objects_path.read_text())
if architecture['input_sha256']!=objects['input_sha256']:raise ValueError('Evidence differs; cannot merge')
if architecture['input_sha256']!=hashlib.sha256((run/'spatiallm-input.ply').read_bytes()).hexdigest():raise ValueError('Prepared input changed')
if any(not x['parse_complete'] or x['token_limit_reached'] for x in (architecture,objects)):raise ValueError('Incomplete inference')
transform=json.loads((run/'transform.json').read_text())
baseline=json.loads((run/'existing-structure.json').read_text())
anchors=vertical_anchors(baseline['components'],transform['vg_world_to_generation_world'],
    assumed_height_m=transform['assumed_height_m'] if a.allow_assumed_ceiling else None)
boxes=[]
for prediction,name,object_only in [(architecture,a.architecture,False),(objects,a.objects,True)]:
    for raw in prediction['boxes']:
        if (raw['class']=='bbox')!=object_only:continue
        box=deepcopy(raw);box['source_prediction']=name;boxes.append(box)
extra_sources=[]
if a.extra_objects:
    if not a.extra_category:raise ValueError('Specify observed categories for supplementary objects')
    path=run/a.extra_objects/'layout.json';extra=json.loads(path.read_text())
    if extra['input_sha256']!=architecture['input_sha256']:raise ValueError('Supplementary evidence differs')
    if not extra['parse_complete'] or extra['token_limit_reached']:raise ValueError('Incomplete supplementary inference')
    for raw in extra['boxes']:
        if raw['class']=='bbox' and raw['label'] in a.extra_category:
            box=deepcopy(raw);box['source_prediction']=a.extra_objects;box['source_id']=box['id']
            box['id']=4000+sum(b['class']=='bbox' for b in boxes);boxes.append(box)
    extra_sources.append(dict(prediction=a.extra_objects,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),selected_categories=a.extra_category))
if a.architecture_check:
    path=run/a.architecture_check/'layout.json';check=json.loads(path.read_text())
    if check['input_sha256']!=architecture['input_sha256']:raise ValueError('Crosscheck evidence differs')
    if not check['parse_complete'] or check['token_limit_reached']:raise ValueError('Incomplete crosscheck inference')
    for box in boxes:
        if box['class'] not in ('door','window'):continue
        matches=[other for other in check['boxes'] if other['class']==box['class']
            and np.linalg.norm(np.asarray(other['center'])-box['center'])<.35]
        box['crosscheck']=dict(model_agreement=bool(matches),other_prediction=a.architecture_check,
            meaning='Model agreement only; not photographic verification')
    extra_sources.append(dict(prediction=a.architecture_check,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),role='opening crosscheck'))
boxes,changes=anchor_wall_heights(boxes,anchors)
inverse=np.asarray(transform['generation_world_to_vg_world']);scale=np.linalg.norm(inverse[:3,0])
rotation=inverse[:3,:3]/scale
for box in boxes:
    box['vg_world']=dict(center=transform_points([box['center']],inverse)[0].tolist(),
        rotation=(rotation@np.asarray(box['rotation'])).tolist(),scale=(np.asarray(box['scale'])*scale).tolist())
    box['constraint']='PROPOSED; requires evidence review'
summary=dict(model='SpatialLM architecture + category-directed objects + geometric anchors',
    layer='INFERRED_LAYOUT',accepted_for_generation=False,parse_complete=True,token_limit_reached=False,
    input_sha256=architecture['input_sha256'],metric_scale_verified=False,
    sources=[dict(prediction=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for name,path in [(a.architecture,arch_path),(a.objects,objects_path)]]+extra_sources,
    vertical_anchors=anchors,wall_height_changes=changes,boxes=boxes,
    counts={label:sum(b['class']==kind for b in boxes) for label,kind in
        [('walls','wall'),('doors','door'),('windows','window'),('bboxes','bbox')]},
    limitations=['Category list was selected after viewing this room, not automatically inferred',
                 'Locker compartment count and object labels remain unverified',
                 'Openings are proposals; no generative stage is enabled'])
out.mkdir();(out/'layout.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(dict(counts=summary['counts'],vertical_anchors=anchors,wall_corrections=len(changes)),indent=2))
