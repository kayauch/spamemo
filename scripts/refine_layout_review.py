"""Apply auditable scene-specific annotations to a separate candidate."""
import argparse
from copy import deepcopy
import hashlib,json
from pathlib import Path
import sys
import numpy as np
from PIL import ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from spatial_memory.layout_review import lift_edge,top_proposal,supported_edge_height
from spatial_memory.generation_evidence import transform_points
from spatial_memory.texturing import preprocess_frame

p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--source',default='prediction-fused-002');p.add_argument('--output',default='prediction-reviewed-001');a=p.parse_args()
run=a.run.resolve()
for name in (a.source,a.output):
    if not name.replace('-','').isalnum():raise ValueError('Invalid name')
out=run/a.output
if out.exists():raise FileExistsError(out)
t=json.loads((run/'transform.json').read_text());source=run/a.source/'layout.json'
layout=json.loads(source.read_text());annotations=run/'boundary-check/annotations.json';notes=json.loads(annotations.read_text())
scene=ROOT/'data/scenes-v2'/t['scene_id'];npz=scene/'reconstruction/reconstruction.npz'
if hashlib.sha256(npz.read_bytes()).hexdigest()!=t['source_npz_sha256']:raise ValueError('Evidence changed')
if hashlib.sha256((run/'spatiallm-input.ply').read_bytes()).hexdigest()!=layout['input_sha256']:raise ValueError('Point evidence changed')
with np.load(npz,allow_pickle=False) as data:
    depth=data['depth'];conf=data['confidence'];ks=data['intrinsics'];es=data['extrinsics']
if notes['coordinate_domain']!=[depth.shape[2],depth.shape[1]]:raise ValueError('Annotation image domain differs')
ids={b['id'] for b in layout['boxes']}
for note in notes['reject']+notes['retain_unverified']+notes['top_edges']:
    if note['id'] not in ids:raise ValueError('Annotation refers to an unknown object')
for edge in notes['top_edges']:
    if edge['role'] not in ('fit','check') or not 0<=edge['frame']<len(depth):raise ValueError('Invalid edge annotation')
frames=json.loads((scene/'frames/manifest.json').read_text())['frames']
out.mkdir();(out/'annotations').mkdir()
observations={};edge_report=[]
for edge in notes['top_edges']:
    i=edge['frame'];xyz=lift_edge(edge['segment'],depth[i],conf[i],ks[i],es[i],t['vg_world_to_generation_world'])
    observations.setdefault(edge['id'],[]).append(dict(frame=i,points=xyz,role=edge['role']))
    edge_report.append(dict(**edge,supported_samples=len(xyz),height_median=float(np.median(xyz[:,2])) if len(xyz) else None))
    im=preprocess_frame(scene/'frames'/frames[i]['name'],*depth.shape[1:]);draw=ImageDraw.Draw(im)
    draw.line([tuple(x) for x in edge['segment']],fill='red',width=2);im.save(out/'annotations'/f'{i:06d}.jpg')
rejected=[];boxes=[];corrections=[]
inverse=np.asarray(t['generation_world_to_vg_world']);scale=np.linalg.norm(inverse[:3,0]);rotation=inverse[:3,:3]/scale
for old in layout['boxes']:
    rejection=next((n for n in notes['reject'] if n['id']==old['id']),None)
    if rejection:
        rejected.append(dict(box=old,review=rejection));continue
    box=deepcopy(old)
    if box['id'] in observations:
        obs=observations[box['id']];fit=[o for o in obs if o['role']=='fit'];check=[o for o in obs if o['role']=='check']
        candidate,report=top_proposal(box,fit)
        report['id']=box['id']
        # Check a third photograph not used for fitting. The depth model still used all frames.
        check_z=[z for o in check if (z:=supported_edge_height(box,o['points'])) is not None]
        report['check_heights']=check_z
        if report['accepted']:
            old_error=float(np.mean(np.abs(np.array(check_z)-report['old_top']))) if check_z else None
            new_error=float(np.mean(np.abs(np.array(check_z)-report['proposed_top']))) if check_z else None
            report.update(check_error_before=old_error,check_error_after=new_error)
            if not check_z or new_error>.12 or new_error>=old_error:
                report.update(accepted=False,reason='Held-out edge does not support improvement')
            else:box=candidate
        corrections.append(report);box['photo_review']=report
    review=next((n for n in notes['retain_unverified'] if n['id']==box['id']),None)
    if review:box['photo_review']=review
    box['vg_world']=dict(center=transform_points([box['center']],inverse)[0].tolist(),rotation=(rotation@np.asarray(box['rotation'])).tolist(),scale=(np.asarray(box['scale'])*scale).tolist())
    boxes.append(box)
layout.update(boxes=boxes,rejected_proposals=rejected,photo_edge_corrections=corrections,
    display_label='実写照合＋上端補正',accepted_for_generation=False,
    annotation_provenance=dict(method=notes['method'],sha256=hashlib.sha256(annotations.read_bytes()).hexdigest(),source_prediction=a.source,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
    edge_observations=edge_report)
layout['depth_review_policy']=dict(min_confidence=1.2,min_unique_samples=20,min_distinct_fit_frames=2,
    max_fit_height_disagreement_m=.12,max_correction_m=.5,
    scale='assumed ceiling height, not measured metric scale',
    evaluation='check frame excluded from edge fitting; VGGT itself used all frames')
layout['counts']={k:sum(b['class']==c for b in boxes) for k,c in [('walls','wall'),('doors','door'),('windows','window'),('bboxes','bbox')]}
(out/'layout.json').write_text(json.dumps(layout,indent=2));(out/'annotations.json').write_text(json.dumps(notes,indent=2))
print(json.dumps(dict(rejected=[r['box']['id'] for r in rejected],edges=edge_report,corrections=corrections),indent=2))
