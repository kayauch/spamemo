"""Export an independent evidence/layout viewer. Never publishes to a scene."""
import argparse
import hashlib
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from PIL import ImageDraw
import cv2
import open3d as o3d
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from spatial_memory.generation_evidence import transform_points, box_evidence
from spatial_memory.texturing import preprocess_frame


def photo_overlays(scene, folder, refs, boxes):
    """Project inferred boxes back with ORIGINAL VGGT cameras, not thumbnail K."""
    with np.load(scene/'reconstruction/reconstruction.npz',allow_pickle=False) as src:
        height,width=src['depth'].shape[1:3]
    edges=[(i,j) for i in range(8) for j in range(i+1,8) if (i^j) in (1,2,4)]
    signs=np.array([[1 if i & (1<<j) else -1 for j in range(3)] for i in range(8)])
    metrics=json.loads((scene/'reconstruction/metrics.json').read_text(encoding='utf-8'))
    for ref in refs:
        im=preprocess_frame(scene/'frames'/ref['source_name'],height,width,
                            preprocessing='aligned_rgbd' if metrics.get('model')=='calibrated-rgbd' else 'vggt')
        im.save(folder/f"{ref['frame']:06d}-crop.jpg",quality=95)
        draw=ImageDraw.Draw(im)
        e=np.asarray(ref['camera_extrinsic_vggt']);k=np.asarray(ref['intrinsic'])
        for box in boxes:
            if not box['evidence']['valid'] or box.get('hidden'):continue
            b=box['vg_world']
            corners=(signs*np.asarray(b['scale'])/2)@np.asarray(b['rotation']).T+b['center']
            camera=corners@e[:,:3].T+e[:,3]
            color={'wall':'#67b5ff','window':'#64ead2','door':'#ffb961'}.get(box['label'],'#ef85df')
            for start,end in edges:
                a,c=camera[start].copy(),camera[end].copy()
                if max(a[2],c[2])<=.001:continue
                if a[2]<.001:a=a+(.001-a[2])/(c[2]-a[2])*(c-a)
                if c[2]<.001:c=c+(.001-c[2])/(a[2]-c[2])*(a-c)
                uv=np.stack((a,c))@k.T;uv=uv[:,:2]/uv[:,2:3]
                if not np.isfinite(uv).all():continue
                xy=np.rint(np.clip(uv,-1e7,1e7)).astype(int)
                visible,one,two=cv2.clipLine((0,0,width,height),tuple(xy[0]),tuple(xy[1]))
                if visible:draw.line([one,two],fill=color,width=2)
        im.save(folder/f"{ref['frame']:06d}-overlay.jpg",quality=95)
        ref['overlay']=f"{ref['frame']:06d}-overlay.jpg"
        ref['processed_photo']=f"{ref['frame']:06d}-crop.jpg"
        ref['overlay_note']='Unoccluded wireframe for alignment diagnosis; not a generated image'


def diagnostics(prediction, run):
    xyz=np.asarray(o3d.io.read_point_cloud(str(run/'spatiallm-input.ply')).points)
    tree=cKDTree(xyz)
    values=np.linspace(-.5,.5,16)
    grid=np.stack(np.meshgrid(values,values),axis=-1).reshape(-1,2)
    for box in prediction['boxes']:
        box.setdefault('evidence',{}).update(box_evidence(box['center'],box['rotation'],box['scale'],xyz,
            allow_planar=box['class'] in ('wall','door','window')))
        if not box['evidence']['valid']:continue
        # Surface support, not object classification confidence or completeness.
        faces=[]
        for axis in range(3):
            other=[i for i in range(3) if i!=axis]
            if any(box['scale'][i]==0 for i in other):continue
            for side in [-.5,.5]:
                face=np.zeros((len(grid),3));face[:,axis]=side;face[:,other]=grid
                faces.append(face*np.asarray(box['scale']))
        samples=np.concatenate(faces)@np.asarray(box['rotation']).T+box['center']
        distance,_=tree.query(samples)
        box['evidence']['sampled_surface_support_fraction_10cm']=float((distance<=.1).mean())
        box['evidence']['sampled_surface_distance_median_m']=float(np.median(distance))
        box['evidence']['scale_warning']='Distances depend on assumed room height'


def structured_layout(prediction, transform, refs):
    result=dict(schema_version=1,layer='INFERRED_LAYOUT',status='review_required',
        scene_id=transform['scene_id'],coordinate_system='generation_world',
        metric_scale_verified=False,transform=transform,room={'height_assumption_m':transform['assumed_height_m']},
        walls=[],openings=[],objects=[],generated_world=None)
    for box in prediction['boxes']:
        if box.get('hidden'):continue
        architectural=box['class'] in ('wall','door','window')
        item=dict(id=f"{box['class']}_{box['id']}",category=box['label'],
            position=box['center'],orientation=box['rotation'],extent=box['scale'],
            vg_world=box['vg_world'],evidence=box['evidence'],
            source_prediction=box.get('source_prediction'),crosscheck=box.get('crosscheck'),
            photo_review=box.get('photo_review'),
            proposed_constraint='HARD' if architectural else 'SOFT',constraint_active=False,
            appearance={'state':'unresolved','reference_frames':[r['frame'] for r in refs]})
        key='walls' if box['class']=='wall' else 'openings' if architectural else 'objects'
        result[key].append(item)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path)
    p.add_argument('--prediction',default='prediction-001')
    args=p.parse_args();run=args.run.resolve()
    transform=json.loads((run/'transform.json').read_text(encoding='utf-8'))
    baseline=json.loads((run/'existing-structure.json').read_text(encoding='utf-8'))
    if not args.prediction.replace('-', '').isalnum():raise ValueError('Invalid prediction name')
    pred=run/args.prediction/'layout.json'
    prediction=json.loads(pred.read_text(encoding='utf-8')) if pred.exists() else None
    if prediction and prediction['input_sha256']!=hashlib.sha256((run/'spatiallm-input.ply').read_bytes()).hexdigest():
        raise ValueError('Prepared input changed after inference')
    if prediction:
        # Human edits invalidate previously duplicated coordinates. Recompute
        # them in the comparison copy from the authoritative room-space box.
        inverse=np.asarray(transform['generation_world_to_vg_world'])
        scale=np.linalg.norm(inverse[:3,0]);rotation=inverse[:3,:3]/scale
        for box in prediction['boxes']:
            box['vg_world']=dict(center=(inverse[:3,:3]@np.asarray(box['center'])+inverse[:3,3]).tolist(),
                rotation=(rotation@np.asarray(box['rotation'])).tolist(),scale=(np.asarray(box['scale'])*scale).tolist())
        diagnostics(prediction,run)
    components=[]
    for comp in baseline['structural_layout']['components']:
        if comp.get('vertices') and comp.get('triangles'):
            components.append(dict(id=comp['id'],kind=comp['kind'],
                vertices=transform_points(comp['vertices'],transform['vg_world_to_generation_world']).tolist(),
                triangles=comp['triangles']))
    refs=json.loads((run/'references.json').read_text(encoding='utf-8'))
    if prediction and prediction.get('edge_observations'):
        scene=ROOT/'data/scenes-v2'/transform['scene_id']
        frames=json.loads((scene/'frames/manifest.json').read_text(encoding='utf-8'))['frames']
        with np.load(scene/'reconstruction/reconstruction.npz',allow_pickle=False) as cameras:
            for i in sorted({e['frame'] for e in prediction['edge_observations']}):
                if any(r['frame']==i for r in refs):continue
                refs.append(dict(frame=i,source_name=frames[i]['name'],
                    camera_extrinsic_vggt=cameras['extrinsics'][i].tolist(),intrinsic=cameras['intrinsics'][i].tolist()))
        refs.sort(key=lambda r:r['frame'])
    if prediction and prediction.get('photo_alignment'):
        alignment=prediction['photo_alignment'];camera_file=run/args.prediction/'alignment-cameras.npz'
        if hashlib.sha256(camera_file.read_bytes()).hexdigest()!=alignment['camera_sha256']:raise ValueError('Aligned cameras changed')
        with np.load(camera_file,allow_pickle=False) as saved:aligned_poses=saved['extrinsics'].copy()
        for ref in refs:
            ref['camera_extrinsic_vggt']=aligned_poses[ref['frame']].tolist()
            ref['camera_source']='validated local alignment; source reconstruction unchanged'
    for ref in refs:
        ref['intrinsic_image_domain']='VGGT processed crop, not reference thumbnail'
        ref['reference_usage']='visual comparison only; no direct pixel projection'
    if prediction:
        scene=ROOT/'data/scenes-v2'/transform['scene_id']
        reference_folder=run/args.prediction/'references'
        shutil.copytree(run/'references',reference_folder,dirs_exist_ok=True)
        if scene.is_dir():
            if hashlib.sha256((scene/'reconstruction/reconstruction.npz').read_bytes()).hexdigest()!=transform['source_npz_sha256']:
                raise ValueError('Original cameras changed; refuse misleading overlays')
            photo_overlays(scene,reference_folder,refs,prediction['boxes'])
            provenance=prediction.get('annotation_provenance')
            if provenance:
                name=provenance['source_prediction']
                if not name.replace('-','').isalnum():raise ValueError('Invalid source prediction')
                source=run/name/'layout.json'
                if hashlib.sha256(source.read_bytes()).hexdigest()!=provenance['source_sha256']:
                    raise ValueError('Comparison source changed')
                original=json.loads(source.read_text(encoding='utf-8'));diagnostics(original,run)
                before=reference_folder/'before';before.mkdir(exist_ok=True)
                photo_overlays(scene,before,deepcopy(refs),original['boxes'])
                for ref in refs:ref['before_overlay']=f"before/{ref['frame']:06d}-overlay.jpg"
    else:reference_folder=run/'references'
    # A review may add annotated photographs; keep the original input bundle fixed.
    if prediction:(run/args.prediction/'references.json').write_text(json.dumps(refs,indent=2))
    if prediction:
        (run/args.prediction/'structured-layout.json').write_text(json.dumps(structured_layout(prediction,transform,refs),indent=2))
    counts=Counter(b['label'] for b in prediction['boxes']) if prediction else {}
    report=dict(schema_version=1,run_id=run.name,prediction_id=args.prediction,transform=transform,prediction=prediction,
        baseline=components,references=refs,category_counts=dict(counts),
        generation_gate=dict(accepted=False,status='REVIEW_REQUIRED' if prediction else 'AWAITING_INFERENCE',
            reasons=['Semantic labels and openings need verification against photographs',
                     'Metric scale is assumed, not measured',
                     'Existing floor/ceiling labels need review after up-sign correction']),
        reality_memory_modified=False,spatialgen_started=False)
    (run/'review.json').write_text(json.dumps(report,indent=2))
    if prediction:(run/args.prediction/'review.json').write_text(json.dumps(report,indent=2))
    web=ROOT/'web/generation-reviews'/run.name
    web.mkdir(parents=True,exist_ok=True)
    shutil.copy2(run/'review.json',web/'review.json')
    shutil.copy2(run/'spatiallm-input.ply',web/'input.ply')
    shutil.copytree(reference_folder,web/'references',dirs_exist_ok=True)
    if prediction:
        comparison=web/args.prediction
        comparison.mkdir(exist_ok=True)
        shutil.copy2(run/'review.json',comparison/'review.json')
        shutil.copy2(run/'spatiallm-input.ply',comparison/'input.ply')
        shutil.copytree(reference_folder,comparison/'references',dirs_exist_ok=True)
    choices=[]
    for file in sorted(web.glob('prediction-*/review.json')):
        prior=json.loads(file.read_text(encoding='utf-8'));model=prior['prediction']['model']
        if prior['prediction'].get('display_label'):label=prior['prediction']['display_label']
        elif prior['prediction'].get('wall_height_changes'):label='統合＋高さ補正'
        elif 'boxes.' not in prior['prediction'].get('task','Detect boxes.'):
            label='種類を指定した推定'
        else:label='全体推定'
        if model.endswith('0.5B'):label+='（0.5B）'
        elif model.endswith('1B'):label+='（1B）'
        choices.append(dict(id=file.parent.name,label=label+' / '+file.parent.name.removeprefix('prediction-')))
    (web/'index.json').write_text(json.dumps(choices,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'http://127.0.0.1:8001/static/generation-review.html?run={run.name}&prediction={args.prediction}')


if __name__=='__main__':main()
