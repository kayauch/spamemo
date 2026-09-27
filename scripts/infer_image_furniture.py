"""Local image detection + segmentation + VGGT lifting; never change source evidence."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Sam2Model, Sam2Processor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from spatial_memory.texturing import preprocess_frame
from spatial_memory.image_furniture import (UP_ROTATIONS, lift_mask, bounds, merge_observations,
                                             FURNITURE_CATEGORIES, classify_detection)
from spatial_memory.storage_geometry import refine_storage_layout
from spatial_memory.storage_fragments import consolidate_storage_fragments
from spatial_memory.furniture_quality import (select_furniture_frames, object_quality,
                                             part_hints, apply_furniture_overrides)

CATEGORIES = [*FURNITURE_CATEGORIES, 'wall']


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path)
    p.add_argument('--architecture',default='prediction-auto-001');p.add_argument('--output',default='prediction-image-001')
    p.add_argument('--frames',type=int,default=24);p.add_argument('--options',type=Path);a=p.parse_args()
    options=json.loads(a.options.read_text(encoding='utf-8')) if a.options else {}
    if not isinstance(options,dict):raise ValueError('Furniture options must be an object')
    frame_budget=options.get('furniture_frame_budget',a.frames)
    if isinstance(frame_budget,bool) or not isinstance(frame_budget,int) or not 2 <= frame_budget <= 32:raise ValueError('Frame count must be 2..32')
    if any(not x.replace('-','').isalnum() for x in (a.architecture,a.output)):raise ValueError('Invalid artifact name')
    run=a.run.resolve();out=run/a.output
    if out.exists():raise FileExistsError(out)
    t=json.loads((run/'transform.json').read_text(encoding='utf-8'))
    source=run/a.architecture/'layout.json';layout=json.loads(source.read_text(encoding='utf-8'))
    scene=ROOT/'data/scenes-v2'/t['scene_id'];npz=scene/'reconstruction/reconstruction.npz'
    if hashlib.sha256(npz.read_bytes()).hexdigest()!=t['source_npz_sha256']:raise ValueError('Source changed')
    if layout['input_sha256']!=hashlib.sha256((run/'spatiallm-input.ply').read_bytes()).hexdigest():raise ValueError('Layout coordinates differ')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    detector_path=ROOT/'checkpoints/grounding-dino-tiny';sam_path=ROOT/'checkpoints/sam2.1-hiera-tiny'
    print('Loading local image models',flush=True)
    processor=AutoProcessor.from_pretrained(detector_path,local_files_only=True)
    detector=AutoModelForZeroShotObjectDetection.from_pretrained(detector_path,local_files_only=True).to(device).eval()
    sam_processor=Sam2Processor.from_pretrained(sam_path,local_files_only=True)
    sam=Sam2Model.from_pretrained(sam_path,local_files_only=True).to(device).eval()
    out.mkdir();(out/'observations').mkdir()
    with np.load(npz,allow_pickle=False) as z:
        depth=z['depth'].squeeze(-1) if z['depth'].ndim==4 else z['depth'].copy()
        conf=z['confidence'].squeeze(-1) if z['confidence'].ndim==4 else z['confidence'].copy()
        ks=z['intrinsics'].copy();es=z['extrinsics'].copy()
    records=json.loads((scene/'frames/manifest.json').read_text(encoding='utf-8'))['frames']
    if len(records)!=len(depth):raise ValueError('Frame count differs')
    selection=select_furniture_frames(records,es,frame_budget)
    rotation=UP_ROTATIONS[t.get('camera_up_axis','image-top')]
    observations=[];reports=[]
    for frame in selection['indices']:
        image=preprocess_frame(scene/'frames'/records[frame]['name'],*depth.shape[1:],
                               preprocessing='aligned_rgbd' if records[frame].get('source_kind')=='sensor' else 'vggt')
        upright=Image.fromarray(np.rot90(np.asarray(image),rotation).copy())
        inputs=processor(images=upright,text='. '.join(CATEGORIES)+'.',return_tensors='pt').to(device)
        with torch.inference_mode():pred=detector(**inputs)
        detection=processor.post_process_grounded_object_detection(pred,inputs.input_ids,threshold=.30,text_threshold=.25,target_sizes=[upright.size[::-1]])[0]
        boxes=detection['boxes'].cpu().numpy();scores=detection['scores'].cpu().numpy()
        labels=detection.get('text_labels',detection.get('labels'))
        # Avoid duplicate boxes from synonymous text prompts within one frame.
        chosen=[]
        for j in np.argsort(-scores):
            kind,label=classify_detection(labels[j]);box=boxes[j]
            duplicate=False
            for prior in chosen:
                other=boxes[prior[0]];inter=np.prod(np.maximum(0,np.minimum(box[2:],other[2:])-np.maximum(box[:2],other[:2])))
                union=np.prod(box[2:]-box[:2])+np.prod(other[2:]-other[:2])-inter
                if prior[2] == kind and inter/max(union,1)> .65:duplicate=True
            if not duplicate:chosen.append((j,label,kind))
        chosen=chosen[:24];display=upright.copy();draw=ImageDraw.Draw(display)
        if chosen:
            inp=sam_processor(images=upright,input_boxes=[[boxes[j].tolist() for j,_,_ in chosen]],return_tensors='pt').to(device)
            with torch.inference_mode():seg=sam(**inp,multimask_output=False)
            masks=sam_processor.post_process_masks(seg.pred_masks.cpu(),inp['original_sizes'].cpu())[0][:,0].numpy()
            for mask,(j,label,kind) in zip(masks,chosen):
                raw_mask=np.rot90(mask,-rotation).copy().astype(np.uint8)
                original=cv2.erode(raw_mask,np.ones((3,3),np.uint8)).astype(bool)
                points=lift_mask(original,depth[frame],conf[frame],ks[frame],es[frame],t['vg_world_to_generation_world'])
                name=f'{frame:06d}-{j:02d}';Image.fromarray(raw_mask*255).save(out/'observations'/f'{name}-mask.png')
                report=dict(frame=int(frame),viewpoint_id=selection['viewpoint_ids'][frame],detection_id=name,
                            label=label,raw_label=str(labels[j]),kind=kind,score=float(scores[j]),box=boxes[j].tolist(),points=len(points),mask=f'observations/{name}-mask.png')
                reports.append(report)
                if kind == 'furniture' and len(points)>=100:
                    lo,hi=bounds(points)
                    if np.max(hi-lo)<6 and np.min(hi-lo)>.025:
                        observations.append(dict(frame=int(frame),viewpoint_id=selection['viewpoint_ids'][frame],detection_id=name,label=label,score=float(scores[j]),points=points))
                draw.rectangle(boxes[j].tolist(),outline='red',width=2);draw.text(tuple(boxes[j][:2]),label,fill='red')
        display.save(out/'observations'/f'{frame:06d}.jpg')
        print(f'frame {frame}: {len(chosen)} detections',flush=True)
    boxes=[deepcopy(b) for b in layout['boxes'] if b['class']!='bbox']
    inverse=np.asarray(t['generation_world_to_vg_world']);scale=np.linalg.norm(inverse[:3,0]);rot=inverse[:3,:3]/scale
    groups=merge_observations(observations);rejected=[];object_reports=[]
    groups.sort(key=lambda group: object_quality(group)['independent_viewpoints'] < 2)
    for group in groups:
        quality=object_quality(group)
        tentative = quality['independent_viewpoints'] < 2
        points=np.concatenate(group['points']);lo,hi=bounds(points);center=(lo+hi)/2;size=np.maximum(hi-lo,.06)
        box=dict(id=6000+len(boxes),label=group['label'],center=center.tolist(),scale=size.tolist(),rotation=np.eye(3).tolist(),
                 evidence=dict(valid=True,points_inside=len(points),confidence=None,evidence_type='image segmentation lifted with VGGT; dimensions cover visible surfaces only',
                               quality=quality,part_hints=part_hints(points,center,np.eye(3),size,group['label'])),
                 source_frames=sorted(group['frames']),source_detection_ids=sorted(group['detection_ids']),
                 detection_scores=group['scores'],constraint='PROPOSED; not complete hidden geometry',
                 vg_world=dict(center=(center@inverse[:3,:3].T+inverse[:3,3]).tolist(),rotation=rot.tolist(),scale=(size*scale).tolist()))
        box['class']='bbox'
        if tentative:
            box.update(hidden=True, review_status='needs_confirmation')
            box['evidence']['requires_confirmation'] = True
            rejected.append(dict(id=box['id'],label=group['label'],frames=group['frames'],reason='fewer than two independent camera positions; retained for review',quality=quality))
        boxes.append(box)
        object_reports.append(dict(id=box['id'],label=box['label'],hidden=bool(box.get('hidden')),review_status=box.get('review_status','multi_view'),dimensions_model_units=size.tolist(),center_model_units=center.tolist(),**quality))
    # Empty rooms are valid inputs. Absence of detections is not proof of absence;
    # retain architecture and report the limitation rather than fabricate furniture.
    layout.update(boxes=boxes,model='Grounding DINO + SAM2 images + VGGT depth; SpatialLM architecture',display_label='元画像から家具を認識（自動）',
                  accepted_for_generation=False,source_architecture=a.architecture,
                  limitations=['Visible-surface bounds, not full recovered furniture', 'Image detections can be wrong; multi-view association is heuristic', 'No manual labels, edges, or per-room prompts used'],
                  image_furniture=dict(version=3,categories=CATEGORIES,frames=selection['selected_count'],frame_selection=selection,up_axis=t.get('camera_up_axis'),detections=reports,rejected=rejected,
                      confirmed_count=sum(b['class']=='bbox' and not b.get('hidden') for b in boxes),
                      quality_report=dict(version=1,objects=object_reports,unresolved_candidates=rejected,
                          recommendations=([dict(code='no_confirmed_furniture',message='家具がある場合は、全体と側面が見える写真を追加してください。検出なしは空室の証明ではありません。')] if not object_reports else []),
                          limitations=['Virtual panorama faces are not independent camera positions', 'Support scores are not calibrated probabilities', 'Unseen furniture details remain inferred']),
                      models=[json.loads((x/'source_revision.json').read_text()) for x in (detector_path,sam_path)]))
    print('Checking storage front/side geometry against independent views',flush=True)
    layout = consolidate_storage_fragments(layout, t)
    layout = refine_storage_layout(layout,out,depth,conf,es,ks,t,records)
    boxes = layout['boxes']
    if options.get('furniture_overrides'):
        from spatial_memory.texture_quality import calibration_scale
        layout=apply_furniture_overrides(layout,options['furniture_overrides'],calibration_scale(layout,options))
    layout['counts']={k:sum(b['class']==c for b in layout['boxes']) for k,c in [('walls','wall'),('doors','door'),('windows','window'),('bboxes','bbox')]}
    (out/'layout.json').write_text(json.dumps(layout,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(layout['counts']),flush=True)


if __name__=='__main__':
    lock=ROOT/'data/appearance-pipeline.lock'
    with lock.open('x') as f:f.write('image-furniture')
    try:main()
    finally:lock.unlink()
