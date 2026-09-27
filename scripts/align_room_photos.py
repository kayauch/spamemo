"""Create reversible photo/plane corrections in a fresh prediction directory."""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from spatial_memory.photo_alignment import align_room_planes
from spatial_memory.semantic_textures import SemanticPhotoMasks
from spatial_memory.camera_refinement import refine_capture_cameras
from spatial_memory.texturing import preprocess_frame
from spatial_memory.solid_room import wall_loop
from spatial_recon.types import ReconstructionResult


def overlay(image,layout,pose,k,inverse):
    picture=image.copy();draw=ImageDraw.Draw(picture)
    p=wall_loop(layout['boxes']);f=layout['vertical_anchors']['floor']['z'];c=layout['vertical_anchors']['ceiling']['z']
    vertices=np.vstack([np.column_stack([p,np.full(len(p),z)]) for z in (f,c)])
    world=vertices@inverse[:3,:3].T+inverse[:3,3];camera=world@pose[:,:3].T+pose[:,3]
    uv=(camera@k.T);uv=uv[:,:2]/np.maximum(camera[:,2,None],1e-8)
    for i in range(len(p)):
        for a,b,color in [(i,(i+1)%len(p),'#28aaff'),(i+len(p),(i+1)%len(p)+len(p),'#ffca66'),(i,i+len(p),'#28aaff')]:
            if min(camera[a,2],camera[b,2])>.01:
                endpoints=np.clip(np.rint(uv[[a,b]]),-100000,100000).astype(int)
                visible,start,end=cv2.clipLine((0,0,picture.width,picture.height),tuple(endpoints[0]),tuple(endpoints[1]))
                if visible:draw.line([start,end],fill=color,width=2)
    return picture


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);parser.add_argument('--prediction',required=True);parser.add_argument('--before-url',default='');args=parser.parse_args()
    if not args.prediction.replace('-','').isalnum():raise ValueError('Invalid prediction')
    run=args.run.resolve();directory=run/args.prediction;path=directory/'layout.json'
    before=json.loads(path.read_text(encoding='utf-8'))
    if before.get('photo_alignment'):raise ValueError('Already aligned; use the unmodified result')
    transform=json.loads((run/'transform.json').read_text(encoding='utf-8'))
    scene=ROOT/'data/scenes-v2'/transform['scene_id'];npz=scene/'reconstruction/reconstruction.npz'
    digest=hashlib.sha256(npz.read_bytes()).hexdigest()
    if digest!=transform['source_npz_sha256']:raise ValueError('Source evidence changed')
    with np.load(npz,allow_pickle=False) as z:
        depth=z['depth'].squeeze().copy();conf=z['confidence'].squeeze().copy();poses=z['extrinsics'].copy();ks=z['intrinsics'].copy()
    records=json.loads((scene/'frames/manifest.json').read_text(encoding='utf-8'))['frames'];h,w=depth.shape[1:]
    metric=json.loads(npz.with_name('metrics.json').read_text(encoding='utf-8'))
    images=[preprocess_frame(scene/'frames'/r['name'],h,w,preprocessing='aligned_rgbd' if metric.get('model')=='calibrated-rgbd' else 'vggt') for r in records]
    result=ReconstructionResult(depth=depth,confidence=conf,extrinsics=poses.copy(),intrinsics=ks,images=np.stack([np.asarray(im) for im in images]),metrics={'model':metric.get('model')})
    print('Checking camera alignment against withheld image matches',flush=True)
    cameras=refine_capture_cameras(result,records)
    print('Checking floor and wall offsets against independent views',flush=True)
    masks=SemanticPhotoMasks(directory,before,h,w)
    aligned,geometry=align_room_planes(before,depth,conf,poses,ks,transform['vg_world_to_generation_world'],records,masks)
    if masks.errors:raise ValueError('Segmentation masks unavailable; preserve original geometry')
    report=dict(version=1,before_result_url=args.before_url,source_npz_sha256=digest,source_unchanged=True,cameras=cameras,geometry=geometry,
                camera_file='alignment-cameras.npz',references=[],
                limitations=['This is bounded local alignment, not recovery of unseen areas.',
                             'Overlay lines have no occlusion handling; hidden room boundaries are also drawn.'])
    # Keep the exact before-model for review; never edit source reconstruction.
    (directory/'alignment-before.json').write_text(json.dumps(before,ensure_ascii=False),encoding='utf-8')
    np.savez_compressed(directory/report['camera_file'],extrinsics=result.extrinsics)
    report['camera_sha256']=hashlib.sha256((directory/report['camera_file']).read_bytes()).hexdigest()
    refs=sorted(set(np.linspace(0,len(records)-1,min(8,len(records))).astype(int).tolist()+[i for i,r in enumerate(records) if r.get('source_kind')=='photo']+[r['index'] for r in cameras['frames'] if r.get('accepted')]))
    destination=ROOT/'web/generation-reviews'/run.name/args.prediction/'alignment';destination.mkdir(parents=True,exist_ok=True)
    inverse=np.asarray(transform['generation_world_to_vg_world'])
    for i in refs:
        images[i].save(destination/f'{i}-photo.jpg',quality=92)
        overlay(images[i],before,poses[i],ks[i],inverse).save(destination/f'{i}-before.jpg',quality=92)
        overlay(images[i],aligned,result.extrinsics[i],ks[i],inverse).save(destination/f'{i}-after.jpg',quality=92)
        report['references'].append(dict(frame=i,photo=f'{i}-photo.jpg',before=f'{i}-before.jpg',after=f'{i}-after.jpg'))
    aligned['photo_alignment']=report
    (directory/'alignment.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (destination/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    path.write_text(json.dumps(aligned,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(camera_corrections=cameras['accepted_count'],plane_corrections=geometry['accepted_count'])),flush=True)


if __name__=='__main__':main()
