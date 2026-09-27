"""Prepare a separate, reversible SpatialLM input and reference-frame bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import open3d as o3d
from PIL import Image
from spatial_memory.generation_evidence import generation_transform,transform_points

p=argparse.ArgumentParser();p.add_argument('scene',type=Path);p.add_argument('output',type=Path)
p.add_argument('--assumed-height',type=float,default=2.5);p.add_argument('--max-points',type=int,default=80000)
p.add_argument('--up-axis',choices=['image-top','image-left','image-right','image-bottom'],default='image-top')
a=p.parse_args();scene=a.scene.resolve();out=a.output.resolve()
if out.is_relative_to(scene):raise ValueError('Keep experiment outside original scene')
if out.exists():raise FileExistsError(out)
if a.max_points<1000:raise ValueError('Insufficient point budget')
metrics=json.loads((scene/'reconstruction/metrics.json').read_text())
if metrics.get('model') not in ('VGGT-1B','calibrated-rgbd') or metrics.get('fallback'):raise ValueError('Real VGGT or calibrated RGB-D evidence required')
manifest=json.loads((scene/'spatial_model/manifest.json').read_text(encoding='utf-8'))
cloud_path=scene/'reconstruction/scene.ply'
cloud=o3d.io.read_point_cloud(str(cloud_path))
if not cloud.has_colors():raise ValueError('RGB evidence missing')
points=np.asarray(cloud.points);colors=np.asarray(cloud.colors)
valid=np.isfinite(points).all(1)&np.isfinite(colors).all(1)
points,colors=points[valid],colors[valid]
npz=scene/'reconstruction/reconstruction.npz'
with np.load(npz,allow_pickle=False) as src:
    poses=src['extrinsics'];ks=src['intrinsics'];confidence=src['confidence']
transform=generation_transform(points,manifest['room_frame'],poses,a.assumed_height,camera_up_axis=a.up_axis,
                               preserve_metric_scale=metrics.get('model')=='calibrated-rgbd')
xyz=transform_points(points,transform['vg_world_to_generation_world'])
keep=((xyz>=0)&(xyz<31.9)).all(1)
prepared=o3d.geometry.PointCloud();prepared.points=o3d.utility.Vector3dVector(xyz[keep]);prepared.colors=o3d.utility.Vector3dVector(colors[keep])
prepared=prepared.voxel_down_sample(.025)
prepared,_=prepared.remove_statistical_outlier(nb_neighbors=15,std_ratio=2)
if len(prepared.points)>a.max_points:
    ids=np.sort(np.random.default_rng(42).choice(len(prepared.points),a.max_points,replace=False))
    prepared=prepared.select_by_index(ids.tolist())
out.mkdir(parents=True)
o3d.io.write_point_cloud(str(out/'spatiallm-input.ply'),prepared)
error=float(np.abs(transform_points(transform_points(points,transform['vg_world_to_generation_world']),transform['generation_world_to_vg_world'])-points).max())
transform.update(scene_id=scene.name,source_pointcloud_sha256=hashlib.sha256(cloud_path.read_bytes()).hexdigest(),
                 source_npz_sha256=hashlib.sha256(npz.read_bytes()).hexdigest(),input_points=len(points),output_points=len(prepared.points),
                 max_roundtrip_error=error,rgb_format='PLY RGB uint8; official loader normalizes to [-1,1]',
                 voxel_m=.025,max_points=a.max_points,seed=42,layer='RECONSTRUCTED',
                 generated_world_state='not generated',positive_domain_filtered=int((~keep).sum()))
(out/'transform.json').write_text(json.dumps(transform,indent=2))
# Independent structural baseline; its semantic labels are not SpatialLM predictions.
(out/'existing-structure.json').write_text(json.dumps(dict(room_frame=manifest['room_frame'],
    structural_layout=manifest['structural_layout'],components=manifest['components']),indent=2))
frames=json.loads((scene/'frames/manifest.json').read_text())['frames']
rank=[];selected=[]
for i,f in enumerate(frames):
    coverage=float((confidence[i]>1.01).mean())
    score=coverage*np.log1p(f.get('sharpness') or 0)
    rank.append((score,i,coverage))
for score,i,coverage in sorted(rank,reverse=True):
    direction=poses[i,2,:3]
    if any(direction@poses[j,2,:3]>.95 and abs(i-j)<12 for j in selected):continue
    selected.append(i)
    if len(selected)==8:break
(out/'references').mkdir()
records=[]
for i in sorted(selected):
    with Image.open(scene/'frames'/frames[i]['name']) as im:
        im.thumbnail((1036,1036));im.save(out/'references'/f'{i:06d}.jpg',quality=92)
    records.append(dict(frame=i,source_name=frames[i]['name'],sharpness=frames[i].get('sharpness'),
                        confidence_coverage=float((confidence[i]>1.01).mean()),
                        camera_extrinsic_vggt=poses[i].tolist(),intrinsic=ks[i].tolist(),
                        intrinsic_image_domain='VGGT processed crop, not reference thumbnail',
                        reference_usage='visual comparison only; no direct pixel projection'))
(out/'references.json').write_text(json.dumps(records,indent=2))
print(json.dumps(transform,indent=2))
