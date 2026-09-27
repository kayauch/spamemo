"""Register a validated local 2DGS PLY in a room; no network publication."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('scene_dir',type=Path)
parser.add_argument('colmap_dir',type=Path)
parser.add_argument('model_dir',type=Path)
args=parser.parse_args()
scene=args.scene_dir.resolve()
alignment=json.loads((args.colmap_dir/'alignment.json').read_text())
report=json.loads((args.model_dir/'evaluation/report.json').read_text())
source_hash=hashlib.sha256((scene/'reconstruction/reconstruction.npz').read_bytes()).hexdigest()
if alignment['scene_id']!=scene.name or source_hash!=alignment['source_npz_sha256']:
    raise ValueError('Model belongs to a different geometry reconstruction')
if not report['finite_parameters'] or report['max_camera_center_error']>1e-4:
    raise ValueError('Model validation failed')
if not np.allclose(report['world_from_2dgs'],np.eye(4)):
    raise ValueError('Viewer currently requires unchanged VGGT coordinates')
if 'alignment_sha256' in report:
    if report['alignment_sha256'] != hashlib.sha256((args.colmap_dir/'alignment.json').read_bytes()).hexdigest():
        raise ValueError('Evaluation used a different camera alignment')
if 'camera_files_sha256' in report:
    camera_digest=hashlib.sha256((args.colmap_dir/'sparse/0/cameras.txt').read_bytes()+(args.colmap_dir/'sparse/0/images.txt').read_bytes()).hexdigest()
    if report['camera_files_sha256'] != camera_digest:
        raise ValueError('Evaluated camera files have changed')
ply=args.model_dir/f"point_cloud/iteration_{report['iteration']}/point_cloud.ply"
digest=hashlib.sha256(ply.read_bytes()).hexdigest()
if report.get('model_ply_sha256', digest) != digest or report.get('geometry_npz_sha256', source_hash) != source_hash:
    raise ValueError('Evaluation report does not match the model or geometry being registered')
path=scene/'spatial_model'/f'appearance-2dgs-{digest[:16]}.ply'
manifest_path=scene/'spatial_model/manifest.json'
manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
with np.load(scene/'reconstruction/reconstruction.npz',allow_pickle=False) as source:
    poses=source['extrinsics']
    intrinsics=source['intrinsics']
    image_height, image_width = source['depth'].shape[1:3]
camera_refinement=alignment.get('appearance_camera_refinement')
if camera_refinement:
    if not camera_refinement['world_geometry_fixed'] or not np.allclose(camera_refinement['intrinsics'],intrinsics,atol=1e-5):
        raise ValueError('Camera correction must retain VGGT intrinsics and coordinate system')
    revised=np.asarray(camera_refinement['extrinsics'])
    if revised.shape != poses.shape or not np.isfinite(revised).all():
        raise ValueError('Invalid refined cameras')
    if camera_refinement['source_npz_sha256'] != source_hash:
        raise ValueError('Camera refinement belongs to different geometry')
    for old,new in zip(poses,revised):
        if not np.allclose(new[:,:3]@new[:,:3].T,np.eye(3),atol=1e-5) or np.linalg.det(new[:,:3]) < .999:
            raise ValueError('Invalid refined rotation')
        shift=np.linalg.norm(-new[:,:3].T@new[:,3]+old[:,:3].T@old[:,3])
        angle=np.degrees(np.arccos(np.clip((np.trace(new[:,:3]@old[:,:3].T)-1)/2,-1,1)))
        if shift > manifest['metrics']['scene_diagonal']*.010001 or angle > 2.001:
            raise ValueError('Camera refinement exceeds bounded correction')
    poses=revised
views=[]
for index in range(0,len(poses),max(1,len(poses)//8)):
    rotation,translation=poses[index,:,:3],poses[index,:,3]
    center=-rotation.T@translation
    forward=rotation.T@np.array([0.,0.,1.])
    up=rotation.T@np.array([0.,-1.,0.])
    views.append({'frame':index,'position':center.tolist(),'forward':forward.tolist(),'up':up.tolist(),
                  'fov_y_degrees':float(np.degrees(2*np.arctan(image_height/(2*intrinsics[index,1,1])))),
                  'image_aspect':float(image_width/image_height)})
if manifest.get('appearance_2dgs') and 'appearance_2dgs_baseline' not in manifest:
    manifest['appearance_2dgs_baseline'] = manifest['appearance_2dgs']
manifest['appearance_2dgs']={'path':str(path),'revision':digest,'world_from_2dgs':alignment['world_from_2dgs'],
                           'geometry_source_npz_sha256':source_hash,'report':report,
                           'capture_views':views,
                           'renderer':'GaussianSplats3D 0.4.7 TwoD, SH degree 2'}
if camera_refinement:
    manifest['appearance_2dgs']['camera_refinement']=camera_refinement
for filename in ('refinement.json', 'filtering.json'):
    metadata = args.model_dir / filename
    if metadata.exists():
        manifest['appearance_2dgs'][filename.removesuffix('.json')] = json.loads(metadata.read_text())
shutil.copy2(ply,path)
temporary=manifest_path.with_suffix('.2dgs.tmp')
temporary.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
temporary.replace(manifest_path)
print(json.dumps(manifest['appearance_2dgs'],indent=2))
