"""Render a frozen current-best model into a separate repair experiment (WSL)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'third_party/2dgs'),str(ROOT/'src')]
import numpy as np
import torch
from PIL import Image
from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel
from gaussian_renderer import render
from spatial_memory.scene_repair import correspondence, sample, classify, bounded_region
from spatial_memory.depth_prior import reliable_depth_mask

parser=argparse.ArgumentParser()
mp,pp=ModelParams(parser),PipelineParams(parser)
parser.add_argument('--iteration',type=int,default=3000)
parser.add_argument('--geometry-npz',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
alignment=json.loads((Path(args.source_path)/'alignment.json').read_text())
source_hash=hashlib.sha256(args.geometry_npz.read_bytes()).hexdigest()
if source_hash!=alignment['source_npz_sha256']: raise ValueError('Wrong geometry')
model=GaussianModel(2)
scene=Scene(mp.extract(args),model,load_iteration=args.iteration,shuffle=False)
with np.load(args.geometry_npz,allow_pickle=False) as src:
    rawdepth,confidence,poses,ks=[src[k] for k in ('depth','confidence','extrinsics','intrinsics')]
if rawdepth.ndim==4: rawdepth=rawdepth[...,0]
if confidence.ndim==4: confidence=confidence[...,0]
frames=[]
test_names={c.image_name for c in scene.getTestCameras()}
with torch.no_grad():
    for cam in scene.getTrainCameras()+scene.getTestCameras():
        index=int(cam.image_name)
        package=render(cam,model,pp.extract(args),torch.zeros(3,device='cuda'))
        rgb=(package['render'].clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        depth=package['surf_depth'][0].cpu().numpy()
        alpha=package['rend_alpha'][0].cpu().numpy()
        e=np.column_stack((cam.R.T,cam.T))
        k=ks[index].copy()
        if depth.shape!=rawdepth[index].shape: raise ValueError('Use native VGGT resolution for this experiment')
        uv,support=correspondence(depth,e,k,rawdepth[index],poses[index],ks[index])
        reliable=reliable_depth_mask(rawdepth[index],confidence[index])
        support &= sample(reliable.astype(np.uint8),uv,True).astype(bool)
        photo=np.asarray(Image.open(Path(args.source_path)/'images'/f'{cam.image_name}.png').convert('RGB'))
        labels,broken,error=classify(rgb,photo,alpha,depth,support)
        region=bounded_region(broken)
        folder=args.output/cam.image_name; folder.mkdir()
        Image.fromarray(rgb).save(folder/'render.png')
        Image.fromarray(photo).save(folder/'observed.png')
        Image.fromarray(region.astype(np.uint8)*255).save(folder/'mask.png')
        overlay=rgb.copy(); overlay[region]=[255,0,180]
        Image.fromarray(overlay).save(folder/'repair-region.png')
        np.savez_compressed(folder/'geometry.npz',depth=depth,alpha=alpha,extrinsic=e,intrinsic=k,
                            labels=labels,broken=broken,support=support,error=error)
        frames.append(dict(frame=cam.image_name,held_out=cam.image_name in test_names,
                           repair_pixels=int(region.sum()),unknown_pixels=int((labels==0).sum())))
        print(cam.image_name, int(region.sum()),flush=True)
ply=Path(args.model_path)/f'point_cloud/iteration_{args.iteration}/point_cloud.ply'
metadata=dict(source_model=str(Path(args.model_path).resolve()),source_ply_sha256=hashlib.sha256(ply.read_bytes()).hexdigest(),
              source_npz_sha256=source_hash,alignment_sha256=hashlib.sha256((Path(args.source_path)/'alignment.json').read_bytes()).hexdigest(),
              frames=frames,labels={'UNKNOWN':0,'RECONSTRUCTED':1,'OBSERVED':2,'AI_REPAIRED':3,'AI_GENERATED':4},
              unknown_policy='defer; no unsupported geometry or synthetic observation')
(args.output/'snapshot.json').write_text(json.dumps(metadata,indent=2))
print('Snapshot complete',flush=True)
