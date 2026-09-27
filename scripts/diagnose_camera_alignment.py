"""Measure real feature reprojection residuals against fixed VGGT depth/cameras.

Read-only geometry diagnostic. It never updates poses or the live room.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('scene_dir', type=Path)
parser.add_argument('colmap_dir', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
cv2.setNumThreads(4)
with np.load(args.scene_dir/'reconstruction/reconstruction.npz', allow_pickle=False) as src:
    depth, confidence, extrinsics, intrinsics = [src[k] for k in ('depth','confidence','extrinsics','intrinsics')]
if depth.ndim == 4: depth = depth[..., 0]
if confidence.ndim == 4: confidence = confidence[..., 0]
sift = cv2.SIFT_create(nfeatures=1800)
features = []
for i in range(len(depth)):
    image = cv2.imread(str(args.colmap_dir/'images'/f'{i:06d}.png'), cv2.IMREAD_GRAYSCALE)
    if image.shape != depth[i].shape: raise ValueError('Use the unscaled COLMAP dataset')
    keys, descriptors = sift.detectAndCompute(image, None)
    features.append((np.array([k.pt for k in keys]), descriptors))
matcher = cv2.BFMatcher()
records = []
for i in range(len(depth)):
    for j in (i+1, i+3):
        if j >= len(depth): continue
        a, da = features[i]; b, db = features[j]
        if da is None or db is None or min(len(da),len(db)) < 2: continue
        ab = {m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(da,db,k=2) if m.distance < .7*n.distance}
        ba = {m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(db,da,k=2) if m.distance < .7*n.distance}
        ids = np.array([(x,y) for x,y in ab.items() if ba.get(y)==x], dtype=int)
        if len(ids)<8: continue
        uv, expected = a[ids[:,0]], b[ids[:,1]]
        p = np.rint(uv).astype(int)
        p[:,0]=p[:,0].clip(0,depth.shape[2]-1);p[:,1]=p[:,1].clip(0,depth.shape[1]-1)
        d,c=depth[i,p[:,1],p[:,0]],confidence[i,p[:,1],p[:,0]]
        valid=np.isfinite(d)&(d>0)&np.isfinite(c)&(c>1.01)
        rays=np.column_stack([uv,np.ones(len(uv))]) @ np.linalg.inv(intrinsics[i]).T
        e=extrinsics[i]
        world=(rays*d[:,None]-e[:,3]) @ e[:,:3]
        cam=world@extrinsics[j,:,:3].T+extrinsics[j,:,3]
        proj=cam@intrinsics[j].T
        predicted=proj[:,:2]/np.maximum(proj[:,2:],1e-8)
        error=np.linalg.norm(predicted-expected,axis=1)
        valid &= np.isfinite(error)&(cam[:,2]>0)
        if not valid.any(): continue
        # Fundamental-matrix RANSAC checks that matches describe camera motion,
        # rather than repetitions in the lockers or independent movement.
        _, inliers=cv2.findFundamentalMat(uv.astype(np.float32),expected.astype(np.float32),cv2.FM_RANSAC,1.5,.999)
        if inliers is not None: valid &= inliers.ravel().astype(bool)
        if not valid.any(): continue
        records.append(dict(source=i,target=j,matches=int(valid.sum()),
                            median_pixels=float(np.median(error[valid])),
                            p90_pixels=float(np.quantile(error[valid],.9))))
report=dict(method='mutual SIFT ratio .7, fundamental RANSAC 1.5px, VGGT depth confidence >1.01',
            median_pair_error=float(np.median([r['median_pixels'] for r in records])),pairs=records)
args.output.write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='pairs'},indent=2))
print('Worst pairs:',sorted(records,key=lambda r:r['median_pixels'],reverse=True)[:8])
