"""Bounded appearance-camera resection against fixed VGGT world anchors.

This is an experimental candidate. World geometry is never optimized. A separate
feature subset must improve before each small pose update is accepted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('scene_dir', type=Path)
parser.add_argument('colmap_dir', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--features',type=int,default=2000)
parser.add_argument('--base-cameras',type=Path)
parser.add_argument('--minimum-features',type=int,default=40)
parser.add_argument('--minimum-inliers',type=int,default=25)
args = parser.parse_args()
if args.output.exists(): raise ValueError('Choose a new output file')
cv2.setNumThreads(4); cv2.setRNGSeed(42)
npz=args.scene_dir/'reconstruction/reconstruction.npz'
with np.load(npz,allow_pickle=False) as src:
    d,c,e,k=[src[key] for key in ('depth','confidence','extrinsics','intrinsics')]
if d.ndim==4:d=d[...,0]
if c.ndim==4:c=c[...,0]
alignment=json.loads((args.colmap_dir/'alignment.json').read_text())
image_scale=alignment.get('appearance_image_scale',1)
initial=e.copy()
if args.base_cameras:
    base=json.loads(args.base_cameras.read_text())
    if base['source_npz_sha256']!=hashlib.sha256(npz.read_bytes()).hexdigest():
        raise ValueError('Base cameras belong to different geometry')
    initial=np.asarray(base['extrinsics'])
manifest=json.loads((args.scene_dir/'spatial_model/manifest.json').read_text(encoding='utf-8'))
diagonal=manifest['metrics']['scene_diagonal']
sift=cv2.SIFT_create(nfeatures=args.features)
features=[]
for i in range(len(d)):
    image=cv2.imread(str(args.colmap_dir/'images'/f'{i:06d}.png'),cv2.IMREAD_GRAYSCALE)
    if image.shape!=(d.shape[1]*image_scale,d.shape[2]*image_scale):raise ValueError('Image scale does not match metadata')
    keys,desc=sift.detectAndCompute(image,None)
    uv=np.array([p.pt for p in keys]).reshape(-1,2)/image_scale
    pixel=np.rint(uv).astype(int)
    pixel[:,0]=pixel[:,0].clip(0,d.shape[2]-1);pixel[:,1]=pixel[:,1].clip(0,d.shape[1]-1)
    depth,conf=d[i,pixel[:,1],pixel[:,0]],c[i,pixel[:,1],pixel[:,0]]
    rays=np.column_stack([uv,np.ones(len(uv))])@np.linalg.inv(k[i]).T
    world=(rays*depth[:,None]-e[i,:,3])@e[i,:,:3]
    valid=np.isfinite(world).all(axis=1)&(depth>0)&(conf>1.01)
    features.append((uv,desc,world,valid))
anchors=[{} for _ in d]
matcher=cv2.BFMatcher()
for i in range(len(d)):
    for delta in (1,2,3,6):
        j=i+delta
        if j>=len(d):continue
        a,da,wa,va=features[i];b,db,wb,vb=features[j]
        if da is None or db is None or min(len(da),len(db))<2:continue
        ab={m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(da,db,k=2) if m.distance<.7*n.distance}
        ba={m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(db,da,k=2) if m.distance<.7*n.distance}
        pairs=np.array([(x,y) for x,y in ab.items() if ba.get(y)==x],dtype=int)
        if len(pairs)<12:continue
        _,valid=cv2.findFundamentalMat(a[pairs[:,0]].astype(np.float32),b[pairs[:,1]].astype(np.float32),cv2.FM_RANSAC,1.5,.999)
        if valid is None:continue
        for ai,bj in pairs[valid.ravel().astype(bool)]:
            if va[ai]:anchors[j].setdefault(int(bj),[]).append(wa[ai])
            if vb[bj]:anchors[i].setdefault(int(ai),[]).append(wb[bj])

refined=initial.copy();records=[]
rng=np.random.default_rng(42)
for i,observed in enumerate(anchors):
    ids=[];points=[]
    for feature,worlds in observed.items():
        if len(worlds)<2:continue
        center=np.median(worlds,axis=0)
        if np.max(np.linalg.norm(np.array(worlds)-center,axis=1))>.02*diagonal:continue
        ids.append(feature);points.append(center)
    if len(points)<args.minimum_features:
        records.append(dict(frame=i,accepted=False,reason='insufficient stable features',features=len(points)));continue
    points=np.asarray(points,dtype=np.float64);uv=features[i][0][ids].astype(np.float64)
    rv,_=cv2.Rodrigues(initial[i,:,:3].astype(np.float64));tv=initial[i,:,3].astype(np.float64)
    projected=cv2.projectPoints(points,rv,tv,k[i],None)[0][:,0]
    good=np.linalg.norm(projected-uv,axis=1)<15
    points,uv=points[good],uv[good]
    if len(points)<args.minimum_features:
        records.append(dict(frame=i,accepted=False,reason='inconsistent anchors',features=len(points)));continue
    if np.linalg.eigvalsh(np.cov(uv.T)).min() < 15**2:
        records.append(dict(frame=i,accepted=False,reason='features concentrated in a small image region'));continue
    order=rng.permutation(len(points));check=order[::5];train=np.setdiff1d(order,check)
    ok,r,t,inliers=cv2.solvePnPRansac(points[train],uv[train],k[i],None,rv.copy(),tv.copy(),True,
                                      iterationsCount=200,reprojectionError=3.,confidence=.999,flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inliers is None or len(inliers)<args.minimum_inliers:
        records.append(dict(frame=i,accepted=False,reason='PnP rejected'));continue
    r,t=cv2.solvePnPRefineLM(points[train[inliers.ravel()]],uv[train[inliers.ravel()]],k[i],None,r,t)
    before=np.linalg.norm(cv2.projectPoints(points[check],rv,tv,k[i],None)[0][:,0]-uv[check],axis=1)
    after=np.linalg.norm(cv2.projectPoints(points[check],r,t,k[i],None)[0][:,0]-uv[check],axis=1)
    rotation=cv2.Rodrigues(r)[0]
    original_center=-e[i,:,:3].T@e[i,:,3]
    new_center=-rotation.T@t.ravel()
    shift=float(np.linalg.norm(new_center-original_center)/diagonal)
    angle=float(np.degrees(np.arccos(np.clip((np.trace(rotation@e[i,:,:3].T)-1)/2,-1,1))))
    accepted=bool(np.median(after)<.8*np.median(before) and shift<.01 and angle<2. and np.median(after)<2.)
    records.append(dict(frame=i,accepted=accepted,features=len(points),validation_features=len(check),
                        before_median_px=float(np.median(before)),after_median_px=float(np.median(after)),
                        center_shift_scene_fraction=shift,rotation_degrees=angle))
    if accepted:refined[i]=np.column_stack([rotation,t])
report=dict(source_npz_sha256=hashlib.sha256(npz.read_bytes()).hexdigest(),world_geometry_fixed=True,
            features_per_image=args.features,image_scale=image_scale,base_cameras=str(args.base_cameras) if args.base_cameras else None,
            minimum_features=args.minimum_features,minimum_inliers=args.minimum_inliers,
            accepted_cameras=sum(r['accepted'] for r in records),total_cameras=len(d),records=records,
            corrected_camera_count=int(np.any(np.abs(refined-e)>1e-5,axis=(1,2)).sum()),
            camera_calibration_scope='Real SIFT matches from all frames, including appearance evaluation frames; each camera uses separate fit/check features',
            extrinsics=refined.tolist(),intrinsics=k.tolist())
args.output.write_text(json.dumps(report,indent=2))
print(json.dumps({key:val for key,val in report.items() if key not in {'records','extrinsics','intrinsics'}},indent=2))
print('Accepted',[(r['frame'],round(r['before_median_px'],2),round(r['after_median_px'],2)) for r in records if r['accepted']])
