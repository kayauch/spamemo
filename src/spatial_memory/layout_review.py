"""Explicit photo annotations and depth evidence; never automatic semantic truth."""
from copy import deepcopy
import numpy as np
from .generation_evidence import transform_points


def lift_edge(segment, depth, confidence, intrinsic, extrinsic, world_transform, min_confidence=1.2):
    uv=np.linspace(segment[0],segment[1],80)
    pixels=np.rint(uv).astype(int)
    h,w=depth.shape
    valid=(pixels[:,0]>=0)&(pixels[:,0]<w)&(pixels[:,1]>=0)&(pixels[:,1]<h)
    pixels=pixels[valid];uv=uv[valid]
    _,unique=np.unique(pixels,axis=0,return_index=True);pixels=pixels[unique];uv=uv[unique]
    d=depth[pixels[:,1],pixels[:,0]];c=confidence[pixels[:,1],pixels[:,0]]
    valid=np.isfinite(d)&(d>0)&np.isfinite(c)&(c>=min_confidence)
    rays=np.column_stack((uv[valid],np.ones(valid.sum())))@np.linalg.inv(intrinsic).T
    camera=rays*d[valid,None]
    world=(camera-extrinsic[:,3])@extrinsic[:,:3]
    return transform_points(world,world_transform)


def supported_edge_height(box, points):
    xyz=np.asarray(points,dtype=float)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)<20:return None
    xyz=xyz[np.isfinite(xyz).all(1)]
    local=(xyz-np.asarray(box['center']))@np.asarray(box['rotation'])
    near=(np.abs(local[:,:2])<=np.asarray(box['scale'])[:2]/2+.25).all(1)
    z=xyz[near,2]
    if len(z)<20 or np.quantile(z,.9)-np.quantile(z,.1)>.2:return None
    return float(np.median(z))


def top_proposal(box, observations, max_shift=.5, agreement=.12):
    """Agree across >=2 distinct photographs; keep bottom, width and thickness."""
    heights=[];frames=[]
    for obs in observations:
        height=supported_edge_height(box,obs['points'])
        if height is None:continue
        if obs['frame'] in frames:continue
        heights.append(height);frames.append(obs['frame'])
    old=box['center'][2]+box['scale'][2]/2
    bottom=box['center'][2]-box['scale'][2]/2
    report=dict(frames=frames,heights=heights,old_top=old,accepted=False)
    if len(heights)<2:return deepcopy(box),dict(report,reason='insufficient independent supported edges')
    top=float(np.median(heights));report['proposed_top']=top
    if np.ptp(heights)>agreement or abs(top-old)>max_shift or top-bottom<.3:
        return deepcopy(box),dict(report,reason='inconsistent or excessive correction')
    result=deepcopy(box);result['center'][2]=(bottom+top)/2;result['scale'][2]=top-bottom
    return result,dict(report,accepted=True,reason='photo-edge heights agree; non-height dimensions unchanged')
