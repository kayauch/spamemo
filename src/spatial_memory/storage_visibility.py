"""Dense ray-box visibility, with explicit target depth contradictions.

No surface sampling and no fitting to evaluation images. Distances along rays
are optical-axis depth in the original saved camera convention.
"""
from __future__ import annotations
import numpy as np


def ray_box_depth(candidate,extrinsic,intrinsic,transform,height,width):
    n=np.asarray(candidate['front_normal'],float);t=np.array([-n[1],n[0]])
    axes=np.array([[n[0],t[0],0],[n[1],t[1],0],[0,0,1.]])
    low=np.array([candidate['back'],candidate['tangent_range'][0],candidate['height_range'][0]])
    high=np.array([candidate['front'],candidate['tangent_range'][1],candidate['height_range'][1]])
    yy,xx=np.indices((height,width));uv=np.stack((xx,yy,np.ones_like(xx)),axis=-1)
    rays=uv@np.linalg.inv(intrinsic).T@extrinsic[:,:3]
    world_center=-extrinsic[:,:3].T@extrinsic[:,3]
    center=np.asarray(transform)[:3,:3]@world_center+np.asarray(transform)[:3,3]
    rays=rays@np.asarray(transform)[:3,:3].T
    origin=center@axes;direction=rays@axes
    parallel=np.abs(direction)<1e-12
    divisor=np.where(parallel,1,direction)
    a=(low-origin)/divisor;b=(high-origin)/divisor
    near=np.where(parallel,-np.inf,np.minimum(a,b));far=np.where(parallel,np.inf,np.maximum(a,b))
    enter=np.max(near,axis=-1);leave=np.min(far,axis=-1)
    invalid_parallel=np.any(parallel&((origin<low-1e-10)|(origin>high+1e-10)),axis=-1)
    # Numerically equal slab entry/exit at edges must remain intersections
    # after a rotated/scaled coordinate round trip.
    hit=(leave+1e-10>=np.maximum(enter,0))&(leave>0)&~invalid_parallel
    depth=np.where(hit,np.where(enter>0,enter,leave),np.inf)
    return depth


def mask_metrics(predicted_depth,observed_depth,confidence,target,other,wall):
    projected=np.isfinite(predicted_depth)
    reliable=np.isfinite(observed_depth)&(observed_depth>0)&np.isfinite(confidence)&(confidence>1.2)
    # Only a separately identified foreground is allowed to occlude the target.
    # A predicted target lying behind its own observed mask is a depth error.
    known_occlusion=projected&other&~target&reliable&(predicted_depth>observed_depth*1.04)
    visible=projected&~known_occlusion
    union=visible|target
    intersection=visible&target
    depth_region=projected&target&reliable
    rel=np.abs(predicted_depth[depth_region]-observed_depth[depth_region])/observed_depth[depth_region]
    wall_block=visible&wall&reliable&(predicted_depth<observed_depth*.97)
    m=dict(projected_pixels=int(projected.sum()),known_occluded_pixels=int(known_occlusion.sum()),
        predicted_visible_pixels=int(visible.sum()),target_pixels=int(target.sum()),
        precision=float(intersection.sum()/max(visible.sum(),1)),
        recall=float(intersection.sum()/max(target.sum(),1)),iou=float(intersection.sum()/max(union.sum(),1)),
        target_depth_pixels=int(depth_region.sum()),
        wall_block_pixels=int(wall_block.sum()),wall_block_fraction=float(wall_block.sum()/max(visible.sum(),1)),
        unverified_predicted_fraction=float((visible&~reliable).sum()/max(visible.sum(),1)))
    if len(rel):m.update(target_relative_depth_median=float(np.median(rel)),target_relative_depth_p90=float(np.quantile(rel,.9)),
        target_depth_contradiction_fraction=float(np.mean(rel>.04)))
    return m,dict(projected=projected,visible=visible,intersection=intersection,false_positive=visible&~target,
        missed=target&~visible,depth_region=depth_region,wall_block=wall_block,known_occlusion=known_occlusion)
