"""Estimate rectangular storage depth from a visible side silhouette.

The front plane and its tangent extent must be fitted independently. This
estimator explicitly assumes an upright side perpendicular to that front.
No observed depth, depth confidence, or wall distance enters the estimator.
"""
from __future__ import annotations
import numpy as np


def trace_side_outline(candidate, extrinsic, intrinsic, transform, storage_mask,
                       wall_mask=None, *, forbidden_mask=None, side='upper', max_depth=None,
                       depth_samples=601, vertical_samples=21, min_rows=4,
                       min_projected_width=12.):
    """Return a per-view side-depth proposal or None if no side edge is seen.

    Coordinates match storage_candidates.py. Optical-axis extrinsics map saved
    world coordinates to camera coordinates; transform maps saved world to
    generation coordinates. ``wall_mask`` is optional diagnostics only.
    Estimates are in the candidate's assumed scene units, not physical metres.
    """
    n=np.asarray(candidate['front_normal'],float);n=n/np.linalg.norm(n)
    tangent=np.array([-n[1],n[0]])
    front=float(candidate['front'])
    low,high=np.asarray(candidate['tangent_range'],float)
    bottom,top=np.asarray(candidate['height_range'],float)
    height=top-bottom;length=high-low
    if height<=0 or length<=0:raise ValueError('Candidate extents must be positive')
    if side not in ('upper','lower'):raise ValueError('side must be upper or lower')
    if depth_samples<30 or vertical_samples<min_rows:raise ValueError('Insufficient trace samples')
    sign=1. if side=='upper' else -1.;edge=high if sign>0 else low
    es=np.asarray(extrinsic,float);ks=np.asarray(intrinsic,float);tr=np.asarray(transform,float)
    storage=np.asarray(storage_mask,bool)
    if storage.ndim!=2:raise ValueError('Expected a 2D storage mask')
    wall=None if wall_mask is None else np.asarray(wall_mask,bool)
    if wall is not None and wall.shape!=storage.shape:raise ValueError('Mask sizes must match')
    forbidden=None if forbidden_mask is None else np.asarray(forbidden_mask,bool)
    if forbidden is not None and forbidden.shape!=storage.shape:raise ValueError('Forbidden mask size must match')
    camera=-es[:,:3].T@es[:,3];camera=tr[:3,:3]@camera+tr[:3,3]
    # Both the front and requested side must face the camera. A side that faces
    # away cannot be measured by following a silhouette behind the front.
    if sign*(camera[:2]@tangent-edge)<=.02*height or camera[:2]@n<=front:return None
    maximum=float(max_depth) if max_depth is not None else min(.30*length,.75*height)
    start=.01*height;minimum=.10*height
    if maximum<=minimum:raise ValueError('Maximum search depth is too small')
    ds=np.linspace(start,maximum,depth_samples);step=ds[1]-ds[0]
    gap=max(3,int(np.ceil(.025*height/step)))
    inv=np.linalg.inv(tr);h,w=storage.shape
    heights=np.linspace(bottom+.10*height,top-.10*height,vertical_samples)
    rows=[]
    for z in heights:
        xy=n[None,:]*(front-ds[:,None])+tangent[None,:]*edge
        points=np.column_stack((xy,np.full(len(ds),z)))
        world=points@inv[:3,:3].T+inv[:3,3]
        cam=world@es[:,:3].T+es[:,3];uv=cam@ks.T
        uv=uv[:,:2]/np.where(np.abs(cam[:,2,None])>1e-12,cam[:,2,None],np.nan)
        finite=np.isfinite(uv).all(axis=1)
        pix=np.rint(np.where(np.isfinite(uv),uv,-1)).astype(int)
        inside=finite&(pix[:,0]>=0)&(pix[:,0]<w)&(pix[:,1]>=0)&(pix[:,1]<h)&(cam[:,2]>0)
        if not inside[0]:continue
        if np.linalg.norm(uv[len(ds)//2]-uv[0])<min_projected_width:continue
        sm=np.zeros(len(ds),bool);sm[inside]=storage[pix[inside,1],pix[inside,0]]
        if sm[:5].mean()<.8:continue
        changes=[j for j in range(5,len(ds)-gap)
                 if inside[j:j+gap].all() and not sm[j:j+gap].any()
                 and sm[max(0,j-gap):j].mean()>.7 and ds[j]>=minimum]
        if not changes:continue
        j=changes[0]
        # Removing ambiguous pixels must not fabricate a new rear endpoint.
        # Nor can a foreground/unknown interruption count as a visible edge.
        if forbidden is not None:
            valid=np.flatnonzero(inside[:j+gap])
            if forbidden[pix[valid,1],pix[valid,0]].any():continue
        row=dict(height=float(z),depth=float(ds[j]),uv=uv[j].tolist())
        if wall is not None:
            row['wall_after_fraction']=float(wall[pix[j:j+gap,1],pix[j:j+gap,0]].mean())
        rows.append(row)
    if len(rows)<min_rows:return None
    values=np.array([r['depth'] for r in rows])
    return dict(depth_median=float(np.median(values)),depth_q10_q90=np.quantile(values,[.1,.9]).tolist(),
                side=side,side_edge=float(edge),rows=rows,
                search_depth_range=[float(start),maximum],sample_height_range=[float(heights[0]),float(heights[-1])],
                assumptions=['vertical rectangular side','side perpendicular to independent front plane','side at independent front tangent endpoint'],
                complete_solid_accepted=False)
