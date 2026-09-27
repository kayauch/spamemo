"""Reversible VGGT-world -> Z-up metric-hypothesis adapter; never edits reality."""
import numpy as np
from copy import deepcopy


def transform_points(points, matrix):
    points=np.asarray(points,dtype=np.float64)
    return points@np.asarray(matrix)[:3,:3].T+np.asarray(matrix)[:3,3]


def generation_transform(points, room_frame, extrinsics, assumed_height_m=2.5, camera_up_axis='image-top', preserve_metric_scale=False):
    if not .5<=assumed_height_m<=10: raise ValueError('Invalid assumed room height')
    points=np.asarray(points,dtype=np.float64)
    poses=np.asarray(extrinsics,dtype=np.float64)
    if points.ndim!=2 or points.shape[1]!=3:
        raise ValueError('Expected Nx3 point evidence')
    if poses.ndim!=3 or poses.shape[1:]!=(3,4) or not len(poses) or not np.isfinite(poses).all():
        raise ValueError('Expected finite OpenCV world-to-camera poses')
    if not room_frame.get('snap_enabled'): raise ValueError('Room frame needs review before normalization')
    if not np.isfinite(points).all() or len(points)<10: raise ValueError('Invalid point evidence')
    basis=np.asarray(room_frame['room_to_world'],dtype=np.float64)[:3,:3]
    if not np.allclose(basis.T@basis,np.eye(3),atol=1e-5) or np.linalg.det(basis)<.999:
        raise ValueError('Room frame must be a proper rotation')
    # OpenCV camera +Y points down; most room captures are upright.
    axes={'image-top':(1,-1),'image-left':(0,-1),'image-right':(0,1),'image-bottom':(1,1)}
    if camera_up_axis not in axes:raise ValueError('Invalid image up direction')
    axis,sign=axes[camera_up_axis]
    camera_ups=sign*poses[:,axis,:3]
    camera_up=np.median(camera_ups,axis=0)
    norm=np.linalg.norm(camera_up)
    if norm<.5: raise ValueError('Camera up directions are ambiguous')
    camera_up/=norm
    vertical_axis=1 if camera_up_axis=='image-top' else int(np.argmax(np.abs(basis.T@camera_up)))
    up=basis[:,vertical_axis].copy()
    dot=float(up@camera_up)
    if abs(dot)<.5: raise ValueError('Camera and structural vertical disagree')
    flipped=dot<0
    if flipped: up=-up
    horizontal_axis=0 if vertical_axis!=0 else 2
    x=basis[:,horizontal_axis]-up*(basis[:,horizontal_axis]@up); x/=np.linalg.norm(x)
    y=np.cross(up,x)
    rotation=np.stack((x,y,up))
    aligned=points@rotation.T
    lower,upper=np.quantile(aligned,[.005,.995],axis=0)
    height=upper[2]-lower[2]
    if height<=1e-6: raise ValueError('Degenerate vertical extent')
    scale=1.0 if preserve_metric_scale else assumed_height_m/height
    matrix=np.eye(4);matrix[:3,:3]=scale*rotation;matrix[:3,3]=-scale*lower
    inverse=np.linalg.inv(matrix)
    return dict(vg_world_to_generation_world=matrix.tolist(),generation_world_to_vg_world=inverse.tolist(),
        scale_m_per_vggt_unit=scale,assumed_height_m=assumed_height_m,metric_scale_verified=False,
        metric_scale_calibrated=bool(preserve_metric_scale),
        unit='meters from calibrated RGB-D input' if preserve_metric_scale else 'meter hypothesis; height assumption, not a measurement',up_sign_flipped_from_existing_frame=flipped,
        camera_up_alignment=abs(dot),camera_up_axis=camera_up_axis,structural_vertical_axis=vertical_axis,room_frame_confidence=float(room_frame['confidence']),
        axis_convention='right-handed Z-up, X/Y aligned to existing Manhattan frame',
        policy='preserve calibrated input scale; rotate and translate only' if preserve_metric_scale else 'robust 0.5%-99.5% height; source frame unchanged')


def box_evidence(center,rotation,extent,points,allow_planar=False):
    center=np.asarray(center);rotation=np.asarray(rotation);extent=np.asarray(extent)
    if center.shape!=(3,) or rotation.shape!=(3,3) or extent.shape!=(3,):
        return dict(valid=False,reason='invalid box shapes')
    dimensions_valid=bool(np.all(extent>0) or (allow_planar and extent[0]>0 and extent[1]==0 and extent[2]>0))
    if not dimensions_valid or not np.isfinite(np.r_[center,extent,rotation.ravel()]).all():
        return dict(valid=False,reason='nonfinite or nonpositive dimensions')
    if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5) or np.linalg.det(rotation)<.999:
        return dict(valid=False,reason='invalid box rotation')
    local=(np.asarray(points)-center)@rotation
    inside=(np.abs(local)<=extent/2+.05).all(axis=1)
    return dict(valid=True,points_inside=int(inside.sum()),surface_only=bool(extent[1]==0),support_margin_m=.05,
                evidence_type='geometric occupancy only; not semantic verification',confidence=None)


def vertical_anchors(components, matrix, assumed_height_m=None):
    """Use observed near-horizontal planes; never trust old floor/ceiling labels."""
    matrix=np.asarray(matrix,dtype=float)
    planes=[]
    for comp in components:
        normal=comp.get('metadata',{}).get('normal')
        if comp.get('source')!='plane' or normal is None or comp.get('point_count',0)<300:
            continue
        n=matrix[:3,:3]@np.asarray(normal)
        if not np.isfinite(n).all() or np.linalg.norm(n)<1e-8:continue
        n/=np.linalg.norm(n)
        if abs(n[2])<.98:continue
        vertices=transform_points(comp['vertices'],matrix)
        if not np.isfinite(vertices).all():continue
        planes.append(dict(source_component=comp['id'],z=float(np.median(vertices[:,2])),
            source_label=comp['kind'],points=comp['point_count'],normal_alignment=abs(float(n[2]))))
    if not planes:raise ValueError('Need at least one supported horizontal plane')
    if len(planes)<2 and assumed_height_m is None:raise ValueError('Need two supported horizontal planes')
    planes.sort(key=lambda p:p['z']);floor,ceiling=planes[0],planes[-1]
    if not 1<=ceiling['z']-floor['z']<=5:
        if assumed_height_m is None:raise ValueError('Ambiguous floor/ceiling separation')
        if not 1<=assumed_height_m<=5:raise ValueError('Invalid assumed ceiling height')
        ceiling=dict(z=floor['z']+assumed_height_m,source_component=None,source_label='assumed_ceiling',points=0)
        return dict(floor=floor,ceiling=ceiling,metric_scale_verified=False,ceiling_observed=False,
                    method='lowest horizontal plane as floor candidate; ceiling height assumed, not observed')
    return dict(floor=floor,ceiling=ceiling,metric_scale_verified=False,
        method='horizontal geometry + camera-aligned vertical; old labels ignored')


def anchor_wall_heights(boxes, anchors):
    """Bounded structural correction only. Furniture and openings are preserved."""
    result=deepcopy(boxes);changes=[]
    low,high=anchors['floor']['z'],anchors['ceiling']['z']
    for box in result:
        if box['class']!='wall':continue
        # This adapter only changes vertical walls; a rotated local Z would be wrong.
        r=np.asarray(box['rotation'])
        if not np.allclose(r[:,2],[0,0,1],atol=1e-4):continue
        old_bottom=box['center'][2]-box['scale'][2]/2
        old_top=box['center'][2]+box['scale'][2]/2
        if abs(low-old_bottom)>.3 or abs(high-old_top)>.6:continue
        box['center'][2]=(low+high)/2;box['scale'][2]=high-low
        changes.append(dict(id=box['id'],before=[old_bottom,old_top],after=[low,high]))
    return result,changes
