"""Conservative image-supported storage solids, with independent-view checks.

This stage keeps SpatialLM architecture unchanged. It can replace a storage
instance's visible-point bounds only when its own masks support a front plane,
a rectangular side silhouette, and a held-out solid. Missing evidence preserves
the original box; rejection is a diagnostic result, never a fabricated depth.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np
from PIL import Image

from .furniture_quality import camera_viewpoints
from .image_furniture import lift_mask
from .storage_side_outline import trace_side_outline
from .storage_visibility import ray_box_depth, mask_metrics

STORAGE_LABELS=frozenset({'wardrobe','cabinet','bookshelf','locker','shelf','box'})


def _planes(points, scale):
    """Find distinct vertical patches without combining parallel offsets."""
    work=np.asarray(points,float);work=work[np.isfinite(work).all(axis=1)]
    if len(work)<80:return []
    work=work[np.linspace(0,len(work)-1,min(len(work),2500),dtype=int)]
    threshold=.012*scale;minimum=max(50,int(len(work)*.08));rng=np.random.default_rng(126)
    result=[]
    for _ in range(3):
        if len(work)<minimum:break
        best=None;count=0
        for _ in range(180):
            a,b=work[rng.choice(len(work),2,replace=False),:2];v=b-a
            if np.linalg.norm(v)<.15*scale:continue
            normal=np.array([-v[1],v[0]])/np.linalg.norm(v)
            distance=normal@a;inside=np.abs(work[:,:2]@normal-distance)<threshold
            if inside.sum()>count:best=inside;count=int(inside.sum())
        if best is None or count<minimum:break
        patch=work[best]
        for _ in range(3):
            center=patch[:,:2].mean(axis=0);_,_,v=np.linalg.svd(patch[:,:2]-center,full_matrices=False)
            normal=v[-1];distance=normal@center
            inside=np.abs(work[:,:2]@normal-distance)<threshold;patch=work[inside]
        work=work[~inside]
        tangent=np.array([-normal[1],normal[0]])
        lo,hi=np.quantile(np.column_stack([patch[:,:2]@tangent,patch[:,2]]),[.01,.99],axis=0)
        if hi[0]-lo[0]<.15*scale or hi[1]-lo[1]<.30*scale:continue
        result.append(dict(normal=normal,offset=float(distance),center=np.median(patch,axis=0),points=patch,
                           area=float(np.prod(hi-lo)),width=float(hi[0]-lo[0])))
    return result


def _front_candidate(observations, train_views, held_views, scale):
    patches=[]
    for obs in observations:
        if obs['viewpoint'] not in train_views:continue
        for p in _planes(obs['points'],scale):patches.append(dict(frame=obs['frame'],viewpoint=obs['viewpoint'],camera=obs['camera'],**p))
    choices=[]
    for seed in patches:
        matches=[p for p in patches if abs(p['normal']@seed['normal'])>np.cos(np.deg2rad(10))
                 and abs(p['center'][:2]@seed['normal']-seed['offset'])<.045*scale]
        views={p['viewpoint'] for p in matches}
        if len(views)<2:continue
        choices.append((len(views),float(np.median([p['area'] for p in matches])),seed,matches))
    if not choices:return None,'insufficient_independent_front_planes'
    _,_,seed,matches=max(choices,key=lambda x:(x[0],x[1]))
    normal=seed['normal'].copy()
    if np.median([normal@(p['camera'][:2]-p['center'][:2]) for p in matches])<0:normal=-normal
    # Give each independent camera group one normal/offset vote.
    view_points={}
    for p in matches:view_points.setdefault(p['viewpoint'],[]).append(p['points'])
    fitted=[]
    for pts in view_points.values():
        pp=np.concatenate(pts);ctr=pp[:,:2].mean(axis=0);_,_,v=np.linalg.svd(pp[:,:2]-ctr,full_matrices=False)
        nn=v[-1]
        if nn@normal<0:nn=-nn
        fitted.append(nn)
    normal=np.median(fitted,axis=0);normal/=np.linalg.norm(normal)
    tangent=np.array([-normal[1],normal[0]])
    front=float(np.median([np.median(np.concatenate(pts)[:,:2]@normal) for pts in view_points.values()]))
    all_points=np.concatenate([p['points'] for p in matches])
    lo,hi=np.quantile(np.column_stack([all_points[:,:2]@tangent,all_points[:,2]]),[.005,.995],axis=0)
    # Front extent and height are observations, with no floor/ceiling snapping.
    candidate=dict(front_normal=normal.tolist(),front=front,tangent_range=[float(lo[0]),float(hi[0])],
                   height_range=[float(lo[1]),float(hi[1])])
    held_support=[]
    for obs in observations:
        if obs['viewpoint'] not in held_views:continue
        pts=obs['points']
        aligned=pts[np.abs(pts[:,:2]@normal-front)<.045*scale]
        if len(aligned)>=60 and np.ptp(aligned[:,2])>.25*scale:held_support.append(obs['viewpoint'])
    if not held_support:return None,'held_front_plane_not_supported'
    candidate.update(front_train_viewpoints=sorted(view_points),front_held_viewpoints=sorted(set(held_support)))
    return candidate,None


def _load_masks(layout, prediction_dir, shape, frame_count):
    root=Path(prediction_dir).resolve();loaded={};invalid=set()
    for detection in layout.get('image_furniture',{}).get('detections',[]):
        identifier=detection.get('detection_id')
        if not identifier:continue
        identifier=str(identifier)
        if identifier in loaded or identifier in invalid:
            loaded.pop(identifier,None);invalid.add(identifier);continue
        try:
            frame=int(detection['frame']);path=(root/detection['mask']).resolve()
            if frame<0 or frame>=frame_count or not path.is_relative_to(root):raise ValueError('Invalid detection source')
            with Image.open(path) as image:mask=np.asarray(image.convert('L'))>0
            if mask.shape!=shape:raise ValueError('Mask coordinates differ from saved depth')
            loaded[identifier]=dict(detection,frame=frame,mask_array=mask)
        except (KeyError,ValueError,OSError,TypeError):invalid.add(identifier)
    return loaded,invalid


def _box_from_candidate(candidate, inverse):
    n=np.asarray(candidate['front_normal']);t=np.array([-n[1],n[0]])
    axes=np.array([[n[0],t[0],0.],[n[1],t[1],0.],[0.,0.,1.]])
    lo=np.array([candidate['back'],candidate['tangent_range'][0],candidate['height_range'][0]])
    hi=np.array([candidate['front'],candidate['tangent_range'][1],candidate['height_range'][1]])
    center=axes@((lo+hi)/2);size=hi-lo
    scale=float(np.linalg.norm(inverse[:3,0]));rotation=inverse[:3,:3]/scale
    return dict(center=center.tolist(),rotation=axes.tolist(),scale=size.tolist(),
                vg_world=dict(center=(inverse[:3,:3]@center+inverse[:3,3]).tolist(),
                              rotation=(rotation@axes).tolist(),scale=(size*scale).tolist()))


def _morph(mask, radius, erode=False):
    h,w=mask.shape;padded=np.pad(mask,radius,constant_values=False)
    result=np.ones_like(mask) if erode else np.zeros_like(mask)
    for y in range(2*radius+1):
        for x in range(2*radius+1):
            if erode:result&=padded[y:y+h,x:x+w]
            else:result|=padded[y:y+h,x:x+w]
    return result


def _fixed_instance_masks(frame_masks, loaded, identifiers):
    """Resolve ownership once, before the training/held split or any fitting.

    Substantial overlap or any interior conflict invalidates the instance.
    Tiny boundary conflicts are removed with a one-pixel guard. Trace guards
    retain the ambiguity so the cut cannot become an invented side endpoint.
    A coarse wall mask overlapping storage is ambiguity, not storage evidence.
    """
    fixed={};guards={};reports=[];reason=None
    for frame,target in frame_masks.items():
        unknown=np.zeros_like(target);other=np.zeros_like(target);wall=np.zeros_like(target)
        for identifier,detection in loaded.items():
            if identifier in identifiers or detection['frame']!=frame:continue
            kind=detection.get('kind','furniture');label=detection.get('label')
            if kind=='wall' or label=='wall':wall|=detection['mask_array']
            elif kind=='furniture':other|=detection['mask_array']
            else:unknown|=detection['mask_array']
        overlap=target&(unknown|other|wall)
        count=int(overlap.sum());total=int(target.sum())
        interior=overlap&_morph(target,2,erode=True)
        substantial=count>max(4,.02*total)
        if substantial or interior.any():reason='ambiguous_instance_mask_ownership'
        removed=_morph(overlap,1)&target
        fixed[frame]=target&~removed
        # Wall beyond the observed target may support a genuine side boundary;
        # only its conflicting overlap is unknown. Other objects and explicit
        # unknown pixels must never become a silhouette depth constraint.
        guards[frame]=_morph(unknown|other|(wall&target),1)
        reports.append(dict(frame=frame,target_pixels=total,ambiguous_pixels=count,
            unknown_overlap_pixels=int((target&unknown).sum()),other_overlap_pixels=int((target&other).sum()),
            wall_overlap_pixels=int((target&wall).sum()),interior_ambiguous_pixels=int(interior.sum()),
            removed_boundary_pixels=int(removed.sum()),fixed_target_pixels=int(fixed[frame].sum()),
            substantial_overlap=bool(substantial)))
    return fixed,guards,reports,reason


def refine_storage_layout(layout, prediction_dir, depths, confidence, extrinsics, intrinsics, transform, records):
    """Return a new layout; geometry is unchanged unless every acceptance gate passes.

    ``transform`` is transform.json (both world transforms), or its forward 4x4
    matrix. All masks must be fixed per-instance detection sources beneath
    prediction_dir. No VGGT rerun, camera mutation or wall correction occurs.
    """
    result=deepcopy(layout)
    depth=np.asarray(depths);conf=np.asarray(confidence);es=np.asarray(extrinsics);ks=np.asarray(intrinsics)
    if depth.ndim==4 and depth.shape[-1]==1:depth=depth[...,0]
    if conf.ndim==4 and conf.shape[-1]==1:conf=conf[...,0]
    if depth.ndim!=3 or conf.shape!=depth.shape or es.shape!=(len(depth),3,4) or ks.shape!=(len(depth),3,3) or len(records)!=len(depth):
        raise ValueError('Storage geometry input dimensions differ')
    tr=np.asarray(transform['vg_world_to_generation_world'] if isinstance(transform,dict) else transform,float)
    inverse=np.linalg.inv(tr)
    if tr.shape!=(4,4) or not np.isfinite(tr).all():raise ValueError('Invalid storage coordinate transform')
    scales=np.linalg.norm(tr[:3,:3],axis=0)
    if min(scales)<=0 or not np.allclose(tr[:3,:3].T@tr[:3,:3],np.eye(3)*scales[0]**2,rtol=1e-5,atol=1e-8):
        raise ValueError('Storage transform must be a uniform-scale rigid transform')
    viewpoints,_=camera_viewpoints(records,es)
    centers=(-np.einsum('nji,nj->ni',es[:,:,:3],es[:,:,3]))@tr[:3,:3].T+tr[:3,3]
    loaded,invalid=_load_masks(result,prediction_dir,depth.shape[1:],len(depth));reports=[]
    for box in result.get('boxes',[]):
        if box.get('class')!='bbox' or box.get('label') not in STORAGE_LABELS:continue
        report=dict(id=box.get('id'),status='preserved_original',accepted=False,reasons=[],wall_correction='deferred',
                    observed_surfaces=[],inferred_surfaces=[],assumptions=['upright rectangular storage','side perpendicular to front'],
                    complete_hidden_geometry_verified=False)
        identifiers=list(dict.fromkeys(str(x) for x in box.get('source_detection_ids',[])))
        report['source_detection_ids']=identifiers
        def finish(reason=None):
            if reason:report['reasons'].append(reason)
            box.setdefault('evidence',{})['storage_geometry']=deepcopy(report);reports.append(deepcopy(report))
            quality=box['evidence'].get('quality')
            summaries=result.get('image_furniture',{}).get('quality_report',{}).get('objects',[])
            matching=[q for q in summaries if str(q.get('id'))==str(box.get('id'))]
            if isinstance(quality,dict):matching.append(quality)
            for q in matching:
                q['geometry_status']='observed_envelope_with_inferred_closures' if report['accepted'] else 'visible_surface_bounds'
                q['storage_geometry_status']=report['status']
                if report['accepted']:
                    q['dimensions_model_units']=deepcopy(box['scale']);q['center_model_units']=deepcopy(box['center'])
                    q['hidden_surfaces_verified']=False
                else:
                    recommendation=dict(code='storage_side_depth_unresolved',message='収納の側面と後ろ端が見える写真を、位置を変えて追加してください。')
                    q.setdefault('needs_photo',[])
                    if not any(r.get('code')==recommendation['code'] for r in q['needs_photo']):q['needs_photo'].append(recommendation)
                    q['storage_geometry_reasons']=list(report['reasons'])
        if box.get('hidden'):finish('hidden_or_unconfirmed_instance');continue
        if not identifiers or any(i not in loaded or i in invalid for i in identifiers):finish('missing_or_invalid_instance_masks');continue
        selected=[loaded[i] for i in identifiers]
        if any(d.get('kind','furniture')!='furniture' or d.get('label') not in STORAGE_LABELS for d in selected):
            finish('ambiguous_instance_class');continue
        frame_masks={}
        for detection in selected:
            frame=detection['frame'];frame_masks[frame]=frame_masks.get(frame,np.zeros(depth.shape[1:],bool))|detection['mask_array']
        # An unreadable non-target mask cannot silently disappear from the
        # ownership check for a source frame.
        unreadable=[d for d in result.get('image_furniture',{}).get('detections',[])
                    if str(d.get('detection_id')) in invalid and d.get('frame') in frame_masks]
        if unreadable:finish('unreadable_conflicting_mask');continue
        frame_masks,outline_guards,ownership,ownership_reason=_fixed_instance_masks(frame_masks,loaded,set(identifiers))
        report['mask_ownership']=ownership
        report['mask_policy']='Fixed instance minus unknown/other/wall conflicts before train/held split; ambiguous outline transitions excluded'
        if ownership_reason:finish(ownership_reason);continue
        ordered=list(dict.fromkeys(viewpoints[f] for f in sorted(frame_masks)))
        if len(ordered)<3:finish('fewer_than_three_independent_viewpoints');continue
        held_views=set(ordered[2::3]);train_views=set(ordered)-held_views
        report.update(training_viewpoints=sorted(train_views),held_viewpoints=sorted(held_views),
                      split='Every third independent camera origin is held out before fitting')
        size=np.asarray(box.get('scale',[0,0,0]),float)
        if size.shape!=(3,) or not np.isfinite(size).all() or min(size)<=0:finish('invalid_original_dimensions');continue
        scale=float(max(size[2],np.linalg.norm(size[:2])*.25,1e-5))
        observations=[]
        for frame,mask in frame_masks.items():
            pts=lift_mask(mask,depth[frame],conf[frame],ks[frame],es[frame],tr)
            observations.append(dict(frame=frame,viewpoint=viewpoints[frame],camera=centers[frame],points=pts))
        candidate,reason=_front_candidate(observations,train_views,held_views,scale)
        if reason:finish(reason);continue
        report['candidate_front']=deepcopy(candidate)
        # Both caps may be observed. They must agree on a single thickness.
        outlines=[]
        width=float(np.diff(candidate['tangent_range'])[0]);height=float(np.diff(candidate['height_range'])[0])
        for frame,mask in frame_masks.items():
            for side in ('upper','lower'):
                # Narrow cabinets and boxes can be deeper than 30% of their
                # width. A wider search is still governed by observed edges
                # and held views, never by the rear wall position.
                trace=trace_side_outline(candidate,es[frame],ks[frame],tr,mask,forbidden_mask=outline_guards[frame],
                                         side=side,max_depth=min(1.25*width,height))
                if trace:outlines.append(dict(frame=frame,viewpoint=viewpoints[frame],held=viewpoints[frame] in held_views,**trace))
        report['side_observations']=[{k:v for k,v in r.items() if k!='rows'} for r in outlines]
        train_outline=[r for r in outlines if not r['held']];held_outline=[r for r in outlines if r['held']]
        side_views=sorted({r['viewpoint'] for r in train_outline})
        if len(side_views)<2 or not held_outline:finish('insufficient_independent_side_outline_support');continue
        by_view=[float(np.median([r['depth_median'] for r in train_outline if r['viewpoint']==v])) for v in side_views]
        thickness=float(np.median(by_view));tolerance=max(.035*scale,.10*thickness)
        if np.ptp(by_view)>2*tolerance:finish('training_side_depth_disagrees');continue
        if any(abs(r['depth_median']-thickness)>tolerance for r in held_outline):finish('held_side_depth_disagrees');continue
        candidate['back']=candidate['front']-thickness
        report.update(depth_estimate=thickness,depth_train_view_range=[min(by_view),max(by_view)],
                      side_depth_interpretation='Observed side silhouette intersected with assumed perpendicular side; no wall distance',
                      observed_surfaces=['front point samples','side outline'],
                      inferred_surfaces=['rear closure','bottom closure','unobserved side/top portions'])
        validation=[]
        for frame,target in sorted(frame_masks.items()):
            other=np.zeros_like(target);wall=np.zeros_like(target)
            for identifier,detection in loaded.items():
                if detection['frame']!=frame or identifier in identifiers:continue
                if detection.get('kind')=='wall' or detection.get('label')=='wall':wall|=detection['mask_array']
                elif detection.get('kind','furniture')=='furniture':other|=detection['mask_array']
            predicted=ray_box_depth(candidate,es[frame],ks[frame],tr,*depth.shape[1:])
            metrics,_=mask_metrics(predicted,depth[frame],conf[frame],target,other,wall)
            metrics.update(frame=frame,viewpoint=viewpoints[frame],held=viewpoints[frame] in held_views)
            validation.append(metrics)
        report['validation']=validation
        usable=[m for m in validation if m['target_pixels']>=100]
        failed=[m for m in usable if m['precision']<.85 or m['recall']<.80 or m['iou']<.72
                or m['target_depth_pixels']<30 or m.get('target_depth_contradiction_fraction',1)>.12
                or m.get('target_relative_depth_p90',1)>.06 or m['wall_block_fraction']>.02]
        if not any(m['held'] for m in usable):finish('insufficient_held_instance_pixels');continue
        if failed:
            report['failed_validation_frames']=[m['frame'] for m in failed]
            finish('source_instance_geometry_contradiction');continue
        # Replacement must not teleport to an unrelated plane in an overmerged group.
        replacement=_box_from_candidate(candidate,inverse)
        if np.linalg.norm(np.asarray(replacement['center'])-np.asarray(box['center']))>np.linalg.norm(size)*.65:
            finish('candidate_disagrees_with_original_instance_location');continue
        box.update(replacement)
        box['constraint']='Observed front and multi-view side outline; hidden closures remain inferred'
        report.update(status='accepted_observed_envelope',accepted=True,candidate=deepcopy(candidate),
                      units='generation model units; physical scale not independently verified')
        finish()
    summary=dict(version=1,objects=reports,status='evaluated' if reports else 'no_storage_instances',
        accepted_count=sum(r['accepted'] for r in reports),preserved_count=sum(not r['accepted'] for r in reports),wall_correction='deferred',
        policy='Preserve original geometry unless unambiguous fixed instance masks, independent side outlines, and every significant source view agree; held views are never fitted',
        thresholds=dict(mask_precision=.85,mask_recall=.80,mask_iou=.72,target_relative_depth_p90=.06,
                        target_depth_contradiction_fraction=.12,wall_block_fraction=.02))
    result.setdefault('image_furniture',{})['storage_geometry']=summary
    result['storage_geometry']=deepcopy(summary)
    return result
