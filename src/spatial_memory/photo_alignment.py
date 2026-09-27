"""Bounded room plane alignment against fixed photographic depth evidence.

This is a local correction, not recovery of unobserved structure. Independent
camera origins are held out before fitting; rejected proposals preserve input.
"""
from copy import deepcopy
import numpy as np
from .solid_room import wall_loop, triangulate_polygon
from .furniture_quality import camera_viewpoints


def inside_polygon(polygon, points):
    inside=np.zeros(len(points),bool);x,y=np.asarray(points).T
    for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
        if abs(b[1]-a[1])<1e-12:continue
        inside^=((a[1]>y)!=(b[1]>y))&(x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])
    return inside


def polygon_area(p):
    return abs(float(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1))))/2


def simple_polygon(p):
    def cross(a,b,c):return float(np.cross(b-a,c-a))
    for i in range(len(p)):
        a,b=p[i],p[(i+1)%len(p)]
        for j in range(i+1,len(p)):
            if j==i+1 or (i==0 and j==len(p)-1):continue
            c,d=p[j],p[(j+1)%len(p)]
            if cross(a,b,c)*cross(a,b,d)<=0 and cross(c,d,a)*cross(c,d,b)<=0:
                if np.all(np.maximum(np.minimum(a,b),np.minimum(c,d))<=np.minimum(np.maximum(a,b),np.maximum(c,d))+1e-9):return False
    return True


def fit_offset(values, groups, old, max_shift=.30):
    values=np.asarray(values, float); groups=np.asarray(groups)
    keep=np.isfinite(values)&(np.abs(values-old)<max_shift)
    values,groups=values[keep],groups[keep]
    # Each camera origin gets equal weight, regardless of pixel density.
    origins=sorted(set(groups.tolist()))
    medians=np.array([np.median(values[groups==g]) for g in origins if np.sum(groups==g)>=20])
    report=dict(accepted=False, before=float(old), after=float(old), points=len(values),
                viewpoints=len(medians), reason='insufficient_independent_views')
    if len(medians)<4:return old,report
    held=np.arange(len(medians))%3==0
    proposed=float(np.median(medians[~held]))
    before=np.abs(medians[held]-old);after=np.abs(medians[held]-proposed)
    report.update(proposed=proposed,heldout_viewpoints=int(held.sum()),
                  before_error=float(np.median(before)),after_error=float(np.median(after)),
                  shift=proposed-float(old))
    if abs(proposed-old)<.005:report['reason']='already_aligned'
    elif abs(proposed-old)>max_shift:report['reason']='correction_too_large'
    elif np.median(after)>.85*np.median(before) or np.max(after-before)>.025:
        report['reason']='heldout_views_disagree'
    elif np.quantile(np.abs(medians-proposed),.8)>.10:
        report['reason']='inconsistent_depth'
    else:
        report.update(accepted=True,after=proposed,reason='heldout_improved')
        return proposed,report
    return old,report


def offset_polygon(footprint, offsets):
    p=np.asarray(footprint,float);edge=np.roll(p,-1,axis=0)-p
    normals=np.column_stack([-edge[:,1],edge[:,0]])/np.linalg.norm(edge,axis=1)[:,None]
    distances=np.einsum('ij,ij->i',normals,p)+np.asarray(offsets)
    moved=[]
    for i in range(len(p)):
        a=np.stack([normals[i-1],normals[i]])
        if abs(np.linalg.det(a))<1e-5:
            if abs(offsets[i-1]-offsets[i])>.005:raise ValueError('Parallel wall proposals conflict')
            moved.append(p[i]+normals[i]*offsets[i])
        else:moved.append(np.linalg.solve(a,[distances[i-1],distances[i]]))
    moved=np.asarray(moved)
    if np.max(np.linalg.norm(moved-p,axis=1))>.45:raise ValueError('Corner correction exceeds bounds')
    if np.any(np.sum((np.roll(moved,-1,axis=0)-moved)*edge,axis=1)<=.02):raise ValueError('Wall would collapse')
    # Validate simple, noncollapsed concave polygons, not just triangle output.
    if not simple_polygon(moved) or not .8<polygon_area(moved)/polygon_area(p)<1.2:raise ValueError('Room topology or area changed')
    triangulate_polygon(moved)
    return moved


def align_room_planes(layout, depths, confidence, es, ks, matrix, records, masks):
    source=deepcopy(layout);p=wall_loop(source['boxes']);floor=source['vertical_anchors']['floor']['z'];ceiling=source['vertical_anchors']['ceiling']['z']
    groups,_=camera_viewpoints(records,es)
    edges=np.roll(p,-1,axis=0)-p;length=np.linalg.norm(edges,axis=1)
    tangent=edges/length[:,None];normal=np.column_stack([-tangent[:,1],tangent[:,0]])
    d=np.einsum('ij,ij->i',normal,p)
    names=['floor','ceiling']+[f'wall-{i}' for i in range(len(p))]
    observations={name:[] for name in names};origins={name:[] for name in names}
    m=np.asarray(matrix)
    for i,depth in enumerate(depths):
        h,w=depth.shape;y,x=np.mgrid[3:h-3:5,3:w-3:5]
        uv=np.stack([x,y,np.ones_like(x)],-1)
        cam=(uv@np.linalg.inv(ks[i]).T)*depth[y,x,None]
        xyz=((cam-es[i,:,3])@es[i,:,:3])@m[:3,:3].T+m[:3,3]
        n=np.cross(xyz[1:-1,2:]-xyz[1:-1,:-2],xyz[2:,1:-1]-xyz[:-2,1:-1])
        n/=np.maximum(np.linalg.norm(n,axis=2,keepdims=True),1e-9)
        q=xyz[1:-1,1:-1].reshape(-1,3);n=n.reshape(-1,3)
        pixels=uv[1:-1,1:-1,:2].reshape(-1,2)
        valid=(confidence[i,y,x][1:-1,1:-1].ravel()>1.2)&np.isfinite(q).all(1)&np.isfinite(n).all(1)
        inside=inside_polygon(p,q[:,:2])
        for name,z in [('floor',floor),('ceiling',ceiling)]:
            keep=valid&inside&(abs(n[:,2])>.97)&(abs(q[:,2]-z)<.30)&masks.allowed(i,pixels,name)
            values=q[keep,2];observations[name].extend(values.tolist());origins[name].extend([groups[i]]*len(values))
        wall_allowed=valid&masks.allowed(i,pixels,'wall')
        for j in range(len(p)):
            relative=q[:,:2]-p[j];along=relative@tangent[j];distance=q[:,:2]@normal[j]-d[j]
            keep=wall_allowed&(abs(n[:,:2]@normal[j])>.95)&(abs(distance)<.30)&(along>.03)&(along<length[j]-.03)&(q[:,2]>floor+.15)&(q[:,2]<ceiling-.15)
            values=q[keep,:2]@normal[j];observations[names[j+2]].extend(values.tolist());origins[names[j+2]].extend([groups[i]]*len(values))
    reports=[];newvalues=[]
    for name,old in zip(names,[floor,ceiling,*d]):
        value,report=fit_offset(observations[name],origins[name],old)
        report['surface_id']=name;reports.append(report);newvalues.append(value)
    if not 1<newvalues[1]-newvalues[0]<5:
        for j in (0,1):reports[j].update(accepted=False,after=[floor,ceiling][j],reason='invalid_room_height');newvalues[j]=[floor,ceiling][j]
    try:newp=offset_polygon(p,np.asarray(newvalues[2:])-d)
    except ValueError as error:
        newp=p
        for report in reports[2:]:report.update(accepted=False,after=report['before'],reason='unsafe_shared_corners',detail=str(error))
    # Update the original wall objects by their endpoints; keep identity and rotation.
    for box in source['boxes']:
        if box['class']!='wall':continue
        r=np.asarray(box['rotation']);axis=int(np.argmax(np.asarray(box['scale'])[:2]));delta=r[:2,axis]*box['scale'][axis]/2
        endpoints=[np.asarray(box['center'])[:2]-delta,np.asarray(box['center'])[:2]+delta]
        ids=[int(np.argmin(np.linalg.norm(p-e,axis=1))) for e in endpoints]
        if any(np.linalg.norm(p[j]-e)>.016 for j,e in zip(ids,endpoints)):raise ValueError('Wall corner correspondence lost')
        a,b=newp[ids];box['center'][:2]=((a+b)/2).tolist();box['scale'][axis]=float(np.linalg.norm(b-a))
        box['center'][2]=(newvalues[0]+newvalues[1])/2;box['scale'][2]=newvalues[1]-newvalues[0]
        box.pop('vg_world',None)
    for name,value in zip(('floor','ceiling'),newvalues):source['vertical_anchors'][name]['z']=float(value)
    wall_loop(source['boxes'])
    return source,dict(method='normal_filtered_depth_planes_with_heldout_camera_origins',surfaces=reports,
        accepted_count=sum(r['accepted'] for r in reports),before_footprint=p.tolist(),after_footprint=newp.tolist(),
        before_heights=[floor,ceiling],after_heights=newvalues[:2],
        limitations=['Local parallel plane offsets only; no invented surfaces or tolerance widening.',
                    'Depth and semantic labels are estimates. Model units are not verified metres.',
                    'Furniture geometry and unseen surfaces remain unchanged.'])
