"""Surfaces from an inferred layout; photographs are evidence, not generated truth."""
import numpy as np


def triangulate_polygon(polygon):
    """Ear clipping for a simple concave perimeter; preserves every boundary edge."""
    p = np.asarray(polygon, dtype=float)
    if len(p) < 3 or not np.isfinite(p).all():
        raise ValueError('Invalid room perimeter')
    cross = lambda a, b: a[0]*b[1]-a[1]*b[0]
    area = sum(cross(a, b) for a, b in zip(p, np.roll(p, -1, axis=0)))
    if abs(area) < 1e-8:
        raise ValueError('Room perimeter has no area')
    indices = list(range(len(p))) if area > 0 else list(range(len(p)-1, -1, -1))
    triangles = []
    while len(indices) > 3:
        for j, b in enumerate(indices):
            a, c = indices[j-1], indices[(j+1) % len(indices)]
            if cross(p[b]-p[a], p[c]-p[b]) <= 1e-9:
                continue
            def contains(k):
                return all(cross(p[y]-p[x], p[k]-p[x]) >= -1e-9
                           for x, y in ((a,b),(b,c),(c,a)))
            if any(contains(k) for k in indices if k not in (a,b,c)):
                continue
            triangles.append([a,b,c]); indices.pop(j); break
        else:
            raise ValueError('Room perimeter intersects itself or cannot be triangulated')
    triangles.append(indices)
    return np.asarray(triangles, dtype=int)


def closed_shell(footprint, floor, ceiling):
    if not np.isfinite([floor, ceiling]).all() or ceiling-floor < .5:
        raise ValueError('Invalid room height')
    p = np.asarray(footprint)
    triangles = triangulate_polygon(p)
    count = len(p)
    vertices = np.vstack([np.column_stack([p, np.full(count,z)]) for z in (floor,ceiling)])
    area = np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1))
    faces = triangles[:,::-1].tolist() + (triangles+count).tolist()
    for i in range(count):
        j = (i+1) % count
        wall = [[i,j,j+count],[i,j+count,i+count]]
        faces.extend(wall if area > 0 else [t[::-1] for t in wall])
    return vertices, np.asarray(faces)


def furniture_parts(box, floor):
    """Simple closed solids. Support legs are explicit inferred completion."""
    if box.get('hidden'):
        return []
    c, r, s = (np.asarray(box[key], dtype=float) for key in ('center','rotation','scale'))
    if not np.isfinite(np.concatenate([c,r.ravel(),s])).all() or np.min(s) <= .015:
        return []
    label = box['label']
    hints = box.get('evidence', {}).get('part_hints', {})
    if hints.get('source') != 'observed_horizontal_band':
        hints = {}
    result = []
    def part(name, size, local, inferred=False):
        result.append(dict(name=name, center=(c+r@np.asarray(local)).tolist(),
                           rotation=r.tolist(), scale=np.asarray(size).tolist(),
                           inferred_completion=bool(inferred),
                           geometry_source='inferred_completion' if inferred else 'observed_surface_bounds'))
    w,d,h = s
    top = c[2]+h/2
    bottom = max(floor, c[2]-h/2)
    if label in ('desk','table','coffee_table'):
        thickness = min(.07,h)
        observed_top = hints.get('top_local_z')
        if observed_top is not None and np.isfinite(observed_top) and h*.2 <= observed_top <= h/2:
            top = c[2]+float(observed_top)
        part('top',[w,d,thickness],[0,0,top-c[2]-thickness/2])
        leg_height = max(.03,top-thickness-floor)
        leg = max(.025,min(w,d)*.1)
        for x in (-1,1):
            for y in (-1,1):
                part(f'leg-{x}-{y}',[leg,leg,leg_height],
                     [x*(w-leg)/2,y*(d-leg)/2,floor+leg_height/2-c[2]],True)
    elif label in ('chair','dining_chair'):
        seat_z = min(top-.06, floor+(top-floor)*.48)
        thickness = min(.06,h*.12)
        observed_seat = hints.get('seat_local_z')
        has_seat = (observed_seat is not None and np.isfinite(observed_seat)
                    and c[2]-h/2 < c[2]+observed_seat < top-thickness)
        if has_seat:
            seat_z = max(floor+thickness, c[2]+float(observed_seat))
        part('seat',[w,d,thickness],[0,0,seat_z-c[2]],not has_seat)
        # Generic back panel; no invented office-chair colour/wheel template.
        back_h = max(.05,top-seat_z)
        part('back',[w,min(.05,d*.15),back_h],[0,d*.42,seat_z+back_h/2-c[2]],True)
        leg_h = max(.03,seat_z-thickness/2-floor)
        leg = max(.025,min(w,d)*.08)
        for x in (-1,1):
            for y in (-1,1):
                part(f'leg-{x}-{y}',[leg,leg,leg_h],[x*w*.38,y*d*.35,floor+leg_h/2-c[2]],True)
    elif label == 'bed':
        full_h = max(.06,top-floor)
        part('base',[w,d,full_h],[0,0,floor+full_h/2-c[2]],bottom>floor+.02)
    else:
        part('body',s,[0,0,0])
    return result


def navigation_spawn(footprint, colliders, floor, eye_height, radius=.18):
    """Find an unoccupied floor location before declaring a room walkable."""
    p=np.asarray(footprint,dtype=float)
    lo,hi=p.min(0),p.max(0)
    step=max(.06,float(np.max(hi-lo))/120)
    candidates=[]
    for x in np.arange(lo[0]+radius,hi[0],step):
        for y in np.arange(lo[1]+radius,hi[1],step):
            inside=False;blocked=False
            for a,b in zip(p,np.roll(p,-1,axis=0)):
                delta=b-a
                t=np.clip(np.dot(np.array([x,y])-a,delta)/max(np.dot(delta,delta),1e-12),0,1)
                if np.linalg.norm(np.array([x,y])-a-t*delta)<radius:blocked=True;break
                if (a[1]>y)!=(b[1]>y) and x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]:inside=not inside
            if blocked or not inside:continue
            for box in colliders:
                if box['top']<=floor+.03 or box['bottom']>=floor+eye_height+.2:continue
                local=np.asarray(box['rotation'])[:2,:2].T@(np.array([x,y])-np.asarray(box['center'])[:2])
                excess=np.maximum(0,np.abs(local)-np.asarray(box['scale'])[:2]/2)
                if np.linalg.norm(excess)<radius:blocked=True;break
            if not blocked:candidates.append((np.linalg.norm(np.array([x,y])-(lo+hi)/2),x,y))
    if not candidates:raise ValueError('No free floor area for walking')
    _,x,y=min(candidates)
    return [float(x),float(y),float(floor+eye_height)]


def wall_loop(boxes, tolerance=.015):
    """Join wall centerlines without filling concavities with a convex hull."""
    segments = []
    for b in boxes:
        if b['class'] != 'wall':
            continue
        c, r = np.asarray(b['center']), np.asarray(b['rotation'])
        axis = int(np.argmax(np.asarray(b['scale'])[:2]))
        delta = r[:, axis] * b['scale'][axis] / 2
        segments.append([(c-delta)[:2], (c+delta)[:2]])
    if len(segments) < 3:
        raise ValueError('A closed wall perimeter is required')
    first, current = segments.pop(0)
    result = [first, current]
    while segments:
        matches = [(i, side) for i, segment in enumerate(segments) for side in (0, 1)
                   if np.linalg.norm(segment[side]-current) < tolerance]
        if len(matches) != 1:
            raise ValueError('Wall perimeter is disconnected or ambiguous')
        i, side = matches[0]
        current = segments.pop(i)[1-side]
        result.append(current)
    if np.linalg.norm(current-first) >= tolerance:
        raise ValueError('Wall perimeter is open')
    return np.asarray(result[:-1])


def photo_visibility(points, extrinsic, intrinsic, depth, confidence, tolerance):
    """Pointwise depth check in VGGT units, with stricter foreground rejection."""
    camera = points @ extrinsic[:, :3].T + extrinsic[:, 3]
    z = camera[:, 2]
    uv = (camera @ intrinsic.T)[:, :2] / np.maximum(z[:, None], 1e-8)
    h, w = depth.shape
    valid = np.isfinite(uv).all(1) & (z > 0) & (uv[:, 0] >= 1) & (uv[:, 0] < w-2) & (uv[:, 1] >= 1) & (uv[:, 1] < h-2)
    pixel = np.rint(np.nan_to_num(uv)).astype(np.int64)
    x, y = np.clip(pixel[:, 0], 0, w-1), np.clip(pixel[:, 1], 0, h-1)
    observed = depth[y, x]
    delta = z-observed
    valid &= np.isfinite(observed) & (observed > 0) & (delta < tolerance*.6) & (delta > -tolerance)
    valid &= np.isfinite(confidence[y, x]) & (confidence[y, x] > 1.2)
    return uv, valid, z


def furniture_occlusion(points, camera_center, boxes):
    """Reject photo rays passing through estimated furniture before a room surface."""
    points=np.asarray(points);blocked=np.zeros(len(points),dtype=bool)
    for box in boxes:
        center,rotation,scale=(np.asarray(box[k]) for k in ('center','rotation','scale'))
        local_camera=(camera_center-center)@rotation
        half=scale/2
        if np.all(np.abs(local_camera)<half):continue
        direction=(points-camera_center)@rotation
        parallel=np.abs(direction)<1e-10
        divisor=np.where(parallel,1.,direction)
        a=(-half-local_camera)/divisor;b=(half-local_camera)/divisor
        low=np.minimum(a,b);high=np.maximum(a,b)
        low=np.where(parallel,-np.inf,low);high=np.where(parallel,np.inf,high)
        misses_parallel=np.any(parallel&(np.abs(local_camera)>half),axis=1)
        near=np.maximum(low.max(1),0);far=high.min(1)
        blocked|=(near<=far)&(far>0)&(near<=1.0001)&~misses_parallel
    return blocked
