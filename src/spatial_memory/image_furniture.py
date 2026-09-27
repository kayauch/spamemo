"""Image-mask evidence lifted into the existing generation coordinate frame."""
import numpy as np
import re

UP_ROTATIONS = {'image-top': 0, 'image-left': 3, 'image-right': 1, 'image-bottom': 2}
FURNITURE_CATEGORIES = ('bed', 'chair', 'sofa', 'desk', 'table', 'wardrobe',
                        'cabinet', 'bookshelf', 'curtain', 'locker', 'shelf', 'box')
STORAGE_CATEGORIES = frozenset(('wardrobe', 'cabinet', 'bookshelf', 'locker', 'shelf', 'box'))
STORAGE_FAMILIES = {'wardrobe': 'cabinet', 'cabinet': 'cabinet', 'locker': 'cabinet',
                    'bookshelf': 'cabinet', 'shelf': 'cabinet', 'box': 'box'}


def classify_detection(text):
    """Keep ambiguous cross-family labels unknown, including wall/furniture mixtures."""
    tokens = set(re.findall(r'[a-z]+', str(text).lower()))
    labels = tokens.intersection(FURNITURE_CATEGORIES)
    if 'wall' in tokens:
        return ('unknown', 'unknown') if labels else ('wall', 'wall')
    if len(labels) == 1:
        return 'furniture', next(iter(labels))
    if labels and labels <= STORAGE_CATEGORIES:
        families = {STORAGE_FAMILIES[label] for label in labels}
        if len(families) == 1:
            return 'furniture', next(iter(families))
    return 'unknown', 'unknown'


def lift_mask(mask, depth, confidence, intrinsic, extrinsic, transform):
    if mask.shape != depth.shape or confidence.shape != depth.shape:
        raise ValueError('Mask and saved camera dimensions differ')
    valid = mask & np.isfinite(depth) & (depth > 0) & np.isfinite(confidence) & (confidence > 1.2)
    y, x = np.nonzero(valid)
    # Deterministic cap, retaining spatially distributed observations.
    keep = np.linspace(0, len(x)-1, min(len(x), 6000), dtype=int)
    x, y = x[keep], y[keep]
    camera = np.column_stack([x, y, np.ones(len(x))]) @ np.linalg.inv(intrinsic).T
    camera *= depth[y, x, None]
    world = (camera-extrinsic[:, 3]) @ extrinsic[:, :3]
    t = np.asarray(transform)
    return world @ t[:3, :3].T + t[:3, 3]


def bounds(points):
    return np.quantile(points, [.02, .98], axis=0)


def merge_observations(observations):
    """Conservative multi-frame association. Single-view proposals stay unconfirmed."""
    # Similar detector categories may refer to the same object. Retain the
    # observed labels for review instead of silently relabeling all tables/desks.
    families = {'desk': 'table-family', 'table': 'table-family', 'coffee_table': 'table-family',
                **{label: family+'-family' for label, family in STORAGE_FAMILIES.items()}}
    def compatible(a, b):
        return families.get(a, a) == families.get(b, b)
    groups = []
    for obs in sorted(observations, key=lambda x: -x['score']):
        lo, hi = bounds(obs['points'])
        matches = []
        for group in groups:
            if not compatible(group['label'], obs['label']) or obs['frame'] in group['frames']:
                continue
            gl, gh = bounds(np.concatenate(group['points']))
            overlap = np.maximum(0, np.minimum(hi, gh)-np.maximum(lo, gl))
            ratio = np.prod(overlap) / max(1e-8, min(np.prod(hi-lo), np.prod(gh-gl)))
            center_distance = np.linalg.norm((lo+hi-gl-gh)/2)
            if ratio > .25 and center_distance < max(.35, np.linalg.norm(gh-gl)*.5):
                matches.append((ratio, group))
        if matches:
            group = max(matches, key=lambda m: m[0])[1]
            group['frames'].append(obs['frame']);group['points'].append(obs['points'])
            group['scores'].append(obs['score'])
            group['labels'].append(obs['label'])
            group['viewpoint_ids'].append(obs.get('viewpoint_id', f"frame-{obs['frame']}"))
            group['detection_ids'].extend([obs['detection_id']] if 'detection_id' in obs else [])
        else:
            groups.append(dict(label=obs['label'], frames=[obs['frame']], points=[obs['points']], scores=[obs['score']],
                               labels=[obs['label']], viewpoint_ids=[obs.get('viewpoint_id', f"frame-{obs['frame']}")],
                               detection_ids=[obs['detection_id']] if 'detection_id' in obs else []))
    # Partial views can start separate tracks. Reconcile only overlapping tracks
    # from disjoint frames; never collapse two chairs observed together.
    changed=True
    while changed:
        changed=False
        for i,left in enumerate(groups):
            ll,lh=bounds(np.concatenate(left['points']))
            for j in range(i+1,len(groups)):
                right=groups[j]
                if not compatible(left['label'],right['label']) or set(left['frames']) & set(right['frames']):continue
                rl,rh=bounds(np.concatenate(right['points']))
                inter=np.prod(np.maximum(0,np.minimum(lh,rh)-np.maximum(ll,rl)))
                union=np.prod(lh-ll)+np.prod(rh-rl)-inter
                if inter/max(union,1e-8)>.35:
                    for key in ('points','frames','scores','labels','viewpoint_ids','detection_ids'):left[key].extend(right[key])
                    groups.pop(j);changed=True;break
            if changed:break
    for group in groups:
        votes = {label: sum(score for candidate, score in zip(group['labels'], group['scores']) if candidate == label)
                 for label in group['labels']}
        group['label'] = max(votes, key=votes.get)
    return groups
