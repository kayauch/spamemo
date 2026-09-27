"""Join complementary image-observed storage fragments without inventing depth.

Thin front tracks have almost no volume, so ordinary 3D overlap association can
miss a side track of the same storage. These conservative planar checks retain
the union of the existing observations; unseen closures remain inferred.
"""
from copy import deepcopy
import numpy as np


_LABELS = frozenset({'cabinet', 'locker', 'wardrobe', 'bookshelf', 'shelf'})


def _candidate(box):
    if (box.get('class') != 'bbox' or box.get('label') not in _LABELS
            or box.get('hidden') or box.get('merged_into') is not None):
        return None
    evidence = box.get('evidence', {})
    if evidence.get('valid') is not True or evidence.get('user_corrected') or box.get('user_correction'):
        return None
    if evidence.get('storage_geometry', {}).get('accepted'):
        return None
    if not box.get('source_detection_ids') or not box.get('source_frames'):
        return None
    center = np.asarray(box.get('center'), float)
    size = np.asarray(box.get('scale'), float)
    rotation = np.asarray(box.get('rotation'), float)
    if (center.shape != (3,) or size.shape != (3,) or rotation.shape != (3, 3)
            or not np.isfinite(center).all() or not np.isfinite(size).all()
            or min(size) <= 0 or not np.allclose(rotation, np.eye(3), atol=1e-5)):
        return None
    axis = int(np.argmin(size[:2]))
    return dict(box=box, center=center, size=size, low=center-size/2,
                high=center+size/2, axis=axis,
                frames=set(box['source_frames']))


def _overlap(left, right, axis):
    span = max(0., min(left['high'][axis], right['high'][axis])
               - max(left['low'][axis], right['low'][axis]))
    return span / min(left['size'][axis], right['size'][axis])


def _thin(candidate):
    size = candidate['size']; axis = candidate['axis']
    return size[axis] <= .12*size[2] and size[axis] <= .15*size[1-axis]


def _matches_front(seed, item):
    axis = seed['axis']
    if item['axis'] != axis or seed['frames'] & item['frames']:
        return False
    if min(seed['size'][2], item['size'][2]) / max(seed['size'][2], item['size'][2]) < .55:
        return False
    return _overlap(seed, item, 2) >= .70 and _overlap(seed, item, 1-axis) >= .60


def _world_mapping(seed, transform):
    if transform is not None:
        if isinstance(transform, dict):
            if 'generation_world_to_vg_world' in transform:
                inverse = np.asarray(transform['generation_world_to_vg_world'], float)
            else:
                inverse = np.linalg.inv(np.asarray(transform['vg_world_to_generation_world'], float))
        else:
            inverse = np.linalg.inv(np.asarray(transform, float))
        if inverse.shape != (4, 4) or not np.isfinite(inverse).all():
            raise ValueError('Invalid storage fragment coordinate transform')
        scales = np.linalg.norm(inverse[:3, :3], axis=0)
        if min(scales) <= 0 or not np.allclose(inverse[:3, :3].T @ inverse[:3, :3],
                                              np.eye(3)*scales[0]**2, rtol=1e-5, atol=1e-8):
            raise ValueError('Storage fragment transform must be uniform-scale rigid')
        return inverse[:3, :3], inverse[:3, 3], float(scales[0])
    vg = seed['box'].get('vg_world')
    if vg is None:
        return None
    ratios = np.asarray(vg['scale'], float) / seed['size']
    if not np.isfinite(ratios).all() or min(ratios) <= 0 or not np.allclose(ratios, ratios[0]):
        raise ValueError('Inconsistent storage fragment coordinate scales')
    rotation = np.asarray(vg['rotation'], float)
    linear = rotation * ratios[0]
    offset = np.asarray(vg['center'], float) - linear @ seed['center']
    return linear, offset, float(ratios[0])


def consolidate_storage_fragments(layout, transform=None):
    """Return a new layout, joining disjoint-view front and side observations.

    The widest planar front retains its ID. Every consumed track stays in the
    layout as hidden with ``merged_into`` for auditability. This does not claim
    measured physical dimensions or verified unseen geometry.
    """
    result = deepcopy(layout)
    candidates = [c for b in result.get('boxes', []) if (c := _candidate(b)) is not None]
    fronts = sorted((c for c in candidates if _thin(c)),
                    key=lambda c: c['size'][1-c['axis']]*c['size'][2], reverse=True)
    used = set(); reports = deepcopy(result.get('storage_fragment_consolidation', {}).get('objects', []))
    for seed in fronts:
        if seed['box']['id'] in used:
            continue
        axis = seed['axis']; height = seed['size'][2]
        tolerance = max(seed['size'][axis], .05*height)
        members = [seed]; frames = set(seed['frames'])
        # Match to the original seed plane, never an expanding transitive chain.
        for item in fronts:
            if item is seed or item['box']['id'] in used or frames & item['frames']:
                continue
            if (_matches_front(seed, item)
                    and abs(seed['center'][axis]-item['center'][axis]) <= tolerance):
                members.append(item); frames.update(item['frames'])
        side_candidates = []
        for item in candidates:
            if item['box']['id'] in used or _thin(item) or frames & item['frames']:
                continue
            if not _matches_front(seed, item):
                continue
            depth = item['size'][axis]
            if depth < max(1.8*seed['size'][axis], .10*height) or depth > .50*height:
                continue
            # A front lies near exactly one edge of a side-observed envelope.
            near = [abs(item[edge][axis]-seed['center'][axis]) <= tolerance
                    for edge in ('low', 'high')]
            if sum(near) != 1:
                continue
            direction = 1 if near[0] else -1
            side_candidates.append((item, direction))
        if not side_candidates or len({direction for _, direction in side_candidates}) != 1:
            continue
        # Side tracks must agree on a rear extent; do not choose one of two
        # incompatible depths merely because one was observed more often.
        direction = side_candidates[0][1]
        rear_edge = 'high' if direction > 0 else 'low'
        rear = [item[rear_edge][axis] for item, _ in side_candidates]
        if max(rear)-min(rear) > max(tolerance, .15*np.median([item['size'][axis] for item, _ in side_candidates])):
            continue
        for item, _ in side_candidates:
            if frames & item['frames']:
                continue
            members.append(item); frames.update(item['frames'])
        low = np.min([item['low'] for item in members], axis=0)
        high = np.max([item['high'] for item in members], axis=0)
        size = high-low
        if size[axis] > .55*size[2]:
            continue
        mapping = _world_mapping(seed, transform)
        box = seed['box']; center = (low+high)/2
        source_ids = sorted(set(identifier for item in members for identifier in item['box']['source_detection_ids']))
        member_ids = [item['box']['id'] for item in members]
        report = dict(id=box['id'], status='associated_observed_fragments',
                      source_object_ids=member_ids, merged_object_ids=member_ids[1:],
                      source_detection_ids=source_ids, source_frames=sorted(frames),
                      depth_axis=axis, observed_front_coordinate=float(seed['center'][axis]),
                      observed_depth_range=[float(low[axis]), float(high[axis])],
                      dimensions_model_units=size.tolist(),
                      hidden_closures_inferred=True, complete_hidden_geometry_verified=False,
                      policy='Union of existing image-observed bounds across disjoint source frames; no wall distance or category depth prior')
        box.update(center=center.tolist(), scale=size.tolist(),
                   source_frames=sorted(frames), source_detection_ids=source_ids,
                   detection_scores=[score for item in members for score in item['box'].get('detection_scores', [])],
                   constraint='Associated observed front and side envelopes; unseen closures remain inferred')
        if mapping is not None:
            linear, offset, unit_scale = mapping
            box['vg_world'] = dict(center=(linear@center+offset).tolist(),
                                   rotation=(linear/unit_scale).tolist(), scale=(size*unit_scale).tolist())
        evidence = box.setdefault('evidence', {})
        qualities = [item['box'].get('evidence', {}).get('quality', {}) for item in members]
        viewpoints = sorted(set(v for q in qualities for v in q.get('viewpoint_ids', [])))
        labels = sorted(set(label for q in qualities for label in q.get('labels_observed', [])))
        evidence['storage_fragment_consolidation'] = deepcopy(report)
        prior = deepcopy(evidence.get('storage_geometry'))
        evidence['storage_geometry'] = dict(id=box['id'], status='associated_observed_fragments',
            accepted=False, current_geometry_validated=False, complete_hidden_geometry_verified=False,
            reasons=['Complementary image-observed fragments associated; original per-track validation superseded'],
            prior_validation=prior, wall_correction='deferred')
        evidence['points_inside'] = sum(item['box'].get('evidence', {}).get('points_inside', 0) or 0 for item in members)
        quality = evidence.setdefault('quality', {})
        quality.update(source_frames=sorted(frames), frame_count=len(frames),
                       viewpoint_ids=viewpoints, independent_viewpoints=len(viewpoints),
                       labels_observed=labels, dimensions_model_units=size.tolist(),
                       center_model_units=center.tolist(), measured_dimensions=False,
                       support_points=evidence['points_inside'],
                       geometry_status='associated_observed_envelope_with_inferred_closures',
                       storage_geometry_status='associated_observed_fragments',
                       current_geometry_validated=False, hidden_surfaces_verified=False)
        for item in members[1:]:
            child = item['box']
            child.update(hidden=True, merged_into=box['id'], review_status='merged_observed_fragment')
            child.setdefault('evidence', {})['storage_fragment_consolidation'] = dict(
                status='merged_observed_fragment', merged_into=box['id'])
            child_prior = deepcopy(child['evidence'].get('storage_geometry'))
            child['evidence']['storage_geometry'] = dict(id=child['id'], status='merged_observed_fragment',
                accepted=False, merged_into=box['id'], current_geometry_validated=False,
                complete_hidden_geometry_verified=False, prior_validation=child_prior,
                reasons=['Fragment retained as provenance of associated storage'], wall_correction='deferred')
        used.update(member_ids); reports.append(report)
        for entry in result.get('image_furniture', {}).get('quality_report', {}).get('objects', []):
            if entry.get('id') == box['id']:
                entry.update(deepcopy(quality))
            elif entry.get('id') in member_ids[1:]:
                entry.update(hidden=True, merged_into=box['id'], review_status='merged_observed_fragment',
                             storage_geometry_status='merged_observed_fragment', current_geometry_validated=False)
    summary = dict(version=1, merged_count=len(reports), objects=reports,
                   policy='Conservative complementary image-observation association; hidden geometry remains inferred')
    result['storage_fragment_consolidation'] = deepcopy(summary)
    result.setdefault('image_furniture', {})['storage_fragment_consolidation'] = summary
    if 'confirmed_count' in result['image_furniture']:
        result['image_furniture']['confirmed_count'] = sum(
            b.get('class') == 'bbox' and not b.get('hidden') for b in result.get('boxes', []))
    index = {str(b.get('id')): b for b in result.get('boxes', [])}
    for geometry in (result.get('storage_geometry'), result['image_furniture'].get('storage_geometry')):
        if isinstance(geometry, dict):
            geometry['objects'] = [deepcopy(index.get(str(item.get('id')), {}).get('evidence', {}).get('storage_geometry', item))
                                   for item in geometry.get('objects', [])]
            geometry['accepted_count'] = sum(bool(item.get('accepted')) for item in geometry['objects'])
            geometry['preserved_count'] = sum(not item.get('accepted') for item in geometry['objects'])
    return result
