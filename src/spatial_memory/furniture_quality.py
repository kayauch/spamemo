"""Deterministic selection and uncertainty reporting for image furniture evidence.

Scores here rank evidence; they are not calibrated probabilities of correctness.
"""
from copy import deepcopy

import numpy as np


FURNITURE_LABELS = frozenset({'bed', 'chair', 'sofa', 'desk', 'table', 'wardrobe',
                            'cabinet', 'bookshelf', 'curtain', 'coffee_table', 'dining_chair',
                            'locker', 'shelf', 'box'})


def camera_viewpoints(records, extrinsics):
    """Group the same physical camera origin, including panorama virtual faces."""
    es = np.asarray(extrinsics, dtype=float)
    if es.shape != (len(records), 3, 4) or not np.isfinite(es).all():
        raise ValueError('Invalid furniture camera poses')
    centers = -np.einsum('nji,nj->ni', es[:, :, :3], es[:, :, 3])
    span = float(np.linalg.norm(np.ptp(centers, axis=0))) if len(centers) else 0.
    tolerance = max(span * .025, 1e-5)
    representatives, panorama_origins, groups = [], {}, []
    for record, center in zip(records, centers):
        panorama = record.get('panorama_group') or record.get('panorama_source_id')
        if not panorama and record.get('source_kind', record.get('kind')) in ('panorama', 'panorama_face'):
            panorama = record.get('source_id')
        if panorama is not None and str(panorama) in panorama_origins:
            group = panorama_origins[str(panorama)]
        else:
            distances = [float(np.linalg.norm(center - prior)) for prior in representatives]
            group = int(np.argmin(distances)) if distances and min(distances) <= tolerance else len(representatives)
            if group == len(representatives):
                representatives.append(center)
            if panorama is not None:
                panorama_origins[str(panorama)] = group
        groups.append(f'viewpoint-{group}')
    return groups, centers


def select_furniture_frames(records, extrinsics, budget=24):
    """Cover the full saved capture, then favor sharp, distinct camera views."""
    if isinstance(budget, bool) or not isinstance(budget, int) or not 2 <= budget <= 32:
        raise ValueError('Furniture frame budget must be 2..32')
    n = len(records)
    if not n:
        raise ValueError('Furniture capture has no frames')
    groups, centers = camera_viewpoints(records, extrinsics)
    sharpness = np.asarray([r.get('sharpness') or 0. for r in records], dtype=float)
    sharpness = np.nan_to_num(sharpness, nan=0., posinf=0., neginf=0.).clip(0)
    quality = .5 + .5 * np.minimum(sharpness / max(float(np.quantile(sharpness, .9)), 1.), 1.)
    directions = np.asarray(extrinsics)[:, 2, :3].astype(float)
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-8)
    span = max(float(np.linalg.norm(np.ptp(centers, axis=0))), 1e-5)
    selected = []
    # Temporal strata ensure that a late room corner is not lost to early detail.
    for bucket in np.array_split(np.arange(n), min(n, max(2, budget // 2))):
        selected.append(int(bucket[np.argmax(quality[bucket])]))
    while len(selected) < min(n, budget):
        position = np.linalg.norm(centers[:, None] - centers[selected][None], axis=2) / span
        angle = (1 - np.clip(directions @ directions[selected].T, -1., 1.)) / 2
        temporal = np.abs(np.arange(n)[:, None] - np.asarray(selected)[None]) / max(n - 1, 1)
        novelty = np.min(.55 * position + .35 * angle + .10 * temporal, axis=1)
        score = quality * (.05 + novelty)
        score[selected] = -1
        selected.append(int(np.argmax(score)))
    # Individually uploaded photos are deliberate evidence, not redundant video
    # samples. Keep the normal view selection AND every photo, even when that
    # exceeds the sampling budget. Otherwise unsegmented photos fail the positive
    # wall-mask check and silently cannot improve the room's wall textures.
    required_photos = [i for i, record in enumerate(records)
                       if record.get('source_kind', record.get('kind')) == 'photo']
    added_photos = sorted(set(required_photos) - set(selected))
    selected = sorted(set(selected).union(required_photos))
    return dict(indices=selected, viewpoint_ids=groups, requested_budget=budget,
                required_photo_indices=required_photos, additional_photo_indices=added_photos,
                selected_count=len(selected), total_frames=n,
                independent_viewpoints=len({groups[i] for i in selected}),
                method='full-capture strata + image sharpness + camera position/direction diversity + all uploaded photos',
                limitation='Camera positions are estimates; view counts are evidence diversity, not confidence probabilities')


def object_quality(group):
    frames = sorted(set(int(x) for x in group['frames']))
    viewpoints = sorted(set(group.get('viewpoint_ids', [f'frame-{x}' for x in frames])))
    labels = sorted(set(group.get('labels', [group['label']])))
    dimensions = np.asarray([np.diff(np.quantile(p, [.02, .98], axis=0), axis=0)[0]
                             for p in group['points']])
    median = np.maximum(np.median(dimensions, axis=0), .02)
    variation = float(np.max(np.ptp(dimensions, axis=0) / median)) if len(dimensions) > 1 else 0.
    issues = ['hidden_surfaces_unobserved']
    recommendations = []
    if len(viewpoints) < 2:
        issues.append('single_camera_origin')
        recommendations.append(dict(code='new_camera_position', message='位置を変えて、この家具の斜め・側面を撮影してください。'))
    if len(labels) > 1:
        issues.append('category_conflict')
        recommendations.append(dict(code='confirm_category', message='家具の種類の推定が一致していません。種類を確認してください。'))
    if variation > .75:
        issues.append('inconsistent_visible_dimensions')
        recommendations.append(dict(code='full_object_photo', message='家具全体が入る写真と側面の写真を追加してください。'))
    if not recommendations:
        recommendations.append(dict(code='side_or_back_photo', message='形を詳しくする場合は、まだ見えていない側面を撮影してください。'))
    return dict(source_frames=frames, frame_count=len(frames), independent_viewpoints=len(viewpoints),
                viewpoint_ids=viewpoints, labels_observed=labels,
                detection_score_range=[float(min(group['scores'])), float(max(group['scores']))],
                visible_dimension_variation=round(variation, 4),
                support_points=int(sum(len(p) for p in group['points'])),
                issues=issues, conflicts=[x for x in issues if x in ('category_conflict', 'inconsistent_visible_dimensions')],
                needs_photo=recommendations, geometry_status='visible_surface_bounds',
                measured_dimensions=False)


def part_hints(points, center, rotation, scale, label):
    """Extract only a supported seat/top level; unseen legs remain completion."""
    local = (np.asarray(points) - np.asarray(center)) @ np.asarray(rotation)
    size = np.asarray(scale)
    if len(local) < 300 or label not in ('chair', 'dining_chair', 'table', 'desk', 'coffee_table'):
        return {}
    hist, edges = np.histogram(local[:, 2], bins=20, range=(-size[2] / 2, size[2] / 2))
    eligible = np.arange(4, 15) if label in ('chair', 'dining_chair') else np.arange(14, 20)
    best = int(eligible[np.argmax(hist[eligible])])
    support = local[(local[:, 2] >= edges[best]) & (local[:, 2] <= edges[best + 1])]
    if len(support) < max(60, len(local) * .08):
        return {}
    extent = np.ptp(support[:, :2], axis=0)
    if np.any(extent < size[:2] * .55):
        return {}
    key = 'seat_local_z' if label in ('chair', 'dining_chair') else 'top_local_z'
    return {key: float(np.median(support[:, 2])), 'source': 'observed_horizontal_band',
            'support_points': len(support), 'limitation': 'Part thickness and unseen supports remain inferred'}


def apply_furniture_overrides(layout, overrides, unit_scale=1.):
    """Return an independent layout with bounded user corrections and provenance."""
    result = deepcopy(layout)
    if not isinstance(overrides, list) or len(overrides) > 100:
        raise ValueError('Furniture overrides must be a list of at most 100 items')
    if not np.isfinite(unit_scale) or unit_scale <= 0:
        raise ValueError('Invalid furniture metric calibration')
    index = {str(b['id']): b for b in result['boxes'] if b.get('class') == 'bbox'}
    used, applied = set(), []
    for change in overrides:
        if not isinstance(change, dict) or str(change.get('id')) not in index:
            raise ValueError('Unknown furniture override id')
        object_id = str(change['id'])
        if object_id in used:
            raise ValueError('Duplicate furniture override id')
        used.add(object_id)
        box = index[object_id]
        changes_geometry = 'dimensions_m' in change or ('label' in change and change['label'] != box.get('label'))
        if 'label' in change:
            if change['label'] not in FURNITURE_LABELS:
                raise ValueError('Unsupported furniture label')
            box['label'] = change['label']
        if 'hidden' in change:
            if not isinstance(change['hidden'], bool):
                raise ValueError('Furniture hidden must be boolean')
            box['hidden'] = change['hidden']
            if box.get('review_status') == 'needs_confirmation' and not change['hidden']:
                box['review_status'] = 'user_accepted'
                box.setdefault('evidence', {})['requires_confirmation'] = False
        if 'dimensions_m' in change:
            dimensions = np.asarray(change['dimensions_m'], dtype=float)
            if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions < .03) or np.any(dimensions > 10):
                raise ValueError('Furniture dimensions must be three lengths in 0.03..10 metres')
            prior = np.asarray(box['scale'], dtype=float)
            dimensions = dimensions / unit_scale
            # Anchor the observed bottom while changing height, keeping XY center.
            box['center'][2] += float((dimensions[2] - prior[2]) / 2)
            box['scale'] = dimensions.tolist()
            box.pop('vg_world', None)  # Stale duplicate coordinate data must not survive edits.
            box.setdefault('evidence', {}).pop('part_hints', None)
        box.setdefault('evidence', {})['user_corrected'] = True
        geometry = box['evidence'].get('storage_geometry')
        if changes_geometry and geometry:
            prior = deepcopy(geometry)
            box['evidence']['storage_geometry'] = dict(
                id=box['id'], status='superseded_by_user_override', accepted=False,
                current_geometry_validated=False, complete_hidden_geometry_verified=False,
                reasons=['Dimensions or category changed after image validation'],
                prior_validation=prior, wall_correction='deferred')
            if isinstance(box['evidence'].get('quality'), dict):
                box['evidence']['quality'].update(geometry_status='superseded_by_user_override',
                    storage_geometry_status='superseded_by_user_override', current_geometry_validated=False,
                    dimensions_model_units=deepcopy(box['scale']), center_model_units=deepcopy(box['center']))
        box['user_correction'] = deepcopy(change)
        applied.append(deepcopy(change))
    result['furniture_overrides'] = applied
    for record in result.get('image_furniture', {}).get('quality_report', {}).get('objects', []):
        box = index.get(str(record.get('id')))
        if box is not None:
            record.update(label=box['label'], hidden=bool(box.get('hidden')),
                          dimensions_model_units=box['scale'], center_model_units=box['center'],
                          user_corrected=bool(box.get('user_correction')), review_status=box.get('review_status'))
            geometry = box.get('evidence', {}).get('storage_geometry')
            if geometry and geometry.get('status') == 'superseded_by_user_override':
                record['geometry_status'] = geometry['status']
                record['storage_geometry_status'] = geometry['status']
                record['current_geometry_validated'] = False
    for geometry_report in (result.get('storage_geometry'), result.get('image_furniture', {}).get('storage_geometry')):
        if isinstance(geometry_report, dict):
            geometry_report['objects'] = [deepcopy(index.get(str(item.get('id')), {}).get('evidence', {}).get('storage_geometry', item))
                                          for item in geometry_report.get('objects', [])]
            geometry_report['accepted_count'] = sum(bool(item.get('accepted')) for item in geometry_report['objects'])
            geometry_report['preserved_count'] = sum(not item.get('accepted') for item in geometry_report['objects'])
    report = result.get('image_furniture', {}).get('quality_report', {})
    if 'unresolved_candidates' in report:
        report['unresolved_candidates'] = [item for item in report['unresolved_candidates']
            if index.get(str(item.get('id')), {}).get('review_status') != 'user_accepted']
    return result
