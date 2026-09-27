"""Conservative photo blending, bounded photo loading and capture feedback.

Images never count as evidence outside the depth-supported mask. Measurements
calibrate a uniform output scale; they do not certify the reconstructed shape.
"""
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class PhotoCache:
    """Load original photographs at the recorded VGGT crop, with an LRU cap."""

    def __init__(self, directory, records, height, width, max_bytes=96 * 1024**2, max_scale=4, preprocessing='vggt'):
        self.directory, self.records = Path(directory), records
        self.height, self.width = int(height), int(width)
        self.max_bytes, self.max_scale = int(max_bytes), int(max_scale)
        self.preprocessing = preprocessing
        self.cache, self.bytes = OrderedDict(), 0
        self.scales = {}

    def get(self, index):
        from .texturing import preprocess_frame

        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        path = (self.directory / self.records[index]['name']).resolve()
        if not path.is_relative_to(self.directory.resolve()):
            raise ValueError('Frame path escapes its directory')
        with Image.open(path) as source:
            # Match individual crops AND mixed-aspect batch padding through the
            # same preprocessing implementation as the saved geometry cameras.
            native_camera = self.preprocessing == 'aligned_rgbd' or self.records[index].get('source_kind') == 'sensor'
            if native_camera:
                if abs((source.width / source.height) / (self.width / self.height) - 1) > .01:
                    raise ValueError('RGB-D photo aspect ratio differs from its camera domain')
                scale = max(1, min(self.max_scale, source.width // self.width, source.height // self.height))
            else:
                scale = max(1, min(self.max_scale, source.width // 518))
        image = preprocess_frame(path, self.height, self.width, scale=scale,
                                 preprocessing='aligned_rgbd' if native_camera else 'vggt')
        photo = np.asarray(image).copy()
        self.scales[index] = scale
        while self.cache and self.bytes + photo.nbytes > self.max_bytes:
            _, (old, _) = self.cache.popitem(last=False)
            self.bytes -= old.nbytes
        if photo.nbytes <= self.max_bytes:
            self.cache[index] = (photo, scale)
            self.bytes += photo.nbytes
        return photo, scale


def sample_photo(photo, scale, uv):
    """Pixel centres scale as (x + .5) * scale - .5, including cropped frames."""
    uv = np.asarray(uv)
    x = ((uv[..., 0] + .5) * scale - .5).astype(np.float32)
    y = ((uv[..., 1] + .5) * scale - .5).astype(np.float32)
    shape = x.shape
    result = np.empty((x.size, 3), np.uint8)
    # OpenCV remap requires each map axis below SHRT_MAX, even for a flat list.
    for start in range(0, x.size, 16000):
        stop = min(x.size, start + 16000)
        result[start:stop] = cv2.remap(photo, x.ravel()[start:stop, None], y.ravel()[start:stop, None],
                                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(-1, 3)
    return result.reshape(*shape, 3)


def overlap_color_gain(source, reference, min_samples=64):
    """Estimate a small color correction only from agreeing shared 3D samples."""
    source, reference = np.asarray(source, float), np.asarray(reference, float)
    valid = ((source > 20) & (source < 235) & (reference > 20) & (reference < 235)).all(axis=1)
    if valid.sum() < min_samples:
        return np.ones(3)
    a, b = source[valid], reference[valid]
    gain = np.clip(np.median(b / a, axis=0), .9, 1.1)
    # Large differences usually indicate bad correspondence, reflections or
    # moving objects. Leave those photographs untouched.
    if np.median(np.abs(a * gain - b)) > 12:
        return np.ones(3)
    brightness = float(np.mean(gain))
    return np.clip(gain, max(.9, brightness - .025), min(1.1, brightness + .025))


def estimate_photo_gains(cache, extrinsics, intrinsics, depths, confidence, tolerance):
    """Use nearby depth-checked views; never globally match unrelated histograms."""
    from .solid_room import photo_visibility

    n, height, width = depths.shape
    gains = np.ones((n, 3), dtype=np.float32)
    centers = np.array([-e[:, :3].T @ e[:, 3] for e in extrinsics])
    yy, xx = np.mgrid[3:height - 3:12, 3:width - 3:12]
    pixels = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
    for i in range(1, n):
        z = depths[i, yy, xx].ravel()
        valid = np.isfinite(z) & (z > 0) & (confidence[i, yy, xx].ravel() > 1.2)
        if valid.sum() < 64:
            continue
        camera = (pixels[valid] @ np.linalg.inv(intrinsics[i]).T) * z[valid, None]
        points = (camera - extrinsics[i, :, 3]) @ extrinsics[i, :, :3]
        source_photo, source_scale = cache.get(i)
        source = sample_photo(source_photo, source_scale, pixels[valid, :2])
        candidates = np.argsort(np.linalg.norm(centers[:i] - centers[i], axis=1))[:3]
        estimates = []
        for j in candidates:
            uv, visible, _ = photo_visibility(points, extrinsics[j], intrinsics[j], depths[j], confidence[j], tolerance)
            if visible.sum() < 64:
                continue
            photo, photo_scale = cache.get(int(j))
            reference = sample_photo(photo, photo_scale, uv[visible]) * gains[j]
            estimates.append(overlap_color_gain(source[visible], reference))
        if estimates:
            gains[i] = np.clip(np.median(estimates, axis=0), .9, 1.1)
    return gains


class SupportedTextureBlend:
    """Keep the two best visible samples; blend only compatible seam pixels."""

    def __init__(self, count):
        self.scores = np.zeros((2, count), np.float32)
        self.colors = np.zeros((2, count, 3), np.uint8)
        self.frames = np.full((2, count), -1, np.int32)

    def add(self, frame, colors, scores, visible):
        scores = np.asarray(scores)
        usable = np.asarray(visible, bool) & np.isfinite(scores) & (scores > 0)
        first = usable & (scores > self.scores[0])
        self.scores[1, first] = self.scores[0, first]
        self.colors[1, first] = self.colors[0, first]
        self.frames[1, first] = self.frames[0, first]
        second = usable & ~first & (scores > self.scores[1])
        for slot, take in ((0, first), (1, second)):
            self.scores[slot, take] = scores[take]
            self.colors[slot, take] = np.clip(np.asarray(colors)[take], 0, 255).astype(np.uint8)
            self.frames[slot, take] = frame

    def finish(self, height, width, material, region=None):
        region = np.ones(height * width, bool) if region is None else np.asarray(region, bool).ravel()
        supported = (self.frames[0] >= 0) & region
        dominant = self.frames[0].reshape(height, width)
        seam = np.zeros_like(dominant, np.uint8)
        horizontal = (dominant[:, 1:] != dominant[:, :-1]) & (dominant[:, 1:] >= 0) & (dominant[:, :-1] >= 0)
        vertical = (dominant[1:, :] != dominant[:-1, :]) & (dominant[1:, :] >= 0) & (dominant[:-1, :] >= 0)
        seam[:, 1:] |= horizontal; seam[:, :-1] |= horizontal
        seam[1:, :] |= vertical; seam[:-1, :] |= vertical
        seam = cv2.dilate(seam, np.ones((5, 5), np.uint8)).ravel().astype(bool)
        difference = np.abs(self.colors[0].astype(float) - self.colors[1])
        blend = (supported & seam & (self.frames[1] >= 0) & (difference.mean(1) <= 18)
                 & (difference.max(1) <= 45) & (self.scores[1] >= self.scores[0] * .35))
        texture = np.tile(np.asarray(material, np.uint8), (height * width, 1))
        texture[supported] = self.colors[0, supported]
        secondary_weight = np.zeros(height * width, np.float32)
        secondary_weight[blend] = self.scores[1, blend] / (self.scores[:, blend].sum(0) + 1e-12)
        texture[blend] = np.rint(self.colors[0, blend] * (1 - secondary_weight[blend, None])
                                  + self.colors[1, blend] * secondary_weight[blend, None]).astype(np.uint8)
        if supported.any():
            texture[~supported] = np.median(texture[supported], axis=0).astype(np.uint8)
        dominant = np.where(supported, self.frames[0], -1)
        secondary = np.where(blend, self.frames[1], -1)
        return texture, supported, dominant, secondary, secondary_weight


def measurement_calibration(footprint, floor, ceiling, options=None):
    options = options or {}
    polygon = np.asarray(footprint, float)
    height = float(ceiling - floor)
    longest = float(np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1).max())
    supplied = {key: options.get(key) for key in ('reference_length_m', 'ceiling_height_m')}
    for key, value in supplied.items():
        bounds = (.3, 50) if key == 'reference_length_m' else (1.5, 10)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not np.isfinite(value) or not bounds[0] <= value <= bounds[1]):
            raise ValueError(f'Invalid {key}')
    if options.get('reference_axis', 'longest_wall') != 'longest_wall':
        raise ValueError('Only longest_wall is supported as the reference axis')
    primary = 'reference_length_m' if supplied['reference_length_m'] is not None else (
        'ceiling_height_m' if supplied['ceiling_height_m'] is not None else None)
    original = {'reference_length_m': longest, 'ceiling_height_m': height}
    if height <= 0 or longest <= 0 or not np.isfinite([height, longest]).all():
        raise ValueError('Room dimensions must be positive')
    factor = float(supplied[primary] / original[primary]) if primary else 1.
    measurements, discrepancies = [], []
    for key, observed in supplied.items():
        if observed is None:
            continue
        predicted = original[key] * factor
        relative = abs(predicted - observed) / observed
        item = dict(name=key, supplied_m=float(observed), calibrated_m=predicted, relative_error=relative)
        measurements.append(item)
        if relative > .1:
            discrepancies.append(item)
    return dict(scale_factor=factor, primary_measurement=primary, measurements=measurements,
                discrepancies=discrepancies, needs_review=bool(discrepancies),
                metric_scale_verified=False, metric_scale_calibrated=primary is not None,
                assumptions=['Uniform scale from user measurement; local geometry remains an estimate.',
                             'Longest wall means the longest individual inferred boundary segment.'])


def calibration_scale(layout, options=None):
    from .solid_room import wall_loop
    return measurement_calibration(wall_loop(layout['boxes']), layout['vertical_anchors']['floor']['z'],
                                   layout['vertical_anchors']['ceiling']['z'], options)['scale_factor']


def scale_room_report(report, factor):
    """Scale every exported room-space position/size, while retaining source evidence."""
    result = deepcopy(report)
    for key in ('floor', 'ceiling'):
        result[key] *= factor
        result['navigation'][key] *= factor
    for key in ('footprint',):
        result[key] = (np.asarray(result[key]) * factor).tolist()
        result['navigation'][key] = (np.asarray(result['navigation'][key]) * factor).tolist()
    result['navigation']['spawn'] = (np.asarray(result['navigation']['spawn']) * factor).tolist()
    # Navigation body dimensions remain physical metres after calibration.
    for surface in result['surfaces']:
        for key in ('origin', 'u', 'v'):
            surface[key] = (np.asarray(surface[key]) * factor).tolist()
        if surface.get('polygon') is not None:
            surface['polygon'] = (np.asarray(surface['polygon']) * factor).tolist()
    for model in result['models']:
        for part in model['parts']:
            for key in ('center', 'scale'):
                part[key] = (np.asarray(part[key]) * factor).tolist()
    for box in result['navigation']['colliders']:
        for key in ('center', 'scale'):
            box[key] = (np.asarray(box[key]) * factor).tolist()
        for key in ('bottom', 'top'):
            box[key] *= factor
    for anchor in result.get('vertical_anchors', {}).values():
        if isinstance(anchor, dict) and isinstance(anchor.get('z'), (int, float)):
            anchor['z'] *= factor
    result['geometry_validation']['shell_volume'] *= factor ** 3
    output_from_source = np.diag([factor, factor, factor, 1])
    result['output_from_generation_world'] = output_from_source.tolist()
    if 'transform' in result:
        result['source_transform'] = deepcopy(result['transform'])
        transform = result['transform']
        if 'vg_world_to_generation_world' in transform:
            transform['vg_world_to_generation_world'] = (output_from_source @ np.asarray(transform['vg_world_to_generation_world'])).tolist()
        if 'generation_world_to_vg_world' in transform:
            transform['generation_world_to_vg_world'] = (np.asarray(transform['generation_world_to_vg_world']) @ np.linalg.inv(output_from_source)).tolist()
        if 'scale_m_per_vggt_unit' in transform:
            transform['scale_m_per_vggt_unit'] *= factor
        transform['output_scale_factor'] = factor
    return result


def capture_quality_report(surfaces, calibration, photo_sources):
    labels = {'floor': '床', 'ceiling': '天井', 'wall': '壁', 'object': '家具'}
    recommendations, details = [], []
    total_area, supported_area = 0., 0.
    for surface in surfaces:
        fraction = float(surface['supported_fraction'])
        area = float(surface.get('surface_area', 1))
        total_area += area
        supported_area += area * fraction
        detail = {key: surface.get(key) for key in ('object_id', 'kind', 'supported_fraction', 'blended_fraction', 'frames_used', 'projection_diagnostics')}
        detail.update(surface_id=surface['id'], missing_fraction=1 - fraction)
        details.append(detail)
        if fraction < .8:
            kind = surface['kind']
            label = surface.get('object_label') or labels.get(kind, kind)
            diagnostics = surface.get('projection_diagnostics') or {}
            code = 'missing_photo_coverage'
            message = f'{label}の写真を貼れていません。未撮影・遮蔽・形状と深度のずれが考えられます。元映像と推定形状を確認してください。'
            if diagnostics.get('in_view_fraction', 0) > .5 and diagnostics.get('depth_supported_fraction', 1) < .25:
                code = 'projection_depth_mismatch'
                message = f'{label}はカメラの画角内ですが、推定面と画像の深度が合っていません。写真追加の前に部屋形状を確認してください。'
            elif diagnostics.get('depth_supported_fraction', 0) - diagnostics.get('after_furniture_mask_fraction', 0) > .2:
                code = 'furniture_mask_occlusion'
                message = f'{label}への貼り込みが家具の推定範囲で遮られています。家具の位置・寸法・重複を確認してください。'
            recommendations.append(dict(id=f'photo-{surface["id"]}', code=code, kind=kind,
                                        severity='high' if fraction < .4 else 'medium', surface_id=surface['id'],
                                        object_id=surface.get('object_id'),
                                        message=message,
                                        missing_fraction=1 - fraction))
    if calibration['needs_review']:
        recommendations.insert(0, dict(id='measurement-conflict', code='measurement_conflict', kind='dimensions', severity='high',
                                      message='壁の長さと天井高が推定形状に合いません。計測値と推定した壁・天井を確認してください。'))
    coverage = supported_area / max(total_area, 1e-12)
    return dict(schema_version=1,
                summary=dict(photo_coverage=coverage, needs_review=bool(recommendations),
                             unsupported_surface_count=sum(s['supported_fraction'] < .8 for s in surfaces)),
                metrics=dict(area_weighted_photo_coverage=coverage, surface_count=len(surfaces),
                             source_frame_count=len(photo_sources), measurement_accuracy_verified=False),
                surfaces=details, recommendations=recommendations, calibration=calibration,
                photo_sources=photo_sources,
                limitations=['Photo coverage measures observed appearance, not geometric accuracy.',
                             'Unseen texels retain plain material; no generative inpainting was used.'])
