"""Target identity masks and bounded-memory, coherent face photo selection.

Semantic masks narrow existing depth/occlusion support; they never create it.
Unlabelled pixels are not positive wall evidence. Legacy layouts explicitly
retain their depth-only fallback, while incomplete modern evidence fails closed.
"""
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class SemanticPhotoMasks:
    def __init__(self, prediction, layout, height, width, max_bytes=32 * 1024**2):
        self.directory = Path(prediction).resolve()
        self.shape = (int(height), int(width))
        evidence = layout.get('image_furniture') or {}
        self.require_wall = int(evidence.get('version', 0)) >= 3
        self.frames, self.ids = {}, set()
        self.cache, self.cache_bytes, self.max_bytes = OrderedDict(), 0, int(max_bytes)
        self.errors = {}
        self.records = evidence.get('detections') or []
        for index, record in enumerate(self.records):
            try:
                frame = int(record.get('frame_index', record.get('frame')))
            except (TypeError, ValueError):
                self.errors[str(index)] = 'missing frame index'
                continue
            identity = str(record.get('detection_id', f'legacy-{index}'))
            kind = record.get('kind')
            if kind not in ('wall', 'unknown', 'furniture'):
                kind = 'wall' if record.get('label') == 'wall' else 'furniture'
            self.frames.setdefault(frame, []).append((identity, kind, record))
            self.ids.add(identity)

    def _frame(self, frame):
        if frame in self.cache:
            self.cache.move_to_end(frame)
            return self.cache[frame]
        loaded, used = [], 0
        for identity, kind, record in self.frames.get(frame, []):
            mask = None
            try:
                reference = record.get('mask_path', record.get('mask'))
                if not isinstance(reference, str) or not reference:
                    raise ValueError('missing mask path')
                path = (self.directory / reference).resolve()
                if not path.is_relative_to(self.directory):
                    raise ValueError('mask path escapes prediction directory')
                with Image.open(path) as image:
                    if image.size != self.shape[::-1]:
                        raise ValueError('mask camera dimensions differ')
                    mask = np.asarray(image.convert('L')) > 0
                used += mask.nbytes
            except (OSError, ValueError) as error:
                self.errors[identity] = str(error)
            loaded.append((identity, kind, mask))
        while self.cache and self.cache_bytes + used > self.max_bytes:
            _, old = self.cache.popitem(last=False)
            self.cache_bytes -= sum(item[2].nbytes for item in old if item[2] is not None)
        if used <= self.max_bytes:
            self.cache[frame] = loaded
            self.cache_bytes += used
        return loaded

    def mode(self, kind, object_box=None):
        if kind == 'object':
            return ('target_instance_mask' if object_box is not None and 'source_detection_ids' in object_box
                    else 'legacy_depth_only_without_object_identity')
        if kind == 'wall':
            return 'positive_wall_mask' if self.require_wall else 'legacy_depth_only_without_wall_labels'
        return 'depth_with_known_furniture_and_unknown_exclusion'

    def allowed(self, frame, uv, kind, object_box=None, wall_margin=0, allow_unlabelled_wall=False):
        """Sample conservative semantic support at camera-domain pixel centres."""
        records = self._frame(int(frame))
        blocked = np.zeros(self.shape, bool)
        positive = np.zeros(self.shape, bool)
        strict_object = kind == 'object' and object_box is not None and 'source_detection_ids' in object_box
        identities = {str(value) for value in (object_box or {}).get('source_detection_ids', [])}
        invalid = bool(strict_object and (not identities or not identities <= self.ids))
        target_seen = False
        for identity, category, mask in records:
            target = strict_object and identity in identities and category == 'furniture'
            relevant = category in ('furniture', 'unknown') or (kind == 'wall' and self.require_wall)
            if mask is None:
                invalid |= relevant
                continue
            if target:
                positive |= mask
                target_seen = True
            elif category in ('furniture', 'unknown'):
                blocked |= mask
            elif kind == 'wall' and category == 'wall':
                positive |= mask
        if strict_object:
            valid = cv2.erode(positive.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            valid &= ~blocked
            invalid |= not target_seen
        else:
            # A guard band keeps resampling from crossing labelled boundaries.
            blocked = cv2.dilate(blocked.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            if kind == 'wall' and self.require_wall and not allow_unlabelled_wall:
                if wall_margin:
                    radius=min(12,max(0,int(wall_margin)))
                    positive=cv2.dilate(positive.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1))).astype(bool)
                valid = cv2.erode(positive.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~blocked
            else:
                valid = ~blocked
        uv = np.asarray(uv)
        finite = np.isfinite(uv).all(axis=1)
        inside = finite & (uv[:, 0] >= 0) & (uv[:, 0] <= self.shape[1]-1) & (uv[:, 1] >= 0) & (uv[:, 1] <= self.shape[0]-1)
        pixel = np.rint(np.clip(np.nan_to_num(uv, nan=-1, posinf=-1, neginf=-1), -1, max(self.shape))).astype(np.int64)
        result = np.zeros(len(uv), bool)
        if not invalid:
            result[inside] = valid[pixel[inside, 1], pixel[inside, 0]]
        return result

    def summary(self):
        return dict(version=1, positive_wall_masks_required=self.require_wall,
                    detection_count=len(self.records), labelled_frame_count=len(self.frames),
                    invalid_masks=[dict(detection_id=identity, reason=reason) for identity, reason in sorted(self.errors.items())],
                    max_mask_cache_bytes=self.max_bytes,
                    limitations=['Segmentation and object association are model estimates, not independent ground truth.',
                                 'Unknown pixels do not support a wall; absent legacy wall labels use an explicit depth-only fallback.'])


def select_face_photos(frame_count, project, max_photos=2):
    """Stream geometry in bounded passes; never retain all-frame RGB.

    ``project(i)`` returns a boolean support mask and nonnegative quality scores.
    Primary maximizes total supported quality. Later views fill only unsupported
    texels, and cannot replace any previously painted texel.
    """
    if isinstance(max_photos, bool) or not isinstance(max_photos, int) or not 1 <= max_photos <= 8:
        raise ValueError('Surface photo limit must be 1..8')
    primary, primary_score, primary_support = None, 0., None
    candidates = []
    for frame in range(frame_count):
        support, scores = project(frame)
        support = np.asarray(support, bool) & np.isfinite(scores) & (np.asarray(scores) > 0)
        score = float(np.asarray(scores)[support].sum())
        if score <= 0:
            continue
        candidates.append(dict(frame=frame, supported_texels=int(support.sum()), quality_sum=score))
        if score > primary_score:
            primary, primary_score, primary_support = frame, score, support.copy()
    if primary is None:
        return [], dict(primary_frame=None, fill_frame=None, candidate_count=0)
    selected = [primary]
    covered = primary_support.copy()
    additional = 0
    # Large walls/floors may need several separate views. Keep existing pixels
    # fixed and accept each next view only where verified coverage is missing.
    while len(selected) < max_photos:
        best, best_score, best_fill = None, 0., None
        for item in candidates:
            frame = item['frame']
            if frame in selected:
                continue
            support, scores = project(frame)
            fill = np.asarray(support, bool) & ~covered & np.isfinite(scores) & (np.asarray(scores) > 0)
            score = float(np.asarray(scores)[fill].sum())
            if score > best_score:
                best, best_score, best_fill = frame, score, fill.copy()
        if best is None:
            break
        selected.append(best)
        additional += int(best_fill.sum())
        covered |= best_fill
        if covered.all():
            break
    return selected, dict(primary_frame=primary, fill_frame=selected[1] if len(selected) > 1 else None,
                          fill_frames=selected[1:],
                          primary_supported_texels=int(primary_support.sum()), additional_fill_texels=additional,
                          candidate_count=len(candidates), max_photos_per_face=max_photos)



def coherent_face_texture(count, material, selected, project, sample):
    """Paint the primary once and use later photographs only to fill holes."""
    texture = np.tile(np.asarray(material, np.uint8), (count, 1))
    supported = np.zeros(count, bool)
    frames = np.full(count, -1, np.int32)
    for frame in selected:
        support, scores = project(frame)
        take = np.asarray(support, bool) & ~supported & np.isfinite(scores) & (np.asarray(scores) > 0)
        if take.any():
            texture[take] = np.clip(sample(frame, take), 0, 255).astype(np.uint8)
            frames[take] = frame
            supported |= take
    return texture, supported, frames, np.full(count, -1, np.int32), np.zeros(count, np.float32)
