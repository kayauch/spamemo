"""Bake fixed surfaces using locally selected, visibility-verified photographs."""
import cv2
import numpy as np


def regional_face_texture(shape, material, frame_count, project, sample):
    """Two streaming passes, O(surface pixels) memory, no averaging across views.

    Smooth quality within a photo's valid footprint to discourage fragmented
    choices. Visibility is never expanded by smoothing. Every output texel
    retains its exact source frame; unsupported texels remain unsupported.
    """
    count = int(np.prod(shape))
    best = np.zeros(count, np.float32)
    sources = np.full(count, -1, np.int32)
    candidates = 0
    for frame in range(frame_count):
        support, score = project(frame)
        score = np.asarray(score, np.float32)
        valid = np.asarray(support, bool) & np.isfinite(score) & (score > 0)
        if not valid.any():
            continue
        candidates += 1
        weights = valid.astype(np.float32).reshape(shape)
        smooth = cv2.GaussianBlur(np.where(valid, score, 0).reshape(shape), (0, 0), 3)
        smooth /= np.maximum(cv2.GaussianBlur(weights, (0, 0), 3), 1e-8)
        quality = smooth.ravel()
        take = valid & (quality > best * 1.05)
        best[take] = quality[take]
        sources[take] = frame
    selected = np.unique(sources[sources >= 0]).tolist()
    texture = np.tile(np.asarray(material, np.uint8), (count, 1))
    for frame in selected:
        take = sources == frame
        texture[take] = np.clip(sample(frame, take), 0, 255).astype(np.uint8)
    supported = sources >= 0
    report = dict(method='regional_quality_selection', candidate_count=candidates,
                  selected_frames=selected, selected_frame_count=len(selected),
                  primary_frame=None, fill_frame=None, max_photos_per_face=frame_count,
                  quality_smoothing_sigma_texels=3, replacement_quality_ratio=1.05)
    return (texture, supported, sources, np.full(count, -1, np.int32),
            np.zeros(count, np.float32), selected, report)
