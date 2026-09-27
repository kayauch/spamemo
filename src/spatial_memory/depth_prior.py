"""Conservative masks for VGGT depth supervision (not ground-truth depth)."""
import numpy as np


def reliable_depth_mask(depth, confidence):
    depth = np.asarray(depth)
    confidence = np.asarray(confidence)
    if depth.ndim != 2 or confidence.shape != depth.shape:
        raise ValueError('Depth and confidence must have matching H,W shapes')
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(confidence) & (confidence > 1.01)
    if not valid.any():
        return valid
    valid &= confidence >= np.quantile(confidence[valid], .5)
    # Avoid silhouettes and sharp depth discontinuities; do not wrap image borders.
    safe = np.where(np.isfinite(depth) & (depth > 0), depth, 0)
    for axis in (0, 1):
        delta = np.abs(np.diff(safe, axis=axis))
        a, b = [slice(None)] * 2, [slice(None)] * 2
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        good = delta <= .08 * np.minimum(safe[tuple(a)], safe[tuple(b)])
        valid[tuple(a)] &= good
        valid[tuple(b)] &= good
    valid[[0, -1], :] = False
    valid[:, [0, -1]] = False
    return valid


def multi_view_depth_masks(depth, confidence, extrinsics, intrinsics, tolerance=.03):
    """Require two other training views to agree; held-out images never vote.

    This checks geometry consistency, not statistical independence of VGGT views.
    """
    depth, confidence = np.asarray(depth), np.asarray(confidence)
    if depth.ndim != 3 or confidence.shape != depth.shape:
        raise ValueError('Expected matching N,H,W depth and confidence')
    n, h, w = depth.shape
    masks = np.zeros_like(depth, dtype=bool)
    counts = np.zeros_like(depth, dtype=np.uint8)
    for i in range(n):
        if i % 8 == 0:
            continue
        base = reliable_depth_mask(depth[i], confidence[i])
        y, x = np.nonzero(base)
        pixel = np.column_stack([x, y, np.ones(len(x))])
        rays = pixel @ np.linalg.inv(intrinsics[i]).T
        camera = rays * depth[i, y, x, None]
        e = extrinsics[i]
        world = (camera - e[:, 3]) @ e[:, :3]
        support = np.zeros(len(x), dtype=np.uint8)
        # Nearby and wider-baseline neighbors; exclude the current/held-out view.
        candidates = sorted({i + delta for delta in (-12,-9,-6,-3,-2,-1,1,2,3,6,9,12)
                             if 0 <= i + delta < n and (i + delta) % 8 != 0})
        for j in candidates:
            cam = world @ extrinsics[j, :, :3].T + extrinsics[j, :, 3]
            z = cam[:, 2]
            uv = cam @ intrinsics[j].T
            uv = uv[:, :2] / np.maximum(z[:, None], 1e-8)
            inside = np.isfinite(uv).all(axis=1) & (z > 0) & (uv[:, 0] >= 1) & (uv[:, 0] < w-2) & (uv[:, 1] >= 1) & (uv[:, 1] < h-2)
            ids = np.flatnonzero(inside)
            p = np.rint(uv[ids]).astype(np.int64)
            d = depth[j, p[:, 1], p[:, 0]]
            c = confidence[j, p[:, 1], p[:, 0]]
            valid = np.isfinite(d) & (d > 0) & np.isfinite(c) & (c > 1.01)
            support[ids] += valid & (np.abs(d-z[ids]) <= tolerance * d)
        masks[i, y, x] = support >= 2
        counts[i, y, x] = support
    return masks, counts
