"""Conservative image-space repair provenance and depth-checked reprojection.

Labels describe evidence, not semantic ground truth. Never invent geometry.
"""
import cv2
import numpy as np

UNKNOWN, RECONSTRUCTED, OBSERVED, AI_REPAIRED, AI_GENERATED = range(5)


def world_points(depth, extrinsic, intrinsic):
    y, x = np.indices(depth.shape)
    rays = np.stack((x, y, np.ones_like(x)), -1) @ np.linalg.inv(intrinsic).T
    camera = rays * depth[..., None]
    return (camera - extrinsic[:, 3]) @ extrinsic[:, :3]


def correspondence(target_depth, target_e, target_k, source_depth, source_e, source_k, tolerance=.025):
    """Inverse warp target pixels to source, rejecting occlusion and depth edges."""
    points = world_points(target_depth, target_e, target_k)
    camera = points @ source_e[:, :3].T + source_e[:, 3]
    projected = camera @ source_k.T
    uv = projected[..., :2] / np.maximum(camera[..., 2:3], 1e-8)
    finite = np.isfinite(uv).all(-1) & np.isfinite(target_depth) & (target_depth > 0)
    uv = np.nan_to_num(uv, nan=-1, posinf=-1, neginf=-1).astype(np.float32)
    h, w = source_depth.shape
    valid = finite & (camera[..., 2] > 0) & (uv[..., 0] >= 1) & (uv[..., 0] < w-2) & (uv[..., 1] >= 1) & (uv[..., 1] < h-2)
    sampled = cv2.remap(source_depth.astype(np.float32), uv[..., 0], uv[..., 1], cv2.INTER_LINEAR)
    valid &= np.isfinite(sampled) & (sampled > 0) & (np.abs(sampled-camera[..., 2]) <= tolerance*np.maximum(sampled, 1e-8))
    return uv, valid


def sample(array, uv, nearest=False):
    return cv2.remap(array, uv[..., 0], uv[..., 1], cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)


def classify(render, photo, alpha, depth, confidence_support):
    """GOOD stays locked; BROKEN needs observed geometric support; UNKNOWN is deferred."""
    finite = np.isfinite(depth) & (depth > 0) & np.isfinite(alpha)
    error = np.abs(render.astype(np.float32)-photo.astype(np.float32)).mean(-1)/255
    observed = finite & confidence_support
    # Remove isolated pixels; keep only errors supported over a small patch.
    broken = observed & ((error > .12) | (alpha < .8))
    broken = cv2.morphologyEx(broken.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3,3),np.uint8)).astype(bool)
    labels = np.full(depth.shape, UNKNOWN, np.uint8)
    labels[finite & (alpha > .5)] = RECONSTRUCTED
    labels[observed] = OBSERVED
    return labels, broken, error


def bounded_region(mask, maximum_fraction=.08):
    """Select a single interior connected region, without changing the rest of the view."""
    mask = mask.copy().astype(np.uint8)
    mask[:3] = 0; mask[-3:] = 0; mask[:,:3] = 0; mask[:,-3:] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    choices = [i for i in range(1,count) if 20 <= stats[i,cv2.CC_STAT_AREA] <= mask.size*maximum_fraction]
    if not choices: return np.zeros_like(mask, dtype=bool)
    return labels == max(choices, key=lambda i:stats[i,cv2.CC_STAT_AREA])


def composite_locked(original, proposal, mask):
    if proposal.shape != original.shape or mask.shape != original.shape[:2]:
        raise ValueError('Repair dimensions differ; do not silently resize proposals')
    result = original.copy()
    result[mask] = proposal[mask]
    return result


def repair_decision(baseline, candidate, consistency):
    """Use real held-out photos, teacher depth stability, holes and worst-view checks."""
    reasons=[]
    for key in ('alignment_sha256','camera_files_sha256','image_size'):
        if not baseline.get(key) or baseline.get(key)!=candidate.get(key): reasons.append(f'Incomparable {key}')
    before={r['frame']:r for r in baseline['test_views']}
    after={r['frame']:r for r in candidate['test_views']}
    if len(before)<3 or before.keys()!=after.keys(): reasons.append('Evaluation frames differ or insufficient')
    if not candidate.get('finite_parameters') or candidate.get('max_camera_center_error', 1)>1e-4: reasons.append('Invalid model')
    if not consistency.get('passed'): reasons.append('Multi-view repair consistency failed')
    if candidate.get('repair_reference_model_sha256')!=baseline.get('model_ply_sha256'):
        reasons.append('Repair reference is not the current best')
    drift=candidate.get('mean_depth_to_best_relative',float('inf'))
    holes=candidate.get('mean_hole_fraction',float('inf'))-baseline.get('mean_hole_fraction',0)
    if not np.isfinite([drift,holes]).all() or drift>.005 or holes>.002:
        reasons.append('Depth drift or hole coverage regressed')
    multiview=candidate.get('repair_multiview_mae',float('inf'))-baseline.get('repair_multiview_mae',0)
    if not np.isfinite(multiview) or multiview>.005:
        reasons.append('Rendered multi-view appearance consistency regressed')
    if reasons: return dict(accepted=False, reasons=reasons)
    delta={k:float(np.mean([after[f][k]-before[f][k] for f in before])) for k in ('psnr','ssim')}
    worst=min(after[f]['psnr']-before[f]['psnr'] for f in before)
    if not np.isfinite([*delta.values(),worst]).all(): reasons.append('Nonfinite metrics')
    if delta['psnr']<-.02 or delta['ssim']<-.002: reasons.append('Mean appearance regressed')
    if worst<-.3: reasons.append('Individual view regressed by more than 0.3 dB')
    if not (delta['psnr']>=.05 or delta['ssim']>=.001): reasons.append('No measurable held-out appearance improvement')
    return dict(accepted=not reasons, reasons=reasons, mean_delta=delta, worst_psnr_delta=worst,
                scope='Appearance frame holdout; VGGT/calibration used all frames; not unseen-room ground truth')
