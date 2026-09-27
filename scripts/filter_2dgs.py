"""Build a reversible candidate removing splats unsupported by training-view depth.

No evaluation RGB or evaluation depth is used for pruning decisions.
The full PLY record is kept verbatim for retained splats, including SH/rotation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def depth_support(xyz, depth, confidence, extrinsics, intrinsics, tolerance):
    support = np.zeros(len(xyz), dtype=np.int32)
    observed = np.zeros_like(support)
    front = np.zeros_like(support)
    for i in range(len(depth)):
        if i % 8 == 0:  # exactly the official appearance hold-out split
            continue
        e, k = extrinsics[i], intrinsics[i]
        cam = xyz @ e[:, :3].T + e[:, 3]
        z = cam[:, 2]
        uv = cam @ k.T
        uv = uv[:, :2] / np.maximum(z[:, None], 1e-8)
        h, w = depth[i].shape
        inside = np.isfinite(uv).all(axis=1) & (z > 0) & (uv[:, 0] >= 1) & (uv[:, 0] < w-2) & (uv[:, 1] >= 1) & (uv[:, 1] < h-2)
        ids = np.flatnonzero(inside)
        pixel = np.rint(uv[ids]).astype(np.int64)
        d = depth[i, pixel[:, 1], pixel[:, 0]]
        c = confidence[i, pixel[:, 1], pixel[:, 0]]
        valid = np.isfinite(d) & (d > 0) & np.isfinite(c) & (c > 1.01)
        relative = (z[ids] - d) / np.maximum(d, 1e-8)
        support[ids] += valid & (np.abs(relative) <= tolerance)
        observed[ids] += valid
        front[ids] += valid & (relative < -tolerance)
    return support, observed, front


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('geometry_npz', type=Path)
    parser.add_argument('--iteration', type=int, default=3000)
    parser.add_argument('--tolerance', type=float, default=.08)
    parser.add_argument('--minimum-views', type=int, default=2)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError('Choose a new output directory')
    source = args.model_dir / f'point_cloud/iteration_{args.iteration}/point_cloud.ply'
    ply = PlyData.read(source)
    values = ply['vertex'].data
    xyz = np.column_stack([values[k] for k in ('x', 'y', 'z')])
    with np.load(args.geometry_npz, allow_pickle=False) as data:
        d, c = data['depth'], data['confidence']
        if d.ndim == 4: d = d[..., 0]
        if c.ndim == 4: c = c[..., 0]
        support, observed, front = depth_support(xyz, d, c, data['extrinsics'], data['intrinsics'], args.tolerance)
    keep = (support >= args.minimum_views) & np.isfinite(xyz).all(axis=1)
    if not keep.any():
        raise ValueError('No supported splats')
    target = args.output_dir / f'point_cloud/iteration_{args.iteration}/point_cloud.ply'
    target.parent.mkdir(parents=True)
    PlyData([PlyElement.describe(values[keep], 'vertex')], text=False).write(target)
    report = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  source_npz_sha256=hashlib.sha256(args.geometry_npz.read_bytes()).hexdigest(),
                  source_splats=len(xyz), retained_splats=int(keep.sum()), removed_splats=int((~keep).sum()),
                  relative_depth_tolerance=args.tolerance, minimum_views=args.minimum_views,
                  support_quantiles=np.quantile(support, [0,.1,.5,.9,1]).tolist(),
                  validation_frames_excluded=True, method='center_depth_support_v1')
    (args.output_dir / 'filtering.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
