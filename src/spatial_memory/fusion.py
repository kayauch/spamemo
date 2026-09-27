"""Keep only mesh faces with multi-view VGGT depth support for display fusion."""
import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from .texturing import project_visible


def supported_faces(triangles, depth, extrinsics, intrinsics, confidence, minimum_views=2):
    if minimum_views < 2 or minimum_views > len(depth):
        raise ValueError('At least two observed views are required')
    samples = np.concatenate([triangles, (triangles + np.roll(triangles, 1, axis=1))/2,
                              triangles.mean(axis=1)[:, None]], axis=1)
    support = np.zeros(len(triangles), dtype=np.int32)
    for d, e, k, c in zip(depth, extrinsics, intrinsics, confidence, strict=True):
        _, valid = project_visible(samples, e, k, d, c)
        support += valid
    return support >= minimum_views, support


def build_fusion_shell(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    scene = manifest_path.parent.parent
    metrics = json.loads((scene / 'reconstruction/metrics.json').read_text(encoding='utf-8'))
    if metrics.get('model') != 'VGGT-1B' or metrics.get('fallback'):
        raise ValueError('Fusion requires real VGGT depth and cameras')
    npz = scene / 'reconstruction/reconstruction.npz'
    with np.load(npz, allow_pickle=False) as src:
        depth, confidence, extrinsics, intrinsics = (src[k] for k in ('depth', 'confidence', 'extrinsics', 'intrinsics'))
    if depth.ndim == 4: depth = depth[..., 0]
    if confidence.ndim == 4: confidence = confidence[..., 0]
    path = Path(manifest.get('hybrid', {}).get('shell_path') or manifest['shell']['path'])
    mesh = o3d.io.read_triangle_mesh(str(path))
    triangles = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]
    keep, counts = supported_faces(triangles, depth, extrinsics, intrinsics, confidence)
    if not keep.any():
        raise ValueError('No faces have multi-view depth support; leave existing manifest unchanged')
    before = len(triangles)
    mesh.remove_triangles_by_mask(~keep)
    mesh.remove_unreferenced_vertices()
    digest = hashlib.sha256(path.read_bytes() + keep.tobytes()).hexdigest()[:16]
    output = manifest_path.parent / f'fusion-supported-{digest}.ply'
    if not o3d.io.write_triangle_mesh(str(output), mesh):
        raise OSError('Could not write supported mesh')
    stats = dict(shell_path=str(output), revision=digest, source_triangles=before,
                 triangles=int(keep.sum()), removed_triangles=int((~keep).sum()),
                 minimum_views=2, maximum_support=int(counts.max()),
                 depth_tolerance=.025, samples_per_face=7,
                 source_npz_sha256=hashlib.sha256(npz.read_bytes()).hexdigest(),
                 method='multi_view_depth_supported_faces_v1')
    manifest['fusion'] = stats
    tmp = manifest_path.with_suffix('.fusion.tmp')
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(manifest_path)
    return stats
