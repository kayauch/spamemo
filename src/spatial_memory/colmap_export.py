"""VGGT world coordinates -> COLMAP text model without another SfM solve."""
import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from .texturing import preprocess_frame


def colmap_pose(extrinsic):
    matrix = np.asarray(extrinsic, dtype=float)
    if matrix.shape != (3, 4) or not np.isfinite(matrix).all():
        raise ValueError('Expected finite world-to-camera 3x4 matrix')
    r = matrix[:, :3]
    if not np.allclose(r @ r.T, np.eye(3), atol=1e-4) or np.linalg.det(r) < .999:
        raise ValueError('Invalid camera rotation')
    xyzw = Rotation.from_matrix(r).as_quat()
    q = xyzw[[3, 0, 1, 2]]
    if q[0] < 0:
        q *= -1
    return q, matrix[:, 3].copy()


def export_scene(scene_dir, output_dir, max_points=50000):
    scene, out = Path(scene_dir).resolve(), Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError(f'Choose a new output directory: {out}')
    if max_points < 1:
        raise ValueError('max_points must be positive')
    metrics = json.loads((scene / 'reconstruction/metrics.json').read_text(encoding='utf-8'))
    if metrics.get('model') != 'VGGT-1B' or metrics.get('fallback'):
        raise ValueError('Requires real VGGT reconstruction')
    npz = scene / 'reconstruction/reconstruction.npz'
    with np.load(npz, allow_pickle=False) as result:
        depth, extrinsics, intrinsics = [result[key] for key in ('depth', 'extrinsics', 'intrinsics')]
    if depth.ndim == 4:
        depth = depth[..., 0]
    n, h, w = depth.shape
    frames = json.loads((scene / 'frames/manifest.json').read_text(encoding='utf-8'))['frames']
    if len(frames) != n or extrinsics.shape != (n, 3, 4) or intrinsics.shape != (n, 3, 3):
        raise ValueError('Camera, depth and frame counts do not match')
    if not np.isfinite(intrinsics).all() or (intrinsics[:, [0, 1], [0, 1]] <= 0).any():
        raise ValueError('Invalid camera intrinsics')
    # Official 2DGS loader ignores principal-point offsets. Refuse silent misalignment.
    if not np.allclose(intrinsics[:, :2, 2], [w/2, h/2], atol=1e-4):
        raise ValueError('Official 2DGS requires centered principal points; rectify first')
    poses = [colmap_pose(e) for e in extrinsics]
    cloud = o3d.io.read_point_cloud(str(scene / 'reconstruction/scene.ply'))
    points = np.asarray(cloud.points)
    if not len(points) or not cloud.has_colors():
        raise ValueError('Missing colored initialization points')
    rng = np.random.default_rng(42)
    selected = np.sort(rng.choice(len(points), min(max_points, len(points)), replace=False))
    points = points[selected]
    rgb = np.rint(np.clip(np.asarray(cloud.colors)[selected], 0, 1) * 255).astype(int)
    if not np.isfinite(points).all():
        raise ValueError('Nonfinite initialization points')
    sparse = out / 'sparse/0'
    sparse.mkdir(parents=True)
    (out / 'images').mkdir()
    tracks = [[] for _ in points]
    camera_lines, image_lines = [], []
    max_projection_error = 0.
    for i, (frame, e, k, pose) in enumerate(zip(frames, extrinsics, intrinsics, poses)):
        name = f'{i:06d}.png'
        preprocess_frame(scene / 'frames' / frame['name'], h, w).save(out / 'images' / name)
        q, t = pose
        camera_lines.append(f'{i+1} PINHOLE {w} {h} ' + ' '.join(f'{v:.12g}' for v in (k[0,0], k[1,1], k[0,2], k[1,2])))
        image_lines.append(f'{i+1} ' + ' '.join(f'{v:.12g}' for v in [*q, *t]) + f' {i+1} {name}')
        camera = points @ e[:, :3].T + e[:, 3]
        z = camera[:, 2]
        uv = (camera @ k.T)[:, :2] / np.maximum(z[:, None], 1e-8)
        inside = (z > 0) & (uv[:,0] >= 0) & (uv[:,0] < w-1) & (uv[:,1] >= 0) & (uv[:,1] < h-1)
        ids = np.flatnonzero(inside)
        pixels = np.rint(uv[ids]).astype(int)
        observed = depth[i, pixels[:,1], pixels[:,0]]
        ids = ids[np.isfinite(observed) & (observed > 0) & (np.abs(observed-z[ids]) <= .03*observed)]
        # Synthetic tracks describe projections of VGGT points, not matched image features.
        entries = []
        for observation, point_id in enumerate(ids):
            entries.append(f'{uv[point_id,0]:.8f} {uv[point_id,1]:.8f} {point_id+1}')
            tracks[point_id].extend([i+1, observation])
        image_lines.append(' '.join(entries))
        rebuilt = Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        test = points @ rebuilt.T + t
        reproj = (test @ k.T)[:, :2] / np.maximum(test[:, 2:3], 1e-8)
        if len(ids):
            max_projection_error = max(max_projection_error, float(np.abs(reproj[ids]-uv[ids]).max()))
    (sparse / 'cameras.txt').write_text('# CAMERA_ID MODEL WIDTH HEIGHT PARAMS\n'+'\n'.join(camera_lines)+'\n')
    (sparse / 'images.txt').write_text('# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n# POINTS2D: X Y POINT3D_ID\n'+'\n'.join(image_lines)+'\n')
    with (sparse / 'points3D.txt').open('w') as f:
        f.write('# POINT3D_ID X Y Z R G B ERROR TRACK; VGGT synthetic observations\n')
        for i, (p, color, track) in enumerate(zip(points, rgb, tracks)):
            f.write(f'{i+1} '+' '.join(f'{v:.12g}' for v in p)+' '+' '.join(map(str,color))+' 0 '+' '.join(map(str,track))+'\n')
    info = {'scene_id': scene.name, 'source_npz_sha256': hashlib.sha256(npz.read_bytes()).hexdigest(),
            'frames': n, 'points': len(points), 'image_size': [w,h], 'seed': 42,
            'world_from_2dgs': np.eye(4).tolist(), 'camera_convention': 'COLMAP/OpenCV world-to-camera, quaternion wxyz',
            'geometry_source': 'VGGT; no new SfM or bundle adjustment', 'tracks': 'synthetic depth-consistent projections, not feature matches',
            'max_pose_roundtrip_pixel_error': max_projection_error,
            'image_preprocessing': 'VGGT crop: width 518, height rounded to 14, centered vertical crop',
            'cameras_fixed': True}
    (out / 'alignment.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    return info
