"""Depth-checked photo projection onto existing structural triangles.

Coordinates remain in reconstruction world space. No geometry is invented.
VGGT crop preprocessing is reproduced from the recorded input dimensions.
"""
import json
import math
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image


def preprocess_frame(path, height, width, scale=1, preprocessing='vggt'):
    if not isinstance(scale, int) or scale < 1:
        raise ValueError('Image scale must be a positive integer')
    with Image.open(path) as source:
        image = source.convert('RGB')
        if preprocessing == 'aligned_rgbd':
            if abs((image.width/image.height)/(width/height)-1) > .01:
                raise ValueError('RGB-D image and depth camera aspect ratios differ')
            return image.resize((width*scale,height*scale), Image.Resampling.BICUBIC)
        resized_height = round(image.height * (518 / image.width) / 14) * 14
        image = image.resize((518 * scale, resized_height * scale), Image.Resampling.BICUBIC)
        if resized_height > 518:
            top = (resized_height - 518) // 2
            image = image.crop((0, top * scale, 518 * scale, (top + 518) * scale))
        # VGGT pads mixed landscape/portrait/cubemap inputs to the batch maximum
        # after each image's crop. Reproduce that exact camera pixel domain.
        if image.width <= width*scale and image.height <= height*scale:
            pad_top=(height-image.height//scale)//2
            pad_left=(width-image.width//scale)//2
            padded=Image.new('RGB',(width*scale,height*scale),'white')
            padded.paste(image,(pad_left*scale,pad_top*scale))
            image=padded
        if image.size != (width * scale, height * scale):
            raise ValueError('Frame preprocessing does not match saved VGGT camera dimensions')
        return image


def project_visible(samples, extrinsic, intrinsic, depth, confidence=None):
    """Reject behind-camera, off-image and depth-inconsistent triangle samples."""
    camera = samples @ extrinsic[:, :3].T + extrinsic[:, 3]
    z = camera[..., 2]
    projected = camera @ intrinsic.T
    uv = projected[..., :2] / np.maximum(z[..., None], 1e-8)
    h, w = depth.shape
    finite = np.isfinite(uv).all(axis=-1) & np.isfinite(z)
    inside = finite & (z > 0) & (uv[..., 0] >= 1) & (uv[..., 0] < w - 2) & (uv[..., 1] >= 1) & (uv[..., 1] < h - 2)
    pixel = np.rint(np.nan_to_num(uv, nan=0, posinf=0, neginf=0)).astype(np.int64)
    x, y = np.clip(pixel[..., 0], 0, w-1), np.clip(pixel[..., 1], 0, h-1)
    observed = depth[y, x]
    valid = inside & np.isfinite(observed) & (observed > 0) & (np.abs(observed-z) <= 0.025 * np.maximum(observed, 1e-8))
    if confidence is not None:
        values = confidence[y, x]
        valid &= np.isfinite(values) & (values > 1.01)
    return uv, valid.all(axis=1)


def structural_faces(triangles, manifest):
    selected = np.zeros(len(triangles), dtype=bool)
    if not manifest.get('structural_layout', {}).get('enabled'):
        return selected
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    raw = {c['id']: c for c in manifest['components']}
    tolerance = manifest['metrics']['scene_diagonal'] * 0.006
    for structure in manifest.get('structural_layout', {}).get('components', []):
        if structure['kind'] not in {'floor', 'wall', 'ceiling'}:
            continue
        for source in structure.get('metadata', {}).get('raw_component_ids', []):
            c = raw.get(source)
            if not c or 'plane' not in c.get('metadata', {}):
                continue
            plane = np.array(c['metadata']['plane'], dtype=float)
            plane /= np.linalg.norm(plane[:3])
            center = np.array([c['center'][k] for k in 'xyz'])
            local = (triangles - center) @ np.array(c['axes'])
            extent = np.array([c['size'][k] for k in 'xy']) / 2
            selected |= ((np.abs(triangles @ plane[:3] + plane[3]) <= tolerance).all(axis=1)
                         & (np.abs(local[..., :2]) <= extent).all(axis=(1, 2))
                         & (np.abs(normals @ plane[:3]) >= .9))
    return selected


def build_textured_scene(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    scene = manifest_path.parent.parent
    metrics = json.loads((scene / 'reconstruction/metrics.json').read_text(encoding='utf-8'))
    if metrics.get('model') != 'VGGT-1B' or metrics.get('fallback'):
        raise ValueError('Photo projection requires recorded VGGT cameras and crop preprocessing')
    with np.load(scene / 'reconstruction/reconstruction.npz', allow_pickle=False) as saved:
        depth, extrinsics, intrinsics = (saved[k] for k in ('depth', 'extrinsics', 'intrinsics'))
        confidence = saved['confidence'] if 'confidence' in saved else None
    if depth.ndim == 4:
        depth = depth[..., 0]
    if confidence is not None and confidence.ndim == 4:
        confidence = confidence[..., 0]
    records = json.loads((scene / 'frames/manifest.json').read_text(encoding='utf-8'))['frames']
    n, h, w = depth.shape
    if len(records) != n or len(extrinsics) != n or len(intrinsics) != n:
        raise ValueError('Frame and camera counts differ')
    shell = Path(manifest.get('hybrid', {}).get('shell_path') or manifest['shell']['path'])
    mesh = o3d.io.read_triangle_mesh(str(shell))
    faces = np.asarray(mesh.triangles)
    triangles = np.asarray(mesh.vertices)[faces]
    colors = np.asarray(mesh.vertex_colors)[faces] if mesh.has_vertex_colors() else np.full_like(triangles, .6)
    eligible = structural_faces(triangles, manifest)
    # Vertices, edge midpoints and centroid: conservative depth checks across each face.
    samples = np.concatenate([triangles, (triangles + np.roll(triangles, 1, axis=1))/2, triangles.mean(axis=1)[:, None]], axis=1)
    choice = np.full(len(faces), -1, dtype=int)
    best = np.zeros(len(faces))
    photo_uv = np.zeros((len(faces), 3, 2))
    sharpness = np.array([max(0, r.get('sharpness') or 0) for r in records])
    sharpness = .5 + .5 * sharpness / max(sharpness.max(), 1)
    for i in range(n):
        uv, visible = project_visible(samples, extrinsics[i], intrinsics[i], depth[i], None if confidence is None else confidence[i])
        a, b = uv[:, 1]-uv[:, 0], uv[:, 2]-uv[:, 0]
        area = np.abs(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]) / 2
        # Pixel footprint favors close, front-facing views; sharpness discourages blur.
        score = area * sharpness[i]
        better = eligible & visible & (area >= 2) & (score > best)
        choice[better], best[better], photo_uv[better] = i, score[better], uv[better, :3]
    used = np.unique(choice[choice >= 0])
    if not len(used):
        raise ValueError('No structural faces pass photo visibility checks')
    # A white tile preserves vertex colors on unsupported surfaces and furniture.
    cols = math.ceil(math.sqrt((len(used)+1) * h / w))
    rows = math.ceil((len(used)+1) / cols)
    if max(cols*w, rows*h) > 8192:
        raise ValueError('Photo atlas exceeds supported size; reconstruct with fewer frames')
    atlas = Image.new('RGB', (cols*w, rows*h), 'white')
    uv_out = np.empty_like(photo_uv)
    uv_out[:] = [.5*w/atlas.width, 1-.5*h/atlas.height]
    output_colors = colors.copy()
    for tile, frame in enumerate(used, start=1):
        image = preprocess_frame(scene / 'frames' / records[frame]['name'], h, w)
        left, top = (tile % cols)*w, (tile // cols)*h
        atlas.paste(image, (left, top))
        mask = choice == frame
        uv_out[mask, :, 0] = (photo_uv[mask, :, 0] + left + .5) / atlas.width
        uv_out[mask, :, 1] = 1 - (photo_uv[mask, :, 1] + top + .5) / atlas.height
        output_colors[mask] = 1
    # Content-addressed files keep old browser requests coherent during rebuilds.
    import hashlib
    payload = json.dumps({'positions': triangles.reshape(-1).tolist(), 'uv': uv_out.reshape(-1).tolist(),
                          'colors': output_colors.reshape(-1).tolist()}, separators=(',', ':'))
    digest = hashlib.sha256(payload.encode() + atlas.tobytes()).hexdigest()[:16]
    output = manifest_path.parent
    geometry_path = output / f'textured-{digest}.json'
    atlas_path = output / f'textured-{digest}.jpg'
    geometry_path.write_text(payload, encoding='utf-8')
    atlas.save(atlas_path, quality=95, subsampling=0)
    areas = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1) / 2
    stats = {'geometry_path': str(geometry_path), 'atlas_path': str(atlas_path), 'revision': digest,
             'triangles': len(faces), 'eligible_triangles': int(eligible.sum()),
             'textured_triangles': int((choice >= 0).sum()), 'frames_used': len(used),
             'textured_area_ratio': float(areas[choice >= 0].sum() / max(areas.sum(), 1e-12)),
             'atlas_size': list(atlas.size), 'method': 'depth_checked_structure_projection_v1'}
    manifest['texture'] = stats
    temporary = manifest_path.with_suffix('.texture.tmp')
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(manifest_path)
    return stats
