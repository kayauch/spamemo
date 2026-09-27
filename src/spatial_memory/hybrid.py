"""Conservative, observation-supported structural correction in world coordinates."""
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.ndimage import binary_fill_holes, label
from scipy.spatial import cKDTree


def correct_structure(points, colors, normals, components, layout, diagonal):
    points = np.asarray(points, dtype=float)
    corrected = points.copy()
    colors = np.asarray(colors, dtype=float)
    normals = np.asarray(normals, dtype=float)
    moved = np.zeros(len(points), dtype=bool)
    patches, patch_colors, faces, records = [], [], [], []
    tolerance = diagonal * 0.004
    cell = diagonal * 0.008
    raw = {c['id']: c for c in components}
    if not layout.get('enabled'):
        return corrected, moved, patches, patch_colors, faces, records
    for structure in layout.get('components', []):
        if structure['kind'] not in {'floor', 'wall', 'ceiling'}:
            continue
        if structure.get('metadata', {}).get('angular_residual_degrees', 180) > 8:
            continue
        for source_id in structure['metadata'].get('raw_component_ids', []):
            component = raw.get(source_id)
            if not component:
                continue
            axes = np.asarray(component['axes'])
            center = np.array([component['center'][k] for k in 'xyz'])
            plane = np.array(component['metadata']['plane'], dtype=float)
            plane /= np.linalg.norm(plane[:3])
            normal = plane[:3]
            distance = points @ normal + plane[3]
            local = (points - center) @ axes
            extent = np.array([component['size'][k] for k in 'xy'])
            supported = ((np.abs(distance) <= tolerance) &
                         (np.abs(local[:, :2]) <= extent / 2).all(axis=1) &
                         (np.abs(normals @ normal) >= 0.94))
            if supported.sum() < 100:
                continue
            selected = supported & ~moved
            corrected[selected] -= distance[selected, None] * normal
            moved |= selected
            uv = local[supported, :2]
            low = uv.min(axis=0) - cell * 0.5
            cells = np.floor((uv - low) / cell).astype(int)
            shape = cells.max(axis=0) + 2
            if np.prod(shape) > 250_000:
                continue
            occupied = np.zeros(shape, dtype=bool)
            occupied[tuple(cells.T)] = True
            # Only enclosed gaps; never extend a surface into unseen borders.
            gaps = binary_fill_holes(occupied) & ~occupied
            labels, count = label(gaps)
            sizes = np.bincount(labels.ravel())
            small = np.flatnonzero((sizes <= 16) & (sizes > 0))
            small = small[small != 0]
            fill_cells = np.argwhere(np.isin(labels, small))
            tree = cKDTree(uv)
            source_colors = colors[supported]
            filled = 0
            for grid in fill_cells:
                middle = low + (grid + 0.5) * cell
                distances, neighbors = tree.query(middle, k=4)
                if distances[-1] > cell * 3:
                    continue
                corners = low + (grid + np.array([[0, 0], [1, 0], [1, 1], [0, 1]])) * cell
                world = center + corners @ axes[:, :2].T
                world -= (world @ normal + plane[3])[:, None] * normal
                offset = len(patches)
                patches.extend(world.tolist())
                patch_colors.extend([np.median(source_colors[neighbors], axis=0).tolist()] * 4)
                faces.extend([[offset, offset+1, offset+2], [offset, offset+2, offset+3]])
                filled += 1
            records.append({'id': source_id, 'kind': structure['kind'],
                            'corrected_points': int(selected.sum()), 'filled_cells': filled})
    return corrected, moved, patches, patch_colors, faces, records


def build_hybrid(source, shell_path, components, layout, diagonal, output):
    cloud = o3d.io.read_point_cloud(str(source))
    xyz = np.asarray(cloud.points)
    if not cloud.has_colors():
        cloud.colors = o3d.utility.Vector3dVector(np.full_like(xyz, 0.6))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=diagonal * 0.015, max_nn=30))
    corrected, moved, vertices, colors, triangles, records = correct_structure(
        xyz, np.asarray(cloud.colors), np.asarray(cloud.normals), components, layout, diagonal)
    # Drop only isolated single points. Dense furniture and thin connected objects survive.
    tree = cKDTree(xyz)
    neighbor_distance = tree.query(xyz, k=2, workers=2)[0][:, 1]
    keep = neighbor_distance <= diagonal * 0.015
    result = cloud.select_by_index(np.flatnonzero(keep))
    result.points = o3d.utility.Vector3dVector(corrected[keep])
    point_path = Path(output) / 'hybrid-points.ply'
    if not o3d.io.write_point_cloud(str(point_path), result):
        raise RuntimeError('Cannot write hybrid points')
    mesh = o3d.io.read_triangle_mesh(str(shell_path))
    mesh.compute_vertex_normals()
    mesh_xyz = np.asarray(mesh.vertices).copy()
    nearest_distance, nearest = tree.query(mesh_xyz, workers=2)
    # Transfer bounded displacements only near observed structural points.
    eligible = moved[nearest] & (nearest_distance <= diagonal * 0.008)
    delta = corrected[nearest] - xyz[nearest]
    aligned = np.abs(np.sum(np.asarray(mesh.vertex_normals) * np.asarray(cloud.normals)[nearest], axis=1)) >= 0.94
    mesh_xyz[eligible & aligned] += delta[eligible & aligned]
    mesh.vertices = o3d.utility.Vector3dVector(mesh_xyz)
    mesh.remove_vertices_by_mask(~keep[nearest])
    if vertices:
        patch = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))
        patch.vertex_colors = o3d.utility.Vector3dVector(colors)
        mesh += patch
    mesh.compute_vertex_normals()
    mesh_path = Path(output) / 'hybrid-shell.ply'
    if not o3d.io.write_triangle_mesh(str(mesh_path), mesh):
        raise RuntimeError('Cannot write hybrid mesh')
    return {'point_path': str(point_path), 'shell_path': str(mesh_path),
            'input_points': len(xyz), 'output_points': len(result.points),
            'corrected_points': int(moved.sum()), 'isolated_points_removed': int((~keep).sum()),
            'patch_triangles': len(triangles), 'planes': records,
            'max_displacement': float(np.linalg.norm(corrected-xyz, axis=1).max()),
            'method': 'observed-planes-small-enclosed-gaps-v1'}
