import numpy as np
from spatial_memory.hybrid import correct_structure


def fixture():
    # Cell-aligned sampling: one enclosed small gap, one large gap and an open edge.
    cell = 0.016
    ij = np.indices((40, 40)).reshape(2, -1).T
    missing = ((ij == [10, 10]).all(axis=1) |
               ((ij[:, 0] >= 20) & (ij[:, 0] < 26) & (ij[:, 1] >= 20) & (ij[:, 1] < 26)) |
               ((ij[:, 0] < 4) & (ij[:, 1] < 4)))
    uv = (ij[~missing] + 0.5) * cell
    points = np.column_stack([uv, np.full(len(uv), 0.003)])
    normals = np.tile([0., 0., 1.], (len(points), 1))
    colors = np.tile([0.2, 0.6, 0.8], (len(points), 1))
    plane = {'id': 'p', 'axes': np.eye(3).tolist(), 'center': dict(x=.32, y=.32, z=0),
             'size': dict(x=.64, y=.64, z=.01), 'metadata': {'plane': [0, 0, 1, 0]}}
    layout = {'enabled': True, 'components': [{'kind': 'floor', 'metadata': {
        'angular_residual_degrees': 0, 'raw_component_ids': ['p']}}]}
    return points, colors, normals, [plane], layout, 2.


def test_small_enclosed_gap_filled_but_large_and_open_gaps_retained():
    args = fixture()
    result, moved, vertices, colors, faces, records = correct_structure(*args)
    assert moved.all()
    np.testing.assert_allclose(result[:, 2], 0, atol=1e-12)
    assert records[0]['filled_cells'] == 1
    assert len(faces) == 2
    np.testing.assert_allclose(colors, np.tile([.2, .6, .8], (4, 1)))
    assert np.max(np.linalg.norm(result-args[0], axis=1)) <= .008


def test_furniture_and_nonparallel_surface_remain_unchanged():
    points, colors, normals, *rest = fixture()
    furniture = np.array([[.2, .3, .1], [.3, .2, .003]])
    points = np.vstack([points, furniture])
    colors = np.vstack([colors, [[1, 0, 0], [1, 0, 0]]])
    normals = np.vstack([normals, [[0, 0, 1], [1, 0, 0]]])
    original = points.copy()
    result, moved, *_ = correct_structure(points, colors, normals, *rest)
    np.testing.assert_array_equal(result[-2:], furniture)
    np.testing.assert_array_equal(points, original)
    assert not moved[-2:].any()


def test_weak_or_sloped_structure_is_not_corrected():
    args = list(fixture())
    args[4]['enabled'] = False
    result, moved, vertices, *_ = correct_structure(*args)
    np.testing.assert_array_equal(result, args[0])
    assert not moved.any() and not vertices
    args[4]['enabled'] = True
    args[4]['components'][0]['metadata']['angular_residual_degrees'] = 15
    result, moved, *_ = correct_structure(*args)
    assert not moved.any()


def test_outputs_preserve_raw_files_and_remove_only_isolated_point(tmp_path):
    import open3d as o3d
    from spatial_memory.hybrid import build_hybrid
    points, colors, normals, components, layout, diagonal = fixture()
    points = np.vstack([points, [[10, 10, 10]]])
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.colors = o3d.utility.Vector3dVector(np.vstack([colors, [[1, 0, 0]]]))
    source = tmp_path / 'raw.ply'
    shell = tmp_path / 'raw-shell.ply'
    o3d.io.write_point_cloud(str(source), cloud)
    furniture = o3d.geometry.TriangleMesh.create_box(.1, .1, .1)
    furniture.translate([.2, .2, .1])
    furniture.paint_uniform_color([.7, .4, .2])
    o3d.io.write_triangle_mesh(str(shell), furniture)
    original = (source.read_bytes(), shell.read_bytes())
    stats = build_hybrid(source, shell, components, layout, diagonal, tmp_path)
    assert stats['isolated_points_removed'] == 1
    assert stats['output_points'] == len(points)-1
    assert original == (source.read_bytes(), shell.read_bytes())
    result = o3d.io.read_triangle_mesh(stats['shell_path'])
    np.testing.assert_allclose(np.asarray(result.vertices)[:8], np.asarray(furniture.vertices))
