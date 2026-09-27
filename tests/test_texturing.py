import json
import numpy as np
import open3d as o3d
import pytest
from PIL import Image
from spatial_memory.texturing import project_visible, preprocess_frame, structural_faces, build_textured_scene


def cameras():
    return np.column_stack([np.eye(3), np.zeros(3)]), np.array([[100., 0, 100], [0, 100, 100], [0, 0, 1]])


def test_projection_rejects_occluded_behind_and_outside_faces():
    e, k = cameras()
    samples = np.array([[[-.1, -.1, 2], [.1, -.1, 2], [0, .1, 2]]])
    depth = np.full((200, 200), 2.)
    uv, valid = project_visible(samples, e, k, depth)
    assert valid.tolist() == [True]
    np.testing.assert_allclose(uv[0, 0], [95, 95])
    assert not project_visible(samples, e, k, depth * .5)[1].any()
    assert not project_visible(-samples, e, k, depth)[1].any()
    assert not project_visible(samples + [10, 0, 0], e, k, depth)[1].any()
    assert not project_visible(samples, e, k, depth, np.ones_like(depth))[1].any()


def test_camera_translation_world_to_camera():
    e, k = cameras()
    e[:, 3] = [-1, 0, 0]
    uv, valid = project_visible(np.array([[[1., 0, 2]]]), e, k, np.full((200, 200), 2.))
    np.testing.assert_allclose(uv[0, 0], [100, 100])
    assert valid[0]


def test_crop_matches_portrait_center_and_rejects_wrong_dimensions(tmp_path):
    path = tmp_path / 'frame.png'
    a = np.zeros((1036, 518, 3), dtype=np.uint8)
    a[259:777] = [255, 0, 0]
    Image.fromarray(a).save(path)
    assert np.array(preprocess_frame(path, 518, 518))[0, 0].tolist() == [255, 0, 0]
    with pytest.raises(ValueError):
        preprocess_frame(path, 294, 518)


def test_higher_resolution_uses_original_detail_not_upscaled_preview(tmp_path):
    path = tmp_path / 'fine-lines.png'
    stripe = np.tile((np.arange(1036) % 2 * 255).astype(np.uint8), (588, 1))
    original = np.repeat(stripe[..., None], 3, axis=2)
    Image.fromarray(original).save(path)
    full = np.array(preprocess_frame(path, 294, 518, scale=2))
    np.testing.assert_array_equal(full, original)
    preview = np.array(preprocess_frame(path, 294, 518))
    assert full.std() > preview.std() * 10


def scene_manifest():
    return {'components': [{'id': 'wall', 'center': dict(x=0, y=0, z=2),
             'axes': np.eye(3).tolist(), 'size': dict(x=4, y=4), 'metadata': {'plane': [0, 0, 1, -2]}}],
            'metrics': {'scene_diagonal': 5}, 'structural_layout': {'enabled': True,
             'components': [{'kind': 'wall', 'metadata': {'raw_component_ids': ['wall']}}]}}


def test_structure_does_not_include_offset_furniture():
    triangle = np.array([[-.2, -.2, 2], [.2, -.2, 2], [0, .2, 2]])
    assert structural_faces(np.array([triangle, triangle + [0, 0, -.2]]), scene_manifest()).tolist() == [True, False]


def test_build_keeps_geometry_and_untextured_colors(tmp_path):
    for folder in ('frames', 'reconstruction', 'spatial_model'):
        (tmp_path / folder).mkdir()
    Image.new('RGB', (518, 294), (230, 30, 10)).save(tmp_path / 'frames/000.jpg')
    (tmp_path / 'frames/manifest.json').write_text(json.dumps({'frames': [{'name': '000.jpg', 'sharpness': 10}]}))
    (tmp_path / 'reconstruction/metrics.json').write_text(json.dumps({'model': 'VGGT-1B'}))
    e, k = cameras()
    k[:2, 2] = [259, 147]
    np.savez(tmp_path / 'reconstruction/reconstruction.npz', depth=np.full((1, 294, 518), 2.), extrinsics=e[None], intrinsics=k[None], confidence=np.full((1, 294, 518), 3.))
    vertices = np.array([[-.2, -.2, 2], [.2, -.2, 2], [0, .2, 2], [-.2, -.2, 1.8], [.2, -.2, 1.8], [0, .2, 1.8]])
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector([[0, 1, 2], [3, 4, 5]]))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.full_like(vertices, .4))
    shell = tmp_path / 'spatial_model/shell.ply'
    o3d.io.write_triangle_mesh(str(shell), mesh)
    before = shell.read_bytes()
    manifest = scene_manifest()
    manifest['shell'] = {'path': str(shell)}
    path = tmp_path / 'spatial_model/manifest.json'
    path.write_text(json.dumps(manifest))
    result = build_textured_scene(path)
    assert result['textured_triangles'] == 1
    assert shell.read_bytes() == before
    data = json.loads(open(result['geometry_path']).read())
    np.testing.assert_allclose(np.array(data['positions']).reshape(-1, 3), vertices)
    colors = np.array(data['colors']).reshape(-1, 3)
    np.testing.assert_allclose(colors[:3], 1)
    np.testing.assert_allclose(colors[3:], .4, atol=.005)
    assert np.isfinite(data['uv']).all()
