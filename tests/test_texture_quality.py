import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import trimesh

from spatial_memory.texture_quality import (PhotoCache, SupportedTextureBlend, sample_photo,
    overlap_color_gain, measurement_calibration, scale_room_report, capture_quality_report)


def test_photo_cache_preserves_crop_and_bounds_memory(tmp_path):
    tall = np.zeros((1600, 1200, 3), np.uint8)
    tall[:300] = [255, 0, 0]
    tall[400:1200] = [0, 255, 0]
    Image.fromarray(tall).save(tmp_path / 'a.png')
    Image.fromarray(np.full_like(tall, 60)).save(tmp_path / 'b.png')
    cache = PhotoCache(tmp_path, [{'name': 'a.png'}, {'name': 'b.png'}], 518, 518,
                       max_bytes=4_000_000, max_scale=4)
    first, scale = cache.get(0)
    assert scale == 2 and first.shape == (1036, 1036, 3)
    assert first[518, 518].tolist() == [0, 255, 0]
    cache.get(1)
    assert cache.bytes <= cache.max_bytes and list(cache.cache) == [1]
    # Returning a high-resolution original cannot silently use mismatched cameras.
    with pytest.raises(ValueError, match='camera dimensions'):
        PhotoCache(tmp_path, [{'name': 'a.png'}], 294, 518).get(0)


def test_photo_sampling_has_pixel_center_convention_and_large_list_support():
    photo = np.zeros((4, 4, 3), np.uint8)
    photo[:2, :2] = 20; photo[:2, 2:] = 60
    photo[2:, :2] = 100; photo[2:, 2:] = 140
    assert sample_photo(photo, 2, np.array([[0, 0], [1, 1]])).tolist() == [[20]*3, [140]*3]
    assert sample_photo(photo, 2, np.zeros((40000, 2))).shape == (40000, 3)


def test_sensor_photos_keep_native_camera_domain_without_vggt_crop(tmp_path):
    photo = np.zeros((480, 640, 3), np.uint8)
    photo[:100] = [255, 0, 0]
    Image.fromarray(photo).save(tmp_path / 'sensor.png')
    cache = PhotoCache(tmp_path, [{'name': 'sensor.png', 'source_kind': 'sensor'}], 480, 640)
    loaded, scale = cache.get(0)
    assert scale == 1 and np.array_equal(loaded, photo)
    with pytest.raises(ValueError, match='aspect ratio'):
        PhotoCache(tmp_path, [{'name': 'sensor.png'}], 518, 518, preprocessing='aligned_rgbd').get(0)


def test_mixed_video_and_panorama_photos_keep_vggt_batch_padding(tmp_path):
    Image.new('RGB', (1036, 588), (30, 90, 120)).save(tmp_path / 'wide.png')
    Image.new('RGB', (1036, 1036), (60, 80, 100)).save(tmp_path / 'cube.png')
    cache = PhotoCache(tmp_path, [{'name': 'wide.png'}, {'name': 'cube.png', 'panorama_face': 'front'}], 518, 518)
    wide, scale = cache.get(0)
    cube, cube_scale = cache.get(1)
    assert scale == cube_scale == 2
    assert wide.shape == cube.shape == (1036, 1036, 3)
    assert wide[223, 500].tolist() == [255]*3
    assert wide[224, 500].tolist() == [30, 90, 120]
    assert wide[811, 500].tolist() == [30, 90, 120]
    assert wide[812, 500].tolist() == [255]*3
    assert sample_photo(wide, scale, np.array([[259, 112]])).tolist() == [[30, 90, 120]]


def test_color_gain_requires_matching_samples_and_stays_conservative():
    source = np.tile([100, 120, 140], (80, 1))
    gain = overlap_color_gain(source, source * 1.08)
    assert np.allclose(gain, 1.08)
    assert np.allclose(overlap_color_gain(source[:10], source[:10] * 1.08), 1)
    assert np.allclose(overlap_color_gain(source, source[:, ::-1] * .6), 1)
    assert (overlap_color_gain(source, source * 1.2) <= 1.1).all()


def test_seam_blends_only_agreeing_visible_samples_and_retains_provenance():
    blend = SupportedTextureBlend(8)
    red = np.tile([110, 90, 80], (8, 1))
    warm = np.tile([120, 95, 85], (8, 1))
    blend.add(2, red, np.array([2, 2, 2, 1, 1, 1, 1, 1]), np.array([1, 1, 1, 1, 1, 0, 0, 0], bool))
    blend.add(3, warm, np.array([1, 1, 1, 2, 2, 2, 2, 2]), np.array([1, 1, 1, 1, 1, 0, 0, 0], bool))
    texture, support, dominant, secondary, weight = blend.finish(2, 4, [50]*3)
    assert support.tolist() == [True]*5 + [False]*3
    assert weight[:5].max() > 0
    assert (dominant[~support] == -1).all() and (secondary[~support] == -1).all()
    assert np.all(weight[~support] == 0)
    assert (texture[0] > red[0]).all() and (texture[0] < warm[0]).all()
    disagree = SupportedTextureBlend(2)
    disagree.add(0, np.array([[0]*3, [0]*3]), np.array([2, 1]), np.ones(2, bool))
    disagree.add(1, np.array([[255]*3, [255]*3]), np.array([1, 2]), np.ones(2, bool))
    assert not disagree.finish(1, 2, [50]*3)[4].any()


def test_measurement_calibration_is_uniform_and_reports_conflicting_height():
    footprint = [[0, 0], [4, 0], [4, 3], [0, 3]]
    result = measurement_calibration(footprint, 0, 2.5, {'reference_length_m': 8, 'ceiling_height_m': 3})
    assert result['scale_factor'] == 2
    assert result['metric_scale_calibrated'] and not result['metric_scale_verified']
    assert result['needs_review'] and result['discrepancies'][0]['name'] == 'ceiling_height_m'
    assert measurement_calibration(footprint, 0, 2.5, {'ceiling_height_m': 3})['scale_factor'] == 1.2
    with pytest.raises(ValueError):
        measurement_calibration(footprint, 0, 2.5, {'reference_length_m': float('nan')})


def test_scale_export_keeps_source_evidence_and_inverse_camera_transform():
    report = dict(floor=.1, ceiling=2, footprint=[[0, 0], [2, 0], [2, 2]],
                  surfaces=[dict(origin=[0, 0, .1], u=[2, 0, 0], v=[0, 2, 0], polygon=[[0, 0], [2, 0], [2, 2]])],
                  models=[dict(parts=[dict(center=[1, 1, .5], scale=[.5, .5, 1])])],
                  navigation=dict(floor=.1, ceiling=2, footprint=[[0, 0], [2, 0], [2, 2]], spawn=[1, 1, 1.5],
                                  radius=.18, eye_height=1.5, colliders=[dict(center=[1, 1, .5], scale=[.5, .5, 1], bottom=0, top=1)]),
                  geometry_validation=dict(shell_volume=4), vertical_anchors=dict(floor=dict(z=.1)),
                  transform=dict(vg_world_to_generation_world=np.eye(4).tolist(), generation_world_to_vg_world=np.eye(4).tolist()))
    out = scale_room_report(report, 2)
    assert report['floor'] == .1 and out['floor'] == .2
    assert out['navigation']['colliders'][0]['top'] == 2
    assert out['navigation']['radius'] == .18
    assert out['geometry_validation']['shell_volume'] == 32
    forward = np.asarray(out['transform']['vg_world_to_generation_world'])
    inverse = np.asarray(out['transform']['generation_world_to_vg_world'])
    assert np.allclose(forward @ inverse, np.eye(4))
    assert out['source_transform'] == report['transform']


def test_quality_feedback_is_area_weighted_and_identifies_missing_surface():
    surfaces = [dict(id='floor', kind='floor', surface_area=9, supported_fraction=1, frames_used=[0]),
                dict(id='bed-side', kind='object', object_id=7, object_label='bed', surface_area=1,
                     supported_fraction=0, frames_used=[])]
    calibration = measurement_calibration([[0, 0], [3, 0], [3, 3]], 0, 2.5)
    quality = capture_quality_report(surfaces, calibration, [{'index': 0}])
    assert quality['summary']['photo_coverage'] == pytest.approx(.9)
    assert quality['recommendations'][0]['surface_id'] == 'bed-side'
    assert quality['recommendations'][0]['object_id'] == 7
    assert quality['summary']['needs_review']


@pytest.mark.parametrize('sensor', [False, True])
def test_builder_exports_closed_calibrated_room_and_only_synthetic_evidence(tmp_path, monkeypatch, sensor):
    script = Path(__file__).resolve().parents[1] / 'scripts/build_solid_room.py'
    spec = importlib.util.spec_from_file_location('test_build_solid_room_quality', script)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    monkeypatch.setattr(builder, 'ROOT', tmp_path)
    scene = tmp_path / 'data/scenes-v2/synthetic'
    (scene / 'frames').mkdir(parents=True)
    (scene / 'reconstruction').mkdir()
    width, height = (640, 480) if sensor else (518, 518)
    if sensor:
        (scene / 'reconstruction/metrics.json').write_text(json.dumps({'model': 'calibrated-rgbd'}))
    for i, color in enumerate((100, 108)):
        Image.new('RGB', (width, height), (color, color, color)).save(scene / f'frames/{i}.jpg')
    (scene / 'frames/manifest.json').write_text(json.dumps({'frames': [{'name': f'{i}.jpg', 'sharpness': 20} for i in range(2)]}))
    extrinsic = np.column_stack([np.eye(3), [-1, -1, -1]])
    intrinsic = np.array([[160, 0, width/2], [0, 160, height/2], [0, 0, 1]])
    reconstruction = scene / 'reconstruction/reconstruction.npz'
    np.savez(reconstruction, depth=np.ones((2, height, width), np.float32), confidence=np.full((2, height, width), 2),
             extrinsics=np.stack([extrinsic]*2), intrinsics=np.stack([intrinsic]*2))
    checksum = hashlib.sha256(reconstruction.read_bytes()).hexdigest()
    run = tmp_path / 'run'
    (run / 'prediction').mkdir(parents=True)
    transform = dict(scene_id='synthetic', source_npz_sha256=checksum,
                     vg_world_to_generation_world=np.eye(4).tolist(), generation_world_to_vg_world=np.eye(4).tolist())
    (run / 'transform.json').write_text(json.dumps(transform))
    corners = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    boxes = []
    for i, (a, b) in enumerate(zip(corners, np.roll(corners, -1, axis=0))):
        direction = (b-a) / np.linalg.norm(b-a)
        x, y = direction
        boxes.append(dict(id=i, **{'class': 'wall'}, center=[*((a+b)/2), 1], scale=[2, 0, 2],
                          rotation=[[x, -y, 0], [y, x, 0], [0, 0, 1]], evidence=dict(valid=False)))
    boxes.append(dict(id='hidden', **{'class': 'bbox'}, label='bed', hidden=True, center=[1, 1, .4],
                      scale=[1, .5, .3], rotation=np.eye(3).tolist(), evidence=dict(valid=True)))
    layout = dict(boxes=boxes, vertical_anchors=dict(floor=dict(z=0), ceiling=dict(z=2)))
    (run / 'prediction/layout.json').write_text(json.dumps(layout))
    # Keep fixture textures small; the projection and exported mesh still use
    # real camera/depth arrays and the exact production functions.
    output = tmp_path / 'result'
    factor = 1 if sensor else 2
    report = builder.build(run, 'prediction', output, {} if sensor else {'reference_length_m': 4})
    assert report['calibration']['scale_factor'] == factor
    assert report['ceiling'] == 2*factor and report['models'] == []
    eye_height = min(1.55, 2*factor*.65)
    assert report['navigation']['eye_height'] == eye_height
    assert report['navigation']['spawn'][2] == pytest.approx(eye_height)
    assert report['metric_scale_calibrated'] and not report['metric_scale_verified']
    assert report['geometry_validation']['shell_volume'] == pytest.approx(8*factor**3)
    if sensor:
        assert report['calibration']['source'] == 'sensor_calibration'
    quality = json.loads((output / 'quality.json').read_text(encoding='utf-8'))
    assert quality['metrics']['source_frame_count'] == 2
    assert any(s['kind'] == 'ceiling' and s['supported_fraction'] > .9 for s in quality['surfaces'])
    floor = next(s for s in report['surfaces'] if s['kind'] == 'floor')
    assert floor['supported_fraction'] == 0
    assert not np.asarray(Image.open(output / floor['photo_provenance']['dominant'])).any()
    assert hashlib.sha256(reconstruction.read_bytes()).hexdigest() == checksum
    loaded = trimesh.load(output / 'room.glb', force='scene')
    mesh = loaded.to_geometry()
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert mesh.volume == pytest.approx(8*factor**3)
    with pytest.raises(FileExistsError):
        builder.build(run, 'prediction', output)
    with pytest.raises(ValueError, match='overrides must be applied'):
        builder.build(run, 'prediction', tmp_path / 'unused', {'furniture_overrides': [{'id': 'hidden', 'hidden': True}]})


def test_quality_separates_depth_mismatch_from_furniture_blocking():
    from spatial_memory.texture_quality import capture_quality_report
    base = dict(kind='wall', supported_fraction=.1, surface_area=1)
    surfaces = [dict(base, id='mismatch', projection_diagnostics=dict(in_view_fraction=.9, depth_supported_fraction=.1, after_furniture_mask_fraction=.1)),
                dict(base, id='blocked', projection_diagnostics=dict(in_view_fraction=.9, depth_supported_fraction=.6, after_furniture_mask_fraction=.1))]
    report = capture_quality_report(surfaces, {'needs_review': False}, [])
    assert [r['code'] for r in report['recommendations']] == ['projection_depth_mismatch', 'furniture_mask_occlusion']
    assert report['surfaces'][0]['projection_diagnostics']['in_view_fraction'] == .9
