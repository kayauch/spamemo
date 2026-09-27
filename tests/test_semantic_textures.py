import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image

from spatial_memory.semantic_textures import SemanticPhotoMasks, coherent_face_texture, select_face_photos


def evidence(tmp_path, masks, version=3):
    detections = []
    for identity, frame, kind, mask in masks:
        Image.fromarray(np.asarray(mask, np.uint8)*255).save(tmp_path/f'{identity}.png')
        detections.append(dict(detection_id=identity, frame=frame, kind=kind, mask=f'{identity}.png'))
    return dict(image_furniture=dict(version=version, detections=detections))


def test_target_identity_excludes_other_storage_even_with_same_label(tmp_path):
    left = np.zeros((12, 12), bool); left[2:10, 1:6] = True
    right = np.zeros_like(left); right[2:10, 7:12] = True
    layout = evidence(tmp_path, [('left', 0, 'furniture', left), ('right', 0, 'furniture', right)])
    policy = SemanticPhotoMasks(tmp_path, layout, 12, 12)
    uv = np.array([[3, 5], [9, 5], [1, 5], [6, 5]])
    box = dict(source_detection_ids=['left'])
    assert policy.allowed(0, uv, 'object', box).tolist() == [True, False, False, False]
    assert not policy.allowed(1, uv, 'object', box).any()
    assert policy.mode('object', box) == 'target_instance_mask'


def test_architecture_requires_positive_wall_and_rejects_known_furniture_unknown(tmp_path):
    wall = np.ones((12, 12), bool)
    furniture = np.zeros_like(wall); furniture[4:7, 4:7] = True
    unknown = np.zeros_like(wall); unknown[1:4, 8:11] = True
    wall[9:] = False
    layout = evidence(tmp_path, [('wall', 0, 'wall', wall), ('box', 0, 'furniture', furniture), ('unknown', 0, 'unknown', unknown)])
    policy = SemanticPhotoMasks(tmp_path, layout, 12, 12)
    uv = np.array([[2, 6], [5, 5], [9, 2], [2, 10]])
    assert policy.allowed(0, uv, 'wall').tolist() == [True, False, False, False]
    assert not policy.allowed(1, uv, 'wall').any()
    # Floor/ceiling classification is not inferred from wall complement.
    assert policy.allowed(0, uv, 'floor').tolist() == [True, False, False, True]


def test_legacy_fallback_is_explicit_and_does_not_leak_known_other_masks(tmp_path):
    policy = SemanticPhotoMasks(tmp_path, {}, 12, 12)
    uv = np.array([[3, 5], [9, 5], [np.nan, 0]])
    assert policy.allowed(0, uv, 'object', {}).tolist() == [True, True, False]
    assert policy.mode('object', {}).startswith('legacy_')
    assert policy.mode('wall').startswith('legacy_')
    # A modern empty target ID list must not silently weaken into fallback.
    assert not policy.allowed(0, uv, 'object', dict(source_detection_ids=[])).any()
    furniture = np.zeros((12, 12), bool); furniture[:, 7:] = True
    layout = evidence(tmp_path, [('other', 0, 'furniture', furniture)], version=2)
    policy = SemanticPhotoMasks(tmp_path, layout, 12, 12)
    assert policy.allowed(0, uv, 'object', {}).tolist() == [True, False, False]


def test_missing_invalid_and_outside_masks_fail_closed_without_crashing(tmp_path):
    layout = dict(image_furniture=dict(version=3, detections=[
        dict(detection_id='broken', frame=0, kind='furniture', mask='../outside.png'),
        dict(detection_id='missing', frame=1, kind='furniture', mask='absent.png')]))
    policy = SemanticPhotoMasks(tmp_path, layout, 12, 12)
    for frame in range(2):
        assert not policy.allowed(frame, np.array([[3, 3]]), 'floor').any()
    assert len(policy.summary()['invalid_masks']) == 2
    Image.new('L', (8, 8), 255).save(tmp_path/'absent.png')
    fresh = SemanticPhotoMasks(tmp_path, layout, 12, 12)
    assert not fresh.allowed(1, np.array([[3, 3]]), 'object', dict(source_detection_ids=['missing'])).any()
    assert 'dimensions' in fresh.summary()['invalid_masks'][0]['reason']


def test_mask_cache_memory_is_bounded(tmp_path):
    full = np.ones((12, 12), bool)
    layout = evidence(tmp_path, [(f'wall{i}', i, 'wall', full) for i in range(4)])
    policy = SemanticPhotoMasks(tmp_path, layout, 12, 12, max_bytes=150)
    for frame in range(4):
        assert policy.allowed(frame, np.array([[3, 3]]), 'wall')[0]
        assert policy.cache_bytes <= 150
    assert list(policy.cache) == [3]


def test_face_primary_is_coherent_and_second_only_fills_holes():
    support = [np.array([1, 1, 1, 1, 0, 0, 0, 0], bool), np.ones(8, bool), np.ones(8, bool)]
    scores = [np.array([8., 8, 8, 8, 0, 0, 0, 0]), np.array([1., 1, 1, 1, 5, 5, 5, 5]), np.ones(8)*2]
    project = lambda frame: (support[frame], scores[frame])
    selected, info = select_face_photos(3, project)
    assert selected == [0, 1]
    assert info['additional_fill_texels'] == 4
    loaded = []
    def sample(frame, take):
        loaded.append(frame)
        return np.tile([30+frame*50]*3, (take.sum(), 1))
    texture, observed, frames, secondary, weights = coherent_face_texture(8, [100]*3, selected, project, sample)
    assert loaded == [0, 1] and observed.all()
    assert frames.tolist() == [0]*4+[1]*4
    assert texture[:, 0].tolist() == [30]*4+[80]*4
    assert (secondary == -1).all() and not weights.any()


def test_empty_sparse_support_stays_plain_and_missing_texels_have_no_provenance():
    project = lambda frame: (np.zeros(5, bool), np.ones(5))
    selected, info = select_face_photos(3, project)
    assert selected == [] and info['candidate_count'] == 0
    texture, observed, frames, _, _ = coherent_face_texture(5, [11, 22, 33], selected, project, lambda *_: None)
    assert texture.tolist() == [[11, 22, 33]]*5 and not observed.any() and (frames == -1).all()


def test_builder_masks_architecture_even_when_geometry_misses_furniture(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1]/'scripts/build_solid_room.py'
    spec = importlib.util.spec_from_file_location('semantic_builder', script)
    builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
    monkeypatch.setattr(builder, 'ROOT', tmp_path)
    scene = tmp_path/'data/scenes-v2/synthetic'
    (scene/'frames').mkdir(parents=True); (scene/'reconstruction').mkdir()
    photo = np.full((64, 64, 3), [20, 80, 120], np.uint8)
    photo[:, :28] = [240, 0, 0]; photo[:, 28:35] = [0, 240, 0]
    Image.fromarray(photo).save(scene/'frames/0.png')
    (scene/'frames/manifest.json').write_text(json.dumps(dict(frames=[dict(name='0.png', source_kind='sensor')])) )
    (scene/'reconstruction/metrics.json').write_text(json.dumps(dict(model='calibrated-rgbd')))
    es = np.array([[1, 0, 0, -1], [0, 0, -1, 1], [0, 1, 0, -1]])
    ks = np.array([[24, 0, 32], [0, 24, 32], [0, 0, 1]])
    reconstruction = scene/'reconstruction/reconstruction.npz'
    np.savez(reconstruction, depth=np.ones((1, 64, 64, 1)), confidence=np.full((1, 64, 64, 1), 2),
             extrinsics=es[None], intrinsics=ks[None])
    checksum = hashlib.sha256(reconstruction.read_bytes()).hexdigest()
    run = tmp_path/'run'; prediction = run/'prediction'; prediction.mkdir(parents=True)
    (run/'transform.json').write_text(json.dumps(dict(scene_id='synthetic', source_npz_sha256=checksum,
        vg_world_to_generation_world=np.eye(4).tolist(), generation_world_to_vg_world=np.eye(4).tolist())))
    corners = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    boxes = []
    for i, (a, b) in enumerate(zip(corners, np.roll(corners, -1, axis=0))):
        x, y = (b-a)/np.linalg.norm(b-a)
        boxes.append(dict(id=i, **{'class': 'wall'}, center=[*((a+b)/2), 1], scale=[2, 0, 2],
            rotation=[[x, -y, 0], [y, x, 0], [0, 0, 1]], evidence=dict(valid=False)))
    furniture = np.zeros((64, 64), bool); furniture[:, :28] = True
    unknown = np.zeros_like(furniture); unknown[:, 28:35] = True
    layout = evidence(prediction, [('wall', 0, 'wall', np.ones_like(furniture)), ('furniture', 0, 'furniture', furniture), ('unknown', 0, 'unknown', unknown)])
    layout.update(boxes=boxes, counts=dict(bboxes=0), vertical_anchors=dict(floor=dict(z=0), ceiling=dict(z=2)))
    (prediction/'layout.json').write_text(json.dumps(layout))
    # Preserve the frame axis on this fixture: production depth loading supports
    # normal multi-frame reconstructions; two duplicate views exercise selection.
    with np.load(reconstruction) as saved:
        data = {key: np.repeat(saved[key], 2, axis=0) for key in saved.files}
    np.savez(reconstruction, **data)
    transform = json.loads((run/'transform.json').read_text())
    transform['source_npz_sha256'] = hashlib.sha256(reconstruction.read_bytes()).hexdigest()
    (run/'transform.json').write_text(json.dumps(transform))
    (scene/'frames/manifest.json').write_text(json.dumps(dict(frames=[dict(name='0.png', source_kind='sensor')]*2)))
    report = builder.build(run, 'prediction', tmp_path/'result')
    wall = next(surface for surface in report['surfaces'] if surface['id'] == 'wall-2')
    support = np.asarray(Image.open(tmp_path/'result'/wall['support'])) > 0
    texture = np.asarray(Image.open(tmp_path/'result'/wall['texture']))
    assert .3 < wall['supported_fraction'] < .5
    assert texture[support, 0].max() < 80 and texture[support, 1].max() < 130
    assert wall['semantic_support_mode'] == 'positive_wall_mask' and wall['frames_used'] == [0]
    assert wall['projection_diagnostics']['after_box_occlusion_fraction'] > .95
    assert report['appearance_processing']['max_photos_per_face'] == 2
    assert report['appearance_processing']['method'] == 'regional_quality_photos_v3'


def test_large_surface_uses_more_views_only_for_verified_missing_pixels():
    masks = [np.array([1, 1, 0, 0, 0, 0, 0], bool),
             np.array([1, 0, 1, 1, 0, 0, 0], bool),
             np.array([0, 0, 1, 0, 1, 1, 0], bool)]
    project = lambda i: (masks[i], np.ones(7) * [8, 3, 2][i])
    limited, _ = select_face_photos(3, project)
    selected, info = select_face_photos(3, project, max_photos=6)
    assert limited == [0, 1]
    assert selected == [0, 1, 2] and info['additional_fill_texels'] == 4
    sample = lambda i, take: np.full((take.sum(), 3), 50+i)
    texture, observed, frames, _, _ = coherent_face_texture(7, [100]*3, selected, project, sample)
    assert frames.tolist() == [0, 0, 1, 1, 2, 2, -1]
    assert observed.tolist() == [True]*6+[False]
    assert texture[:, 0].tolist() == [50, 50, 51, 51, 52, 52, 100]
