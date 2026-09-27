import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location('scene_pipeline', Path(__file__).resolve().parents[1]/'scripts/run_scene_appearance.py')
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def make_scene(root, count=17):
    for directory in ('frames', 'reconstruction', 'spatial_model'):
        (root/directory).mkdir(parents=True)
    frames = []
    for i in range(count):
        name = f'{i:06d}.jpg'
        (root/'frames'/name).write_bytes(b'preflight only')
        frames.append(dict(name=name))
    (root/'frames/manifest.json').write_text(json.dumps(dict(frames=frames)))
    (root/'reconstruction/metrics.json').write_text(json.dumps(dict(model='VGGT-1B')))
    (root/'reconstruction/scene.ply').touch()
    (root/'spatial_model/manifest.json').write_text('{}')
    np.savez(root/'reconstruction/reconstruction.npz', depth=np.ones((count,2,4)), confidence=np.ones((count,2,4)),
             extrinsics=np.tile(np.column_stack((np.eye(3),np.zeros(3))), (count,1,1)),
             intrinsics=np.tile([[4,0,2],[0,4,1],[0,0,1]], (count,1,1)))
    return root


@pytest.mark.parametrize('name', ['room-one', '別の部屋 with spaces'])
def test_preflight_uses_requested_scene_without_writing(tmp_path, name):
    scene = make_scene(tmp_path/name)
    before = sorted(p.relative_to(scene) for p in scene.rglob('*'))
    info = pipeline.preflight(scene)
    assert info['scene'] == str(scene)
    assert info['held_out_views'] == 3
    assert sorted(p.relative_to(scene) for p in scene.rglob('*')) == before


def test_short_capture_rejected_before_training(tmp_path):
    with pytest.raises(ValueError, match='17 frames'):
        pipeline.preflight(make_scene(tmp_path/'short', count=2))


def test_synthetic_fallback_rejected(tmp_path):
    scene = make_scene(tmp_path/'fallback')
    (scene/'reconstruction/metrics.json').write_text('{"model":"VGGT-1B","fallback":true}')
    with pytest.raises(ValueError, match='real VGGT'):
        pipeline.preflight(scene)


def test_missing_frame_rejected(tmp_path):
    scene = make_scene(tmp_path/'missing')
    (scene/'frames/000003.jpg').unlink()
    with pytest.raises(ValueError, match='source frame'):
        pipeline.preflight(scene)
