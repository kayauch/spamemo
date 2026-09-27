import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from spatial_memory.capture_api import capture_photo_budget
from spatial_memory.capture_inputs import CaptureInput, inherited_inputs, prepare_capture
from spatial_memory.models import Scene
from test_capture_jobs import setup
from test_capture_target import model


def parent_with_sources(tmp_path, db, photo_count):
    source_dir = tmp_path / 'parent-inputs'
    source_dir.mkdir()
    sources = []
    for index, kind in enumerate(['video'] + ['photo'] * photo_count):
        path = source_dir / f'{index}.{"mp4" if kind == "video" else "jpg"}'
        path.write_bytes(f'original {index}'.encode())
        sources.append(dict(path=str(path), kind=kind))
    manifest = source_dir / 'capture.json'
    manifest.write_text(json.dumps(dict(sources=sources)))
    scene = Scene(name='parent', capture_manifest_path=str(manifest))
    db.upsert_scene(scene)
    return scene


def uploads(count):
    return [('photos', (f'{index}.jpg', f'photo {index}'.encode(), 'image/jpeg')) for index in range(count)]


def test_64_targeted_photos_keep_matching_targets_and_raise_only_required_budget(tmp_path, monkeypatch):
    client, jobs, _, db = setup(tmp_path, monkeypatch)
    scene = Scene(name='parent')
    db.upsert_scene(scene)
    target = json.loads(model(tmp_path, scene.id))
    response = client.post('/capture-jobs', data={
        'parent_scene_id': scene.id, 'capture_targets': json.dumps([target] * 64)}, files=uploads(64))
    assert response.status_code == 200
    job = jobs.status(response.json()['id'])
    assert job['options']['max_frames'] == 64
    assert len(job['inputs']) == 64
    assert all(item['metadata']['capture_target']['key'] == 'surface:floor' for item in job['inputs'])


@pytest.mark.parametrize('parent_count,count,expected_budget', [(10, 10, 60), (10, 60, 72), (60, 58, 120)])
def test_saved_photos_reserved_and_parent_video_uses_remaining_frames(tmp_path, monkeypatch, parent_count, count, expected_budget):
    client, jobs, _, db = setup(tmp_path, monkeypatch)
    parent = parent_with_sources(tmp_path, db, parent_count)
    budget = capture_photo_budget(parent)
    assert budget['existing_photo_count'] == parent_count
    assert budget['min_parent_frames'] == parent_count + 2
    assert budget['max_additional_photos'] == min(64, 118 - parent_count)
    response = client.post('/capture-jobs', data={'parent_scene_id': parent.id}, files=uploads(count))
    assert response.status_code == 200
    assert jobs.status(response.json()['id'])['options']['max_frames'] == expected_budget


@pytest.mark.parametrize('with_parent,count', [(False, 65), (True, 59)])
def test_over_capacity_rejected_before_creating_job_or_upload_directory(tmp_path, monkeypatch, with_parent, count):
    client, jobs, _, db = setup(tmp_path, monkeypatch)
    data = {'parent_scene_id': parent_with_sources(tmp_path, db, 60).id} if with_parent else {}
    response = client.post('/capture-jobs', data=data, files=uploads(count))
    assert response.status_code == 422
    assert ('120' if with_parent else '64') in response.json()['detail']
    assert jobs.active is None
    assert not list(jobs.directory.glob('*/uploads'))
    assert not list(jobs.directory.glob('*/job.json'))


def test_prepare_retains_every_unique_still_and_samples_video_to_minimum(tmp_path, monkeypatch):
    originals = []
    for index in range(70):
        path = tmp_path / f'photo-{index}.png'
        Image.new('RGB', (16, 12), (index, 100, 200)).save(path)
        originals.append(CaptureInput(path))
    video = tmp_path / 'source.mp4'
    video.write_bytes(b'video')
    quotas = []

    def extract(path, output, fps, quota, quality_filter):
        quotas.append(quota)
        output.mkdir()
        frames = []
        for index in range(quota):
            frame = output / f'{index}.jpg'
            Image.new('RGB', (16, 12), (200, index, 0)).save(frame)
            frames.append(SimpleNamespace(path=frame, source_frame_index=index, timestamp_seconds=index,
                                          sharpness=100, motion_score=1))
        return frames

    monkeypatch.setattr('spatial_memory.capture_inputs.extract_video_frames_with_metadata', extract)
    frames, records, manifest = prepare_capture([CaptureInput(video, 'video')] + originals, tmp_path / 'child')
    assert quotas == [2]
    assert len(frames) == 72 and manifest['max_frames'] == 72
    assert sum(record['source_kind'] == 'photo' for record in records) == 70
    assert len(manifest['sources']) == 71


def test_legacy_fallback_does_not_sample_away_known_photo_frames(tmp_path):
    directory = tmp_path / 'frames'
    directory.mkdir()
    records = []
    for index in range(50):
        path = directory / f'{index:06d}.jpg'
        path.write_bytes(bytes([index]))
        records.append(dict(name=path.name, source_kind='photo' if index < 40 else 'video'))
    (directory / 'manifest.json').write_text(json.dumps(dict(frames=records)))
    scene = Scene(name='legacy', frames_dir=str(directory))
    values = inherited_inputs(scene, 12)
    assert sum(not value.metadata.get('reused_frame') for value in values) == 40
    assert sum(bool(value.metadata.get('reused_frame')) for value in values) == 6
