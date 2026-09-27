import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from spatial_memory.capture_api import build_capture_router, capture_quality
from spatial_memory.capture_jobs import CaptureJobs
from spatial_memory.capture_options import CaptureOptions
from spatial_memory.models import Scene
from spatial_memory.storage import SpatialMemoryDB


class RoomStub:
    active = None
    calls = []

    def start(self, scene, up_axis='image-top', **kwargs):
        self.calls.append((scene, up_axis, kwargs))
        return dict(state='complete', message='complete', result_url='/static/solid-room.html?run=r&result=s')

    def status(self, scene):
        return dict(state='idle')


def setup(tmp_path, monkeypatch):
    class NoThread:
        def __init__(self, **kwargs): pass
        def start(self): pass
    monkeypatch.setattr('spatial_memory.capture_jobs.threading.Thread', NoThread)
    db = SpatialMemoryDB(tmp_path/'test.db')
    service = SimpleNamespace(db=db)
    rooms = RoomStub()
    rooms.calls = []
    jobs = CaptureJobs(tmp_path, service, rooms)
    app = FastAPI()
    app.include_router(build_capture_router(tmp_path, db, jobs, rooms))
    return TestClient(app), jobs, rooms, db


def test_upload_durable_id_safe_filenames_and_no_path_leak(tmp_path, monkeypatch):
    client, jobs, _, _ = setup(tmp_path, monkeypatch)
    response = client.post('/capture-jobs', data={'name': '追加写真'},
                           files=[('photos', ('../../photo.jpg', b'image-data', 'image/jpeg'))])
    assert response.status_code == 200
    data = response.json()
    assert 'inputs' not in data
    saved = jobs.status(data['id'])
    path = Path(saved['inputs'][0]['path'])
    assert path.is_relative_to(jobs.directory)
    assert path.read_bytes() == b'image-data'
    assert client.get(f'/capture-jobs/{data["id"]}').json()['state'] == 'queued'
    jobs.active = None
    assert client.get(f'/capture-jobs/{data["id"]}').json()['state'] == 'interrupted'
    assert client.post(f'/capture-jobs/{data["id"]}/resume').status_code == 200


def test_capture_validates_limits_and_missing_parent(tmp_path, monkeypatch):
    client, jobs, _, _ = setup(tmp_path, monkeypatch)
    assert client.post('/capture-jobs').status_code == 422
    assert client.post('/capture-jobs', data={'options':'{"max_frames":999}'}, files={'videos':('x.mp4',b'x')}).status_code == 422
    assert client.post('/capture-jobs', data={'parent_scene_id':str(uuid4())}, files={'photos':('x.jpg',b'x')}).status_code == 422
    assert client.post('/capture-jobs', files=[('depth_bundle',('sensor.zip',b'x')),('photos',('x.jpg',b'x'))]).status_code == 422
    assert client.post('/capture-jobs', files={'photos':('x.exe',b'x')}).status_code == 422
    jobs.active = 'occupied'
    assert client.post('/capture-jobs', files={'videos':('x.mp4',b'x')}).status_code == 409


def test_worker_passes_options_and_keeps_completed_scene_on_room_failure(tmp_path, monkeypatch):
    _, jobs, rooms, db = setup(tmp_path, monkeypatch)
    scene = Scene(name='new', id=str(uuid4()))
    inputs = [{'kind':'photo','path':str(tmp_path/'new.jpg')}]
    seen = []
    def reconstruct(values, name, **kwargs):
        seen.append(kwargs)
        db.upsert_scene(scene)
        return scene, {'test':True}
    jobs.reconstruction.reconstruct_capture = reconstruct
    job = jobs.start(inputs, 'new', options={'ceiling_height_m':2.4})
    jobs.work(job)
    assert jobs.status(job['id'])['state'] == 'complete'
    assert seen[0]['capture_metadata']['ceiling_height_m'] == 2.4
    assert rooms.calls[0][2]['options']['ceiling_height_m'] == 2.4
    assert jobs.active is None
    # A resumed failed model stage reuses the successful geometry.
    job['state'] = 'failed'
    jobs.active = job['id']
    jobs.work(job)
    assert len(seen) == 1


def test_old_scene_quality_no_regeneration(tmp_path, monkeypatch):
    client, jobs, rooms, db = setup(tmp_path, monkeypatch)
    scene = Scene(name='existing', frames_dir=str(tmp_path/'frames'))
    db.upsert_scene(scene)
    folder=Path(scene.frames_dir);folder.mkdir()
    (folder/'manifest.json').write_text(json.dumps({'frames':[{'name':'a.jpg','timestamp_seconds':12.}]}))
    report=client.get(f'/scenes/{scene.id}/capture-quality')
    assert report.status_code==200 and report.json()['frame_count']==1
    assert rooms.calls==[] and jobs.active is None


@pytest.mark.parametrize('options', [
    {'reference_length_m':float('nan')}, {'ceiling_height_m':0}, {'unknown_option':True},
    {'furniture_overrides':[{'id':'../outside','hidden':True}]},
    {'furniture_overrides':[{'id':'6','label':'fantasy' }]},
    {'furniture_overrides':[{'id':'6','hidden':True},{'id':'6','hidden':False}]},
    {'furniture_overrides':[{'id':'6','dimensions_m':[1,-1,1]}]},
])
def test_capture_options_reject_invalid_constraints(options):
    with pytest.raises(ValueError): CaptureOptions.model_validate(options)


def test_options_omit_null_overrides_for_furniture_helper():
    result=CaptureOptions.model_validate({'furniture_overrides':[{'id':'6001','hidden':True}]}).model_dump(exclude_none=True)
    assert result['furniture_overrides']==[{'id':'6001','hidden':True}]


def test_sensor_worker_honors_user_frame_budget(tmp_path,monkeypatch):
    _,jobs,_,db=setup(tmp_path,monkeypatch)
    seen=[];scene=Scene(name='sensor')
    def sensor(path,name,**kwargs):
        seen.append(kwargs);db.upsert_scene(scene);return scene,{}
    jobs.reconstruction.reconstruct_sensor_capture=sensor
    job=jobs.start([dict(kind='sensor',path=str(tmp_path/'sensor.zip'))],'sensor',options={'max_frames':12})
    jobs.work(job)
    assert seen[0]['max_frames']==12 and jobs.status(job['id'])['state']=='complete'


def test_capture_room_and_legacy_share_admission_lock(tmp_path,monkeypatch):
    from spatial_memory.room_jobs import RoomJobs
    _,_,_,db=setup(tmp_path,monkeypatch)
    rooms=RoomJobs(tmp_path,tmp_path/'scenes')
    jobs=CaptureJobs(tmp_path,SimpleNamespace(db=db),rooms)
    assert jobs.lock is rooms.lock
    with jobs.legacy_reconstruction():
        with pytest.raises(RuntimeError):rooms.start(str(uuid4()))
        with pytest.raises(RuntimeError):jobs.start([], 'other')
    assert jobs.active is None
    with pytest.raises(ValueError):
        with jobs.legacy_reconstruction():raise ValueError('simulate failure')
    assert jobs.active is None


def test_failed_furniture_review_can_return_to_last_successful_model(tmp_path,monkeypatch):
    client,_,rooms,db=setup(tmp_path,monkeypatch)
    scene=Scene(name='room');db.upsert_scene(scene)
    output=tmp_path/'web/generation-reviews/room-success/solid-good';output.mkdir(parents=True)
    (output/'room.json').write_text(json.dumps({'source_prediction':'prediction-good','transform':{'camera_up_axis':'image-left'}}))
    layout=tmp_path/'data/generation-runs/room-success/prediction-good';layout.mkdir(parents=True)
    (layout/'layout.json').write_text(json.dumps({'boxes':[{'id':6001,'class':'bbox','label':'bed','scale':[1,2,.5]}]}))
    failed=dict(state='failed',up_axis='image-left',run='room-success',image_prediction='prediction-failed',
                previous_result_url='/static/solid-room.html?run=room-success&result=solid-good')
    rooms.status=lambda scene_id:failed
    report=client.get(f'/scenes/{scene.id}/capture-quality').json()
    assert report['options']['up_axis']=='image-left' and report['objects'][0]['id']==6001
    response=client.post(f'/scenes/{scene.id}/capture-review',json={'up_axis':'image-left','furniture_overrides':[{'id':'6001','hidden':True}]})
    assert response.status_code==200
    assert rooms.calls[-1][2]['reuse_run']=='room-success'
    assert rooms.calls[-1][2]['reuse_prediction']=='prediction-good'


def test_job_save_retries_windows_replace_contention(tmp_path, monkeypatch):
    _, jobs, _, _ = setup(tmp_path, monkeypatch)
    original = Path.replace
    attempts = []
    def replace(path, target):
        attempts.append(path)
        if len(attempts) < 3:
            raise PermissionError('sharing violation')
        return original(path, target)
    monkeypatch.setattr(Path, 'replace', replace)
    monkeypatch.setattr('spatial_memory.capture_jobs.time.sleep', lambda _: None)
    job = {'id':str(uuid4()), 'state':'complete'}
    jobs.save(job)
    assert jobs.status(job['id'])['state'] == 'complete'
    assert len(attempts) == 3
    assert not list(jobs.folder(job['id']).glob('*.tmp'))


def test_completed_room_repairs_stale_capture_without_restart(tmp_path, monkeypatch):
    _, jobs, rooms, _ = setup(tmp_path, monkeypatch)
    job = {'id':str(uuid4()), 'state':'running', 'stage':'room', 'scene_id':str(uuid4()), 'message':'2/6'}
    jobs.save(job)
    monkeypatch.setattr(rooms, 'status', lambda _:dict(state='complete',message='done',result_url='/static/solid-room.html?run=r&result=s'))
    recovered=jobs.status(job['id'])
    assert recovered['state']=='complete' and recovered['message']=='done'
    assert json.loads((jobs.folder(job['id'])/'job.json').read_text())['state']=='complete'
    assert rooms.calls == []


def test_final_save_failure_always_releases_active_job(tmp_path, monkeypatch):
    _, jobs, _, _ = setup(tmp_path, monkeypatch)
    job={'id':str(uuid4()),'state':'running'}
    jobs.active=job['id']
    def fail(_):raise PermissionError('cannot save')
    monkeypatch.setattr(jobs,'save',fail)
    with pytest.raises(PermissionError):jobs.work(job)
    assert jobs.active is None


def test_room_type_round_trips_with_capture_options(tmp_path, monkeypatch):
    client, jobs, _, _ = setup(tmp_path, monkeypatch)
    response = client.post('/capture-jobs', data={'name': 'UI room', 'options': json.dumps({'room_type': 'research'})}, files={'videos': ('test.mp4', b'test-video', 'video/mp4')})
    assert response.status_code == 200
    assert jobs.status(response.json()['id'])['options']['room_type'] == 'research'
    assert CaptureOptions().room_type is None
    with pytest.raises(ValueError):
        CaptureOptions(room_type='unsupported')


def test_deleted_parent_rejects_before_storage_and_same_video_can_start_new_room(tmp_path, monkeypatch):
    client, jobs, _, _ = setup(tmp_path, monkeypatch)
    video = b'recorded-video-retained-in-browser'
    response = client.post('/capture-jobs', data={'name': 'new recording', 'parent_scene_id': str(uuid4())}, files={'videos': ('recording.webm', video, 'video/webm')})
    assert response.status_code == 422
    assert response.json()['detail'] == '追加先の部屋が見つかりません'
    assert not list(jobs.directory.iterdir())
    retry = client.post('/capture-jobs', data={'name': 'new recording'}, files={'videos': ('recording.webm', video, 'video/webm')})
    assert retry.status_code == 200
    saved = jobs.status(retry.json()['id'])
    assert not saved.get('parent_scene_id')
    assert Path(saved['inputs'][0]['path']).read_bytes() == video
