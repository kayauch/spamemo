import json
from uuid import uuid4
import pytest
from spatial_memory.room_jobs import RoomJobs, PIPELINE_VERSION, copy_prediction_masks


def test_rejects_invalid_ids_and_missing_inputs(tmp_path):
    jobs=RoomJobs(tmp_path,tmp_path/'scenes')
    with pytest.raises(ValueError):jobs.start('../outside')
    with pytest.raises(ValueError,match='動画'):jobs.start(str(uuid4()))
    assert jobs.active is None


def test_durable_result_and_interrupted_state(tmp_path):
    scene=str(uuid4());jobs=RoomJobs(tmp_path,tmp_path/'scenes')
    jobs.save(scene,dict(state='running',run='room-a'))
    assert jobs.status(scene)['state']=='interrupted'
    jobs.active=scene
    assert jobs.status(scene)['state']=='running'
    jobs.save(scene,dict(state='complete',result_url='/static/result',pipeline_version=PIPELINE_VERSION))
    assert jobs.start(scene)['result_url']=='/static/result'


def test_subprocess_failure_preserves_failure_and_releases_slot(tmp_path,monkeypatch):
    def fail(*args,**kwargs):raise RuntimeError('expected test failure')
    monkeypatch.setattr('spatial_memory.room_jobs.subprocess.run',fail)
    scene=str(uuid4());jobs=RoomJobs(tmp_path,tmp_path/'scenes');jobs.active=scene
    jobs.work(scene,dict(state='running',scene=scene,run='room-test'))
    result=jobs.status(scene)
    assert result['state']=='failed' and 'expected test failure' in result['error']
    assert jobs.active is None


def test_duplicate_start_does_not_spawn_and_other_scene_is_blocked(tmp_path,monkeypatch):
    jobs=RoomJobs(tmp_path,tmp_path/'scenes');scene=str(uuid4());jobs.active=scene
    jobs.save(scene,dict(state='running',run='room-test'))
    assert jobs.start(scene)['run']=='room-test'
    with pytest.raises(RuntimeError,match='別の部屋'):jobs.start(str(uuid4()))


def test_resume_reads_japanese_layout_and_skips_gpu(tmp_path,monkeypatch):
    calls=[]
    def mock_run(command,**kwargs):
        calls.append(command)
        if 'infer_image_furniture.py' in command[1]:
            output=command[-1];folder=run/output;folder.mkdir()
            (folder/'layout.json').write_text('{}')
        if 'build_solid_room.py' in command[1]:
            folder=tmp_path/'web/generation-reviews/room-resume'/command[-1];folder.mkdir(parents=True)
            (folder/'room.json').write_text(json.dumps(dict(geometry_validation=dict(shell_watertight=True),navigation=dict(floor=0))))
            (folder/'room.glb').write_bytes(b'glTF')
    monkeypatch.setattr('spatial_memory.room_jobs.subprocess.run',mock_run)
    run=tmp_path/'data/generation-runs/room-resume'
    for name in ('prediction-001','prediction-auto-001'):
        (run/name).mkdir(parents=True)
        (run/name/'layout.json').write_text(json.dumps({'display_label':'自動推定（実写照合前）'},ensure_ascii=False),encoding='utf-8')
    (run/'transform.json').write_text('{}',encoding='utf-8')
    scene=str(uuid4());jobs=RoomJobs(tmp_path,tmp_path/'scenes');jobs.active=scene
    jobs.work(scene,dict(state='running',scene=scene,run='room-resume'))
    assert jobs.status(scene)['state']=='complete'
    assert len(calls)==3
    assert 'infer_image_furniture.py' in calls[0][1]
    assert all('run_spatiallm.py' not in ' '.join(c) for c in calls)


def test_force_preserves_previous_result_and_history(tmp_path,monkeypatch):
    scene=str(uuid4());source=tmp_path/'scenes'/scene
    for relative in ('reconstruction/reconstruction.npz','reconstruction/metrics.json','spatial_model/manifest.json','frames/manifest.json'):
        file=source/relative;file.parent.mkdir(parents=True,exist_ok=True)
        file.write_text(json.dumps({'model':'VGGT-1B'}))
    class NoThread:
        def __init__(self,*a,**k):pass
        def start(self):pass
    monkeypatch.setattr('spatial_memory.room_jobs.threading.Thread',NoThread)
    jobs=RoomJobs(tmp_path,tmp_path/'scenes');old=dict(state='complete',result_url='/old')
    jobs.save(scene,old)
    new=jobs.start(scene,force=True)
    assert new['previous_result_url']=='/old' and new['pipeline_version']==PIPELINE_VERSION
    assert json.loads(next((jobs.directory/'history').glob('*.json')).read_text())==old


def test_missing_model_export_cannot_report_success(tmp_path,monkeypatch):
    run=tmp_path/'data/generation-runs/room-test'
    for name in ('prediction-001','prediction-auto-001','prediction-image-test'):
        (run/name).mkdir(parents=True)
        (run/name/'layout.json').write_text('{}')
    (run/'transform.json').write_text('{}')
    monkeypatch.setattr('spatial_memory.room_jobs.subprocess.run',lambda *a,**k:None)
    jobs=RoomJobs(tmp_path,tmp_path/'scenes');scene=str(uuid4());jobs.active=scene
    jobs.work(scene,dict(state='running',scene=scene,run='room-test',image_prediction='prediction-image-test',previous_result_url='/old'))
    result=jobs.status(scene)
    assert result['state']=='failed' and result['previous_result_url']=='/old'
    assert jobs.active is None


def test_review_keeps_identity_masks_and_rejects_escape(tmp_path):
    source = tmp_path/'original'; source.mkdir()
    (source/'observations').mkdir()
    (source/'observations/a.png').write_bytes(b'saved-mask')
    target = tmp_path/'reviewed'; target.mkdir()
    layout = {'image_furniture': {'version': 3, 'detections': [
        {'detection_id': 'a', 'mask': 'observations/a.png'}]}}
    copy_prediction_masks(source, target, layout)
    assert (target/'observations/a.png').read_bytes() == b'saved-mask'
    layout['image_furniture']['detections'][0]['mask'] = '../outside.png'
    with pytest.raises(ValueError, match='escapes'):
        copy_prediction_masks(source, target, layout)
    layout['image_furniture']['detections'][0]['mask'] = 'missing.png'
    with pytest.raises(ValueError, match='missing'):
        copy_prediction_masks(source, target, layout)


def test_regeneration_keeps_preferences_but_not_old_object_overrides(tmp_path, monkeypatch):
    scene = str(uuid4()); source = tmp_path/'scenes'/scene
    for relative in ('reconstruction/reconstruction.npz', 'reconstruction/metrics.json',
                     'spatial_model/manifest.json', 'frames/manifest.json'):
        path = source/relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'model': 'VGGT-1B'}))
    class NoThread:
        def __init__(self, *args, **kwargs): pass
        def start(self): pass
    monkeypatch.setattr('spatial_memory.room_jobs.threading.Thread', NoThread)
    jobs = RoomJobs(tmp_path, tmp_path/'scenes')
    jobs.save(scene, dict(state='complete', pipeline_version=4, result_url='/old', options={
        'ceiling_height_m': 2.6, 'furniture_frame_budget': 32,
        'furniture_overrides': [{'id': '42', 'hidden': True}]}))
    result = jobs.start(scene, force=True)
    assert result['options']['ceiling_height_m'] == 2.6
    assert result['options']['furniture_frame_budget'] == 32
    assert result['options']['furniture_overrides'] == []
    assert result['previous_result_url'] == '/old'
