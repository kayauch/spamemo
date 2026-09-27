import json
from test_capture_jobs import setup
from spatial_memory.models import Scene


def prepare(tmp_path,monkeypatch):
    client,jobs,rooms,db=setup(tmp_path,monkeypatch);scene=Scene(name='test');db.upsert_scene(scene)
    folder=tmp_path/'web/generation-reviews/room-test/solid-test';folder.mkdir(parents=True)
    (folder/'room.json').write_text(json.dumps(dict(source_prediction='prediction-test')))
    rooms.status=lambda _:dict(state='complete',result_url='/static/solid-room.html?run=room-test&result=solid-test',options={'texture_quality':'standard'})
    return client,jobs,rooms,scene,folder


def test_alignment_uses_exact_existing_prediction_and_preserves_options(tmp_path,monkeypatch):
    client,jobs,rooms,scene,_=prepare(tmp_path,monkeypatch)
    r=client.post(f'/scenes/{scene.id}/photo-alignment',json=dict(run='room-test',result='solid-test'))
    assert r.status_code==200
    call=rooms.calls[0][2]
    assert call['align_photos'] and call['reuse_prediction']=='prediction-test'
    assert call['reuse_run']=='room-test' and call['options']['texture_quality']=='standard'


def test_stale_wrong_and_already_aligned_results_do_not_start_jobs(tmp_path,monkeypatch):
    client,jobs,rooms,scene,folder=prepare(tmp_path,monkeypatch)
    url=f'/scenes/{scene.id}/photo-alignment'
    assert client.post(url,json=dict(run='room-other',result='solid-test')).status_code==409
    assert client.post(url,json=dict(run='../room-test',result='solid-test')).status_code==422
    (folder/'room.json').write_text(json.dumps(dict(photo_alignment={'version':1})))
    assert client.post(url,json=dict(run='room-test',result='solid-test')).status_code==409
    jobs.active='busy'
    assert client.post(url,json=dict(run='room-test',result='solid-test')).status_code==409
    assert not rooms.calls
