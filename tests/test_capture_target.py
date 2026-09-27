import json
from pathlib import Path
import pytest
from spatial_memory.capture_target import resolve_capture_target
from test_capture_jobs import setup
from spatial_memory.models import Scene

def model(root,scene):
    folder=root/'web/generation-reviews/run/result';folder.mkdir(parents=True)
    (folder/'room.json').write_text(json.dumps({'transform':{'scene_id':scene},'surfaces':[{'id':'floor','kind':'floor'},{'id':'wall-1','kind':'wall'},{'id':'chair-face','kind':'object','object_id':7}],'furniture':[{'id':7},{'id':8,'hidden':True}]}))
    return json.dumps({'run':'run','result':'result','key':'surface:floor'})

def test_target_upload_persists_verified_source_on_every_photo(tmp_path,monkeypatch):
    client,jobs,_,db=setup(tmp_path,monkeypatch);scene=Scene(name='parent');db.upsert_scene(scene)
    target=model(tmp_path,scene.id)
    response=client.post('/capture-jobs',data={'parent_scene_id':scene.id,'capture_target':target},files=[('photos',('one.jpg',b'one')),('photos',('two.jpg',b'two'))])
    assert response.status_code==200
    saved=jobs.status(response.json()['id'])
    for item in saved['inputs']:
        context=item['metadata']['capture_target']
        assert context['scene_id']==scene.id and context['surface_ids']==['floor']
        assert context['status']=='user_selected_not_verified'
    assert 'inputs' not in response.json()

@pytest.mark.parametrize('change',[{'run':'../run'},{'key':'surface:missing'},{'key':'object:8'},{'result':'other'}])
def test_rejects_unknown_hidden_or_unsafe_targets(tmp_path,change):
    raw=json.loads(model(tmp_path,'scene'));raw.update(change)
    with pytest.raises(ValueError):resolve_capture_target(tmp_path,json.dumps(raw),'scene')

def test_rejects_cross_room_and_requires_parent(tmp_path):
    raw=model(tmp_path,'scene')
    for parent in [None,'other']:
        with pytest.raises(ValueError):resolve_capture_target(tmp_path,raw,parent)
    assert resolve_capture_target(tmp_path,None,'scene') is None

def test_furniture_target_collects_source_faces(tmp_path):
    raw=json.loads(model(tmp_path,'scene'));raw['key']='object:7'
    context=resolve_capture_target(tmp_path,json.dumps(raw),'scene')
    assert context['kind']=='object' and context['surface_ids']==['chair-face']

def test_invalid_target_is_rejected_before_upload_is_written(tmp_path,monkeypatch):
    client,jobs,_,db=setup(tmp_path,monkeypatch);scene=Scene(name='parent');db.upsert_scene(scene)
    raw=json.loads(model(tmp_path,scene.id));raw['key']='surface:missing'
    response=client.post('/capture-jobs',data={'parent_scene_id':scene.id,'capture_target':json.dumps(raw)},files={'photos':('one.jpg',b'one')})
    assert response.status_code==422 and not list(jobs.directory.glob('*/uploads'))
