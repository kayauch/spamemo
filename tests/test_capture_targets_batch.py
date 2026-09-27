import json
import pytest
from test_capture_jobs import setup
from test_capture_target import model
from spatial_memory.models import Scene

def test_each_photo_retains_its_own_target(tmp_path,monkeypatch):
    client,jobs,_,db=setup(tmp_path,monkeypatch);scene=Scene(name='parent');db.upsert_scene(scene)
    first=json.loads(model(tmp_path,scene.id));second={**first,'key':'object:7'}
    response=client.post('/capture-jobs',data={'parent_scene_id':scene.id,'capture_targets':json.dumps([first,second])},files=[('photos',('same.jpg',b'floor')),('photos',('same.jpg',b'chair'))])
    assert response.status_code==200
    inputs=jobs.status(response.json()['id'])['inputs']
    assert [i['metadata']['capture_target']['key'] for i in inputs]==['surface:floor','object:7']

@pytest.mark.parametrize('mode',['too_few','too_many','null','null_item','unknown','both','no_parent'])
def test_batch_rejects_mismatched_or_invalid_targets_before_saving(tmp_path,monkeypatch,mode):
    client,jobs,_,db=setup(tmp_path,monkeypatch);scene=Scene(name='parent');db.upsert_scene(scene)
    target=json.loads(model(tmp_path,scene.id));values=[target,target]
    if mode=='too_few':values=[target]
    if mode=='too_many':values=[target]*3
    if mode=='null':values=None
    if mode=='null_item':values=[target,None]
    if mode=='unknown':values=[target,{**target,'key':'surface:unknown'}]
    data={'parent_scene_id':scene.id,'capture_targets':json.dumps(values)}
    if mode=='both':data['capture_target']=json.dumps(target)
    if mode=='no_parent':data.pop('parent_scene_id')
    response=client.post('/capture-jobs',data=data,files=[('photos',('a.jpg',b'a')),('photos',('b.jpg',b'b'))])
    assert response.status_code==422 and not list(jobs.directory.glob('*/uploads'))
