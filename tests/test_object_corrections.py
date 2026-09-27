from __future__ import annotations

import asyncio
import importlib

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from spatial_memory.llm.rules import RuleBasedInterpreter
from spatial_memory.models import Scene, SpatialEntity, Vec3
from spatial_memory.navigation import route_to_entity
from spatial_memory.object_corrections import ObjectCorrectionUpdate
from spatial_memory.search_answer import entity_answer
from spatial_memory.service import SpatialMemoryService
from spatial_memory.storage import SpatialMemoryDB


def entity(id, name, **kwargs):
    return SpatialEntity(id=id, name=name, scene_id=kwargs.pop('scene_id', 'room'),
                         position=Vec3(x=1, y=0, z=2), **kwargs)


@pytest.fixture
def db(tmp_path):
    database = SpatialMemoryDB(tmp_path / 'test.db')
    database.upsert_scene(Scene(id='room', name='Room'))
    database.upsert_scene(Scene(id='other-room', name='Other'))
    database.upsert_entity(entity('cabinet', 'キャビネット 1', kind='container',
        metadata={'shelf_ui': 1, 'source_prediction': 'prediction-a', 'source_object_id': '6001',
                  'storage_photo': {'image': 'data:image/jpeg;base64,example'}, 'note': 'キャビネット 1 を購入した'}))
    database.upsert_entity(entity('cable', '充電ケーブル', parent_id='cabinet', relation_to_parent='inside',
        description='キャビネット 1・上から2段目・右',
        metadata={'shelf_item': 1, 'level': 2, 'side': '右', 'location': 'キャビネット 1・上から2段目',
                  'note': 'キャビネット 1 と同じ日に購入'}))
    database.upsert_entity(entity('box', '箱', parent_id='cabinet', relation_to_parent='inside'))
    database.upsert_entity(entity('scissors', 'はさみ', parent_id='box', relation_to_parent='inside'))
    return database


def update(db, **kwargs):
    payload = {'source_prediction': 'prediction-a', 'object_id': '6001', **kwargs}
    return db.update_object_correction('room', ObjectCorrectionUpdate(**payload))


def test_rename_updates_registered_storage_and_generated_locations(db):
    correction = update(db, name='  入口の収納  ')
    assert correction.name == '入口の収納'
    root, child = db.get_entity('cabinet'), db.get_entity('cable')
    assert root.name == '入口の収納'
    assert child.description == '入口の収納・上から2段目・右'
    assert child.metadata['location'] == '入口の収納・上から2段目'
    assert child.metadata['note'] == 'キャビネット 1 と同じ日に購入'
    assert root.metadata['note'] == 'キャビネット 1 を購入した'
    assert root.metadata['storage_photo']['image'] == 'data:image/jpeg;base64,example'
    answer = entity_answer(child, db.get_entity)
    assert '入口の収納' in answer and 'キャビネット 1' not in answer
    search = SpatialMemoryService(db, RuleBasedInterpreter())
    result = asyncio.run(search.query('room', '入口の収納'))
    # A location query can also match an item recorded inside that storage.
    assert any(hit.entity.id == 'cabinet' for hit in result.hits + result.candidates)


def test_hide_removes_root_and_nested_items_from_search_and_navigation_and_restores(db):
    search = SpatialMemoryService(db, RuleBasedInterpreter())
    assert asyncio.run(search.query('room', '充電ケーブル')).target.id == 'cable'
    update(db, hidden=True)
    assert db.list_entities('room') == []
    for identifier in ('cabinet', 'cable', 'box', 'scissors'):
        assert db.get_entity(identifier) is None
    assert asyncio.run(search.query('room', '充電ケーブル')).status == 'not_found'
    with pytest.raises(ValueError, match='Target entity not found'):
        route_to_entity(db, 'room', Vec3(x=0, y=0, z=0), 'scissors')
    update(db, hidden=False)
    assert {item.id for item in db.list_entities('room')} == {'cabinet', 'cable', 'box', 'scissors'}
    assert route_to_entity(db, 'room', Vec3(x=0, y=0, z=0), 'scissors').target_entity_id == 'scissors'
    assert db.get_entity('cabinet').metadata['storage_photo']['image'].startswith('data:image')


def test_object_correction_does_not_cross_prediction_or_scene_boundaries(db):
    for identifier, scene_id, prediction in [('other-prediction', 'room', 'prediction-b'),
                                            ('other-scene', 'other-room', 'prediction-a')]:
        db.upsert_entity(entity(identifier, 'Original', scene_id=scene_id,
            metadata={'source_prediction': prediction, 'source_object_id': '6001'}))
    db.upsert_entity(entity('missing-prediction', 'Legacy', metadata={'source_object_id': '6001'}))
    db.upsert_entity(entity('independent', '手動の棚'))
    update(db, name='訂正名', hidden=True)
    for identifier in ('other-prediction', 'other-scene', 'missing-prediction', 'independent'):
        assert db.get_entity(identifier) is not None
    assert db.get_entity('other-prediction').name == 'Original'
    assert db.list_object_corrections('room', 'prediction-b') == []
    assert db.list_object_corrections('other-room', 'prediction-a') == []


def test_partial_updates_restart_and_stale_browser_writes_preserve_correction(db):
    original = db.get_entity('cabinet')
    original_child = db.get_entity('cable')
    update(db, name='窓側の棚')
    update(db, hidden=True)
    reopened = SpatialMemoryDB(db.path)
    correction = reopened.list_object_corrections('room', 'prediction-a')[0]
    assert correction.name == '窓側の棚' and correction.hidden
    assert reopened.upsert_entity(original).name == '窓側の棚'
    assert reopened.upsert_entity(original_child).description == '窓側の棚・上から2段目・右'
    assert reopened.get_entity('cabinet') is None
    update(reopened, hidden=False)
    assert reopened.get_entity('cabinet').name == '窓側の棚'
    assert reopened.get_entity('cable').description == '窓側の棚・上から2段目・右'


def test_correction_can_precede_storage_registration_and_hide_future_children(db):
    update(db, object_id='7001', name='机', hidden=True)
    saved = db.upsert_entity(entity('later', '机候補', metadata={
        'source_prediction': 'prediction-a', 'source_object_id': 7001}))
    assert saved.name == '机'
    db.upsert_entity(entity('later-child', '鍵', parent_id='later'))
    assert db.get_entity('later') is None
    assert db.get_entity('later-child') is None
    update(db, object_id='7001', hidden=False)
    assert db.get_entity('later').name == '机'
    assert db.get_entity('later-child').name == '鍵'


def test_atomic_failure_rolls_back_correction_and_linked_rename(db, monkeypatch):
    original_write = db._write_entity
    writes = []
    def fail_second(conn, item):
        writes.append(item.id)
        if len(writes) == 2:
            raise RuntimeError('simulated write failure')
        original_write(conn, item)
    monkeypatch.setattr(db, '_write_entity', fail_second)
    with pytest.raises(RuntimeError, match='simulated write failure'):
        update(db, name='新しい名前')
    assert db.list_object_corrections('room', 'prediction-a') == []
    assert db.get_entity('cabinet').name == 'キャビネット 1'
    assert db.get_entity('cable').description == 'キャビネット 1・上から2段目・右'


def test_multiple_hidden_ancestors_and_cycles_do_not_leak_or_loop(db):
    box = db.get_entity('box')
    box.metadata.update(source_prediction='prediction-a', source_object_id='6002')
    db.upsert_entity(box)
    update(db, hidden=True)
    update(db, object_id='6002', hidden=True)
    update(db, hidden=False)
    assert db.get_entity('cabinet') is not None
    assert db.get_entity('box') is None and db.get_entity('scissors') is None
    update(db, object_id='6002', hidden=False)
    box.parent_id = 'scissors'
    db.upsert_entity(box)
    update(db, object_id='6002', hidden=True)
    assert db.get_entity('scissors') is None


def test_missing_and_deleted_scenes_reject_corrections(db):
    payload = ObjectCorrectionUpdate(source_prediction='p', object_id='1', hidden=True)
    with pytest.raises(ValueError, match='部屋が見つかりません'):
        db.update_object_correction('missing', payload)
    db.delete_scene('room')
    with pytest.raises(ValueError, match='部屋が見つかりません'):
        db.update_object_correction('room', payload)
    assert db.list_object_corrections('room', 'p') == []


@pytest.mark.parametrize('changes', [{}, {'name': ''}, {'name': '  '}, {'name': None},
                                      {'name': 'a' * 121}, {'hidden': None}, {'hidden': 'false'},
                                      {'hidden': False, 'extra': 1}])
def test_invalid_partial_update_is_rejected(changes):
    with pytest.raises(ValidationError):
        ObjectCorrectionUpdate(source_prediction='p', object_id='1', **changes)


def test_api_contract_and_scene_guards(db, tmp_path, monkeypatch):
    monkeypatch.setenv('SPATIAL_DB', str(tmp_path / 'api.db'))
    monkeypatch.setenv('SCENE_ROOT', str(tmp_path / 'scenes'))
    monkeypatch.setenv('SEMANTIC_SEARCH_ENABLED', '0')
    api = importlib.import_module('spatial_memory.api')
    monkeypatch.setattr(api, 'db', db)
    client = TestClient(api.app)
    url = '/scenes/room/object-corrections'
    assert client.get(url, params={'source_prediction': 'prediction-a'}).json() == []
    payload = {'source_prediction': 'prediction-a', 'object_id': '6001', 'name': '本棚'}
    response = client.post(url, json=payload)
    assert response.status_code == 200
    assert response.json() == {**payload, 'hidden': False}
    payload.pop('name')
    response = client.post(url, json={**payload, 'hidden': True})
    assert response.json()['name'] == '本棚'
    assert client.get(url, params={'source_prediction': 'prediction-a'}).json() == [response.json()]
    assert client.get('/scenes/room/entities').json() == []
    assert client.post(url, json={**payload, 'name': ''}).status_code == 422
    assert client.get(url).status_code == 422
    db.delete_scene('room')
    assert client.get(url, params={'source_prediction': 'prediction-a'}).status_code == 404
    assert client.post(url, json={**payload, 'hidden': False}).status_code == 404
