"""Room deletion hides records without removing files shared by revisions."""
from __future__ import annotations

import importlib
import sqlite3

from fastapi.testclient import TestClient
import pytest

from spatial_memory.models import NavEdge, NavNode, Observation, Scene, SpatialEntity, Vec3
from spatial_memory.storage import SpatialMemoryDB


def populate(db, scene):
    db.upsert_scene(scene)
    point = Vec3(x=1, y=0, z=2)
    db.upsert_entity(SpatialEntity(id=f'{scene.id}-item', scene_id=scene.id,
                                  name='充電ケーブル', position=point))
    db.add_observation(Observation(id=f'{scene.id}-observation', scene_id=scene.id,
                                   position=point))
    db.upsert_nav_node(NavNode(id=f'{scene.id}-node', scene_id=scene.id,
                               name='入口', position=point))
    db.upsert_nav_edge(NavEdge(scene_id=scene.id, source_id=f'{scene.id}-node',
                               target_id=f'{scene.id}-destination'))


def test_delete_hides_room_and_all_related_readers(tmp_path):
    db = SpatialMemoryDB(tmp_path / 'test.db')
    populate(db, Scene(id='room', name='応理ラウンジ'))
    assert db.delete_scene('room')
    assert db.get_scene('room') is None
    assert db.list_scenes() == []
    assert db.get_entity('room-item') is None
    assert db.list_entities('room') == []
    assert db.list_observations('room') == []
    assert db.list_nav_nodes('room') == []
    assert db.list_nav_edges('room') == []


def test_delete_keeps_other_revision_and_shared_files(tmp_path):
    db = SpatialMemoryDB(tmp_path / 'test.db')
    photo = tmp_path / 'shared-photo.jpg'
    video = tmp_path / 'shared-video.webm'
    photo.write_bytes(b'photo')
    video.write_bytes(b'video')
    parent = Scene(id='parent', name='old', source_video_path=str(video),
                   frames_dir=str(photo.parent))
    child = parent.model_copy(update={'id': 'child', 'name': 'new', 'parent_scene_id': 'parent'})
    populate(db, parent)
    populate(db, child)
    assert db.delete_scene('parent')
    assert [room.id for room in db.list_scenes()] == ['child']
    assert db.get_scene('child') == child
    assert db.get_entity('child-item') is not None
    assert len(db.list_observations('child')) == 1
    assert len(db.list_nav_nodes('child')) == 1
    assert len(db.list_nav_edges('child')) == 1
    assert photo.read_bytes() == b'photo'
    assert video.read_bytes() == b'video'


def test_delete_survives_restart_and_late_generation_upsert(tmp_path):
    path = tmp_path / 'test.db'
    db = SpatialMemoryDB(path)
    scene = Scene(id='room', name='original')
    db.upsert_scene(scene)
    stale_worker = SpatialMemoryDB(path)
    assert db.delete_scene(scene.id)
    stale_worker.upsert_scene(scene.model_copy(update={'name': 'late completion'}))
    reopened = SpatialMemoryDB(path)
    assert reopened.is_scene_deleted(scene.id)
    assert reopened.get_scene(scene.id) is None
    assert reopened.list_scenes() == []
    with reopened.connect() as conn:
        stored = Scene.model_validate_json(conn.execute('SELECT payload FROM scenes WHERE id=?',
                                                       (scene.id,)).fetchone()['payload'])
    assert stored.name == 'original'


def test_repeated_delete_and_unknown_room(tmp_path):
    db = SpatialMemoryDB(tmp_path / 'test.db')
    db.upsert_scene(Scene(id='room', name='room'))
    assert db.delete_scene('room')
    assert db.delete_scene('room')
    assert not db.delete_scene('missing')
    assert not db.is_scene_deleted('missing')
    db.upsert_scene(Scene(id='missing', name='new room'))
    assert db.get_scene('missing') is not None


def test_existing_database_migrates_without_losing_rooms(tmp_path):
    path = tmp_path / 'test.db'
    room = Scene(id='room', name='existing')
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE scenes (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        conn.execute('INSERT INTO scenes VALUES (?, ?)', (room.id, room.model_dump_json()))
    db = SpatialMemoryDB(path)
    assert db.get_scene('room') == room
    assert db.delete_scene('room')
    assert db.list_scenes() == []


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Importing the app must not initialize or alter the user's real database.
    monkeypatch.setenv('SPATIAL_DB', str(tmp_path / 'api.db'))
    monkeypatch.setenv('SCENE_ROOT', str(tmp_path / 'scenes'))
    monkeypatch.setenv('SEMANTIC_SEARCH_ENABLED', '0')
    api = importlib.import_module('spatial_memory.api')
    monkeypatch.setattr(api, 'db', SpatialMemoryDB(tmp_path / 'api.db'))
    return TestClient(api.app)


def test_api_delete_is_idempotent_and_rejects_recreating_deleted_id(client):
    assert client.post('/scenes', json={'id': 'room', 'name': 'room'}).status_code == 200
    assert client.post('/scenes', json={'id': 'other', 'name': 'other'}).status_code == 200
    first = client.delete('/scenes/room')
    assert first.status_code == 200
    assert first.json() == {'deleted': 'room'}
    assert client.delete('/scenes/room').json() == first.json()
    assert client.get('/scenes/room').status_code == 404
    assert [scene['id'] for scene in client.get('/scenes').json()] == ['other']
    assert client.get('/scenes/room/entities').json() == []
    assert client.get('/scenes/room/navigation').json() == {'nodes': [], 'edges': []}
    assert client.post('/query', json={'scene_id': 'room', 'text': 'ケーブル'}).status_code == 404
    assert client.post('/scenes', json={'id': 'room', 'name': 'resurrect'}).status_code == 409


def test_api_delete_unknown_room_returns_404(client):
    assert client.delete('/scenes/missing').status_code == 404
    assert client.get('/scenes').json() == []
