import asyncio
import pytest
import torch
from spatial_memory.models import Scene, SpatialEntity, Vec3, SearchHit
from spatial_memory.storage import SpatialMemoryDB
from spatial_memory.service import SpatialMemoryService
from spatial_memory.llm.rules import RuleBasedInterpreter
from spatial_memory.search import search_entities, normalize
from spatial_memory.semantic import SemanticSearchEngine
from spatial_memory.search_answer import search_answer, entity_answer


def entity(name, **kwargs):
    return SpatialEntity(name=name, scene_id='room', position=Vec3(x=0,y=0,z=0), **kwargs)


def service(tmp_path, entities, semantic=None):
    db=SpatialMemoryDB(tmp_path/'search.db')
    db.upsert_scene(Scene(id='room',name='Room'))
    for e in entities: db.upsert_entity(e)
    return SpatialMemoryService(db,RuleBasedInterpreter(),semantic=semantic)


class Scores:
    def __init__(self, scores): self.scores=scores
    def search(self, query, entities, limit=5):
        return sorted([SearchHit(entity=e,score=self.scores.get(e.id,0),reasons=['semantic']) for e in entities],key=lambda h:-h.score)[:limit]


def test_duplicate_names_require_choice_even_limit_one(tmp_path):
    s=service(tmp_path,[entity('充電ケーブル',id='one'),entity('充電ケーブル',id='two')])
    r=asyncio.run(s.query('room','充電ケーブルどこ？',limit=1))
    assert r.status=='ambiguous' and r.target is None and r.hits==[]
    assert len(r.candidates)==2
    assert '判断できません' in search_answer(r,s.db.get_entity)


@pytest.mark.parametrize('scores,status', [({'one':.79,'two':.2},'not_found'),({'one':.82,'two':.79},'ambiguous'),({'one':.86,'two':.7},'matched')])
def test_raw_semantic_gate_before_cutoff_and_limit(tmp_path,scores,status):
    s=service(tmp_path,[entity('USBコード',id='one'),entity('眼鏡',id='two')],Scores(scores))
    r=asyncio.run(s.query('room','スマホを充電するやつ',limit=1))
    assert r.status==status
    assert (r.target is not None)==(status=='matched')


def test_wall_photos_and_binary_metadata_are_not_searchable(tmp_path):
    wall=entity('充電ケーブル',id='wall',metadata={'wall_photo':{'image':'data:image/png;base64,abc'}})
    item=entity('はさみ',id='item',metadata={'shelf_photo':'data:image/jpeg;base64,充電ケーブル','note':'紙を切る道具'})
    s=service(tmp_path,[wall,item])
    assert asyncio.run(s.query('room','充電ケーブル')).status=='not_found'
    assert search_entities('紙を切る道具',[item])[0].entity.id=='item'
    assert 'data:image' not in SemanticSearchEngine.entity_text(item)


def test_answer_uses_nested_record_and_shelf_position(tmp_path):
    shelf=entity('キャビネット1',id='shelf',kind='container')
    box=entity('青い箱',id='box',parent_id='shelf',relation_to_parent='inside',kind='container')
    item=entity('充電ケーブル',id='item',parent_id='box',relation_to_parent='inside',metadata={'shelf_item':1,'level':2,'side':'右'})
    s=service(tmp_path,[shelf,box,item])
    r=asyncio.run(s.query('room','充電ケーブルどこ？'))
    answer=search_answer(r,s.db.get_entity)
    assert 'キャビネット1' in answer and '青い箱' in answer and '上から2段目・右' in answer
    assert '記録されています' in answer and '(0' not in answer


def test_cross_room_parent_never_leaks(tmp_path):
    s=service(tmp_path,[entity('はさみ',id='item',parent_id='other')])
    s.db.upsert_scene(Scene(id='other-room',name='Other'))
    s.db.upsert_entity(entity('秘密の箱',id='other').model_copy(update={'scene_id':'other-room'}))
    assert '秘密' not in entity_answer(s.db.get_entity('item'),s.db.get_entity)


def test_cache_tracks_text_edits_and_ignores_image_changes():
    e=SemanticSearchEngine(device='cpu');calls=[]
    def embed(texts,prefix):
        calls.append((prefix,texts[:]))
        return torch.tensor([[1.,0.] for _ in texts])
    e._embed=embed
    item=entity('はさみ',id='item')
    e.search('切る',[item]);e.search('切る',[item])
    item.metadata['shelf_photo']='data:image/jpeg;base64,new'
    e.search('切る',[item])
    assert len([c for c in calls if c[0]=='passage'])==1
    item.metadata['note']='紙を切る'
    e.search('切る',[item])
    assert len([c for c in calls if c[0]=='passage'])==2


def test_scissors_word_not_destroyed():
    assert normalize('where is scissors?')=='scissors'


def test_unavailable_semantic_disclosed(tmp_path):
    class Broken:
        def search(self,*args,**kwargs): raise RuntimeError('offline')
    s=service(tmp_path,[entity('はさみ')],Broken())
    r=asyncio.run(s.query('room','はさみどこ？'))
    assert r.status=='matched' and r.search_mode=='lexical_fallback'
