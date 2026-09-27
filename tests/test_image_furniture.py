import numpy as np
import pytest
from spatial_memory.image_furniture import lift_mask, merge_observations, UP_ROTATIONS, classify_detection


def test_upright_mask_roundtrip_and_lifting():
    mask=np.zeros((4,6),bool);mask[1,2]=True
    depth=np.ones((4,6))*2;conf=np.ones_like(depth)*2
    for rotation in UP_ROTATIONS.values():
        assert np.array_equal(np.rot90(np.rot90(mask,rotation),-rotation),mask)
    e=np.column_stack([np.eye(3),[1,0,0]])
    t=np.eye(4);t[:3,3]=[3,4,5]
    assert np.allclose(lift_mask(mask,depth,conf,np.eye(3),e,t),[[6,6,7]])
    conf[1,2]=0
    assert lift_mask(mask,depth,conf,np.eye(3),e,t).shape==(0,3)
    with pytest.raises(ValueError):lift_mask(mask[:2],depth,conf,np.eye(3),e,t)


def test_association_keeps_separate_chairs_and_does_not_double_count_frame():
    rng=np.random.default_rng(1);points=rng.uniform(0,1,(1000,3))
    obs=[dict(frame=0,label='chair',score=.9,points=points),dict(frame=1,label='chair',score=.8,points=points+.03),
         dict(frame=2,label='chair',score=.8,points=points+4),dict(frame=0,label='chair',score=.7,points=points),
         dict(frame=3,label='bed',score=.8,points=points)]
    groups=merge_observations(obs)
    assert len(groups)==4
    assert groups[0]['frames']==[0,1]
    assert all(len(g['frames'])==len(set(g['frames'])) for g in groups)


def test_semantic_labels_keep_cross_family_conflicts_unknown():
    assert classify_detection('locker') == ('furniture', 'locker')
    assert classify_detection('cabinet locker') == ('furniture', 'cabinet')
    assert classify_detection('cabinet bookshelf shelf') == ('furniture', 'cabinet')
    assert classify_detection('cabinet box') == ('unknown', 'unknown')
    assert classify_detection('wall cabinet') == ('unknown', 'unknown')
    assert classify_detection('chair shelf') == ('unknown', 'unknown')
    assert classify_detection('wall') == ('wall', 'wall')
    assert classify_detection('unrecognized') == ('unknown', 'unknown')


def test_storage_association_preserves_exact_mask_ids():
    points = np.random.default_rng(4).uniform(0, 1, (500, 3))
    records = [dict(frame=i, label=label, detection_id=f'mask-{i}',
                    score=.9, points=points + i*.001)
               for i, label in enumerate(('locker', 'cabinet', 'wardrobe'))]
    groups = merge_observations(records)
    assert len(groups) == 1
    assert groups[0]['detection_ids'] == ['mask-0', 'mask-1', 'mask-2']


def test_small_box_inside_cabinet_stays_a_separate_instance():
    rng = np.random.default_rng(2)
    cabinet = rng.uniform([-1,-.4,0], [1,.4,2], (1000,3))
    box = rng.uniform([-.2,-.2,.8], [.2,.2,1.2], (1000,3))
    groups = merge_observations([dict(frame=0,label='cabinet',detection_id='large',score=.9,points=cabinet),
                                 dict(frame=1,label='box',detection_id='small',score=.8,points=box)])
    assert len(groups) == 2
    assert {tuple(g['detection_ids']) for g in groups} == {('large',),('small',)}
