from copy import deepcopy

import numpy as np
import pytest

from spatial_memory.furniture_quality import (select_furniture_frames, camera_viewpoints,
    object_quality, part_hints, apply_furniture_overrides)
from spatial_memory.image_furniture import merge_observations
from spatial_memory.solid_room import furniture_parts


def cameras(centers):
    return np.asarray([np.column_stack([np.eye(3), -np.asarray(c)]) for c in centers])


def test_selection_covers_late_capture_and_prefers_sharp_frame_in_each_region():
    records = [dict(sharpness=10 if i % 10 else 500) for i in range(60)]
    poses = cameras([[i / 10, 0, 0] for i in range(60)])
    report = select_furniture_frames(records, poses, 12)
    assert len(report['indices']) == 12
    assert set(range(0, 60, 10)).issubset(report['indices'])
    assert any(i >= 50 for i in report['indices'])
    assert report == select_furniture_frames(records, poses, 12)


def test_panorama_faces_do_not_multiply_independent_support_even_with_pose_drift():
    records = [dict(panorama_group='pano-a') for _ in range(6)] + [dict(kind='video')]
    poses = cameras([[i * .1, 0, 0] for i in range(6)] + [[2, 0, 0]])
    groups, _ = camera_viewpoints(records, poses)
    assert len(set(groups[:6])) == 1
    assert groups[-1] != groups[0]
    points = np.random.default_rng(7).uniform(-1, 1, (1000, 3))
    observations = [dict(frame=i, viewpoint_id=groups[i], label='table', score=.8, points=points) for i in range(6)]
    group = merge_observations(observations)[0]
    quality = object_quality(group)
    assert quality['frame_count'] == 6 and quality['independent_viewpoints'] == 1
    assert 'single_camera_origin' in quality['issues']
    assert quality['needs_photo'][0]['code'] == 'new_camera_position'


def test_camera_rotation_in_place_is_not_independent_position():
    poses = cameras([[1, 2, 3], [1, 2, 3]])
    poses[1, :, :3] = np.diag([-1, -1, 1])
    poses[1, :, 3] = -poses[1, :, :3] @ [1, 2, 3]
    groups, _ = camera_viewpoints([{}, {}], poses)
    assert groups[0] == groups[1]


def test_similar_category_conflict_retains_evidence_instead_of_coercing_labels():
    points = np.random.default_rng(2).uniform(0, 1, (1000, 3))
    observations = [dict(frame=0,label='cabinet',score=.95,points=points),
                    dict(frame=1,label='wardrobe',score=.8,points=points)]
    groups = merge_observations(observations)
    assert len(groups) == 1 and groups[0]['label'] == 'cabinet'
    assert object_quality(groups[0])['conflicts'] == ['category_conflict']


def test_visible_bounds_variation_requires_additional_object_photo():
    points = np.random.default_rng(1).uniform(0, 1, (1000, 3))
    quality = object_quality(dict(frames=[0, 1], label='chair', points=[points, points * 3], scores=[.8,.9]))
    assert 'inconsistent_visible_dimensions' in quality['conflicts']
    assert any(x['code'] == 'full_object_photo' for x in quality['needs_photo'])


def test_user_dimensions_keep_bottom_and_do_not_mutate_source_or_keep_stale_coordinates():
    layout = dict(boxes=[dict(id=3, **{'class':'bbox'}, label='chair', center=[1,2,.6], scale=[1,1,1],
                             rotation=np.eye(3).tolist(), vg_world={'scale':[9,9,9]})],
                  image_furniture={'quality_report':{'objects':[{'id':3,'label':'chair'}]}})
    original = deepcopy(layout)
    result = apply_furniture_overrides(layout, [dict(id=3,label='table',hidden=True,dimensions_m=[1,.6,.8])],unit_scale=2)
    box = result['boxes'][0]
    assert box['scale'] == [.5,.3,.4] and box['label'] == 'table'
    assert box['center'][2] - box['scale'][2]/2 == pytest.approx(.1)
    assert 'vg_world' not in box and box['hidden']
    report = result['image_furniture']['quality_report']['objects'][0]
    assert report['label'] == 'table' and report['hidden'] and report['user_corrected']
    assert report['dimensions_model_units'] == box['scale']
    assert layout == original
    assert furniture_parts(box,0) == []
    with pytest.raises(ValueError): apply_furniture_overrides(layout,[dict(id=3,dimensions_m=[1,float('nan'),1])])
    with pytest.raises(ValueError): apply_furniture_overrides(layout,[dict(id=99,label='chair')])


def test_observed_seat_level_guides_part_but_legs_remain_explicit_completion():
    rng = np.random.default_rng(3)
    points = rng.uniform(-.5,.5,(1200,3));points[:900,2] = -.12
    hints = part_hints(points+[0,0,.5],[0,0,.5],np.eye(3),[1,1,1],'chair')
    assert hints['seat_local_z'] == pytest.approx(-.12)
    box = dict(label='chair',center=[0,0,.5],scale=[1,1,1],rotation=np.eye(3),evidence={'part_hints':hints})
    parts = furniture_parts(box,0)
    assert parts[0]['center'][2] == pytest.approx(.38)
    assert not parts[0]['inferred_completion']
    assert all(p['inferred_completion'] for p in parts[1:])
    box['evidence']={}
    assert furniture_parts(box,0)[0]['inferred_completion']


@pytest.mark.parametrize('budget',[True,1,33,2.5])
def test_invalid_selection_budget_rejected(budget):
    with pytest.raises(ValueError):select_furniture_frames([{},{}],cameras([[0,0,0],[1,0,0]]),budget)


def test_user_can_accept_tentative_furniture_without_changing_source():
    from spatial_memory.furniture_quality import apply_furniture_overrides
    source = {'boxes': [{'id': 7, 'class': 'bbox', 'label': 'wardrobe', 'hidden': True,
                        'review_status': 'needs_confirmation', 'center': [1,2,1], 'scale': [1,.5,2],
                        'evidence': {'valid': True, 'requires_confirmation': True}}],
              'image_furniture': {'quality_report': {'objects': [], 'unresolved_candidates': [{'id':7}]}}}
    updated = apply_furniture_overrides(source, [{'id':'7', 'hidden':False}])
    assert updated['boxes'][0]['review_status'] == 'user_accepted'
    assert not updated['boxes'][0]['hidden']
    assert not updated['boxes'][0]['evidence']['requires_confirmation']
    assert updated['image_furniture']['quality_report']['unresolved_candidates'] == []
    assert source['boxes'][0]['hidden'] and source['boxes'][0]['review_status'] == 'needs_confirmation'


def test_user_shape_change_does_not_keep_old_geometry_acceptance():
    geometry = dict(id=7, accepted=True, status='accepted_observed_envelope', candidate={'front': 1})
    layout = {'boxes': [dict(id=7, label='locker', **{'class':'bbox'}, center=[1,1,1], scale=[1,.4,2],
                            evidence={'storage_geometry': geometry})],
              'image_furniture': {'storage_geometry': {'accepted_count': 1, 'objects': [geometry]},
                                  'quality_report': {'objects': [{'id': 7}]}},
              'storage_geometry': {'accepted_count': 1, 'objects': [geometry]}}
    hidden = apply_furniture_overrides(layout, [{'id':7, 'hidden':True}])
    assert hidden['boxes'][0]['evidence']['storage_geometry']['accepted']
    revised = apply_furniture_overrides(layout, [{'id':7, 'dimensions_m':[1,.6,2]}])
    current = revised['boxes'][0]['evidence']['storage_geometry']
    assert not current['accepted'] and not current['current_geometry_validated']
    assert current['prior_validation']['candidate'] == {'front':1}
    assert revised['image_furniture']['storage_geometry']['accepted_count'] == 0
    assert revised['storage_geometry']['accepted_count'] == 0
    assert revised['image_furniture']['quality_report']['objects'][0]['geometry_status'] == 'superseded_by_user_override'
    assert layout['boxes'][0]['evidence']['storage_geometry']['accepted']


def test_uploaded_photos_survive_sampling_without_displacing_video_evidence():
    records = [dict(source_kind='video', sharpness=100) for _ in range(54)]
    records += [dict(source_kind='photo', sharpness=1) for _ in range(6)]
    poses = cameras([[i / 10, 0, 0] for i in range(60)])
    baseline = select_furniture_frames([dict(r, source_kind='video') for r in records], poses, 24)
    report = select_furniture_frames(records, poses, 24)
    assert set(baseline['indices']).issubset(report['indices'])
    assert set(range(54, 60)).issubset(report['indices'])
    assert report['required_photo_indices'] == list(range(54, 60))
    assert report['selected_count'] == len(set(report['indices']))
    assert report == select_furniture_frames(records, poses, 24)


def test_photo_only_capture_is_not_silently_limited_by_video_sampling_budget():
    records = [dict(kind='photo', sharpness=1) for _ in range(35)]
    poses = cameras([[i / 10, 0, 0] for i in range(35)])
    report = select_furniture_frames(records, poses, 2)
    assert report['indices'] == list(range(35))
    assert report['requested_budget'] == 2
    assert report['selected_count'] == 35
