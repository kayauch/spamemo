import numpy as np
import pytest
from spatial_memory.photo_alignment import fit_offset,offset_polygon


def samples(levels):
    return np.repeat(levels,30),np.repeat([f'view-{i}' for i in range(len(levels))],30)


def test_plane_correction_requires_other_camera_origins_to_improve():
    values,groups=samples([.10,.11,.09,.10,.12,.10])
    corrected,report=fit_offset(values,groups,0)
    assert corrected==pytest.approx(.105)
    assert report['accepted'] and report['after_error']<report['before_error']
    assert report['heldout_viewpoints']==2


def test_disagreeing_heldout_views_keep_original_plane():
    values,groups=samples([0,.15,.15,0,.15,.15])
    corrected,report=fit_offset(values,groups,0)
    assert corrected==0 and not report['accepted']
    assert report['reason']=='heldout_views_disagree'


def test_repeated_pixels_from_one_view_are_not_independent_evidence():
    corrected,report=fit_offset(np.full(10000,.1),np.full(10000,'same-camera'),0)
    assert corrected==0 and report['reason']=='insufficient_independent_views'


def test_already_aligned_and_large_changes_are_preserved():
    for levels in ([.001]*6,[.6]*6):
        values,groups=samples(levels)
        corrected,report=fit_offset(values,groups,0)
        assert corrected==0 and not report['accepted']


def test_shared_wall_corners_stay_connected_without_changing_topology():
    polygon=np.array([[0.,0],[4,0],[4,3],[0,3]])
    updated=offset_polygon(polygon,[.1,0,0,0])
    assert updated.tolist()==[[0,.1],[4,.1],[4,3],[0,3]]
    assert polygon.tolist()==[[0,0],[4,0],[4,3],[0,3]]
    with pytest.raises(ValueError):offset_polygon(polygon,[1.5,0,0,0])
