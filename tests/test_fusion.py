import numpy as np
import pytest
from spatial_memory.fusion import supported_faces
from spatial_memory.depth_prior import reliable_depth_mask, multi_view_depth_masks


def test_supported_face_requires_two_views_and_all_samples():
    triangle = np.array([[-.1, -.1, 2], [.1, -.1, 2], [0, .1, 2]])
    faces = np.array([triangle, triangle + [0, 0, 1], triangle + [0, 0, -3]])
    e = np.column_stack([np.eye(3), np.zeros(3)])
    k = np.array([[100., 0, 100], [0, 100, 100], [0, 0, 1]])
    d = np.full((2, 200, 200), 2.)
    c = np.full_like(d, 3.)
    args = (faces, d, np.repeat(e[None], 2, axis=0), np.repeat(k[None], 2, axis=0), c)
    keep, count = supported_faces(*args)
    assert keep.tolist() == [True, False, False]
    assert count.tolist() == [2, 0, 0]
    d[1, 98, 100] = 1  # projected centroid (100, 98.33), nearest pixel
    assert not supported_faces(*args)[0].any()


def test_depth_mask_excludes_invalid_edges_and_low_confidence():
    d = np.full((10, 10), 2.)
    c = np.full_like(d, 3.)
    d[:, 6:] = 4
    d[3, 3] = np.nan
    c[7, 3] = 1
    mask = reliable_depth_mask(d, c)
    assert mask[7, 4]
    assert not mask[:, 5:7].any()
    assert not mask[3, 3]
    assert not mask[7, 3]
    assert not mask[0].any()
    assert not reliable_depth_mask(d, np.ones_like(c)).any()
    with pytest.raises(ValueError):
        reliable_depth_mask(d, c[:2])


def test_multiview_mask_excludes_held_out_views_and_occluded_votes():
    d = np.full((4, 20, 20), 2.)
    c = np.full_like(d, 3.)
    e = np.repeat(np.column_stack([np.eye(3), np.zeros(3)])[None], 4, axis=0)
    k = np.repeat(np.array([[10.,0,10],[0,10,10],[0,0,1]])[None], 4, axis=0)
    masks, votes = multi_view_depth_masks(d, c, e, k)
    assert not masks[0].any()
    assert masks[1:, 10, 10].all()
    assert votes[1, 10, 10] == 2
    d[3] = 1  # two train views and one held-out view alone are insufficient
    assert not multi_view_depth_masks(d, c, e, k)[0].any()
