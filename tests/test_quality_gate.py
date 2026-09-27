from copy import deepcopy

import pytest

from spatial_memory.quality_gate import compare_reports


def report():
    return dict(image_size=[518,294], alignment_sha256='a', camera_files_sha256='b',
                geometry_npz_sha256='c', finite_parameters=True, max_camera_center_error=0,
                test_views=[dict(frame=str(i), psnr=23., ssim=.85, depth_abs_relative=.1, alpha_coverage=.99)
                            for i in (0,8,16)])


def test_depth_improvement_accepts_same_coordinates():
    baseline = report()
    candidate = deepcopy(baseline)
    for row in candidate['test_views']: row['depth_abs_relative'] = .08
    assert compare_reports(baseline, candidate)['accepted']


def test_average_gain_does_not_hide_bad_view():
    baseline, candidate = report(), report()
    for row, gain in zip(candidate['test_views'], (3,3,-1)):
        row['psnr'] += gain
    decision = compare_reports(baseline, candidate)
    assert not decision['accepted']
    assert decision['worst_view_psnr_delta'] == -1


@pytest.mark.parametrize('change', ['nan', 'camera', 'frames', 'coverage', 'unchanged', 'missing'])
def test_rejects_unreliable_or_incomparable_candidate(change):
    baseline, candidate = report(), report()
    if change == 'nan': candidate['test_views'][0]['psnr'] = float('nan')
    if change == 'camera': candidate['camera_files_sha256'] = 'other'
    if change == 'frames': candidate['test_views'].pop()
    if change == 'coverage': candidate['test_views'][0]['alpha_coverage'] = .5
    if change == 'missing': candidate.pop('geometry_npz_sha256')
    assert not compare_reports(baseline, candidate)['accepted']
