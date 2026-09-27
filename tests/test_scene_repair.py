from copy import deepcopy
import numpy as np
import pytest
from spatial_memory.scene_repair import (
    correspondence, composite_locked, classify, bounded_region, repair_decision, OBSERVED, UNKNOWN,
)


def test_warp_rejects_occlusion_and_behind_camera():
    depth=np.ones((10,12),np.float32)
    e=np.column_stack((np.eye(3),np.zeros(3)))
    k=np.array([[8,0,6],[0,8,5],[0,0,1.]])
    uv,valid=correspondence(depth,e,k,depth,e,k)
    assert valid[5,6] and np.allclose(uv[5,6],[6,5])
    _,occluded=correspondence(depth,e,k,depth*.5,e,k)
    assert not occluded.any()
    behind=e.copy(); behind[2,3]=-2
    _,invalid=correspondence(depth,e,k,depth,behind,k)
    assert not invalid.any()


def test_invalid_depth_never_qualifies_for_repair():
    a=np.zeros((12,12,3),np.uint8); b=np.full_like(a,255)
    depth=np.ones((12,12)); depth[:6]=np.nan
    labels,broken,_=classify(a,b,np.ones((12,12)),depth,np.ones((12,12),bool))
    assert not broken[:6].any()
    assert np.all(labels[:6]==UNKNOWN)
    assert labels[8,8]==OBSERVED and broken[8,8]


def test_good_pixels_locked_even_if_generator_changes_whole_image():
    source=np.arange(300,dtype=np.uint8).reshape(10,10,3)
    proposal=np.zeros_like(source); mask=np.zeros((10,10),bool); mask[4:6,4:6]=True
    result=composite_locked(source,proposal,mask)
    assert np.array_equal(result[~mask],source[~mask])
    assert np.array_equal(result[mask],proposal[mask])
    with pytest.raises(ValueError): composite_locked(source,proposal[:2],mask)


def report():
    return dict(alignment_sha256='a',camera_files_sha256='b',image_size=[518,294],
        finite_parameters=True,max_camera_center_error=0,model_ply_sha256='best',
        repair_reference_model_sha256='best',mean_depth_to_best_relative=0,mean_hole_fraction=.01,repair_multiview_mae=.05,
        test_views=[dict(frame=str(i),psnr=23.,ssim=.85) for i in range(3)])


def test_repair_requires_heldout_improvement_and_rejects_bad_view():
    before=report(); after=deepcopy(before)
    assert not repair_decision(before,after,{'passed':True})['accepted']
    for row in after['test_views']: row['psnr']+=.1
    assert repair_decision(before,after,{'passed':True})['accepted']
    after['test_views'][0]['psnr']-=.5
    assert not repair_decision(before,after,{'passed':True})['accepted']


@pytest.mark.parametrize('problem',['geometry','holes','camera','multiview','render_consistency','nan'])
def test_repair_rejects_nonappearance_regressions(problem):
    before=report(); after=deepcopy(before)
    for row in after['test_views']: row['psnr']+=1
    if problem=='geometry': after['mean_depth_to_best_relative']=.1
    if problem=='holes': after['mean_hole_fraction']=.1
    if problem=='camera': after['alignment_sha256']='other'
    if problem=='render_consistency': after['repair_multiview_mae']=.2
    if problem=='nan': after['mean_depth_to_best_relative']=float('nan')
    assert not repair_decision(before,after,{'passed':problem!='multiview'})['accepted']
