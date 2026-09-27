import numpy as np

from spatial_memory.camera_refinement import refine_capture_cameras, refine_pose_from_correspondences
from spatial_recon.types import ReconstructionResult


def synthetic_correspondences():
    rng = np.random.default_rng(429)
    points = rng.uniform([-1, -.7, 2], [1, .7, 4], (150, 3))
    intrinsic = np.array([[300., 0, 160], [0, 300, 120], [0, 0, 1]])
    exact = np.eye(4)[:3].copy()
    projected = points @ intrinsic.T
    pixels = projected[:, :2] / projected[:, 2:]
    observed_depth = points[:, 2].copy()
    initial = exact.copy()
    initial[0, 3] = .06
    return points, pixels, observed_depth, intrinsic, initial


def test_local_pose_error_improves_on_heldout_matches():
    points, pixels, depth, intrinsic, initial = synthetic_correspondences()
    original = initial.copy()
    refined, report = refine_pose_from_correspondences(points, pixels, depth, intrinsic, initial)
    assert report["accepted"] is True
    assert report["heldout_matches"] == 30
    assert report["after_reprojection_px"] < report["before_reprojection_px"] * .1
    assert report["after_relative_depth_error"] <= report["before_relative_depth_error"] + .005
    assert np.linalg.norm(refined[:, 3]) < .001
    assert np.array_equal(initial, original)


def test_depth_disagreement_rejects_an_image_only_improvement():
    points, pixels, depth, intrinsic, initial = synthetic_correspondences()
    depth *= 1.5
    refined, report = refine_pose_from_correspondences(points, pixels, depth, intrinsic, initial)
    assert report["accepted"] is False
    assert report["reason"] == "cross_view_depth_disagreement"
    assert np.array_equal(refined, initial)


def test_large_corrections_are_rejected_without_mutation():
    points, pixels, depth, intrinsic, initial = synthetic_correspondences()
    initial[0, 3] = .8
    original = initial.copy()
    refined, report = refine_pose_from_correspondences(points, pixels, depth, intrinsic, initial)
    assert not report["accepted"]
    assert report["reason"] == "correction_exceeds_local_bounds"
    assert np.array_equal(refined, original)
    assert np.array_equal(initial, original)


def test_heldout_corruption_is_not_hidden_by_training_success():
    points, pixels, depth, intrinsic, initial = synthetic_correspondences()
    pixels[::5, 0] += 20
    refined, report = refine_pose_from_correspondences(points, pixels, depth, intrinsic, initial)
    assert not report["accepted"]
    assert report["reason"] == "heldout_reprojection_did_not_improve"
    assert np.array_equal(refined, initial)


def test_blank_images_and_panorama_rig_remain_unchanged():
    result = ReconstructionResult(
        depth=np.ones((4, 64, 64), np.float32),
        extrinsics=np.repeat(np.eye(4)[None, :3], 4, axis=0),
        intrinsics=np.repeat(np.array([[[50., 0, 32], [0, 50, 32], [0, 0, 1]]]), 4, axis=0),
        images=np.full((4, 64, 64, 3), 180, np.uint8),
    )
    before = result.extrinsics.copy()
    report = refine_capture_cameras(result, [{}, {}, {"panorama_group": "p"}, {"panorama_group": "p"}])
    assert report["accepted_count"] == 0
    assert report["frames"][1]["reason"] == "insufficient_texture"
    assert report["frames"][2]["reason"] == "panorama_rig_preserved"
    assert np.array_equal(result.extrinsics, before)


def test_sensor_cameras_are_not_refined():
    result = ReconstructionResult(np.ones((2, 4, 4)), np.repeat(np.eye(4)[None, :3], 2, axis=0), np.repeat(np.eye(3)[None], 2, axis=0), metrics={"model": "calibrated-rgbd"})
    assert refine_capture_cameras(result, [{}, {}])["status"] == "skipped_calibrated_sensor"
