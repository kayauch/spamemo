import cv2
import numpy as np

from spatial_memory.models import SpatialEntity, Vec3
from spatial_memory.relocalization import (
    PoseEstimate,
    PoseSmoother,
    SceneRelocalizer,
    target_guidance,
)


def test_pnp_recovers_camera_pose() -> None:
    rng = np.random.default_rng(42)
    world = rng.uniform(
        low=[-2.0, -1.5, 3.0],
        high=[2.0, 1.5, 8.0],
        size=(80, 3),
    ).astype(np.float64)

    camera_matrix = np.array([
        [700.0, 0.0, 320.0],
        [0.0, 700.0, 240.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    true_rvec = np.array([0.04, -0.08, 0.03], dtype=np.float64)
    true_tvec = np.array([0.25, -0.12, 0.45], dtype=np.float64)
    image, _ = cv2.projectPoints(
        world,
        true_rvec,
        true_tvec,
        camera_matrix,
        None,
    )
    image = image.reshape(-1, 2)
    image += rng.normal(0.0, 0.25, size=image.shape)

    solved = SceneRelocalizer.estimate_pose_from_correspondences(
        world,
        image,
        camera_matrix,
        reprojection_error=2.0,
        min_inliers=20,
    )
    assert solved is not None
    rotation, translation, inliers, error = solved

    true_rotation, _ = cv2.Rodrigues(true_rvec)
    true_center = -true_rotation.T @ true_tvec
    estimated_center = -rotation.T @ translation

    assert len(inliers) > 65
    assert error < 1.0
    assert np.linalg.norm(estimated_center - true_center) < 0.05


def test_target_guidance_projects_in_front_target() -> None:
    camera_matrix = np.array([
        [500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0],
    ])
    pose = PoseEstimate(
        reference_frame=0,
        rotation_world_to_camera=np.eye(3),
        translation_world_to_camera=np.zeros(3),
        camera_position_world=Vec3(x=0, y=0, z=0),
        inliers=30,
        matches=40,
        reprojection_error=0.5,
        image_width=640,
        image_height=480,
        camera_matrix=camera_matrix,
        seconds=0.1,
    )
    target = SpatialEntity(
        scene_id="room",
        name="target",
        position=Vec3(x=1.0, y=0.0, z=5.0),
    )
    guide = target_guidance(pose, target)

    assert guide["in_front"] is True
    assert guide["on_screen"] is True
    assert guide["screen_target"]["x"] > 320
    assert guide["arrow"]["x"] > 0
    assert guide["distance"] > 5.0


def test_target_guidance_marks_behind_target() -> None:
    pose = PoseEstimate(
        reference_frame=0,
        rotation_world_to_camera=np.eye(3),
        translation_world_to_camera=np.zeros(3),
        camera_position_world=Vec3(x=0, y=0, z=0),
        inliers=30,
        matches=40,
        reprojection_error=0.5,
        image_width=640,
        image_height=480,
        camera_matrix=np.array([
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0],
        ]),
        seconds=0.1,
    )
    target = SpatialEntity(
        scene_id="room",
        name="behind",
        position=Vec3(x=-2.0, y=0.0, z=-4.0),
    )
    guide = target_guidance(pose, target)

    assert guide["in_front"] is False
    assert guide["on_screen"] is False
    assert abs(guide["arrow"]["x"]) > 0.5



def test_relocalization_features_persist_across_instances(tmp_path) -> None:
    from pathlib import Path

    from spatial_memory.models import Scene

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    rng = np.random.default_rng(7)
    base = rng.integers(0, 256, size=(120, 160), dtype=np.uint8)
    for index in range(4):
        image = base.copy()
        cv2.putText(
            image,
            f"F{index}",
            (12 + index * 3, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            255,
            2,
        )
        cv2.imwrite(str(frames_dir / f"{index:06d}.jpg"), image)

    reconstruction_dir = tmp_path / "reconstruction"
    reconstruction_dir.mkdir()
    reconstruction = reconstruction_dir / "reconstruction.npz"
    np.savez(
        reconstruction,
        depth=np.ones((4, 10, 10), dtype=np.float32),
        intrinsics=np.repeat(np.eye(3, dtype=np.float32)[None], 4, axis=0),
        extrinsics=np.repeat(
            np.concatenate(
                [
                    np.eye(3, dtype=np.float32),
                    np.zeros((3, 1), dtype=np.float32),
                ],
                axis=1,
            )[None],
            4,
            axis=0,
        ),
    )
    scene = Scene(
        id="persistent-room",
        name="Persistent Room",
        frames_dir=str(frames_dir),
        reconstruction_npz=str(reconstruction),
        frame_count=4,
    )

    first = SceneRelocalizer()
    refs1 = first.prepare(scene, max_reference_frames=4)
    assert len(refs1) == 4
    assert first.last_prepare_info["source"] == "generated"
    index_dir = Path(str(first.last_prepare_info["index_dir"]))
    assert (index_dir / "manifest.json").exists()
    assert (index_dir / "features.npz").exists()

    second = SceneRelocalizer()
    refs2 = second.prepare(scene, max_reference_frames=4)
    assert len(refs2) == 4
    assert second.last_prepare_info["source"] == "disk"
    assert np.allclose(refs1[0].keypoints, refs2[0].keypoints)
    assert np.allclose(refs1[0].descriptors, refs2[0].descriptors)



def test_pose_smoother_reduces_position_jump() -> None:
    camera_matrix = np.eye(3, dtype=np.float64)
    smoother = PoseSmoother(
        position_alpha=0.5,
        rotation_alpha=0.5,
        reset_after_seconds=60.0,
    )
    first = PoseEstimate(
        reference_frame=0,
        rotation_world_to_camera=np.eye(3),
        translation_world_to_camera=np.zeros(3),
        camera_position_world=Vec3(x=0.0, y=0.0, z=0.0),
        inliers=50,
        matches=60,
        reprojection_error=0.5,
        image_width=640,
        image_height=480,
        camera_matrix=camera_matrix,
        seconds=0.1,
        confidence=1.0,
    )
    second = PoseEstimate(
        reference_frame=1,
        rotation_world_to_camera=np.eye(3),
        translation_world_to_camera=np.array([-10.0, 0.0, 0.0]),
        camera_position_world=Vec3(x=10.0, y=0.0, z=0.0),
        inliers=50,
        matches=60,
        reprojection_error=0.5,
        image_width=640,
        image_height=480,
        camera_matrix=camera_matrix,
        seconds=0.1,
        confidence=1.0,
    )

    first_smoothed, first_info = smoother.update("session", first)
    second_smoothed, second_info = smoother.update("session", second)

    assert first_info["applied"] is False
    assert first_smoothed.camera_position_world.x == 0.0
    assert second_info["applied"] is True
    assert abs(second_smoothed.camera_position_world.x - 5.0) < 1e-6

    smoother.reset("session")
    reset_pose, reset_info = smoother.update("session", second)
    assert reset_info["applied"] is False
    assert reset_pose.camera_position_world.x == 10.0


def test_rotated_portrait_intrinsics_recover_same_camera_center() -> None:
    rng = np.random.default_rng(123)
    world = rng.uniform(
        low=[-2.0, -1.5, 3.0],
        high=[2.0, 1.5, 8.0],
        size=(120, 3),
    ).astype(np.float64)

    landscape_k = np.array([
        [720.0, 0.0, 320.0],
        [0.0, 700.0, 240.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    true_rvec = np.array([0.05, -0.07, 0.02], dtype=np.float64)
    true_tvec = np.array([0.15, -0.09, 0.35], dtype=np.float64)
    image, _ = cv2.projectPoints(
        world,
        true_rvec,
        true_tvec,
        landscape_k,
        None,
    )
    image = image.reshape(-1, 2)

    # Rotate the displayed image 90 degrees clockwise:
    # landscape 640x480 -> portrait 480x640.
    rotated = np.column_stack([
        479.0 - image[:, 1],
        image[:, 0],
    ])

    candidates = dict(SceneRelocalizer._candidate_camera_matrices(
        landscape_k,
        depth_shape=(480, 640),
        reference_size=(640, 480),
        query_size=(480, 640),
    ))
    assert "cw90" in candidates

    solved = SceneRelocalizer.estimate_pose_from_correspondences(
        world,
        rotated,
        candidates["cw90"],
        reprojection_error=2.0,
        min_inliers=20,
    )
    assert solved is not None

    rotation, translation, inliers, error = solved
    true_rotation, _ = cv2.Rodrigues(true_rvec)
    true_center = -true_rotation.T @ true_tvec
    estimated_center = -rotation.T @ translation

    assert len(inliers) > 100
    assert error < 0.5
    assert np.linalg.norm(estimated_center - true_center) < 0.05


def test_camera_position_plausibility_rejects_absurd_pose() -> None:
    reference_centers = np.array([
        [0.0, 0.0, 0.0],
        [0.4, 0.0, 0.1],
        [0.8, 0.1, 0.2],
        [1.2, 0.1, 0.3],
    ], dtype=np.float64)

    ok, nearest, maximum = SceneRelocalizer._camera_position_plausible(
        np.array([1.0, 0.3, 0.5], dtype=np.float64),
        reference_centers,
    )
    assert ok is True
    assert nearest < maximum

    bad, bad_nearest, same_maximum = (
        SceneRelocalizer._camera_position_plausible(
            np.array([1e8, -1e8, 1e8], dtype=np.float64),
            reference_centers,
        )
    )
    assert bad is False
    assert bad_nearest > same_maximum


def test_pose_smoother_exposes_only_recent_pose() -> None:
    pose = PoseEstimate(
        reference_frame=3,
        rotation_world_to_camera=np.eye(3),
        translation_world_to_camera=np.zeros(3),
        camera_position_world=Vec3(x=0.1, y=0.2, z=0.3),
        inliers=25,
        matches=40,
        reprojection_error=1.0,
        image_width=1080,
        image_height=1920,
        camera_matrix=np.eye(3),
        seconds=0.2,
        confidence=0.8,
        image_rotation="cw90",
    )
    smoother = PoseSmoother(reset_after_seconds=3.0)
    stored, _ = smoother.update("room:session", pose)

    recent = smoother.recent_pose("room:session", max_age_seconds=1.5)
    assert recent is not None
    recent_pose, age = recent
    assert recent_pose.camera_position_world == stored.camera_position_world
    assert recent_pose.image_rotation == "cw90"
    assert age >= 0.0

    smoother._states["room:session"].updated_at -= 2.0
    assert smoother.recent_pose(
        "room:session",
        max_age_seconds=1.5,
    ) is None

    smoother.reset("room:session")
    assert smoother.recent_pose("room:session") is None


def test_pose_smoother_refresh_recent_pose_extends_window() -> None:
    pose = PoseEstimate(
        reference_frame=1,
        rotation_world_to_camera=np.eye(3),
        translation_world_to_camera=np.zeros(3),
        camera_position_world=Vec3(x=0.0, y=0.0, z=0.0),
        inliers=20,
        matches=30,
        reprojection_error=1.0,
        image_width=640,
        image_height=480,
        camera_matrix=np.eye(3),
        seconds=0.1,
        confidence=0.8,
    )
    smoother = PoseSmoother()
    smoother.update("room:track", pose)
    smoother._states["room:track"].updated_at -= 10.0
    assert smoother.recent_pose(
        "room:track",
        max_age_seconds=5.0,
    ) is None

    assert smoother.refresh_recent_pose("room:track") is True
    assert smoother.recent_pose(
        "room:track",
        max_age_seconds=5.0,
    ) is not None
    assert smoother.refresh_recent_pose("missing") is False
