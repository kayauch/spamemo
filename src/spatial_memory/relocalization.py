from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .models import Scene, SpatialEntity, Vec3
from .projection import load_reconstruction


@dataclass
class ReferenceFeatures:
    frame_index: int
    path: Path
    width: int
    height: int
    keypoints: list | np.ndarray
    descriptors: np.ndarray | None
    descriptor_centroid: np.ndarray | None = None


@dataclass
class PoseEstimate:
    reference_frame: int
    rotation_world_to_camera: np.ndarray
    translation_world_to_camera: np.ndarray
    camera_position_world: Vec3
    inliers: int
    matches: int
    reprojection_error: float
    image_width: int
    image_height: int
    camera_matrix: np.ndarray
    seconds: float
    inlier_ratio: float = 0.0
    confidence: float = 0.0
    references_tested: int = 0
    image_rotation: str = "none"
    nearest_reference_camera_distance: float = 0.0
    max_plausible_camera_distance: float = 0.0


class SceneRelocalizer:
    def __init__(
        self,
        ratio_test: float = 0.75,
        min_matches: int = 12,
        min_inliers: int = 8,
        ransac_reprojection_error: float = 6.0,
        candidate_limit: int = 12,
        min_pose_confidence: float = 0.25,
    ):
        self.ratio_test = ratio_test
        self.min_matches = min_matches
        self.min_inliers = min_inliers
        self.ransac_reprojection_error = ransac_reprojection_error
        self.candidate_limit = candidate_limit
        self.min_pose_confidence = min_pose_confidence
        self.sift = cv2.SIFT_create(nfeatures=3500)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self._cache: dict[str, list[ReferenceFeatures]] = {}
        self._cache_lock = threading.Lock()
        self.last_prepare_info: dict[str, object] = {}

    def _frames(self, scene: Scene) -> list[Path]:
        if not scene.frames_dir:
            raise ValueError("Scene has no frames_dir")
        frames_dir = Path(scene.frames_dir)
        frames = sorted(
            item for item in frames_dir.iterdir()
            if item.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        if not frames:
            raise ValueError("Scene has no reference frames")
        return frames

    @staticmethod
    def _descriptor_centroid(descriptors: np.ndarray | None) -> np.ndarray | None:
        if descriptors is None or len(descriptors) == 0:
            return None
        centroid = descriptors.astype(np.float32).mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm <= 1e-8:
            return None
        return centroid / norm

    @staticmethod
    def _keypoint_xy(point) -> tuple[float, float]:
        if hasattr(point, "pt"):
            return float(point.pt[0]), float(point.pt[1])
        return float(point[0]), float(point[1])

    def _index_dir(self, scene: Scene, max_reference_frames: int) -> Path:
        if scene.frames_dir:
            scene_root = Path(scene.frames_dir).parent
        elif scene.reconstruction_npz:
            reconstruction_parent = Path(scene.reconstruction_npz).parent
            scene_root = (
                reconstruction_parent.parent
                if reconstruction_parent.name == "reconstruction"
                else reconstruction_parent
            )
        else:
            raise ValueError("Scene has no persistent storage path")
        return (
            scene_root
            / "relocalization"
            / f"sift-v1-{max_reference_frames}"
        )

    @staticmethod
    def _index_signature(
        scene: Scene,
        frames: list[Path],
        indices: list[int],
    ) -> str:
        digest = hashlib.sha1()
        digest.update(scene.id.encode("utf-8"))
        if scene.reconstruction_npz:
            reconstruction = Path(scene.reconstruction_npz)
            if reconstruction.exists():
                stat = reconstruction.stat()
                digest.update(
                    f"{reconstruction.name}:{stat.st_size}:{stat.st_mtime_ns}".encode()
                )
        for index in indices:
            frame = frames[index]
            stat = frame.stat()
            digest.update(
                f"{index}:{frame.name}:{stat.st_size}:{stat.st_mtime_ns}".encode()
            )
        return digest.hexdigest()

    def _load_index(
        self,
        scene: Scene,
        frames: list[Path],
        max_reference_frames: int,
        signature: str,
    ) -> list[ReferenceFeatures] | None:
        index_dir = self._index_dir(scene, max_reference_frames)
        manifest_path = index_dir / "manifest.json"
        data_path = index_dir / "features.npz"
        if not manifest_path.exists() or not data_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("signature") != signature:
                return None
            data = np.load(data_path, allow_pickle=True)
            prepared: list[ReferenceFeatures] = []
            for offset, frame_index in enumerate(data["frame_indices"].tolist()):
                descriptors = data["descriptors"][offset]
                if descriptors is not None and getattr(descriptors, "size", 0) == 0:
                    descriptors = None
                centroid = data["centroids"][offset]
                if centroid is not None and getattr(centroid, "size", 0) == 0:
                    centroid = None
                prepared.append(
                    ReferenceFeatures(
                        frame_index=int(frame_index),
                        path=frames[int(frame_index)],
                        width=int(data["widths"][offset]),
                        height=int(data["heights"][offset]),
                        keypoints=np.asarray(data["keypoints"][offset], dtype=np.float32),
                        descriptors=(
                            np.asarray(descriptors, dtype=np.float32)
                            if descriptors is not None
                            else None
                        ),
                        descriptor_centroid=(
                            np.asarray(centroid, dtype=np.float32)
                            if centroid is not None
                            else None
                        ),
                    )
                )
            return prepared or None
        except Exception:
            return None

    def _save_index(
        self,
        scene: Scene,
        max_reference_frames: int,
        signature: str,
        prepared: list[ReferenceFeatures],
    ) -> Path:
        index_dir = self._index_dir(scene, max_reference_frames)
        index_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            index_dir / "features.npz",
            frame_indices=np.asarray([item.frame_index for item in prepared], dtype=np.int32),
            widths=np.asarray([item.width for item in prepared], dtype=np.int32),
            heights=np.asarray([item.height for item in prepared], dtype=np.int32),
            keypoints=np.asarray([item.keypoints for item in prepared], dtype=object),
            descriptors=np.asarray(
                [
                    item.descriptors
                    if item.descriptors is not None
                    else np.empty((0, 128), dtype=np.float32)
                    for item in prepared
                ],
                dtype=object,
            ),
            centroids=np.asarray(
                [
                    item.descriptor_centroid
                    if item.descriptor_centroid is not None
                    else np.empty((0,), dtype=np.float32)
                    for item in prepared
                ],
                dtype=object,
            ),
        )
        (index_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "signature": signature,
                    "reference_frames": len(prepared),
                    "max_reference_frames": max_reference_frames,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return index_dir

    def prepare(
        self,
        scene: Scene,
        max_reference_frames: int = 60,
    ) -> list[ReferenceFeatures]:
        started = time.perf_counter()
        frames = self._frames(scene)
        if len(frames) > max_reference_frames:
            indices = np.linspace(
                0,
                len(frames) - 1,
                max_reference_frames,
                dtype=int,
            ).tolist()
        else:
            indices = list(range(len(frames)))

        signature = self._index_signature(scene, frames, indices)
        cache_key = f"{scene.id}:{max_reference_frames}:{signature}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            self.last_prepare_info = {
                "source": "memory",
                "seconds": time.perf_counter() - started,
                "reference_frames": len(cached),
                "index_dir": str(self._index_dir(scene, max_reference_frames)),
            }
            return cached

        persisted = self._load_index(
            scene,
            frames,
            max_reference_frames,
            signature,
        )
        if persisted is not None:
            with self._cache_lock:
                self._cache[cache_key] = persisted
            self.last_prepare_info = {
                "source": "disk",
                "seconds": time.perf_counter() - started,
                "reference_frames": len(persisted),
                "index_dir": str(self._index_dir(scene, max_reference_frames)),
            }
            return persisted

        prepared: list[ReferenceFeatures] = []
        for frame_index in indices:
            path = frames[frame_index]
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            keypoints, descriptors = self.sift.detectAndCompute(image, None)
            points = np.asarray(
                [keypoint.pt for keypoint in (keypoints or [])],
                dtype=np.float32,
            ).reshape(-1, 2)
            prepared.append(
                ReferenceFeatures(
                    frame_index=frame_index,
                    path=path,
                    width=image.shape[1],
                    height=image.shape[0],
                    keypoints=points,
                    descriptors=descriptors,
                    descriptor_centroid=self._descriptor_centroid(descriptors),
                )
            )

        if not prepared:
            raise ValueError("Could not extract reference image features")

        index_dir = self._save_index(
            scene,
            max_reference_frames,
            signature,
            prepared,
        )
        with self._cache_lock:
            self._cache[cache_key] = prepared
        self.last_prepare_info = {
            "source": "generated",
            "seconds": time.perf_counter() - started,
            "reference_frames": len(prepared),
            "index_dir": str(index_dir),
        }
        return prepared

    def clear(self, scene_id: str | None = None) -> None:
        with self._cache_lock:
            if scene_id is None:
                self._cache.clear()
            else:
                for key in list(self._cache):
                    if key.startswith(f"{scene_id}:"):
                        self._cache.pop(key, None)

    @staticmethod
    def _scale_intrinsics(
        intrinsic: np.ndarray,
        depth_shape: tuple[int, int],
        query_size: tuple[int, int],
    ) -> np.ndarray:
        depth_h, depth_w = depth_shape
        query_w, query_h = query_size
        sx = query_w / max(depth_w, 1)
        sy = query_h / max(depth_h, 1)
        k = intrinsic.astype(np.float64).copy()
        k[0, 0] *= sx
        k[0, 2] *= sx
        k[1, 1] *= sy
        k[1, 2] *= sy
        return k

    @classmethod
    def _candidate_camera_matrices(
        cls,
        intrinsic: np.ndarray,
        depth_shape: tuple[int, int],
        reference_size: tuple[int, int],
        query_size: tuple[int, int],
    ) -> list[tuple[str, np.ndarray]]:
        """Build intrinsics candidates for normal and 90-degree display rotation."""
        query_w, query_h = query_size
        reference_w, reference_h = reference_size

        candidates: list[tuple[str, np.ndarray]] = [
            (
                "none",
                cls._scale_intrinsics(
                    intrinsic,
                    depth_shape,
                    query_size,
                ),
            )
        ]

        reference_landscape = reference_w >= reference_h
        query_landscape = query_w >= query_h
        if reference_landscape == query_landscape:
            return candidates

        k_ref = cls._scale_intrinsics(
            intrinsic,
            depth_shape,
            reference_size,
        )
        fx = float(k_ref[0, 0])
        fy = float(k_ref[1, 1])
        cx = float(k_ref[0, 2])
        cy = float(k_ref[1, 2])

        rotated_w = max(float(reference_h), 1.0)
        rotated_h = max(float(reference_w), 1.0)
        sx = float(query_w) / rotated_w
        sy = float(query_h) / rotated_h

        clockwise = np.array([
            [fy * sx, 0.0, (reference_h - 1.0 - cy) * sx],
            [0.0, fx * sy, cx * sy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        counterclockwise = np.array([
            [fy * sx, 0.0, cy * sx],
            [0.0, fx * sy, (reference_w - 1.0 - cx) * sy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        candidates.extend([
            ("cw90", clockwise),
            ("ccw90", counterclockwise),
        ])
        return candidates

    @staticmethod
    def _camera_centers_from_extrinsics(
        extrinsics: np.ndarray,
    ) -> np.ndarray:
        centers = []
        for extrinsic in np.asarray(extrinsics):
            rotation = np.asarray(extrinsic[:3, :3], dtype=np.float64)
            translation = np.asarray(extrinsic[:3, 3], dtype=np.float64)
            center = -rotation.T @ translation
            if np.isfinite(center).all():
                centers.append(center)
        if not centers:
            return np.empty((0, 3), dtype=np.float64)
        return np.asarray(centers, dtype=np.float64)

    @staticmethod
    def _camera_position_plausible(
        camera_position: np.ndarray,
        reference_centers: np.ndarray,
    ) -> tuple[bool, float, float]:
        position = np.asarray(camera_position, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            return False, float("inf"), 0.0
        if len(reference_centers) == 0:
            return True, 0.0, 0.0

        distances = np.linalg.norm(
            reference_centers - position[None, :],
            axis=1,
        )
        nearest = float(np.min(distances))
        span = np.ptp(reference_centers, axis=0)
        path_diagonal = float(np.linalg.norm(span))
        max_distance = max(5.0, min(50.0, path_diagonal * 10.0))
        return nearest <= max_distance, nearest, max_distance

    @staticmethod
    def _reference_pixel_to_world(
        u: float,
        v: float,
        reference_size: tuple[int, int],
        depth_map: np.ndarray,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
    ) -> np.ndarray | None:
        ref_w, ref_h = reference_size
        depth_h, depth_w = depth_map.shape[-2:]
        du = u * depth_w / max(ref_w, 1)
        dv = v * depth_h / max(ref_h, 1)
        x = int(round(du))
        y = int(round(dv))
        if x < 0 or x >= depth_w or y < 0 or y >= depth_h:
            return None

        y1, y2 = max(0, y - 2), min(depth_h, y + 3)
        x1, x2 = max(0, x - 2), min(depth_w, x + 3)
        crop = depth_map[y1:y2, x1:x2]
        valid = crop[np.isfinite(crop) & (crop > 0)]
        if valid.size == 0:
            return None
        depth = float(np.median(valid))

        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        camera = np.array([
            (du - cx) / fx * depth,
            (dv - cy) / fy * depth,
            depth,
        ], dtype=np.float64)
        rotation = extrinsic[:3, :3]
        translation = extrinsic[:3, 3]
        return rotation.T @ (camera - translation)

    def _candidate_correspondences(
        self,
        query_keypoints,
        query_descriptors: np.ndarray,
        reference: ReferenceFeatures,
        depth_map: np.ndarray,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        if reference.descriptors is None or len(reference.descriptors) < 2:
            return np.empty((0, 3)), np.empty((0, 2)), 0

        pairs = self.matcher.knnMatch(
            reference.descriptors,
            query_descriptors,
            k=2,
        )
        good = []
        for pair in pairs:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < self.ratio_test * second.distance:
                good.append(first)

        world_points: list[np.ndarray] = []
        image_points: list[tuple[float, float]] = []
        for match in good:
            ref_pt = self._keypoint_xy(reference.keypoints[match.queryIdx])
            query_pt = self._keypoint_xy(query_keypoints[match.trainIdx])
            world = self._reference_pixel_to_world(
                ref_pt[0],
                ref_pt[1],
                (reference.width, reference.height),
                depth_map,
                intrinsic,
                extrinsic,
            )
            if world is None or not np.isfinite(world).all():
                continue
            world_points.append(world)
            image_points.append(query_pt)

        if not world_points:
            return np.empty((0, 3)), np.empty((0, 2)), len(good)
        return (
            np.asarray(world_points, dtype=np.float64),
            np.asarray(image_points, dtype=np.float64),
            len(good),
        )

    @staticmethod
    def estimate_pose_from_correspondences(
        world_points: np.ndarray,
        image_points: np.ndarray,
        camera_matrix: np.ndarray,
        reprojection_error: float = 6.0,
        min_inliers: int = 8,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
        if len(world_points) < max(6, min_inliers):
            return None
        success, rvec, tvec, inlier_idx = cv2.solvePnPRansac(
            world_points,
            image_points,
            camera_matrix,
            None,
            flags=cv2.SOLVEPNP_EPNP,
            iterationsCount=200,
            reprojectionError=reprojection_error,
            confidence=0.999,
        )
        if not success or inlier_idx is None or len(inlier_idx) < min_inliers:
            return None

        inliers = inlier_idx.reshape(-1)
        object_inliers = world_points[inliers]
        image_inliers = image_points[inliers]
        cv2.solvePnP(
            object_inliers,
            image_inliers,
            camera_matrix,
            None,
            rvec,
            tvec,
            True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        projected, _ = cv2.projectPoints(
            object_inliers,
            rvec,
            tvec,
            camera_matrix,
            None,
        )
        projected = projected.reshape(-1, 2)
        error = float(
            np.mean(np.linalg.norm(projected - image_inliers, axis=1))
        )
        rotation, _ = cv2.Rodrigues(rvec)
        translation = tvec.reshape(3)
        return rotation, translation, inliers, error

    def relocalize(
        self,
        scene: Scene,
        image: Image.Image,
        max_reference_frames: int = 60,
        search_all_references: bool = True,
    ) -> PoseEstimate:
        if not scene.reconstruction_npz:
            raise ValueError("Scene has no reconstruction_npz")

        started = time.perf_counter()
        query_rgb = np.asarray(image.convert("RGB"))
        query_gray = cv2.cvtColor(query_rgb, cv2.COLOR_RGB2GRAY)
        query_keypoints, query_descriptors = self.sift.detectAndCompute(
            query_gray,
            None,
        )
        if query_descriptors is None or len(query_keypoints or []) < self.min_matches:
            raise ValueError("Not enough visual features in current camera image")

        data = load_reconstruction(scene.reconstruction_npz)
        depth = data["depth"]
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        intrinsics = data["intrinsics"]
        extrinsics = data["extrinsics"]
        reference_camera_centers = self._camera_centers_from_extrinsics(
            extrinsics
        )

        references = self.prepare(
            scene,
            max_reference_frames=max_reference_frames,
        )
        query_centroid = self._descriptor_centroid(query_descriptors)
        if query_centroid is not None:
            ranked = sorted(
                references,
                key=lambda reference: (
                    float(
                        np.dot(query_centroid, reference.descriptor_centroid)
                    )
                    if reference.descriptor_centroid is not None
                    else -1.0
                ),
                reverse=True,
            )
        else:
            ranked = list(references)

        candidate_limit = max(1, min(self.candidate_limit, len(ranked)))
        primary = ranked[:candidate_limit]
        remaining = ranked[candidate_limit:]
        batches = [primary]
        if remaining and search_all_references:
            batches.append(remaining)

        best: PoseEstimate | None = None
        references_tested = 0

        for batch in batches:
            for reference in batch:
                references_tested += 1
                frame_index = reference.frame_index
                if frame_index >= len(depth):
                    continue
                world_points, image_points, match_count = (
                    self._candidate_correspondences(
                        query_keypoints,
                        query_descriptors,
                        reference,
                        depth[frame_index],
                        intrinsics[frame_index],
                        extrinsics[frame_index],
                    )
                )
                if len(world_points) < self.min_matches:
                    continue

                matrix_candidates = self._candidate_camera_matrices(
                    intrinsics[frame_index],
                    depth[frame_index].shape[-2:],
                    (reference.width, reference.height),
                    (image.width, image.height),
                )
                for image_rotation, camera_matrix in matrix_candidates:
                    solved = self.estimate_pose_from_correspondences(
                        world_points,
                        image_points,
                        camera_matrix,
                        reprojection_error=self.ransac_reprojection_error,
                        min_inliers=self.min_inliers,
                    )
                    if solved is None:
                        continue

                    rotation, translation, inliers, error = solved
                    camera_position = -rotation.T @ translation
                    plausible, nearest_camera_distance, max_camera_distance = (
                        self._camera_position_plausible(
                            camera_position,
                            reference_camera_centers,
                        )
                    )
                    if not plausible:
                        continue

                    inlier_ratio = float(len(inliers)) / max(match_count, 1)
                    inlier_score = min(float(len(inliers)) / 80.0, 1.0)
                    ratio_score = min(inlier_ratio / 0.35, 1.0)
                    reprojection_score = math.exp(-error / 3.0)
                    confidence = (
                        0.35 * inlier_score
                        + 0.35 * ratio_score
                        + 0.30 * reprojection_score
                    )
                    if confidence < self.min_pose_confidence:
                        continue

                    candidate = PoseEstimate(
                        reference_frame=frame_index,
                        rotation_world_to_camera=rotation,
                        translation_world_to_camera=translation,
                        camera_position_world=Vec3(
                            x=float(camera_position[0]),
                            y=float(camera_position[1]),
                            z=float(camera_position[2]),
                        ),
                        inliers=int(len(inliers)),
                        matches=int(match_count),
                        reprojection_error=error,
                        image_width=image.width,
                        image_height=image.height,
                        camera_matrix=camera_matrix,
                        seconds=time.perf_counter() - started,
                        inlier_ratio=inlier_ratio,
                        confidence=confidence,
                        references_tested=references_tested,
                        image_rotation=image_rotation,
                        nearest_reference_camera_distance=nearest_camera_distance,
                        max_plausible_camera_distance=max_camera_distance,
                    )

                    if best is None:
                        best = candidate
                        continue
                    current_rank = (
                        candidate.confidence,
                        candidate.inliers,
                        -candidate.reprojection_error,
                        -candidate.nearest_reference_camera_distance,
                    )
                    best_rank = (
                        best.confidence,
                        best.inliers,
                        -best.reprojection_error,
                        -best.nearest_reference_camera_distance,
                    )
                    if current_rank > best_rank:
                        best = candidate

            if best is not None:
                break

        if best is None:
            raise ValueError("Could not relocalize camera in this Scene")
        best.seconds = time.perf_counter() - started
        best.references_tested = references_tested
        return best



@dataclass
class _PoseSmoothingState:
    camera_position: np.ndarray
    quaternion: np.ndarray
    updated_at: float
    pose: PoseEstimate


class PoseSmoother:
    def __init__(
        self,
        position_alpha: float = 0.45,
        rotation_alpha: float = 0.35,
        reset_after_seconds: float = 3.0,
    ):
        self.position_alpha = position_alpha
        self.rotation_alpha = rotation_alpha
        self.reset_after_seconds = reset_after_seconds
        self._states: dict[str, _PoseSmoothingState] = {}
        self._lock = threading.Lock()

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._states.clear()
            else:
                self._states.pop(key, None)

    def recent_pose(
        self,
        key: str,
        max_age_seconds: float = 1.5,
    ) -> tuple[PoseEstimate, float] | None:
        now = time.monotonic()
        with self._lock:
            state = self._states.get(key)
            if state is None:
                return None
            age = now - state.updated_at
            if age > max(0.05, float(max_age_seconds)):
                return None
            return state.pose, float(age)

    def refresh_recent_pose(
        self,
        key: str,
    ) -> bool:
        now = time.monotonic()
        with self._lock:
            state = self._states.get(key)
            if state is None:
                return False
            state.updated_at = now
            return True

    def update(
        self,
        key: str,
        pose: PoseEstimate,
    ) -> tuple[PoseEstimate, dict]:
        now = time.monotonic()
        raw_position = np.array(
            [
                pose.camera_position_world.x,
                pose.camera_position_world.y,
                pose.camera_position_world.z,
            ],
            dtype=np.float64,
        )
        raw_quaternion = Rotation.from_matrix(
            pose.rotation_world_to_camera
        ).as_quat()

        with self._lock:
            stale_after = max(self.reset_after_seconds * 10.0, 30.0)
            for stale_key, stale_state in list(self._states.items()):
                if now - stale_state.updated_at > stale_after:
                    self._states.pop(stale_key, None)

            previous = self._states.get(key)
            if (
                previous is None
                or now - previous.updated_at > self.reset_after_seconds
            ):
                state = _PoseSmoothingState(
                    camera_position=raw_position,
                    quaternion=raw_quaternion,
                    updated_at=now,
                    pose=pose,
                )
                self._states[key] = state
                return pose, {
                    "applied": False,
                    "position_alpha": 1.0,
                    "rotation_alpha": 1.0,
                }

            confidence_factor = 0.5 + 0.5 * max(
                0.0,
                min(1.0, pose.confidence),
            )
            position_alpha = min(
                1.0,
                self.position_alpha * confidence_factor,
            )
            rotation_alpha = min(
                1.0,
                self.rotation_alpha * confidence_factor,
            )

            smoothed_position = (
                (1.0 - position_alpha) * previous.camera_position
                + position_alpha * raw_position
            )

            previous_quaternion = previous.quaternion
            if float(np.dot(previous_quaternion, raw_quaternion)) < 0:
                raw_quaternion = -raw_quaternion
            smoothed_quaternion = (
                (1.0 - rotation_alpha) * previous_quaternion
                + rotation_alpha * raw_quaternion
            )
            norm = float(np.linalg.norm(smoothed_quaternion))
            if norm <= 1e-8:
                smoothed_quaternion = raw_quaternion
            else:
                smoothed_quaternion /= norm

            smoothed_rotation = Rotation.from_quat(
                smoothed_quaternion
            ).as_matrix()
            smoothed_translation = -smoothed_rotation @ smoothed_position

            smoothed_pose = PoseEstimate(
                reference_frame=pose.reference_frame,
                rotation_world_to_camera=smoothed_rotation,
                translation_world_to_camera=smoothed_translation,
                camera_position_world=Vec3(
                    x=float(smoothed_position[0]),
                    y=float(smoothed_position[1]),
                    z=float(smoothed_position[2]),
                ),
                inliers=pose.inliers,
                matches=pose.matches,
                reprojection_error=pose.reprojection_error,
                image_width=pose.image_width,
                image_height=pose.image_height,
                camera_matrix=pose.camera_matrix,
                seconds=pose.seconds,
                inlier_ratio=pose.inlier_ratio,
                confidence=pose.confidence,
                references_tested=pose.references_tested,
                image_rotation=pose.image_rotation,
                nearest_reference_camera_distance=pose.nearest_reference_camera_distance,
                max_plausible_camera_distance=pose.max_plausible_camera_distance,
            )
            self._states[key] = _PoseSmoothingState(
                camera_position=smoothed_position,
                quaternion=smoothed_quaternion,
                updated_at=now,
                pose=smoothed_pose,
            )
        return smoothed_pose, {
            "applied": True,
            "position_alpha": position_alpha,
            "rotation_alpha": rotation_alpha,
            "raw_to_smoothed_position_delta": float(
                np.linalg.norm(raw_position - smoothed_position)
            ),
        }


def target_guidance(
    pose: PoseEstimate,
    target: SpatialEntity,
) -> dict:
    point_world = np.array([
        target.position.x,
        target.position.y,
        target.position.z,
    ], dtype=np.float64)
    camera_point = (
        pose.rotation_world_to_camera @ point_world
        + pose.translation_world_to_camera
    )
    x, y, z = (float(v) for v in camera_point)

    k = pose.camera_matrix
    width = pose.image_width
    height = pose.image_height

    if abs(z) > 1e-8:
        u = float(k[0, 0] * x / z + k[0, 2])
        v = float(k[1, 1] * y / z + k[1, 2])
    else:
        u = width / 2
        v = height / 2

    in_front = z > 0
    on_screen = (
        in_front
        and 0 <= u < width
        and 0 <= v < height
    )

    cx = width / 2
    cy = height / 2
    dx = u - cx
    dy = v - cy

    if not in_front:
        dx = -x
        dy = y
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            dx = 1.0

    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        arrow = {"x": 0.0, "y": 0.0}
    else:
        arrow = {"x": float(dx / norm), "y": float(dy / norm)}

    distance = math.sqrt(
        (target.position.x - pose.camera_position_world.x) ** 2
        + (target.position.y - pose.camera_position_world.y) ** 2
        + (target.position.z - pose.camera_position_world.z) ** 2
    )

    return {
        "camera_vector": {"x": x, "y": y, "z": z},
        "screen_target": {"x": u, "y": v},
        "normalized_screen_target": {
            "x": u / max(width, 1),
            "y": v / max(height, 1),
        },
        "in_front": in_front,
        "on_screen": on_screen,
        "arrow": arrow,
        "distance": float(distance),
    }


def pose_to_dict(pose: PoseEstimate) -> dict:
    return {
        "reference_frame": pose.reference_frame,
        "camera_position_world": pose.camera_position_world.model_dump(),
        "rotation_world_to_camera": pose.rotation_world_to_camera.tolist(),
        "translation_world_to_camera": pose.translation_world_to_camera.tolist(),
        "camera_matrix": pose.camera_matrix.tolist(),
        "inliers": pose.inliers,
        "matches": pose.matches,
        "inlier_ratio": pose.inlier_ratio,
        "confidence": pose.confidence,
        "reprojection_error": pose.reprojection_error,
        "references_tested": pose.references_tested,
        "image_rotation": pose.image_rotation,
        "nearest_reference_camera_distance": pose.nearest_reference_camera_distance,
        "max_plausible_camera_distance": pose.max_plausible_camera_distance,
        "image_width": pose.image_width,
        "image_height": pose.image_height,
        "seconds": pose.seconds,
    }
