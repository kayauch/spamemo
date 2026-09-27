from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import DA3Adapter, VGGTAdapter, VGGTOmegaAdapter
from .benchmark_metrics import (
    heldout_relocalization_consistency,
    matched_world_consistency,
    reconstruction_health,
)
from .geometry import save_point_cloud
from .io import save_result


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def image_paths(directory: Path, max_frames: int) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.suffix.lower() in IMAGE_EXTS
    )[:max_frames]


def make_adapter(args):
    if args.model == "da3":
        return DA3Adapter(model_id=args.da3_model)
    if args.model == "vggt":
        if not args.vggt_checkpoint:
            raise ValueError("--vggt-checkpoint is required")
        return VGGTAdapter(
            checkpoint=args.vggt_checkpoint,
            repo_path=args.vggt_repo,
        )
    if args.model == "vggt-omega":
        if not args.vggt_omega_checkpoint:
            raise ValueError("--vggt-omega-checkpoint is required")
        return VGGTOmegaAdapter(
            checkpoint=args.vggt_omega_checkpoint,
            image_resolution=args.vggt_omega_resolution,
        )
    raise ValueError(f"Unknown model: {args.model}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["da3", "vggt", "vggt-omega"])
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=35)
    parser.add_argument("--point-stride", type=int, default=4)
    parser.add_argument("--da3-model", default="depth-anything/DA3-LARGE-1.1")
    parser.add_argument("--vggt-checkpoint", type=Path)
    parser.add_argument("--vggt-repo", type=Path, default=Path("/workspace/third_party/vggt"))
    parser.add_argument("--vggt-omega-checkpoint", type=Path)
    parser.add_argument("--vggt-omega-resolution", type=int, default=512)
    args = parser.parse_args()

    frames = image_paths(args.frames_dir, args.max_frames)
    if len(frames) < 2:
        raise ValueError("At least two frames are required")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter = make_adapter(args)
    result = adapter.infer(frames)

    reconstruction_path = save_result(result, args.output_dir)
    point_count = save_point_cloud(
        result,
        args.output_dir / "scene.ply",
        stride=args.point_stride,
    )

    evaluation = {}
    evaluation.update(reconstruction_health(result))
    evaluation.update(matched_world_consistency(result, frames))
    evaluation.update(heldout_relocalization_consistency(result, frames))
    evaluation["point_count"] = int(point_count)
    evaluation["reconstruction_npz"] = str(reconstruction_path)

    result.metrics.update(evaluation)
    save_result(result, args.output_dir)
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )

    print(json.dumps(result.metrics, indent=2, ensure_ascii=False, allow_nan=True))


if __name__ == "__main__":
    main()
