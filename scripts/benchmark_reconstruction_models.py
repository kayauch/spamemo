from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def gpu_used_mib() -> float | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        first = completed.stdout.strip().splitlines()[0]
        return float(first)
    except Exception:
        return None


def run_model(
    model: str,
    frames_dir: Path,
    output_dir: Path,
    max_frames: int,
    vggt_checkpoint: Path,
    vggt_omega_checkpoint: Path | None,
) -> dict[str, Any]:
    target = output_dir / model
    target.mkdir(parents=True, exist_ok=True)
    log_path = target / "worker.log"

    command = [
        sys.executable,
        "-m",
        "spatial_recon.benchmark_worker",
        "--model",
        model,
        "--frames-dir",
        str(frames_dir),
        "--output-dir",
        str(target),
        "--max-frames",
        str(max_frames),
    ]
    if model == "vggt":
        command += ["--vggt-checkpoint", str(vggt_checkpoint)]
    elif model == "vggt-omega":
        if vggt_omega_checkpoint is None or not vggt_omega_checkpoint.exists():
            return {
                "model": model,
                "status": "blocked",
                "reason": "VGGT-Omega checkpoint unavailable; gated Hugging Face access required.",
            }
        command += [
            "--vggt-omega-checkpoint",
            str(vggt_omega_checkpoint),
        ]

    baseline = gpu_used_mib()
    peak = baseline
    started = time.perf_counter()

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
        while process.poll() is None:
            current = gpu_used_mib()
            if current is not None:
                peak = current if peak is None else max(peak, current)
            time.sleep(0.2)

    wall_seconds = time.perf_counter() - started
    return_code = process.returncode
    metrics_path = target / "metrics.json"

    if return_code != 0 or not metrics_path.exists():
        tail = ""
        try:
            tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-40:])
        except Exception:
            pass
        return {
            "model": model,
            "status": "failed",
            "return_code": return_code,
            "wall_seconds": wall_seconds,
            "worker_log_tail": tail,
        }

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.update(
        {
            "model_key": model,
            "status": "ok",
            "wall_seconds": wall_seconds,
            "gpu_total_baseline_mib": baseline,
            "gpu_total_peak_mib": peak,
            "gpu_total_delta_mib": (
                peak - baseline
                if peak is not None and baseline is not None
                else None
            ),
            "worker_log": str(log_path),
        }
    )
    return metrics


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def report_markdown(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Reconstruction model benchmark",
        "",
        "All runnable models used the same source frame directory and frame cap.",
        "GPU total-memory measurements include already-running services such as the local Qwen LLM.",
        "",
        "| Model | Status | Inference s | Proc peak GiB | GPU total peak MiB | Depth valid | Reloc success | Reloc pos err / depth | Reproj px | 3D match err / depth | Point count |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        if item.get("status") != "ok":
            lines.append(
                f"| {item.get('model_key', item.get('model'))} | {item.get('status')} | - | - | - | - | - | - | - | - | - |"
            )
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("model_key", item.get("model"))),
                    "ok",
                    fmt(item.get("inference_seconds"), 3),
                    fmt(item.get("inference_peak_reserved_gib"), 3),
                    fmt(item.get("gpu_total_peak_mib"), 0),
                    fmt(item.get("depth_valid_fraction"), 4),
                    fmt(item.get("relocalization_success_rate"), 3),
                    fmt(item.get("relocalization_position_error_median_normalized"), 5),
                    fmt(item.get("relocalization_reprojection_error_median_px"), 3),
                    fmt(item.get("matched_world_median_normalized"), 5),
                    fmt(item.get("point_count"), 0),
                ]
            )
            + " |"
        )

    lines += ["", "## Blockers", ""]
    blockers = [item for item in results if item.get("status") != "ok"]
    if blockers:
        for item in blockers:
            lines.append(
                f"- **{item.get('model_key', item.get('model'))}**: {item.get('reason') or 'worker failed; see log'}"
            )
    else:
        lines.append("- None.")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=Path("/results/da3-video-smoke/frames"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/results/benchmarks/reconstruction"),
    )
    parser.add_argument("--max-frames", type=int, default=35)
    parser.add_argument(
        "--models",
        default="da3,vggt,vggt-omega",
        help="comma-separated model keys",
    )
    parser.add_argument(
        "--vggt-checkpoint",
        type=Path,
        default=Path("/results/checkpoints/vggt/model.pt"),
    )
    parser.add_argument(
        "--vggt-omega-checkpoint",
        type=Path,
        default=Path("/results/checkpoints/vggt-omega/vggt_omega_1b_512.pt"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    requested = [item.strip() for item in args.models.split(",") if item.strip()]
    results: list[dict[str, Any]] = []

    for model in requested:
        print(f"=== {model} ===", flush=True)
        result = run_model(
            model=model,
            frames_dir=args.frames_dir,
            output_dir=args.output_dir,
            max_frames=args.max_frames,
            vggt_checkpoint=args.vggt_checkpoint,
            vggt_omega_checkpoint=args.vggt_omega_checkpoint,
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    summary = {
        "frames_dir": str(args.frames_dir),
        "max_frames": args.max_frames,
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )
    (args.output_dir / "REPORT.md").write_text(
        report_markdown(results),
        encoding="utf-8",
    )
    print(f"Saved: {args.output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
