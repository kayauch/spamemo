import json
from pathlib import Path

import torch
import typer
from rich import print

from .adapters import DA3Adapter, VGGTAdapter, VGGTOmegaAdapter
from .geometry import save_point_cloud
from .io import save_result
from .video import extract_video_frames

app = typer.Typer(no_args_is_help=True)


def _images(input_path: Path, frame_dir: Path, fps: float, max_frames: int) -> list[Path]:
    if input_path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        return sorted(p for p in input_path.iterdir() if p.suffix.lower() in exts)[:max_frames]
    return extract_video_frames(input_path, frame_dir, fps, max_frames)


@app.command()
def preflight() -> None:
    info = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update({
            "gpu": props.name,
            "vram_gib": props.total_memory / 2**30,
            "compute_capability": f"{props.major}.{props.minor}",
        })
    print(json.dumps(info, indent=2))


@app.command()
def benchmark(
    model: str = typer.Option("da3", help="da3, vggt, or vggt-omega"),
    input_path: Path = typer.Option(..., exists=True),
    output_dir: Path = typer.Option(Path("/results/benchmark")),
    sample_fps: float = typer.Option(2.0),
    max_frames: int = typer.Option(60),
    point_stride: int = typer.Option(4),
    da3_model: str = typer.Option("depth-anything/DA3-LARGE-1.1"),
    vggt_checkpoint: Path | None = typer.Option(None),
    vggt_repo: Path = typer.Option(Path("/workspace/third_party/vggt")),
    vggt_omega_checkpoint: Path | None = typer.Option(None),
    vggt_omega_resolution: int = typer.Option(512),
) -> None:
    frames = _images(input_path, output_dir / "frames", sample_fps, max_frames)
    if len(frames) < 2:
        raise typer.BadParameter("At least two frames are required.")
    print(f"[cyan]Using {len(frames)} frames[/cyan]")
    if model == "da3":
        adapter = DA3Adapter(model_id=da3_model)
    elif model == "vggt":
        if vggt_checkpoint is None:
            raise typer.BadParameter("--vggt-checkpoint is required for VGGT.")
        adapter = VGGTAdapter(
            checkpoint=vggt_checkpoint,
            repo_path=vggt_repo,
        )
    elif model == "vggt-omega":
        if vggt_omega_checkpoint is None:
            raise typer.BadParameter(
                "--vggt-omega-checkpoint is required for VGGT-Omega."
            )
        adapter = VGGTOmegaAdapter(
            checkpoint=str(vggt_omega_checkpoint),
            image_resolution=vggt_omega_resolution,
        )
    else:
        raise typer.BadParameter(
            "--model must be da3, vggt, or vggt-omega."
        )

    result = adapter.infer(frames)
    save_result(result, output_dir)
    count = save_point_cloud(result, output_dir / "scene.ply", stride=point_stride)
    result.metrics["point_count"] = count
    save_result(result, output_dir)
    print(json.dumps(result.metrics, indent=2))
    print(f"[green]Saved reconstruction to {output_dir}[/green]")


if __name__ == "__main__":
    app()
