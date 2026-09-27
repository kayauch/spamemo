from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError


VGGT_REPO_URL = "https://github.com/facebookresearch/vggt.git"
VGGT_MODEL_REPO = "facebook/VGGT-1B"
VGGT_MODEL_FILE = "model.pt"
OMEGA_MODEL_REPO = "facebook/VGGT-Omega"
OMEGA_MODEL_FILE = "vggt_omega_1b_512.pt"


def ensure_git_repo(path: Path, url: str) -> None:
    if (path / ".git").exists():
        print(f"repo ready: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(path)],
        check=True,
    )
    print(f"cloned: {path}")


def ensure_hf_file(repo_id: str, filename: str, target_dir: Path) -> Path:
    target = target_dir / filename
    if target.exists() and target.stat().st_size > 0:
        print(f"checkpoint ready: {target}")
        return target

    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id,
        filename,
        local_dir=str(target_dir),
    )
    print(f"downloaded: {downloaded}")
    return Path(downloaded)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VGGT_REPO = Path("/workspace/third_party/vggt") if Path("/workspace/third_party/vggt").exists() else PROJECT_ROOT / "third_party" / "vggt"
DEFAULT_CHECKPOINT_ROOT = Path("/results/checkpoints") if Path("/results/checkpoints").exists() else PROJECT_ROOT / "checkpoints"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vggt-repo",
        type=Path,
        default=DEFAULT_VGGT_REPO,
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
    )
    parser.add_argument(
        "--omega",
        action="store_true",
        help="Also attempt the gated VGGT-Omega checkpoint.",
    )
    args = parser.parse_args()

    ensure_git_repo(args.vggt_repo, VGGT_REPO_URL)
    ensure_hf_file(
        VGGT_MODEL_REPO,
        VGGT_MODEL_FILE,
        args.checkpoint_root / "vggt",
    )

    if args.omega:
        try:
            ensure_hf_file(
                OMEGA_MODEL_REPO,
                OMEGA_MODEL_FILE,
                args.checkpoint_root / "vggt-omega",
            )
        except GatedRepoError as exc:
            print(
                "VGGT-Omega remains blocked: request model access on Hugging Face "
                "and configure HF authentication in the runtime."
            )
            print(type(exc).__name__)


if __name__ == "__main__":
    main()
