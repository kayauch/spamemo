import json
from pathlib import Path

import numpy as np

from .types import ReconstructionResult


def save_result(result: ReconstructionResult, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    npz_path = output / "reconstruction.npz"
    payload = {
        "depth": result.depth,
        "extrinsics": result.extrinsics,
        "intrinsics": result.intrinsics,
    }
    if result.confidence is not None:
        payload["confidence"] = result.confidence
    np.savez_compressed(npz_path, **payload)
    (output / "metrics.json").write_text(
        json.dumps(result.metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return npz_path
