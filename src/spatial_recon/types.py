from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ReconstructionResult:
    depth: np.ndarray
    extrinsics: np.ndarray
    intrinsics: np.ndarray
    confidence: np.ndarray | None = None
    images: np.ndarray | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def frame_count(self) -> int:
        return int(self.depth.shape[0])
