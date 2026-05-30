"""Block + timestep aggregation of the normalized updates (Eq. 12)."""
from __future__ import annotations

import numpy as np

from ..io.artifact import Artifact


def klip_score_map(
    artifact: Artifact,
    *,
    block_size: int,
    t_start: int,
    t_end: int,
) -> np.ndarray:
    upd = artifact.normalized_updates
    T, B, C, H, W = upd.shape

    if H % block_size != 0 or W % block_size != 0:
        raise ValueError(
            f"image size ({H}, {W}) not divisible by block_size={block_size}"
        )
    if not (0 <= t_start < t_end <= T):
        raise ValueError(f"invalid window [{t_start}, {t_end}) for T={T}")

    Hb, Wb = H // block_size, W // block_size
    windowed = upd[t_start:t_end]
    blocked = windowed.reshape(t_end - t_start, B, C, Hb, block_size, Wb, block_size)
    return (
        (blocked ** 2)
        .sum(axis=(2, 4, 6))
        .mean(axis=(0, 1))
        .astype(np.float32)
    )
