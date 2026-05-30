"""Block + timestep aggregation of the normalized updates.

Implements Equation 12 of the paper:

    KLIP(B_i, [t0, t1]; y)
        = (1/2) ∫_{t0}^{t1} E_{x ~ p_t(x|y)} [|| g(t) * s_l(x, y; t) ||²_{B_i, 2}] dt

Discretized as: square the per-step normalized updates (which already equal
g(t)·s_l), restrict to the time window [t0, t1], block-reshape spatially,
average over (samples × time × within-block).

This is the single-shape computation that song22, PaDIS, and CelebA all
implement separately with their own indexing tricks. Here it lives once.
"""
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
    """Returns the per-block KLIP score map for a single image.

    Args:
        artifact:    Canonical (T, B, C, H, W) of normalized updates.
        block_size:  D_B in the paper; 1 = pixel-level.
        t_start:     t_0, inclusive index into stored timesteps.
        t_end:       t_1, exclusive index into stored timesteps.

    Returns:
        (H // block_size, W // block_size) float32 — the KLIP heatmap.
    """
    upd = artifact.normalized_updates  # (T, B, C, H, W)
    T, B, C, H, W = upd.shape

    if H % block_size != 0 or W % block_size != 0:
        raise ValueError(
            f"image size ({H}, {W}) not divisible by block_size={block_size}"
        )
    if not (0 <= t_start < t_end <= T):
        raise ValueError(f"invalid window [{t_start}, {t_end}) for T={T}")

    Hb, Wb = H // block_size, W // block_size

    # Paper Eq. 12: (1/2) ∫ E_x [||g s_l||²_{B_i, 2}] dt
    #   ||·||²_2 over the block B_i is a SUM over within-block coords and channels.
    #   E_x is a Monte Carlo MEAN over the B posterior samples.
    #   The time integral discretizes as MEAN over the (t_start, t_end) window.
    # Matches CT/Klip_PaDIS.ipynb bit-for-bit when C=1, B=1. Constant 1/2 is
    # dropped (irrelevant for AUROC).
    windowed = upd[t_start:t_end]  # (Tw, B, C, H, W)
    blocked = windowed.reshape(t_end - t_start, B, C, Hb, block_size, Wb, block_size)
    return (
        (blocked ** 2)
        .sum(axis=(2, 4, 6))       # ||·||²_2 over channels + within-block
        .mean(axis=(0, 1))         # E_x + ∫dt (discretized)
        .astype(np.float32)
    )
