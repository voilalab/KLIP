"""Canonical sampler-output artifact.

One `.npz` per (image, run) pair. Holds *already-normalized* per-timestep
likelihood-score proxies (delta_x_t / g(t)), which is what the KLIP integrand
squares. Samplers convert their native output into this schema at write time:

  - song22 (PC) stores diff/g already → write-through.
  - PaDIS (DPS) stores delta_x_t + sigma separately → divide by sigma^p at write.
  - CelebA (DPS+DDPM) stores delta_x_t + alphas_cumprod  → divide by sqrt(beta_t).

Once written, the scoring stage is framework-agnostic NumPy.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np


@dataclasses.dataclass(frozen=True)
class Artifact:
    """A single (image, sampler-run) trajectory in canonical form."""

    # (T, B, C, H, W) float32
    #   T = number of stored timesteps
    #   B = posterior-sample count averaged into this artifact (>=1)
    #   C = image channels (1 for CT, 3 for RGB)
    #   H, W = spatial resolution
    # Each value is delta_x_t / g(t) — i.e. the per-step likelihood-score proxy
    # already normalized by the SDE/DDPM diffusion coefficient.
    normalized_updates: np.ndarray

    # (T,) float32 — sigma or sqrt(beta_t) values per timestep, for diagnostics
    # only. The square root is already absorbed into normalized_updates; this
    # is kept so callers can recover the raw delta or apply a different
    # normalization without re-running the sampler.
    g_values: np.ndarray

    # Optional, useful for debugging only.
    source_image: np.ndarray | None = None    # (C, H, W) uint8
    reconstruction: np.ndarray | None = None  # (H, W) or (H, W, C) float32 — sampler's final mean image

    def __post_init__(self) -> None:
        if self.normalized_updates.ndim != 5:
            raise ValueError(
                f"normalized_updates must be (T, B, C, H, W); got shape {self.normalized_updates.shape}"
            )
        if self.g_values.shape != (self.normalized_updates.shape[0],):
            raise ValueError(
                f"g_values must be (T,) with T = {self.normalized_updates.shape[0]}; "
                f"got {self.g_values.shape}"
            )

    @property
    def num_timesteps(self) -> int:
        return self.normalized_updates.shape[0]

    @property
    def num_samples(self) -> int:
        return self.normalized_updates.shape[1]

    @property
    def num_channels(self) -> int:
        return self.normalized_updates.shape[2]

    @property
    def image_size(self) -> tuple[int, int]:
        return self.normalized_updates.shape[3], self.normalized_updates.shape[4]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        kw = dict(
            normalized_updates=self.normalized_updates.astype(np.float32, copy=False),
            g_values=self.g_values.astype(np.float32, copy=False),
        )
        if self.source_image is not None:
            kw["source_image"] = self.source_image
        if self.reconstruction is not None:
            kw["reconstruction"] = self.reconstruction
        np.savez_compressed(path, **kw)

    @classmethod
    def load(cls, path: str | Path) -> "Artifact":
        d = np.load(path, allow_pickle=False)
        return cls(
            normalized_updates=d["normalized_updates"],
            g_values=d["g_values"],
            source_image=d["source_image"] if "source_image" in d.files else None,
            reconstruction=d["reconstruction"] if "reconstruction" in d.files else None,
        )
