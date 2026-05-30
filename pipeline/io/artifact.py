"""Canonical sampler-output artifact."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np


@dataclasses.dataclass(frozen=True)
class Artifact:
    normalized_updates: np.ndarray
    g_values: np.ndarray
    source_image: np.ndarray | None = None
    reconstruction: np.ndarray | None = None

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
