"""Helpers for reading the unified chaos_*.npy / celeba_*.npy files.

Schema (all `.npy` files are pickled dicts):
    imgs    (N, H, W) uint8         OR (N, H, W, 3) for CelebA
    masks   (N, H, W) uint8         body / foreground / scar mask
    labels  (N, H, W) uint8 or bool OOD label (e.g. tumor voxels = 2)
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np


@dataclasses.dataclass
class DatasetSplit:
    imgs: np.ndarray                       # (N, H, W) or (N, H, W, 3) uint8
    masks: np.ndarray | None = None        # (N, H, W) uint8 — body mask
    labels: np.ndarray | None = None       # (N, H, W) uint8 / bool — OOD label

    def __len__(self) -> int:
        return self.imgs.shape[0]

    def slice(self, idx: slice | list[int]) -> "DatasetSplit":
        return DatasetSplit(
            imgs=self.imgs[idx],
            masks=self.masks[idx] if self.masks is not None else None,
            labels=self.labels[idx] if self.labels is not None else None,
        )


def load_split(path: str | Path) -> DatasetSplit:
    d = np.load(path, allow_pickle=True).item()
    if "imgs" not in d:
        raise ValueError(f"{path}: expected dict with key 'imgs', got keys {list(d)}")
    return DatasetSplit(
        imgs=d["imgs"],
        masks=d.get("masks"),
        labels=d.get("labels"),
    )
