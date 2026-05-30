"""AUROC computation — both dataset-level and image-level.

Dataset-level (Table 1 main results):
    - Each image is scored by the MAX KLIP across its blocks.
    - ROC over (ID, OOD) image scores.

Image-level (Table 1):
    - Each image gets a per-image ROC over its blocks, masked to body.
    - Final number is the mean AUROC across the OOD image pool.
"""
from __future__ import annotations

import dataclasses
from typing import Sequence

import numpy as np
from sklearn.metrics import auc, roc_curve


def _downsample_mask_to_blocks(mask: np.ndarray, block_size: int) -> np.ndarray:
    """Reduce a pixel-level binary mask to one cell per block via 'any pixel set'."""
    H, W = mask.shape
    if H % block_size != 0 or W % block_size != 0:
        raise ValueError(f"mask shape {mask.shape} not divisible by block_size={block_size}")
    return (
        (mask > 0).reshape(H // block_size, block_size, W // block_size, block_size).any(axis=(1, 3))
    )


@dataclasses.dataclass(frozen=True)
class DatasetLevelResult:
    auroc: float
    id_scores: np.ndarray                  # (N_id,)
    ood_scores: np.ndarray                 # (N_ood,)


@dataclasses.dataclass(frozen=True)
class ImageLevelResult:
    mean_auroc: float
    per_image_auroc: np.ndarray            # (N_ood,) — NaN where AUROC undefined
    valid_count: int                       # how many images had both classes in the mask


def dataset_level(
    id_score_maps: Sequence[np.ndarray],
    ood_score_maps: Sequence[np.ndarray],
) -> DatasetLevelResult:
    """One AUROC across the full (ID, OOD) image pool, score = max-over-blocks."""
    id_arr  = np.array([sm.max() for sm in id_score_maps],  dtype=np.float64)
    ood_arr = np.array([sm.max() for sm in ood_score_maps], dtype=np.float64)
    y_true = np.concatenate([np.zeros(len(id_arr)), np.ones(len(ood_arr))])
    y_score = np.concatenate([id_arr, ood_arr])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return DatasetLevelResult(auroc=float(auc(fpr, tpr)), id_scores=id_arr, ood_scores=ood_arr)


def image_level(
    ood_score_maps: Sequence[np.ndarray],
    ood_label_masks: Sequence[np.ndarray],   # (H, W) pixel-level: 1 = OOD pixel
    ood_body_masks: Sequence[np.ndarray] | None,  # (H, W) pixel-level; None → full image
    *,
    block_size: int,
) -> ImageLevelResult:
    """Per-image AUROC over body-masked blocks, averaged across OOD images."""
    if len(ood_score_maps) != len(ood_label_masks):
        raise ValueError("ood_score_maps and ood_label_masks must align")

    per_image: list[float] = []
    for i, score_map in enumerate(ood_score_maps):
        label_blocks = _downsample_mask_to_blocks(ood_label_masks[i], block_size)
        if ood_body_masks is not None:
            body_blocks = _downsample_mask_to_blocks(ood_body_masks[i], block_size)
        else:
            body_blocks = np.ones_like(label_blocks, dtype=bool)
        scores = score_map[body_blocks]
        labels = label_blocks[body_blocks]
        if len(np.unique(labels)) < 2:
            per_image.append(float("nan"))
            continue
        fpr, tpr, _ = roc_curve(labels.astype(int), scores)
        per_image.append(float(auc(fpr, tpr)))

    arr = np.array(per_image, dtype=np.float64)
    valid = arr[~np.isnan(arr)]
    return ImageLevelResult(
        mean_auroc=float(np.mean(valid)) if len(valid) else float("nan"),
        per_image_auroc=arr,
        valid_count=int(len(valid)),
    )
