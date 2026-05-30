"""Confirms `pipeline.scoring` matches the original PaDIS-notebook recipe."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, roc_curve

sys.modules.setdefault("numpy._core", __import__("numpy").core)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.io.artifact import Artifact
from pipeline.io.datasets import load_split
from pipeline.scoring.aggregate import klip_score_map
from pipeline.scoring.auroc import image_level

ROOT = Path("/usr/scratch/jhong392/workspace/KLIP")
PADIS_OUT = Path("/tmp/padis_refactored_out")
BLOCK_SIZE = 2
T0, T1 = 65, 85
SIGMA_POWER = 0.5


def _notebook_compute_patch_energy_grid(artifact, sigma_values, ws, we, ps, sp=0.5):
    normalized = artifact[ws:we, 0] / (sigma_values[ws:we, None, None] ** sp)
    t, h, w = normalized.shape
    return (
        normalized.reshape(t, h // ps, ps, w // ps, ps) ** 2
    ).sum(axis=(0, 2, 4)) / t


def _notebook_downsample(mask, ps):
    h, w = mask.shape
    return (mask.reshape(h // ps, ps, w // ps, ps).mean(axis=(1, 3)) > 0)


def main() -> int:
    sigma_values = np.load(PADIS_OUT / "img_0000_sigma_values.npy")
    print(f"sigma_values: {sigma_values.shape}, range=[{sigma_values.min():.4f}, {sigma_values.max():.4f}]")
    mu_files = sorted(PADIS_OUT.glob("img_*_mean_measurement_updates.npy"))
    print(f"OOD artifact files: {len(mu_files)}")

    artifacts = {}
    raw_mu = {}
    for p in mu_files:
        mu = np.load(p)
        raw_mu[p.name.replace("_mean_measurement_updates.npy", "")] = mu
        normalized = mu / (sigma_values.reshape(-1, 1, 1, 1) ** SIGMA_POWER)
        art = Artifact(
            normalized_updates=normalized[:, None, ...].astype(np.float32),
            g_values=(sigma_values ** SIGMA_POWER).astype(np.float32),
        )
        artifacts[p.name.replace("_mean_measurement_updates.npy", "")] = art

    print("\n--- heatmap parity ---")
    for stem in list(artifacts)[:3]:
        ours = klip_score_map(artifacts[stem], block_size=BLOCK_SIZE, t_start=T0, t_end=T1)
        theirs = _notebook_compute_patch_energy_grid(
            raw_mu[stem], sigma_values, T0, T1, BLOCK_SIZE, SIGMA_POWER
        ).astype(np.float32)
        diff = np.abs(ours - theirs)
        print(
            f"  {stem}: shape={ours.shape} max|d|={diff.max():.3e} "
            f"mean|d|={diff.mean():.3e} bit-id={np.array_equal(ours, theirs)}"
        )

    print("\n--- image-level AUROC parity ---")
    ood = load_split(ROOT / "data/chaos_ood_tumor.npy")
    score_maps = []
    label_masks = []
    body_masks = []
    sample_idxs = []
    for stem, art in artifacts.items():
        i = int(stem.split("_")[1])
        sample_idxs.append(i)
        score_maps.append(klip_score_map(art, block_size=BLOCK_SIZE, t_start=T0, t_end=T1))
        label_masks.append((ood.labels[i, ::2, ::2] > 1).astype(np.uint8) * 255)
        body_masks.append((ood.masks[i, ::2, ::2] > 0).astype(np.uint8) * 255)

    result = image_level(
        ood_score_maps=score_maps,
        ood_label_masks=label_masks,
        ood_body_masks=body_masks,
        block_size=BLOCK_SIZE,
    )

    ref_aurocs = []
    for stem, score_map in zip(artifacts, score_maps):
        i = int(stem.split("_")[1])
        body_blocks = _notebook_downsample(ood.masks[i, ::2, ::2] > 0, BLOCK_SIZE)
        tumor_blocks = _notebook_downsample(ood.labels[i, ::2, ::2] > 1, BLOCK_SIZE)
        scores = score_map[body_blocks]
        labels = tumor_blocks[body_blocks]
        if len(np.unique(labels)) < 2:
            ref_aurocs.append(float("nan"))
            continue
        fpr, tpr, _ = roc_curve(labels.astype(int), scores)
        ref_aurocs.append(float(auc(fpr, tpr)))

    print(f"  pipeline  mean_auroc = {result.mean_auroc:.6f}  ({result.valid_count} valid)")
    print(f"  notebook  mean_auroc = {np.nanmean(ref_aurocs):.6f}")
    print(f"  per-image: pipeline={result.per_image_auroc}, notebook={ref_aurocs}")
    if not np.allclose(result.per_image_auroc, ref_aurocs, equal_nan=True):
        print("FAIL: per-image AUROC mismatch")
        return 1

    print("\nAll scoring + AUROC outputs match the notebook recipe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
