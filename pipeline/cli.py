"""Unified-pipeline command-line entry.

Stages:
  sample : run the configured sampler over the requested image indices,
           write canonical Artifact .npz files to {output}/{name}/artifacts/.
  score  : load artifacts, compute per-image KLIP score maps, save .npy under
           {output}/{name}/scores/.
  auroc  : compute dataset- and image-level AUROC from the score maps + masks.
  all    : sample -> score -> auroc.

Usage:
  python -m pipeline.cli --config pipeline/configs/chaos_padis_image.yaml \
                         --stage all \
                         --ood-indices 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow `python pipeline/cli.py ...` as well as `python -m pipeline.cli ...`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipeline.io.artifact import Artifact
    from pipeline.io.config import Config
    from pipeline.io.datasets import load_split
    from pipeline.samplers import ddpm_torch, padis_torch, song22_jax
    from pipeline.scoring.aggregate import klip_score_map
    from pipeline.scoring.auroc import dataset_level, image_level
else:
    from .io.artifact import Artifact
    from .io.config import Config
    from .io.datasets import load_split
    from .samplers import ddpm_torch, padis_torch, song22_jax
    from .scoring.aggregate import klip_score_map
    from .scoring.auroc import dataset_level, image_level


def parse_indices(spec: str, total: int) -> list[int]:
    if spec == "all":
        return list(range(total))
    out: list[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return out


_BACKENDS = {
    "padis_torch": padis_torch,
    "song22_jax": song22_jax,
    "ddpm_torch": ddpm_torch,
}


def _backend(cfg: Config):
    if cfg.sampler.backend not in _BACKENDS:
        raise NotImplementedError(f"sampler backend {cfg.sampler.backend!r} not wired in CLI yet")
    return _BACKENDS[cfg.sampler.backend]


def stage_sample(cfg: Config, ood_idxs: list[int], id_idxs: list[int]) -> None:
    print(f"[stage sample] OOD indices {ood_idxs}, ID indices {id_idxs}")
    backend = _backend(cfg)
    ood = backend.load_split_for(cfg, "ood")
    backend.run(cfg, ood_idxs, ood)
    if cfg.task == "dataset" and id_idxs:
        id_ = backend.load_split_for(cfg, "id")
        original_subdir = cfg.output.artifacts_subdir
        cfg.output.artifacts_subdir = "artifacts_id"
        try:
            backend.run(cfg, id_idxs, id_)
        finally:
            cfg.output.artifacts_subdir = original_subdir


def _score_one(cfg: Config, artifact: Artifact) -> np.ndarray:
    return klip_score_map(
        artifact,
        block_size=cfg.klip.block_size,
        t_start=cfg.klip.t_start,
        t_end=cfg.klip.t_end,
    )


def stage_score(cfg: Config, ood_idxs: list[int], id_idxs: list[int]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    print(f"[stage score] computing KLIP score maps")
    scores_dir = cfg.scores_dir()
    scores_dir.mkdir(parents=True, exist_ok=True)

    ood_scores: dict[int, np.ndarray] = {}
    for i in ood_idxs:
        art = Artifact.load(cfg.artifacts_dir() / f"img_{i:04d}.npz")
        sm = _score_one(cfg, art)
        np.save(scores_dir / f"ood_{i:04d}.npy", sm)
        ood_scores[i] = sm

    id_scores: dict[int, np.ndarray] = {}
    if cfg.task == "dataset":
        id_artifacts_dir = cfg.artifacts_dir().parent / "artifacts_id"
        for i in id_idxs:
            art = Artifact.load(id_artifacts_dir / f"img_{i:04d}.npz")
            sm = _score_one(cfg, art)
            np.save(scores_dir / f"id_{i:04d}.npy", sm)
            id_scores[i] = sm

    return ood_scores, id_scores


def stage_auroc(
    cfg: Config,
    ood_idxs: list[int],
    id_idxs: list[int],
    ood_scores: dict[int, np.ndarray],
    id_scores: dict[int, np.ndarray],
) -> dict:
    print(f"[stage auroc] computing AUROC ({cfg.task!r})")
    out: dict = {"task": cfg.task, "name": cfg.name}

    ood_split = load_split(cfg.dataset.ood_npy)
    # Dataset stored at sampler.image_size or a multiple thereof; decimate to sampler resolution.
    stride = ood_split.imgs.shape[1] // cfg.sampler.image_size
    ood_score_list = [ood_scores[i] for i in ood_idxs]
    if ood_split.labels is not None:
        # CHAOS convention: labels uint8/bool, OOD voxels have value > 1 (tumor) or True (star).
        ood_label_masks = [
            ((ood_split.labels[i, ::stride, ::stride] > 1).astype(np.uint8) * 255)
            if ood_split.labels[i].dtype == np.uint8
            else (ood_split.labels[i, ::stride, ::stride].astype(np.uint8) * 255)
            for i in ood_idxs
        ]
        ood_body_masks = [
            ((ood_split.masks[i, ::stride, ::stride] > 0).astype(np.uint8) * 255)
            for i in ood_idxs
        ] if ood_split.masks is not None else None
    else:
        # CelebA convention: no separate body mask; `masks` is the per-pixel OOD (scar) label.
        if ood_split.masks is None:
            raise ValueError("dataset has neither labels nor masks; cannot compute AUROC")
        ood_label_masks = [
            ((ood_split.masks[i, ::stride, ::stride] > 127).astype(np.uint8) * 255)
            for i in ood_idxs
        ]
        ood_body_masks = None

    image_res = image_level(
        ood_score_list, ood_label_masks, ood_body_masks, block_size=cfg.klip.block_size
    )
    out["image_level"] = {
        "mean_auroc": image_res.mean_auroc,
        "valid_count": image_res.valid_count,
        "per_image": image_res.per_image_auroc.tolist(),
        "ood_indices": ood_idxs,
    }
    print(f"  image-level mean AUROC = {image_res.mean_auroc:.6f} ({image_res.valid_count} valid)")

    if cfg.task == "dataset":
        id_score_list = [id_scores[i] for i in id_idxs]
        ds_res = dataset_level(id_score_list, ood_score_list)
        out["dataset_level"] = {
            "auroc": ds_res.auroc,
            "id_scores": ds_res.id_scores.tolist(),
            "ood_scores": ds_res.ood_scores.tolist(),
            "id_indices": id_idxs,
            "ood_indices": ood_idxs,
        }
        print(f"  dataset-level AUROC = {ds_res.auroc:.6f}")

    json_path = cfg.output_root() / "auroc.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(out, indent=2))
    print(f"  wrote {json_path.relative_to(Path.cwd()) if json_path.is_relative_to(Path.cwd()) else json_path}")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="path to .yaml")
    p.add_argument("--stage", default="all", choices=["sample", "score", "auroc", "all"])
    p.add_argument("--ood-indices", default="all",
                   help="comma-separated indices or ranges (e.g. '0,2-4'); 'all' = full set")
    p.add_argument("--id-indices", default="all",
                   help="only used when task='dataset'")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config)

    ood_split = load_split(cfg.dataset.ood_npy)
    ood_idxs = parse_indices(args.ood_indices, len(ood_split))
    id_idxs: list[int] = []
    if cfg.task == "dataset":
        id_split = load_split(cfg.dataset.id_npy)
        id_idxs = parse_indices(args.id_indices, len(id_split))

    if args.stage in ("sample", "all"):
        stage_sample(cfg, ood_idxs, id_idxs)
    ood_scores: dict[int, np.ndarray] = {}
    id_scores: dict[int, np.ndarray] = {}
    if args.stage in ("score", "all"):
        ood_scores, id_scores = stage_score(cfg, ood_idxs, id_idxs)
    elif args.stage == "auroc":
        scores_dir = cfg.scores_dir()
        ood_scores = {i: np.load(scores_dir / f"ood_{i:04d}.npy") for i in ood_idxs}
        if cfg.task == "dataset":
            id_scores = {i: np.load(scores_dir / f"id_{i:04d}.npy") for i in id_idxs}
    if args.stage in ("auroc", "all"):
        stage_auroc(cfg, ood_idxs, id_idxs, ood_scores, id_scores)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
