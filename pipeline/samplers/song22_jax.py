"""song22 (predictor-corrector, JAX) sampler wrapper."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from ..io.artifact import Artifact
from ..io.config import Config
from ..io.datasets import DatasetSplit, load_split

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_temp_split(split: DatasetSplit, indices: list[int], orig_npy: Path) -> Path:
    fd, tmp = tempfile.mkstemp(suffix=".npy", prefix="song22_input_")
    os.close(fd)
    sliced = split.slice(indices)
    payload = {"imgs": sliced.imgs}
    if sliced.masks is not None:
        payload["masks"] = sliced.masks
    if sliced.labels is not None:
        payload["labels"] = sliced.labels
    np.save(tmp, payload, allow_pickle=True)
    return Path(tmp)


def run(cfg: Config, indices: list[int], split: DatasetSplit, *, conda_env: str = "song22") -> dict[int, Path]:
    out_dir = cfg.artifacts_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    song22_config_rel = cfg.sampler.extra.get("song22_config")
    if not song22_config_rel:
        raise ValueError("sampler.extra.song22_config must point at a song22 ml_collections config (e.g. configs/ve/chaos_eval_image.py)")
    song22_config = (REPO_ROOT / "song22" / song22_config_rel).resolve()
    ckpt = Path(cfg.sampler.checkpoint)
    if not ckpt.is_absolute():
        ckpt = (REPO_ROOT / ckpt).resolve()

    temp_input = _write_temp_split(split, indices, Path(cfg.dataset.ood_npy))
    print(f"[song22] temp input = {temp_input}")

    cmd = (
        f"source /usr/scratch/jhong392/src_conda.sh && conda activate {conda_env} && "
        f"cd {REPO_ROOT / 'song22'} && python3 _pipeline_sample.py "
        f"--config={song22_config} "
        f"--ckpt={ckpt} "
        f"--image_npy={temp_input} "
        f"--indices={','.join(str(i) for i in indices)} "
        f"--out_dir={out_dir.resolve()}"
    )
    print(f"[song22] launching: {cmd[:200]}...")
    rc = subprocess.run(["bash", "-c", cmd], stdout=sys.stdout, stderr=sys.stderr).returncode
    if rc != 0:
        raise RuntimeError(f"song22 sampler exited with rc={rc}")

    temp_input.unlink()
    return {i: (out_dir / f"img_{i:04d}.npz").resolve() for i in indices}


def load_split_for(cfg: Config, kind: str) -> DatasetSplit:
    if kind == "ood":
        return load_split(cfg.dataset.ood_npy)
    if kind == "id":
        if cfg.dataset.id_npy is None:
            raise ValueError("config.dataset.id_npy required for kind='id'")
        return load_split(cfg.dataset.id_npy)
    raise ValueError(f"unknown kind {kind!r}")
