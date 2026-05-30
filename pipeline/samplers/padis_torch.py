"""PaDIS sampler wrapper around CT/dps_sampling_test.py."""
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


def _write_temp_split(split: DatasetSplit, indices: list[int]) -> Path:
    sliced = split.slice(indices)
    fd, tmp = tempfile.mkstemp(suffix=".npy", prefix="padis_input_")
    os.close(fd)
    np.save(tmp, {"imgs": sliced.imgs}, allow_pickle=True)
    return Path(tmp)


def _convert_raw_to_artifact(
    raw_path: Path,
    sigma_path: Path,
    recon_path: Path | None,
    sigma_power: float,
) -> Artifact:
    raw = np.load(raw_path)
    if raw.ndim != 4:
        raise ValueError(f"{raw_path}: expected (T, C, H, W); got {raw.shape}")
    sigma = np.load(sigma_path)
    if sigma.shape != (raw.shape[0],):
        raise ValueError(f"sigma shape {sigma.shape} != raw T={raw.shape[0]}")
    g = sigma ** sigma_power
    normalized = raw / g.reshape(-1, 1, 1, 1)
    recon = None
    if recon_path is not None and recon_path.exists():
        recon = np.clip(np.load(recon_path), 0, 1).astype(np.float32)
    return Artifact(
        normalized_updates=normalized[:, None, ...].astype(np.float32),
        g_values=g.astype(np.float32),
        reconstruction=recon,
    )


def run(cfg: Config, indices: list[int], split: DatasetSplit, *, conda_env: str = "song22") -> dict[int, Path]:
    out_dir = cfg.artifacts_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    temp_input = _write_temp_split(split, indices)
    temp_outdir = Path(tempfile.mkdtemp(prefix="padis_output_"))
    print(f"[padis] temp input = {temp_input}, temp outdir = {temp_outdir}")

    network = Path(cfg.sampler.checkpoint)
    if not network.is_absolute():
        network = (REPO_ROOT / network).resolve()

    cmd = (
        f"source /usr/scratch/jhong392/src_conda.sh && conda activate {conda_env} && "
        f"cd {REPO_ROOT / 'CT'} && python3 dps_sampling_test.py "
        f"--network={network} "
        f"--outdir={temp_outdir} "
        f"--image_npy={temp_input} "
        f"--image_size={cfg.sampler.image_size} "
        f"--views={cfg.forward_op.views} "
        f"--name={cfg.forward_op.name} "
        f"--steps={cfg.sampler.num_steps} "
        f"--sigma_min={cfg.sampler.sigma_min} "
        f"--sigma_max={cfg.sampler.sigma_max} "
        f"--zeta={cfg.sampler.zeta} "
        f"--pad={cfg.sampler.pad} "
        f"--num_runs={cfg.sampler.num_samples} "
        f"--psize={cfg.sampler.psize}"
    )
    print(f"[padis] launching: {cmd[:200]}...")
    rc = subprocess.run(["bash", "-c", cmd], stdout=sys.stdout, stderr=sys.stderr).returncode
    if rc != 0:
        raise RuntimeError(f"PaDIS sampler exited with rc={rc}")

    written: dict[int, Path] = {}
    for local_i, original_i in enumerate(indices):
        raw_path = temp_outdir / f"img_{local_i:04d}_mean_measurement_updates.npy"
        sig_path = temp_outdir / f"img_{local_i:04d}_sigma_values.npy"
        recon_path = temp_outdir / f"img_{local_i:04d}_mean_recon.npy"
        if not raw_path.exists() or not sig_path.exists():
            raise FileNotFoundError(f"missing PaDIS output: {raw_path} / {sig_path}")
        artifact = _convert_raw_to_artifact(raw_path, sig_path, recon_path, cfg.klip.sigma_power)
        dst = (out_dir / f"img_{original_i:04d}.npz").resolve()
        artifact.save(dst)
        written[original_i] = dst
        try:
            shown = dst.relative_to(REPO_ROOT)
        except ValueError:
            shown = dst
        print(f"[padis]  img {original_i:>4d}: wrote {shown}")

    temp_input.unlink()
    return written


def load_artifacts(cfg: Config, indices: list[int]) -> dict[int, Artifact]:
    return {
        i: Artifact.load(cfg.artifacts_dir() / f"img_{i:04d}.npz") for i in indices
    }


def load_split_for(cfg: Config, kind: str) -> DatasetSplit:
    if kind == "ood":
        return load_split(cfg.dataset.ood_npy)
    if kind == "id":
        if cfg.dataset.id_npy is None:
            raise ValueError("config.dataset.id_npy required for kind='id'")
        return load_split(cfg.dataset.id_npy)
    raise ValueError(f"unknown kind {kind!r}")
