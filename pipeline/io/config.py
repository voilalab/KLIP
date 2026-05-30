"""Unified config schema (dataclasses + YAML loader).

One YAML config per (model × dataset × task) combination. Fields are grouped:
  - sampler: which backend, checkpoint path, sampler-specific hyperparams
  - dataset: paths to the unified chaos_*.npy / celeba_*.npy
  - forward_op: which inverse problem (ct_parbeam, gaussian_blur, ...)
  - klip:    block_size, t_start, t_end, num_samples (Equation 12 knobs)
  - task:    'image' (per-block AUROC) or 'dataset' (max-over-blocks AUROC)
  - output:  where to put canonical artifacts + score arrays
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml


@dataclasses.dataclass
class SamplerCfg:
    backend: str                           # 'padis_torch' | 'song22_jax' | 'ddpm_torch'
    checkpoint: str                        # path or HF id
    num_samples: int = 1                   # posterior samples per image (B in artifact)
    num_steps: int = 100                   # diffusion sampling steps
    sigma_min: float = 0.003               # PaDIS EDM schedule
    sigma_max: float = 10.0
    zeta: float = 0.3                      # DPS step size
    pad: int = 24                          # PaDIS patch padding
    psize: int = 56                        # PaDIS patch size
    image_size: int = 256                  # sampler operating resolution
    image_channels: int = 1
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)  # backend-specific


@dataclasses.dataclass
class DatasetCfg:
    id_npy: str | None = None              # required if task='dataset'
    ood_npy: str = ""                      # required (the test OOD set)
    # If ood_npy points at chaos_ood_*.npy it carries imgs+masks+labels in one file.


@dataclasses.dataclass
class ForwardOpCfg:
    name: str                              # 'ct_parbeam' | 'ct_fanbeam' | 'gaussian_blur'
    views: int = 24                        # CT: number of projection views
    blursize: int = 21                     # Gaussian: kernel size
    blursigma: float = 9.0                 # Gaussian: kernel sigma
    sigma_y: float = 0.0                   # measurement noise


@dataclasses.dataclass
class KlipCfg:
    block_size: int = 2                    # D_B in the paper (1 = pixel-level)
    t_start: int = 65                      # t_0 (inclusive, indexed in stored timesteps)
    t_end: int = 85                        # t_1 (exclusive)
    sigma_power: float = 0.5               # Only for PaDIS-style normalization; ignored elsewhere


@dataclasses.dataclass
class OutputCfg:
    root: str = "./output"                 # where to write {artifacts/, scores/, auroc.json}
    artifacts_subdir: str = "artifacts"
    scores_subdir: str = "scores"


@dataclasses.dataclass
class Config:
    name: str                              # config identifier (slug)
    task: str                              # 'image' | 'dataset'
    sampler: SamplerCfg
    dataset: DatasetCfg
    forward_op: ForwardOpCfg
    klip: KlipCfg
    output: OutputCfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            name=raw["name"],
            task=raw["task"],
            sampler=SamplerCfg(**raw["sampler"]),
            dataset=DatasetCfg(**raw["dataset"]),
            forward_op=ForwardOpCfg(**raw["forward_op"]),
            klip=KlipCfg(**raw["klip"]),
            output=OutputCfg(**raw.get("output", {})),
        )

    def output_root(self) -> Path:
        return Path(self.output.root) / self.name

    def artifacts_dir(self) -> Path:
        return self.output_root() / self.output.artifacts_subdir

    def scores_dir(self) -> Path:
        return self.output_root() / self.output.scores_subdir
