"""Unified config schema (dataclasses + YAML loader)."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml


@dataclasses.dataclass
class SamplerCfg:
    backend: str
    checkpoint: str
    num_samples: int = 1
    num_steps: int = 100
    sigma_min: float = 0.003
    sigma_max: float = 10.0
    zeta: float = 0.3
    pad: int = 24
    psize: int = 56
    image_size: int = 256
    image_channels: int = 1
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class DatasetCfg:
    id_npy: str | None = None
    ood_npy: str = ""


@dataclasses.dataclass
class ForwardOpCfg:
    name: str
    views: int = 24
    blursize: int = 21
    blursigma: float = 9.0
    sigma_y: float = 0.0


@dataclasses.dataclass
class KlipCfg:
    block_size: int = 2
    t_start: int = 65
    t_end: int = 85
    sigma_power: float = 0.5


@dataclasses.dataclass
class OutputCfg:
    root: str = "./output"
    artifacts_subdir: str = "artifacts"
    scores_subdir: str = "scores"


@dataclasses.dataclass
class Config:
    name: str
    task: str
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
