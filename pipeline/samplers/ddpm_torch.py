"""CelebA-HQ DPS + DDPM sampler (port of CelebA/celebA.ipynb).

In-process port — no shell-out — because there's no existing CLI to wrap.
Loads `google/ddpm-celebahq-256` (or any DDPMPipeline-compatible HF id), runs DPS
for Gaussian deblurring with measurement updates captured per timestep, and
writes the canonical Artifact .npz.

Normalization to canonical form: the notebook divides each measurement update
by sqrt(beta_t) before squaring. So normalized_updates = delta_x_t / sqrt(beta_t).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..io.artifact import Artifact
from ..io.config import Config
from ..io.datasets import DatasetSplit, load_split

REPO_ROOT = Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────────────────────────
# Blur operator (verbatim from CelebA/celebA.ipynb cell 7, trimmed).
def _make_gaussian_kernel(ksize: int, sigma: float, device, dtype):
    ax = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="xy")
    k = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return (k / k.sum())[None, None]


class BlurOp(nn.Module):
    """Depthwise Gaussian blur with circular padding. Symmetric: H^T = H."""

    def __init__(self, ksize: int, sigma: float):
        super().__init__()
        self.ksize = ksize
        self.sigma = sigma
        self._kernel = None

    def _get_kernel(self, x: torch.Tensor) -> torch.Tensor:
        if self._kernel is None or self._kernel.device != x.device or self._kernel.dtype != x.dtype:
            self._kernel = _make_gaussian_kernel(self.ksize, self.sigma, x.device, x.dtype)
        return self._kernel

    def H(self, x: torch.Tensor) -> torch.Tensor:
        k = self._get_kernel(x)
        C = x.shape[1]
        pad = self.ksize // 2
        x_pad = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="circular")
        return torch.nn.functional.conv2d(x_pad, k.expand(C, 1, -1, -1), groups=C)

    def Ht(self, x: torch.Tensor) -> torch.Tensor:
        return self.H(x)


# ─────────────────────────────────────────────────────────────────────────────
# DPS update (verbatim from notebook cell 9).
def _dps_update_xt(
    xt, t_idx, eps_t, alphas_cumprod, H, Ht, y, sigma_y, dps_scale, match_prior_norm
):
    a_bar = alphas_cumprod[t_idx]
    sqrt_a_bar = a_bar.sqrt().view(1, 1, 1, 1)
    sigma_t = (1.0 - a_bar).sqrt().view(1, 1, 1, 1)

    x0_hat = (xt - sigma_t * eps_t) / sqrt_a_bar
    resid = y - H(x0_hat)
    grad_x0 = Ht(resid) / (sigma_y ** 2 + 1e-12)
    grad_xt = grad_x0 / sqrt_a_bar

    if match_prior_norm:
        prior_norm = (eps_t / sigma_t).reshape(xt.shape[0], -1).norm(dim=1).mean()
        like_norm = grad_xt.reshape(xt.shape[0], -1).norm(dim=1).mean()
        zeta = dps_scale * (prior_norm.detach() / (like_norm.detach() + 1e-12))
    else:
        resid_norm = resid.reshape(xt.shape[0], -1).norm(dim=1).mean()
        zeta = dps_scale / (resid_norm.detach() + 1e-12)

    return xt + zeta * grad_xt


# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _sample_one(
    pipe,
    img_uint8: np.ndarray,                 # (H, W, 3) uint8
    *,
    num_inference_steps: int,
    num_samples: int,
    base_seed: int,
    blur_ksize_true: int,
    blur_sigma_true: float,
    blur_sigma_model: float,
    dps_scale: float,
    sigma_y: float,
) -> Artifact:
    """Run DPS-deblur on one image and return its canonical Artifact."""
    device = pipe.unet.device
    dtype = next(pipe.unet.parameters()).dtype

    # tensor in [-1, 1], shape (1, 3, H, W)
    x0 = torch.from_numpy(img_uint8).to(device=device, dtype=dtype) / 127.5 - 1.0
    x0 = x0.permute(2, 0, 1).unsqueeze(0)

    blur_true = BlurOp(blur_ksize_true, blur_sigma_true)
    blur_model = BlurOp(blur_ksize_true, blur_sigma_model)
    y = blur_true.H(x0).clamp(-1, 1)
    y_batch = y.expand(num_samples, -1, -1, -1).contiguous()

    # Initialize x_T with per-sample deterministic seeds.
    B, C, H, W = y_batch.shape
    xt = torch.empty_like(y_batch)
    for i in range(B):
        g = torch.Generator(device=device).manual_seed(base_seed + i)
        xt[i] = torch.randn((C, H, W), device=device, dtype=dtype, generator=g)

    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_inference_steps, device=device)
    betas = scheduler.betas.to(device)
    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    measurement_list: list[torch.Tensor] = []
    sqrt_betas: list[float] = []
    for t in scheduler.timesteps:
        eps = pipe.unet(xt, t).sample
        t_idx = int(t.item())
        x_before = xt.clone()
        xt = _dps_update_xt(
            xt, t_idx, eps, alphas_cumprod,
            blur_model.H, blur_model.Ht, y_batch,
            sigma_y, dps_scale, match_prior_norm=False,
        )
        measurement_list.append((xt - x_before).cpu().float())   # (B, C, H, W)
        sqrt_betas.append(float(math.sqrt(betas[t_idx].item())))
        # Scheduler diffusion step.
        gen = torch.Generator(device=device).manual_seed(base_seed + 99991)
        xt = scheduler.step(eps, t, xt, generator=gen).prev_sample

    # Stack to (T, B, C, H, W), normalize by sqrt(beta_t).
    stacked = torch.stack(measurement_list, dim=0).numpy().astype(np.float32)  # (T, B, C, H, W)
    g_values = np.array(sqrt_betas, dtype=np.float32)
    normalized = stacked / g_values.reshape(-1, 1, 1, 1, 1)
    # Final reconstruction: x_t after last scheduler step. Mean over the posterior batch,
    # rescaled from [-1, 1] back to [0, 1] for inspection.
    recon = ((xt.mean(dim=0).cpu().float() + 1.0) / 2.0).clamp(0, 1).permute(1, 2, 0).numpy().astype(np.float32)
    return Artifact(normalized_updates=normalized, g_values=g_values, reconstruction=recon)


def run(cfg: Config, indices: list[int], split: DatasetSplit, *, conda_env: str | None = None) -> dict[int, Path]:
    """In-process DPS-deblur on `split.imgs[indices]`. Writes canonical Artifacts."""
    out_dir = cfg.artifacts_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    from diffusers import DDPMPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ddpm] loading {cfg.sampler.checkpoint} on {device}")
    pipe = DDPMPipeline.from_pretrained(cfg.sampler.checkpoint, torch_dtype=torch.float32)
    pipe = pipe.to(device)
    pipe.unet.eval()

    extra = cfg.sampler.extra
    written: dict[int, Path] = {}
    for i in indices:
        img = split.imgs[i]
        if img.dtype != np.uint8 or img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(f"celeba image[{i}] must be (H, W, 3) uint8; got {img.shape} {img.dtype}")
        artifact = _sample_one(
            pipe, img,
            num_inference_steps=cfg.sampler.num_steps,
            num_samples=cfg.sampler.num_samples,
            base_seed=int(extra.get("base_seed", 44)),
            blur_ksize_true=int(extra.get("blur_ksize_true", cfg.forward_op.blursize)),
            blur_sigma_true=float(extra.get("blur_sigma_true", cfg.forward_op.blursigma)),
            blur_sigma_model=float(extra.get("blur_sigma_model", cfg.forward_op.blursigma)),
            dps_scale=cfg.sampler.zeta,
            sigma_y=float(extra.get("sigma_y_dps", 1.0)),
        )
        dst = (out_dir / f"img_{i:04d}.npz").resolve()
        artifact.save(dst)
        written[i] = dst
        print(f"[ddpm]  img {i:>4d}: wrote {dst.name} (T={artifact.num_timesteps}, B={artifact.num_samples})")
    return written


def load_split_for(cfg: Config, kind: str) -> DatasetSplit:
    if kind == "ood":
        return load_split(cfg.dataset.ood_npy)
    if kind == "id":
        if cfg.dataset.id_npy is None:
            raise ValueError("config.dataset.id_npy required for kind='id'")
        return load_split(cfg.dataset.id_npy)
    raise ValueError(f"unknown kind {kind!r}")
