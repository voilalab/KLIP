"""Subprocess entry for pipeline.samplers.song22_jax."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy.core
sys.modules.setdefault("numpy._core", numpy.core)

import numpy as np
import jax
import jax.numpy as jnp

from run_klip import load_config_from_file, setup
import sde_lib


def _run_sampling_capture_both(config, cs_solver, scaler, sampling_shape, pstate, rng, test_imgs):
    test_imgs = test_imgs.reshape((jax.process_count(), -1, *test_imgs.shape[1:]))[jax.process_index()]
    hyper_params = {
        "projection": [config.sampling.coeff, config.sampling.snr],
        "langevin_projection": [config.sampling.coeff, config.sampling.snr],
        "langevin": [config.sampling.projection_sigma_rate, config.sampling.snr],
        "baseline": [config.sampling.projection_sigma_rate, config.sampling.snr],
    }[config.sampling.cs_solver]

    per_host_batch_size = config.eval.batch_size // jax.process_count()
    num_batches = int(np.ceil(len(test_imgs) / per_host_batch_size))

    all_samples = []
    all_diffs = []
    for batch in range(num_batches):
        current_batch = jnp.asarray(
            test_imgs[batch * per_host_batch_size: min((batch + 1) * per_host_batch_size, len(test_imgs))],
            dtype=jnp.float32,
        ) / 255.0
        if len(current_batch) < per_host_batch_size:
            pad_len = per_host_batch_size - len(current_batch)
            current_batch = jnp.pad(current_batch, ((0, pad_len), (0, 0), (0, 0)), mode="constant")
        current_batch = current_batch.reshape((-1, *sampling_shape))
        img = scaler(current_batch)
        rng, *step_rng = jax.random.split(rng, jax.local_device_count() + 1)
        step_rng = jnp.asarray(step_rng)
        samples, diffs = cs_solver(step_rng, pstate, img, *hyper_params)
        all_samples.extend(samples)
        all_diffs.extend(diffs)
    return all_samples, all_diffs


def _g_values_for_stored_timesteps(config, sampling_eps: float) -> np.ndarray:
    sde_name = config.training.sde.lower()
    N = config.model.num_scales
    diff_every = config.eval.diff_every
    n_stored = N // diff_every

    if sde_name == "vesde":
        sigma_min = config.model.sigma_min
        sigma_max = config.model.sigma_max
        t_grid = np.linspace(1.0, sampling_eps, N)[::diff_every][:n_stored]
        sigma = sigma_min * (sigma_max / sigma_min) ** t_grid
        g = sigma * np.sqrt(2 * (np.log(sigma_max) - np.log(sigma_min)))
        return g.astype(np.float32)

    raise NotImplementedError(f"g_values for SDE={sde_name!r} not implemented yet")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image_npy", required=True)
    ap.add_argument("--indices", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    indices = [int(x) for x in args.indices.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config_from_file(args.config)
    sampling_eps = 1e-3 if config.training.sde.lower() in ("vpsde", "subvpsde") else 1e-5

    cs_solver, scaler, sampling_shape, pstate, rng = setup(config, checkpoint_dir=args.ckpt)
    print(f"[song22-sample] setup OK; processing {len(indices)} images")

    full = np.load(args.image_npy, allow_pickle=True).item()["imgs"]
    img_size = config.data.image_size

    g_vals = _g_values_for_stored_timesteps(config, sampling_eps)
    print(f"[song22-sample] g_values: {g_vals.shape}, range=[{g_vals.min():.3f}, {g_vals.max():.3f}]")

    device_cnt = len(jax.devices("gpu"))
    sample_per_img = config.eval.samples_per_img
    img_per_device = config.eval.img_per_device

    for original_i in indices:
        single = full[original_i]
        test_imgs_batch = np.vstack([
            np.repeat(single[None, ...], sample_per_img, axis=0)
            for _ in range(device_cnt * img_per_device)
        ])
        samples, diffs = _run_sampling_capture_both(
            config, cs_solver, scaler, sampling_shape, pstate, rng, test_imgs_batch
        )

        diffs = np.asarray(diffs)[..., 0]
        diffs = diffs.transpose(0, 2, 1, 3, 4)
        diffs = diffs[0].reshape(img_per_device, sample_per_img, -1, img_size, img_size)[0]
        T = diffs.shape[1]
        normalized = diffs.transpose(1, 0, 2, 3)[:, :, None, ...].astype(np.float32)

        samples_arr = np.asarray(samples)[..., 0]
        recon_mean = samples_arr[0].reshape(img_per_device, sample_per_img, img_size, img_size)[0].mean(axis=0)
        recon_mean = np.clip(recon_mean, 0, 1).astype(np.float32)

        np.savez_compressed(
            out_dir / f"img_{original_i:04d}.npz",
            normalized_updates=normalized,
            g_values=g_vals[:T],
            reconstruction=recon_mean,
        )
        print(f"[song22-sample]  img {original_i:>4d}: wrote (T={T}, B={diffs.shape[0]}), "
              f"recon range=[{recon_mean.min():.3f}, {recon_mean.max():.3f}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
