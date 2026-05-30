"""Library entry points for the song22 PC sampler — used by the unified pipeline.

This file used to host the standalone CLI (`python run_klip.py <config> <ckpt>`)
that did sampling + KLIP scoring + AUROC in one. That CLI is gone; the unified
pipeline at `pipeline/cli.py` replaces it. Only the model + SDE setup remains
here, exposed for `song22/_pipeline_sample.py`.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import flax
import jax
from flax.core import freeze
from flax.training import checkpoints

import datasets
import losses
import ml_collections
import sde_lib
from cs import get_cs_solver
from models import ncsnpp  # noqa: F401  (side effect: registers the architecture)
from models import utils as mutils


def load_config_from_file(config_path: str) -> ml_collections.ConfigDict:
    config_path = os.path.abspath(config_path)
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["config_module"] = config_module
    spec.loader.exec_module(config_module)
    return config_module.get_config()


def setup(config, checkpoint_dir):
    seed = config.seed

    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    if config.training.sde.lower() == "vpsde":
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == "subvpsde":
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == "vesde":
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    sample_per_img = config.eval.samples_per_img
    img_per_device = config.eval.img_per_device
    config.eval.batch_size = sample_per_img * img_per_device * len(jax.devices("gpu"))
    sampling_shape = (
        sample_per_img * img_per_device,
        config.data.image_size,
        config.data.image_size,
        config.data.num_channels,
    )

    rng = jax.random.PRNGKey(seed + 1)
    rng = jax.random.fold_in(rng, jax.process_index())

    rng, model_rng = jax.random.split(rng)
    score_model, init_model_state, initial_params = mutils.init_model(model_rng, config)
    cs_solver = get_cs_solver(config, sde, score_model, sampling_shape, inverse_scaler, eps=sampling_eps)
    tx = losses.get_optimizer(config)
    state = mutils.State.create(
        apply_fn=score_model.apply,
        params=initial_params,
        tx=tx,
        optimizer=None,
        lr=config.optim.lr,
        model_state=init_model_state,
        ema_rate=config.model.ema_rate,
        params_ema=initial_params,
        rng=rng,
    )

    state = state.replace(step=0)
    state_restored = checkpoints.restore_checkpoint(
        checkpoint_dir, state, step=getattr(config.eval, "ckpt_id", None)
    )

    # Legacy raw-dict checkpoint compatibility (pre-State-object snapshots).
    if isinstance(state_restored, dict):
        raw = state_restored
        old_opt = raw["optimizer"]
        params = flax.serialization.from_state_dict(initial_params, old_opt["target"])
        model_st = raw.get("model_state", init_model_state)
        if isinstance(model_st, dict):
            model_st = freeze(model_st)
        params_ema = raw.get("params_ema", params)
        rng = raw.get("rng", rng)

        state_restored = mutils.State(
            apply_fn=score_model.apply,
            params=params,
            tx=tx,
            opt_state=tx.init(params),
            optimizer=None,
            lr=config.optim.lr,
            model_state=model_st,
            ema_rate=config.model.ema_rate,
            params_ema=params_ema,
            rng=rng,
        )

    state = state_restored
    pstate = flax.jax_utils.replicate(state)
    return cs_solver, scaler, sampling_shape, pstate, rng
