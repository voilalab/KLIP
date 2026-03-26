import os
import sys
import importlib.util

import jax
import flax
from flax.training import checkpoints
from flax.core import freeze

import datasets
import sde_lib
import losses
from models import utils as mutils
from models import ncsnpp
from cs import get_cs_solver

import ml_collections
import numpy as np
import jax.numpy as jnp
import tqdm
from sklearn.metrics import roc_curve, auc


def load_config_from_file(config_path: str) -> ml_collections.ConfigDict:
    config_path = os.path.abspath(config_path)
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["config_module"] = config_module
    spec.loader.exec_module(config_module)
    return config_module.get_config()

def setup(config, checkpoint_dir):

  seed = config.seed
   
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
      sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
      sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
      sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
      sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
      sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
      sampling_eps = 1e-5
  else:
      raise NotImplementedError(f"SDE {config.training.sde} unknown.")
      
  sample_per_img = config.eval.samples_per_img
  img_per_device = config.eval.img_per_device
  config.eval.batch_size = sample_per_img * img_per_device * len(jax.devices('gpu'))
  sampling_shape = (sample_per_img * img_per_device,
                  config.data.image_size, config.data.image_size,
                  config.data.num_channels)

  rng = jax.random.PRNGKey(seed + 1)
  rng = jax.random.fold_in(rng, jax.process_index())

  # Initialize model
  rng, model_rng = jax.random.split(rng)
  score_model, init_model_state, initial_params = mutils.init_model(model_rng, config)
  cs_solver = get_cs_solver(config, sde, score_model, sampling_shape, inverse_scaler, 
                          eps=sampling_eps)
  tx = losses.get_optimizer(config)
  state = mutils.State.create(
      apply_fn    = score_model.apply,
      params      = initial_params,        # replaced by restore
      tx          = tx,
      optimizer   = None,                  # unused in eval after refactor
      lr          = config.optim.lr,
      model_state = init_model_state,      # replaced by restore
      ema_rate    = config.model.ema_rate,
      params_ema  = initial_params,        # replaced by restore
      rng         = rng,
  )

  state = state.replace(step=0)
  state_restored = checkpoints.restore_checkpoint(checkpoint_dir, state, step=getattr(config.eval, "ckpt_id", None))

  # LEGACY: raw dict checkpoint with "optimizer" etc. keys, instead of a State object. We want to support both for now, but eventually we should switch to only saving State objects.
  if isinstance(state_restored, dict):
    raw = state_restored
    old_opt    = raw["optimizer"]
    params     = flax.serialization.from_state_dict(initial_params, old_opt["target"])
    model_st   = raw.get("model_state", init_model_state)
    if isinstance(model_st, dict):          # keep FrozenDict type for scan/pmap
        model_st = freeze(model_st)
    params_ema = raw.get("params_ema", params)
    step       = raw.get("step", 0)
    rng        = raw.get("rng", rng)

    # Rebuild a proper State object
    state_restored = mutils.State(
        apply_fn    = score_model.apply,
        params      = params,
        tx          = tx,
        opt_state   = tx.init(params),      # not used in eval, but State requires it
        optimizer   = None,
        lr          = config.optim.lr,
        model_state = model_st,
        ema_rate    = config.model.ema_rate,
        params_ema  = params_ema,
        rng         = rng,
    )

  state = state_restored

  pstate = flax.jax_utils.replicate(state)

  return cs_solver, scaler, sampling_shape, pstate, rng


def run_sampling(config, cs_solver, scaler, sampling_shape, pstate, rng, test_imgs):
    
    test_imgs = test_imgs.reshape((jax.process_count(), -1, *test_imgs.shape[1:]))[jax.process_index()]
    hyper_params = {
        'projection': [config.sampling.coeff, config.sampling.snr],
        'langevin_projection': [config.sampling.coeff, config.sampling.snr],
        'langevin': [config.sampling.projection_sigma_rate, config.sampling.snr],
        'baseline': [config.sampling.projection_sigma_rate, config.sampling.snr]
    }[config.sampling.cs_solver]

    per_host_batch_size = config.eval.batch_size // jax.host_count()
    num_batches = int(np.ceil(len(test_imgs) / per_host_batch_size))

    all_diffs = []

    for batch in range(num_batches):
        current_batch = jnp.asarray(test_imgs[batch * per_host_batch_size:
                                            min((batch + 1) * per_host_batch_size,
                                                len(test_imgs))], dtype=jnp.float32) / 255.
        
        n_effective_samples = len(current_batch)
        if n_effective_samples < per_host_batch_size:
            pad_len = per_host_batch_size - len(current_batch)
            current_batch = jnp.pad(current_batch, ((0, pad_len), (0, 0), (0, 0)),
                                    mode='constant', constant_values=0.)

        current_batch = current_batch.reshape((-1, *sampling_shape))
        img = scaler(current_batch)

        rng, *step_rng = jax.random.split(rng, jax.local_device_count() + 1)
        step_rng = jnp.asarray(step_rng)

        samples, diffs = cs_solver(step_rng, pstate, img, *hyper_params)
        all_diffs.extend(diffs)

    return all_diffs


def run_sampling_batched(config, cs_solver, scaler, sampling_shape, pstate, rng, imgs):

    device_cnt = len(jax.devices('gpu'))

    img_size = config.data.image_size
    block_size = config.eval.block_size

    sample_per_img = config.eval.samples_per_img
    img_per_device = config.eval.img_per_device

    t_start, t_end = config.eval.t_start, config.eval.t_end

    all_klips = []
    for i in tqdm.tqdm(range(0, len(imgs), device_cnt * img_per_device)):

        test_imgs_batch = np.vstack([np.repeat(np.expand_dims(imgs[i+j], axis=0), sample_per_img, 0) for j in range(device_cnt * img_per_device)])

        diffs = run_sampling(config, cs_solver, scaler, sampling_shape, pstate, rng, test_imgs_batch)
        diffs = np.asarray(diffs)[..., 0].transpose(0, 2, 1, 3, 4)  # device_cnt, img_per_device, sample_per_img * t, img_size, img_size
        diffs = diffs.reshape(device_cnt * img_per_device, 
                              sample_per_img, 
                              -1, 
                              img_size//block_size, 
                              block_size, 
                              img_size//block_size,
                              block_size)
        diffs = (diffs[:, :, t_start:t_end, ...] ** 2).mean(axis=(1, 2, 4, 6))

        all_klips.append(diffs)

    all_klips = np.concatenate(all_klips)
    return all_klips




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("checkpoint_dir", type=str)
    args = parser.parse_args()

    config = load_config_from_file(args.config_path)
    cs_solver, scaler, sampling_shape, pstate, rng = setup(config, checkpoint_dir=args.checkpoint_dir)
    
    ood_data_dir = './data/tumor_imgs.npy' if config.eval.ood == 'tumor' else './data/star_imgs.npy'
    ood_imgs = np.load(ood_data_dir, allow_pickle=True).item()['imgs']

    if config.eval.task == "dataset":

        id_imgs = np.load("./data/id_imgs.npy", allow_pickle=True).item()['imgs']

        print("Running sampling for ID images...")
        klips_id = run_sampling_batched(config, cs_solver, scaler, sampling_shape, pstate, rng, id_imgs)
        
        print("Running sampling for OOD images...")
        klips_ood = run_sampling_batched(config, cs_solver, scaler, sampling_shape, pstate, rng, ood_imgs)

        klips_id = klips_id.max(axis=(-2, -1))
        klips_ood = klips_ood.max(axis=(-2, -1))

        y_test_id = np.zeros(len(klips_id))
        y_test_ood = np.ones(len(klips_ood))
        y_true = np.append(y_test_id, y_test_ood)
        sample_score = np.append(klips_id, klips_ood)
        fpr, tpr, _ = roc_curve(y_true, sample_score)
        auroc = auc(fpr, tpr)
        print(f"Dataset Level: {auroc}")

    elif config.eval.task == "image":

        ood_labels = np.load(ood_data_dir, allow_pickle=True).item()['labels']
        body_masks = np.load(ood_data_dir, allow_pickle=True).item()['masks']

        print("Running sampling...")
        klips_ood = run_sampling_batched(config, cs_solver, scaler, sampling_shape, pstate, rng, ood_imgs)

        img_size = config.data.image_size
        block_size = config.eval.block_size
        ood_labels_block = (ood_labels.reshape(-1, img_size//block_size, block_size, img_size//block_size, block_size)==2).any(axis=(2, 4))
        body_masks_block = (body_masks.reshape(-1, img_size//block_size, block_size, img_size//block_size, block_size)>0).all(axis=(2, 4))

        aurocs = []
        for i in range(len(klips_ood)):
            fpr, tpr, _ = roc_curve(ood_labels_block[i].flatten(), (klips_ood[i] * body_masks_block[i]).flatten())
            aurocs.append(auc(fpr, tpr))
        print(f"Image Level: {np.average(aurocs)}")
        
    else:
        print("Invalid eval task specified. Must be 'dataset' or 'image'.")