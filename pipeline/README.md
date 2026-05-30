# Unified KLIP pipeline

Consolidates the three legacy code paths (`song22/`, `CT/`, `CelebA/`) into one
config-driven pipeline. Same model checkpoints, same data files, same KLIP
algorithm — but one CLI and one scoring module.

## Quick start

```bash
# PaDIS sparse-view CT, image-level AUROC, one OOD image
CUDA_VISIBLE_DEVICES=0 python -m pipeline.cli \
    --config pipeline/configs/chaos_padis_image.yaml \
    --stage all --ood-indices 0

# song22 predictor-corrector, image-level AUROC
CUDA_VISIBLE_DEVICES=0 python -m pipeline.cli \
    --config pipeline/configs/chaos_song_image.yaml \
    --stage all --ood-indices 0

# CelebA Gaussian deblur (DPS + DDPM), image-level AUROC
CUDA_VISIBLE_DEVICES=0 HF_HOME=/usr/scratch/jhong392/huggingface \
    python -m pipeline.cli \
    --config pipeline/configs/celeba_image.yaml \
    --stage all --ood-indices 0

# Dataset-level AUROC (needs both ID and OOD samples)
CUDA_VISIBLE_DEVICES=0 python -m pipeline.cli \
    --config pipeline/configs/chaos_song_dataset.yaml \
    --stage all --ood-indices 0-9 --id-indices 0-9
```

## Layout

```
pipeline/
├── cli.py               # `python -m pipeline.cli ...`
├── io/
│   ├── artifact.py      # Canonical (T, B, C, H, W) per-step normalized-update tensor
│   ├── config.py        # dataclass schema, YAML loader
│   └── datasets.py      # reads unified data/chaos_*.npy and data/celeba_*.npy
├── scoring/
│   ├── aggregate.py     # Eq. 12: block reshape + [t0, t1] restriction + ||·||² + E
│   └── auroc.py         # dataset-level (max-over-blocks) and image-level (per-image, masked)
├── samplers/
│   ├── padis_torch.py   # subprocess wrapper over CT/dps_sampling_test.py
│   ├── song22_jax.py    # subprocess wrapper over song22/_pipeline_sample.py
│   └── ddpm_torch.py    # in-process port of CelebA/celebA.ipynb
├── configs/             # one YAML per (model × dataset × task)
└── tests/               # parity tests vs the legacy notebook recipe
```

## Stages

The CLI runs four stages, individually or in sequence (`--stage all`):

1. **sample** — invoke the configured sampler on the requested image indices,
   write canonical `Artifact` `.npz` files to `output/<name>/artifacts/`.
2. **score** — load artifacts, compute per-image KLIP score maps, save
   `.npy` under `output/<name>/scores/`.
3. **auroc** — compute dataset- and/or image-level AUROC from the score maps
   and masks, write `output/<name>/auroc.json`.

## Canonical artifact (`pipeline/io/artifact.py`)

```python
Artifact(
    normalized_updates: np.ndarray,   # (T, B, C, H, W) float32, = delta_x_t / g(t)
    g_values: np.ndarray,             # (T,) float32, per-step diffusion coefficient
    source_image: np.ndarray | None,  # optional, for debugging
)
```

The per-sampler normalization (what divides `delta_x_t`) lives in each sampler
wrapper so that downstream scoring is framework-agnostic NumPy:

| sampler        | normalization at write time |
| --- | --- |
| `padis_torch`  | divide by `sigma(t)^p` (default `p=0.5`) |
| `song22_jax`   | already normalized in-place by the predictor-corrector solver — write-through |
| `ddpm_torch`   | divide by `sqrt(beta_t)` |

## Scoring formula

[`scoring/aggregate.py`](scoring/aggregate.py) implements Eq. 12 of the paper:

```
KLIP(B_i, [t0, t1]; y) ≈
    (1/2) × mean_{t ∈ [t0, t1]} mean_{samples} sum_{(c, i, j) ∈ B_i}
        (normalized_updates[t, sample, c, i, j])²
```

The constant `1/2` is dropped (it doesn't affect AUROC). On C=1, B=1 inputs
this matches `CT/Klip_PaDIS.ipynb`'s `compute_patch_energy_grid` bit-for-bit
within float32 precision (verified in
[`tests/test_scoring_matches_notebook.py`](tests/test_scoring_matches_notebook.py)).

## AUROC modes ([`scoring/auroc.py`](scoring/auroc.py))

- `dataset_level`: image score = max KLIP across blocks; ROC over the full
  (ID, OOD) image pool. Matches `song22/run_klip.py` task='dataset'.
- `image_level`: per-image ROC over body-masked blocks, averaged across the
  OOD set. Matches `CT/Klip_PaDIS.ipynb` and `CelebA/celebA.ipynb`.

## Tests

```bash
# scoring + AUROC bit-equivalence against the original PaDIS-notebook recipe
python pipeline/tests/test_scoring_recipe.py
```

The recipe under test is preserved inline (the notebook itself is gone).

## Configs

Each YAML has `name`, `task`, `sampler`, `dataset`, `forward_op`, `klip`,
`output`. See [`configs/chaos_padis_image.yaml`](configs/chaos_padis_image.yaml)
for the canonical example. Path fields are resolved relative to the repo root.

Sampler-backend-specific knobs go under `sampler.extra` so the top-level
schema stays uniform:
- `song22_jax`: `extra.song22_config` points at a ml_collections config in
  `song22/configs/ve/`. The song22 config governs all model/SDE settings;
  the unified config only adds paths + KLIP window knobs.
- `ddpm_torch`: `extra.{base_seed, blur_ksize_true, blur_sigma_true, blur_sigma_model, sigma_y_dps}`.

## Status

| pipeline | sampler | scoring | AUROC (image) | AUROC (dataset) | end-to-end smoke-tested |
| --- | --- | --- | --- | --- | --- |
| song22 PC       | ✅ | ✅ | ✅ | ✅ | ✅ on `chaos_ood_tumor` (1 image); recon PSNR 32.0 dB |
| PaDIS DPS       | ✅ | ✅ | ✅ | ✅ | ✅ on `chaos_ood_tumor` (2 images); recon PSNR 33.1 dB |
| CelebA DPS+DDPM | ✅ | ✅ | ✅ | ✅ | ✅ pipeline runs; needs real CelebA data for meaningful numbers |

## Legacy removed

These were deleted as part of consolidation — same behavior is now in the
unified pipeline:

- `CT/Klip_PaDIS.ipynb` → `pipeline/scoring/{aggregate,auroc}.py`
- `CelebA/celebA.ipynb` → `pipeline/samplers/ddpm_torch.py` (sampler) + `pipeline/scoring/` (scoring)
- `inspect_chaos_test.ipynb` → `data/chaos_*.npy` are materialized; recipe in git history
- `CT/sampling_script.sh` → `python -m pipeline.cli --config X --stage sample`
- `song22/run_klip.py` standalone CLI → `python -m pipeline.cli --config X --stage all` (file kept as library; only `load_config_from_file` + `setup` remain)
- `CT/inverse_nodist.py`, `CT/requirements_CelebA.txt`, `CelebA/README.md` → unused

## What stays in the legacy directories

The unified pipeline shells out to the existing samplers — it doesn't reimplement
the SDE / DDPM / patch-denoise logic. So these stay:

- `CT/dps_sampling_test.py` + its `dnnlib/`, `torch_utils/`, `training/`, `odlstuff/`, `denoise_padding.py`, `inverse_operators.py`, `parbeam_updated.py` — invoked by `pipeline.samplers.padis_torch`.
- `song22/run_klip.py` (now library-only, 115 lines) + its `cs.py`, `sampling.py`, `datasets.py`, `sde_lib.py`, `losses.py`, `models/`, `transforms/`, `mar/`, `configs/ve/` — invoked by `pipeline.samplers.song22_jax`.
- Trained checkpoints in `checkpoints/`.
