# KLIP: Localized Distribution Shift Detection via KL-Divergence with Diffusion Priors in Inverse Problems

Official code release for the CVPR 2026 paper:

> [**KLIP: Localized Distribution Shift Detection via KL-Divergence with Diffusion Priors in Inverse Problems**](https://arxiv.org/abs/2605.31596)
> Alireza Kheirandish\*, Jihoon Hong\*, Sara Fridovich-Keil
> *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.*

KLIP is a calibration-free OOD detection metric that estimates the prior–posterior KL divergence during diffusion-based posterior sampling for an inverse problem. We evaluate it across three (model, dataset, inverse problem) combinations:

| Backend | Model | Inverse problem | Dataset |
| --- | --- | --- | --- |
| `song22_jax` | NCSNPP predictor-corrector (Song et al. 2022) | sparse-view CT (24 views) | CHAOS abdominal CT (ID) + synthetic tumors / star artifacts (OOD) |
| `padis_torch` | PaDIS patch-based EDM (Hu et al. 2023) | sparse-view CT (24 views) | same CHAOS data |
| `ddpm_torch` | DDPM on CelebA-HQ (HF: `google/ddpm-celebahq-256`) | Gaussian deblur | CelebA-HQ (ID) + synthetic scars / film & TV characters (OOD) |

All three are driven by a single CLI ([pipeline/cli.py](pipeline/cli.py)) with one YAML config per task. See [pipeline/README.md](pipeline/README.md) for pipeline internals and [data/README.md](data/README.md) for the data schema.

## Repository layout

```
KLIP/
├── pipeline/                # the unified pipeline (CLI, scoring, samplers, configs)
├── CT/                      # PaDIS sampler — invoked by pipeline.samplers.padis_torch
├── song22/                  # song22 PC sampler — invoked by pipeline.samplers.song22_jax
├── checkpoints/             # trained model checkpoints
│   ├── chaos_song_ckpt/     # song22 Orbax checkpoint on CHAOS
│   └── chaos_padis_ckpt/    # PaDIS .pkl on CHAOS
├── data/                    # unified .npy datasets (see data/README.md)
└── output/                  # per-config: artifacts/, scores/, auroc.json
```

The unified pipeline does not reimplement the sampler internals — it shells out to the existing `CT/dps_sampling_test.py` and a tiny adapter in `song22/_pipeline_sample.py`, both of which use the library code in those subdirectories. So those directories stay as-is.

## Setup

A single conda environment satisfies all three backends.

```bash
# 1. Create the base env (Python 3.9 + JAX 0.4.30 + Flax 0.8.5 + the song22 deps;
#    also installs a small patch to flax.linen.normalization)
bash song22/setup.sh klip
conda activate klip

# 2. Add the pieces the unified pipeline + PaDIS + CelebA need
pip install torch>=2.8 diffusers click pyyaml
# (torch>=2.8 is required for the PaDIS .pkl checkpoint to unpickle cleanly)
```

### Hugging Face cache redirect (CelebA)

The CelebA pipeline downloads `google/ddpm-celebahq-256` from the HF hub on first run. If your home directory has a small quota, point the cache at a larger filesystem:

```bash
export HF_HOME=/path/with/space/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME
```

### PaDIS vendored files

The PaDIS sampler depends on a few files from the upstream [jasonhu4/PaDIS](https://github.com/jasonhu4/PaDIS) repo that are vendored under `CT/`: `denoise_padding.py`, `training/pos_embedding.py`, and the local `dnnlib/`, `torch_utils/`, `odlstuff/`, `parbeam_updated.py`. No additional install needed — they're checked in.

### song22 metal-masks asset

The song22 sampler eagerly loads metal-artifact-removal masks from `song22/assets/metal_masks/`. If that directory is missing on a fresh clone, point a symlink at a copy of the `.mat` files from the original Song et al. release.

## Downloading checkpoints and data

The model checkpoints and CHAOS data files are too large for git and live on
Google Drive instead:

- **Checkpoints** — [drive.google.com/drive/folders/1xdW2m027QpvCGsP-9xbdl_EMwQ6GyYw4](https://drive.google.com/drive/folders/1xdW2m027QpvCGsP-9xbdl_EMwQ6GyYw4?usp=drive_link)
  - `chaos_song_ckpt/` — song22 NCSNPP checkpoint on CHAOS (Orbax format, ~59 MB)
  - `chaos_padis_ckpt/network-snapshot-001345.pkl` — PaDIS EDM checkpoint on CHAOS (~208 MB)
- **Data** — [drive.google.com/drive/folders/1cIyPd_eR1JFTk6ekQRJ0mzNxKsiaXtVU](https://drive.google.com/drive/folders/1cIyPd_eR1JFTk6ekQRJ0mzNxKsiaXtVU?usp=drive_link)
  - `chaos_id.npy`, `chaos_ood_tumor.npy`, `chaos_ood_star.npy` — derived from the
    [CHAOS Grand Challenge](https://chaos.grand-challenge.org/) training split
    (`CC-BY-NC-SA-4.0`; please cite the dataset — see [data/README.md](data/README.md))

After download, the on-disk layout should be:

```
KLIP/
├── checkpoints/
│   ├── chaos_song_ckpt/
│   │   └── checkpoints/checkpoint_3/...                   # Orbax checkpoint shards
│   └── chaos_padis_ckpt/
│       └── network-snapshot-001345.pkl                    # PaDIS pickle
└── data/
    ├── chaos_id.npy
    ├── chaos_ood_tumor.npy
    └── chaos_ood_star.npy
```

CLI download (optional):

```bash
pip install gdown
gdown --folder 1xdW2m027QpvCGsP-9xbdl_EMwQ6GyYw4 -O checkpoints
gdown --folder 1cIyPd_eR1JFTk6ekQRJ0mzNxKsiaXtVU -O data
```

These paths are also referenced in [pipeline/configs/*.yaml](pipeline/configs/),
so once the files are in place no further configuration is needed.

### Data schema

Each `.npy` is a pickled `dict` with keys `imgs`, optionally `masks`, optionally
`labels`. See [data/README.md](data/README.md) for shapes and per-file details.

The CelebA face data (`data/celeba_test.npy`, `data/celeba_characters.npy`)
will be added in a later release; until then, the
[celeba_image.yaml](pipeline/configs/celeba_image.yaml) config can be used as a
template once the files are placed under `data/`.

## Quick start

```bash
# image-level OOD localization on sparse-view CT, song22 PC sampler:
CUDA_VISIBLE_DEVICES=0 python -m pipeline.cli \
    --config pipeline/configs/chaos_song_image.yaml \
    --stage all --ood-indices 0

# dataset-level OOD detection on sparse-view CT, song22 PC sampler:
CUDA_VISIBLE_DEVICES=0 python -m pipeline.cli \
    --config pipeline/configs/chaos_song_dataset.yaml \
    --stage all --ood-indices 0-2 --id-indices 0-2

# image-level OOD localization on sparse-view CT, PaDIS DPS sampler:
CUDA_VISIBLE_DEVICES=0 python -m pipeline.cli \
    --config pipeline/configs/chaos_padis_image.yaml \
    --stage all --ood-indices 0

# image-level OOD localization on Gaussian deblur (CelebA):
CUDA_VISIBLE_DEVICES=0 python -m pipeline.cli \
    --config pipeline/configs/celeba_image.yaml \
    --stage all --ood-indices 0
```

Each invocation writes:
- `output/<name>/artifacts/img_NNNN.npz` — canonical `(T, B, C, H, W)` normalized updates + per-step `g(t)` + the sampler's mean reconstruction
- `output/<name>/scores/{ood,id}_NNNN.npy` — per-image KLIP score maps
- `output/<name>/auroc.json` — dataset-level and/or image-level AUROC

To run stages independently (useful if sampling is expensive and you want to tune the KLIP window without re-sampling):

```bash
python -m pipeline.cli --config X.yaml --stage sample --ood-indices 0-99 --id-indices 0-99
python -m pipeline.cli --config X.yaml --stage score
python -m pipeline.cli --config X.yaml --stage auroc
```

## Verifying the pipeline

```bash
# scoring + AUROC bit-equivalence against the original PaDIS-notebook recipe
# (the notebook is gone; the recipe is preserved inline in the test)
python pipeline/tests/test_scoring_recipe.py
```

## Citation

```bibtex
@inproceedings{klip2026,
  title  = {KLIP: Localized Distribution Shift Detection via KL-Divergence with Diffusion Priors in Inverse Problems},
  author = {Kheirandish, Alireza and Hong, Jihoon and Fridovich-Keil, Sara},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year   = {2026},
}
```

## Acknowledgments

The three samplers wrap or build on prior open-source work — please cite the originals when relevant:

- **song22**: Song, Y., Shen, L., Xing, L., Ermon, S. *Solving Inverse Problems in Medical Imaging with Score-Based Generative Models*, ICLR 2022. [code](https://github.com/yang-song/score_sde)
- **PaDIS**: Hu, J., et al. *PaDIS: Patch-based Diffusion Inverse Solver*, 2023. [code](https://github.com/jasonhu4/PaDIS)
- **CelebA-HQ DDPM**: `google/ddpm-celebahq-256` on the Hugging Face hub.

```bibtex
@inproceedings{song2022solving,
  title  = {Solving Inverse Problems in Medical Imaging with Score-Based Generative Models},
  author = {Song, Yang and Shen, Liyue and Xing, Lei and Ermon, Stefano},
  booktitle = {International Conference on Learning Representations},
  year   = {2022},
}

@article{hu2023padis,
  title  = {PaDIS: Patch-based Diffusion Inverse Solver},
  author = {Hu, Jason and others},
  year   = {2023},
  url    = {https://github.com/jasonhu4/PaDIS},
}
```
