# KLIP: Localized Distribution Shift Detection via KL-Divergence with Diffusion Priors in Inverse Problems

## Acknowledgements

This repository builds on the codebase of [Solving Inverse Problems in Medical Imaging with Score-Based Generative Models](https://openreview.net/forum?id=vaRCHVj0uGI) (Song et al., ICLR 2022). We thank the authors for making their code publicly available. If you are interested in the original work, please refer to their [repository](https://github.com/yang-song/score_sde).

---

## Setup

**1. Clone the repository**

```bash
git clone <repo-url>
cd <repo-name>
```

**2. Run the setup script**

This will create a conda environment named `klip_song22` (Python 3.9), install all dependencies, and apply a required patch to flax.

```bash
bash setup.sh
```

To use a custom environment name:

```bash
bash setup.sh my_env_name
```

Activate the environment:

```bash
conda activate klip_song22
```

---

## Running KLIP

KLIP evaluation is run via `run_klip.py`, which takes a config file and a checkpoint directory as arguments.

```bash
python run_klip.py <config_path> <checkpoint_dir>
```

**Arguments**

- `config_path`: Path to the evaluation config file (see `configs/ve/`).
- `checkpoint_dir`: Path to the directory containing the model checkpoint.

**Example**

```bash
# Dataset-level OOD detection
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_klip.py configs/ve/chaos_eval_dataset.py /path/to/checkpoint

# Image-level (localized) OOD detection
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_klip.py configs/ve/chaos_eval_image.py /path/to/checkpoint
```

The eval task (`"dataset"` or `"image"`) is controlled by the `config.eval.task` field in the config file:

- `"dataset"`: Computes dataset-level AUROC for distinguishing in-distribution vs. OOD images.
- `"image"`: Computes image-level AUROC for localizing OOD regions within individual images.

Config files are located in `configs/ve/` and follow the [`ml_collections`](https://github.com/google/ml_collections) format.

---

## References

If you find this code useful, please consider citing our paper:

```bibtex
@inproceedings{klip2026,
  title={KLIP: Localized Distribution Shift Detection via KL-Divergence with Diffusion Priors in Inverse Problems},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026},
}
```

and the original work this repository builds upon:

```bibtex
@inproceedings{song2022solving,
  title={Solving Inverse Problems in Medical Imaging with Score-Based Generative Models},
  author={Yang Song and Liyue Shen and Lei Xing and Stefano Ermon},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=vaRCHVj0uGI}
}
```
