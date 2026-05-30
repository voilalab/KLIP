# `data/` — KLIP datasets

All evaluation code reads `.npy` files only. The CHAOS files are shared between
the song22 and PaDIS pipelines — same on-disk source, different runtime
resolutions.

## Canonical schema

Every dataset `.npy` is a pickled `dict`. Keys follow a consistent convention:

| Key      | Shape                                                                 | Dtype                       | Meaning                                                |
| ---      | ---                                                                   | ---                         | ---                                                    |
| `imgs`   | `(N, H, W)` for grayscale CT, `(N, H, W, 3)` for RGB CelebA            | `uint8`, range `[0, 255]`   | Input images                                           |
| `masks`  | `(N, H, W)`                                                           | `uint8`                     | Foreground / body / scar mask                          |
| `labels` | `(N, H, W)`                                                           | `uint8` or `bool`           | Per-pixel OOD label (e.g. tumor voxels = 2 in CHAOS)   |

## CHAOS (shared between song22 and PaDIS)

| File                  | Shape                  | Keys                                                                        |
| ---                   | ---                    | ---                                                                         |
| `chaos_id.npy`        | `(100, 512, 512)` uint8 | `imgs`, `masks`                                                            |
| `chaos_ood_tumor.npy` | `(250, 512, 512)` uint8 | `imgs`, `masks`, `labels` (values in `{0, 1, 2}`; tumor voxels labeled `2`) |
| `chaos_ood_star.npy`  | `(100, 512, 512)` uint8 | `imgs`, `masks`, `labels` (bool tuning labels)                              |

- **song22** consumes the 512×512 arrays directly; paths come from
  `config.eval.{id,tumor,star}_npy` in
  [song22/configs/ve/](../song22/configs/ve/).
- **PaDIS** ([CT/dps_sampling_test.py](../CT/dps_sampling_test.py)) takes one
  of these via `--image_npy` and **LANCZOS-resizes to `--image_size` (default
  256) at load time**, matching the DICOM→PNG preprocessing in
  [CT/process_AAPM.py](../CT/process_AAPM.py).
- **PaDIS scoring** reads the body/tumor masks from the same file: the
  `masks` / `labels` keys are decimated to the sampler's working resolution
  by [pipeline/cli.py](../pipeline/cli.py)'s `stage_auroc`.

## CelebA (DPS + Gaussian deblur)

> **Not yet included** — the face data will be added in a later release. The
> schema below is what the pipeline expects; the YAML config
> [pipeline/configs/celeba_image.yaml](../pipeline/configs/celeba_image.yaml)
> already points at these paths.

| File                    | Shape                                                                 | Keys                       |
| ---                     | ---                                                                   | ---                        |
| `celeba_test.npy`       | `(N, 256, 256, 3)` uint8 imgs + `(N, 256, 256)` uint8 masks            | `imgs`, `masks`            |
| `celeba_characters.npy` | `(N, 256, 256, 3)` uint8 (whole-image OOD; no masks)                  | `imgs`                     |

`masks > 127` is the OOD label (e.g. scar pixels). The DDPM weights are pulled
from `google/ddpm-celebahq-256` on the Hugging Face hub at runtime — no local
checkpoint is needed.

## Path mapping for the unified configs

Each pipeline config in [pipeline/configs/](../pipeline/configs/) points at the
right `.npy` via:

```yaml
dataset:
  ood_npy: data/chaos_ood_tumor.npy   # or chaos_ood_star.npy, or celeba_test.npy
  id_npy:  data/chaos_id.npy          # required when task: dataset
```
