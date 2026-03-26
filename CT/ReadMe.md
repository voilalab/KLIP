# KLIP: KLIP implementation over PaDIS model via DPS for inverse problems


This repository is a **modified** of [jasonhu4/PaDIS](https://github.com/jasonhu4/PaDIS), a Patch-based Position-Aware Diffusion Inverse Solver for solving inverse imaging problems (CT reconstruction, deblurring, superresolution).

## What's Different From the Original PaDIS

The original PaDIS performs reconstruction by sampling from a patch-based diffusion model. This fork introduces the following changes:

1. **Modified sampling**: The sampler now **stores intermediate measurement updates** (i.e., the data-consistency corrections applied at each diffusion step). These stored updates are the inputs required by the **KLIP** post-processing method.

2. **KLIP post-processing notebook**: A Jupyter notebook is included that loads the stored measurement updates and runs the KLIP algorithm for OOD detection.


## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/KheirAli/Klip_test.git
cd Klip_test
```

### 2. Create the Conda Environment

create the environment for PaDIS model and all of the requirements from there. 


### 3. Prepare Directories

```bash
# Create a directory for your test images (input)
mkdir image_dir

# Create a directory for results (output)
mkdir results

# Create the checkpoint directory
mkdir training-runs
```

Place your test images (`.png`) inside `image_dir/`. The reconstruction script will process all images in this folder.

---

## Usage

### Step 1 — Train or Download a Checkpoint

Train a model (see the original [PaDIS repo](https://github.com/jasonhu4/PaDIS) for training details), or download a pre-trained CT checkpoint from [here](https://drive.google.com/file/d/1-WBjiXYTOelRIWS38Ft6J_TGGqkPy-ou/view?usp=sharing).

Place the checkpoint inside `training-runs/`.

### Step 2 — Run PaDIS Reconstruction (Modified)

Set your paths at the top of the command. The two key variables are:

run the bash script for cinditional over measurement sampling.


**What the modified script saves to `OUTDIR`:**
- Reconstructed images (same as original PaDIS)
- **`measurement_updates/`** — a subdirectory containing the stored measurement update tensors at each diffusion step, required for the KLIP method

### Step 3 — Run KLIP Post-processing

After reconstruction is complete, open and run the Jupyter notebook:


The notebook will:
1. Load the stored measurement updates from `OUTDIR/measurement_updates/`
2. Apply the KLIP algorithm
3. Save the KLIP-processed reconstructions to `KLIP_OUTDIR`

---

## Key Parameters (Reference)

| Parameter | Description | Default |
|---|---|---|
| `--network` | Path to trained model checkpoint (`.pkl`) | required |
| `--outdir` | Output directory for results | required |
| `--image_dir` | Directory of input PNG images | required |
| `--image_size` | Image resolution (pixels) | `256` |
| `--views` | Number of CT views (for CT tasks) | `20` |
| `--steps` | Number of diffusion sampling steps | `100` |
| `--sigma_min` | Minimum noise level | `0.003` |
| `--sigma_max` | Maximum noise level | `10` |
| `--zeta` | Data-consistency step size | `0.3` |
| `--pad` | Patch padding size | `24` |
| `--psize` | Patch size | `56` |
| `--name` | Inverse problem type (e.g. `ct_parbeam`, `deblur`, `superres`) | required |

> All other hyperparameters inherit from the original PaDIS — see [jasonhu4/PaDIS](https://github.com/jasonhu4/PaDIS) for full documentation.

---

## Why We Store Measurement Updates

In the original PaDIS, data-consistency corrections are computed and applied at each diffusion timestep but **discarded** afterwards. For the KLIP method to work, these measurement updates must be retained across all timesteps for each reconstructed image.

The modification in sampling, intercepts these updates during the sampling loop and writes them to disk. The KLIP notebook then loads them for computing ood measure.

---

## Citation

If you use this code, please cite the original PaDIS paper:

```bibtex
@article{hu2023padis,
  title   = {PaDIS: Patch-based Diffusion Inverse Solver},
  author  = {Hu, Jason and others},
  year    = {2023},
  url     = {https://github.com/jasonhu4/PaDIS}
}
