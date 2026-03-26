# CelebA DPS Deblurring Experiment

This repository contains an implementation of Diffusion Posterior Sampling (DPS) for image deblurring on the CelebA-HQ dataset, with anomaly detection capabilities through measurement update heatmaps and AUC evaluation.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output](#output)
- [Citations](#citations)
- [License](#license)

---

## 🔍 Overview

This project implements **Diffusion Posterior Sampling (DPS)** for solving inverse problems in image processing, specifically targeting motion blur removal on facial images from the CelebA-HQ dataset. The implementation includes:

- **DPS-based deblurring**: Uses pretrained diffusion models to recover sharp images from blurry observations
- **Anomaly detection**: Generates heatmaps from measurement updates to identify anomalous regions
- **Quantitative evaluation**: Computes Area Under the Curve (AUC) metrics against binary anomaly masks

---

## ✨ Features

- ✅ Pretrained DDPM model (CelebA-HQ 256×256) from Hugging Face
- ✅ Configurable motion blur kernel simulation
- ✅ Step-by-step diffusion process with measurement update tracking
- ✅ Anomaly heatmap visualization
- ✅ AUC metric evaluation for anomaly detection
- ✅ Optional synthetic artifact generation for testing
- ✅ Comprehensive visualization of results

---

## 📦 Requirements

### System Requirements
- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster inference)
- 8GB+ RAM (16GB recommended)

### Python Dependencies

```txt
torch>=2.0.0
torchvision>=0.15.0
diffusers>=0.21.0
transformers>=4.30.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
Pillow>=9.5.0
tqdm>=4.65.0
jupyter>=1.0.0
```

---

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/celeba-dps-deblurring.git
cd celeba-dps-deblurring
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n dps-deblur python=3.9
conda activate dps-deblur
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 🎯 Quick Start

### Option 1: Run the Complete Notebook

1. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

2. Open `CelebA_experiment.ipynb`

3. Run all cells sequentially (Cell → Run All)

### Option 2: Run Specific Sections

1. **Configuration Only**: Run the first cell to set parameters
2. **Model Loading**: Execute cells up to "Load Model"
3. **Single Image Test**: Run through one deblurring iteration
4. **Full Evaluation**: Execute all cells for complete analysis

---

## ⚙️ Configuration

At the top of `CelebA_experiment.ipynb`, modify the **Configuration Cell**:

```python
# ========== CONFIGURATION ==========

# Model Configuration
MODEL_ID = "google/ddpm-celebahq-256"  # Pretrained DDPM from Hugging Face
IMAGE_SIZE = 256                        # Image resolution (256x256)

# DPS Parameters
NUM_DIFFUSION_STEPS = 1000              # Number of diffusion steps
GUIDANCE_SCALE = 1.0                    # Measurement guidance strength
ZETA = 0.5                              # Step size for gradient updates

# Blur Parameters
BLUR_KERNEL_SIZE = 15                   # Size of motion blur kernel
BLUR_ANGLE = 45                         # Angle of motion blur (degrees)
NOISE_LEVEL = 0.01                      # Gaussian noise std added to measurement

# Paths
DATA_DIR = "./data"                     # Directory for CelebA images
OUTPUT_DIR = "./outputs"                # Directory for results
MASK_DIR = "./masks"                    # Directory for anomaly masks (if available)

# Evaluation
EVAL_NUM_SAMPLES = 100                  # Number of images for AUC evaluation
GENERATE_SYNTHETIC = True               # Generate synthetic artifacts for testing

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====================================
```

### Key Parameters Explained

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `NUM_DIFFUSION_STEPS` | Reverse diffusion iterations | 1000 (default), 500 (faster) |
| `GUIDANCE_SCALE` | Strength of measurement conditioning | 0.5 - 2.0 |
| `ZETA` | Learning rate for gradient step | 0.1 - 1.0 |
| `BLUR_KERNEL_SIZE` | Motion blur kernel size (odd number) | 9, 15, 21 |
| `BLUR_ANGLE` | Direction of motion blur | 0° - 180° |
| `NOISE_LEVEL` | Measurement noise standard deviation | 0.01 - 0.05 |

---

## 📖 Usage

### Basic Workflow

The notebook follows this pipeline:

```
1. Load Pretrained Model
   ↓
2. Load/Generate Test Image
   ↓
3. Create Blurry Measurement (Forward Operator)
   ↓
4. Initialize Diffusion Process
   ↓
5. Run DPS Deblurring (Reverse Process)
   │  ├─ Track measurement updates at each step
   │  └─ Apply gradient-based corrections
   ↓
6. Generate Anomaly Heatmap
   │  └─ Aggregate measurement updates
   ↓
7. Compute AUC (if ground truth masks available)
   ↓
8. Visualize Results
```

### Step-by-Step Execution

#### 1️⃣ **Load the Model**

```python
from diffusers import DDPMPipeline

# Load pretrained CelebA-HQ DDPM
pipeline = DDPMPipeline.from_pretrained(MODEL_ID)
pipeline = pipeline.to(DEVICE)
```

#### 2️⃣ **Create Blurry Measurement**

```python
# Apply motion blur + noise
blurry_image = apply_motion_blur(clean_image, kernel_size=15, angle=45)
blurry_image = add_gaussian_noise(blurry_image, noise_level=0.01)
```

#### 3️⃣ **Run DPS Deblurring**

```python
# Initialize and run reverse diffusion
restored_image, measurement_updates = dps_deblur(
    blurry_measurement=blurry_image,
    pipeline=pipeline,
    num_steps=1000,
    guidance_scale=1.0
)
```

#### 4️⃣ **Generate Heatmap**

```python
# Compute anomaly heatmap from updates
heatmap = compute_anomaly_heatmap(measurement_updates)
```

#### 5️⃣ **Evaluate AUC**

```python
# If ground truth masks are available
auc_score = compute_auc(heatmap, ground_truth_mask)
print(f"AUC: {auc_score:.4f}")
```

---

## 📊 Output

The notebook generates the following outputs in `./outputs/`:

### Visualizations

1. **`original_image.png`**: Clean input image (if available)
2. **`blurry_measurement.png`**: Degraded observation with motion blur
3. **`restored_image.png`**: Deblurred result from DPS
4. **`anomaly_heatmap.png`**: Heat map highlighting anomalous regions
5. **`comparison_grid.png`**: Side-by-side comparison of all results

### Metrics

- **`auc_scores.txt`**: AUC metrics for anomaly detection
- **`experiment_log.json`**: Full configuration and results log

### Example Output

```
outputs/
├── experiment_001/
│   ├── original_image.png
│   ├── blurry_measurement.png
│   ├── restored_image.png
│   ├── anomaly_heatmap.png
│   ├── comparison_grid.png
│   └── metrics.json
```

---

## 📚 Citations

This implementation builds upon the following research papers. **Please cite them if you use this code**:

### 1. Diffusion Posterior Sampling (DPS)

```bibtex
@inproceedings{chung2023diffusion,
  title={Diffusion posterior sampling for general noisy inverse problems},
  author={Chung, Hyungjin and Kim, Jeongsol and Mccann, Michael T and Klasky, Marc L and Ye, Jong Chul},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2023}
}
```

**Paper**: [Diffusion Posterior Sampling for General Noisy Inverse Problems (ICLR 2023)](https://openreview.net/forum?id=OnD9zGAGT0k)

### 2. Denoising Diffusion Probabilistic Models (DDPM)

```bibtex
@inproceedings{ho2020denoising,
  title={Denoising diffusion probabilistic models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  booktitle={Advances in Neural Information Processing Systems},
  volume={33},
  pages={6840--6851},
  year={2020}
}
```

**Paper**: [Denoising Diffusion Probabilistic Models (NeurIPS 2020)](https://arxiv.org/abs/2006.11239)

### 3. CelebA-HQ Dataset

```bibtex
@inproceedings{karras2018progressive,
  title={Progressive growing of GANs for improved quality, stability, and variation},
  author={Karras, Tero and Aila, Timo and Laine, Samuli and Lehtinen, Jaakko},
  booktitle={International Conference on Learning Representations},
  year={2018}
}

@inproceedings{liu2015faceattributes,
  title={Deep learning face attributes in the wild},
  author={Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
  booktitle={Proceedings of International Conference on Computer Vision (ICCV)},
  year={2015}
}
```

**Dataset**: [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans) and [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

### 4. Diffusers Library

```bibtex
@misc{von-platen-etal-2022-diffusers,
  author = {Patrick von Platen and Suraj Patil and Anton Lozhkov and Pedro Cuenca and Nathan Lambert and Kashif Rasul and Mishig Davaadorj and Thomas Wolf},
  title = {Diffusers: State-of-the-art diffusion models},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/diffusers}}
}
```

---

## 🔬 Methodology

### DPS Algorithm Overview

The Diffusion Posterior Sampling method solves inverse problems by:

1. **Forward Process**: Simulate image degradation (blur + noise)
2. **Reverse Diffusion**: Iteratively denoise using pretrained DDPM
3. **Measurement Guidance**: At each step, compute gradient of measurement loss
4. **Posterior Correction**: Update sample to be consistent with observation

**Mathematical Formulation**:

```
x̂_{t-1} = μ_θ(x_t, t) + Σ_θ(x_t, t) · [∇_{x_t} log p(y|x_0(x_t)) + z_t]
```

Where:
- `y` is the blurry measurement
- `x_t` is the noisy sample at step `t`
- `μ_θ, Σ_θ` are learned mean and variance
- The gradient term enforces data consistency

---

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Reduce batch size or image resolution
IMAGE_SIZE = 128  # Instead of 256
```

**2. Model Download Fails**
```bash
# Manually download model
huggingface-cli download google/ddpm-celebahq-256
```

**3. Import Errors**
```bash
# Reinstall dependencies
pip install --upgrade diffusers transformers
```

**4. Slow Inference**
```python
# Reduce diffusion steps
NUM_DIFFUSION_STEPS = 500  # Faster, slightly lower quality
```

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` file for details.

**Note**: The pretrained models and datasets have their own licenses:
- DDPM Model: Apache 2.0
- CelebA Dataset: Non-commercial research only

---

## 🙏 Acknowledgments

- **Hugging Face** for the Diffusers library and pretrained models
- **KAIST AIMLab** for the DPS algorithm
- **NVIDIA** for CelebA-HQ dataset curation

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].

---

## 🔗 Additional Resources

- [DPS Official Repository](https://github.com/DPS2022/diffusion-posterior-sampling)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers/index)
- [CelebA-HQ Dataset](https://github.com/tkarras/progressive_growing_of_gans)

---

**Last Updated**: March 2024
