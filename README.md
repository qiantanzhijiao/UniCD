# UniCD-PR: Detection-Friendly NUC with Parameter-Residual Dual Decomposition

Reproduction and improvement of **UniCD** (CVPR 2025) — *Detection-Friendly
Nonuniformity Correction: A Union Framework for Infrared UAV Target Detection*.

This repository contains the complete training, evaluation, and benchmarking
code for **UniCD-PR**, which augments the polynomial bias model with a
lightweight convolutional residual branch and a detection-driven gate.

## Key Features

- **Parameter-residual dual decomposition**: interpretable polynomial
  correction + free-form residual branch for non-polynomial bias (stripes,
  compression artifacts)
- **Detection-driven gate**: adaptively controls residual strength per image
  to preserve detection-relevant content
- **Perturbation regularization**: improves robustness to unmodeled
  nonuniformity patterns
- **Zero parameter overhead**: residual branch shares the polynomial
  parameterization (0.458M params, 136.7 MFLOPs, 114 FPS on A40 GPU)

## Repository Structure

```
UniCD/
├── unied/                  # Core library
│   ├── models/             #   NUC model, detector backbone, heads
│   ├── datasets/           #   IRBFD dataset loader, coefficient fitting
│   ├── losses/             #   TEBs loss, detection loss
│   ├── poly.py             #   Polynomial bias model
│   ├── augment.py          #   Perturbation augmentation
│   └── config.py           #   YAML config parser
├── tools/                  # Entry-point scripts
│   ├── train.py            #   Training (pretrain / joint / real-finetune)
│   ├── eval_nuc.py         #   NUC evaluation (PSNR / SSIM / SCRG)
│   ├── eval_det.py         #   Detection evaluation (AP50 / P / R)
│   ├── benchmark_speed.py  #   Params / FLOPs / Latency / FPS
│   ├── classical_nuc_baseline.py  # MM / TPC / FPR classical baselines
│   ├── eval_gate_analysis.py      # Gate causality analysis
│   └── prepare_coefs.py    #   Polynomial coefficient pre-fitting
├── configs/                # Baseline (poly-only) configs
├── configs_improved/       # UniCD-PR configs (full + ablations)
├── tests/                  # Smoke tests and data verification
└── scripts (*.sh)          # Pipeline automation scripts
```

## Data

The experiments use the **IRBFD** benchmark (Fang et al., CVPR 2025),
which provides infrared UAV images with synthetic and real nonuniformity.

### Data Sources

| Dataset | Description | Location (local) |
|---|---|---|
| **IRBFD-syn** | Synthetic nonuniformity, VOC2007 format | `<DATA_ROOT>/IRBFD-syn/VOC2007/` |
| **IRBFD-real** | Real nonuniformity, VOC2007 format | `<DATA_ROOT>/IRBFD-real/VOC2007/` |
| **Coefficient cache** | Pre-fitted polynomial coefficients | `<DATA_ROOT>/UniCD-repro/outputs/coef_cache_syn.npz` |

> **Note**: The IRBFD dataset is distributed by the original UniCD authors.
> Request access from the original UniCD paper (CVPR 2025) repository.

### Directory Layout Expected by Configs

```
<DATA_ROOT>/
├── IRBFD-syn/VOC2007/
│   ├── JPEGImages-bias/     # Corrupted images (input)
│   ├── JPEGImages-clear/    # Clean images (ground truth)
│   ├── Annotations/         # XML bounding box labels
│   └── ImageSets/Main/      # train/val/test splits
├── IRBFD-real/VOC2007/
│   ├── JPEGImages-bias/
│   ├── JPEGImages-clear/
│   ├── Annotations/
│   └── ImageSets/Main/
└── UniCD-repro/
    └── outputs/coef_cache_syn.npz
```

### Server-Specific Data Paths

| Server | Data Root |
|---|---|
| Server 1 (A40-8Q) | `/root/work/UniCD/` |
| Server 2 (RTX 3060) | `/home/jgxy/work/UniCD/` |

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.12 (CUDA recommended)
- See `requirements.txt` for full dependencies

```bash
conda create -n unied python=3.10
conda activate unied
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Offline / Older Environments

For Python 3.6 or offline clusters, see `requirements_offline_py36.txt`.

## Configuration

All experiment parameters are defined in YAML config files. The key fields:

### Data Configuration

```yaml
data:
  syn_root: /path/to/IRBFD-syn/VOC2007     # Synthetic dataset root
  real_root: /path/to/IRBFD-real/VOC2007   # Real dataset root
  coef_cache: /path/to/coef_cache_syn.npz   # Pre-fitted polynomial coefficients
  batch_size: 4
  num_workers: 8
  hflip: 0.5                                 # Horizontal flip augmentation
  scale: 1.0                                 # Scale augmentation
```

### Model Configuration

```yaml
model:
  degree: 3               # Polynomial degree (D=3, 10 coefficients)
  residual:
    enabled: true          # Enable residual branch
    gate: true             # Enable detection-driven gate
```

### Training Configuration

```yaml
train:
  stage: joint             # pretrain / joint / real
  epochs: 50
  lr: 0.001
  weight_decay: 0.0001
  lambda_tebs_first: 1.0   # TEBs loss weight (phase 1)
  lambda_tebs_after: 0.01  # TEBs loss weight (phase 2)
  lambda_switch_epoch: 20  # Phase switch epoch
  use_br: true             # Bias regularization
  lambda_res: 1.0          # Residual loss weight
  seed: 42
  resume: path/to/checkpoint.pt  # Pretrained checkpoint
```

## Usage

### 1. Pre-fit Polynomial Coefficients

```bash
python tools/prepare_coefs.py \
    --data-root /path/to/IRBFD-syn/VOC2007 \
    --output outputs/coef_cache_syn.npz \
    --degree 3
```

### 2. Training Pipeline

**Full UniCD-PR (recommended):**

```bash
# Stage 1: NUC pretrain (30 epochs)
python tools/train.py --config configs_improved/pretrain_pr.yaml

# Stage 2: Joint training with detection (50 epochs)
python tools/train.py --config configs_improved/joint_pr.yaml

# Stage 3: Real-data fine-tuning (30 epochs)
python tools/train.py --config configs_improved/finetune_real_pr.yaml
```

**Baseline (poly-only):**

```bash
python tools/train.py --config configs/pretrain_nuc.yaml
python tools/train.py --config configs/joint_train.yaml
python tools/train.py --config configs/finetune_real.yaml
```

**Ablation experiments:**

```bash
# Residual branch only (no gate)
python tools/train.py --config configs_improved/joint_pr_nogate.yaml

# Residual + gate (no perturbation)
python tools/train.py --config configs_improved/joint_pr_noperturb.yaml
```

Or run the automated pipeline:

```bash
bash run_improved_pipeline.sh   # Full improved pipeline
bash start_joint.sh             # Baseline + ablation pipeline
```

### 3. Evaluation

```bash
# NUC quality (PSNR / SSIM / SCRG)
python tools/eval_nuc.py --config configs_improved/eval_pr.yaml \
    --ckpt outputs_improved/joint_pr/epoch_050.pt

# Detection performance (AP50 / P / R)
python tools/eval_det.py --config configs_improved/eval_pr.yaml \
    --ckpt outputs_improved/joint_pr/epoch_050.pt --mode syn
python tools/eval_det.py --config configs_improved/eval_pr.yaml \
    --ckpt outputs_improved/joint_pr/epoch_050.pt --mode real

# Classical NUC baselines (MM / TPC / FPR)
python tools/classical_nuc_baseline.py --data-root /path/to/IRBFD-syn/VOC2007 \
    --output results/classical_nuc.json

# Speed benchmark (params / FLOPs / FPS)
python tools/benchmark_speed.py --config configs_improved/eval_pr.yaml \
    --ckpt outputs_improved/joint_pr/epoch_050.pt \
    --output results/speed_benchmark.json
```

### 4. Gate Causality Analysis

```bash
python tools/eval_gate_analysis.py --config configs_improved/eval_pr.yaml \
    --ckpt outputs_improved/joint_pr/epoch_050.pt
```

## Main Results

### NUC Quality (IRBFD-syn test, 2880 images)

| Method | PSNR (dB) | SSIM | SCRG |
|---|---|---|---|
| No correction | 11.40 | 0.640 | 0.99 |
| Moment Matching | 11.58 | 0.670 | 1.00 |
| Two-Point (oracle) | 16.14 | 0.690 | 0.25 |
| Fixed-Pattern Removal | 12.90 | 0.670 | 1.00 |
| UniCD (baseline) | 11.38 | 0.639 | 1.00 |
| **UniCD-PR (ours)** | **17.44** | **0.736** | **1.60** |

### Detection Performance

| Method | AP50 (syn) | AP50 (real) | P (real) | R (real) |
|---|---|---|---|---|
| UniCD (baseline) | 0.010 | 0.605 | 0.872 | 0.638 |
| **UniCD-PR (ours)** | **0.432** | **0.840** | **0.863** | **0.869** |

### Ablation Study

| Configuration | PSNR (dB) | AP50 (syn) | AP50 (real) |
|---|---|---|---|
| Baseline (poly only) | 11.38 | 0.010 | 0.605 |
| + Residual (no gate) | 17.66 | 0.443 | — |
| + Residual + Gate (no pert.) | 17.36 | 0.444 | — |
| + Full (ours) | 17.44 | 0.432 | **0.840** |

### Efficiency

| Model | Params (M) | FLOPs (M) | FPS (A40) |
|---|---|---|---|
| Baseline | 0.458 | 136.7 | 118 |
| UniCD-PR | 0.458 | 136.7 | 114 |

## Citation

If you use this code, please cite:

```bibtex
@article{yang2026unicdpr,
  title={Detection-Friendly Nonuniformity Correction with Parameter-Residual
         Dual Decomposition for Infrared UAV Target Detection},
  author={Yang, Peide and Ye, Xiubin and Dong, Han},
  journal={Infrared Physics \& Technology},
  year={2026}
}
```

And the original UniCD paper:

```bibtex
@inproceedings{fang2025unicd,
  title={Detection-Friendly Nonuniformity Correction: A Union Framework for
         Infrared UAV Target Detection},
  author={Fang, Zhen and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and
             Pattern Recognition (CVPR)},
  pages={11898--11907},
  year={2025}
}
```

## License

For academic research use only.

## Contact

- Peide Yang — yangpd@xujc.com
- Xiubin Ye — yexiubin@xujc.com
- Han Dong — donghan@mail.nankai.edu.cn

Key Laboratory of Intelligent Manufacturing Equipment and Industrial Internet
Technology, Fujian Provincial Universities / School of Electronic Science and
Technology, Xiamen University Tan Kah Kee College, Xiamen, China.
