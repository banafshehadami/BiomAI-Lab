# rPPG-SysDiaGAN (PyTorch)

**Systolic–Diastolic Feature Localization in rPPG Using GAN with Multi-Domain Discriminators**

This repository provides a clean PyTorch reimplementation of the rPPG-SysDiaGAN framework described in the paper. The goal is to reconstruct photoplethysmography (rPPG) signals from facial video while accurately preserving systolic and diastolic features through multi-domain adversarial training.

---

## Key Features

- **Generator: Swin-AUNet** — 3D U-Net backbone with attention gates; temporal Swin-style self-attention at the bottleneck; maps 3D video inputs → 1D rPPG via spatial global average pooling.  
- **Discriminators: Multi-Domain PatchGANs** — time domain (raw rPPG sequences), second derivative domain (SDPPG, morphology), frequency domain (db4 wavelet coefficient maps).  
- **Losses** — Soft-DTW (differentiable sequence alignment), sparsity constraints across time/SD/frequency domains, differentiable CDF-variance loss, adversarial loss.  
- **Signal Utilities** — second derivative transform, wavelet-based maps, heart-rate estimation; metrics: RMSE, Pearson correlation, Fréchet distance.  
- **Training Strategy** — multi-domain loss weighting (α=1.5, β=0.8, γ=1.2), Adam optimizer with gradient clipping, YAML configuration.

---

## Repository Structure

```
rrppg-sysdiagan/
├─ README.md
├─ requirements.txt
├─ configs/
│  └─ default.yaml
├─ src/rrppg_gan/
│  ├─ __init__.py
│  ├─ datasets.py
│  ├─ losses.py
│  ├─ signal_utils.py
│  └─ models/
│     ├─ generator.py
│     ├─ attention.py
│     └─ discriminators.py
├─ train.py
├─ eval.py
└─ LICENSE
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scriptsctivate
pip install -r requirements.txt
```

---

## Dataset Preparation

The model expects **pre-cropped face clips** (e.g., 128×128 RGB frames) with aligned ground-truth PPG signals.

Structure:

```
DATA_ROOT/
  subject_001/
    clip_000.pt   # dict: {"video": (T,3,H,W), "ppg": (T,)}
    clip_001.pt
  subject_002/
    ...
```

Each `.pt` file contains:
- `video`: float tensor `(T, 3, H, W)` in [0,1]
- `ppg`: float tensor `(T,)` ground-truth PPG

Preprocessing can be done with OpenFace or similar face-tracking pipelines.

---

## Training

```bash
python train.py --config configs/default.yaml --data_root /path/to/DATA_ROOT --out runs/exp1
```

- Checkpoints saved to `runs/exp1/`  
- Best checkpoint selected by lowest generator loss

---

## Evaluation

```bash
python eval.py --ckpt runs/exp1/best.ckpt --data_root /path/to/DATA_ROOT
```

Reports dataset-wide averages of:
- Pearson correlation (R)
- RMSE
- Fréchet distance

---

## Configuration

Key options in `configs/default.yaml`:

- `model.*` — generator and attention settings  
- `loss.*` — α, β, γ weights; Soft-DTW gamma; CDF bins  
- `train.*` — batch size, learning rate, epochs, sample rate, etc.

---

## Citation

If you use this codebase, please cite:

```bibtex
@inproceedings{adami2024rppg,
  title={rPPG-SysDiaGAN: Systolic-Diastolic Feature Localization in rPPG Using Generative Adversarial Network with Multi-domain Discriminator},
  author={Adami, Banafsheh and Karimian, Nima},
  booktitle={European Conference on Computer Vision},
  pages={193--210},
  year={2024}
}

```

---

## License

Released under the MIT License. See `LICENSE` for details.
