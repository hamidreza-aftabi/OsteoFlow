# OsteoFlow: Lyapunov-Guided Flow Distillation for Predicting Bone Remodeling after Mandibular Reconstruction

[![arXiv](https://img.shields.io/badge/arXiv-2603.22421-b31b1b.svg)](https://arxiv.org/abs/2603.22421)
[![arXiv version](https://img.shields.io/badge/arXiv%20version-v1-b31b1b.svg)](https://arxiv.org/abs/2603.22421v1)
[![License](https://img.shields.io/badge/license-PolyForm%20NC%201.0.0-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-required-EE4C2C.svg)](requirements.txt)

OsteoFlow is a teacher-student framework for predicting Year-1 post-operative
CT from Day-5 CT after mandibular reconstruction. It combines diffeomorphic
registration and rectified flow modeling to learn bone remodeling at the
graft-host interface. During training, a registration-based teacher provides
trajectory supervision, while the student learns an image-space transport field
with Lyapunov regularization. At inference time, only the student model is used.

Publication: [OsteoFlow on ResearchGate](https://www.researchgate.net/publication/403111765_OsteoFlow_Lyapunov-Guided_Flow_Distillation_for_Predicting_Bone_Remodeling_after_Mandibular_Reconstruction).

## Repository Layout

- `code/` - teacher and student OsteoFlow training scripts.
- `assets/` - README figures.
- `requirements.txt` - Python dependencies.

## Method Overview

![Method overview](assets/method.png)

OsteoFlow has two stages:

- **Teacher:** diffeomorphic registration with stationary velocity fields
  (SVF) to generate trajectory supervision.
- **Student:** rectified flow trained with teacher guidance and Lyapunov
  regularization.

Only the student model is needed at test time.

## Results Preview

![Qualitative results](assets/results.png)

Representative predictions for union, partial union, and nonunion cases are
shown on the resection plane and orthogonal central slices.

## Setup

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the teacher model:

```bash
python code/OsteoFlow_Teacher_V0.py
```

Run the student model:

```bash
python code/OsteoFlow_Student_V0.py
```

## Model Input and Resolution

- Input volume size: `48 x 48 x 48` ROI patch.
- Input/output channels: one CT volume channel.
- Intensity normalization: HU clipped to `[-100, 1100]` and mapped to `[-1, 1]`.
- Student UNet spatial path: `48 -> 24 -> 12 -> 24 -> 48`.
- Inference integration: one-step update or multi-step ODE integration
  (`Euler`, `Heun`, or `RK4`).

## Configurable Parameters

`LOSS_MODE` controls the training objective:

- `fm_only` - rectified flow only.
- `lqr_only` - Lyapunov-guided teacher only.
- `both` - joint training.

```python
LOSS_MODE = 'both'  # 'both' | 'lqr_only' | 'fm_only'
```

## Baseline Implementation Notes

- Baselines follow the same ROI-level split rule: augmented units for training
  and `aug0`-only units for testing.
- **MedVAE-3D:** Fine-tuned from the released
  [MedVAE](https://github.com/StanfordMIMI/MedVAE) `medvae_4x_1c_3d_finetuning`
  checkpoint for CT modality using the MedVAE-recommended loss setup
  (`L1 + LPIPS + PatchGAN`); a resection-aware loss ablation was also tested.
- **cDDPM(delta)-3D:** MONAI-based conditional DDPM conditioned on POD5,
  trained to predict `Delta = POY1 - POD5`, and sampled with DDIM inference.
- **Pix2Pix-3D:** 3D POD5-to-POY1 ROI translation adapted from
  [pix2pix](https://github.com/phillipi/pix2pix), with a 3D ResUNet generator,
  3D PatchGAN discriminator, adversarial BCE loss, and `L1` reconstruction.
- **GRIT-3D adapted:** 3D adaptation based on the GRIT paper/project page; this
  is not a strict reproduction because the official code was unavailable and
  the adaptation uses reconstruction-plus-residual GAN training without the
  original style encoder pathway.
- **SegGuidedDiff-3D adapted:** 3D POD5-to-POY1 diffusion guided by a Day-5
  bone mask derived from POD5 through concatenation-based guidance; the same
  ablation mask was applied using the Day-5-derived mask.
- **Rectified Flow:** Based on
  [RectifiedFlow](https://github.com/gnobitab/RectifiedFlow), which also
  underlies the OsteoFlow student formulation.

## Reproducibility Notes

- Keep configured directory names consistent under `BASE_DIR`.
- The teacher is used only during training; inference uses the student alone.

## Data and Checkpoints

Pretrained checkpoints will be released in this repository.

The dataset used in this study is internal and cannot be publicly distributed.
Access requests may be directed to the corresponding author.

## Citation

If you use OsteoFlow, please cite:

```bibtex
@misc{aftabi2026osteoflow,
  title = {OsteoFlow: Lyapunov-Guided Flow Distillation for Predicting Bone Remodeling after Mandibular Reconstruction},
  author = {Aftabi, Hamidreza and Yu, Faye and Switzer, Brooke and Fishman, Zachary and Prisman, Eitan and Hodgson, Antony and Whyne, Cari and Fels, Sidney and Hardisty, Michael},
  year = {2026},
  eprint = {2603.22421},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  doi = {10.48550/arXiv.2603.22421},
  url = {https://www.researchgate.net/publication/403111765_OsteoFlow_Lyapunov-Guided_Flow_Distillation_for_Predicting_Bone_Remodeling_after_Mandibular_Reconstruction}
}
```

## License

OsteoFlow is source-available for noncommercial research and educational use
under the [PolyForm Noncommercial License 1.0.0](LICENSE). Commercial use
requires separate written permission from the authors. See [NOTICE](NOTICE) for
the required copyright notice.
