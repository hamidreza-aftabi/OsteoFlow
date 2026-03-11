# OsteoFlow: Lyapunov-Guided Flow Distillation for Predicting Bone Remodeling after Mandibular Reconstruction

OsteoFlow is a teacher–student framework for predicting Year-1 post-operative CT from Day-5 CT after mandibular reconstruction. It combines diffeomorphic registration and rectified flow modeling to learn bone remodeling at the graft–host interface. During training, a registration-based teacher provides trajectory supervision, while the student learns an image-space transport field with Lyapunov regularization. At inference time, only the student model is used.

## Method Overview

The framework has two stages:

- **Teacher:** diffeomorphic registration with stationary velocity fields (SVF) to generate supervision trajectories
- **Student:** rectified flow trained with teacher guidance and Lyapunov regularization

Only the student is needed at test time.

### Framework
![Method overview](assets/method.png)

Overview of the preprocessing pipeline and teacher–student distillation framework used to guide the student velocity field.

### Qualitative Results
![Qualitative results](assets/results.png)

Representative predictions for union, partial union, and nonunion cases, shown on the resection plane and orthogonal central slices.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the teacher model:

```bash
python Code/OsteoFlow_Teacher_V0.py
```

Run the student model:

```bash
python Code/OsteoFlow_Student_V0.py
```

## Repository Structure

```text
OsteoFlow/
├── README.md
├── requirements.txt
├── Code/
│   ├── OsteoFlow_Teacher_V0.py
│   └── OsteoFlow_Student_V0.py
└── assets/
    ├── method.png
    └── results.png
```

## Configurable Parameters

Some parameters and flags can be changed in the code.

In particular, `LOSS_MODE` controls the training setup:

- `fm_only` — rectified flow only
- `lqr_only` — Lyapunov-guided teacher only
- `both` — joint training

```python
# The parameters and flags in this code can be adjusted depending on the training setup.
# LOSS_MODE controls which supervision is used:
#   'fm_only'  -> rectified flow only
#   'lqr_only' -> Lyapunov-guided teacher only
#   'both'     -> joint training
LOSS_MODE = 'both'  # 'both' | 'lqr_only' | 'fm_only'
```

## Data and Checkpoints

Pretrained checkpoints will be released here.

The dataset used in this study is internal and cannot be publicly distributed. In the case of acceptance, access requests may be directed to the corresponding author.

## Reproducibility Notes

- Keep directory names consistent under `BASE_DIR` if paths are modified.
- The teacher is used only during training.
- Inference uses the student model alone.

## Citation

If you use this repository in your research, please cite the corresponding paper once available.

```bibtex
@article{osteoflow2026,
  title={OsteoFlow: Lyapunov-Guided Flow Distillation for Predicting Bone Remodeling after Mandibular Reconstruction},
  author={Aftabi et al.},
  journal={arXiv},
  year={2026}
}
```
