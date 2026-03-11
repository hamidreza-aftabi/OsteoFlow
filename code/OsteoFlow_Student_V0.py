# =============================================================
# OsteoFlow_Student — Rectified Flow + Lyapunov in IMAGE SPACE
# =============================================================

import os, re, math, csv, sys
import random
from contextlib import contextmanager
import numpy as np
from pathlib import Path

import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler

# ===================== MODE SELECTION =====================
# Change MODE to control script behavior
MODE = 'train'  # 'train' | 'inference'

# ===================== LOCAL PATHS =====================
BASE_DIR = Path(__file__).resolve().parent.parent
AUG_ROOT = BASE_DIR / "output_rois_augmented"
POD5_DIR = AUG_ROOT / "POD5"
POY1_DIR = AUG_ROOT / "POY1"
OUT_ROOT = BASE_DIR / "OsteoFlow_Student"
FM_OUT_DIR = OUT_ROOT / "FM_results"


# Create output directories
FM_OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"📁 POD5 directory: {POD5_DIR}")
print(f"📁 POY1 directory: {POY1_DIR}")
print(f"📊 Output directory: {OUT_ROOT}")

# Keep a reference to the ORIGINAL Day5 directory for visualization.
# When one-shot preprocessing is enabled, training/inference will use the cached
# modified POD5 volumes, but we still want to visualize the true original Day5.
POD5_DIR_ORIGINAL = Path(POD5_DIR)

# ----------------------- Device & Seeds -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0); np.random.seed(0)


# ----------------------- Configuration -----------------------
ROI_SHAPE = (48, 48, 48)
HU_RANGE = (-100, 1100)  # Standard 12-bit CT range (matches augmentation clipping)
METRICS_BONE_HU_THRESHOLD = 300  # HU threshold for bone segmentation (DICE, HD95, etc.)

IMAGE_CHANNELS = 1
IMAGE_SPATIAL_SIZE = 48

# Metrics/Logging
EXCEL_UPDATE_INTERVAL = 1
EVAL_NUM_WORKERS = 0
COMPUTE_TRAIN_METRICS = False
COMPUTE_TEST_METRICS = True
EVAL_AT_EPOCH0 = True


# Flow training
BATCH_SIZE = 1
EVAL_BATCH_SIZE = BATCH_SIZE
LEARNING_RATE = 1e-4
USE_COSINE_LR = True
WARMUP_FRACTION = 0.05
NUM_EPOCHS = 50

# UNet architecture
UNET_USE_ATTENTION = False
UNET_BASE_CHANNELS = 48

# Memory & Checkpointing
USE_AMP = True
USE_EMA = False
EMA_DECAY = 0.999
EVAL_USE_EMA = False
USE_GRADIENT_CHECKPOINTING = False
CKPT_SAVE_UNIT = 'epoch'
CKPT_SAVE_INTERVAL = 10
RECON_SAVE_UNIT = 'epoch'
RECON_SAVE_INTERVAL = 1
NUM_SAMPLES_TO_SAVE = 0
RESUME_FROM_CHECKPOINT = False

# Inference / ODE integration
USE_DIRECT_ONE_STEP_INFERENCE = False  # True: single-step x1=x0+v(x0,0); False: multi-step ODE
INTEGRATION_METHOD = 'rk4'            # 'euler' | 'heun' | 'rk4'
EVAL_INTEGRATION_STEPS = 10           # ODE steps when USE_DIRECT_ONE_STEP_INFERENCE=False

# ===================== PLATE MASK HANDLING =====================
EXCLUDE_PLATE_FROM_LOSS = False
EXCLUDE_PLATE_FROM_METRICS = False

# Train/Test Split
SPLIT_BY_PATIENT = True  # True: by patient, False: by ROI
TRAIN_SPLIT = 0.85
TEST_SPLIT = 0
RANDOM_SEED = 123

# Semi-online augmentation
USE_SEMI_ONLINE_AUG = False
NUM_AUG_PER_ROI = 40


# Image Space Loss
USE_IMAGE_SPACE_LOSS = True
USE_IMG_LOSS_BONE_WEIGHTED = True
IMAGE_SPACE_LOSS_WEIGHT = 1.0
IMAGE_SPACE_LOSS_FREQ = 1

# Bone-weighted loss
BONE_LOSS_LAMBDA = 10.0
BONE_WEIGHT_ALPHA = 1.0
BONE_SURFACE_WEIGHT = 0.5

# ================= FM Resection Plane Weighting =================
FM_RESECTION_PLANE_CONSTRAINT = True
FM_RESECTION_PLANE_SIGMA = 30.0

# ================= Middle-Slab Prior Channel =================
USE_MIDDLE_SLAB_PRIOR_CHANNEL = True
MIDDLE_SLAB_PRIOR_MODE = 'concat'  # 'concat' | 'controlnet'
USE_MIDDLE_SLAB_PRIOR_MULTI_STAGE_INJECTION = False
MIDDLE_SLAB_IMAGE_SLICE_START = 18
MIDDLE_SLAB_IMAGE_SLICE_END = 30
MIDDLE_SLAB_PROFILE = 'cosine'  # 'cosine' | 'gaussian'
MIDDLE_SLAB_FALLOFF = 5.0

# ================= RF + Analytical Lyapunov =================
LOSS_MODE = 'both'  # 'both' | 'lqr_only' | 'fm_only'

# FM loss weight decay schedule
FM_LOSS_DECAY_ENABLE = True
FM_LOSS_DECAY_START_EPOCH = 10
FM_LOSS_DECAY_END_EPOCH = 15
FM_LOSS_DECAY_SHAPE = 'linear'  # 'linear' | 'cosine'

LYAPUNOV_ENABLED = True
USE_ANALYTICAL_LYAPUNOV = LYAPUNOV_ENABLED
LYAPUNOV_ALPHA = 1.0
LYAPUNOV_LAMBDA_MAX = 1.0
LYAPUNOV_WARMUP_EPOCHS = 10
LYAPUNOV_DT = 0.05
LYAPUNOV_DT_MIN = LYAPUNOV_DT
LYAPUNOV_TEACHER_TANGENT_SCHEME = 'forward'  # 'forward' | 'centered' | 'forward2'
LYAPUNOV_TERMINAL_FADE_START = 0.75

# Student init from SVF teacher
INIT_STUDENT_FROM_SVF_TEACHER = True
INIT_STUDENT_FROM_SVF_TEACHER_MODE = 'middle3'  # 'all' | 'middle_only' | 'middle3' | 'input_only'
INIT_STUDENT_FROM_SVF_TEACHER_VERBOSE = True

# On-policy Lyapunov training
LYAPUNOV_ON_POLICY_TRAINING = True
LYAPUNOV_ON_POLICY_PROB = 1.0
LYAPUNOV_ON_POLICY_STEPS = 8

LYAPUNOV_R_INV = 1.0  # Legacy compat

# SVF Teacher
LYAPUNOV_SVF_TEACHER_CHECKPOINT = BASE_DIR / "Results" / "teacher_123_m100_1100_SVF_final.pth"
LYAPUNOV_SVF_TEACHER_BASE_CHANNELS = 48
LYAPUNOV_SVF_TEACHER_SS_SQUARINGS = 7
LYAPUNOV_SVF_TEACHER_FLOW_CAP = 7.0

print(f"🎯 FM Config: {IMAGE_CHANNELS}×{IMAGE_SPATIAL_SIZE}³ | UNet={UNET_BASE_CHANNELS}ch | BS={BATCH_SIZE} | Epochs={NUM_EPOCHS}")
print(f"   Lyapunov={LYAPUNOV_ENABLED} α={LYAPUNOV_ALPHA} λ_max={LYAPUNOV_LAMBDA_MAX} | Teacher: {LYAPUNOV_SVF_TEACHER_CHECKPOINT.name}")
print(f"   Split: {TRAIN_SPLIT:.0%}/{TEST_SPLIT:.0%} seed={RANDOM_SEED}")
print()


@contextmanager
def _preserve_rng_state():
    """Preserve/restore RNG state so optional steps can't perturb training/eval determinism."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = None
    if torch.cuda.is_available():
        try:
            cuda_states = torch.cuda.get_rng_state_all()
        except Exception:
            cuda_states = None
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            try:
                torch.cuda.set_rng_state_all(cuda_states)
            except Exception:
                pass

def _fm_loss_weight_factor(epoch: int) -> float:
    """Return FM loss weight in [0,1] (optionally decays FM over epochs)."""
    if not bool(globals().get('FM_LOSS_DECAY_ENABLE', False)):
        return 1.0
    start_e = int(globals().get('FM_LOSS_DECAY_START_EPOCH', 0))
    end_e = int(globals().get('FM_LOSS_DECAY_END_EPOCH', start_e))
    shape = str(globals().get('FM_LOSS_DECAY_SHAPE', 'linear')).lower().strip()

    e = int(epoch)
    if e < start_e:
        return 1.0
    if end_e <= start_e:
        return 0.0
    if e >= end_e:
        return 0.0

    p = (e - start_e) / float(max(1, end_e - start_e))  # in (0,1)
    if shape == 'linear':
        return float(1.0 - p)
    if shape == 'cosine':
        return float(0.5 * (1.0 + math.cos(math.pi * p)))
    raise ValueError(f"Unknown FM_LOSS_DECAY_SHAPE: {shape} (use 'linear' or 'cosine')")

# ----------------------- Utils -------------------------------
def _clip_and_norm_to_unit(v_np, hu_range=HU_RANGE):
    lo, hi = hu_range
    v = np.clip(v_np.astype(np.float32), lo, hi)
    v = 2.0 * (v - lo) / (hi - lo) - 1.0
    return v

def _maybe_resample_to_roi(v_np, roi_shape=ROI_SHAPE):
    """Safety check only: enforce expected ROI shape and do NOT resample.
    If the incoming volume isn't already the desired size, raise an error so
    data can be fixed upstream. This avoids any hidden interpolation.
    """
    if tuple(v_np.shape) != tuple(roi_shape):
        raise ValueError(
            f"Input volume shape {v_np.shape} does not match expected ROI_SHAPE {roi_shape}. "
            "Please resample upstream to avoid implicit interpolation."
        )
    return v_np

def denorm_to_hu(v_np: np.ndarray, hu_range=HU_RANGE):
    lo, hi = hu_range
    return (((v_np.astype(np.float32) + 1.0) * 0.5) * (hi - lo) + lo)


def hu_to_denorm_scalar(hu: float, hu_range=HU_RANGE) -> float:
    """Convert an HU threshold to normalized [-1,1] threshold using HU_RANGE."""
    lo, hi = hu_range
    hu = float(hu)
    return ((hu - lo) / (hi - lo)) * 2.0 - 1.0


def make_bone_mask_from_norm(x_norm: torch.Tensor, hu_threshold: float = METRICS_BONE_HU_THRESHOLD) -> torch.Tensor:
    """Hard bone mask from a normalized [-1,1] image tensor.

    Args:
        x_norm: [B,1,D,H,W] normalized to [-1,1]
        hu_threshold: bone threshold in HU (e.g., 300)
    Returns:
        mask: float mask [B,1,D,H,W] in {0,1}
    """
    thr = hu_to_denorm_scalar(float(hu_threshold), HU_RANGE)
    # x_norm expected in [-1,1]; thresholding happens in normalized space for GPU efficiency
    return (x_norm > thr).to(dtype=x_norm.dtype)


def masked_mean_abs(x: torch.Tensor, weight_map: torch.Tensor | None = None) -> torch.Tensor:
    """Per-sample mean(|x|) optionally restricted to a voxel mask.

    Args:
        x: [B, C, D, H, W]
        weight_map: [B,1,D,H,W] or [B,C,D,H,W] with nonnegative weights (typically 0/1)
    Returns:
        [B] tensor
    """
    x_abs = x.abs()
    if weight_map is None:
        return x_abs.flatten(1).mean(dim=1)

    w = weight_map
    if w.ndim == 4:
        w = w.unsqueeze(1)
    if w.ndim != 5:
        raise ValueError(f"weight_map must have shape [B,1,D,H,W] or [B,C,D,H,W] (got {tuple(w.shape)})")
    w = w.to(dtype=x_abs.dtype, device=x_abs.device)
    if w.shape[1] == 1 and x_abs.shape[1] != 1:
        w = w.expand(-1, x_abs.shape[1], -1, -1, -1)
    num = (x_abs * w).flatten(1).sum(dim=1)
    den = w.flatten(1).sum(dim=1)
    # If mask is empty for a sample, fall back to full-volume mean(|x|)
    full = x_abs.flatten(1).mean(dim=1)
    return torch.where(den > 1e-7, num / den.clamp_min(1e-8), full)


def velocity_magnitude_ratio_stable(
    v: torch.Tensor,
    v_teacher: torch.Tensor,
    weight_map: torch.Tensor | None = None,
    denom_floor: float = 1e-4,
) -> torch.Tensor:
    """Stable per-sample magnitude ratio using mean-absolute magnitudes.

    This avoids the heavy-tailed behavior of global L2 norms and reduces noise
    when teacher tangent magnitudes are very small.
    """
    num = masked_mean_abs(v, weight_map=weight_map)
    den = masked_mean_abs(v_teacher, weight_map=weight_map).clamp_min(float(denom_floor))
    return num / den


def velocity_scale_agreement_01(
    v: torch.Tensor,
    v_teacher: torch.Tensor,
    weight_map: torch.Tensor | None = None,
    denom_floor: float = 1e-4,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Bounded velocity-scale agreement in [0,1], best at 1.

    Let r = mean(|v|) / mean(|v_teacher|). We return min(r, 1/r).
    """
    r = velocity_magnitude_ratio_stable(v, v_teacher, weight_map=weight_map, denom_floor=denom_floor)
    r_inv = 1.0 / r.clamp_min(float(eps))
    return torch.minimum(r, r_inv)


def fm_create_resection_plane_mask(shape: tuple, sigma: float, device: torch.device | None = None) -> torch.Tensor:
    """FM training loss mask: smooth exp falloff along W axis.

    Weight formula (along W axis): w(n) = exp(-((n - center)/sigma)^2)
    where center = W/2.

    Hard-band (binary) option uses threshold tau=exp(-1) (≈0.3679).

    Args:
        shape: (D,H,W) or (B,C,D,H,W)
        sigma: falloff parameter in voxels
        device: torch device
    Returns:
        mask broadcastable to the given shape, values in [0,1]
    """
    if len(shape) == 5:
        B, C, D, H, W = shape
    elif len(shape) == 3:
        D, H, W = shape
        B, C = 1, 1
    else:
        raise ValueError(f"Unexpected shape for FM resection plane mask: {shape}")

    sigma = max(float(sigma), 0.1)
    center_n = W / 2.0

    w_indices = torch.arange(W, dtype=torch.float32, device=device)
    distance_from_center = (w_indices - center_n) / sigma
    weight_1d = torch.exp(-(distance_from_center ** 2))

    tau = math.exp(-1.0)


def fm_create_middle_slab_prior_mask(shape: tuple, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Create a smooth, deterministic middle-slab emphasis mask for the UNet input.

    The slab is defined in IMAGE slice indices (0..ROI_SHAPE[2]-1) along the W axis
    (the same axis used by the FM resection-plane mask and the script's "Axial" view),
    and mapped into the provided W dimension (typically IMAGE_SPATIAL_SIZE).

    Args:
        shape: (D,H,W) or (B,C,D,H,W) (C is ignored; output is (B,1,D,H,W))
        device: torch device
        dtype: dtype of returned tensor
    Returns:
        mask in [0,1], shape (B,1,D,H,W)
    """
    if len(shape) == 5:
        B, _, D, H, W = shape
    elif len(shape) == 3:
        D, H, W = shape
        B = 1
    else:
        raise ValueError(f"Unexpected shape for middle-slab prior mask: {shape}")

    device = device if device is not None else torch.device('cpu')
    dtype = dtype if dtype is not None else torch.float32

    # Define slab in IMAGE space (W axis, same as visualization's "Axial" slice index)
    # then resample to the requested latent W resolution. This guarantees the latent
    # prior has the exact same orientation as the image-space map.
    img_W = int(ROI_SHAPE[2])
    start_img = int(globals().get('MIDDLE_SLAB_IMAGE_SLICE_START', 20))
    end_img = int(globals().get('MIDDLE_SLAB_IMAGE_SLICE_END', 28))
    if end_img < start_img:
        start_img, end_img = end_img, start_img
    start_img = max(0, min(img_W - 1, start_img))
    end_img = max(0, min(img_W - 1, end_img))

    falloff = float(globals().get('MIDDLE_SLAB_FALLOFF', 2.0))
    falloff = max(falloff, 1e-6)
    profile = str(globals().get('MIDDLE_SLAB_PROFILE', 'cosine')).lower().strip()

    falloff_img = float(falloff) * (float(img_W) / float(max(int(W), 1)))
    falloff_img = max(falloff_img, 1e-6)

    idx_img = torch.arange(int(img_W), device=device, dtype=torch.float32)
    dist_img = torch.zeros_like(idx_img)
    dist_img = torch.where(idx_img < float(start_img), float(start_img) - idx_img, dist_img)
    dist_img = torch.where(idx_img > float(end_img), idx_img - float(end_img), dist_img)

    if profile == 'gaussian':
        w1d_img = torch.exp(-((dist_img / falloff_img) ** 2))
    elif profile == 'cosine':
        x = torch.clamp(dist_img / falloff_img, 0.0, 1.0)
        w1d_img = 0.5 * (1.0 + torch.cos(math.pi * x))
    else:
        raise ValueError(f"Unknown MIDDLE_SLAB_PROFILE: {profile} (use 'cosine' or 'gaussian')")

    # Resample image-space 1D prior to target W
    w1d_img_t = w1d_img.view(1, 1, int(img_W)).to(dtype=torch.float32)
    if int(img_W) != int(W):
        w1d_lat = F.interpolate(w1d_img_t, size=int(W), mode='linear', align_corners=False)
    else:
        w1d_lat = w1d_img_t
    w1d_lat = w1d_lat.view(int(W)).to(dtype=dtype)

    w = w1d_lat.view(1, 1, 1, 1, int(W)).expand(int(B), 1, int(D), int(H), int(W)).contiguous()
    return w


def load_raw_hu_nifti(path: Path) -> np.ndarray:
    """Load a NIfTI volume in raw HU space and enforce ROI_SHAPE (no implicit resample)."""
    img = nib.load(str(path))
    v = img.get_fdata().astype(np.float32)
    v = _maybe_resample_to_roi(v, ROI_SHAPE)
    return v


# ===================== Teacher Model =====================

class TeacherTimeEmbedding(nn.Module):
    """Sinusoidal time embedding with learned projection (Teacher FM style)."""
    def __init__(self, d_t=64, M=1000, out_channels=256):
        super().__init__()
        self.d_t = d_t
        self.M = M
        self.proj = nn.Sequential(
            nn.Linear(d_t, out_channels * 4),
            nn.GELU(),
            nn.Linear(out_channels * 4, out_channels)
        )
    
    def forward(self, t):
        h = self.d_t // 2
        freqs = torch.exp(-math.log(self.M) * torch.arange(h, device=t.device) / max(h - 1, 1))
        ang = (t[:, None] * self.M) * freqs[None]
        emb = torch.cat([ang.sin(), ang.cos()], dim=-1)
        if self.d_t % 2:
            emb = F.pad(emb, (0, 1))
        return self.proj(emb)


class TeacherResBlock3DPlain(nn.Module):
    def __init__(self, ch, pdrop=0.3):
        super().__init__()
        self.n1 = nn.GroupNorm(min(8, ch), ch)
        self.c1 = nn.Conv3d(ch, ch, 3, padding=1)
        self.n2 = nn.GroupNorm(min(8, ch), ch)
        self.d = nn.Dropout3d(pdrop)
        self.c2 = nn.Conv3d(ch, ch, 3, padding=1)

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(self.d(F.silu(self.n2(h))))
        return x + h


class TeacherResBlock3DWithTimeEmb(nn.Module):
    """Residual block with time embedding modulation."""
    def __init__(self, channels, emb_channels, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.emb_proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, channels * 2))
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.dropout = nn.Dropout3d(dropout)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        emb_out = self.emb_proj(emb)[:, :, None, None, None]
        scale, shift = emb_out.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return x + h


class TeacherUNet3D(nn.Module):
    """
    Teacher UNet for SVF diffeomorphic registration.
    Input: concat(x0, x1) [B, 2*in_channels, D, H, W]
    Output: velocity [B, out_channels, D, H, W]
    """
    def __init__(self, base_channels=48, latent_channels=4, use_time_emb=True):
        super().__init__()
        c = base_channels
        self.latent_channels = latent_channels
        self.use_time_emb = use_time_emb

        # Time embedding
        emb_channels = 256
        if use_time_emb:
            self.time_emb = TeacherTimeEmbedding(d_t=64, M=1000, out_channels=emb_channels)
            self.emb_channels = emb_channels
        else:
            self.emb_channels = None

        # Input: 2 * latent_channels (POD5 + POY1 concatenated)
        in_ch = 2 * latent_channels
        self.in_conv = nn.Conv3d(in_ch, c, 3, padding=1)

        # LATENT SPACE: 12³ -> 6³ -> 3³ -> 6³ -> 12³
        if use_time_emb:
            self.e1_1 = TeacherResBlock3DWithTimeEmb(c, emb_channels)
            self.e1_2 = TeacherResBlock3DWithTimeEmb(c, emb_channels)
            self.e2_1 = TeacherResBlock3DWithTimeEmb(c * 2, emb_channels)
            self.e2_2 = TeacherResBlock3DWithTimeEmb(c * 2, emb_channels)
            self.b1 = TeacherResBlock3DWithTimeEmb(c * 4, emb_channels)
            self.b2 = TeacherResBlock3DWithTimeEmb(c * 4, emb_channels)
            self.d2_1 = TeacherResBlock3DWithTimeEmb(c * 4, emb_channels)
            self.d2_2 = TeacherResBlock3DWithTimeEmb(c * 4, emb_channels)
            self.d1_1 = TeacherResBlock3DWithTimeEmb(c * 2, emb_channels)
            self.d1_2 = TeacherResBlock3DWithTimeEmb(c * 2, emb_channels)
        else:
            self.e1_1 = TeacherResBlock3DPlain(c)
            self.e1_2 = TeacherResBlock3DPlain(c)
            self.e2_1 = TeacherResBlock3DPlain(c * 2)
            self.e2_2 = TeacherResBlock3DPlain(c * 2)
            self.b1 = TeacherResBlock3DPlain(c * 4)
            self.b2 = TeacherResBlock3DPlain(c * 4)
            self.d2_1 = TeacherResBlock3DPlain(c * 4)
            self.d2_2 = TeacherResBlock3DPlain(c * 4)
            self.d1_1 = TeacherResBlock3DPlain(c * 2)
            self.d1_2 = TeacherResBlock3DPlain(c * 2)

        self.down1 = nn.Conv3d(c, c * 2, 3, stride=2, padding=1)  # 12->6
        self.down2 = nn.Conv3d(c * 2, c * 4, 3, stride=2, padding=1)  # 6->3
        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, 3, stride=2, padding=1, output_padding=1)  # 3->6
        self.p2 = nn.Conv3d(c * 4, c * 2, 1)
        self.up1 = nn.ConvTranspose3d(c * 2, c, 3, stride=2, padding=1, output_padding=1)  # 6->12
        self.p1 = nn.Conv3d(c * 2, c, 1)

        # Output: velocity channels (must match latent_channels passed at construction).
        out_ch = latent_channels
        self.out = nn.Sequential(nn.GroupNorm(min(8, c), c), nn.SiLU(), nn.Conv3d(c, out_ch, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x0, x1, t=None):
        """Forward pass. x0=POD5 latent, x1=POY1 latent (goal), t=time scalar."""
        x = torch.cat([x0, x1], dim=1)
        h0 = self.in_conv(x)

        t_emb = None
        if self.use_time_emb and t is not None:
            t_emb = self.time_emb(t)

        if self.use_time_emb and t_emb is not None:
            h1 = self.e1_1(h0, t_emb); h1 = self.e1_2(h1, t_emb); s1 = h1
            h2 = self.down1(h1)
            h2 = self.e2_1(h2, t_emb); h2 = self.e2_2(h2, t_emb); s2 = h2
            hb = self.down2(h2)
            hb = self.b1(hb, t_emb); hb = self.b2(hb, t_emb)
            u2 = self.up2(hb); u2 = torch.cat([u2, s2], dim=1)
            u2 = self.d2_1(u2, t_emb); u2 = self.d2_2(u2, t_emb); u2 = self.p2(u2)
            u1 = self.up1(u2); u1 = torch.cat([u1, s1], dim=1)
            u1 = self.d1_1(u1, t_emb); u1 = self.d1_2(u1, t_emb); u1 = self.p1(u1)
        else:
            h1 = self.e1_1(h0); h1 = self.e1_2(h1); s1 = h1
            h2 = self.down1(h1)
            h2 = self.e2_1(h2); h2 = self.e2_2(h2); s2 = h2
            hb = self.down2(h2)
            hb = self.b1(hb); hb = self.b2(hb)
            u2 = self.up2(hb); u2 = torch.cat([u2, s2], dim=1)
            u2 = self.d2_1(u2); u2 = self.d2_2(u2); u2 = self.p2(u2)
            u1 = self.up1(u2); u1 = torch.cat([u1, s1], dim=1)
            u1 = self.d1_1(u1); u1 = self.d1_2(u1); u1 = self.p1(u1)

        v_out = self.out(u1)
        return v_out


# ===================== SVF Image-Space Teacher Wrapper =====================

class SVFTeacherWrapper:
    """Wrapper for SVF image-space teacher model."""
    
    def __init__(self, svf_model: nn.Module, device: torch.device,
                 ss_squarings: int = 7, flow_cap: float = 5.0):
        """
        Args:
            svf_model: Trained SVF UNet (image space, outputs [B,3,48,48,48] velocity + optional intensity)
            device: torch device
            ss_squarings: Number of scaling-and-squaring iterations for exp(v)
            flow_cap: Velocity cap in voxels per axis
        """
        self.svf_model = svf_model
        self.device = device
        self.ss_squarings = ss_squarings
        self.flow_cap = flow_cap
        
        # Ensure SVF model is in eval mode and frozen
        self.svf_model.eval()
        for p in self.svf_model.parameters():
            p.requires_grad_(False)
    
    def _predict_svf_velocity(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Predict SVF velocity field from (x0, x1) pair."""
        with torch.no_grad():
            out = self.svf_model(x0, x1)
            if isinstance(out, tuple):
                v_vox = out[0]  # SVF returns (v, a_raw)
            else:
                v_vox = out
            # Ensure v_vox is [B, 3, D, H, W]
            if v_vox.shape[1] > 3:
                v_vox = v_vox[:, :3]
        return v_vox
    
    def _expv_at_t(self, v_vox: torch.Tensor, t_scalar: float) -> torch.Tensor:
        """
        Compute exp(t * v) using scaling-and-squaring.
        
        Args:
            v_vox: Velocity field [B, 3, D, H, W] in voxel units
            t_scalar: Time in [0, 1]
        Returns:
            phi_norm: Normalized displacement field [B, D, H, W, 3]
        """
        if abs(t_scalar) < 1e-8:
            # Identity at t=0
            B, _, D, H, W = v_vox.shape
            return torch.zeros(B, D, H, W, 3, device=v_vox.device)
        
        # Scale velocity by t
        v_scaled = v_vox * t_scalar
        
        # Cap velocity for stability
        v_capped = self.flow_cap * torch.tanh(v_scaled / max(self.flow_cap, 1e-6))
        
        # Convert to normalized displacement
        B, C, D, H, W = v_capped.shape
        # v_capped is (z, y, x) order in channels
        # grid_sample expects (x, y, z) order in last dim
        v_xyz = v_capped.permute(0, 2, 3, 4, 1)  # [B, D, H, W, 3]
        # Convert from voxel to normalized coords: divide by (size-1)/2 * 2 = size-1
        # Actually for normalized: displacement / ((size-1)/2) but grid is in [-1,1]
        v_norm = torch.zeros_like(v_xyz)
        v_norm[..., 0] = v_xyz[..., 2] / ((W - 1) / 2)  # x from voxel z
        v_norm[..., 1] = v_xyz[..., 1] / ((H - 1) / 2)  # y from voxel y
        v_norm[..., 2] = v_xyz[..., 0] / ((D - 1) / 2)  # z from voxel x
        
        # Scaling-and-squaring
        phi = v_norm / (2.0 ** self.ss_squarings)
        for _ in range(self.ss_squarings):
            phi = self._compose_fields(phi, phi)
        
        return phi
    
    def _compose_fields(self, phi_a: torch.Tensor, phi_b: torch.Tensor) -> torch.Tensor:
        """Compose two displacement fields: phi_a ∘ phi_b."""
        B, D, H, W, _ = phi_a.shape
        base = self._make_base_grid(D, H, W, phi_a.device).unsqueeze(0).expand(B, -1, -1, -1, -1)
        sample_coords = base + phi_b
        # Sample phi_a at displaced coordinates
        phi_a_sampled = F.grid_sample(
            phi_a.permute(0, 4, 1, 2, 3),  # [B, 3, D, H, W]
            sample_coords,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        ).permute(0, 2, 3, 4, 1)  # [B, D, H, W, 3]
        return phi_a_sampled + phi_b
    
    def _make_base_grid(self, D: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Create base grid in [-1, 1] normalized coordinates."""
        z = torch.linspace(-1, 1, D, device=device)
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
        return torch.stack([grid_x, grid_y, grid_z], dim=-1)  # [D, H, W, 3] (x, y, z order)
    
    def _warp_image(self, x: torch.Tensor, phi_norm: torch.Tensor) -> torch.Tensor:
        """Warp image x by normalized displacement field phi."""
        B, C, D, H, W = x.shape
        base = self._make_base_grid(D, H, W, x.device).unsqueeze(0).expand(B, -1, -1, -1, -1)
        grid = base + phi_norm
        x_warped = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return x_warped
    
    @torch.no_grad()
    def get_warped_at_t(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Get warped image at time t.
        
        Args:
            x0: POD5 image [B, 1, 48, 48, 48] normalized to [-1, 1]
            x1: POY1 image [B, 1, 48, 48, 48] normalized to [-1, 1]
            t: Time tensor [B] in [0, 1]
        Returns:
            x_t: Warped image [B, 1, 48, 48, 48]
        """
        B = x0.size(0)
        
        # Predict SVF velocity (same for all t since it's stationary)
        v_vox = self._predict_svf_velocity(x0, x1)
        
        # Handle batch of different t values
        x_t = torch.zeros_like(x0)
        for i in range(B):
            t_i = float(t[i].item())
            phi_i = self._expv_at_t(v_vox[i:i+1], t_i)
            x_t[i:i+1] = self._warp_image(x0[i:i+1], phi_i)
        
        return x_t
    
    @torch.no_grad()
    def get_teacher_state_and_tangent(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, dt: float = 0.05
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get teacher IMAGE state x*(t) and tangent dx*/dt.
        
        Args:
            x0: POD5 image [B, 1, 48, 48, 48]
            x1: POY1 image [B, 1, 48, 48, 48]
            t: Time tensor [B] in [0, 1]
            dt: Time step for finite difference
        Returns:
            x_star: Teacher image state at t [B, 1, 48, 48, 48]
            dx_star_dt: Teacher tangent (finite difference) [B, 1, 48, 48, 48]
        """
        dt_val = float(dt)
        if dt_val <= 0.0:
            raise ValueError(f"dt must be > 0 (got {dt_val})")

        scheme = str(globals().get("LYAPUNOV_TEACHER_TANGENT_SCHEME", "forward")).strip().lower()
        if scheme not in ("forward", "centered", "forward2"):
            raise ValueError("LYAPUNOV_TEACHER_TANGENT_SCHEME must be 'forward', 'centered', or 'forward2' "
                             f"(got {globals().get('LYAPUNOV_TEACHER_TANGENT_SCHEME')!r})")

        # Always compute x*(t) since the Lyapunov correction term uses it.
        x_t = self.get_warped_at_t(x0, x1, t.clamp(0.0, 1.0))
        x_star = x_t

        if scheme == "forward":
            # Forward difference: dx*/dt ≈ (x*(t+dt) - x*(t)) / dt
            t_next = (t + dt_val).clamp(0.0, 1.0)
            x_t_next = self.get_warped_at_t(x0, x1, t_next)
            dx_star_dt = (x_t_next - x_star) / dt_val
            return x_star, dx_star_dt

        if scheme == "forward2":
            # 2nd-order forward difference (more accurate than forward; avoids t-dt):
            # interior:   dx*/dt ≈ (-3 x*(t) + 4 x*(t+dt) - x*(t+2dt)) / (2 dt)
            # near t=1:   fall back to forward/backward 1st-order to avoid clamping artifacts.
            t_det = t.detach().to(dtype=torch.float32)
            use_forward2 = t_det <= (1.0 - 2.0 * dt_val)
            use_forward1 = (t_det > (1.0 - 2.0 * dt_val)) & (t_det <= (1.0 - dt_val))
            use_backward1 = t_det > (1.0 - dt_val)

            # Always compute x(t+dt) because it's used by both forward2 and forward1 regions.
            t_next = (t + dt_val).clamp(0.0, 1.0)
            x_t_next = self.get_warped_at_t(x0, x1, t_next)

            x_t_next2 = None
            if bool(use_forward2.any()):
                t_next2 = (t + 2.0 * dt_val).clamp(0.0, 1.0)
                x_t_next2 = self.get_warped_at_t(x0, x1, t_next2)

            x_t_prev = None
            if bool(use_backward1.any()):
                t_prev = (t - dt_val).clamp(0.0, 1.0)
                x_t_prev = self.get_warped_at_t(x0, x1, t_prev)

            dx_star_dt = torch.empty_like(x_star)
            if bool(use_forward2.any()):
                assert x_t_next2 is not None
                dx_star_dt[use_forward2] = (
                    (-3.0 * x_star[use_forward2]) + (4.0 * x_t_next[use_forward2]) - x_t_next2[use_forward2]
                ) / (2.0 * dt_val)
            if bool(use_forward1.any()):
                dx_star_dt[use_forward1] = (x_t_next[use_forward1] - x_star[use_forward1]) / dt_val
            if bool(use_backward1.any()):
                assert x_t_prev is not None
                dx_star_dt[use_backward1] = (x_star[use_backward1] - x_t_prev[use_backward1]) / dt_val

            return x_star, dx_star_dt

        # Centered difference (with boundary fallbacks):
        # - interior:   (x*(t+dt) - x*(t-dt)) / (2 dt)
        # - near t=0:   forward difference
        # - near t=1:   backward difference
        t_prev = (t - dt_val).clamp(0.0, 1.0)
        t_next = (t + dt_val).clamp(0.0, 1.0)
        x_t_prev = self.get_warped_at_t(x0, x1, t_prev)
        x_t_next = self.get_warped_at_t(x0, x1, t_next)

        t_det = t.detach().to(dtype=torch.float32)
        use_forward = t_det < dt_val
        use_backward = t_det > (1.0 - dt_val)
        use_center = ~(use_forward | use_backward)

        dx_star_dt = torch.empty_like(x_star)
        if bool(use_center.any()):
            dx_star_dt[use_center] = (x_t_next[use_center] - x_t_prev[use_center]) / (2.0 * dt_val)
        if bool(use_forward.any()):
            dx_star_dt[use_forward] = (x_t_next[use_forward] - x_star[use_forward]) / dt_val
        if bool(use_backward.any()):
            dx_star_dt[use_backward] = (x_star[use_backward] - x_t_prev[use_backward]) / dt_val

        return x_star, dx_star_dt


def load_svf_teacher(device: torch.device) -> SVFTeacherWrapper | None:
    """
    Load SVF teacher model. Teacher operates entirely in image space.
    
    This is used for:
    - Lyapunov regularization (when LYAPUNOV_ENABLED / USE_ANALYTICAL_LYAPUNOV are on)
    - Teacher-alignment diagnostics (always computed if the teacher is available)
    
    Returns None if the checkpoint is not found.
    """
    ckpt_path = LYAPUNOV_SVF_TEACHER_CHECKPOINT
    if ckpt_path is None or not Path(ckpt_path).exists():
        print(f"⚠️ SVF Teacher checkpoint not found at {ckpt_path}")
        print("   Teacher-based diagnostics/Lyapunov will be disabled for this run.")
        return None
    
    print(f"🧠 Loading SVF Teacher from {ckpt_path}")
    
    # Load checkpoint
    data = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if isinstance(data, dict):
        state_dict = data.get("model_state_dict") or data.get("model_state") or data.get("state_dict") or data
    else:
        state_dict = data
    
    # Normalize common prefixes (DataParallel etc.)
    if isinstance(state_dict, dict) and any(str(k).startswith("module.") for k in state_dict.keys()):
        state_dict = {str(k).replace("module.", "", 1): v for k, v in state_dict.items()}

    # Detect base channels from checkpoint
    in_w = state_dict.get("in_conv.weight", None)
    if isinstance(in_w, torch.Tensor) and in_w.ndim == 5:
        detected_base_ch = int(in_w.shape[0])
    else:
        detected_base_ch = LYAPUNOV_SVF_TEACHER_BASE_CHANNELS

    # Detect 2-level vs 3-level architecture from checkpoint keys
    detected_num_downs = 3 if ("down3.weight" in state_dict or any(k.startswith("down3.") for k in state_dict.keys())) else 2

    # Detect output channels from checkpoint (teacher is typically velocity-only, i.e., 3)
    out_w = state_dict.get("out.2.weight", None)
    if isinstance(out_w, torch.Tensor) and out_w.ndim == 5:
        detected_out_channels = int(out_w.shape[0])
    else:
        detected_out_channels = 3
    
    # Create SVF model (image space, 48³)
    svf_model = TeacherUNet3D(
        base_channels=detected_base_ch,
        use_time_emb=False,  # SVF mode doesn't use time embedding
        in_channels=2,  # concat(x0, x1)
        num_downs=detected_num_downs,
        out_channels=detected_out_channels,
    ).to(device)
    
    # Load weights (strict after choosing the right architecture)
    try:
        svf_model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            "SVF Teacher checkpoint does not match the instantiated teacher architecture. "
            "This usually means you changed the teacher UNet depth/base-channels but are still pointing to an old checkpoint.\n"
            f"Checkpoint: {ckpt_path}\n"
            f"Detected base_channels={detected_base_ch}, num_downs={detected_num_downs}.\n"
            "Fix: update LYAPUNOV_SVF_TEACHER_CHECKPOINT to the matching .pth or regenerate the teacher.\n"
            f"Original load error: {e}"
        )
    svf_model.eval()
    for p in svf_model.parameters():
        p.requires_grad_(False)
    
    print(f"   SVF Teacher parameters: {sum(p.numel() for p in svf_model.parameters()):,} (frozen)")
    print(f"   Base channels: {detected_base_ch}")
    print(f"   UNet depth: {detected_num_downs} downsamples")
    print(f"   Scaling-squaring iterations: {LYAPUNOV_SVF_TEACHER_SS_SQUARINGS}")
    print(f"   Teacher operates in IMAGE space")
    
    wrapper = SVFTeacherWrapper(
        svf_model=svf_model,
        device=device,
        ss_squarings=LYAPUNOV_SVF_TEACHER_SS_SQUARINGS,
        flow_cap=LYAPUNOV_SVF_TEACHER_FLOW_CAP
    )
    
    return wrapper


@torch.no_grad()
def maybe_initialize_student_from_svf_teacher(
    flow: nn.Module,
    svf_teacher: SVFTeacherWrapper | None,
    *,
    verbose: bool = True,
) -> dict:
    """Initialize selected student layers from the frozen SVF teacher (optional)."""
    # Make init dependent on the master loss mode.
    # If LOSS_MODE is pure FM (no teacher-based objective), we skip init to keep the baseline clean.
    loss_mode = str(globals().get('LOSS_MODE', 'both')).strip().lower()
    if loss_mode == 'fm_only':
        if verbose and bool(globals().get("INIT_STUDENT_FROM_SVF_TEACHER", False)):
            print("   [INIT] Skipping student init from SVF teacher because LOSS_MODE='fm_only'.")
        return {"enabled": False, "copied": [], "skipped": ["loss_mode_fm_only"]}

    if not bool(globals().get("INIT_STUDENT_FROM_SVF_TEACHER", False)):
        return {"enabled": False, "copied": [], "skipped": []}

    init_mode = str(globals().get("INIT_STUDENT_FROM_SVF_TEACHER_MODE", "middle_only")).strip().lower()
    if init_mode not in ("all", "middle_only", "middle3", "input_only"):
        raise ValueError(
            "INIT_STUDENT_FROM_SVF_TEACHER_MODE must be 'all', 'middle_only', 'middle3', or 'input_only' "
            f"(got {init_mode!r})"
        )

    if svf_teacher is None or getattr(svf_teacher, "svf_model", None) is None:
        if verbose:
            print("   [INIT] INIT_STUDENT_FROM_SVF_TEACHER=True but SVF teacher is not loaded; skipping init.")
        return {"enabled": True, "copied": [], "skipped": ["teacher_not_loaded"]}

    teacher = svf_teacher.svf_model
    copied: list[str] = []
    skipped: list[str] = []

    def _wshape(m: nn.Module | None) -> str:
        try:
            if m is not None and hasattr(m, "weight") and isinstance(m.weight, torch.Tensor):
                return str(tuple(m.weight.shape))
        except Exception:
            pass
        return "?"

    def _log(action: str, label: str, dst: nn.Module | None, src: nn.Module | None, reason: str | None = None) -> None:
        if action == "copied":
            copied.append(label)
        else:
            skipped.append(label if reason is None else f"{label} [{reason}]")
        if verbose:
            extra = f" reason={reason}" if reason else ""
            print(f"   [INIT] {action.upper():6s} {label} (student={_wshape(dst)}, teacher={_wshape(src)}){extra}")

    def _copy_conv_exact(dst: nn.Module, src: nn.Module, label: str) -> None:
        if not (isinstance(dst, (nn.Conv3d, nn.ConvTranspose3d)) and isinstance(src, (nn.Conv3d, nn.ConvTranspose3d))):
            _log("skipped", label, dst, src, "not_conv")
            return
        if dst.weight.shape != src.weight.shape:
            _log("skipped", label, dst, src, "shape_mismatch")
            return
        dst.weight.copy_(src.weight)
        if (dst.bias is not None) and (src.bias is not None) and (dst.bias.shape == src.bias.shape):
            dst.bias.copy_(src.bias)
        _log("copied", label, dst, src)

    def _copy_groupnorm_exact(dst: nn.Module, src: nn.Module, label: str) -> None:
        if not (isinstance(dst, nn.GroupNorm) and isinstance(src, nn.GroupNorm)):
            _log("skipped", label, dst, src, "not_groupnorm")
            return
        if (dst.weight is None) or (src.weight is None) or (dst.bias is None) or (src.bias is None):
            _log("skipped", label, dst, src, "missing_params")
            return
        if dst.weight.shape != src.weight.shape or dst.bias.shape != src.bias.shape:
            _log("skipped", label, dst, src, "shape_mismatch")
            return
        dst.weight.copy_(src.weight)
        dst.bias.copy_(src.bias)
        _log("copied", label, dst, src)

    if verbose:
        print(f"   [INIT] Starting student init from SVF teacher (mode={init_mode})")

    # (A) Input conv (skipped in middle_only)
    label_in = "input_conv <- teacher.in_conv (adapt)"
    if init_mode in ("all", "input_only"):
        dst = getattr(flow, "input_conv", None)
        src = getattr(teacher, "in_conv", None)
        if isinstance(dst, nn.Conv3d) and isinstance(src, nn.Conv3d):
            w_dst = dst.weight
            w_src = src.weight
            if (w_dst.ndim == 5) and (w_src.ndim == 5) and (w_dst.shape[0] == w_src.shape[0]) and (w_dst.shape[2:] == w_src.shape[2:]):
                in_ch_dst = int(w_dst.shape[1])
                in_ch_src = int(w_src.shape[1])
                new_w = torch.empty_like(w_dst)
                if in_ch_dst == 1 and in_ch_src >= 2:
                    new_w[:, 0] = w_src[:, :2].mean(dim=1)
                elif in_ch_dst <= in_ch_src:
                    if in_ch_dst == 1:
                        new_w[:, 0] = w_src.mean(dim=1)
                    else:
                        new_w[:, :in_ch_dst] = w_src[:, :in_ch_dst]
                else:
                    new_w[:, :in_ch_src] = w_src[:, :in_ch_src]
                    fill = w_src.mean(dim=1, keepdim=True)
                    new_w[:, in_ch_src:] = fill.expand(-1, in_ch_dst - in_ch_src, -1, -1, -1)
                dst.weight.copy_(new_w)
                if (dst.bias is not None) and (src.bias is not None) and (dst.bias.shape == src.bias.shape):
                    dst.bias.copy_(src.bias)
                _log("copied", label_in, dst, src)
            else:
                _log("skipped", label_in, dst, src, "shape_mismatch")
        else:
            _log("skipped", label_in, dst, src, "missing_or_not_conv")
    else:
        _log("skipped", label_in, getattr(flow, "input_conv", None), getattr(teacher, "in_conv", None), "mode_no_input")

    # (B) Middle structural layers.
    # - middle_only: copy 6 layers (enc1_down, enc2_down, dec2_up, dec1_up, dec2_proj, dec1_proj)
    # - middle3: copy 3 deeper layers only (enc2_down, dec2_up, dec2_proj)
    if init_mode in ("all", "middle_only", "middle3"):
        if init_mode == "middle3":
            pairs = [
                ("enc2_down", "down2"),
                ("dec2_up", "up2"),
                ("dec2_proj", "p2"),
            ]
        else:
            pairs = [
                ("enc1_down", "down1"),
                ("enc2_down", "down2"),
                ("dec2_up", "up2"),
                ("dec1_up", "up1"),
                ("dec2_proj", "p2"),
                ("dec1_proj", "p1"),
            ]
        for dst_name, src_name in pairs:
            dst = getattr(flow, dst_name, None)
            src = getattr(teacher, src_name, None)
            label = f"{dst_name} <- teacher.{src_name}"
            if dst is None or src is None:
                _log("skipped", label, dst, src, "missing_attr")
            else:
                _copy_conv_exact(dst, src, label)

    # (C) Output normalization only (NOT the final output conv) - only in all
    label_gn = "output_conv[0] GroupNorm <- teacher.out[0] GroupNorm"
    if init_mode == "all":
        out_student = getattr(flow, "output_conv", None)
        out_teacher = getattr(teacher, "out", None)
        if isinstance(out_student, nn.Sequential) and isinstance(out_teacher, nn.Sequential) and len(out_student) >= 1 and len(out_teacher) >= 1:
            _copy_groupnorm_exact(out_student[0], out_teacher[0], label_gn)
        else:
            _log("skipped", label_gn, out_student, out_teacher, "missing_or_not_sequential")
    else:
        _log("skipped", label_gn, getattr(flow, "output_conv", None), getattr(teacher, "out", None), "mode_not_all")

    if verbose:
        print(f"   [INIT] Summary mode={init_mode}: copied={len(copied)} skipped={len(skipped)}")

    return {"enabled": True, "copied": copied, "skipped": skipped}


# ===================== ANALYTICAL Lyapunov LOSS (IMAGE SPACE) =====================
# This implements the Lyapunov loss using an explicit value function V(x,t) = (α/2)||x - x*(t)||²
# No value network is learned - the gradient ∇V = α(x - x*(t)) is computed analytically.
#
# The Lyapunov-optimal velocity is: v_lyapunov = dx*/dt - α(x - x*(t))
# This includes:
#   - dx*/dt: Teacher tangent (follow the moving path)
#   - -α(x - x*(t)): Correction term (pull back to path if drifting)
# =============================================================================

def compute_LYAPUNOV_analytical_loss(
    v_pred: torch.Tensor,           # Predicted velocity from student [B, C, D, H, W]
    z_t: torch.Tensor,              # Current student state [B, C, D, H, W]
    z_star: torch.Tensor,           # Teacher state at time t [B, C, D, H, W]
    dz_star_dt: torch.Tensor,       # Teacher tangent [B, C, D, H, W]
    t: torch.Tensor,                # Time [B]
    warmup_frac: float,             # Warmup fraction in [0,1] (step-based; 0=start, 1=full)
    weight_map: torch.Tensor | None = None,  # Optional voxel mask/weights [B,1,D,H,W] or [B,C,D,H,W]
) -> tuple[torch.Tensor, dict]:
    """
    Compute analytical Lyapunov loss without learned critic.
    
    Loss = ||v_pred - v_lyapunov||²
    where v_lyapunov = dz*/dt - α(z - z*(t))
    
    Args:
        v_pred: Student predicted velocity
        z_t: Current interpolated state (student explores around this)
        z_star: Teacher reference state at time t
        dz_star_dt: Teacher tangent velocity
        t: Time values
        warmup_frac: Warmup fraction for Lyapunov weight (0=start, 1=full)
    
    Returns:
        loss: Scalar Lyapunov loss (weighted)
        info: Dictionary with diagnostics
    """
    B = v_pred.size(0)
    device = v_pred.device
    
    # Warmup weight: ramp from 0 to 1 over LYAPUNOV_WARMUP_EPOCHS (STEP-based; caller provides warmup_frac).
    # If LYAPUNOV_WARMUP_EPOCHS <= 0, skip warmup (full weight immediately).
    if LYAPUNOV_WARMUP_EPOCHS <= 0:
        warmup_frac = 1.0
    warmup_frac = float(max(0.0, min(1.0, float(warmup_frac))))
    w_warmup = 0.5 * (1.0 - math.cos(math.pi * warmup_frac))  # Cosine ramp
    
    # Terminal fade: reduce λ near t=1 so RF controls endpoint
    t_np = t.detach().cpu().numpy()
    fade = np.ones(B, dtype=np.float32)
    mask = t_np > LYAPUNOV_TERMINAL_FADE_START
    if mask.any():
        fade[mask] = 1.0 - (t_np[mask] - LYAPUNOV_TERMINAL_FADE_START) / (1.0 - LYAPUNOV_TERMINAL_FADE_START)
        fade = np.clip(fade, 0.0, 1.0)
    fade = torch.tensor(fade, device=device, dtype=torch.float32)
    
    # Final weight per sample
    w_lyapunov = LYAPUNOV_LAMBDA_MAX * w_warmup * fade  # [B]
    
    if w_lyapunov.max() < 1e-8:
        return torch.tensor(0.0, device=device), {
            # Preferred keys (used by training logs)
            'raw_lyapunov_loss': 0.0,
            'lambda': 0.0,
            'weight': 0.0,
            # Backward-compatible keys
            'w_LYAPUNOV_mean': 0.0,
            'loss_LYAPUNOV_raw': 0.0,
            'loss_LYAPUNOV_weighted': 0.0,
            'warmup_frac': float(warmup_frac),
        }
    
    # Compute deviation from teacher path
    deviation = z_t - z_star  # [B, C, D, H, W]
    
    # Lyapunov-optimal velocity: v_lyapunov = dz*/dt - α * (z - z*)
    # This is the velocity that minimizes cost-to-go for the chosen V and ℓ
    v_lyapunov = dz_star_dt - LYAPUNOV_ALPHA * deviation
    
    # Loss: MSE between predicted and Lyapunov-optimal velocity.
    # If weight_map is provided, compute a masked mean over voxels (e.g., bone-only distillation).
    diff2 = (v_pred - v_lyapunov) ** 2  # [B, C, D, H, W]
    loss_per_sample_full = diff2.flatten(1).mean(dim=1)  # [B]

    if weight_map is not None:
        w = weight_map
        if w.ndim == 4:
            w = w.unsqueeze(1)
        if w.ndim != 5:
            raise ValueError(f"weight_map must have shape [B,1,D,H,W] or [B,C,D,H,W] (got {tuple(w.shape)})")
        w = w.to(device=device, dtype=diff2.dtype)
        if w.shape[1] == 1 and diff2.shape[1] != 1:
            w = w.expand(-1, diff2.shape[1], -1, -1, -1)
        # Weighted mean; if a sample has no masked voxels, fall back to full-volume loss.
        num = (diff2 * w).flatten(1).sum(dim=1)
        den = w.flatten(1).sum(dim=1).clamp_min(1e-8)
        loss_per_sample_masked = num / den
        empty = (den <= 1e-7)
        loss_per_sample = torch.where(empty, loss_per_sample_full, loss_per_sample_masked)
    else:
        loss_per_sample = loss_per_sample_full
    
    # Weighted loss
    weighted_loss = (w_lyapunov * loss_per_sample).mean()
    
    # Diagnostics
    with torch.no_grad():
        # Cosine similarity between v_pred and v_lyapunov (masked if weight_map provided)
        if weight_map is not None:
            w = weight_map
            if w.ndim == 4:
                w = w.unsqueeze(1)
            w = w.to(device=device, dtype=v_pred.dtype)
            if w.shape[1] == 1 and v_pred.shape[1] != 1:
                w = w.expand(-1, v_pred.shape[1], -1, -1, -1)
            v_pred_flat = (v_pred * w).view(B, -1)
            v_LYAPUNOV_flat = (v_lyapunov * w).view(B, -1)
        else:
            v_pred_flat = v_pred.view(B, -1)
            v_LYAPUNOV_flat = v_lyapunov.view(B, -1)
        dot = (v_pred_flat * v_LYAPUNOV_flat).sum(dim=1)
        norm_pred = v_pred_flat.norm(dim=1).clamp(min=1e-8)
        norm_lyapunov = v_LYAPUNOV_flat.norm(dim=1).clamp(min=1e-8)
        cosine_sim = (dot / (norm_pred * norm_lyapunov)).mean()
        
        # Deviation magnitude
        dev_mag = deviation.pow(2).flatten(1).mean(dim=1).sqrt().mean()
        
        # Teacher tangent magnitude
        tangent_mag = dz_star_dt.pow(2).flatten(1).mean(dim=1).sqrt().mean()
    
    info = {
        # Preferred keys (used by training logs)
        'raw_lyapunov_loss': float(loss_per_sample.mean().item()),
        'lambda': float(w_lyapunov.mean().item()),
        'weight': float(w_lyapunov.mean().item()),
        # Backward-compatible keys
        'w_LYAPUNOV_mean': float(w_lyapunov.mean().item()),
        'loss_LYAPUNOV_raw': float(loss_per_sample.mean().item()),
        'loss_LYAPUNOV_weighted': float(weighted_loss.item()),
        'deviation_mag': float(dev_mag.item()),
        'tangent_mag': float(tangent_mag.item()),
        'cosine_sim_v_lyapunov': float(cosine_sim.item()),
        'warmup_frac': float(warmup_frac),
    }
    
    return weighted_loss, info


@torch.no_grad()
def rollout_student_to_time_euler(
    flow: nn.Module,
    x0: torch.Tensor,
    t_target: torch.Tensor,
    case_ids: torch.Tensor,
    bone_mask_img: torch.Tensor | None = None,
    steps: int = 8,
) -> torch.Tensor:
    """Roll out student dynamics x' = v_theta(x,t) from t=0 to t=t_target using Euler.

    This is used only to *choose the state* for the Lyapunov/LQR loss (on-policy training).
    Gradients are NOT propagated through this rollout.
    """
    if int(steps) <= 0:
        return x0

    was_training = bool(flow.training)
    flow.eval()
    x = x0.clone()
    B = x.shape[0]
    t_target = t_target.to(device=x.device, dtype=torch.float32).clamp(0.0, 1.0)
    dt = 1.0 / float(int(steps))
    try:
        for s in range(int(steps)):
            t_s = torch.full((B,), float(s) * dt, device=x.device, dtype=torch.float32)
            step_size = (t_target - t_s).clamp(min=0.0, max=dt)
            if not (step_size > 0).any():
                break
            v = flow(
                x,
                t_s,
                case_ids,
                bone_mask=bone_mask_img,
            )
            x = x + step_size.view(B, 1, 1, 1, 1) * v
        return x
    finally:
        if was_training:
            flow.train()



def _bone_weight_map(gt_norm_t: torch.Tensor, hu_threshold: float, alpha: float, surface_weight: float) -> torch.Tensor:
    """Bone-surface weight map for weighted image-space MSE loss."""
    lo, hi = HU_RANGE
    gt_hu = (gt_norm_t.clamp(-1, 1) + 1.0) * 0.5 * (hi - lo) + lo  # [B,1,D,H,W]
    bone_mask = (gt_hu > hu_threshold).float()
    # Surface band: 1-voxel wide using binary erosion
    bm = bone_mask.squeeze(1).detach().cpu().numpy().astype(np.uint8)  # [B,D,H,W]
    surface = []
    from scipy.ndimage import binary_erosion
    for b in range(bm.shape[0]):
        er = binary_erosion(bm[b], iterations=1, border_value=0)
        s = (bm[b] & (~er)).astype(np.float32)
        surface.append(s)
    surface = torch.from_numpy(np.stack(surface, axis=0)).to(bone_mask.device).unsqueeze(1)
    # Base weights: 1 + alpha in bone; add surface bonus
    w = torch.ones_like(bone_mask)
    w = w + alpha * bone_mask
    w = w + surface_weight * surface
    return w

def save_orthogonal_png(vol_np, save_path, title):
    """vol_np: [D,H,W] in HU or normalized; just visualize 3 middle slices."""
    D, H, W = vol_np.shape
    md, mh, mw = D//2, H//2, W//2
    # Use fixed HU range from HU_RANGE for consistent visualization
    vmin, vmax = HU_RANGE[0], HU_RANGE[1]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    fig.suptitle(title, fontsize=18, fontweight='bold')
    axes[0].imshow(vol_np[:, mh, :].T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    axes[0].set_title('Sagittal', fontsize=14, fontweight='bold')
    axes[1].imshow(vol_np[md, :, :].T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    axes[1].set_title('Coronal', fontsize=14, fontweight='bold')
    im = axes[2].imshow(vol_np[:, :, mw], cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    axes[2].set_title('Axial', fontsize=14, fontweight='bold')
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label('HU', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    for ax in axes: ax.set_xticks([]); ax.set_yticks([])
    plt.savefig(save_path, dpi=150)
    plt.close()


def _mid_slices_orthogonal(vol_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (sagittal, coronal, axial) 2D slices from a [D,H,W] volume."""
    if vol_np.ndim != 3:
        raise ValueError(f"Expected vol_np as [D,H,W]; got shape={getattr(vol_np, 'shape', None)}")
    D, H, W = vol_np.shape
    md, mh, mw = D // 2, H // 2, W // 2
    sag = vol_np[:, mh, :].T
    cor = vol_np[md, :, :].T
    ax = vol_np[:, :, mw]
    return sag, cor, ax


    base_rows = len(volumes)
    num_rows = base_rows
    if attn_map is not None:
        num_rows += 1
    if saliency_map is not None:
        num_rows += 1
    if error_map is not None:
        num_rows += 1
    if fm_weight_np is not None:
        num_rows += 1
    if slab_prior_np is not None:
        num_rows += 1
    if slab_prior_bone_np is not None:
        num_rows += 1
    if add_bone_mask:
        num_rows += 1
    if plate_mask is not None:
        num_rows += 1  # Add row for plate mask visualization
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 4 * num_rows), constrained_layout=True)
    fig.suptitle(f"Case {case_id} ROI {roi_num} - Training Progress (Epoch {epoch})", 
                 fontsize=20, fontweight='bold')
    
    for row_idx, (vol_np, row_title) in enumerate(volumes):
        D, H, W = vol_np.shape
        md, mh, mw = D//2, H//2, W//2
        # Use fixed HU range from HU_RANGE for consistent visualization
        vmin, vmax = HU_RANGE[0], HU_RANGE[1]
        norm = Normalize(vmin=vmin, vmax=vmax)
        
        # Sagittal
        axes[row_idx, 0].imshow(vol_np[:, mh, :].T, cmap='gray', norm=norm, origin='lower')
        axes[row_idx, 0].set_title('Sagittal' if row_idx == 0 else '', fontsize=14, fontweight='bold')
        axes[row_idx, 0].set_ylabel(row_title, fontsize=13, fontweight='bold')
        axes[row_idx, 0].set_xticks([]); axes[row_idx, 0].set_yticks([])
        
        # Coronal
        axes[row_idx, 1].imshow(vol_np[md, :, :].T, cmap='gray', norm=norm, origin='lower')
        axes[row_idx, 1].set_title('Coronal' if row_idx == 0 else '', fontsize=14, fontweight='bold')
        axes[row_idx, 1].set_xticks([]); axes[row_idx, 1].set_yticks([])
        
        # Axial
        im = axes[row_idx, 2].imshow(vol_np[:, :, mw], cmap='gray', norm=norm, origin='lower')
        axes[row_idx, 2].set_title('Axial' if row_idx == 0 else '', fontsize=14, fontweight='bold')
        axes[row_idx, 2].set_xticks([]); axes[row_idx, 2].set_yticks([])
        
        # Add colorbar for each row
        cbar = plt.colorbar(im, ax=axes[row_idx, :], shrink=0.8, pad=0.02)
        cbar.set_label('HU', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)
    
    # Optional attention heatmap row (projected to volume slices)
    next_row = base_rows
    if attn_map is not None:
        # attn_map expected shape [D,H,W] (e.g., 12^3). Upsample to 48^3 for display
        attn_t = torch.as_tensor(attn_map).float().unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
        attn_up = F.interpolate(attn_t, size=pod5_hu.shape, mode='trilinear', align_corners=False)
        attn_np = attn_up.squeeze().cpu().numpy()
        # normalize 0-1
        a_min, a_max = float(attn_np.min()), float(attn_np.max())
        if a_max > a_min:
            attn_np = (attn_np - a_min) / (a_max - a_min)
        D, H, W = attn_np.shape
        md, mh, mw = D//2, H//2, W//2
        # overlay on FM prediction context
        axes[next_row, 0].imshow(poy1_fm_pred_hu[:, mh, :].T, cmap='gray', origin='lower')
        im0 = axes[next_row, 0].imshow(attn_np[:, mh, :].T, cmap='magma', alpha=0.6, origin='lower')
        axes[next_row, 0].set_title('Attention (Sagittal)')
        axes[next_row, 0].set_ylabel('Attention Heatmap', fontsize=11, fontweight='bold')
        axes[next_row, 0].set_xticks([]); axes[next_row, 0].set_yticks([])
        axes[next_row, 1].imshow(poy1_fm_pred_hu[md, :, :].T, cmap='gray', origin='lower')
        im1 = axes[next_row, 1].imshow(attn_np[md, :, :].T, cmap='magma', alpha=0.6, origin='lower')
        axes[next_row, 1].set_title('Attention (Coronal)')
        axes[next_row, 1].set_xticks([]); axes[next_row, 1].set_yticks([])
        axes[next_row, 2].imshow(poy1_fm_pred_hu[:, :, mw], cmap='gray', origin='lower')
        im2 = axes[next_row, 2].imshow(attn_np[:, :, mw], cmap='magma', alpha=0.6, origin='lower')
        axes[next_row, 2].set_title('Attention (Axial)')
        axes[next_row, 2].set_xticks([]); axes[next_row, 2].set_yticks([])
        # shared colorbar for attention row
        cbar = plt.colorbar(im2, ax=axes[next_row, :], shrink=0.8, pad=0.02)
        cbar.set_label('Attention', rotation=270, labelpad=15, fontsize=9)
        next_row += 1

    # Optional saliency map (gradient wrt POD5)
    if saliency_map is not None:
        s_np = np.asarray(saliency_map)
        D, H, W = s_np.shape
        md, mh, mw = D//2, H//2, W//2
        axes[next_row, 0].imshow(s_np[:, mh, :].T, cmap='inferno', origin='lower')
        axes[next_row, 0].set_title('Saliency (Sagittal)')
        axes[next_row, 0].set_ylabel('Input Saliency', fontsize=11, fontweight='bold')
        axes[next_row, 0].set_xticks([]); axes[next_row, 0].set_yticks([])
        axes[next_row, 1].imshow(s_np[md, :, :].T, cmap='inferno', origin='lower')
        axes[next_row, 1].set_title('Saliency (Coronal)')
        axes[next_row, 1].set_xticks([]); axes[next_row, 1].set_yticks([])
        imS = axes[next_row, 2].imshow(s_np[:, :, mw], cmap='inferno', origin='lower')
        axes[next_row, 2].set_title('Saliency (Axial)')
        axes[next_row, 2].set_xticks([]); axes[next_row, 2].set_yticks([])
        cbar = plt.colorbar(imS, ax=axes[next_row, :], shrink=0.8, pad=0.02)
        cbar.set_label('Saliency', rotation=270, labelpad=15, fontsize=9)
        next_row += 1

    # Optional error map |Pred - GT|
    if error_map is not None:
        e_np = np.asarray(error_map)
        D, H, W = e_np.shape
        md, mh, mw = D//2, H//2, W//2
        axes[next_row, 0].imshow(e_np[:, mh, :].T, cmap='viridis', origin='lower')
        axes[next_row, 0].set_title('Error (Sagittal)')
        axes[next_row, 0].set_ylabel('|Pred-GT| HU', fontsize=11, fontweight='bold')
        axes[next_row, 0].set_xticks([]); axes[next_row, 0].set_yticks([])
        axes[next_row, 1].imshow(e_np[md, :, :].T, cmap='viridis', origin='lower')
        axes[next_row, 1].set_title('Error (Coronal)')
        axes[next_row, 1].set_xticks([]); axes[next_row, 1].set_yticks([])
        imE = axes[next_row, 2].imshow(e_np[:, :, mw], cmap='viridis', origin='lower')
        axes[next_row, 2].set_title('Error (Axial)')
        axes[next_row, 2].set_xticks([]); axes[next_row, 2].set_yticks([])
        cbar = plt.colorbar(imE, ax=axes[next_row, :], shrink=0.8, pad=0.02)
        cbar.set_label('|Pred-GT| HU', rotation=270, labelpad=15, fontsize=9)
        next_row += 1

    # Optional FM resection-plane weight map row
    if fm_weight_np is not None:
        w_np = np.asarray(fm_weight_np, dtype=np.float32)
        # clamp just in case of numerical noise
        w_np = np.clip(w_np, 0.0, 1.0)
        D, H, W = w_np.shape
        md, mh, mw = D//2, H//2, W//2
        im0 = axes[next_row, 0].imshow(w_np[:, mh, :].T, cmap='magma', vmin=0.0, vmax=1.0, origin='lower')
        axes[next_row, 0].set_title('Weight (Sagittal)')
        axes[next_row, 0].set_ylabel(fm_weight_title or 'FM Weight', fontsize=11, fontweight='bold')
        axes[next_row, 0].set_xticks([]); axes[next_row, 0].set_yticks([])
        axes[next_row, 1].imshow(w_np[md, :, :].T, cmap='magma', vmin=0.0, vmax=1.0, origin='lower')
        axes[next_row, 1].set_title('Weight (Coronal)')
        axes[next_row, 1].set_xticks([]); axes[next_row, 1].set_yticks([])
        im2 = axes[next_row, 2].imshow(w_np[:, :, mw], cmap='magma', vmin=0.0, vmax=1.0, origin='lower')
        axes[next_row, 2].set_title('Weight (Axial)')
        axes[next_row, 2].set_xticks([]); axes[next_row, 2].set_yticks([])
        cbar = plt.colorbar(im2, ax=axes[next_row, :], shrink=0.8, pad=0.02)
        cbar.set_label('Weight', rotation=270, labelpad=15, fontsize=9)
        next_row += 1

    # Optional middle-slab prior row (LATENT space, true resolution)
    if slab_prior_np is not None:
        s_np = np.asarray(slab_prior_np, dtype=np.float32)
        s_np = np.clip(s_np, 0.0, 1.0)
        D, H, W = s_np.shape
        md, mh, mw = D//2, H//2, W//2
        axes[next_row, 0].imshow(s_np[:, mh, :].T, cmap='magma', vmin=0.0, vmax=1.0, origin='lower', interpolation='nearest')
        axes[next_row, 0].set_title('Prior (Sagittal)')
        axes[next_row, 0].set_ylabel(slab_prior_title or 'Middle-Slab Prior (Latent)', fontsize=11, fontweight='bold')
        axes[next_row, 0].set_xticks([]); axes[next_row, 0].set_yticks([])
        axes[next_row, 1].imshow(s_np[md, :, :].T, cmap='magma', vmin=0.0, vmax=1.0, origin='lower', interpolation='nearest')
        axes[next_row, 1].set_title('Prior (Coronal)')
        axes[next_row, 1].set_xticks([]); axes[next_row, 1].set_yticks([])
        imP = axes[next_row, 2].imshow(s_np[:, :, mw], cmap='magma', vmin=0.0, vmax=1.0, origin='lower', interpolation='nearest')
        axes[next_row, 2].set_title('Prior (Axial)')
        axes[next_row, 2].set_xticks([]); axes[next_row, 2].set_yticks([])
        cbar = plt.colorbar(imP, ax=axes[next_row, :], shrink=0.8, pad=0.02)
        cbar.set_label('Prior', rotation=270, labelpad=15, fontsize=9)
        next_row += 1

    # Optional prior × bone row (IMAGE space)
    if slab_prior_bone_np is not None:
        s_np = np.asarray(slab_prior_bone_np, dtype=np.float32)
        s_np = np.clip(s_np, 0.0, 1.0)
        D, H, W = s_np.shape
        md, mh, mw = D//2, H//2, W//2
        axes[next_row, 0].imshow(s_np[:, mh, :].T, cmap='magma', vmin=0.0, vmax=1.0, origin='lower')
        axes[next_row, 0].set_title('Prior×Bone (Sagittal)')
        axes[next_row, 0].set_ylabel(slab_prior_bone_title or 'Prior×Bone', fontsize=11, fontweight='bold')
        axes[next_row, 0].set_xticks([]); axes[next_row, 0].set_yticks([])
        axes[next_row, 1].imshow(s_np[md, :, :].T, cmap='magma', vmin=0.0, vmax=1.0, origin='lower')
        axes[next_row, 1].set_title('Prior×Bone (Coronal)')
        axes[next_row, 1].set_xticks([]); axes[next_row, 1].set_yticks([])
        imPB = axes[next_row, 2].imshow(s_np[:, :, mw], cmap='magma', vmin=0.0, vmax=1.0, origin='lower')
        axes[next_row, 2].set_title('Prior×Bone (Axial)')
        axes[next_row, 2].set_xticks([]); axes[next_row, 2].set_yticks([])
        cbar = plt.colorbar(imPB, ax=axes[next_row, :], shrink=0.8, pad=0.02)
        cbar.set_label('Prior×Bone', rotation=270, labelpad=15, fontsize=9)
        next_row += 1

    # Optional plate mask overlay row (shows plate on POD5 and prediction)
    if plate_mask is not None:
        plate_np = np.asarray(plate_mask, dtype=np.float32)
        plate_np = (plate_np > 0.5).astype(np.float32)  # Ensure binary
        D, H, W = plate_np.shape
        md, mh, mw = D//2, H//2, W//2
        vmin, vmax = HU_RANGE[0], HU_RANGE[1]
        norm = Normalize(vmin=vmin, vmax=vmax)
        
        # Sagittal: Show POD5 with plate mask overlay in red
        axes[next_row, 0].imshow(pod5_hu[:, mh, :].T, cmap='gray', norm=norm, origin='lower')
        axes[next_row, 0].imshow(plate_np[:, mh, :].T, cmap='Reds', alpha=0.4, origin='lower', vmin=0, vmax=1)
        axes[next_row, 0].set_title('Plate on POD5 (Sagittal)')
        axes[next_row, 0].set_ylabel('Plate Mask Overlay', fontsize=11, fontweight='bold')
        axes[next_row, 0].set_xticks([]); axes[next_row, 0].set_yticks([])
        
        # Coronal: Show POD5 with plate mask overlay
        axes[next_row, 1].imshow(pod5_hu[md, :, :].T, cmap='gray', norm=norm, origin='lower')
        axes[next_row, 1].imshow(plate_np[md, :, :].T, cmap='Reds', alpha=0.4, origin='lower', vmin=0, vmax=1)
        axes[next_row, 1].set_title('Plate on POD5 (Coronal)')
        axes[next_row, 1].set_xticks([]); axes[next_row, 1].set_yticks([])
        
        # Axial: Show FM prediction with plate mask overlay
        axes[next_row, 2].imshow(poy1_fm_pred_hu[:, :, mw], cmap='gray', norm=norm, origin='lower')
        imPlate = axes[next_row, 2].imshow(plate_np[:, :, mw], cmap='Reds', alpha=0.4, origin='lower', vmin=0, vmax=1)
        axes[next_row, 2].set_title('Plate on FM Pred (Axial)')
        axes[next_row, 2].set_xticks([]); axes[next_row, 2].set_yticks([])
        
        cbar = plt.colorbar(imPlate, ax=axes[next_row, :], shrink=0.8, pad=0.02)
        cbar.set_label('Plate Mask', rotation=270, labelpad=15, fontsize=9)
        next_row += 1

    # Optional bone mask row (using POY1 Ground Truth)
    if add_bone_mask:
        bone = (poy1_gt_hu > bone_threshold).astype(np.float32)
        D, H, W = bone.shape
        md, mh, mw = D//2, H//2, W//2
        axes[next_row, 0].imshow(poy1_gt_hu[:, mh, :].T, cmap='gray', origin='lower')
        axes[next_row, 0].imshow(bone[:, mh, :].T, cmap='Reds', alpha=0.35, origin='lower')
        axes[next_row, 0].set_title('Bone Mask on GT (Sagittal)')
        axes[next_row, 0].set_ylabel(f'Bone Mask on GT (> {bone_threshold} HU)', fontsize=11, fontweight='bold')
        axes[next_row, 0].set_xticks([]); axes[next_row, 0].set_yticks([])
        axes[next_row, 1].imshow(poy1_gt_hu[md, :, :].T, cmap='gray', origin='lower')
        axes[next_row, 1].imshow(bone[md, :, :].T, cmap='Reds', alpha=0.35, origin='lower')
        axes[next_row, 1].set_title('Bone Mask on GT (Coronal)')
        axes[next_row, 1].set_xticks([]); axes[next_row, 1].set_yticks([])
        axes[next_row, 2].imshow(poy1_gt_hu[:, :, mw], cmap='gray', origin='lower')
        imB = axes[next_row, 2].imshow(bone[:, :, mw], cmap='Reds', alpha=0.35, origin='lower')
        axes[next_row, 2].set_title('Bone Mask on GT (Axial)')
        axes[next_row, 2].set_xticks([]); axes[next_row, 2].set_yticks([])
        cbar = plt.colorbar(imB, ax=axes[next_row, :], shrink=0.8, pad=0.02)
        cbar.set_label('Bone (GT)', rotation=270, labelpad=15, fontsize=9)

    # With constrained_layout enabled, avoid tight_layout and bbox_inches='tight'.
    # Slightly reduce top to ensure suptitle fits well in some backends.
    fig.subplots_adjust(top=0.95)
    plt.savefig(save_path, dpi=150)
    plt.close()

# ----------------------- Comprehensive Metrics Functions -----
def compute_ssim_3d(pred, target, window_size=11, data_range=None):
    """
    Compute 3D SSIM between predicted and target volumes.
    Simplified 3D SSIM calculation using Gaussian filters.
    """
    if data_range is None:
        data_range = max(pred.max() - pred.min(), target.max() - target.min())
    
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    # Convert to float64 for numerical stability
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    
    # Compute local means
    mu1 = gaussian_filter(pred, sigma=1.5)
    mu2 = gaussian_filter(target, sigma=1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute local variances and covariance
    sigma1_sq = gaussian_filter(pred ** 2, sigma=1.5) - mu1_sq
    sigma2_sq = gaussian_filter(target ** 2, sigma=1.5) - mu2_sq
    sigma12 = gaussian_filter(pred * target, sigma=1.5) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return float(np.mean(ssim_map))

def compute_psnr(pred, target, data_range=None):
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    
    if data_range is None:
        data_range = max(pred.max() - pred.min(), target.max() - target.min())
    
    psnr = 20 * np.log10(data_range / np.sqrt(mse))
    return float(psnr)

def compute_dice_score(pred_hu, target_hu, threshold=METRICS_BONE_HU_THRESHOLD):
    """
    Compute overall DICE score for bone segmentation.
    Uses HU threshold to segment bone regions (> 200 HU for mandibular bone).
    
    Args:
        pred_hu: Predicted volume in HU
        target_hu: Target volume in HU
        threshold: HU threshold for bone segmentation (default 200 for bone)
    
    Returns:
        float: DICE coefficient (0-1, higher is better)
    """
    # Segment bone regions
    pred_bone = (pred_hu > threshold).astype(np.float32)
    target_bone = (target_hu > threshold).astype(np.float32)
    
    # Compute DICE: 2*|A∩B| / (|A| + |B|)
    intersection = np.sum(pred_bone * target_bone)
    union_size = np.sum(pred_bone) + np.sum(target_bone)
    
    if union_size == 0:
        return float('nan')  # Both empty - undefined (use nanmean for aggregation)
    
    dice = (2.0 * intersection) / (union_size + 1e-8)
    return float(dice)

def compute_hd95(pred_hu, target_hu, threshold=METRICS_BONE_HU_THRESHOLD, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Compute 95th percentile Hausdorff Distance (HD95) for bone segmentation.
    HD95 measures surface distance between predicted and ground truth segmentations.
    
    Args:
        pred_hu: Predicted volume in HU
        target_hu: Target volume in HU
        threshold: HU threshold for bone segmentation
        voxel_spacing: Physical voxel spacing in mm (D, H, W)
    
    Returns:
        float: HD95 in mm (lower is better)
    """
    from scipy.ndimage import distance_transform_edt
    
    # Segment bone regions as boolean masks (safer for ~ and XOR)
    pred_bone = (pred_hu > threshold)
    target_bone = (target_hu > threshold)
    
    # If either segmentation is empty, return large penalty
    if (not pred_bone.any()) or (not target_bone.any()):
        return 100.0  # Large penalty for missing segmentation
    
    # Compute boundaries (surface voxels)
    from scipy.ndimage import binary_erosion
    pred_surface = np.logical_xor(pred_bone, binary_erosion(pred_bone))
    target_surface = np.logical_xor(target_bone, binary_erosion(target_bone))
    
    # If no surface found, return 0
    if (not pred_surface.any()) or (not target_surface.any()):
        return 0.0
    
    # Distance from pred surface to nearest target surface (invert boolean properly)
    target_dt = distance_transform_edt(~target_surface, sampling=voxel_spacing)
    pred_surface_coords = np.where(pred_surface)
    distances_pred_to_target = target_dt[pred_surface_coords]
    
    # Distance from target surface to nearest pred surface (invert boolean properly)
    pred_dt = distance_transform_edt(~pred_surface, sampling=voxel_spacing)
    target_surface_coords = np.where(target_surface)
    distances_target_to_pred = pred_dt[target_surface_coords]
    
    # Combine all distances
    all_distances = np.concatenate([distances_pred_to_target, distances_target_to_pred])
    
    # Compute 95th percentile
    hd95 = float(np.percentile(all_distances, 95))
    
    return hd95

def ms_ssim3d_simple(x: torch.Tensor, y: torch.Tensor, data_range: float = 2.0) -> torch.Tensor:
    """Simple MS-SSIM.

    NOTE: Inputs are expected to be normalized volumes (typically in [-1,1]),
    so default data_range=2.0.
    """
    C1 = (0.01 * float(data_range)) ** 2
    C2 = (0.03 * float(data_range)) ** 2
    def ssim3d_basic(a, b, C1=C1, C2=C2, win=3):
        pad = win // 2
        pool = nn.AvgPool3d(win, stride=1, padding=0)
        apad = F.pad(a, (pad,)*6, mode='replicate')
        bpad = F.pad(b, (pad,)*6, mode='replicate')
        mu_a = pool(apad)
        mu_b = pool(bpad)
        mu_a2, mu_b2, mu_ab = mu_a**2, mu_b**2, mu_a*mu_b
        sigma_a2 = pool(apad*apad) - mu_a2
        sigma_b2 = pool(bpad*bpad) - mu_b2
        sigma_ab = pool(apad*bpad) - mu_ab
        ssim = ((2*mu_ab + C1)*(2*sigma_ab + C2)) / ((mu_a2 + mu_b2 + C1)*(sigma_a2 + sigma_b2 + C2))
        return ssim.mean()
    s1 = ssim3d_basic(x, y)
    s2 = ssim3d_basic(F.avg_pool3d(x, 2), F.avg_pool3d(y, 2))
    return 0.5 * (s1 + s2)

def ssim3d_map(x: torch.Tensor, y: torch.Tensor, data_range: float = 2.0, win: int = 3) -> torch.Tensor:
    """Return per-voxel SSIM map (same spatial shape as x/y)."""
    C1 = (0.01 * float(data_range)) ** 2
    C2 = (0.03 * float(data_range)) ** 2
    pad = win // 2
    pool = nn.AvgPool3d(win, stride=1, padding=0)
    xpad = F.pad(x, (pad,) * 6, mode='replicate')
    ypad = F.pad(y, (pad,) * 6, mode='replicate')
    mu_x = pool(xpad)
    mu_y = pool(ypad)
    mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y
    sigma_x2 = pool(xpad * xpad) - mu_x2
    sigma_y2 = pool(ypad * ypad) - mu_y2
    sigma_xy = pool(xpad * ypad) - mu_xy
    ssim_val = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return ssim_val

def ms_ssim3d_masked(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Masked MS-SSIM: weight SSIM maps by a soft mask (e.g., bone voxels) across scales."""
    if mask.dim() == 4:
        mask = mask.unsqueeze(1)
    mask = mask.to(dtype=x.dtype, device=x.device)
    # Scale 1
    s1_map = ssim3d_map(x, y, data_range=2.0)
    w1 = mask
    s1 = (s1_map * w1).sum() / (w1.sum() + eps)
    # Scale 2
    x2, y2 = F.avg_pool3d(x, 2), F.avg_pool3d(y, 2)
    w2 = F.avg_pool3d(w1, 2)
    s2_map = ssim3d_map(x2, y2, data_range=2.0)
    s2 = (s2_map * w2).sum() / (w2.sum() + eps)
    return 0.5 * (s1 + s2)

def compute_comprehensive_metrics(pred_hu, target_hu, pred_norm=None, target_norm=None, plate_mask=None):
    """
    Compute comprehensive metrics.
    Returns MAE_all_HU, MAE_bone_HU, MS_SSIM, MS_SSIM_bone, Dice_bone
    
    Args:
        pred_hu: Predicted HU volume [D, H, W]
        target_hu: Target HU volume [D, H, W]
        pred_norm: Predicted normalized volume (optional)
        target_norm: Target normalized volume (optional)
        plate_mask: Binary plate mask [D, H, W] to exclude from metrics (optional)
    """
    metrics = {}
    
    # Create exclusion mask (True = include, False = exclude)
    if plate_mask is not None and EXCLUDE_PLATE_FROM_METRICS:
        include_mask = ~(plate_mask.astype(bool))  # Exclude plate voxels
    else:
        include_mask = np.ones_like(target_hu, dtype=bool)  # Include all
    
    # MAE in HU
    if include_mask.any():
        metrics['MAE_all_HU'] = float(np.mean(np.abs(pred_hu[include_mask] - target_hu[include_mask])))
    else:
        metrics['MAE_all_HU'] = float(np.mean(np.abs(pred_hu - target_hu)))
    
    # Bone-specific MAE (excluding plate if enabled)
    bone_mask = (target_hu > METRICS_BONE_HU_THRESHOLD)
    bone_and_include = bone_mask & include_mask
    if bone_and_include.any():
        metrics['MAE_bone_HU'] = float(np.mean(np.abs(pred_hu[bone_and_include] - target_hu[bone_and_include])))
    else:
        metrics['MAE_bone_HU'] = float('nan')  # Undefined when no bone voxels exist
    
    # MS-SSIM: compute on normalized [-1,1] tensors
    if pred_norm is not None and target_norm is not None:
        pred_t = torch.from_numpy(pred_norm).unsqueeze(0).unsqueeze(0).float().to(device)
        target_t = torch.from_numpy(target_norm).unsqueeze(0).unsqueeze(0).float().to(device)
    else:
        # fallback: normalize HU to [-1,1] (clip to HU_RANGE)
        pred_clip = np.clip(pred_hu, HU_RANGE[0], HU_RANGE[1])
        target_clip = np.clip(target_hu, HU_RANGE[0], HU_RANGE[1])
        pred_norm_tmp = 2.0 * (pred_clip - HU_RANGE[0]) / (HU_RANGE[1] - HU_RANGE[0]) - 1.0
        target_norm_tmp = 2.0 * (target_clip - HU_RANGE[0]) / (HU_RANGE[1] - HU_RANGE[0]) - 1.0
        pred_t = torch.from_numpy(pred_norm_tmp).unsqueeze(0).unsqueeze(0).float().to(device)
        target_t = torch.from_numpy(target_norm_tmp).unsqueeze(0).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        # MS-SSIM on full volume (with plate exclusion if enabled)
        if plate_mask is not None and EXCLUDE_PLATE_FROM_METRICS:
            include_mask_t = torch.from_numpy(include_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            ms_ssim_val = ms_ssim3d_masked(pred_t, target_t, include_mask_t)
        else:
            ms_ssim_val = ms_ssim3d_simple(pred_t, target_t)
        metrics['MS_SSIM'] = float(ms_ssim_val.item())
        
        # MS-SSIM on bone region only (excluding plate if enabled)
        if bone_and_include.any():
            # Create combined bone+include mask tensor
            bone_include_mask_t = torch.from_numpy(bone_and_include.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            # Use proper masked MS-SSIM computation
            ms_ssim_bone_val = ms_ssim3d_masked(pred_t, target_t, bone_include_mask_t)
            metrics['MS_SSIM_bone'] = float(ms_ssim_bone_val.item())
        else:
            metrics['MS_SSIM_bone'] = float('nan')  # Undefined when no bone voxels exist
    
    # Dice on bone (>HU threshold), excluding plate if enabled
    if plate_mask is not None and EXCLUDE_PLATE_FROM_METRICS:
        # Compute dice only on non-plate regions
        pred_bone = (pred_hu > METRICS_BONE_HU_THRESHOLD) & include_mask
        target_bone = (target_hu > METRICS_BONE_HU_THRESHOLD) & include_mask
        intersection = np.sum(pred_bone & target_bone)
        union = np.sum(pred_bone) + np.sum(target_bone)
        if union == 0:
            metrics['Dice_bone'] = float('nan') if (not pred_bone.any() and not target_bone.any()) else 0.0
        else:
            metrics['Dice_bone'] = float(2.0 * intersection / (union + 1e-8))
    else:
        metrics['Dice_bone'] = compute_dice_score(pred_hu, target_hu, threshold=METRICS_BONE_HU_THRESHOLD)
    
    return metrics


def _make_w_slab_mask_np(shape: tuple[int, int, int], w_start: int, w_end: int) -> np.ndarray:
    """Hard slab mask selecting W slices in [w_start, w_end] inclusive."""
    if len(shape) != 3:
        raise ValueError(f"Expected 3D shape (D,H,W), got: {shape}")
    D, H, W = shape
    if W <= 0:
        return np.zeros((D, H, W), dtype=bool)
    w_start_i = int(w_start)
    w_end_i = int(w_end)
    if w_end_i < w_start_i:
        w_start_i, w_end_i = w_end_i, w_start_i
    w_start_i = max(0, min(W - 1, w_start_i))
    w_end_i = max(0, min(W - 1, w_end_i))
    mask = np.zeros((D, H, W), dtype=bool)
    mask[:, :, w_start_i:w_end_i + 1] = True
    return mask


def compute_comprehensive_metrics_middle_slab(pred_hu, target_hu, pred_norm=None, target_norm=None, plate_mask=None):
    """Same 5 metrics as compute_comprehensive_metrics, but restricted to the middle-slab W range.
    
    Args:
        pred_hu: Predicted HU volume [D, H, W]
        target_hu: Target HU volume [D, H, W]
        pred_norm: Predicted normalized volume (optional)
        target_norm: Target normalized volume (optional)
        plate_mask: Binary plate mask [D, H, W] to exclude from metrics (optional)
    """
    w_start = int(globals().get('MIDDLE_SLAB_IMAGE_SLICE_START', 20))
    w_end = int(globals().get('MIDDLE_SLAB_IMAGE_SLICE_END', 28))

    slab_mask = _make_w_slab_mask_np(target_hu.shape, w_start, w_end)
    
    # Create exclusion mask combining slab and plate exclusion
    if plate_mask is not None and EXCLUDE_PLATE_FROM_METRICS:
        include_mask = slab_mask & ~(plate_mask.astype(bool))  # Slab AND not plate
    else:
        include_mask = slab_mask
    
    metrics = {}

    # MAE in HU (slab only, excluding plate if enabled)
    if include_mask.any():
        metrics['MAE_all_HU_mid'] = float(np.mean(np.abs(pred_hu[include_mask] - target_hu[include_mask])))
    else:
        metrics['MAE_all_HU_mid'] = float('nan')

    # Bone-specific MAE (slab only, excluding plate if enabled)
    bone_mask = (target_hu > METRICS_BONE_HU_THRESHOLD) & include_mask
    if bone_mask.any():
        metrics['MAE_bone_HU_mid'] = float(np.mean(np.abs(pred_hu[bone_mask] - target_hu[bone_mask])))
    else:
        metrics['MAE_bone_HU_mid'] = float('nan')  # Undefined when no bone voxels in slab

    # MS-SSIM (slab only) in normalized [-1,1]
    if pred_norm is not None and target_norm is not None:
        pred_t = torch.from_numpy(pred_norm).unsqueeze(0).unsqueeze(0).float().to(device)
        target_t = torch.from_numpy(target_norm).unsqueeze(0).unsqueeze(0).float().to(device)
    else:
        pred_clip = np.clip(pred_hu, HU_RANGE[0], HU_RANGE[1])
        target_clip = np.clip(target_hu, HU_RANGE[0], HU_RANGE[1])
        pred_norm_tmp = 2.0 * (pred_clip - HU_RANGE[0]) / (HU_RANGE[1] - HU_RANGE[0]) - 1.0
        target_norm_tmp = 2.0 * (target_clip - HU_RANGE[0]) / (HU_RANGE[1] - HU_RANGE[0]) - 1.0
        pred_t = torch.from_numpy(pred_norm_tmp).unsqueeze(0).unsqueeze(0).float().to(device)
        target_t = torch.from_numpy(target_norm_tmp).unsqueeze(0).unsqueeze(0).float().to(device)

    include_mask_t = torch.from_numpy(include_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        metrics['MS_SSIM_mid'] = float(ms_ssim3d_masked(pred_t, target_t, include_mask_t).item()) if include_mask.any() else float('nan')
        if bone_mask.any():
            bone_mask_t = torch.from_numpy(bone_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            metrics['MS_SSIM_bone_mid'] = float(ms_ssim3d_masked(pred_t, target_t, bone_mask_t).item())
        else:
            metrics['MS_SSIM_bone_mid'] = float('nan')  # Undefined when no bone in slab

    # Dice on bone (>HU threshold), slab only, excluding plate if enabled
    pred_bin = (pred_hu > METRICS_BONE_HU_THRESHOLD) & include_mask
    tgt_bin = (target_hu > METRICS_BONE_HU_THRESHOLD) & include_mask
    inter = float(np.logical_and(pred_bin, tgt_bin).sum())
    denom = float(pred_bin.sum() + tgt_bin.sum())
    if denom == 0:
        metrics['Dice_bone_mid'] = float('nan') if (not pred_bin.any() and not tgt_bin.any()) else 0.0
    else:
        metrics['Dice_bone_mid'] = float((2.0 * inter) / (denom + 1e-8))

    return metrics

def create_metrics_excel_with_footnotes(metrics_list, save_path):
    """
    Create metrics Excel.
    Columns: epoch, train metrics, test metrics
    """
    df = pd.DataFrame(metrics_list)
    
    col_order = ['epoch']
    
    # Training/eval losses + diagnostics
    # Teacher-alignment diagnostics are computed from the SVF teacher trajectory in latent space
    # and are comparable across different LOSS_MODE and LYAPUNOV_ALPHA.
    train_cols = [
        'avg_total_loss',
        'avg_lyapunov_loss',
        'avg_rmse_v_teacher_tangent',
        'avg_cos_v_teacher_tangent',
        'avg_velocity_magnitude_ratio',
        'avg_endpoint_mae_to_teacher',
        'avg_fm_loss',
        'avg_endpoint_loss',
    ]
    for c in train_cols:
        if c in df.columns:
            col_order.append(c)
    
    # Test metrics (updated: MAE_all_HU, MAE_bone_HU, MS_SSIM, MS_SSIM_bone, Dice_bone)
    # + Middle-slab-only versions (W slices in [MIDDLE_SLAB_IMAGE_SLICE_START, MIDDLE_SLAB_IMAGE_SLICE_END])
    test_cols = [
        'MAE_all_HU', 'MAE_bone_HU', 'MS_SSIM', 'MS_SSIM_bone', 'Dice_bone',
        'MAE_all_HU_mid', 'MAE_bone_HU_mid', 'MS_SSIM_mid', 'MS_SSIM_bone_mid', 'Dice_bone_mid'
    ]
    missing_mid = [
        c for c in ['MAE_all_HU_mid', 'MAE_bone_HU_mid', 'MS_SSIM_mid', 'MS_SSIM_bone_mid', 'Dice_bone_mid']
        if c not in df.columns
    ]
    if missing_mid:
        print(f"⚠️ Middle-slab metric columns missing (won't appear in Excel): {missing_mid}")
    for c in test_cols:
        if c in df.columns:
            col_order.append(c)
    
    # Any remaining columns
    remaining = [c for c in df.columns if c not in col_order]
    col_order.extend(remaining)
    
    df = df[col_order]
    
    # Add summary rows (use nanmean/nanstd to handle undefined bone metrics).
    # Exclude explicit baseline rows (epoch < 0) from summary stats.
    summary_df = df
    if 'epoch' in df.columns:
        epoch_num = pd.to_numeric(df['epoch'], errors='coerce')
        keep_mask = (epoch_num >= 0)
        if keep_mask.any():
            summary_df = df.loc[keep_mask]

    avg_row = summary_df.select_dtypes(include=[np.number]).apply(lambda x: np.nanmean(x))
    avg_row['epoch'] = 'AVERAGE'
    std_row = summary_df.select_dtypes(include=[np.number]).apply(lambda x: np.nanstd(x))
    std_row['epoch'] = 'STD_DEV'
    df = pd.concat([df, pd.DataFrame([avg_row]), pd.DataFrame([std_row])], ignore_index=True)
    
    # Round numeric columns to 4 decimals for readability
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].round(4)
    
    # Write to Excel with a definitions sheet
    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Epoch Metrics')
        
        # Metric definitions (updated)
        notes = pd.DataFrame({
            'Metric': [
                'avg_rmse_v_teacher_tangent',
                'avg_cos_v_teacher_tangent',
                'avg_velocity_magnitude_ratio',
                'avg_endpoint_mae_to_teacher',
                'MAE_all_HU', 'MAE_bone_HU', 'MS_SSIM', 'MS_SSIM_bone', 'Dice_bone',
                'MAE_all_HU_mid', 'MAE_bone_HU_mid', 'MS_SSIM_mid', 'MS_SSIM_bone_mid', 'Dice_bone_mid',
                '⚠️ IMPORTANT', '⚠️ NaN handling'
            ],
            'Meaning': [
                'TeacherTangentRMSE: E_t[ sqrt( mean((v_student(x*(t), t) - dx*(t)/dt)^2) ) ] (lower is better; comparable across LOSS_MODE and LYAPUNOV_ALPHA)',
                'TangentCosSim: E_t[ cos( v_student(x*(t), t), dx*(t)/dt ) ] on flattened volumes (higher is better; range [-1,1]; comparable across LOSS_MODE)',
                'VelocityScaleAgreement01: E_t[ min(r,1/r) ], r=mean(|v_student(x*(t),t)|)/mean(|dx*/dt|) in [0,1] (higher is better; comparable across LOSS_MODE)',
                "EndMAE_toTeacher: mean(|x1_pred - x*(1)|) comparing the student's endpoint x1_pred (computed using USE_DIRECT_ONE_STEP_INFERENCE settings) to teacher endpoint warp x*(1)=Warp(x0,exp(v_svf)) (lower is better; shows if student learns teacher deformation)",
                'Mean absolute error in HU (all voxels)',
                'Mean absolute error in HU (bone voxels only, >threshold; NaN if no bone)',
                'Multi-scale SSIM (similarity, 0-1, higher is better)',
                'Multi-scale SSIM computed on bone region only (NaN if no bone)',
                'Dice coefficient for bone segmentation (NaN if both pred/GT have no bone)',
                'MAE_all_HU computed only on the middle slab W slices',
                'MAE_bone_HU computed only on the middle slab W slices (NaN if no bone in slab)',
                'MS_SSIM computed only on the middle slab W slices',
                'MS_SSIM_bone computed only on the middle slab W slices (NaN if no bone in slab)',
                'Dice_bone computed only on the middle slab W slices (NaN if both have no bone in slab)',
                "Metrics use the same endpoint inference mode as USE_DIRECT_ONE_STEP_INFERENCE: if True, ONE-STEP endpoint x1_pred=x0+v(x0,0); if False, multi-step ODE-style integration (Euler/Heun/RK4) with EVAL_INTEGRATION_STEPS and INTEGRATION_METHOD.",
                'Bone-specific metrics return NaN when undefined (no bone voxels in region). Aggregation uses np.nanmean() to exclude undefined values from averages. This prevents biasing averages toward 0.0 for regions where bone metrics are not applicable.'
            ]
        })
        notes.to_excel(writer, index=False, sheet_name='Metric Definitions')
        
        # Widen columns for readability
        ws = writer.book["Epoch Metrics"]
        for col_cells in ws.columns:
            max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col_cells)
            ws.column_dimensions[col_cells[0].column_letter].width = min(max(10, max_len + 2), 35)
    
    print(f"Metrics saved to: {save_path}")
    print("   - 'Epoch Metrics' sheet: per-epoch rows + AVERAGE/STD_DEV")
    print("   - 'Metric Definitions' sheet: metric meanings")
    return
# ----------------------- Dataset -----------------------------
class ROI3DDataset(Dataset):
    """
    Robust pairing with augmented file support:
    - Accepts POD5 like: {case}_POD5_{ROI}[_aug{N}][_RAS].nii.gz
    - Finds POY1 with same pattern
    - Optionally generates bone masks from POY1 or POD5 using HU threshold
    """
    def __init__(self, pod5_dir, poy1_dir, normalize=True, recursive=False):
        self.pod5_dir = Path(pod5_dir)
        self.poy1_dir = Path(poy1_dir)
        self.normalize = normalize
        self.pairs = []

        pod5_pat = re.compile(
            r'^(?P<case>\d+)_POD5_(?P<roi>ROI\d+)(?:_(?P<aug>aug\d+))?(?:_RAS)?\.nii(?:\.gz)?$',
            re.IGNORECASE
        )

        globber = self.pod5_dir.rglob if recursive else self.pod5_dir.glob
        pod5_files = list(globber("*.nii")) + list(globber("*.nii.gz"))

        def first_existing(cands):
            for c in cands:
                if c.exists():
                    return c
            return None

        for pod5_file in pod5_files:
            m = pod5_pat.match(pod5_file.name)
            if not m:
                continue

            case_str = m.group("case")
            roi_str = m.group("roi")
            aug = m.group("aug")
            
            # Extract augmentation ID (default to 0 for original)
            aug_id = 0
            if aug:
                aug_id = int(aug.replace('aug', ''))

            # Build candidate POY1 names
            name_roots = []
            if aug:
                name_roots += [
                    f"{case_str}_POY1_{roi_str}_{aug}_RAS",
                    f"{case_str}_POY1_{roi_str}_{aug}",
                ]
            name_roots += [
                f"{case_str}_POY1_{roi_str}_RAS",
                f"{case_str}_POY1_{roi_str}",
            ]

            candidates = []
            for root in name_roots:
                candidates += [self.poy1_dir / f"{root}.nii.gz",
                               self.poy1_dir / f"{root}.nii"]

            target = first_existing(candidates)
            if target is None:
                continue

            # Search for corresponding plate mask file (Year-1 / POY1 only).
            # IMPORTANT: As requested, we only use the POY1 (Year-1) plate mask for loss/metrics/visualization.
            # Expected patterns include:
            # - {case}_POY1_{ROI}_plateMask_{aug}.nii.gz          (augmented_v4)
            # - {case}_POY1_{ROI}_plateMask.nii.gz               (non-aug)
            plate_mask_path = None
            plate_mask_roots: list[str] = []
            prefixes = ["POY1"]
            if aug:
                for pref in prefixes:
                    plate_mask_roots += [
                        f"{case_str}_{pref}_{roi_str}_plateMask_{aug}",
                        f"{case_str}_{pref}_{roi_str}_plateMask_{aug}_RAS",
                    ]
            for pref in prefixes:
                plate_mask_roots += [
                    f"{case_str}_{pref}_{roi_str}_plateMask",
                    f"{case_str}_{pref}_{roi_str}_plateMask_RAS",
                ]

            plate_mask_candidates: list[Path] = []
            for root in plate_mask_roots:
                base_dir = self.poy1_dir
                plate_mask_candidates += [
                    base_dir / f"{root}.nii.gz",
                    base_dir / f"{root}.nii",
                ]
            plate_mask_path = first_existing(plate_mask_candidates)

            self.pairs.append({
                "pod5_path": pod5_file,
                "poy1_path": target,
                "plate_mask_path": plate_mask_path,  # None if not found
                "case_id": int(case_str),
                "roi_num": int(roi_str.replace('ROI', '')),
                "aug_id": aug_id,  # Track augmentation ID
            })

        # CRITICAL: Sort pairs for deterministic train/test split
        # Sort by (case_id, roi_num, filename) to ensure identical ordering
        self.pairs.sort(key=lambda p: (p["case_id"], p["roi_num"], str(p["pod5_path"])))

        # Count plate masks found
        plate_mask_count = sum(1 for p in self.pairs if p.get("plate_mask_path") is not None)
        print(f"📊 Found {len(self.pairs)} paired ROI volumes")
        print(f"   Plate masks: {plate_mask_count}/{len(self.pairs)} ({100*plate_mask_count/max(1,len(self.pairs)):.1f}%)")
        if plate_mask_count == 0 and len(self.pairs) > 0:
            print(
                "⚠️  Plate masks not found. Expected names like: "
                "{case}_POY1_{ROI}_plateMask_augN.nii.gz (or non-aug {case}_POY1_{ROI}_plateMask.nii.gz) "
                "located in the POY1 directory you configured."
            )
        self.unique_cases = sorted({p["case_id"] for p in self.pairs})
        self.unique_rois = sorted({p["roi_num"] for p in self.pairs})
        self.n_cases = len(self.unique_cases)
        self.n_rois = len(self.unique_rois)
        print(f"   Cases: {self.n_cases}, ROI types: {self.n_rois}")


    def __len__(self): return len(self.pairs)

    def _load(self, path: Path):
        img = nib.load(str(path))
        v = img.get_fdata().astype(np.float32)
        v = _maybe_resample_to_roi(v, ROI_SHAPE)
        if self.normalize:
            v = _clip_and_norm_to_unit(v, HU_RANGE)
        return v
    
    def _load_raw_hu(self, path: Path):
        """Load raw HU values (no normalization) for bone mask generation"""
        img = nib.load(str(path))
        v = img.get_fdata().astype(np.float32)
        v = _maybe_resample_to_roi(v, ROI_SHAPE)
        return v

    def _load_plate_mask(self, path: Path | None):
        """Load plate mask from NIfTI file, return binary mask [D,H,W]"""
        if path is None or not path.exists():
            return None
        img = nib.load(str(path))
        v = img.get_fdata().astype(np.float32)
        v = _maybe_resample_to_roi(v, ROI_SHAPE)
        # Ensure binary (threshold at 0.5 in case of interpolation)
        v = (v > 0.5).astype(np.float32)
        return v

    def __getitem__(self, idx):
        p = self.pairs[idx]
        pod5 = self._load(p["pod5_path"])
        poy1 = self._load(p["poy1_path"])

        bone_mask = torch.zeros(1, *pod5.shape, dtype=torch.float32)
        
        # Load plate mask if available (for loss/metrics exclusion and visualization)
        plate_mask_np = self._load_plate_mask(p.get("plate_mask_path"))
        if plate_mask_np is not None:
            plate_mask = torch.from_numpy(plate_mask_np).unsqueeze(0)  # [1,D,H,W]
            has_plate_mask = True
        else:
            plate_mask = torch.zeros(1, *pod5.shape, dtype=torch.float32)
            has_plate_mask = False
        
        result = {
            "pod5": torch.from_numpy(pod5).unsqueeze(0),  # [1,D,H,W]
            "poy1": torch.from_numpy(poy1).unsqueeze(0),
            "bone_mask": bone_mask,  # [1,D,H,W] (zeros if conditioning disabled)
            "plate_mask": plate_mask,  # [1,D,H,W] binary plate mask (zeros if not found)
            "has_plate_mask": torch.tensor(1.0 if has_plate_mask else 0.0, dtype=torch.float32),
            "case_id": self.unique_cases.index(p["case_id"]),
            "roi_num": self.unique_rois.index(p["roi_num"]),
            "meta": {
                "case_id": p["case_id"],
                "roi_num": p["roi_num"],
                "aug_id": p["aug_id"],  # Track augmentation ID
                "pod5_path": str(p["pod5_path"]),
                "poy1_path": str(p["poy1_path"]),
                "plate_mask_path": str(p["plate_mask_path"]) if p.get("plate_mask_path") else "",
                "has_plate_mask": has_plate_mask,
            }
        }
        return result

# ==================== Semi-Online Augmentation Sampler ====================
class SemiOnlineAugmentationSampler(torch.utils.data.Sampler):
    """
    Sampler for semi-online augmentation:
    - For each ROI, picks aug0 + NUM_AUG_PER_ROI random augmentations each epoch
    - Different random selection each epoch
    - Works with Subset objects from train_test_split
    """
    def __init__(self, dataset, num_random_aug_per_roi: int = NUM_AUG_PER_ROI, seed: int = RANDOM_SEED):
        self.dataset = dataset
        self.num_random_aug_per_roi = max(0, int(num_random_aug_per_roi))
        self.rng = np.random.default_rng(seed)
        
        # Handle Subset vs raw Dataset
        from torch.utils.data import Subset
        self.is_subset = isinstance(dataset, Subset)
        if self.is_subset:
            self.base_dataset = dataset.dataset
            self.valid_indices = set(dataset.indices)
            self.base_to_subset_idx = {base_idx: subset_idx for subset_idx, base_idx in enumerate(dataset.indices)}
        else:
            self.base_dataset = dataset
            self.valid_indices = set(range(len(dataset)))
            self.base_to_subset_idx = None
        
        self._build_group_indices()
    
    def _build_group_indices(self):
        """Build grouping of indices by (case_id, roi_num), pre-separated by aug_id."""
        self.group_indices = {}
        for idx in self.valid_indices:
            pair = self.base_dataset.pairs[idx]
            key = (pair["case_id"], pair["roi_num"])
            if key not in self.group_indices:
                self.group_indices[key] = {"aug0": [], "others": []}
            
            if pair["aug_id"] == 0:
                self.group_indices[key]["aug0"].append(idx)
            else:
                self.group_indices[key]["others"].append(idx)
    
    def __len__(self):
        total = 0
        for key, group in self.group_indices.items():
            aug0_count = len(group["aug0"])
            other_count = len(group["others"])
            if aug0_count == 0:
                k = min(aug0_count + other_count, 1 + self.num_random_aug_per_roi)
                total += k
            else:
                k = min(other_count, self.num_random_aug_per_roi)
                total += 1 + k
        return total
    
    def __iter__(self):
        all_indices = []
        for key, group in self.group_indices.items():
            orig_indices = group["aug0"]
            other_indices = group["others"]
            
            # Always include first aug0 if available
            if orig_indices:
                all_indices.append(orig_indices[0])
            
            # Randomly sample from other augmentations
            if other_indices and self.num_random_aug_per_roi > 0:
                n_pick = min(len(other_indices), self.num_random_aug_per_roi)
                picked = self.rng.choice(other_indices, size=n_pick, replace=False)
                all_indices.extend(picked.tolist())
            
            # Edge case fallback
            elif not orig_indices and other_indices:
                n_pick = min(len(other_indices), 1 + self.num_random_aug_per_roi)
                picked = self.rng.choice(other_indices, size=n_pick, replace=False)
                all_indices.extend(picked.tolist())
        
        # Shuffle and convert to subset indices if needed
        self.rng.shuffle(all_indices)
        if self.is_subset:
            for base_idx in all_indices:
                yield self.base_to_subset_idx[base_idx]
        else:
            for idx in all_indices:
                yield idx

# ----------------------- UNet Flow Network -------------------
class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding with learned projection."""
    def __init__(self, d_t=64, M=1000, out_channels=128):
        super().__init__()
        self.d_t = d_t
        self.M = M
        self.proj = nn.Sequential(
            nn.Linear(d_t, out_channels * 4),
            nn.GELU(),
            nn.Linear(out_channels * 4, out_channels)
        )
    
    def forward(self, t):
        # t: [B]
        h = self.d_t // 2
        freqs = torch.exp(-math.log(self.M) * torch.arange(h, device=t.device) / max(h-1, 1))
        ang = (t[:, None] * self.M) * freqs[None]
        emb = torch.cat([ang.sin(), ang.cos()], dim=-1)
        if self.d_t % 2:
            emb = F.pad(emb, (0, 1))
        return self.proj(emb)  # [B, out_channels]


class ResBlock3D(nn.Module):
    """Residual block with time and conditioning modulation."""
    def __init__(self, channels, emb_channels, dropout=0.1, use_bone_mask_film=False):
        super().__init__()
        self.use_bone_mask_film = use_bone_mask_film
        
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        
        # Time + conditioning projection (for time embedding)
        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, channels * 2)  # scale and shift
        )
        
        # Optional bone mask FiLM modulation (separate from time)
        if use_bone_mask_film:
            self.bone_mask_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_channels, channels * 2)  # additional scale and shift
            )
        
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.dropout = nn.Dropout3d(dropout)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
    
    def forward(self, x, emb, bone_mask_emb=None, spatial_gate=None):
        """
        x: [B, C, D, H, W]
        emb: [B, emb_channels] (time + case conditioning combined)
        bone_mask_emb: [B, emb_channels] (optional bone mask FiLM embedding)
        spatial_gate: [B, 1, D, H, W] (optional). If provided, applies spatial gating
            AFTER norm1 and BEFORE conv1 so it cannot be trivially cancelled by norm.
        """
        x_n = self.norm1(x)
        if spatial_gate is not None:
            # spatial_gate is [B,1,D,H,W] and broadcasts across channels
            x_n = x_n * spatial_gate
        h = self.conv1(F.silu(x_n))
        
        # Apply time/conditioning modulation (scale and shift)
        emb_out = self.emb_proj(emb)[:, :, None, None, None]  # [B, 2*C, 1, 1, 1]
        scale, shift = emb_out.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        
        # Apply bone mask modulation if available (additive FiLM)
        if self.use_bone_mask_film and bone_mask_emb is not None:
            bone_emb_out = self.bone_mask_proj(bone_mask_emb)[:, :, None, None, None]
            bone_scale, bone_shift = bone_emb_out.chunk(2, dim=1)
            h = h * (1 + bone_scale) + bone_shift
        
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return x + h

class UNetFlowNetwork(nn.Module):
    """
    UNet architecture for flow matching in IMAGE SPACE.
    Input: [B, 1, 48, 48, 48] image + time + optional case ID conditioning
    Output: [B, 1, 48, 48, 48] velocity field
    
    Architecture:
    - Encoder: 48 -> 24 -> 12 (2 downsamples with Conv3d stride=2)
    - Bottleneck: 12×12×12 (with optional Flash Attention for global context)
    - Decoder: 12 -> 24 -> 48 (2 upsamples with ConvTranspose3d stride=2)
    
    """
    def __init__(self, image_channels=1, base_channels=UNET_BASE_CHANNELS,
                 use_attention=UNET_USE_ATTENTION,
                 ):
        super().__init__()
        self.image_channels = image_channels
        self.use_attention = use_attention
        self.use_middle_slab_prior = bool(globals().get('USE_MIDDLE_SLAB_PRIOR_CHANNEL', False))
        self.middle_slab_prior_mode = str(globals().get('MIDDLE_SLAB_PRIOR_MODE', 'concat')).lower().strip()

        # Validate prior mode
        if self.use_middle_slab_prior and self.middle_slab_prior_mode not in ['concat', 'controlnet']:
            raise ValueError(f"MIDDLE_SLAB_PRIOR_MODE must be 'concat' or 'controlnet', got: {self.middle_slab_prior_mode}")

        want_multi_inject = bool(globals().get('USE_MIDDLE_SLAB_PRIOR_MULTI_STAGE_INJECTION', False))
        if want_multi_inject and (not self.use_middle_slab_prior):
            print("   ⚠️  USE_MIDDLE_SLAB_PRIOR_MULTI_STAGE_INJECTION=True but USE_MIDDLE_SLAB_PRIOR_CHANNEL=False; ignoring multi-stage injection.")
        if want_multi_inject and self.middle_slab_prior_mode == 'controlnet':
            print("   ⚠️  USE_MIDDLE_SLAB_PRIOR_MULTI_STAGE_INJECTION is superseded by MIDDLE_SLAB_PRIOR_MODE='controlnet'; ignoring.")
            want_multi_inject = False
        self.use_middle_slab_prior_multi_stage_injection = bool(want_multi_inject and self.use_middle_slab_prior)
        
        emb_channels = 256  # Time/conditioning embedding width
        self.time_emb = TimeEmbedding(out_channels=emb_channels)
        
        # Prior channel ControlNet
        self.prior_controlnet = None
        if self.use_middle_slab_prior and self.middle_slab_prior_mode == 'controlnet':
            print(f"   🧭 Middle-slab prior enabled: controlnet mode (learned multi-scale injection)")
        
        # Combined embedding: time + case conditioning
        combine_in = emb_channels * 2
        self.emb_combine = nn.Linear(combine_in, emb_channels)
        
        # ResBlocks do not use bone mask FiLM (bone mask conditioning removed)
        use_bone_film = False
        
        # Initial projection
        in_channels = image_channels
        if self.use_middle_slab_prior and self.middle_slab_prior_mode == 'concat':
            in_channels = int(in_channels) + 1
            print("   🧭 Middle-slab prior channel enabled: concat mode (extra UNet input channel)")
        self.input_conv = nn.Conv3d(in_channels, base_channels, 3, padding=1)

        # Multi-stage prior injection (residual): add a learned projection of the prior at multiple scales.
        # We zero-init these projections so enabling it won't immediately destabilize training.
        if self.use_middle_slab_prior_multi_stage_injection:
            self.prior_inject_12 = nn.Conv3d(1, base_channels, kernel_size=1)
            self.prior_inject_6 = nn.Conv3d(1, base_channels * 2, kernel_size=1)
            self.prior_inject_3 = nn.Conv3d(1, base_channels * 4, kernel_size=1)
            nn.init.zeros_(self.prior_inject_12.weight)
            nn.init.zeros_(self.prior_inject_12.bias)
            nn.init.zeros_(self.prior_inject_6.weight)
            nn.init.zeros_(self.prior_inject_6.bias)
            nn.init.zeros_(self.prior_inject_3.weight)
            nn.init.zeros_(self.prior_inject_3.bias)
            print("   🧭 Middle-slab prior multi-stage injection enabled (48/24/12/6 + decoder re-injection)")
        else:
            self.prior_inject_12 = None
            self.prior_inject_6 = None
            self.prior_inject_3 = None
        
        # Encoder (2 levels: 48 -> 24 -> 12)
        # Level 1: 48³ → 24³
        self.enc1_res1 = ResBlock3D(base_channels, emb_channels, use_bone_mask_film=use_bone_film)
        self.enc1_res2 = ResBlock3D(base_channels, emb_channels, use_bone_mask_film=use_bone_film)

        # Axial attention removed (USE_AXIAL_W_ATTENTION disabled)
        self.use_axial_w_attention_prior_bias = False
        self.axial_w_attn_enc12 = nn.Identity()
        self.enc1_down = nn.Conv3d(base_channels, base_channels * 2, 3, stride=2, padding=1)  # 48->24
        
        # Level 2: 24³ → 12³
        self.enc2_res1 = ResBlock3D(base_channels * 2, emb_channels, use_bone_mask_film=use_bone_film)
        self.enc2_res2 = ResBlock3D(base_channels * 2, emb_channels, use_bone_mask_film=use_bone_film)
        self.enc2_down = nn.Conv3d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)  # 24->12
        
        # Bottleneck (12×12×12) with optional Flash Attention
        self.bottleneck_res1 = ResBlock3D(base_channels * 4, emb_channels, use_bone_mask_film=use_bone_film)
        self.bottleneck_attn = nn.Identity()
        self.bottleneck_res2 = ResBlock3D(base_channels * 4, emb_channels, use_bone_mask_film=use_bone_film)
        
        # Decoder (2 levels: 12 -> 24 -> 48)
        # Use ConvTranspose3d with output_padding=1 for exact size matching with encoder
        self.dec2_up = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=3, stride=2, padding=1, output_padding=1)  # 12->24
        self.dec2_res1 = ResBlock3D(base_channels * 4, emb_channels, use_bone_mask_film=use_bone_film)  # *4 because of skip concat
        self.dec2_res2 = ResBlock3D(base_channels * 4, emb_channels, use_bone_mask_film=use_bone_film)
        self.dec2_proj = nn.Conv3d(base_channels * 4, base_channels * 2, 1)  # project back down
        
        self.dec1_up = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1)  # 24->48
        self.dec1_res1 = ResBlock3D(base_channels * 2, emb_channels, use_bone_mask_film=use_bone_film)  # *2 because of skip concat
        self.dec1_res2 = ResBlock3D(base_channels * 2, emb_channels, use_bone_mask_film=use_bone_film)
        self.dec1_proj = nn.Conv3d(base_channels * 2, base_channels, 1)

        # Mirror axial attention at 48^3 in decoder (removed)
        self.axial_w_attn_dec12 = nn.Identity()
        
        # Output projection
        self.output_conv = nn.Sequential(
            nn.GroupNorm(min(8, base_channels), base_channels),
            nn.SiLU(),
            nn.Conv3d(base_channels, image_channels, 3, padding=1)
        )
        
        # Initialize output to zero (start with identity flow)
        nn.init.zeros_(self.output_conv[-1].weight)
        nn.init.zeros_(self.output_conv[-1].bias)
        # attention map holder for visualization
        self.last_attn_map = None
    
    def forward(
        self,
        z_t,
        t,
        case_ids=None,
        bone_mask=None,
        capture_attn: bool = False,
        z0=None,
        u=None,
    ):
        """
        Image-space forward pass.
        z_t: [B, 1, 48, 48, 48] interpolated image
        t: [B] time in [0,1]
        case_ids: unused (kept for API compatibility)
        Returns: velocity [B, 1, 48, 48, 48]
        """
        # Get embeddings
        t_emb = self.time_emb(t)  # [B, emb_channels]
        c_emb = torch.zeros_like(t_emb)  # Case ID conditioning disabled

        emb_parts = [t_emb, c_emb]

        # Combine embeddings
        emb = self.emb_combine(torch.cat(emb_parts, dim=1))  # [B, emb_channels]
        
        # Get prior channel residuals if using ControlNet mode
        prior_r12 = prior_r6 = prior_r3 = None
        if self.use_middle_slab_prior and self.middle_slab_prior_mode == 'controlnet':
            prior_mask = fm_create_middle_slab_prior_mask(
                (z_t.size(0), 1, z_t.size(2), z_t.size(3), z_t.size(4)),
                device=z_t.device,
                dtype=z_t.dtype,
            )
            prior_r12, prior_r6, prior_r3 = self.prior_controlnet(prior_mask)
        
        # Bone mask conditioning removed
        bone_mask_emb = None
        bone_r12 = bone_r6 = bone_r3 = None
        
        # Initial conv
        net_in = z_t
        prior = None
        if self.use_middle_slab_prior and self.middle_slab_prior_mode == 'concat':
            prior = fm_create_middle_slab_prior_mask(
                (z_t.size(0), 1, z_t.size(2), z_t.size(3), z_t.size(4)),
                device=z_t.device,
                dtype=z_t.dtype,
            )
            net_in = torch.cat([net_in, prior], dim=1)
        h = self.input_conv(net_in)  # [B, base_ch, 12, 12, 12]

        def _prior_to(feat: torch.Tensor) -> torch.Tensor | None:
            if prior is None:
                return None
            if tuple(prior.shape[2:]) == tuple(feat.shape[2:]):
                return prior
            return F.interpolate(prior, size=feat.shape[2:], mode='trilinear', align_corners=False)

        # Prior ControlNet injection at 12^3
        if prior_r12 is not None:
            if tuple(prior_r12.shape[2:]) != tuple(h.shape[2:]):
                prior_r12 = F.interpolate(prior_r12, size=h.shape[2:], mode='trilinear', align_corners=False)
            h = h + prior_r12
        
        # Bone mask ControlNet injection at 12^3
        if bone_r12 is not None:
            if tuple(bone_r12.shape[2:]) != tuple(h.shape[2:]):
                bone_r12 = F.interpolate(bone_r12, size=h.shape[2:], mode='trilinear', align_corners=False)
            h = h + bone_r12

        # Multi-stage prior injection at 12^3
        if self.use_middle_slab_prior_multi_stage_injection and (self.prior_inject_12 is not None):
            p12 = _prior_to(h)
            if p12 is not None:
                h = h + self.prior_inject_12(p12)

        # Optional prior feature gating.
        # IMPORTANT: applying gating here (before ResBlocks) can be largely neutralized by GroupNorm.
        # We therefore build a spatial gate and apply it INSIDE ResBlocks after norm1 and before conv1.
        spatial_gate_12 = None
        def _gate_like(feat: torch.Tensor) -> torch.Tensor | None:
            if spatial_gate_12 is None:
                return None
            if tuple(spatial_gate_12.shape[2:]) == tuple(feat.shape[2:]):
                return spatial_gate_12
            return F.interpolate(spatial_gate_12, size=feat.shape[2:], mode='trilinear', align_corners=False)
        
        # Encoder
        g12 = _gate_like(h)
        h1 = self.enc1_res1(h, emb, bone_mask_emb, spatial_gate=g12)
        h1 = self.enc1_res2(h1, emb, bone_mask_emb, spatial_gate=g12)

        # Axial W attention at 12^3 (optionally biased by slab prior)
        if isinstance(self.axial_w_attn_enc12, nn.Identity):
            pass
        else:
            p_bias = prior if (self.use_axial_w_attention_prior_bias and prior is not None) else None
            h1 = self.axial_w_attn_enc12(h1, prior=p_bias)
        skip1 = h1  # Save for skip connection [B, base_ch, 12, 12, 12]
        h2 = self.enc1_down(h1)  # [B, base_ch*2, 7, 7, 7]

        # Prior ControlNet injection at 6^3
        if prior_r6 is not None:
            if tuple(prior_r6.shape[2:]) != tuple(h2.shape[2:]):
                prior_r6 = F.interpolate(prior_r6, size=h2.shape[2:], mode='trilinear', align_corners=False)
            h2 = h2 + prior_r6
        
        # Bone mask ControlNet injection at 6^3
        if bone_r6 is not None:
            if tuple(bone_r6.shape[2:]) != tuple(h2.shape[2:]):
                bone_r6 = F.interpolate(bone_r6, size=h2.shape[2:], mode='trilinear', align_corners=False)
            h2 = h2 + bone_r6

        # Multi-stage prior injection at ~6^3
        if self.use_middle_slab_prior_multi_stage_injection and (self.prior_inject_6 is not None):
            p6 = _prior_to(h2)
            if p6 is not None:
                h2 = h2 + self.prior_inject_6(p6)

        g7 = _gate_like(h2)
        h2 = self.enc2_res1(h2, emb, bone_mask_emb, spatial_gate=g7)
        h2 = self.enc2_res2(h2, emb, bone_mask_emb, spatial_gate=g7)
        skip2 = h2  # Save for skip connection [B, base_ch*2, 7, 7, 7]
        h3 = self.enc2_down(h2)  # [B, base_ch*4, 4, 4, 4]

        # Prior ControlNet injection at 3^3
        if prior_r3 is not None:
            if tuple(prior_r3.shape[2:]) != tuple(h3.shape[2:]):
                prior_r3 = F.interpolate(prior_r3, size=h3.shape[2:], mode='trilinear', align_corners=False)
            h3 = h3 + prior_r3
        
        # Bone mask ControlNet injection at 3^3
        if bone_r3 is not None:
            if tuple(bone_r3.shape[2:]) != tuple(h3.shape[2:]):
                bone_r3 = F.interpolate(bone_r3, size=h3.shape[2:], mode='trilinear', align_corners=False)
            h3 = h3 + bone_r3

        # Multi-stage prior injection at ~3^3
        if self.use_middle_slab_prior_multi_stage_injection and (self.prior_inject_3 is not None):
            p3 = _prior_to(h3)
            if p3 is not None:
                h3 = h3 + self.prior_inject_3(p3)
        
        # Bottleneck with optional attention
        g4 = _gate_like(h3)
        h3 = self.bottleneck_res1(h3, emb, bone_mask_emb, spatial_gate=g4)
        # capture attention map optionally
        h3 = self.bottleneck_attn(h3)
        self.last_attn_map = None
        h3 = self.bottleneck_res2(h3, emb, bone_mask_emb, spatial_gate=g4)
        
        # Decoder with skip connections
        h2_up = self.dec2_up(h3)  # [B, base_ch*2, 6, 6, 6]
        h2_up = torch.cat([h2_up, skip2], dim=1)  # [B, base_ch*4, 6, 6, 6]
        g6 = _gate_like(h2_up)
        h2_up = self.dec2_res1(h2_up, emb, bone_mask_emb, spatial_gate=g6)
        h2_up = self.dec2_res2(h2_up, emb, bone_mask_emb, spatial_gate=g6)
        h2_up = self.dec2_proj(h2_up)  # [B, base_ch*2, 6, 6, 6]

        # Multi-stage prior re-injection in decoder at ~6^3
        if self.use_middle_slab_prior_multi_stage_injection and (self.prior_inject_6 is not None):
            p6d = _prior_to(h2_up)
            if p6d is not None:
                h2_up = h2_up + self.prior_inject_6(p6d)
        
        h1_up = self.dec1_up(h2_up)  # [B, base_ch, 12, 12, 12]
        h1_up = torch.cat([h1_up, skip1], dim=1)  # [B, base_ch*2, 12, 12, 12]
        g12b = _gate_like(h1_up)
        h1_up = self.dec1_res1(h1_up, emb, bone_mask_emb, spatial_gate=g12b)
        h1_up = self.dec1_res2(h1_up, emb, bone_mask_emb, spatial_gate=g12b)
        h1_up = self.dec1_proj(h1_up)  # [B, base_ch, 12, 12, 12]

        # Multi-stage prior re-injection in decoder at 12^3
        if self.use_middle_slab_prior_multi_stage_injection and (self.prior_inject_12 is not None):
            p12d = _prior_to(h1_up)
            if p12d is not None:
                h1_up = h1_up + self.prior_inject_12(p12d)

        # Mirror axial W attention at 12^3 in decoder
        if isinstance(self.axial_w_attn_dec12, nn.Identity):
            pass
        else:
            p_bias = prior if (self.use_axial_w_attention_prior_bias and prior is not None) else None
            h1_up = self.axial_w_attn_dec12(h1_up, prior=p_bias)
        
        # Output
        v = self.output_conv(h1_up)  # [B, image_channels, D, H, W]
        return v


def LYAPUNOV_velocity_from_valuenet(valuenet: nn.Module, z: torch.Tensor, t: torch.Tensor,
                               case_ids=None, bone_mask=None,
                               R_inv: float = None, capture_attn: bool = False,
) -> torch.Tensor:
    """
    Compute velocity from flow model or value network.
    
    This function handles:
    1. UNetFlowNetwork: outputs velocity directly [B, C, D, H, W]
    2. UNetFlowNetwork (legacy): outputs scalar V, velocity = -∇V * R_inv
    
    Detection: If model output has same shape as input z, it's velocity.
               If model output is scalar [B], it's value (legacy).
    
    Args:
        valuenet: UNetFlowNetwork or UNetFlowNetwork instance
        x: Current state [B, C, D, H, W]
        t: Time [B]
        case_ids: Case IDs for conditioning
        bone_mask: Optional bone mask for conditioning
        R_inv: Control gain (only used for legacy ValueNet)
        capture_attn: Whether to capture attention (passed to forward)
    Returns:
        velocity: [B, C, D, H, W] detached velocity field
    """
    # UNetFlowNetwork outputs velocity directly
    # Call the model and check output shape to determine mode
    with torch.no_grad():
        output = valuenet(
            z, t, case_ids,
            bone_mask=bone_mask,
        )
    
    # If output has same shape as z, it's direct velocity prediction
    if output.shape == z.shape:
        return output.detach()
    
    # Legacy: UNetFlowNetwork outputs scalar V, compute velocity as -∇V * R_inv
    if R_inv is None:
        R_inv = float(LYAPUNOV_R_INV)
    
    with torch.enable_grad():
        z_req = z.detach().requires_grad_(True)
        V = valuenet(
            z_req, t, case_ids,
            bone_mask=bone_mask,
            capture_attn=capture_attn,
        )
        grad_V = torch.autograd.grad(V.sum(), z_req, create_graph=False, retain_graph=False)[0]
    velocity = -grad_V * R_inv
    return velocity.detach()


# ----------------------- Dataset Splitting -------------------
def split_dataset(full_dataset, train_split=TRAIN_SPLIT, random_seed=RANDOM_SEED, split_by_patient=SPLIT_BY_PATIENT,
                  use_semi_online=USE_SEMI_ONLINE_AUG, max_aug_id=NUM_AUG_PER_ROI):
    """
    Split dataset BY PATIENT or BY ROI to prevent data leakage from augmentations.
    
    Strategy:
    - If split_by_patient=True: Split by patient ID (all ROIs of same patient stay together)
    - If split_by_patient=False: Split by individual ROI (each ROI is independent)
    - Train set: Uses augmentations based on use_semi_online flag
      - If use_semi_online=True: ALL augmentations available (sampler picks randomly each epoch)
      - If use_semi_online=False: Only aug0 through max_aug_id (deterministic subset)
    - Test set: ONLY original (aug0) - no augmentations
    
    Args:
        full_dataset: Complete ROI3DDataset
        train_split: Fraction of units for training (0.85 = 85%)
        random_seed: Random seed for reproducibility
        split_by_patient: True=split by patient ID, False=split by ROI independently
        use_semi_online: If True, allow all augmentations (sampler controls usage); if False, filter to max_aug_id
        max_aug_id: Maximum augmentation ID to include when use_semi_online=False
    
    Returns:
        train_dataset, test_dataset (as Subset objects)
    """
    from torch.utils.data import Subset
    import numpy as np
    
    if split_by_patient:
        # Extract unique patient IDs (case_id)
        all_units = sorted(set(p["case_id"] for p in full_dataset.pairs))
        unit_name = "patient"
        unit_key = "case_id"
        # Create unit identifier function
        get_unit_id = lambda p: p["case_id"]
    else:
        # Extract unique ROIs (case_id + roi_num combination)
        all_units = sorted(set((p["case_id"], p["roi_num"]) for p in full_dataset.pairs))
        unit_name = "ROI"
        unit_key = "(case_id, roi_num)"
        # Create unit identifier function
        get_unit_id = lambda p: (p["case_id"], p["roi_num"])
    
    n_units = len(all_units)
    
    # Split units into train/test using RandomState for exact reproducibility
    # IMPORTANT: Use RandomState.shuffle() for deterministic split
    rng = np.random.RandomState(random_seed)
    all_units_list = list(all_units)
    rng.shuffle(all_units_list)
    n_train_units = int(train_split * n_units)
    train_units = set(all_units_list[:n_train_units])
    test_units = set(all_units_list[n_train_units:])
    
    mode_str = f"semi-online (all augs available)" if use_semi_online else f"fixed (aug0-aug{max_aug_id})"
    print(f"📊 {'Patient' if split_by_patient else 'ROI'}-Based Dataset Split ({mode_str}):")
    print(f"   Total unique {unit_name}s: {n_units}")
    print(f"   Train {unit_name}s ({len(train_units)}): {sorted(list(train_units)[:10])}{'...' if len(train_units) > 10 else ''}")
    print(f"   Test {unit_name}s ({len(test_units)}): {sorted(list(test_units)[:10])}{'...' if len(test_units) > 10 else ''}")
    
    # Build train indices based on augmentation mode
    train_indices = []
    for idx, pair in enumerate(full_dataset.pairs):
        if get_unit_id(pair) in train_units:
            aug_id = pair.get("aug_id", 0)
            # Filter augmentations based on mode
            if not use_semi_online and aug_id > max_aug_id:
                continue  # Skip augmentations beyond max_aug_id when not using semi-online
            train_indices.append(idx)
    
    # Build test indices: ONLY original (aug0) for test units
    test_indices = []
    for idx, pair in enumerate(full_dataset.pairs):
        if get_unit_id(pair) in test_units:
            # Only include original (aug_id == 0)
            if pair["aug_id"] == 0:
                test_indices.append(idx)
    
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # ===================== VALIDATION: NO LEAKAGE =====================
    # 1) Train/test units are disjoint
    train_units_seen = set(get_unit_id(full_dataset.pairs[i]) for i in train_indices)
    test_units_seen = set(get_unit_id(full_dataset.pairs[i]) for i in test_indices)
    intersect_units = train_units_seen.intersection(test_units_seen)
    if intersect_units:
        raise RuntimeError(f"❌ Data leakage detected: units present in both train and test: {sorted(list(intersect_units))[:5]} ...")

    # 2) Test set must contain only original samples (aug_id == 0)
    non_original_in_test = [i for i in test_indices if full_dataset.pairs[i].get("aug_id", 0) != 0]
    if non_original_in_test:
        raise RuntimeError(f"❌ Test set contains non-original augmentations at indices: {non_original_in_test[:10]} ...")

    # 3) (Soft) Check train contains all available augmentations for its units (can't know expected count,
    #    but we can report distribution and ensure at least one sample per unit)
    train_aug_counts = {}
    for idx in train_indices:
        unit_id = get_unit_id(full_dataset.pairs[idx])
        train_aug_counts[unit_id] = train_aug_counts.get(unit_id, 0) + 1
    missing_train_units = [u for u in train_units if u not in train_aug_counts]
    if missing_train_units:
        # Not fatal: warn if any selected unit has no samples (shouldn't happen)
        print(f"⚠️ Warning: {len(missing_train_units)} selected train {unit_name}(s) had no samples. First few: {missing_train_units[:5]}")

    # 4) Report a small sample of units and counts for quick human verification
    def _fmt_unit(u):
        return f"{u}" if isinstance(u, tuple) else str(u)
    sample_units = list(train_aug_counts.keys())[:5]

    print(f"   Train samples: {len(train_dataset)} (all augmentations)")
    if train_aug_counts:
        print(f"     Augmentations per {unit_name}: min={min(train_aug_counts.values())}, max={max(train_aug_counts.values())}, avg={sum(train_aug_counts.values())/len(train_aug_counts):.1f}")
        print("     Examples:")
        for u in sample_units:
            print(f"       - {unit_name} {_fmt_unit(u)} -> {train_aug_counts[u]} samples")
    print(f"   Test samples: {len(test_dataset)} (original only, no augmentations)")
    print(f"   Random seed: {random_seed}")
    print(f"   ✅ NO DATA LEAKAGE: Train and test {unit_name}s are completely separate\n")

    return train_dataset, test_dataset

# ----------------------- Flow Training -----------------------
def train_flow_matching(train_dataset, test_dataset):
    """Train flow matching with UNet."""
    print(f"\n{'='*70}")
    print(f"STEP 2: Training Flow Matching with UNet")
    print(f"{'='*70}")
    
    # Create dataloaders with optional semi-online augmentation
    if USE_SEMI_ONLINE_AUG:
        print(f"🔁 Semi-online augmentation enabled: original + {NUM_AUG_PER_ROI} random augs per ROI each epoch")
        train_sampler = SemiOnlineAugmentationSampler(train_dataset, num_random_aug_per_roi=NUM_AUG_PER_ROI, seed=RANDOM_SEED)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0)
        actual_train_samples_per_epoch = len(train_sampler)
        print(f"📊 Training: {len(train_dataset)} available samples → {actual_train_samples_per_epoch} used per epoch (~{actual_train_samples_per_epoch // BATCH_SIZE} batches)")
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        print(f"📊 Training: {len(train_dataset)} samples per epoch (~{len(train_dataset) // BATCH_SIZE} batches)")
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


    @torch.no_grad()
    def _predict_endpoint_for_metrics(
        eval_model: nn.Module,
        x0: torch.Tensor,
        case_id_t: torch.Tensor,
        *,
        bone_mask_img: torch.Tensor | None,
        v0_at_t0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict x(1) used for TRAIN/TEST Excel metrics and training visualizations.

        If USE_DIRECT_ONE_STEP_INFERENCE=True: one-step endpoint x1 = x0 + v(x0, t=0).
        Else: multi-step ODE-style integration (Euler/Heun/RK4) from t=0..1.
        """
        use_direct = bool(globals().get("USE_DIRECT_ONE_STEP_INFERENCE", True))
        if use_direct:
            if v0_at_t0 is None:
                t0 = torch.zeros(x0.size(0), device=x0.device)
                v0_at_t0 = LYAPUNOV_velocity_from_valuenet(
                    eval_model, x0, t0, case_id_t,
                    bone_mask=bone_mask_img,
                )
            return x0 + v0_at_t0

        steps = int(globals().get("EVAL_INTEGRATION_STEPS", 30))
        if steps <= 0:
            raise ValueError(f"EVAL_INTEGRATION_STEPS must be > 0 (got {steps})")
        method = str(globals().get("INTEGRATION_METHOD", "rk4")).strip().lower()

        def _vel(x_state: torch.Tensor, t_vec: torch.Tensor) -> torch.Tensor:
            return LYAPUNOV_velocity_from_valuenet(
                eval_model, x_state, t_vec, case_id_t,
                bone_mask=bone_mask_img,
            )

        x_t = x0.clone()
        dt = 1.0 / float(steps)
        batch_size = x_t.size(0)

        if method == "rk4":
            for s in range(steps):
                t0 = torch.full((batch_size,), s * dt, device=x_t.device)
                k1 = _vel(x_t, t0)
                k2 = _vel(x_t + 0.5 * dt * k1, t0 + 0.5 * dt)
                k3 = _vel(x_t + 0.5 * dt * k2, t0 + 0.5 * dt)
                k4 = _vel(x_t + dt * k3, t0 + dt)
                x_t = x_t + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        elif method == "heun":
            for s in range(steps):
                t0 = torch.full((batch_size,), s * dt, device=x_t.device)
                v0 = _vel(x_t, t0)
                x_euler = x_t + dt * v0
                t1 = torch.full((batch_size,), (s + 1) * dt, device=x_t.device)
                v1 = _vel(x_euler, t1)
                x_t = x_t + 0.5 * dt * (v0 + v1)
        elif method == "euler":
            for s in range(steps):
                t = torch.full((batch_size,), s * dt, device=x_t.device)
                v = _vel(x_t, t)
                x_t = x_t + dt * v
        else:
            raise ValueError(f"Unknown INTEGRATION_METHOD={method!r} (expected 'euler', 'heun', or 'rk4')")

        return x_t

    @torch.no_grad()
    def _compute_copy_day5_baseline_row(eval_dataset: Dataset) -> dict | None:
        """Compute direct POD5->POY1 copy baseline metrics on eval_dataset."""
        if len(eval_dataset) == 0:
            return None

        baseline_metrics_list = []
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=EVAL_NUM_WORKERS,
        )

        for batch in eval_loader:
            pred_norm = batch["pod5"].cpu().numpy()   # Direct copy baseline: pred = POD5
            target_norm = batch["poy1"].cpu().numpy() # Target: POY1
            plate_mask_batch = batch.get("plate_mask", None)

            for i in range(pred_norm.shape[0]):
                pred_hu = denorm_to_hu(pred_norm[i, 0])
                target_hu = denorm_to_hu(target_norm[i, 0])

                plate_mask_i = None
                if plate_mask_batch is not None and EXCLUDE_PLATE_FROM_METRICS:
                    plate_mask_i = plate_mask_batch[i, 0].cpu().numpy()

                sample_metrics = compute_comprehensive_metrics(
                    pred_hu, target_hu, pred_norm[i, 0], target_norm[i, 0], plate_mask=plate_mask_i
                )
                sample_metrics.update(
                    compute_comprehensive_metrics_middle_slab(
                        pred_hu, target_hu, pred_norm[i, 0], target_norm[i, 0], plate_mask=plate_mask_i
                    )
                )
                baseline_metrics_list.append(sample_metrics)

        if not baseline_metrics_list:
            return None

        baseline_row = {
            'epoch': -1,
            'loss_mode': 'copy_day5_baseline',
            'baseline_name': 'pred=pod5',
            'avg_total_loss': float('nan'),
        }
        for key in baseline_metrics_list[0].keys():
            values = [m[key] for m in baseline_metrics_list]
            baseline_row[key] = np.nanmean(values)

        return baseline_row
    print(f"📊 Test: {len(test_dataset)} samples (~{len(test_dataset) // BATCH_SIZE} batches)")
    print()

    # Choose a visualization dataset: prefer TEST, fall back to TRAIN if no test samples
    has_test = len(test_dataset) > 0
    vis_dataset = test_dataset if has_test else train_dataset
    if not has_test:
        print("⚠️ No test set available — will use TRAIN samples for visualizations during training.")
    
    # Initialize UNet flow network
    # Use full dataset to get total number of unique cases for conditioning
    full_dataset = train_dataset.dataset  # Access underlying dataset from Subset
    
    # ==================== RF + Analytical Lyapunov ====================
    # UNet directly outputs velocity v(z, t)
    # NO learned value network - we use analytical V(z,t) = (α/2)||z - z*(t)||²
    # Lyapunov regularization comes from matching v_lyapunov = dz*/dt - α(z - z*)
    
    # Initialize standard velocity UNet (outputs velocity, not scalar V)
    flow = UNetFlowNetwork(
        image_channels=IMAGE_CHANNELS,
        base_channels=UNET_BASE_CHANNELS,
        n_cases=full_dataset.n_cases,
        use_attention=UNET_USE_ATTENTION
    ).to(device)
    
    # Load SVF teacher for Lyapunov regularization
    svf_teacher = load_svf_teacher(device)

    # Optional: initialize parts of the student from the SVF teacher weights.
    maybe_initialize_student_from_svf_teacher(
        flow,
        svf_teacher,
        verbose=bool(globals().get("INIT_STUDENT_FROM_SVF_TEACHER_VERBOSE", True)),
    )

    # We always compute a single teacher-alignment cosine metric during evaluation.
    # That requires the teacher checkpoint to be available in ALL modes (fm_only, lqr_only, both).
    if svf_teacher is None:
        raise RuntimeError(
            "SVF teacher checkpoint is required to compute the teacher-alignment cosine metric. "
            "Please set LYAPUNOV_SVF_TEACHER_CHECKPOINT to a valid path."
        )

    # Hard guard: lqr_only must actually have a teacher-driven analytical Lyapunov objective
    _loss_mode_init = str(globals().get('LOSS_MODE', 'both')).strip().lower()
    _lyapunov_enabled_flag = bool(globals().get('LYAPUNOV_ENABLED', True))
    if _loss_mode_init == 'lqr_only':
        if (not _lyapunov_enabled_flag) or (not bool(USE_ANALYTICAL_LYAPUNOV)):
            raise RuntimeError(
                "LOSS_MODE='lqr_only' requires LYAPUNOV_ENABLED=True and USE_ANALYTICAL_LYAPUNOV=True. "
                "Otherwise there is no training signal."
            )
        if svf_teacher is None:
            raise RuntimeError(
                "LOSS_MODE='lqr_only' requires loading the SVF teacher (check LYAPUNOV_SVF_TEACHER_CHECKPOINT path)."
            )
    
    optimizer = torch.optim.AdamW(flow.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Mixed Precision Training (AMP) for memory efficiency
    # torch 2.x deprecates torch.cuda.amp.GradScaler in favor of torch.amp.GradScaler("cuda").
    try:
        from torch.amp import GradScaler as _AmpGradScaler  # type: ignore
        scaler = _AmpGradScaler('cuda' if torch.cuda.is_available() else 'cpu', enabled=USE_AMP)
    except Exception:
        scaler = GradScaler(enabled=USE_AMP)
    if USE_AMP:
        print(f"⚡ Mixed Precision Training (AMP) ENABLED - uses ~50% less GPU memory")
    
    print(f"🚀 Image-Space RF + Analytical Lyapunov")
    print(f"   UNet parameters: {sum(p.numel() for p in flow.parameters()):,}")
    if svf_teacher is not None:
        print(f"   SVF Teacher: loaded (Lyapunov enabled)")
    else:
        print(f"   SVF Teacher: not loaded (Lyapunov disabled, pure RF mode)")
    
    # EMA model for flow
    ema_flow = deepcopy(flow).to(device) if USE_EMA else None
    if ema_flow is not None:
        for p in ema_flow.parameters():
            p.requires_grad_(False)
    
    def _ema_update(ema_model, online_model, decay=EMA_DECAY):
        if ema_model is None:
            return
        with torch.no_grad():
            for p_ema, p in zip(ema_model.parameters(), online_model.parameters()):
                p_ema.data.mul_(decay).add_(p.data, alpha=(1.0 - decay))
    
    print(f"Training for {NUM_EPOCHS} epochs...")
    print(f"Batch size: {BATCH_SIZE}, Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print(f"Steps per epoch: {len(train_loader)}")
    
    # Print loss mode
    loss_mode = globals().get('LOSS_MODE', 'both')
    print(f"\n{'='*50}")
    print(f"🎯 LOSS MODE: {loss_mode.upper()}")
    if loss_mode == 'lqr_only':
        print(f"   Pure LQR tracking (teacher-only): NO FM, NO endpoint")
        print(f"   Expect: good geometry, endpoint may drift")
    elif loss_mode == 'fm_only':
        print(f"   Pure FM - NO LQR/teacher guidance")
        print(f"   Expect: Straight paths, endpoint accuracy")
    else:
        print(f"   FM + LQR - Both losses active (recommended)")
    print(f"{'='*50}\n")
    
    if bool(globals().get('FM_LOSS_DECAY_ENABLE', False)):
        s = int(globals().get('FM_LOSS_DECAY_START_EPOCH', 0))
        e = int(globals().get('FM_LOSS_DECAY_END_EPOCH', s))
        sh = str(globals().get('FM_LOSS_DECAY_SHAPE', 'linear')).strip().lower()
        print(f"   FM decay enabled: epoch {s} -> {e} ({sh})")
     
    flow.train()
    
    # Training output directories
    checkpoints_dir = FM_OUT_DIR / "checkpoints"
    recon_dir = FM_OUT_DIR / "training_reconstructions"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize metrics tracking for TRAIN and TEST separately
    all_train_metrics = []
    all_test_metrics = []
    
    # Select fixed samples for visualization from both TEST and TRAIN sets
    # Test set samples
    test_vis_count = len(test_dataset)
    num_test_vis = min(NUM_SAMPLES_TO_SAVE, test_vis_count)
    test_sample_indices = np.random.choice(test_vis_count, num_test_vis, replace=False) if test_vis_count > 0 else []
    
    # Train set samples
    train_vis_count = len(train_dataset)
    num_train_vis = min(NUM_SAMPLES_TO_SAVE, train_vis_count)
    train_sample_indices = np.random.choice(train_vis_count, num_train_vis, replace=False) if train_vis_count > 0 else []

    # If HQ export is enabled, ensure the requested case/ROI pairs are included
    # in the per-epoch visualization samples so exports are actually produced.
    # For backward compatibility (used in step-based saves)
    vis_dataset = test_dataset if test_vis_count > 0 else train_dataset
    sample_indices = test_sample_indices if len(test_sample_indices) > 0 else train_sample_indices

    # ---------------- Resume logic ----------------
    def _get_latest_checkpoint(ckpt_dir: Path):
        if not ckpt_dir.exists():
            return None
        cks = list(ckpt_dir.glob('flow_unet_epoch_*.pth')) + list(ckpt_dir.glob('flow_unet_step_*.pth'))
        if not cks:
            return None
        # sort by numeric token at end
        def _key(p: Path):
            m = re.search(r'_(\d+)\.pth$', p.name)
            return int(m.group(1)) if m else -1
        cks.sort(key=_key)
        return cks[-1]

    start_epoch = 1
    # IMPORTANT: `global_step` is used for both LR scheduling and Lyapunov warmup.
    # Initialize here, and do not reset it later (especially after resume).
    global_step = 0
    if RESUME_FROM_CHECKPOINT:
        latest = _get_latest_checkpoint(checkpoints_dir)
        if latest is not None:
            print(f"🔄 Resuming from latest checkpoint: {latest}")
            ckpt = torch.load(latest, map_location=device, weights_only=False)
            # Check for architecture mismatch before loading
            try:
                flow.load_state_dict(ckpt['model_state_dict'])
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except RuntimeError as e:
                print(f"⚠️ Failed to load checkpoint (likely architecture mismatch): {e}")
                print(f"   Starting fresh training from epoch 1...")
                start_epoch = 1
                global_step = 0
                # Don't try to load metrics
                latest = None
                # Don't try to load metrics
                latest = None
            
            if latest is not None:  # Only continue if checkpoint loaded successfully
                prev_epoch = int(ckpt.get('epoch', 0))
                global_step = int(ckpt.get('global_step', 0))
                if prev_epoch >= 1:
                    start_epoch = prev_epoch + 1
                    print(f"   -> Starting at epoch {start_epoch} (previous avg loss: {ckpt.get('avg_loss', float('nan')):.6f})")
                # Try loading existing Excel metrics to preserve history (fail-fast on errors)
                train_metrics_excel_path = FM_OUT_DIR / "training_metrics_TRAIN.xlsx"
                test_metrics_excel_path = FM_OUT_DIR / "training_metrics_TEST.xlsx"
                if train_metrics_excel_path.exists():
                    df_train_hist = pd.read_excel(train_metrics_excel_path, sheet_name='Epoch Metrics')
                    # Keep only numeric epoch rows (drop AVERAGE/STD_DEV)
                    df_train_hist = df_train_hist[pd.to_numeric(df_train_hist['epoch'], errors='coerce').notna()]
                    # Convert epoch to int for consistency
                    df_train_hist['epoch'] = df_train_hist['epoch'].astype(int)
                    all_train_metrics = df_train_hist.to_dict(orient='records')
                    print(f"   -> Loaded {len(all_train_metrics)} prior TRAIN metric rows from Excel")
                if test_metrics_excel_path.exists():
                    df_test_hist = pd.read_excel(test_metrics_excel_path, sheet_name='Epoch Metrics')
                    df_test_hist = df_test_hist[pd.to_numeric(df_test_hist['epoch'], errors='coerce').notna()]
                    df_test_hist['epoch'] = df_test_hist['epoch'].astype(int)
                    all_test_metrics = df_test_hist.to_dict(orient='records')
                    print(f"   -> Loaded {len(all_test_metrics)} prior TEST metric rows from Excel")
        else:
            print("ℹ️ No existing checkpoint found; starting from scratch.")
    
    # Setup LR scheduler with cosine decay and warmup.
    # NOTE: LambdaLR's `step` argument is effectively a 0-based counter of how many times
    # `scheduler.step()` has been called. Since this code calls `scheduler.step()` AFTER
    # `optimizer.step()`, using `step` directly would make the *next* step's LR factor hit 0
    # right after the first optimizer update (step=0 -> factor=0). We therefore use (step+1)
    # in the warmup and decay math, and clamp progress to avoid cosine wraparound.
    if USE_COSINE_LR:
        total_steps = len(train_loader) * (NUM_EPOCHS - start_epoch + 1)
        warmup_steps = max(1, int(WARMUP_FRACTION * total_steps))

        def lr_lambda(step: int) -> float:
            s = int(step) + 1  # 1-based step count
            if s <= warmup_steps:
                return float(s) / float(max(1, warmup_steps))
            # cosine decay from 1 to 0 over remaining steps
            denom = float(max(1, total_steps - warmup_steps))
            progress = float(s - warmup_steps) / denom
            progress = max(0.0, min(1.0, progress))
            return float(0.5 * (1.0 + math.cos(math.pi * progress)))

        # If resuming, keep scheduler aligned with the already-completed optimizer steps.
        # This matches the code path where we call `scheduler.step()` AFTER `optimizer.step()`.
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda,
            last_epoch=int(global_step) - 1,
        )

    # Add explicit TEST baseline row: direct POD5->POY1 copy (pred = POD5).
    if COMPUTE_TEST_METRICS:
        has_copy_baseline = any(int(m.get('epoch', -999999)) == -1 for m in all_test_metrics)
        if not has_copy_baseline:
            print("📏 Computing TEST baseline row (direct POD5->POY1 copy)...")
            baseline_row = _compute_copy_day5_baseline_row(test_dataset)
            if baseline_row is not None:
                all_test_metrics.append(baseline_row)
                print("   -> Added baseline row at epoch -1 (copy_day5_baseline).")
            else:
                print("   -> Skipped baseline row (empty test dataset).")
    
    # Optional: Evaluate the randomly initialized model on the TEST set at epoch 0
    if EVAL_AT_EPOCH0 and COMPUTE_TEST_METRICS and start_epoch == 1:
        print(f"\n{'='*70}")
        print("Epoch 0 (pre-training) evaluation on TEST set:")
        # NOTE: Early in training, EMA can look almost identical to init (EMA_DECAY=0.999).
        # Use the online model for eval so progress is visible.
        eval_flow = flow
        eval_flow.eval()
        loss_mode_eval = str(globals().get('LOSS_MODE', 'both')).strip().lower()
        try:
            test_metrics_list = []
            test_losses = {
                'fm': [], 'endpoint': [], 'lyapunov': [],
                'teacher_tangent_rmse': [],
                'teacher_tangent_cossim': [],
                'velocity_magnitude_ratio': [],
                'endpoint_mae_to_teacher': [],
                'img': [], 'ssim': [], 'perc': [], 'bone': []
            }
            test_eval_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=EVAL_NUM_WORKERS)
            with torch.no_grad(), autocast(enabled=USE_AMP):
                for batch in test_eval_loader:
                    pod5_t = batch["pod5"].to(device)
                    poy1_t = batch["poy1"].to(device)
                    batch_size = pod5_t.size(0)
                    _case_id = batch["case_id"]
                    case_id_t = _case_id.to(device) if torch.is_tensor(_case_id) else torch.as_tensor(_case_id, device=device)
                    case_id_t = case_id_t.long()

                    # Evaluation/inference policy:
                    # - POY1-derived masks are forbidden
                    # - POD5-derived masks are allowed
                    bone_mask_t = None
                    bone_mask_img = None
                    x0 = pod5_t
                    x1_gt = poy1_t

                    # Predict velocity and endpoint at t=0 (Lyapunov: velocity from gradient)
                    t0 = torch.zeros(pod5_t.size(0), device=device)
                    v0 = LYAPUNOV_velocity_from_valuenet(
                        eval_flow, x0, t0, case_id_t,
                        bone_mask=bone_mask_img,
                    )
                    
                    # Get plate mask for evaluation (not for zeroing v0, but for loss exclusion)
                    plate_mask_eval = batch.get("plate_mask")
                    plate_mask_t = None
                    if plate_mask_eval is not None and EXCLUDE_PLATE_FROM_LOSS:
                        plate_mask_t = plate_mask_eval.to(device)
                    
                    x1_pred = _predict_endpoint_for_metrics(
                        eval_flow,
                        x0,
                        case_id_t,
                        bone_mask_img=bone_mask_img,
                        v0_at_t0=v0,
                    )

                    # For comparable diagnostics across modes/epochs, compute velocity at random t
                    t_rand = torch.rand(pod5_t.size(0), device=device)
                    x_t = (1 - t_rand[:, None, None, None, None]) * x0 + t_rand[:, None, None, None, None] * x1_gt

                    # Get plate mask for evaluation (not for zeroing v0, but for loss exclusion)
                    plate_mask_eval = batch.get("plate_mask")
                    plate_mask_t = None
                    if plate_mask_eval is not None and EXCLUDE_PLATE_FROM_LOSS:
                        plate_mask_t = plate_mask_eval.to(device)
                    
                    # Losses (with plate exclusion if enabled)
                    u_target = x1_gt - x0
                    if plate_mask_t is not None:
                        # Completely exclude plate from loss - model doesn't learn anything about plate regions
                        plate_exclusion = (1.0 - plate_mask_t)
                        fm_loss_val = ((v0 - u_target) ** 2 * plate_exclusion).flatten(1).sum(dim=1) / plate_exclusion.flatten(1).sum(dim=1).clamp_min(1e-6)
                    else:
                        fm_loss_val = F.mse_loss(v0, u_target, reduction='none').flatten(1).mean(dim=1)
                    test_losses['fm'].extend(fm_loss_val.tolist())
                    
                    # Endpoint loss: only if enabled (with plate exclusion)
                    # Single comparable distillation metric (all modes / all LYAPUNOV_ALPHA):
                    # TeacherTangentRMSE = E_t[ RMSE( v_student(x*(t), t), dx*(t)/dt ) ]
                    x_star, dx_star_dt = svf_teacher.get_teacher_state_and_tangent(
                        pod5_t, poy1_t, t_rand, dt=LYAPUNOV_DT_MIN
                    )
                    v_star = LYAPUNOV_velocity_from_valuenet(
                        eval_flow, x_star, t_rand, case_id_t,
                        bone_mask=bone_mask_img,
                    )
                    mse_per_sample = ((v_star - dx_star_dt) ** 2).flatten(1).mean(dim=1)
                    rmse_per_sample = torch.sqrt(mse_per_sample.clamp(min=0.0) + 1e-12)
                    test_losses['teacher_tangent_rmse'].extend(rmse_per_sample.detach().cpu().tolist())

                    # Additional simple, comparable teacher-alignment metrics (all modes):
                    # 1) TangentCosSim between v_student(x*(t),t) and dx*/dt (higher is better)
                    cos_per_sample = F.cosine_similarity(
                        v_star.flatten(1),
                        dx_star_dt.flatten(1),
                        dim=1,
                        eps=1e-8,
                    )
                    test_losses['teacher_tangent_cossim'].extend(cos_per_sample.detach().cpu().tolist())

                    # 2) VelocityScaleAgreement01: min(r,1/r) in [0,1], r=mean(|v_student|)/mean(|dx*/dt|)
                    vel_mag_ratio_per_sample = velocity_scale_agreement_01(v_star, dx_star_dt)
                    test_losses['velocity_magnitude_ratio'].extend(vel_mag_ratio_per_sample.detach().cpu().tolist())

                    # 3) EndMAE_toTeacher: compare one-step endpoint x1_pred to teacher endpoint x*(1)
                    t1 = torch.ones(pod5_t.size(0), device=device)
                    x_star_1 = svf_teacher.get_warped_at_t(pod5_t, poy1_t, t1)
                    end_mae_per_sample = (x1_pred - x_star_1).abs().flatten(1).mean(dim=1)
                    test_losses['endpoint_mae_to_teacher'].extend(end_mae_per_sample.detach().cpu().tolist())

                    # Optional: Lyapunov objective only when analytical Lyapunov is enabled
                    if bool(USE_ANALYTICAL_LYAPUNOV) and (loss_mode_eval != 'fm_only'):
                        # Step-based warmup fraction for Lyapunov
                        if LYAPUNOV_WARMUP_EPOCHS <= 0:
                            warmup_frac_eval = 1.0
                        else:
                            warmup_steps_eval = max(1, int(LYAPUNOV_WARMUP_EPOCHS * len(train_loader)))
                            warmup_frac_eval = min(1.0, float(global_step + 1) / float(warmup_steps_eval))

                        lyapunov_loss_batch, _LYAPUNOV_info = compute_LYAPUNOV_analytical_loss(
                            v_pred=v_t,
                            z_t=x_t,       # x_t (image space)
                            z_star=x_star,  # x_star (image space)
                            dz_star_dt=dx_star_dt,  # image space
                            t=t_rand,
                            warmup_frac=warmup_frac_eval,
                        )
                        LYAPUNOV_scalar = float(lyapunov_loss_batch.item())
                        test_losses['lyapunov'].extend([LYAPUNOV_scalar] * batch_size)

                    # x1_pred IS the prediction (no decode needed)
                    poy1_pred = x1_pred

                    if USE_IMAGE_SPACE_LOSS:
                        img_loss_val = F.mse_loss(poy1_pred, poy1_t)
                        img_loss_scalar = img_loss_val.item()

                        test_losses['img'].extend([img_loss_scalar] * batch_size)

                        if USE_IMG_LOSS_BONE_WEIGHTED:
                            w = _bone_weight_map(poy1_t, hu_threshold=METRICS_BONE_HU_THRESHOLD,
                                                 alpha=BONE_WEIGHT_ALPHA, surface_weight=BONE_SURFACE_WEIGHT)
                            bone_mse = ((poy1_pred - poy1_t) ** 2) * w
                            bone_loss_vals = bone_mse.flatten(1).mean(dim=1)
                            test_losses['bone'].extend(bone_loss_vals.tolist())

                    # Metrics
                    pred_norm = poy1_pred.cpu().numpy()
                    target_norm = poy1_t.cpu().numpy()
                    # Get plate mask for metrics if enabled
                    plate_mask_batch = batch.get("plate_mask", None)
                    for i in range(pred_norm.shape[0]):
                        pred_hu = denorm_to_hu(pred_norm[i, 0])
                        target_hu = denorm_to_hu(target_norm[i, 0])
                        # Extract plate mask for this sample if available
                        plate_mask_i = None
                        if plate_mask_batch is not None and EXCLUDE_PLATE_FROM_METRICS:
                            plate_mask_i = plate_mask_batch[i, 0].cpu().numpy()
                        sample_metrics = compute_comprehensive_metrics(pred_hu, target_hu, pred_norm[i, 0], target_norm[i, 0], plate_mask=plate_mask_i)
                        sample_metrics.update(compute_comprehensive_metrics_middle_slab(pred_hu, target_hu, pred_norm[i, 0], target_norm[i, 0], plate_mask=plate_mask_i))
                        test_metrics_list.append(sample_metrics)

            # Aggregate epoch-0 metrics
            avg_fm = float(np.mean(test_losses['fm'])) if len(test_losses['fm']) > 0 else float('nan')
            avg_ep = float(np.mean(test_losses['endpoint'])) if len(test_losses['endpoint']) > 0 else float('nan')
            avg_lyapunov = float(np.mean(test_losses['lyapunov'])) if len(test_losses['lyapunov']) > 0 else float('nan')
            avg_tan_rmse = float(np.mean(test_losses['teacher_tangent_rmse'])) if len(test_losses.get('teacher_tangent_rmse', [])) > 0 else float('nan')
            avg_tan_cos = float(np.mean(test_losses['teacher_tangent_cossim'])) if len(test_losses.get('teacher_tangent_cossim', [])) > 0 else float('nan')
            avg_vel_mag_ratio = float(np.mean(test_losses['velocity_magnitude_ratio'])) if len(test_losses.get('velocity_magnitude_ratio', [])) > 0 else float('nan')
            avg_end_mae_teacher = float(np.mean(test_losses['endpoint_mae_to_teacher'])) if len(test_losses.get('endpoint_mae_to_teacher', [])) > 0 else float('nan')

            avg_img = float(np.mean(test_losses['img'])) if (USE_IMAGE_SPACE_LOSS and len(test_losses['img']) > 0) else float('nan')
            avg_bone = float(np.mean(test_losses['bone'])) if (USE_IMAGE_SPACE_LOSS and USE_IMG_LOSS_BONE_WEIGHTED and len(test_losses['bone']) > 0) else float('nan')

            avg_total = 0.0
            if loss_mode_eval == 'lqr_only':
                avg_total = avg_lyapunov if not math.isnan(avg_lyapunov) else 0.0
            elif loss_mode_eval == 'fm_only':
                avg_total = avg_fm if not math.isnan(avg_fm) else 0.0
            else:
                avg_total = (avg_fm if not math.isnan(avg_fm) else 0.0) + (avg_lyapunov if not math.isnan(avg_lyapunov) else 0.0)
            
            # Add endpoint (mode-independent, only if weight > 0)
            if USE_IMAGE_SPACE_LOSS:
                if not math.isnan(avg_img):
                    avg_total += IMAGE_SPACE_LOSS_WEIGHT * avg_img
                if USE_IMG_LOSS_BONE_WEIGHTED and not math.isnan(avg_bone):
                    avg_total += BONE_LOSS_LAMBDA * avg_bone

            test_epoch0_metrics = {
                'epoch': 0,
                'loss_mode': loss_mode_eval,
                'avg_total_loss': avg_total,
                'avg_rmse_v_teacher_tangent': avg_tan_rmse,
                'avg_cos_v_teacher_tangent': avg_tan_cos,
                'avg_velocity_magnitude_ratio': avg_vel_mag_ratio,
                'avg_endpoint_mae_to_teacher': avg_end_mae_teacher,
            }
            # Only show active losses
            if loss_mode_eval != 'fm_only' and not math.isnan(avg_lyapunov):
                test_epoch0_metrics['avg_lyapunov_loss'] = avg_lyapunov
            if loss_mode_eval != 'lqr_only' and not math.isnan(avg_fm):
                test_epoch0_metrics['avg_fm_loss'] = avg_fm
            if USE_IMAGE_SPACE_LOSS:
                if not math.isnan(avg_img):
                    test_epoch0_metrics['avg_img_loss'] = avg_img
                if USE_IMG_LOSS_BONE_WEIGHTED and not math.isnan(avg_bone):
                    test_epoch0_metrics['avg_bone_loss'] = avg_bone

            if test_metrics_list:
                for key in test_metrics_list[0].keys():
                    values = [m[key] for m in test_metrics_list]
                    test_epoch0_metrics[key] = np.nanmean(values)  # Use nanmean to handle undefined metrics

            # De-duplicate and append epoch 0 metrics
            all_test_metrics = [m for m in all_test_metrics if int(m.get('epoch', -1)) != 0]
            all_test_metrics.append(test_epoch0_metrics)

            # Write Excel row for epoch 0
            test_metrics_excel_path = FM_OUT_DIR / "training_metrics_TEST.xlsx"
            create_metrics_excel_with_footnotes(all_test_metrics, test_metrics_excel_path)
            print("  📊 Updated TEST metrics Excel (epoch 0)")

            # Print brief summary
            if 'MAE_all_HU' in test_epoch0_metrics:
                print(f"  Test Loss (epoch 0): {test_epoch0_metrics['avg_total_loss']:.6f}")
                print(f"  MAE_all_HU: {test_epoch0_metrics.get('MAE_all_HU', float('nan')):.2f} | MAE_bone_HU: {test_epoch0_metrics.get('MAE_bone_HU', float('nan')):.2f}")
                print(f"  MS_SSIM: {test_epoch0_metrics.get('MS_SSIM', float('nan')):.4f} | MS_SSIM_bone: {test_epoch0_metrics.get('MS_SSIM_bone', float('nan')):.4f}")
                print(f"  Dice_bone: {test_epoch0_metrics.get('Dice_bone', float('nan')):.4f}")
                if 'MAE_all_HU_mid' in test_epoch0_metrics:
                    print(f"  Mid-slab MAE_all_HU_mid: {test_epoch0_metrics.get('MAE_all_HU_mid', float('nan')):.2f} | Dice_bone_mid: {test_epoch0_metrics.get('Dice_bone_mid', float('nan')):.4f}")
        finally:
            eval_flow.train()
        print(f"{'='*70}\n")
    
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        # CRITICAL: Ensure model is in training mode
        flow.train()
        
        epoch_loss = 0.0
        epoch_fm_loss = 0.0
        epoch_endpoint_loss = 0.0
        epoch_img_loss = 0.0
        epoch_bone_loss = 0.0
        epoch_bone_batches = 0
        num_batches = 0
        num_img_loss_batches = 0  # Track how many batches computed image losses
        
        # Track most recent batch losses for logging (persist across batches)
        last_img_loss = None
        last_bone_loss = None
        last_bone_loss_weighted = None
        
        for batch_idx, batch in enumerate(train_loader):
            pod5 = batch["pod5"].to(device)  # [B, 1, 48, 48, 48]
            poy1 = batch["poy1"].to(device)
            case_ids = batch["case_id"].to(device)

            # Get bone mask (POY1 bone structure via HU threshold) if enabled
            bone_mask = None
            # Get plate mask for loss exclusion if enabled
            plate_mask = None
            if EXCLUDE_PLATE_FROM_LOSS:
                plate_mask = batch["plate_mask"].to(device)  # [B, 1, 48, 48, 48]
            
            # pod5 and poy1 are already in normalized [-1, 1] range
            x0 = pod5  # [B, 1, 48, 48, 48]
            x1 = poy1  # [B, 1, 48, 48, 48]
            
            bone_mask_img = bone_mask  # [B, 1, 48, 48, 48] or None
            
            # Sample random time t ~ Uniform[0, 1]
            t = torch.rand(pod5.size(0), device=device)
            
            # Interpolate in image space: x_t = (1-t)*x0 + t*x1
            x_t = (1 - t[:, None, None, None, None]) * x0 + t[:, None, None, None, None] * x1
            
            # Target velocity: u_t = x1 - x0 (constant for straight paths)
            u_target = x1 - x0
            # NOTE: Plate regions are excluded from loss via masking (not by zeroing u_target)
            # This means the model doesn't learn anything about plate regions at all
            
            # ==================== RF + Analytical Lyapunov (IMAGE SPACE) ====================
            # flow outputs velocity directly in image space
            # Lyapunov regularization uses analytical value function
            
            # Check loss mode early (used to decide what state to feed the model)
            loss_mode = str(globals().get('LOSS_MODE', 'both')).strip().lower()

            # Optionally compute Lyapunov on-policy at student rollout state x_theta(t).
            # We tie the on-policy probability to the same warmup schedule as the Lyapunov weight.
            #
            # IMPORTANT:
            # - Lyapunov/LQR may be computed on on-policy rollout states x_theta(t)
            # - FM/RF loss should stay on the straight interpolation bridge x_t=(1-t)x0+t*x1
            x_t_for_lyapunov = x_t
            LYAPUNOV_used_on_policy_state = False
            use_lyapunov_available = (loss_mode != 'fm_only') and (svf_teacher is not None) and USE_ANALYTICAL_LYAPUNOV
            if use_lyapunov_available and bool(globals().get('LYAPUNOV_ON_POLICY_TRAINING', False)):
                if LYAPUNOV_WARMUP_EPOCHS <= 0:
                    warmup_frac_for_policy = 1.0
                else:
                    warmup_steps_train = max(1, int(LYAPUNOV_WARMUP_EPOCHS * len(train_loader)))
                    warmup_frac_for_policy = min(1.0, float(global_step + 1) / float(warmup_steps_train))

                p_max = float(globals().get('LYAPUNOV_ON_POLICY_PROB', 0.0))
                # Ramp: gradually introduce on-policy states as Lyapunov warms up
                p_on = max(0.0, min(1.0, p_max * warmup_frac_for_policy))
                if (p_on >= 1.0) or (torch.rand((), device=device).item() < p_on):
                    steps_on = int(globals().get('LYAPUNOV_ON_POLICY_STEPS', 8))
                    x_t_for_lyapunov = rollout_student_to_time_euler(
                        flow=flow,
                        x0=x0,
                        t_target=t,
                        case_ids=case_ids,
                        bone_mask_img=bone_mask_img,
                        steps=steps_on,
                    )
                    LYAPUNOV_used_on_policy_state = True

            # Use AMP autocast for memory-efficient forward pass
            with autocast(enabled=USE_AMP):
                v_pred_fm = None
                v_pred_lyapunov = None

                # Predict velocity for FM/RF objective (always on straight interpolation x_t)
                if loss_mode != 'lqr_only':
                    v_pred_fm = flow(
                        x_t,
                        t,
                        case_ids,
                        bone_mask=bone_mask_img,
                    )

                # Predict velocity for Lyapunov objective (straight interpolation or on-policy rollout)
                if use_lyapunov_available:
                    if LYAPUNOV_used_on_policy_state or (v_pred_fm is None):
                        v_pred_lyapunov = flow(
                            x_t_for_lyapunov,
                            t,
                            case_ids,
                            bone_mask=bone_mask_img,
                        )
                    else:
                        v_pred_lyapunov = v_pred_fm

                # Keep a backward-compatible alias for downstream logging/debug.
                v_pred = v_pred_fm if (v_pred_fm is not None) else v_pred_lyapunov
                
                # Note: For STATIC plate behavior, we zero u_target in plate regions (above)
                # so the model learns to predict zero velocity there. No need to modify v_pred here.
                
                # Flow matching loss
                # Skip FM loss computation if in lqr_only mode
                if loss_mode == 'lqr_only':
                    fm_loss = torch.tensor(0.0, device=device)  # Dummy for code compatibility
                else:
                    # MSE loss with optional plate exclusion
                    err2 = (v_pred_fm - u_target) ** 2
                    if plate_mask is not None and EXCLUDE_PLATE_FROM_LOSS:
                        plate_exclusion = 1.0 - plate_mask
                        fm_loss = (err2 * plate_exclusion).sum() / plate_exclusion.sum().clamp_min(1e-6)
                    else:
                        fm_loss = F.mse_loss(v_pred_fm, u_target)
                
                # Endpoint loss: penalize error at t=0
                # Direct velocity prediction at t=0 in IMAGE SPACE
                t0_batch = torch.zeros(pod5.size(0), device=device)
                v0 = flow(
                    x0,
                    t0_batch,
                    case_ids,
                    bone_mask=bone_mask_img,
                )
                
                # Note: v0 is NOT zeroed in plate regions. The model should learn to predict
                # zero velocity there via the FM loss (where u_target is zeroed in plate regions).
                # Endpoint loss excludes plate regions via plate_exclusion weighting below.
                    
                x0_pushed = x0 + v0  # x0 + v(x0, 0) should equal x1
                
                # Simple endpoint loss with optional plate exclusion
                x_err2 = (x0_pushed - x1) ** 2
                if plate_mask is not None and EXCLUDE_PLATE_FROM_LOSS:
                    plate_exclusion = 1.0 - plate_mask
                    endpoint_loss = (x_err2 * plate_exclusion).sum() / plate_exclusion.sum().clamp_min(1e-6)
                else:
                    endpoint_loss = F.mse_loss(x0_pushed, x1)
                
                # ==================== LOSS MODE SELECTION ====================
            # loss_mode already set above to skip FM computation
            
            fm_w = _fm_loss_weight_factor(epoch)
            if loss_mode == 'lqr_only':
                # LQR-only: NO FM loss; Lyapunov/LQR is added below
                loss = torch.tensor(0.0, device=device)
            elif loss_mode == 'fm_only':
                # Pure FM: standard FM loss
                loss = fm_w * fm_loss
            else:  # 'both' (default)
                loss = fm_w * fm_loss
            
            # Endpoint loss: independent of LOSS_MODE (added if weight > 0)
            # Optional image-space losses (MSE/SSIM/Perceptual/Bone)
            img_ramp = 1.0
            img_w = IMAGE_SPACE_LOSS_WEIGHT
            bone_w = BONE_LOSS_LAMBDA * img_ramp

            step_matches_img = ((batch_idx + 1) % IMAGE_SPACE_LOSS_FREQ == 0)
            want_mse = USE_IMAGE_SPACE_LOSS
            want_bone = USE_IMAGE_SPACE_LOSS and USE_IMG_LOSS_BONE_WEIGHTED
            # If ramp is still 0, keep training as pure FM (skip expensive decode)
            do_img_losses = (img_ramp > 0.0) and step_matches_img and (want_mse or want_bone)

            img_loss = None; bone_loss_term = None
            if do_img_losses:
                recon_pred = x0_pushed

                # NOTE: We now use proper normalized weighted losses instead of pre-multiplying inputs
                if FM_RESECTION_PLANE_CONSTRAINT:
                    plane_mask = fm_create_resection_plane_mask(
                        recon_pred.shape, sigma=FM_RESECTION_PLANE_SIGMA, device=recon_pred.device
                    )
                else:
                    plane_mask = None
                
                use_norm_weight = bool(globals().get('USE_NORMALIZED_WEIGHTED_LOSS', False))
                
                # Compute enabled image-space terms with proper weighting
                if want_mse:
                    mse_err2 = (recon_pred - poy1) ** 2
                    # Also add gradient loss component
                    gx_pred = torch.diff(recon_pred, dim=2); gy_pred = torch.diff(recon_pred, dim=3); gz_pred = torch.diff(recon_pred, dim=4)
                    gx_tgt = torch.diff(poy1, dim=2); gy_tgt = torch.diff(poy1, dim=3); gz_tgt = torch.diff(poy1, dim=4)
                    grad_err = (torch.abs(gx_pred - gx_tgt).mean() + torch.abs(gy_pred - gy_tgt).mean() + torch.abs(gz_pred - gz_tgt).mean()) / 3.0
                    if plane_mask is not None and use_norm_weight:
                        img_loss = (mse_err2 * plane_mask).sum() / plane_mask.sum().clamp_min(1e-6) + 0.3 * grad_err
                    elif plane_mask is not None:
                        img_loss = (mse_err2 * plane_mask).mean() + 0.3 * grad_err
                    else:
                        img_loss = mse_err2.mean() + 0.3 * grad_err
                    last_img_loss = img_loss.item()
                    
                # Bone-weighted loss with normalized weighting
                if want_bone:
                    w = _bone_weight_map(poy1, hu_threshold=METRICS_BONE_HU_THRESHOLD,
                                         alpha=BONE_WEIGHT_ALPHA, surface_weight=BONE_SURFACE_WEIGHT)
                    if plane_mask is not None:
                        w = w * plane_mask
                    bone_mse = ((recon_pred - poy1) ** 2) * w
                    if use_norm_weight:
                        bone_loss_term = bone_mse.sum() / w.sum().clamp_min(1e-6)
                    else:
                        bone_loss_term = bone_mse.mean()
                    last_bone_loss = bone_loss_term.item()
                    last_bone_loss_weighted = (BONE_LOSS_LAMBDA * bone_loss_term).item()

                # Add enabled terms to total loss with their weights
                if img_loss is not None:
                    loss = loss + img_w * img_loss
                if bone_loss_term is not None:
                    loss = loss + (bone_w * bone_loss_term)
            
            # ==================== Analytical Lyapunov/LQR Loss ====================
            # Uses analytical value function V(z,t) = (α/2)||z - z*(t)||²
            # Optimal velocity: v_lyapunov = dz*/dt - α(z - z*(t))
            # Teacher provides ground-truth z*(t) and dz*/dt
            lyapunov_loss_val = 0.0
            lyapunov_info = {'raw_lyapunov_loss': 0.0, 'weight': 0.0, 'lambda': 0.0}

            # Skip Lyapunov for 'fm_only' mode
            use_lyapunov_this_step = (loss_mode != 'fm_only') and (svf_teacher is not None) and USE_ANALYTICAL_LYAPUNOV

            if use_lyapunov_this_step:
                # Step-based warmup fraction for Lyapunov (smooth within epoch)
                if LYAPUNOV_WARMUP_EPOCHS <= 0:
                    warmup_frac_train = 1.0
                else:
                    warmup_steps_train = max(1, int(LYAPUNOV_WARMUP_EPOCHS * len(train_loader)))
                    warmup_frac_train = min(1.0, float(global_step + 1) / float(warmup_steps_train))

                # Get teacher's ground-truth state x*(t) and tangent dx*/dt
                x_star, dx_star_dt = svf_teacher.get_teacher_state_and_tangent(
                    pod5, poy1, t, dt=LYAPUNOV_DT_MIN
                )

                LYAPUNOV_weight_map = None

                # Compute analytical Lyapunov loss (in image space)
                lyapunov_loss, lyapunov_info = compute_LYAPUNOV_analytical_loss(
                    v_pred=v_pred_lyapunov,
                    z_t=x_t_for_lyapunov,  # Lyapunov state (straight interpolation or on-policy rollout)
                    z_star=x_star,
                    dz_star_dt=dx_star_dt,
                    t=t,
                    warmup_frac=warmup_frac_train,
                    weight_map=LYAPUNOV_weight_map,
                )

                lyapunov_loss_val = lyapunov_loss.item()
                loss = loss + lyapunov_loss
            
            # End of autocast block - backward and optimizer are outside
            # Optimize UNet (single optimizer for flow = UNetFlowNetwork)
            optimizer.zero_grad()
            # Use AMP scaler for backward pass
            scaler.scale(loss).backward()
            
            # DEBUG: Check gradient magnitudes on first batch of first epoch
            if epoch == start_epoch and batch_idx == 0:
                grad_norms = []
                for name, p in flow.named_parameters():
                    if p.grad is not None:
                        grad_norms.append((name, p.grad.abs().mean().item(), p.grad.abs().max().item()))
                print(f"\n[DEBUG] Gradient check after first backward:")
                print(f"  v_pred mean={v_pred.abs().mean().item():.6f}, max={v_pred.abs().max().item():.6f}")
                print(f"  u_target mean={u_target.abs().mean().item():.6f}")
                fm_w_dbg = float(fm_w) if loss_mode != 'lqr_only' else 0.0
                print(f"  fm_loss={fm_loss.item() if loss_mode != 'lqr_only' else 0.0:.6f} (w={fm_w_dbg:.3f}), lyapunov_loss={lyapunov_loss_val:.6f}")
                if use_lyapunov_this_step and lyapunov_info:
                    raw_loss = lyapunov_info.get('loss_LYAPUNOV_raw', lyapunov_info.get('raw_lyapunov_loss', 0.0))
                    weight = lyapunov_info.get('w_LYAPUNOV_mean', lyapunov_info.get('weight', 0.0))
                    print(f"  Lyapunov info: raw={raw_loss:.6f}, weight={weight:.4f}")
                total_grad_norm = sum(gn[1] for gn in grad_norms) / max(len(grad_norms), 1)
                params_with_grad = len([g for g in grad_norms if g[1] > 0])
                print(f"  Params with grad: {params_with_grad}/{len(grad_norms)}, avg grad norm: {total_grad_norm:.8f}")
                if grad_norms:
                    top_grads = sorted(grad_norms, key=lambda x: -x[2])[:5]
                    print(f"  Top 5 grads: {[(n, f'{m:.6f}') for n, m, _ in top_grads]}")
                print()
            
            # Use AMP scaler for gradient unscaling and optimizer step
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()
            # EMA update
            if USE_EMA:
                _ema_update(ema_flow, flow)
            
            # Accumulate losses
            epoch_loss += loss.item()
            # fm_loss may not exist in 'lqr_only' mode
            if loss_mode != 'lqr_only':
                epoch_fm_loss += float(fm_w) * fm_loss.item()
            epoch_endpoint_loss += endpoint_loss.item()
            # Accumulate image-space losses if computed this batch
            if do_img_losses:
                if img_loss is not None:
                    epoch_img_loss += img_loss.item()
                num_img_loss_batches += 1
            # Accumulate bone loss independently (only if image-space bone loss was computed)
            if want_bone and (bone_loss_term is not None):
                epoch_bone_loss += bone_loss_term.item()
                epoch_bone_batches += 1
            num_batches += 1
            
            # Step bookkeeping and optional scheduler
            global_step += 1
            if USE_COSINE_LR:
                scheduler.step()
            
            # Log progress within epoch
            if (batch_idx + 1) % 10 == 0:
                current_lr = float(optimizer.param_groups[0].get('lr', 0.0))
                fm_str = f"{fm_loss.item():.6f}" if loss_mode != 'lqr_only' else "N/A"
                log_msg = (f"  Epoch [{epoch}/{NUM_EPOCHS}] Batch [{batch_idx+1}/{len(train_loader)}] - "
                          f"Loss: {loss.item():.6f} | FM: {fm_str} | Endpoint: {endpoint_loss.item():.6f} | LR: {current_lr:.2e}")
                # Add image-space/bone losses to log if they have been computed at least once
                any_img_logged = False
                if 'last_img_loss' in locals() and last_img_loss is not None:
                    log_msg += f" | ImgLoss: {last_img_loss:.6f}"
                    any_img_logged = True
                if 'last_bone_loss' in locals() and last_bone_loss is not None:
                    log_msg += f" | BoneLoss: {last_bone_loss:.6f} (×{BONE_LOSS_LAMBDA:.1f}={last_bone_loss_weighted:.6f})"
                # Log analytical Lyapunov loss
                if lyapunov_loss_val > 0:
                    raw_lyapunov = float(lyapunov_info.get('raw_lyapunov_loss', lyapunov_info.get('loss_LYAPUNOV_raw', 0.0)))
                    lam_lyapunov = float(lyapunov_info.get('lambda', lyapunov_info.get('w_LYAPUNOV_mean', 0.0)))
                    log_msg += f" | Lyapunov: {lyapunov_loss_val:.4f} (raw:{raw_lyapunov:.4f} λ:{lam_lyapunov:.3f})"
                print(log_msg)

            # Step-based saving logic
            def _should_save(unit: str, interval: int, epoch_idx: int, step_idx: int) -> bool:
                if interval <= 0:
                    return False
                if unit == 'epoch':
                    return False  # handled after epoch
                # step-based
                return (step_idx % interval == 0)

            # Save checkpoint by step
            if _should_save(CKPT_SAVE_UNIT, CKPT_SAVE_INTERVAL, epoch, global_step):
                checkpoint_path = checkpoints_dir / f"flow_unet_step_{global_step:08d}.pth"
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': flow.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': {
                        'use_middle_slab_prior_channel': bool(globals().get('USE_MIDDLE_SLAB_PRIOR_CHANNEL', False)),
                        'use_middle_slab_prior_multi_stage_injection': bool(globals().get('USE_MIDDLE_SLAB_PRIOR_MULTI_STAGE_INJECTION', False)),
                        'middle_slab_image_slice_start': int(globals().get('MIDDLE_SLAB_IMAGE_SLICE_START', 20)),
                        'middle_slab_image_slice_end': int(globals().get('MIDDLE_SLAB_IMAGE_SLICE_END', 28)),
                        'middle_slab_profile': str(globals().get('MIDDLE_SLAB_PROFILE', 'cosine')),
                        'middle_slab_falloff': float(globals().get('MIDDLE_SLAB_FALLOFF', 5.0)),
                        'roi_shape': tuple(globals().get('ROI_SHAPE', (48, 48, 48))),
                        'LYAPUNOV_mode': 'analytical',
                        'LYAPUNOV_alpha': float(LYAPUNOV_ALPHA),
                        'LYAPUNOV_lambda_max': float(LYAPUNOV_LAMBDA_MAX),
                    },
                }, checkpoint_path)
                print(f"💾 Saved checkpoint (step): {checkpoint_path}")

            # Save reconstructions by step (single sample for speed)
            if _should_save(RECON_SAVE_UNIT, RECON_SAVE_INTERVAL, epoch, global_step):
                # Use online model for visualizations; EMA can lag heavily early on.
                flow.eval()
                try:
                    if len(sample_indices) == 0:
                        # Nothing to visualize this step
                        continue
                    sample_idx = int(sample_indices[0])
                    s = vis_dataset[sample_idx]
                    pod5_t = s["pod5"].unsqueeze(0).to(device)
                    poy1_t = s["poy1"].unsqueeze(0).to(device)
                    case_id_idx = s["case_id"]
                    case_id_t = torch.as_tensor([case_id_idx], device=device, dtype=torch.long)
                    
                    # IMPORTANT: never use POY1-derived bone masks for inference/visualization.
                    bone_mask_t = None
                    bone_mask_img = None
                    
                    x0 = pod5_t
                    x1_gt = poy1_t
                    t0 = torch.zeros(1, device=device)
                    # Lyapunov: velocity from gradient
                    eval_model = flow
                    v0 = LYAPUNOV_velocity_from_valuenet(
                        eval_model, x0, t0, case_id_t,
                        bone_mask=None,
                        capture_attn=True,
                    )
                    attn_map = None
                    if hasattr(eval_model, 'last_attn_map') and eval_model.last_attn_map is not None:
                        attn_map = eval_model.last_attn_map.squeeze().cpu().numpy()
                    x1_pred = _predict_endpoint_for_metrics(
                        eval_model,
                        x0,
                        case_id_t,
                        bone_mask_img=None,
                        v0_at_t0=v0,
                    )
                    poy1_pred = x1_pred
                    poy1_recon_gt = x1_gt
                    pod5_hu = denorm_to_hu(pod5_t.squeeze().cpu().numpy())
                    poy1_gt_hu = denorm_to_hu(poy1_t.squeeze().cpu().numpy())
                    poy1_recon_gt_hu = denorm_to_hu(poy1_recon_gt.squeeze().cpu().numpy())
                    poy1_pred_hu = denorm_to_hu(poy1_pred.squeeze().cpu().numpy())
                    pod5_original_hu = None
                    pod5_modified_hu = None
                    case_id = s["meta"]["case_id"]; roi_num = s["meta"]["roi_num"]
                    step_recon_dir = recon_dir / f"step_{global_step:08d}"
                    step_recon_dir.mkdir(exist_ok=True)
                    save_orthogonal_png(pod5_hu, str(step_recon_dir / f"Case_{case_id:03d}_ROI{roi_num:02d}_POD5.png"), f"POD5 (case {case_id}, ROI {roi_num})")
                    save_orthogonal_png(poy1_gt_hu, str(step_recon_dir / f"Case_{case_id:03d}_ROI{roi_num:02d}_GT.png"), f"GT POY1 (case {case_id}, ROI {roi_num})")
                    save_orthogonal_png(poy1_pred_hu, str(step_recon_dir / f"Case_{case_id:03d}_ROI{roi_num:02d}_pred.png"), f"Predicted POY1 (case {case_id}, ROI {roi_num})")
                    print(f"🖼️  Saved step reconstruction to: {step_recon_dir}")
                finally:
                    (ema_flow if USE_EMA else flow).train()
        
        # Compute epoch averages
        avg_loss = epoch_loss / num_batches
        avg_fm_loss = epoch_fm_loss / num_batches
        avg_endpoint_loss = epoch_endpoint_loss / num_batches
        
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{NUM_EPOCHS} Summary:")
        print(f"  Avg Total Loss: {avg_loss:.6f}")
        print(f"  Avg FM Loss: {avg_fm_loss:.6f}")
        print(f"  Avg Endpoint Loss: {avg_endpoint_loss:.6f}")
        # Print image-space / bone loss averages if computed
        if num_img_loss_batches > 0:
            if epoch_img_loss > 0:
                avg_img_loss = epoch_img_loss / num_img_loss_batches
                print(f"  Avg Img Loss: {avg_img_loss:.6f}")
        if USE_IMG_LOSS_BONE_WEIGHTED and epoch_bone_batches > 0:
            avg_bone_loss = epoch_bone_loss / epoch_bone_batches
            print(f"  Avg Bone Loss: {avg_bone_loss:.6f}")

        
        # Decide whether to run full train/test evaluation this epoch
        should_eval_epoch = (epoch == 1) or (epoch % EXCEL_UPDATE_INTERVAL == 0) or (epoch == NUM_EPOCHS)
        if should_eval_epoch:
            # Compute comprehensive metrics on TRAINING SET (use EMA if enabled)
            # Default to EMA for more stable curves (can disable via EVAL_USE_EMA=False)
            eval_flow = (ema_flow if (USE_EMA and EVAL_USE_EMA and ema_flow is not None) else flow)
            eval_flow.eval()
            loss_mode_eval = str(globals().get('LOSS_MODE', 'both')).strip().lower()
            # Lyapunov warmup is step-based (uses global_step), so no epoch_progress needed here.
            
            # ============= TRAIN SET EVALUATION =============
            if COMPUTE_TRAIN_METRICS:
                print(f"  Computing metrics on training set...")
                
                train_metrics_list = []
                train_losses = {
                    'fm': [], 'endpoint': [], 'lyapunov': [],
                    'teacher_tangent_rmse': [],
                    'teacher_tangent_cossim': [],
                    'velocity_magnitude_ratio': [],
                    'endpoint_mae_to_teacher': [],
                    'img': [], 'ssim': [], 'perc': [], 'bone': []
                }
                
                # Batched evaluation over the entire training set
                train_eval_loader = DataLoader(train_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=EVAL_NUM_WORKERS)
                with torch.no_grad(), autocast(enabled=USE_AMP):
                    for batch in train_eval_loader:
                        pod5_t = batch["pod5"].to(device)
                        poy1_t = batch["poy1"].to(device)
                        batch_size = pod5_t.size(0)
                        # Avoid copy-construct warning: if already a tensor, just move to device; else wrap with as_tensor
                        _case_id = batch["case_id"]
                        case_id_t = _case_id.to(device) if torch.is_tensor(_case_id) else torch.as_tensor(_case_id, device=device)
                        case_id_t = case_id_t.long()
                        
                        # Evaluation/inference policy:
                        # - POY1-derived masks are forbidden
                        # - POD5-derived masks are allowed
                        bone_mask_t = None
                        bone_mask_img = None
                        x0 = pod5_t
                        x1_gt = poy1_t

                        # Predict endpoint at t=0 (one-step)
                        t0 = torch.zeros(pod5_t.size(0), device=device)
                        v0 = LYAPUNOV_velocity_from_valuenet(
                            eval_flow, x0, t0, case_id_t,
                            bone_mask=bone_mask_img,
                        )
                        x1_pred = _predict_endpoint_for_metrics(
                            eval_flow,
                            x0,
                            case_id_t,
                            bone_mask_img=bone_mask_img,
                            v0_at_t0=v0,
                        )
                        
                        # Compute losses for this batch
                        # Get plate mask for evaluation (not for zeroing v0, but for loss exclusion)
                        plate_mask_eval = batch.get("plate_mask")
                        plate_mask_t = None
                        if plate_mask_eval is not None and EXCLUDE_PLATE_FROM_LOSS:
                            plate_mask_t = plate_mask_eval.to(device)
                        
                        u_target = x1_gt - x0  # straight-line target velocity
                        # IMPORTANT: In earlier versions, fm_loss at t=0 was identical to endpoint_loss.
                        # For meaningful diagnostics, compute FM loss at a random t using x_t.
                        t_rand = torch.rand(pod5_t.size(0), device=device)
                        x_t = (1 - t_rand[:, None, None, None, None]) * x0 + t_rand[:, None, None, None, None] * x1_gt
                        v_t = LYAPUNOV_velocity_from_valuenet(
                            eval_flow, x_t, t_rand, case_id_t,
                            bone_mask=bone_mask_img,
                        )
                        if plate_mask_t is not None:
                            plate_exclusion = (1.0 - plate_mask_t)
                            fm_loss_val = ((v_t - u_target) ** 2 * plate_exclusion).flatten(1).sum(dim=1) / plate_exclusion.flatten(1).sum(dim=1).clamp_min(1e-6)
                        else:
                            fm_loss_val = F.mse_loss(v_t, u_target, reduction='none').flatten(1).mean(dim=1)
                        train_losses['fm'].extend(fm_loss_val.tolist())
                        
                        # Endpoint loss: only if enabled
                        # Single comparable distillation metric (all modes / all LYAPUNOV_ALPHA):
                        # TeacherTangentRMSE = E_t[ RMSE( v_student(x*(t), t), dx*(t)/dt ) ]
                        x_star, dx_star_dt = svf_teacher.get_teacher_state_and_tangent(
                            pod5_t, poy1_t, t_rand, dt=LYAPUNOV_DT_MIN
                        )
                        v_star = LYAPUNOV_velocity_from_valuenet(
                            eval_flow, x_star, t_rand, case_id_t,
                            bone_mask=bone_mask_img,
                        )
                        mse_per_sample = ((v_star - dx_star_dt) ** 2).flatten(1).mean(dim=1)
                        rmse_per_sample = torch.sqrt(mse_per_sample.clamp(min=0.0) + 1e-12)
                        train_losses['teacher_tangent_rmse'].extend(rmse_per_sample.detach().cpu().tolist())

                        # Additional teacher-alignment metrics (all modes)
                        cos_per_sample = F.cosine_similarity(
                            v_star.flatten(1),
                            dx_star_dt.flatten(1),
                            dim=1,
                            eps=1e-8,
                        )
                        train_losses['teacher_tangent_cossim'].extend(cos_per_sample.detach().cpu().tolist())

                        # VelocityScaleAgreement01: min(r,1/r) in [0,1]
                        vel_mag_ratio_per_sample = velocity_scale_agreement_01(v_star, dx_star_dt)
                        train_losses['velocity_magnitude_ratio'].extend(vel_mag_ratio_per_sample.detach().cpu().tolist())

                        t1 = torch.ones(pod5_t.size(0), device=device)
                        x_star_1 = svf_teacher.get_warped_at_t(pod5_t, poy1_t, t1)
                        end_mae_per_sample = (x1_pred - x_star_1).abs().flatten(1).mean(dim=1)
                        train_losses['endpoint_mae_to_teacher'].extend(end_mae_per_sample.detach().cpu().tolist())

                        # Optional: Lyapunov objective only when analytical Lyapunov is enabled
                        if bool(USE_ANALYTICAL_LYAPUNOV) and (loss_mode_eval != 'fm_only'):
                            # Step-based warmup fraction for Lyapunov
                            if LYAPUNOV_WARMUP_EPOCHS <= 0:
                                warmup_frac_eval = 1.0
                            else:
                                warmup_steps_eval = max(1, int(LYAPUNOV_WARMUP_EPOCHS * len(train_loader)))
                                warmup_frac_eval = min(1.0, float(global_step + 1) / float(warmup_steps_eval))

                            lyapunov_loss_batch, _LYAPUNOV_info = compute_LYAPUNOV_analytical_loss(
                                v_pred=v_t,
                                z_t=x_t,       # x_t (image space)
                                z_star=x_star,  # x_star (image space)
                                dz_star_dt=dx_star_dt,  # image space
                                t=t_rand,
                                warmup_frac=warmup_frac_eval,
                                weight_map=None,
                            )
                            LYAPUNOV_scalar = float(lyapunov_loss_batch.item())
                            train_losses['lyapunov'].extend([LYAPUNOV_scalar] * batch_size)
                        
                            poy1_pred = x1_pred
                        
                        # Compute image-space losses if enabled
                        if USE_IMAGE_SPACE_LOSS:
                            img_loss_val = F.mse_loss(poy1_pred, poy1_t)
                            img_loss_scalar = img_loss_val.item()

                            train_losses['img'].extend([img_loss_scalar] * batch_size)

                            # Bone-weighted loss if enabled
                            if USE_IMG_LOSS_BONE_WEIGHTED:
                                w = _bone_weight_map(poy1_t, hu_threshold=METRICS_BONE_HU_THRESHOLD,
                                                     alpha=BONE_WEIGHT_ALPHA, surface_weight=BONE_SURFACE_WEIGHT)
                                bone_mse = ((poy1_pred - poy1_t) ** 2) * w
                                bone_loss_vals = bone_mse.flatten(1).mean(dim=1)
                                train_losses['bone'].extend(bone_loss_vals.tolist())
                        
                        # Convert to numpy and compute metrics per-sample
                        pred_norm = poy1_pred.cpu().numpy()
                        target_norm = poy1_t.cpu().numpy()
                        # Get plate mask for metrics if enabled
                        plate_mask_batch = batch.get("plate_mask", None)
                        for i in range(pred_norm.shape[0]):
                            pred_hu = denorm_to_hu(pred_norm[i, 0])
                            target_hu = denorm_to_hu(target_norm[i, 0])
                            # Extract plate mask for this sample if available
                            plate_mask_i = None
                            if plate_mask_batch is not None and EXCLUDE_PLATE_FROM_METRICS:
                                plate_mask_i = plate_mask_batch[i, 0].cpu().numpy()
                            sample_metrics = compute_comprehensive_metrics(pred_hu, target_hu, pred_norm[i, 0], target_norm[i, 0], plate_mask=plate_mask_i)
                            sample_metrics.update(compute_comprehensive_metrics_middle_slab(pred_hu, target_hu, pred_norm[i, 0], target_norm[i, 0], plate_mask=plate_mask_i))
                            train_metrics_list.append(sample_metrics)
                
                # Average training metrics
                avg_fm = float(np.mean(train_losses['fm'])) if len(train_losses['fm']) > 0 else float('nan')
                avg_ep = float(np.mean(train_losses['endpoint'])) if len(train_losses['endpoint']) > 0 else float('nan')
                avg_lyapunov = float(np.mean(train_losses['lyapunov'])) if len(train_losses['lyapunov']) > 0 else float('nan')
                avg_tan_rmse = float(np.mean(train_losses['teacher_tangent_rmse'])) if len(train_losses.get('teacher_tangent_rmse', [])) > 0 else float('nan')
                avg_tan_cos = float(np.mean(train_losses['teacher_tangent_cossim'])) if len(train_losses.get('teacher_tangent_cossim', [])) > 0 else float('nan')
                avg_vel_mag_ratio = float(np.mean(train_losses['velocity_magnitude_ratio'])) if len(train_losses.get('velocity_magnitude_ratio', [])) > 0 else float('nan')
                avg_end_mae_teacher = float(np.mean(train_losses['endpoint_mae_to_teacher'])) if len(train_losses.get('endpoint_mae_to_teacher', [])) > 0 else float('nan')

                # Optional image-space term means (only when enabled)
                avg_img = float(np.mean(train_losses['img'])) if (USE_IMAGE_SPACE_LOSS and len(train_losses['img']) > 0) else float('nan')
                avg_bone = float(np.mean(train_losses['bone'])) if (USE_IMAGE_SPACE_LOSS and USE_IMG_LOSS_BONE_WEIGHTED and len(train_losses['bone']) > 0) else float('nan')

                # Define avg_total_loss to match the selected training objective
                avg_total = 0.0
                if loss_mode_eval == 'lqr_only':
                    avg_total = avg_lyapunov if not math.isnan(avg_lyapunov) else 0.0
                elif loss_mode_eval == 'fm_only':
                    avg_total = avg_fm if not math.isnan(avg_fm) else 0.0
                else:  # 'both'
                    avg_total = (avg_fm if not math.isnan(avg_fm) else 0.0) + (avg_lyapunov if not math.isnan(avg_lyapunov) else 0.0)
                
                # Add endpoint (mode-independent, only if weight > 0)
                # Add enabled image-space terms into total if they are enabled
                if USE_IMAGE_SPACE_LOSS:
                    if not math.isnan(avg_img):
                        avg_total += IMAGE_SPACE_LOSS_WEIGHT * avg_img
                    if USE_IMG_LOSS_BONE_WEIGHTED and not math.isnan(avg_bone):
                        avg_total += BONE_LOSS_LAMBDA * avg_bone

                train_epoch_metrics = {
                    'epoch': epoch,
                    'loss_mode': loss_mode_eval,
                    'avg_total_loss': avg_total,
                    'avg_rmse_v_teacher_tangent': avg_tan_rmse,
                    'avg_cos_v_teacher_tangent': avg_tan_cos,
                    'avg_velocity_magnitude_ratio': avg_vel_mag_ratio,
                    'avg_endpoint_mae_to_teacher': avg_end_mae_teacher,
                }
                # Only show active losses in Excel
                if loss_mode_eval != 'fm_only' and not math.isnan(avg_lyapunov):
                    train_epoch_metrics['avg_lyapunov_loss'] = avg_lyapunov
                if loss_mode_eval != 'lqr_only' and not math.isnan(avg_fm):
                    train_epoch_metrics['avg_fm_loss'] = avg_fm
                # Add image-space losses if enabled and computed
                if USE_IMAGE_SPACE_LOSS:
                    if not math.isnan(avg_img):
                        train_epoch_metrics['avg_img_loss'] = avg_img
                    if USE_IMG_LOSS_BONE_WEIGHTED and not math.isnan(avg_bone):
                        train_epoch_metrics['avg_bone_loss'] = avg_bone
                
                if train_metrics_list:
                    for key in train_metrics_list[0].keys():
                        values = [m[key] for m in train_metrics_list]
                        train_epoch_metrics[key] = np.nanmean(values)  # Use nanmean to handle undefined metrics
                
                # De-duplicate any existing entry for this epoch before appending
                all_train_metrics = [m for m in all_train_metrics if int(m.get('epoch', -1)) != epoch]
                all_train_metrics.append(train_epoch_metrics)
            else:
                train_epoch_metrics = None
            
            # ============= TEST SET EVALUATION =============
            if COMPUTE_TEST_METRICS:
                print(f"  Computing metrics on test set...")
                
                test_metrics_list = []
                test_losses = {
                    'fm': [], 'endpoint': [], 'lyapunov': [],
                    'teacher_tangent_rmse': [],
                    'teacher_tangent_cossim': [],
                    'velocity_magnitude_ratio': [],
                    'endpoint_mae_to_teacher': [],
                    'img': [], 'ssim': [], 'perc': [], 'bone': []
                }
                
                # Batched evaluation over the entire test set
                test_eval_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=EVAL_NUM_WORKERS)
                with torch.no_grad(), autocast(enabled=USE_AMP):
                    for batch in test_eval_loader:
                        pod5_t = batch["pod5"].to(device)
                        poy1_t = batch["poy1"].to(device)
                        batch_size = pod5_t.size(0)
                        # Avoid copy-construct warning: if already a tensor, just move to device; else wrap with as_tensor
                        _case_id = batch["case_id"]
                        case_id_t = _case_id.to(device) if torch.is_tensor(_case_id) else torch.as_tensor(_case_id, device=device)
                        case_id_t = case_id_t.long()
                        
                        # Evaluation/inference policy:
                        # - POY1-derived masks are forbidden
                        # - POD5-derived masks are allowed
                        bone_mask_t = None
                        bone_mask_img = None
                        x0 = pod5_t
                        x1_gt = poy1_t
                        
                        # Predict endpoint at t=0 (one-step)
                        t0 = torch.zeros(pod5_t.size(0), device=device)
                        v0 = LYAPUNOV_velocity_from_valuenet(
                            eval_flow, x0, t0, case_id_t,
                            bone_mask=bone_mask_img,
                        )
                        x1_pred = _predict_endpoint_for_metrics(
                            eval_flow,
                            x0,
                            case_id_t,
                            bone_mask_img=bone_mask_img,
                            v0_at_t0=v0,
                        )
                        
                        # Get plate mask for evaluation (not for zeroing v0, but for loss exclusion)
                        plate_mask_eval = batch.get("plate_mask")
                        plate_mask_t = None
                        if plate_mask_eval is not None and EXCLUDE_PLATE_FROM_LOSS:
                            plate_mask_t = plate_mask_eval.to(device)
                        
                        # Compute losses for this batch
                        u_target = x1_gt - x0  # straight-line target velocity
                        # Compute FM loss at a random t for meaningful diagnostics
                        t_rand = torch.rand(pod5_t.size(0), device=device)
                        x_t = (1 - t_rand[:, None, None, None, None]) * x0 + t_rand[:, None, None, None, None] * x1_gt
                        v_t = LYAPUNOV_velocity_from_valuenet(
                            eval_flow, x_t, t_rand, case_id_t,
                            bone_mask=bone_mask_img,
                        )
                        if plate_mask_t is not None:
                            plate_exclusion = (1.0 - plate_mask_t)
                            fm_loss_val = ((v_t - u_target) ** 2 * plate_exclusion).flatten(1).sum(dim=1) / plate_exclusion.flatten(1).sum(dim=1).clamp_min(1e-6)
                        else:
                            fm_loss_val = F.mse_loss(v_t, u_target, reduction='none').flatten(1).mean(dim=1)
                        test_losses['fm'].extend(fm_loss_val.tolist())
                        
                        # Endpoint loss: only if enabled
                        # Single comparable distillation metric (all modes / all LYAPUNOV_ALPHA):
                        # TeacherTangentRMSE = E_t[ RMSE( v_student(x*(t), t), dx*(t)/dt ) ]
                        x_star, dx_star_dt = svf_teacher.get_teacher_state_and_tangent(
                            pod5_t, poy1_t, t_rand, dt=LYAPUNOV_DT_MIN
                        )
                        v_star = LYAPUNOV_velocity_from_valuenet(
                            eval_flow, x_star, t_rand, case_id_t,
                            bone_mask=bone_mask_img,
                        )
                        mse_per_sample = ((v_star - dx_star_dt) ** 2).flatten(1).mean(dim=1)
                        rmse_per_sample = torch.sqrt(mse_per_sample.clamp(min=0.0) + 1e-12)
                        test_losses['teacher_tangent_rmse'].extend(rmse_per_sample.detach().cpu().tolist())

                        # Additional teacher-alignment metrics (all modes)
                        cos_per_sample = F.cosine_similarity(
                            v_star.flatten(1),
                            dx_star_dt.flatten(1),
                            dim=1,
                            eps=1e-8,
                        )
                        test_losses['teacher_tangent_cossim'].extend(cos_per_sample.detach().cpu().tolist())

                        # VelocityScaleAgreement01: min(r,1/r) in [0,1]
                        vel_mag_ratio_per_sample = velocity_scale_agreement_01(v_star, dx_star_dt)
                        test_losses['velocity_magnitude_ratio'].extend(vel_mag_ratio_per_sample.detach().cpu().tolist())

                        t1 = torch.ones(pod5_t.size(0), device=device)
                        x_star_1 = svf_teacher.get_warped_at_t(pod5_t, poy1_t, t1)
                        end_mae_per_sample = (x1_pred - x_star_1).abs().flatten(1).mean(dim=1)
                        test_losses['endpoint_mae_to_teacher'].extend(end_mae_per_sample.detach().cpu().tolist())

                        # Optional: Lyapunov objective only when analytical Lyapunov is enabled
                        if bool(USE_ANALYTICAL_LYAPUNOV) and (loss_mode_eval != 'fm_only'):
                            # Step-based warmup fraction for Lyapunov
                            if LYAPUNOV_WARMUP_EPOCHS <= 0:
                                warmup_frac_eval = 1.0
                            else:
                                warmup_steps_eval = max(1, int(LYAPUNOV_WARMUP_EPOCHS * len(train_loader)))
                                warmup_frac_eval = min(1.0, float(global_step + 1) / float(warmup_steps_eval))

                            lyapunov_loss_batch, _LYAPUNOV_info = compute_LYAPUNOV_analytical_loss(
                                v_pred=v_t,
                                z_t=x_t,       # x_t (image space)
                                z_star=x_star,  # x_star (image space)
                                dz_star_dt=dx_star_dt,  # image space
                                t=t_rand,
                                warmup_frac=warmup_frac_eval,
                                weight_map=None,
                            )
                            LYAPUNOV_scalar = float(lyapunov_loss_batch.item())
                            test_losses['lyapunov'].extend([LYAPUNOV_scalar] * batch_size)
                        
                            poy1_pred = x1_pred
                        
                        # Compute image-space losses if enabled
                        if USE_IMAGE_SPACE_LOSS:
                            img_loss_val = F.mse_loss(poy1_pred, poy1_t)
                            img_loss_scalar = img_loss_val.item()

                            test_losses['img'].extend([img_loss_scalar] * batch_size)

                            # Bone-weighted loss if enabled
                            if USE_IMG_LOSS_BONE_WEIGHTED:
                                w = _bone_weight_map(poy1_t, hu_threshold=METRICS_BONE_HU_THRESHOLD,
                                                     alpha=BONE_WEIGHT_ALPHA, surface_weight=BONE_SURFACE_WEIGHT)
                                bone_mse = ((poy1_pred - poy1_t) ** 2) * w
                                bone_loss_vals = bone_mse.flatten(1).mean(dim=1)
                                test_losses['bone'].extend(bone_loss_vals.tolist())
                        
                        # Convert to numpy and compute metrics per-sample
                        pred_norm = poy1_pred.cpu().numpy()
                        target_norm = poy1_t.cpu().numpy()
                        # Get plate mask for metrics if enabled
                        plate_mask_batch = batch.get("plate_mask", None)
                        for i in range(pred_norm.shape[0]):
                            pred_hu = denorm_to_hu(pred_norm[i, 0])
                            target_hu = denorm_to_hu(target_norm[i, 0])
                            # Extract plate mask for this sample if available
                            plate_mask_i = None
                            if plate_mask_batch is not None and EXCLUDE_PLATE_FROM_METRICS:
                                plate_mask_i = plate_mask_batch[i, 0].cpu().numpy()
                            sample_metrics = compute_comprehensive_metrics(pred_hu, target_hu, pred_norm[i, 0], target_norm[i, 0], plate_mask=plate_mask_i)
                            sample_metrics.update(compute_comprehensive_metrics_middle_slab(pred_hu, target_hu, pred_norm[i, 0], target_norm[i, 0], plate_mask=plate_mask_i))
                            test_metrics_list.append(sample_metrics)
                
                # Average test metrics
                avg_fm = float(np.mean(test_losses['fm'])) if len(test_losses['fm']) > 0 else float('nan')
                avg_ep = float(np.mean(test_losses['endpoint'])) if len(test_losses['endpoint']) > 0 else float('nan')
                avg_lyapunov = float(np.mean(test_losses['lyapunov'])) if len(test_losses['lyapunov']) > 0 else float('nan')
                avg_tan_rmse = float(np.mean(test_losses['teacher_tangent_rmse'])) if len(test_losses.get('teacher_tangent_rmse', [])) > 0 else float('nan')
                avg_tan_cos = float(np.mean(test_losses['teacher_tangent_cossim'])) if len(test_losses.get('teacher_tangent_cossim', [])) > 0 else float('nan')
                avg_vel_mag_ratio = float(np.mean(test_losses['velocity_magnitude_ratio'])) if len(test_losses.get('velocity_magnitude_ratio', [])) > 0 else float('nan')
                avg_end_mae_teacher = float(np.mean(test_losses['endpoint_mae_to_teacher'])) if len(test_losses.get('endpoint_mae_to_teacher', [])) > 0 else float('nan')

                avg_img = float(np.mean(test_losses['img'])) if (USE_IMAGE_SPACE_LOSS and len(test_losses['img']) > 0) else float('nan')
                avg_bone = float(np.mean(test_losses['bone'])) if (USE_IMAGE_SPACE_LOSS and USE_IMG_LOSS_BONE_WEIGHTED and len(test_losses['bone']) > 0) else float('nan')

                if loss_mode_eval == 'lqr_only':
                    avg_total = avg_lyapunov if not math.isnan(avg_lyapunov) else 0.0
                elif loss_mode_eval == 'fm_only':
                    avg_total = avg_fm if not math.isnan(avg_fm) else 0.0
                else:
                    avg_total = (avg_fm if not math.isnan(avg_fm) else 0.0) + (avg_lyapunov if not math.isnan(avg_lyapunov) else 0.0)
                
                # Add endpoint (mode-independent, only if weight > 0)
                if USE_IMAGE_SPACE_LOSS:
                    if not math.isnan(avg_img):
                        avg_total += IMAGE_SPACE_LOSS_WEIGHT * avg_img
                    if USE_IMG_LOSS_BONE_WEIGHTED and not math.isnan(avg_bone):
                        avg_total += BONE_LOSS_LAMBDA * avg_bone

                test_epoch_metrics = {
                    'epoch': epoch,
                    'loss_mode': loss_mode_eval,
                    'avg_total_loss': avg_total,
                    'avg_rmse_v_teacher_tangent': avg_tan_rmse,
                    'avg_cos_v_teacher_tangent': avg_tan_cos,
                    'avg_velocity_magnitude_ratio': avg_vel_mag_ratio,
                    'avg_endpoint_mae_to_teacher': avg_end_mae_teacher,
                }
                # Only show active losses in Excel
                if loss_mode_eval != 'fm_only' and not math.isnan(avg_lyapunov):
                    test_epoch_metrics['avg_lyapunov_loss'] = avg_lyapunov
                if loss_mode_eval != 'lqr_only' and not math.isnan(avg_fm):
                    test_epoch_metrics['avg_fm_loss'] = avg_fm
                # Add image-space losses if enabled and computed
                if USE_IMAGE_SPACE_LOSS:
                    if not math.isnan(avg_img):
                        test_epoch_metrics['avg_img_loss'] = avg_img
                    if USE_IMG_LOSS_BONE_WEIGHTED and not math.isnan(avg_bone):
                        test_epoch_metrics['avg_bone_loss'] = avg_bone
                
                # Average all metric values across test samples
                if test_metrics_list:
                    for key in test_metrics_list[0].keys():
                        values = [m[key] for m in test_metrics_list]
                        test_epoch_metrics[key] = np.nanmean(values)  # Use nanmean to handle undefined metrics
                
                # De-duplicate any existing entry for this epoch before appending
                all_test_metrics = [m for m in all_test_metrics if int(m.get('epoch', -1)) != epoch]
                all_test_metrics.append(test_epoch_metrics)
            else:
                test_epoch_metrics = None
            
            # Print key metrics (Train vs Test comparison)
            if COMPUTE_TRAIN_METRICS and COMPUTE_TEST_METRICS:
                # Both enabled: print side-by-side comparison
                if train_epoch_metrics and test_epoch_metrics and 'MAE_all_HU' in test_epoch_metrics:
                    print(f"  Train Loss: {train_epoch_metrics['avg_total_loss']:.6f} | Test Loss: {test_epoch_metrics['avg_total_loss']:.6f}")
                    print(
                        f"  MAE_all_HU: {train_epoch_metrics.get('MAE_all_HU', float('nan')):.2f} | {test_epoch_metrics.get('MAE_all_HU', float('nan')):.2f} "
                        f"(bone: {train_epoch_metrics.get('MAE_bone_HU', float('nan')):.2f} | {test_epoch_metrics.get('MAE_bone_HU', float('nan')):.2f})"
                    )
                    print(
                        f"  MS_SSIM: {train_epoch_metrics.get('MS_SSIM', float('nan')):.4f} | {test_epoch_metrics.get('MS_SSIM', float('nan')):.4f} "
                        f"(bone: {train_epoch_metrics.get('MS_SSIM_bone', float('nan')):.4f} | {test_epoch_metrics.get('MS_SSIM_bone', float('nan')):.4f})"
                    )
                    print(f"  Dice_bone: {train_epoch_metrics.get('Dice_bone', float('nan')):.4f} | {test_epoch_metrics.get('Dice_bone', float('nan')):.4f}")
            elif COMPUTE_TRAIN_METRICS and train_epoch_metrics:
                # Only train enabled
                if 'MAE_all_HU' in train_epoch_metrics:
                    print(f"  Train Loss: {train_epoch_metrics['avg_total_loss']:.6f}")
                    print(f"  MAE_all_HU: {train_epoch_metrics.get('MAE_all_HU', float('nan')):.2f} | MAE_bone_HU: {train_epoch_metrics.get('MAE_bone_HU', float('nan')):.2f}")
                    print(f"  MS_SSIM: {train_epoch_metrics.get('MS_SSIM', float('nan')):.4f} | MS_SSIM_bone: {train_epoch_metrics.get('MS_SSIM_bone', float('nan')):.4f}")
                    print(f"  Dice_bone: {train_epoch_metrics.get('Dice_bone', float('nan')):.4f}")
            elif COMPUTE_TEST_METRICS and test_epoch_metrics:
                # Only test enabled
                if 'MAE_all_HU' in test_epoch_metrics:
                    print(f"  Test Loss: {test_epoch_metrics['avg_total_loss']:.6f}")
                    print(f"  MAE_all_HU: {test_epoch_metrics.get('MAE_all_HU', float('nan')):.2f} | MAE_bone_HU: {test_epoch_metrics.get('MAE_bone_HU', float('nan')):.2f}")
                    print(f"  MS_SSIM: {test_epoch_metrics.get('MS_SSIM', float('nan')):.4f} | MS_SSIM_bone: {test_epoch_metrics.get('MS_SSIM_bone', float('nan')):.4f}")
                    print(f"  Dice_bone: {test_epoch_metrics.get('Dice_bone', float('nan')):.4f}")
            
            # Update Excel files only for enabled metric computations
            if COMPUTE_TRAIN_METRICS:
                train_metrics_excel_path = FM_OUT_DIR / "training_metrics_TRAIN.xlsx"
                create_metrics_excel_with_footnotes(all_train_metrics, train_metrics_excel_path)
                print(f"  📊 Updated TRAIN metrics Excel (epoch {epoch})")
            
            if COMPUTE_TEST_METRICS:
                test_metrics_excel_path = FM_OUT_DIR / "training_metrics_TEST.xlsx"
                create_metrics_excel_with_footnotes(all_test_metrics, test_metrics_excel_path)
                print(f"  📊 Updated TEST metrics Excel (epoch {epoch})")
            
            if not COMPUTE_TRAIN_METRICS and not COMPUTE_TEST_METRICS:
                print(f"  ⚠️ Both train and test metric computation disabled - no Excel files generated")
            
            print(f"{'='*70}\n")
            eval_flow.train() if not USE_EMA else flow.train()
        else:
            # Skip heavy evaluation this epoch
            if epoch < NUM_EPOCHS:
                next_eval = max(1, ((epoch // EXCEL_UPDATE_INTERVAL) + 1) * EXCEL_UPDATE_INTERVAL)
                if next_eval <= epoch:
                    next_eval += EXCEL_UPDATE_INTERVAL
                print(f"  Skipping full train/test metrics this epoch. Next evaluation at epoch {min(next_eval, NUM_EPOCHS)}.")
            else:
                print("  Skipping full train/test metrics this epoch.")
            print(f"{'='*70}\n")
        
        # Save checkpoint and reconstructions at specified epoch intervals
        def _should_save_epoch(unit: str, interval: int, epoch_idx: int) -> bool:
            if interval <= 0:
                return False
            if unit != 'epoch':
                return False
            return (epoch_idx % interval == 0) or (epoch_idx == NUM_EPOCHS)

        if _should_save_epoch(CKPT_SAVE_UNIT, CKPT_SAVE_INTERVAL, epoch):
            # Save model checkpoint
            checkpoint_path = checkpoints_dir / f"flow_unet_epoch_{epoch:04d}.pth"
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': flow.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_loss': avg_loss,
                'avg_fm_loss': avg_fm_loss,
                'avg_endpoint_loss': avg_endpoint_loss,
                'config': {
                    'use_middle_slab_prior_channel': bool(globals().get('USE_MIDDLE_SLAB_PRIOR_CHANNEL', False)),
                    'use_middle_slab_prior_multi_stage_injection': bool(globals().get('USE_MIDDLE_SLAB_PRIOR_MULTI_STAGE_INJECTION', False)),
                    'middle_slab_image_slice_start': int(globals().get('MIDDLE_SLAB_IMAGE_SLICE_START', 20)),
                    'middle_slab_image_slice_end': int(globals().get('MIDDLE_SLAB_IMAGE_SLICE_END', 28)),
                    'middle_slab_profile': str(globals().get('MIDDLE_SLAB_PROFILE', 'cosine')),
                    'middle_slab_falloff': float(globals().get('MIDDLE_SLAB_FALLOFF', 5.0)),
                    'roi_shape': tuple(globals().get('ROI_SHAPE', (48, 48, 48))),
                    'LYAPUNOV_mode': 'analytical',
                    'LYAPUNOV_alpha': float(LYAPUNOV_ALPHA),
                    'LYAPUNOV_lambda_max': float(LYAPUNOV_LAMBDA_MAX),
                },
            }, checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")
        if _should_save_epoch(RECON_SAVE_UNIT, RECON_SAVE_INTERVAL, epoch):
            # Generate and save reconstructions from latent space
            # Use online model for visualizations; EMA can lag heavily early on.
            flow.eval()
            with torch.no_grad(), autocast(enabled=USE_AMP):
                # Prepare epoch directory once
                epoch_recon_dir = (FM_OUT_DIR / "training_reconstructions" / f"epoch_{epoch:04d}")
                epoch_recon_dir.mkdir(parents=True, exist_ok=True)
                # Save TEST samples
                test_recon_dir = epoch_recon_dir / "test"
                test_recon_dir.mkdir(exist_ok=True)
                for idx_in_samples, sample_idx in enumerate(test_sample_indices):
                    s = test_dataset[sample_idx]
                    pod5_t = s["pod5"].unsqueeze(0).to(device)
                    poy1_t = s["poy1"].unsqueeze(0).to(device)
                    case_id_idx = s["case_id"]
                    case_id_t = torch.as_tensor([case_id_idx], device=device, dtype=torch.long)
                    
                    # IMPORTANT: never use POY1-derived bone masks for inference/visualization.
                    bone_mask_t = None
                    bone_mask_img = None
                    
                    x0 = pod5_t
                    x1_gt = poy1_t
                    
                    t0 = torch.zeros(1, device=device)
                    eval_model = flow
                    v0 = LYAPUNOV_velocity_from_valuenet(
                        eval_model, x0, t0, case_id_t,
                        bone_mask=None,
                        capture_attn=True,
                    )
                    attn_map = None
                    if hasattr(eval_model, 'last_attn_map') and eval_model.last_attn_map is not None:
                        attn_map = eval_model.last_attn_map.squeeze().cpu().numpy()
                    x1_pred = _predict_endpoint_for_metrics(
                        eval_model,
                        x0,
                        case_id_t,
                        bone_mask_img=None,
                        v0_at_t0=v0,
                    )
                    
                    poy1_pred = x1_pred
                    poy1_recon_gt = x1_gt
                    
                    case_id = s["meta"]["case_id"]
                    roi_num = s["meta"]["roi_num"]
                    pod5_hu = denorm_to_hu(pod5_t.squeeze().cpu().numpy())
                    poy1_gt_hu = denorm_to_hu(poy1_t.squeeze().cpu().numpy())
                    poy1_recon_gt_hu = denorm_to_hu(poy1_recon_gt.squeeze().cpu().numpy())
                    poy1_pred_hu = denorm_to_hu(poy1_pred.squeeze().cpu().numpy())
                    pod5_original_hu = None
                    pod5_modified_hu = None
                    save_orthogonal_png(pod5_hu, str(test_recon_dir / f"Case_{case_id:03d}_ROI{roi_num:02d}_POD5.png"), f"POD5 (case {case_id}, ROI {roi_num})")
                    save_orthogonal_png(poy1_gt_hu, str(test_recon_dir / f"Case_{case_id:03d}_ROI{roi_num:02d}_GT.png"), f"GT POY1 (case {case_id}, ROI {roi_num})")
                    save_orthogonal_png(poy1_pred_hu, str(test_recon_dir / f"Case_{case_id:03d}_ROI{roi_num:02d}_pred.png"), f"Predicted POY1 (case {case_id}, ROI {roi_num})")
                
                # Save TRAIN samples
                train_recon_dir = epoch_recon_dir / "train"
                train_recon_dir.mkdir(exist_ok=True)
                for idx_in_samples, sample_idx in enumerate(train_sample_indices):
                    s = train_dataset[sample_idx]
                    pod5_t = s["pod5"].unsqueeze(0).to(device)
                    poy1_t = s["poy1"].unsqueeze(0).to(device)
                    case_id_idx = s["case_id"]
                    case_id_t = torch.as_tensor([case_id_idx], device=device, dtype=torch.long)
                    
                    # IMPORTANT: never use POY1-derived bone masks for inference/visualization.
                    bone_mask_t = None
                    bone_mask_img = None
                    
                    x0 = pod5_t
                    x1_gt = poy1_t
                    
                    t0 = torch.zeros(1, device=device)
                    eval_model = flow
                    v0 = LYAPUNOV_velocity_from_valuenet(
                        eval_model, x0, t0, case_id_t,
                        bone_mask=None,
                        capture_attn=True,
                    )
                    attn_map = None
                    if hasattr(eval_model, 'last_attn_map') and eval_model.last_attn_map is not None:
                        attn_map = eval_model.last_attn_map.squeeze().cpu().numpy()
                    x1_pred = _predict_endpoint_for_metrics(
                        eval_model,
                        x0,
                        case_id_t,
                        bone_mask_img=None,
                        v0_at_t0=v0,
                    )
                    
                    poy1_pred = x1_pred
                    poy1_recon_gt = x1_gt
                    
                    case_id = s["meta"]["case_id"]
                    roi_num = s["meta"]["roi_num"]
                    pod5_hu = denorm_to_hu(pod5_t.squeeze().cpu().numpy())
                    poy1_gt_hu = denorm_to_hu(poy1_t.squeeze().cpu().numpy())
                    poy1_recon_gt_hu = denorm_to_hu(poy1_recon_gt.squeeze().cpu().numpy())
                    poy1_pred_hu = denorm_to_hu(poy1_pred.squeeze().cpu().numpy())
                    pod5_original_hu = None
                    pod5_modified_hu = None
                    save_orthogonal_png(pod5_hu, str(train_recon_dir / f"Case_{case_id:03d}_ROI{roi_num:02d}_POD5.png"), f"POD5 (case {case_id}, ROI {roi_num})")
                    save_orthogonal_png(poy1_gt_hu, str(train_recon_dir / f"Case_{case_id:03d}_ROI{roi_num:02d}_GT.png"), f"GT POY1 (case {case_id}, ROI {roi_num})")
                    save_orthogonal_png(poy1_pred_hu, str(train_recon_dir / f"Case_{case_id:03d}_ROI{roi_num:02d}_pred.png"), f"Predicted POY1 (case {case_id}, ROI {roi_num})")
            
            print(f"🖼️  Saved {len(test_sample_indices)} TEST + {len(train_sample_indices)} TRAIN sample reconstructions to: {epoch_recon_dir}\n")
            flow.train()
    
    # Final message (Excel files already updated after each epoch)
    train_metrics_excel_path = FM_OUT_DIR / "training_metrics_TRAIN.xlsx"
    test_metrics_excel_path = FM_OUT_DIR / "training_metrics_TEST.xlsx"
    
    print(f"\n✅ Flow training complete!")
    print(f"📊 Final metrics saved to:")
    print(f"   - Training: {train_metrics_excel_path}")
    print(f"   - Test: {test_metrics_excel_path}\n")
    return flow

# ----------------------- Inference Helper Functions ----------
@torch.no_grad()
def map_and_decode_direct(
    flow,
    pod5_vol,
    case_id,
    poy1_vol=None,
    guidance_scale=None,
):
    """
    Direct one-step inference: x1 = x0 + v(x0, t=0)
    
    Args:
        flow: Flow matching model (works in 48³ image space)
        pod5_vol: POD5 volume (input CT, [48,48,48] or [1,1,48,48,48])
        case_id: Case ID for conditioning
        poy1_vol: MUST be None. POY1-based conditioning is forbidden at inference.
        guidance_scale: Not used.
    """
    if poy1_vol is not None:
        raise RuntimeError(
            "POY1-based conditioning is forbidden at inference. "
            "Pass poy1_vol=None (Day-5-only inference)."
        )
    flow.eval()
    
    # Prepare input: [B, 1, 48, 48, 48]
    if pod5_vol.dim() == 3:
        x = pod5_vol[None, None].to(device)
    elif pod5_vol.dim() == 4:
        x = pod5_vol[None].to(device)
    else:
        x = pod5_vol.to(device)
    
    c = torch.tensor([case_id], device=device)
    
    # IMPORTANT: POY1-derived conditioning is forbidden at inference.
    # If source is POD5, we can derive a mask directly from the POD5 input volume.
    bone_mask_img = None
    x0 = x  # [1, 1, 48, 48, 48]
    
    # Predict velocity at t=0 (Lyapunov: velocity from gradient)
    t0 = torch.zeros(1, device=device)
    if guidance_scale is not None and guidance_scale > 1.0 and bone_mask_img is not None:
        v_guided = LYAPUNOV_velocity_from_valuenet(
            flow, x0, t0, c,
            bone_mask=bone_mask_img,
        )
        v_unguided = LYAPUNOV_velocity_from_valuenet(
            flow, x0, t0, c,
            bone_mask=None,
        )
        v0 = v_unguided + guidance_scale * (v_guided - v_unguided)
    else:
        v0 = LYAPUNOV_velocity_from_valuenet(
            flow, x0, t0, c,
            bone_mask=bone_mask_img,
        )
    
    # Direct prediction - result is already the image
    x1_pred = x0 + v0
    
    return x1_pred.squeeze().cpu()

@torch.no_grad()
def map_and_decode_integrated(
    flow,
    pod5_vol,
    case_id,
    poy1_vol=None,
    guidance_scale=None,
    steps=EVAL_INTEGRATION_STEPS,
):
    """Multi-step ODE integration in 48³ image space."""
    if poy1_vol is not None:
        raise RuntimeError(
            "POY1-based conditioning is forbidden at inference. "
            "Pass poy1_vol=None (Day-5-only inference)."
        )
    flow.eval()
    
    if pod5_vol.dim() == 3:
        x = pod5_vol[None, None].to(device)
    elif pod5_vol.dim() == 4:
        x = pod5_vol[None].to(device)
    else:
        x = pod5_vol.to(device)
    
    c = torch.tensor([case_id], device=device)

    bone_mask_img = None
    def _expand_mask(mask, batch_size):
        if mask is None:
            return None
        if mask.shape[0] == batch_size:
            return mask
        return mask.expand(batch_size, -1, -1, -1, -1)

    def _velocity(x_state, t_vec):
        batch_size = x_state.shape[0]
        case_vec = c.expand(batch_size)
        mask_arg = _expand_mask(bone_mask_img, batch_size)

        if guidance_scale is not None and guidance_scale > 1.0 and mask_arg is not None:
            v_guided = LYAPUNOV_velocity_from_valuenet(
                flow, x_state, t_vec, case_vec,
                bone_mask=mask_arg,
            )
            v_unguided = LYAPUNOV_velocity_from_valuenet(
                flow, x_state, t_vec, case_vec,
                bone_mask=None,
            )
            return v_unguided + guidance_scale * (v_guided - v_unguided)

        return LYAPUNOV_velocity_from_valuenet(
            flow, x_state, t_vec, case_vec,
            bone_mask=mask_arg,
        )
    
    x_t = x.clone()
    dt = 1.0 / steps
    batch_size = x_t.shape[0]

    if INTEGRATION_METHOD == 'rk4':
        for s in range(steps):
            t0 = torch.full((batch_size,), s * dt, device=device)
            k1 = _velocity(x_t, t0)
            k2 = _velocity(x_t + 0.5 * dt * k1, t0 + 0.5 * dt)
            k3 = _velocity(x_t + 0.5 * dt * k2, t0 + 0.5 * dt)
            k4 = _velocity(x_t + dt * k3, t0 + dt)
            x_t = x_t + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    elif INTEGRATION_METHOD == 'heun':
        for s in range(steps):
            t0 = torch.full((batch_size,), s * dt, device=device)
            v0 = _velocity(x_t, t0)
            x_euler = x_t + dt * v0
            t1 = torch.full((batch_size,), (s + 1) * dt, device=device)
            v1 = _velocity(x_euler, t1)
            x_t = x_t + 0.5 * dt * (v0 + v1)
    else:  # euler
        for s in range(steps):
            t = torch.full((batch_size,), s * dt, device=device)
            v = _velocity(x_t, t)
            x_t = x_t + dt * v
    
    return x_t.squeeze().cpu()


@torch.no_grad()
def integrate_image_space_flow_trajectory(
    flow: nn.Module,
    pod5_vol: torch.Tensor,
    case_id,
    *,
    t0: float = 0.0,
    t1: float = 1.0,
    step: float = 0.1,
    guidance_scale=None,
) -> tuple[list[float], list[torch.Tensor]]:
    """Integrate x' = v_theta(x,t) and return a trajectory at fixed time steps.

    Returns:
        (t_values, x_values) where x_values are CPU tensors of shape [D,H,W] (or [1,D,H,W] if present).
    """
    flow.eval()

    t0f = float(t0)
    t1f = float(t1)
    st = float(step)
    if st <= 0:
        raise ValueError(f"step must be > 0 (got {step!r})")
    if t1f < t0f:
        raise ValueError(f"t1 must be >= t0 (got t0={t0f}, t1={t1f})")

    # Accept 3D/4D/5D input shapes.
    if pod5_vol.dim() == 3:
        x = pod5_vol[None, None].to(device)
    elif pod5_vol.dim() == 4:
        x = pod5_vol[None].to(device)
    else:
        x = pod5_vol.to(device)

    case_id_int = int(case_id.item()) if torch.is_tensor(case_id) else int(case_id)
    c = torch.tensor([case_id_int], device=device)

    bone_mask_img = None
    def _expand_mask(mask, batch_size):
        if mask is None:
            return None
        if mask.shape[0] == batch_size:
            return mask
        return mask.expand(batch_size, -1, -1, -1, -1)

    def _velocity(x_state, t_vec):
        batch_size = x_state.shape[0]
        case_vec = c.expand(batch_size)
        mask_arg = _expand_mask(bone_mask_img, batch_size)

        if guidance_scale is not None and guidance_scale > 1.0 and mask_arg is not None:
            v_guided = LYAPUNOV_velocity_from_valuenet(
                flow, x_state, t_vec, case_vec,
                bone_mask=mask_arg,
            )
            v_unguided = LYAPUNOV_velocity_from_valuenet(
                flow, x_state, t_vec, case_vec,
                bone_mask=None,
            )
            return v_unguided + guidance_scale * (v_guided - v_unguided)

        return LYAPUNOV_velocity_from_valuenet(
            flow, x_state, t_vec, case_vec,
            bone_mask=mask_arg,
        )

    # Build fixed time grid inclusive of endpoints using the requested step size.
    # Use an epsilon guard to include t1 despite floating point drift.
    eps = 1e-9
    if abs(t1f - t0f) <= eps:
        return [t0f], [x.squeeze().detach().cpu()]

    t_values: list[float] = []
    t_curr = t0f
    while t_curr < (t1f - eps):
        t_values.append(float(t_curr))
        t_curr += st
    t_values.append(float(t1f))

    x_t = x.clone()
    batch_size = x_t.shape[0]
    x_values: list[torch.Tensor] = [x_t.squeeze().detach().cpu()]

    method = str(globals().get("INTEGRATION_METHOD", "rk4")).lower().strip()
    for k in range(len(t_values) - 1):
        t_curr = float(t_values[k])
        t_next = float(t_values[k + 1])
        dt = float(max(t_next - t_curr, 0.0))
        if dt <= 0.0:
            x_values.append(x_t.squeeze().detach().cpu())
            continue

        t_vec = torch.full((batch_size,), float(t_curr), device=device)

        if method == "rk4":
            k1 = _velocity(x_t, t_vec)
            k2 = _velocity(x_t + 0.5 * dt * k1, t_vec + 0.5 * dt)
            k3 = _velocity(x_t + 0.5 * dt * k2, t_vec + 0.5 * dt)
            k4 = _velocity(x_t + dt * k3, t_vec + dt)
            x_t = x_t + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        elif method == "heun":
            v0 = _velocity(x_t, t_vec)
            x_euler = x_t + dt * v0
            t1_vec = torch.full((batch_size,), float(t_next), device=device)
            v1 = _velocity(x_euler, t1_vec)
            x_t = x_t + 0.5 * dt * (v0 + v1)
        else:  # euler
            v = _velocity(x_t, t_vec)
            x_t = x_t + dt * v

        x_values.append(x_t.squeeze().detach().cpu())

    return t_values, x_values



# ----------------------- Inference ---------------------------
@torch.no_grad()
def run_inference(dataset, flow_model_path, out_dir, use_direct=True, max_samples=None):
    """
    Evaluate a trained flow model.
    
    Args:
        dataset: ROI3DDataset instance
        flow_model_path: Path to trained flow model checkpoint
        out_dir: Output directory for results
        use_direct: If True, use direct one-step inference; otherwise use ODE integration
        max_samples: Maximum number of samples to evaluate (None = all)
    """
    print(f"\n{'='*70}")
    print(f"STEP 3: Inference ({'Direct' if use_direct else INTEGRATION_METHOD.upper()})")
    print(f"{'='*70}")
    print(f"Loading flow model from: {flow_model_path}")
    
    # Get n_cases from underlying dataset (handles both Subset and full Dataset)
    if hasattr(dataset, 'dataset'):
        # It's a Subset from random_split
        full_dataset = dataset.dataset
        n_cases = full_dataset.n_cases
    else:
        # It's a full ROI3DDataset
        n_cases = dataset.n_cases
        full_dataset = dataset

    # Load flow model
    flow = UNetFlowNetwork(
        image_channels=IMAGE_CHANNELS,
        base_channels=UNET_BASE_CHANNELS,
        n_cases=n_cases,
        use_attention=UNET_USE_ATTENTION
    ).to(device)
    
    checkpoint = torch.load(flow_model_path, map_location=device, weights_only=False)
    ck_cfg = checkpoint.get('config', {}) if isinstance(checkpoint, dict) else {}
    if ck_cfg:
        want = bool(globals().get('USE_MIDDLE_SLAB_PRIOR_CHANNEL', False))
        got = bool(ck_cfg.get('use_middle_slab_prior_channel', False))
        if want != got:
            print(
                "⚠️  Checkpoint/config mismatch for middle-slab prior channel: "
                f"checkpoint={got} vs current={want}. "
                "This can cause load_state_dict shape errors unless they match."
            )

        want_ms = bool(globals().get('USE_MIDDLE_SLAB_PRIOR_MULTI_STAGE_INJECTION', False))
        got_ms = bool(ck_cfg.get('use_middle_slab_prior_multi_stage_injection', False))
        if want_ms != got_ms:
            print(
                "⚠️  Checkpoint/config mismatch for USE_MIDDLE_SLAB_PRIOR_MULTI_STAGE_INJECTION: "
                f"checkpoint={got_ms} vs current={want_ms}. "
                "This changes model parameters and can affect load_state_dict."
            )


    flow.load_state_dict(checkpoint['model_state_dict'])
    flow.eval()
    
    print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"   Training loss: {checkpoint.get('avg_loss', 'N/A'):.6f}\n")
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Determine samples to evaluate
    num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    idxs = list(range(num_samples))
    
    rows = [("index", "case_id", "roi_num", "mode", "mse_norm", "mae_norm", "png_pred", "png_gt", "png_pod5")]

    for i, idx in enumerate(idxs):
        s = dataset[idx]
        case_id = s["meta"]["case_id"]
        roi_num = s["meta"]["roi_num"]

        pod5_tensor = s["pod5"].float()
        poy1_tensor = s.get("poy1")
        if poy1_tensor is None:
            raise RuntimeError("Inference dataset must provide POY1 targets for evaluation metrics.")
        poy1_tensor = poy1_tensor.float()

        pod5_vol = pod5_tensor.squeeze().cpu()
        gt_vol = poy1_tensor.squeeze().cpu()

        pod5_hu = denorm_to_hu(pod5_vol.numpy())
        gt_hu = denorm_to_hu(gt_vol.numpy())
        png_pod5 = os.path.join(out_dir, f"Case_{case_id:03d}_ROI{roi_num:02d}_POD5.png")
        png_pod5_orig = None
        png_pod5_mod = None
        pod5_original_hu = None
        pod5_modified_hu = None
        png_gt = os.path.join(out_dir, f"Case_{case_id:03d}_ROI{roi_num:02d}_GT.png")

        # IMPORTANT: POY1 must NOT be used as an inference input.
        # We keep POY1 only as a target for metrics/visualization.
        guidance_variants = [("unguided", None, None)]

        saved_common_views = False

        for mode_idx, (mode_name, guidance_tensor, cfg_scale) in enumerate(guidance_variants):
            if use_direct:
                pred = map_and_decode_direct(
                    flow,
                    pod5_tensor,
                    s["case_id"],
                    poy1_vol=None,
                    guidance_scale=None,
                )
            else:
                pred = map_and_decode_integrated(
                    flow,
                    pod5_tensor,
                    s["case_id"],
                    poy1_vol=None,
                    guidance_scale=None,
                    steps=EVAL_INTEGRATION_STEPS,
                )

            pred = pred.squeeze()
            mse_val = float(torch.mean((pred - gt_vol) ** 2))
            mae_val = float(torch.mean(torch.abs(pred - gt_vol)))

            print(f"  [{i+1}/{len(idxs)}] Case {case_id} ROI {roi_num} [{mode_name}]: MSE={mse_val:.6f}, MAE={mae_val:.6f}")

            pred_hu = denorm_to_hu(pred.numpy())

            if mode_idx == 0:
                png_pred = os.path.join(out_dir, f"Case_{case_id:03d}_ROI{roi_num:02d}_FM.png")
            else:
                safe_suffix = re.sub(r"[^0-9A-Za-z]+", "_", mode_name.upper()).strip("_")
                if not safe_suffix:
                    safe_suffix = f"MODE{mode_idx}"
                png_pred = os.path.join(out_dir, f"Case_{case_id:03d}_ROI{roi_num:02d}_FM_{safe_suffix}.png")

            if not saved_common_views:
                save_orthogonal_png(pod5_hu, png_pod5, f"POD5 Input (case {case_id}, ROI {roi_num})")
                save_orthogonal_png(gt_hu, png_gt, f"GT POY1 (case {case_id}, ROI {roi_num})")
                saved_common_views = True
            save_orthogonal_png(pred_hu, png_pred, f"FM Predicted POY1 ({mode_name}, case {case_id}, ROI {roi_num})")

            rows.append((i, case_id, roi_num, mode_name, mse_val, mae_val, png_pred, png_gt, png_pod5))
    
    csv_path = os.path.join(out_dir, "inference_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    
    print(f"\n✅ Inference complete! Results saved to: {out_dir}")
    print(f"   Metrics: {csv_path}\n")
    
    return rows

# ----------------------- Main --------------------------------
def main():
    """
    Main training pipeline:
    - Train Flow Matching with UNet
    
    For inference on a trained model, use run_inference() separately.
    """
    print(f"\n{'='*70}")
    print(f"Flow Matching UNet (IMAGE SPACE)")
    print(f"{'='*70}\n")
    
    # Load full dataset
    full_dataset = ROI3DDataset(POD5_DIR, POY1_DIR)
    if len(full_dataset) == 0:
        raise RuntimeError("❌ No paired volumes found. Check POD5_DIR and POY1_DIR paths.")

    # Split dataset
    train_dataset, test_dataset = split_dataset(full_dataset, TRAIN_SPLIT, RANDOM_SEED, SPLIT_BY_PATIENT)
    
    # Train Flow Matching with UNet
    flow = train_flow_matching(train_dataset, test_dataset)
    
    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"📁 Training outputs: {FM_OUT_DIR}")
    print(f"   - Checkpoints: {FM_OUT_DIR / 'checkpoints'}")
    print(f"   - Training reconstructions: {FM_OUT_DIR / 'training_reconstructions'}")
    print(f"{'='*70}\n")
    
    return flow, train_dataset, test_dataset

def run_inference_only(checkpoint_path, max_samples=10):
    """
    Run inference using a trained checkpoint.
    
    Args:
        checkpoint_path: Path to the trained flow model checkpoint (.pth file)
        max_samples: Maximum number of samples to evaluate
    """
    print(f"\n{'='*70}")
    print(f"Running Inference Only")
    print(f"{'='*70}\n")
    
    # Load full dataset and split
    full_dataset = ROI3DDataset(POD5_DIR, POY1_DIR)
    if len(full_dataset) == 0:
        raise RuntimeError("❌ No paired volumes found. Check POD5_DIR and POY1_DIR paths.")

    # Split using SAME method as training (this defines which augmentations are used)
    train_dataset, test_dataset = split_dataset(full_dataset, TRAIN_SPLIT, RANDOM_SEED, SPLIT_BY_PATIENT)
    # Run inference on TEST SET
    inference_out_dir = FM_OUT_DIR / "inference_results"
    results = run_inference(
        dataset=test_dataset,
        flow_model_path=checkpoint_path,
        out_dir=str(inference_out_dir),
        use_direct=USE_DIRECT_ONE_STEP_INFERENCE,
        max_samples=max_samples
    )
    
    print(f"\n{'='*70}")
    print(f"✅ INFERENCE COMPLETE!")
    print(f"{'='*70}")
    print(f"📁 Results saved to: {inference_out_dir}")
    print(f"{'='*70}\n")
    
    return results


if __name__ == "__main__":
    if MODE == 'train':
        # Run full training pipeline
        main()
    
    elif MODE == 'inference':
        # Run inference only on a trained checkpoint
        # Specify the checkpoint path (update the epoch number as needed)
        CHECKPOINT_PATH = FM_OUT_DIR / "checkpoints" / "flow_unet_epoch_0100.pth"
        run_inference_only(checkpoint_path=str(CHECKPOINT_PATH), max_samples=10)
    
    else:
        print(f"❌ Invalid MODE: {MODE}. Use 'train' or 'inference'.")
