"""
fusion/F1_gated_stack.py
========================
Weighted channel stacking with learnable phase gates (strategy a2).

Each phase is multiplied by a learned scalar gate before stacking.
The gates are global (one scalar per phase, shared across all spatial
locations and patients) and passed through a sigmoid so they stay in
(0, 1).  This lets the network down-weight uninformative phases during
training without any additional inputs.

Clinical motivation
-------------------
Arterial phase carries the most discriminative signal for HCC
(non-rim arterial phase hyperenhancement), while portal venous phase
dominates for metastasis detection (BCLM / CRLM washout).  A single
global gate per phase is the minimal learnable mechanism that can
capture this prior without overfitting.

Ablation role
-------------
F0 → F1 isolates the effect of phase weighting while keeping
everything else (channel count, encoder, decoder) identical.
Gates are logged at the end of training to inspect which phases
the network found most informative.

Contract
--------
    FUSION_ID    = "F1"
    OUT_CHANNELS = 3
    build_load_fn(image_size) → load_fn(patient_root, slice_idx)
                                → (img_t (3,H,W), mask_t (1,H,W))

Note: the gate parameters live inside the GatedFusion nn.Module returned
by build_gate_module().  The notebook must:
  1. Call build_load_fn()  to get the per-slice loader (for caching).
  2. Call build_gate_module().to(device) and include its parameters in
     the optimizer so the gates are actually learned.
  3. Call gate_module(img_batch) inside the training loop BEFORE the
     model forward pass.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import nibabel as nib

# ---------------------------------------------------------------------------
# Contract constants
# ---------------------------------------------------------------------------

FUSION_ID:    str = "F1"
OUT_CHANNELS: int = 3

# ---------------------------------------------------------------------------
# HU windowing — liver preset (WW=150, WL=60)
# ---------------------------------------------------------------------------

_LO = 60.0 - 150.0 / 2   # -15 HU
_HI = 60.0 + 150.0 / 2   # +135 HU


def _window(arr: np.ndarray) -> np.ndarray:
    return np.clip(
        (arr.astype(np.float32) - _LO) / (_HI - _LO), 0.0, 1.0
    )


# ---------------------------------------------------------------------------
# Learnable gate module
# ---------------------------------------------------------------------------

class GatedFusion(nn.Module):
    """
    Applies a learned scalar gate to each of the 3 input channels.

    Input:  FloatTensor (B, 3, H, W)  — stacked NC / ART / PVP
    Output: FloatTensor (B, 3, H, W)  — each channel scaled by sigmoid(gate)

    The output is re-normalised channel-wise so the total energy is
    preserved (avoids the network simply learning to suppress all phases
    to minimise loss).

    Usage
    -----
        gate = GatedFusion().to(device)
        # include gate.parameters() in the optimizer
        fused = gate(raw_stack)   # call before model(fused)
    """

    def __init__(self, n_phases: int = 3):
        super().__init__()
        # Initialise gates at 0 so sigmoid starts at 0.5 (equal weighting)
        self.gates = nn.Parameter(torch.zeros(n_phases))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        weights = torch.sigmoid(self.gates)          # (3,)
        weights = weights / (weights.sum() + 1e-8)   # normalise to sum=1
        return x * weights.view(1, -1, 1, 1)

    def gate_values(self) -> dict[str, float]:
        """Return current gate values for logging."""
        w = torch.sigmoid(self.gates).detach().cpu()
        w = w / (w.sum() + 1e-8)
        return {"nc": w[0].item(), "art": w[1].item(), "pvp": w[2].item()}


def build_gate_module() -> GatedFusion:
    """
    Instantiate the gate module.  Call .to(device) and include
    its parameters in the optimiser.
    """
    return GatedFusion(n_phases=3)


# ---------------------------------------------------------------------------
# load_fn factory  (identical to F0 — gates are applied at train time)
# ---------------------------------------------------------------------------

def build_load_fn(image_size: int = 256):
    """
    Return a callable that loads the raw stacked phases (same as F0).

    The gating is NOT applied here — it happens inside the training loop
    via the GatedFusion module so the gates are actually learned.
    The cache therefore stores the same (3, H, W) tensors as F0.

    Args:
        image_size: spatial size (square) to resize to.

    Returns:
        load_fn(patient_root, slice_idx)
            → (img_t (3, H, W) float32,  mask_t (1, H, W) float32)
    """
    img_tf = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])
    msk_tf = T.Resize((image_size, image_size))

    def load_fn(patient_root: str, slice_idx: int):
        pr = Path(patient_root)

        nc_s  = _window(nib.load(str(pr / "NIFTI" / "nc.nii.gz" )).get_fdata()[:, :, slice_idx])
        art_s = _window(nib.load(str(pr / "NIFTI" / "art.nii.gz")).get_fdata()[:, :, slice_idx])
        pvp_s = _window(nib.load(str(pr / "NIFTI" / "pvp.nii.gz")).get_fdata()[:, :, slice_idx])

        rgb   = np.stack([nc_s, art_s, pvp_s], axis=-1)
        img_t = img_tf(Image.fromarray((rgb * 255).astype(np.uint8)))

        mask_arr = (nib.load(str(pr / "mask_pvp.nii.gz")).get_fdata()[:, :, slice_idx] > 0).astype(np.uint8)
        mask_pil = msk_tf(Image.fromarray(mask_arr * 255))
        mask_t   = (torch.from_numpy(np.array(mask_pil)).float() > 0).float().unsqueeze(0)

        return img_t, mask_t

    return load_fn
