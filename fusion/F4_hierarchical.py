"""
fusion/F4_hierarchical.py
==========================
F4 — Hierarchical multi-scale feature fusion.

Unlike F0–F3 which fuse phases before the encoder, F4 keeps the three
phase streams separate through the CNN encoder and fuses them at every
decoder skip level via a Phase Fusion Module (PFM).

Because F4 requires three separate encoder passes (one per phase), the
fusion cannot be encapsulated in a simple load_fn that returns a single
tensor.  Instead:

  - build_load_fn() returns a loader that caches the three phases
    as separate channels (same 3-channel tensor as F0) — the runner
    notebook uses this for caching.
  - The model variants (D*_F4 files) unpack the 3 channels back into
    individual phase tensors, run the encoder on each, and pass all
    three to the HierarchicalFusionDecoder.

Contract
--------
    FUSION_ID    = "F4"
    OUT_CHANNELS = 3   # same cache format as F0 — phases stacked
    build_load_fn(image_size) → load_fn identical to F0
"""

import numpy as np
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import nibabel as nib

FUSION_ID:    str = "F4"
OUT_CHANNELS: int = 3   # stacked NC+ART+PVP, same as F0

_LO = 60.0 - 150.0 / 2
_HI = 60.0 + 150.0 / 2


def _window(arr: np.ndarray) -> np.ndarray:
    return np.clip(
        (arr.astype(np.float32) - _LO) / (_HI - _LO), 0.0, 1.0
    )


def build_load_fn(image_size: int = 256):
    """
    Identical to F0 — caches stacked (3,H,W) tensors.
    The model variant separates them again in its forward() pass.
    """
    img_tf = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
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
