"""
fusion/F0_early_fusion.py
=========================
Early fusion: NC + ART + PVP are windowed and stacked into a single
(3, H, W) tensor before the model sees them.

    Channel 0 (R) → NC   (non-contrast)
    Channel 1 (G) → ART  (arterial)
    Channel 2 (B) → PVP  (portal venous)

This is the simplest possible multi-phase strategy and serves as the
F0 baseline for comparing against later fusion approaches.

Required contract
-----------------
Every fusion module must expose:

    FUSION_ID:   str         short identifier used in filenames / logs
    OUT_CHANNELS: int        number of channels the model will receive
    build_load_fn(image_size) → callable(patient_root, slice_idx)
                               returns (img_tensor, mask_tensor)

The notebook calls build_load_fn() once, then passes the resulting
callable to shared.dataset.build_cached_dataset().
"""

import numpy as np
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import nibabel as nib

# ---------------------------------------------------------------------------
# Module-level constants (read by the notebook)
# ---------------------------------------------------------------------------

FUSION_ID:    str = "F0"
OUT_CHANNELS: int = 3       # model receives (3, H, W) — NC + ART + PVP


# ---------------------------------------------------------------------------
# HU windowing — liver preset  (WW=150, WL=60)
# ---------------------------------------------------------------------------

_LO = 60.0 - 150.0 / 2    # -15 HU
_HI = 60.0 + 150.0 / 2    # +135 HU


def _window(arr: np.ndarray) -> np.ndarray:
    return np.clip((arr.astype(np.float32) - _LO) / (_HI - _LO), 0.0, 1.0)


# ---------------------------------------------------------------------------
# load_fn factory
# ---------------------------------------------------------------------------

def build_load_fn(image_size: int = 256):
    """
    Return a callable that loads and fuses one axial slice.

    The returned function is passed to shared.dataset.build_cached_dataset()
    which calls it once per slice while caching patient volumes.

    Args:
        image_size: spatial size (square) to resize to.

    Returns:
        load_fn(patient_root: str, slice_idx: int)
            → (img_t: FloatTensor (3, H, W),
               mask_t: FloatTensor (1, H, W))
    """
    img_tf = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),        # PIL (H,W,3) uint8 → FloatTensor (3,H,W) in [0,1]
    ])
    msk_tf = T.Resize((image_size, image_size))

    def load_fn(patient_root: str, slice_idx: int):
        pr = Path(patient_root)

        # Load each phase volume and extract the axial slice
        nc_s  = _window(nib.load(str(pr / "NIFTI" / "nc.nii.gz" )).get_fdata()[:, :, slice_idx])
        art_s = _window(nib.load(str(pr / "NIFTI" / "art.nii.gz")).get_fdata()[:, :, slice_idx])
        pvp_s = _window(nib.load(str(pr / "NIFTI" / "pvp.nii.gz")).get_fdata()[:, :, slice_idx])

        # Stack into (H, W, 3) → PIL → tensor
        rgb   = np.stack([nc_s, art_s, pvp_s], axis=-1)
        img_t = img_tf(Image.fromarray((rgb * 255).astype(np.uint8)))

        # Load mask
        mask_arr = (nib.load(str(pr / "mask_pvp.nii.gz")).get_fdata()[:, :, slice_idx] > 0).astype(np.uint8)
        mask_pil = msk_tf(Image.fromarray(mask_arr * 255))
        mask_t   = (torch.from_numpy(np.array(mask_pil)).float() > 0).float().unsqueeze(0)

        return img_t, mask_t

    return load_fn
