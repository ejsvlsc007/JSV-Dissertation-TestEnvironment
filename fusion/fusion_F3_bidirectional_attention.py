"""
fusion/F3_bidirectional_attention.py
=====================================
Bidirectional cross-phase attention — DIB (Dual Interaction Block) module
(strategy c3).

Unlike F2 where each phase independently queries the others, here the
three phases are updated simultaneously and symmetrically.  Each phase
both attends to and is attended by the others in a single joint
operation, better reflecting the co-registered nature of NC / ART / PVP
(they are not a causal sequence — they are parallel views of the same
anatomy).

DIB module design
-----------------
1. Project each phase independently: 1 → embed_dim channels.
2. Concatenate all three along the token dimension.
3. Run joint self-attention across all 3×T tokens simultaneously.
   Every token (from any phase) can attend to every other token
   (from any phase) — this is the bidirectional step.
4. Split the output back into three per-phase streams.
5. Each stream is broadcast to full resolution and summed with its
   original phase (residual).
6. Final 1×1 conv projects the 3 residual-updated channels to output.

Difference from F2
------------------
F2: phase_i queries [phase_j, phase_k] → attended_i  (3 separate attention ops)
F3: [phase_i, phase_j, phase_k] jointly attend each other → all updated at once
    Parameter overhead over F2: one attention layer instead of three,
    but operating on 3× the token count.

Ablation role
-------------
F2 → F3 isolates the effect of joint vs. independent attention updates.
All other components (backbone, decoder, loss, data) are identical.

Contract
--------
    FUSION_ID    = "F3"
    OUT_CHANNELS = 3
    build_load_fn(image_size) → load_fn(patient_root, slice_idx)
                                → (img_t (3,H,W), mask_t (1,H,W))
    build_fusion_module(...) → DIBFusion nn.Module
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import nibabel as nib

# ---------------------------------------------------------------------------
# Contract constants
# ---------------------------------------------------------------------------

FUSION_ID:    str = "F3"
OUT_CHANNELS: int = 3

# ---------------------------------------------------------------------------
# HU windowing
# ---------------------------------------------------------------------------

_LO = 60.0 - 150.0 / 2
_HI = 60.0 + 150.0 / 2


def _window(arr: np.ndarray) -> np.ndarray:
    return np.clip(
        (arr.astype(np.float32) - _LO) / (_HI - _LO), 0.0, 1.0
    )


# ---------------------------------------------------------------------------
# DIB module
# ---------------------------------------------------------------------------

class DIBFusion(nn.Module):
    """
    Dual Interaction Block — bidirectional cross-phase attention.

    All three phases are projected, tokenised, and jointly attended in
    one self-attention operation.  Each phase's output tokens are
    extracted from the joint output, broadcast to full resolution,
    and blended back with the original phase via a learned residual.

    Args:
        spatial_tokens: spatial resolution for attention (default 8 → 64 tokens).
        embed_dim:      internal embedding dimension.
        num_heads:      attention heads for joint self-attention.
        image_size:     input / output spatial size.

    Input:  (B, 3, H, W)
    Output: (B, 3, H, W)
    """

    def __init__(
        self,
        spatial_tokens: int = 8,
        embed_dim:      int = 64,
        num_heads:      int = 4,
        image_size:     int = 256,
    ):
        super().__init__()
        self.spatial_tokens = spatial_tokens
        self.image_size     = image_size
        self.embed_dim      = embed_dim
        T_sq                = spatial_tokens * spatial_tokens   # tokens per phase

        # Independent per-phase projection: 1ch → embed_dim
        self.phase_proj = nn.ModuleList([
            nn.Conv2d(1, embed_dim, 1) for _ in range(3)
        ])

        # Joint self-attention across all 3×T_sq tokens
        self.joint_norm = nn.LayerNorm(embed_dim)
        self.joint_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=0.0
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # Per-phase broadcast: embed_dim → full spatial resolution
        self.broadcast = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, image_size * image_size),
            )
            for _ in range(3)
        ])

        # Final projection: 3 blended channels → 3 output channels
        self.out_proj = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 3, 1),
        )

        # Per-phase learned residual blend (sigmoid, init at 0 → 0.5)
        self.blend = nn.Parameter(torch.zeros(3))

    def _to_tokens(self, feat: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, s*s, C) via adaptive avg pool."""
        s      = self.spatial_tokens
        pooled = F.adaptive_avg_pool2d(feat, (s, s))
        return pooled.flatten(2).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) — stacked NC(0) / ART(1) / PVP(2)
        Returns:
            (B, 3, H, W)
        """
        B, _, H, W = x.shape
        T_sq = self.spatial_tokens ** 2

        # 1. Project each phase to embed_dim
        phase_feats = [
            self.phase_proj[i](x[:, i:i+1, :, :])
            for i in range(3)
        ]                                                # 3 × (B, embed_dim, H, W)

        # 2. Tokenise each phase
        phase_tokens = [self._to_tokens(f) for f in phase_feats]
        # 3 × (B, T_sq, embed_dim)

        # 3. Concatenate all phases along token axis
        joint = torch.cat(phase_tokens, dim=1)          # (B, 3*T_sq, embed_dim)

        # 4. Joint self-attention — every token attends to every other
        joint_norm = self.joint_norm(joint)
        attn_out, _ = self.joint_attn(joint_norm, joint_norm, joint_norm)
        joint = joint + attn_out                        # residual
        joint = joint + self.ffn(joint)                 # FFN residual

        # 5. Split back into per-phase streams
        streams = joint.split(T_sq, dim=1)              # 3 × (B, T_sq, embed_dim)

        # 6. Per-phase: pool to global context → broadcast to full resolution
        out_channels = []
        alphas = torch.sigmoid(self.blend)
        for i, stream in enumerate(streams):
            ctx     = stream.mean(dim=1)                # (B, embed_dim)
            spatial = self.broadcast[i](ctx)            # (B, H*W)
            spatial = spatial.view(B, 1, H, W)          # (B, 1, H, W)

            # Residual blend with original phase
            orig    = x[:, i:i+1, :, :]
            blended = alphas[i] * spatial + (1 - alphas[i]) * orig
            out_channels.append(blended)

        # 7. Stack and project to output
        out = torch.cat(out_channels, dim=1)            # (B, 3, H, W)
        return self.out_proj(out)


def build_fusion_module(
    spatial_tokens: int = 8,
    embed_dim:      int = 64,
    num_heads:      int = 4,
    image_size:     int = 256,
) -> DIBFusion:
    """
    Instantiate the DIB module.
    Call .to(device) and include its parameters in the optimiser.
    """
    return DIBFusion(
        spatial_tokens=spatial_tokens,
        embed_dim=embed_dim,
        num_heads=num_heads,
        image_size=image_size,
    )


# ---------------------------------------------------------------------------
# load_fn factory  (identical to F0/F1/F2 — module applied at train time)
# ---------------------------------------------------------------------------

def build_load_fn(image_size: int = 256):
    """
    Load raw stacked phases.  The DIB module is applied at train time,
    not during caching.

    Returns:
        load_fn(patient_root, slice_idx)
            → (img_t (3, H, W),  mask_t (1, H, W))
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
