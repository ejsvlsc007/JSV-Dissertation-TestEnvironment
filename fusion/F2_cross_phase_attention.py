"""
fusion/F2_cross_phase_attention.py
===================================
Unidirectional cross-phase attention (strategy c2).

Each phase acts as a query that attends to the other two phases
(keys and values).  The attention is applied at the input level
before the encoder sees anything, and the output is projected back
to 3 channels so the model interface is identical to F0/F1.

Attention flow
--------------
    NC  queries {ART, PVP}  → attended NC
    ART queries {NC,  PVP}  → attended ART
    PVP queries {NC,  ART}  → attended PVP
    → concat attended channels → 1×1 projection → (B, 3, H, W)

"Unidirectional" here means each phase attends to the others but the
attention is computed independently per query phase — there is no
simultaneous bidirectional update (that is F3).

Spatial efficiency
------------------
Full spatial self-attention on 256×256 feature maps is O((HW)²) which
is prohibitive.  Instead, attention is computed on a spatially pooled
representation (default 8×8 = 64 tokens per phase) and the resulting
context vector is broadcast back to full resolution via a learned MLP.
This keeps the parameter count low while still capturing global
inter-phase dependencies.

Ablation role
-------------
F0 → F1 → F2 → F3 traces the progression:
  naive stack → gated stack → unidirectional attention → bidirectional attention

Contract
--------
    FUSION_ID    = "F2"
    OUT_CHANNELS = 3
    build_load_fn(image_size) → load_fn(patient_root, slice_idx)
                                → (img_t (3,H,W), mask_t (1,H,W))

The CrossPhaseAttentionFusion nn.Module must be instantiated by the
notebook and its parameters included in the optimizer.
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

FUSION_ID:    str = "F2"
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
# Attention module
# ---------------------------------------------------------------------------

class CrossPhaseAttentionFusion(nn.Module):
    """
    Unidirectional cross-phase attention applied at input level.

    Each phase queries the other two, producing an attended version
    of itself.  All three attended phases are concatenated and
    projected back to 3 output channels.

    Args:
        spatial_tokens: spatial resolution for attention (default 8→64 tokens).
                        Attention is O(tokens²) so keep this small.
        embed_dim:      internal embedding dimension per phase.
        num_heads:      attention heads.
        image_size:     input spatial size (needed for upsampling back).

    Input:  (B, 3, H, W)  — raw stacked phases from the cache
    Output: (B, 3, H, W)  — attended and projected
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

        # Project each phase from 1 channel → embed_dim
        self.phase_proj = nn.ModuleList([
            nn.Conv2d(1, embed_dim, 1) for _ in range(3)
        ])

        # Cross-attention: each phase as query, others as key/value
        # We use 3 separate attention layers — one per query phase
        self.attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            for _ in range(3)
        ])
        self.attn_norm = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(3)
        ])

        # MLP to broadcast attended context back to full resolution
        # Input: embed_dim (pooled context) → image_size × image_size
        self.broadcast = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, image_size * image_size),
            )
            for _ in range(3)
        ])

        # Final projection: 3 attended maps (each 1ch) → 3 output channels
        self.out_proj = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 3, 1),
        )

        # Residual blend scalar (learned, starts at 0 → identity at init)
        self.blend = nn.Parameter(torch.zeros(1))

    def _to_tokens(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Spatially pool feature map to (B, spatial_tokens², embed_dim) tokens.
        feat: (B, embed_dim, H, W)
        """
        B, C, H, W = feat.shape
        s = self.spatial_tokens
        pooled = F.adaptive_avg_pool2d(feat, (s, s))   # (B, C, s, s)
        return pooled.flatten(2).transpose(1, 2)        # (B, s*s, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) — stacked NC(0) / ART(1) / PVP(2)
        Returns:
            (B, 3, H, W) — attended output, same spatial size
        """
        B, _, H, W = x.shape

        # Project each phase to embed_dim feature maps
        phase_feats = [
            self.phase_proj[i](x[:, i:i+1, :, :])   # (B, embed_dim, H, W)
            for i in range(3)
        ]

        # Tokenise
        phase_tokens = [self._to_tokens(f) for f in phase_feats]  # 3 × (B, T, C)

        attended = []
        for i in range(3):
            # Query: phase i — Keys/Values: the other two phases concatenated
            query  = phase_tokens[i]                                      # (B, T, C)
            others = torch.cat([phase_tokens[j] for j in range(3) if j != i], dim=1)  # (B, 2T, C)

            q_norm = self.attn_norm[i](query)
            o_norm = self.attn_norm[i](others)   # reuse same norm (cross-norm)
            ctx, _ = self.attn[i](q_norm, o_norm, o_norm)               # (B, T, C)
            ctx    = (query + ctx).mean(dim=1)                           # (B, C) global context

            # Broadcast context to full spatial resolution
            spatial = self.broadcast[i](ctx)                             # (B, H*W)
            spatial = spatial.view(B, 1, H, W)                          # (B, 1, H, W)
            attended.append(spatial)

        attended_stack = torch.cat(attended, dim=1)   # (B, 3, H, W)
        out = self.out_proj(attended_stack)            # (B, 3, H, W)

        # Residual: blend learned output with original input
        alpha = torch.sigmoid(self.blend)
        return alpha * out + (1 - alpha) * x


def build_fusion_module(
    spatial_tokens: int = 8,
    embed_dim:      int = 64,
    num_heads:      int = 4,
    image_size:     int = 256,
) -> CrossPhaseAttentionFusion:
    """
    Instantiate the attention module.
    Call .to(device) and include its parameters in the optimiser.
    """
    return CrossPhaseAttentionFusion(
        spatial_tokens=spatial_tokens,
        embed_dim=embed_dim,
        num_heads=num_heads,
        image_size=image_size,
    )


# ---------------------------------------------------------------------------
# load_fn factory  (identical to F0/F1 — attention applied at train time)
# ---------------------------------------------------------------------------

def build_load_fn(image_size: int = 256):
    """
    Load raw stacked phases.  Attention is applied at train time via
    the CrossPhaseAttentionFusion module, not during caching.

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
