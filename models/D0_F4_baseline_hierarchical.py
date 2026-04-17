"""
models/D0_F4_baseline_hierarchical.py
=======================================
D0 + F4 — DenseNet-style CNN + Swin Transformer with hierarchical
multi-scale phase fusion.

This model variant uses the HierarchicalFusionDecoder instead of the
standard decoder.  The three phases (NC, ART, PVP) are run through
separate CNN encoder branches and fused at each decoder skip level via
Phase Fusion Modules (PFMs).

The transformer encoder runs on the full early-fused (3-channel) input
and contributes only at the bottleneck — global attention at the deepest
level captures long-range anatomy context while local phase differences
are resolved at each PFM.

Forward pass
------------
    Input x: (B, 3, H, W)  — stacked NC(0) / ART(1) / PVP(2)
    1. Split into 3 × (B, 1, H, W) single-phase tensors
    2. Run each through the shared CNN encoder  → 3 × [s1,s2,s3,s4]
    3. Run full (3-ch) input through transformer → [t1,t2,t3,t4]
    4. HierarchicalFusionDecoder:
         - PFM fuses nc/art/pvp skips at each level
         - Bottleneck = pvp_s4 + aligned(t4)
    5. Output: (B, 1, H, W) logits

Note: sharing CNN encoder weights across all three phases triples the
effective training signal on the encoder while keeping parameter count
the same as D0.  The three phases are processed in a single batched
call (3B instead of B) for efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.D0_baseline import CNNEncoder
from shared.swin_encoder import SwinEncoder
from shared.hierarchical_fusion_decoder import HierarchicalFusionDecoder
from shared.model_base import Loss

MODEL_ID: str = "D0_F4"


class D0_F4(nn.Module):
    def __init__(
        self,
        in_channels:   int   = 3,    # always 3 for F4 (NC+ART+PVP stack)
        img_size:      int   = 256,
        cnn_channels:  int   = 32,
        swin_channels: int   = 24,
        num_layers:    tuple = (4, 4, 4, 4),
        window_size:   int   = 8,
        mlp_ratio:     float = 4.0,
        drop_rate:     float = 0.1,
        deep_sup:      bool  = True,
        **_,
    ):
        super().__init__()

        # Shared CNN encoder — used for all 3 phases (weight sharing)
        self.cnn_encoder = CNNEncoder(
            in_channels=1,          # each phase fed separately as 1-ch
            base_ch=cnn_channels,
            num_layers=num_layers,
        )
        cnn_ch = [cnn_channels * 2 ** (i + 1) for i in range(len(num_layers))]
        self.cnn_encoder.out_channels = cnn_ch

        # Transformer encoder — sees the full 3-channel input
        self.transformer = SwinEncoder(
            in_channels=in_channels,
            embed_dim=swin_channels,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
        )

        # Hierarchical fusion decoder
        self.decoder = HierarchicalFusionDecoder(
            cnn_channels=cnn_ch,
            trans_channels=self.transformer.out_channels,
            deep_sup=deep_sup,
        )

    def forward(self, x: torch.Tensor):
        orig_size = x.shape[2:]

        # Split into single-phase tensors
        nc, art, pvp = x[:, 0:1], x[:, 1:2], x[:, 2:3]

        # Encode all three phases in one batched forward pass
        # Stack along batch dim for efficiency: (3B, 1, H, W)
        phases_batched = torch.cat([nc, art, pvp], dim=0)
        all_feats      = self.cnn_encoder(phases_batched)

        B = x.shape[0]
        # Split back into per-phase feature lists
        phase_cnn_feats = [
            [f[:B], f[B:2*B], f[2*B:]]   # [nc_feats, art_feats, pvp_feats] per stage
            for f in all_feats
        ]
        # Reformat: 3 lists of 4 tensors → list-of-phases, each containing list-of-stages
        nc_feats  = [phase_cnn_feats[s][0] for s in range(len(all_feats))]
        art_feats = [phase_cnn_feats[s][1] for s in range(len(all_feats))]
        pvp_feats = [phase_cnn_feats[s][2] for s in range(len(all_feats))]

        trans_feats = self.transformer(x)

        return self.decoder(
            phase_cnn_feats=[nc_feats, art_feats, pvp_feats],
            trans_feats=trans_feats,
            orig_size=orig_size,
        )


def build_model(in_channels=3, img_size=256, **cfg) -> nn.Module:
    return D0_F4(in_channels=in_channels, img_size=img_size, **cfg)
