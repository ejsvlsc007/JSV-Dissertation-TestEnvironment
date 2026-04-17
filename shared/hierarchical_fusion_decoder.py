"""
shared/hierarchical_fusion_decoder.py
======================================
F4 — Hierarchical multi-scale feature fusion decoder.

This module provides the HierarchicalFusionDecoder that replaces the
standard decoder in DualEncoderBase for F4 model variants.

Instead of fusing the three phases into a single tensor at the input
and then running a standard decoder, F4 keeps the three phase streams
separate through the encoder stages and injects a Phase Fusion Module
(PFM) at each skip connection level before it enters the decoder.

At each decoder stage i:
    skip_fused_i = PFM_i(skip_nc_i, skip_art_i, skip_pvp_i)

where each skip_*_i comes from running the CNN encoder on the
corresponding single-phase input.

Clinical motivation
-------------------
Tumour margins require fine-scale inter-phase fusion (early encoder
stages carry texture/edge detail); lesion localisation benefits from
coarse semantic fusion (deep stages).  Fusing at every scale captures
both simultaneously.

This is implemented as a standalone decoder class so any CNN encoder +
transformer pairing (D0–D7) can use it by swapping the decoder, keeping
the experiment matrix clean.

Usage in model files
--------------------
    class D0_F4(nn.Module):
        def __init__(self, in_channels=3, img_size=256, **cfg):
            self.cnn_encoder  = ...   # same as D0
            self.transformer  = ...   # same as D0
            self.decoder      = HierarchicalFusionDecoder(
                                    cnn_channels=self.cnn_encoder.out_channels,
                                    trans_channels=self.transformer.out_channels,
                                    deep_sup=cfg.get('deep_sup', True),
                                )
        def forward(self, x):
            # x: (B, 3, H, W) — raw stacked phases
            phases = x.split(1, dim=1)   # 3 × (B, 1, H, W)
            ...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Phase Fusion Module (applied at each skip level)
# ---------------------------------------------------------------------------

class PhaseFusionModule(nn.Module):
    """
    Fuses three per-phase feature maps at one scale level.

    Uses a lightweight cross-phase attention: concatenate all three maps,
    apply a channel attention (SE-style) to produce per-channel weights,
    then sum the weighted maps.

    Args:
        in_ch:    channels of each per-phase feature map at this scale.
        out_ch:   output channels after fusion (= in_ch for skip compatibility).
        reduction: SE reduction ratio.
    """

    def __init__(self, in_ch: int, out_ch: int, reduction: int = 4):
        super().__init__()
        # Spatial alignment: each phase map → out_ch
        self.phase_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
            for _ in range(3)
        ])
        # Channel attention over concatenated features (3 × out_ch)
        mid = max(out_ch * 3 // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_ch * 3, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, out_ch * 3),
            nn.Sigmoid(),
        )
        # Output projection
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_ch * 3, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        f_nc:  torch.Tensor,
        f_art: torch.Tensor,
        f_pvp: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            f_nc, f_art, f_pvp: (B, in_ch, H, W)  per-phase feature maps
        Returns:
            (B, out_ch, H, W)  fused feature map
        """
        # Align each phase to out_ch
        p = [conv(f) for conv, f in
             zip(self.phase_convs, [f_nc, f_art, f_pvp])]

        # Concatenate → channel attention → weighted
        cat = torch.cat(p, dim=1)                    # (B, 3*out_ch, H, W)
        w   = self.se(cat).view(cat.shape[0], -1, 1, 1)
        cat = cat * w

        return self.out_conv(cat)


# ---------------------------------------------------------------------------
# Decoder block (same as model_base.DecoderBlock)
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ---------------------------------------------------------------------------
# Hierarchical Fusion Decoder
# ---------------------------------------------------------------------------

class HierarchicalFusionDecoder(nn.Module):
    """
    UNet decoder with per-scale Phase Fusion Modules at each skip level.

    The three per-phase CNN encoders (one per phase) produce skip
    connections at each stage.  Before each decoder block, the three
    per-phase skips are fused by a PFM.  The transformer skip is fused
    with the CNN skips at the bottleneck (same as DualEncoderBase).

    Args:
        cnn_channels:   list of CNN encoder output channels [c1,c2,c3,c4]
        trans_channels: list of transformer output channels [t1,t2,t3,t4]
        deep_sup:       whether to add auxiliary output heads.
    """

    def __init__(
        self,
        cnn_channels:   list[int],
        trans_channels: list[int],
        deep_sup:       bool = True,
    ):
        super().__init__()
        self.deep_sup = deep_sup
        n_stages      = len(cnn_channels)

        # Align transformer → CNN at each stage (for bottleneck fusion)
        self.align = nn.ModuleList([
            nn.Conv2d(tc, cc, 1)
            for tc, cc in zip(trans_channels, cnn_channels)
        ])

        # PFM at each skip level (all except bottleneck)
        self.pfm = nn.ModuleList([
            PhaseFusionModule(cnn_channels[i], cnn_channels[i])
            for i in range(n_stages - 1)
        ])

        # Decoder blocks
        dec_in = cnn_channels[-1]
        self.decoder_blocks = nn.ModuleList()
        self.aux_heads      = nn.ModuleList()
        for i in range(n_stages - 1, 0, -1):
            skip_ch = cnn_channels[i - 1]
            out_ch  = skip_ch
            self.decoder_blocks.append(DecoderBlock(dec_in, skip_ch, out_ch))
            self.aux_heads.append(nn.Conv2d(out_ch, 1, 1))
            dec_in = out_ch

        self.final_up   = nn.ConvTranspose2d(dec_in, dec_in // 2, 2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(dec_in // 2, dec_in // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(dec_in // 2),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(dec_in // 2, 1, 1)

    def forward(
        self,
        phase_cnn_feats: list[list[torch.Tensor]],
        trans_feats:     list[torch.Tensor],
        orig_size:       tuple[int, int],
    ):
        """
        Args:
            phase_cnn_feats: list of 3 encoder outputs, each a list of 4 tensors
                             [[nc_s1..s4], [art_s1..s4], [pvp_s1..s4]]
            trans_feats:     4 transformer feature maps
            orig_size:       (H, W) of the original input

        Returns:
            main_logits  or  [main_logits, aux1, aux2, aux3]
        """
        nc_feats, art_feats, pvp_feats = phase_cnn_feats

        # Fuse transformer into PVP stream at bottleneck (deepest stage)
        n = len(nc_feats)
        tf_aligned = [
            F.interpolate(self.align[i](trans_feats[i]),
                          size=pvp_feats[i].shape[2:],
                          mode="bilinear", align_corners=False)
            for i in range(n)
        ]
        # Bottleneck: use PVP CNN + transformer (PVP is most discriminative)
        bottleneck = pvp_feats[-1] + tf_aligned[-1]

        # Build fused skip connections (PFM at each non-bottleneck level)
        fused_skips = [
            self.pfm[i](nc_feats[i], art_feats[i], pvp_feats[i])
            for i in range(n - 1)
        ]

        # Decode
        x = bottleneck
        aux_logits = []
        for block, aux_head, skip in zip(
            self.decoder_blocks,
            self.aux_heads,
            reversed(fused_skips),
        ):
            x = block(x, skip)
            if self.deep_sup and self.training:
                aux_logits.append(
                    F.interpolate(aux_head(x), size=orig_size,
                                  mode="bilinear", align_corners=False)
                )

        x    = self.final_up(x)
        x    = F.interpolate(x, size=orig_size, mode="bilinear", align_corners=False)
        x    = self.final_conv(x)
        main = self.head(x)

        if self.deep_sup and self.training:
            return [main] + aux_logits
        return main
