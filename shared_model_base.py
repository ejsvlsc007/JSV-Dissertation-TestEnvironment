"""
shared/model_base.py
====================
Shared decoder and DualEncoderBase class.

All D1–D7 model files inherit from DualEncoderBase and only need to:
  1. Define self.cnn_encoder  (pretrained timm backbone)
  2. Define self.transformer  (SwinEncoder or CSwinEncoder)
  3. Call super().__init__() after setting those attributes

The decoder, fusion alignment layers, deep supervision heads,
output head, and Loss class are identical across all variants —
they live here so they are never duplicated.

DualEncoderBase architecture
-----------------------------
  Input (B, C, H, W)
    ├─ CNN encoder      → [c1, c2, c3, c4]  (pretrained timm)
    └─ Transformer enc  → [t1, t2, t3, t4]  (from scratch)
  Align transformer channels to CNN channels via 1×1 convs
  Fuse at each scale: fused_i = cnn_i + align_i(transformer_i)
  UNet decoder with skip connections from fused features
  Output: (B, 1, H, W) logits

The CNN encoder must expose:
    .out_channels: list[int]   — [ch_stage1, ch_stage2, ch_stage3, ch_stage4]
    forward(x) → list[Tensor]  — feature maps at each stage

The transformer must expose:
    .out_channels: list[int]
    forward(x) → list[Tensor]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Decoder building block
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
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
# Loss (shared across all model variants)
# ---------------------------------------------------------------------------

class DualEncoderLoss(nn.Module):
    """
    Dice + BCE blend with optional deep supervision.

    Args:
        dice_weight: blend factor (0 = pure BCE, 1 = pure Dice).
        aux_weight:  weight for each auxiliary deep supervision loss.
        pos_weight:  BCEWithLogitsLoss positive-class weight.
    """

    SMOOTH = 1e-5

    def __init__(
        self,
        dice_weight: float = 0.5,
        aux_weight:  float = 0.4,
        pos_weight:  float = 500.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.aux_weight  = aux_weight
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

    def _single(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce  = self.bce(logits, target)
        pred = torch.sigmoid(logits)
        inter = (pred * target).sum()
        dice = 1 - (2 * inter + self.SMOOTH) / (
            pred.sum() + target.sum() + self.SMOOTH
        )
        return (1 - self.dice_weight) * bce + self.dice_weight * dice

    def forward(self, output, target: torch.Tensor) -> torch.Tensor:
        if isinstance(output, (list, tuple)):
            main = self._single(output[0], target)
            aux  = sum(self._single(o, target) for o in output[1:])
            return main + self.aux_weight * aux
        return self._single(output, target)


# Alias so model files can do:  from shared.model_base import Loss
Loss = DualEncoderLoss


# ---------------------------------------------------------------------------
# DualEncoderBase
# ---------------------------------------------------------------------------

class DualEncoderBase(nn.Module):
    """
    Base class for all D1–D7 dual-encoder segmentation models.

    Subclass usage
    --------------
        class MyModel(DualEncoderBase):
            def __init__(self, in_channels, img_size, **cfg):
                # 1. Build CNN encoder
                self.cnn_encoder = ...     # must have .out_channels list
                # 2. Build transformer encoder
                self.transformer = ...     # must have .out_channels list
                # 3. Call super().__init__()
                super().__init__(in_channels, img_size, **cfg)

    After super().__init__() the following are available:
        self.align          nn.ModuleList of 1×1 alignment convs
        self.decoder_blocks nn.ModuleList of DecoderBlocks
        self.aux_heads      nn.ModuleList of auxiliary output heads
        self.final_up       ConvTranspose2d for last upsampling
        self.final_conv     post-upsampling conv block
        self.head           1×1 output conv
        self.deep_sup       bool
    """

    def __init__(
        self,
        in_channels: int,
        img_size:    int,
        deep_sup:    bool = True,
        **_,
    ):
        super().__init__()
        self.deep_sup = deep_sup

        cnn_ch   = self.cnn_encoder.out_channels    # [c1, c2, c3, c4]
        trans_ch = self.transformer.out_channels    # [t1, t2, t3, t4]
        n_stages = len(cnn_ch)

        # 1×1 convs to align transformer channels → CNN channels
        self.align = nn.ModuleList([
            nn.Conv2d(tc, cc, 1)
            for tc, cc in zip(trans_ch, cnn_ch)
        ])

        # Decoder — built coarse-to-fine
        dec_in = cnn_ch[-1]
        self.decoder_blocks = nn.ModuleList()
        self.aux_heads      = nn.ModuleList()
        for i in range(n_stages - 1, 0, -1):
            skip_ch = cnn_ch[i - 1]
            out_ch  = skip_ch
            self.decoder_blocks.append(DecoderBlock(dec_in, skip_ch, out_ch))
            self.aux_heads.append(nn.Conv2d(out_ch, 1, 1))
            dec_in = out_ch

        # Final upsampling to original resolution
        self.final_up   = nn.ConvTranspose2d(dec_in, dec_in // 2, 2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(dec_in // 2, dec_in // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(dec_in // 2),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(dec_in // 2, 1, 1)

    def forward(self, x: torch.Tensor):
        orig_size = x.shape[2:]

        cnn_feats   = self.cnn_encoder(x)    # [f1, f2, f3, f4] coarse-first
        trans_feats = self.transformer(x)

        # Fuse: align transformer → CNN resolution, then add
        fused = []
        for cf, tf, align in zip(cnn_feats, trans_feats, self.align):
            tf_r = F.interpolate(align(tf), size=cf.shape[2:],
                                 mode="bilinear", align_corners=False)
            fused.append(cf + tf_r)

        # Decode
        x = fused[-1]
        aux_logits = []
        for block, aux_head, skip in zip(
            self.decoder_blocks,
            self.aux_heads,
            reversed(fused[:-1]),
        ):
            x = block(x, skip)
            if self.deep_sup and self.training:
                aux_up = F.interpolate(
                    aux_head(x), size=orig_size,
                    mode="bilinear", align_corners=False,
                )
                aux_logits.append(aux_up)

        x    = self.final_up(x)
        x    = F.interpolate(x, size=orig_size, mode="bilinear", align_corners=False)
        x    = self.final_conv(x)
        main = self.head(x)

        if self.deep_sup and self.training:
            return [main] + aux_logits
        return main
