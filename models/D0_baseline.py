"""
models/D0_baseline.py
=====================
D0 baseline model: CaT-Net — Dual Encoder (CNN + Swin Transformer)
with UNet-style decoder and optional deep supervision.

Required contract  (same as every model module)
-----------------
    MODEL_ID:  str
    build_model(in_channels, img_size, **cfg) → nn.Module
    Loss(**cfg)                                → nn.Module

The notebook reads MODEL_ID and calls build_model() / Loss() without
needing to know anything else about the architecture.

Architecture overview
---------------------
    Input (B, C, H, W)
        │
        ├─ CNN Encoder      — DenseNet-style blocks, 4 stages
        │                     progressively halves spatial resolution
        │
        └─ Swin Encoder     — Swin Transformer, patch-based attention
                              matches CNN encoder's 4-stage resolution pyramid
        │
    Fusion at each scale    — element-wise addition of CNN + Swin feature maps
        │
    UNet Decoder            — skip connections from fused encoder features,
                              4× upsampling stages back to original resolution
        │
    Output head             — 1×1 conv → (B, 1, H, W) logits
        │  (optional)
    Deep supervision heads  — auxiliary logits at each decoder stage
                              (only used during training)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_ID: str = "D0"


# ---------------------------------------------------------------------------
# CNN Encoder building blocks
# ---------------------------------------------------------------------------

class DenseLayer(nn.Module):
    def __init__(self, in_ch: int, growth: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, growth, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return torch.cat([x, self.conv(x)], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_ch: int, growth: int, n_layers: int):
        super().__init__()
        layers, ch = [], in_ch
        for _ in range(n_layers):
            layers.append(DenseLayer(ch, growth))
            ch += growth
        self.block   = nn.Sequential(*layers)
        self.out_ch  = ch
        # 1×1 bottleneck to keep channel count manageable
        self.squeeze = nn.Conv2d(ch, in_ch * 2, 1, bias=False)

    def forward(self, x):
        return self.squeeze(self.block(x))


class CNNEncoder(nn.Module):
    """4-stage DenseNet encoder.  Returns feature maps at each stage."""

    def __init__(self, in_channels: int, base_ch: int, num_layers: tuple[int, ...]):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, base_ch, 3, padding=1, bias=False)

        self.stages    = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base_ch
        for n in num_layers:
            block = DenseBlock(ch, ch, n)
            self.stages.append(block)
            out_ch = block.out_ch if hasattr(block, "out_ch") else ch * 2
            # Use squeeze output channels
            out_ch = ch * 2
            self.downsamples.append(nn.MaxPool2d(2))
            ch = out_ch

        self.out_channels = [base_ch * 2 ** (i + 1) for i in range(len(num_layers))]

    def forward(self, x):
        feats = []
        x = self.stem(x)
        for stage, down in zip(self.stages, self.downsamples):
            x = stage(x)
            feats.append(x)
            x = down(x)
        return feats   # list of feature maps, coarsest last


# ---------------------------------------------------------------------------
# Swin Transformer Encoder
# ---------------------------------------------------------------------------

class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.ws   = window_size
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.ws
        # Pad if necessary
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, Hp, Wp = x.shape

        # Partition into windows
        x = x.view(B, C, Hp // ws, ws, Wp // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()   # B, nH, nW, ws, ws, C
        nH, nW = Hp // ws, Wp // ws
        x = x.view(B * nH * nW, ws * ws, C)

        # Self-attention within each window
        x_norm = self.norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Reverse partition
        x = x.view(B, nH, nW, ws, ws, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, C, Hp, Wp)

        # Remove padding
        if pad_h or pad_w:
            x = x[:, :, :H, :W]
        return x


class SwinStage(nn.Module):
    def __init__(self, in_ch: int, dim: int, window_size: int,
                 num_heads: int, mlp_ratio: float, drop: float):
        super().__init__()
        self.proj    = nn.Conv2d(in_ch, dim, 1)
        self.attn    = WindowAttention(dim, window_size, num_heads)
        self.norm1   = nn.LayerNorm(dim)
        self.norm2   = nn.LayerNorm(dim)
        self.mlp     = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )
        self.down    = nn.Conv2d(dim, dim * 2, 3, stride=2, padding=1)
        self.out_dim = dim * 2

    def forward(self, x):
        x = self.proj(x)
        # Attention
        B, C, H, W = x.shape
        residual = x
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_flat = self.norm1(x_flat)
        x = x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = self.attn(x) + residual
        # MLP
        B, C, H, W = x.shape
        residual = x
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_flat = self.norm2(x_flat)
        x_flat = self.mlp(x_flat)
        x = x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2) + residual
        feat = x
        x    = self.down(x)
        return x, feat


class SwinEncoder(nn.Module):
    """4-stage Swin encoder, returns feature maps matching CNN encoder stages."""

    def __init__(self, in_channels: int, embed_dim: int, window_size: int,
                 n_stages: int, mlp_ratio: float, drop: float):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, 4, stride=4)
        self.stages = nn.ModuleList()
        dim = embed_dim
        for _ in range(n_stages):
            heads = max(1, dim // 32)
            self.stages.append(SwinStage(dim, dim, window_size, heads, mlp_ratio, drop))
            dim *= 2
        self.out_channels = [embed_dim * 2 ** i for i in range(n_stages)]

    def forward(self, x):
        x = self.patch_embed(x)
        feats = []
        for stage in self.stages:
            x, feat = stage(x)
            feats.append(feat)
        return feats


# ---------------------------------------------------------------------------
# Decoder
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

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch from padding
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class CaT_Net_with_Decoder_DeepSup(nn.Module):
    """
    CaT-Net: CNN + Swin dual encoder with UNet decoder and deep supervision.

    Forward output (training):
        [main_logits, aux1, aux2, aux3]   when deep_sup=True
    Forward output (inference):
        main_logits                       when deep_sup=False or model.eval()

    All outputs are raw logits — apply torch.sigmoid() before thresholding.
    """

    def __init__(
        self,
        in_channels:  int   = 3,
        num_classes:  int   = 1,
        img_size:     int   = 256,
        cnn_channels: int   = 32,
        swin_channels: int  = 24,
        num_layers:   tuple = (4, 4, 4, 4),
        window_size:  int   = 8,
        mlp_ratio:    float = 4.0,
        drop_rate:    float = 0.1,
        deep_sup:     bool  = True,
    ):
        super().__init__()
        self.deep_sup = deep_sup
        n_stages      = len(num_layers)

        self.cnn_encoder  = CNNEncoder(in_channels, cnn_channels, num_layers)
        self.swin_encoder = SwinEncoder(
            in_channels, swin_channels, window_size,
            n_stages, mlp_ratio, drop_rate,
        )

        # Fusion: align Swin channels to CNN channels with 1×1 convs
        cnn_ch  = [cnn_channels * 2 ** (i + 1) for i in range(n_stages)]
        swin_ch = [swin_channels * 2 **  i      for i in range(n_stages)]
        self.align = nn.ModuleList([
            nn.Conv2d(sc, cc, 1) for sc, cc in zip(swin_ch, cnn_ch)
        ])

        # Decoder — built in reverse stage order
        dec_in  = cnn_ch[-1]
        self.decoder_blocks = nn.ModuleList()
        self.aux_heads      = nn.ModuleList()
        for i in range(n_stages - 1, 0, -1):
            skip_ch = cnn_ch[i - 1]
            out_ch  = skip_ch
            self.decoder_blocks.append(DecoderBlock(dec_in, skip_ch, out_ch))
            self.aux_heads.append(nn.Conv2d(out_ch, num_classes, 1))
            dec_in = out_ch

        # Final upsampling to original resolution + output head
        self.final_up   = nn.ConvTranspose2d(dec_in, dec_in // 2, 2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(dec_in // 2, dec_in // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(dec_in // 2),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(dec_in // 2, num_classes, 1)

    def forward(self, x):
        orig_size = x.shape[2:]

        cnn_feats  = self.cnn_encoder(x)   # coarse to fine: [f1, f2, f3, f4]
        swin_feats = self.swin_encoder(x)

        # Fuse: align Swin maps to CNN resolution and add
        fused = []
        for cf, sf, align in zip(cnn_feats, swin_feats, self.align):
            sf_r = F.interpolate(align(sf), size=cf.shape[2:],
                                 mode="bilinear", align_corners=False)
            fused.append(cf + sf_r)

        # Decode (skip connections from fused features, reverse order)
        x = fused[-1]
        aux_logits = []
        for block, aux, skip in zip(
            self.decoder_blocks, self.aux_heads,
            reversed(fused[:-1])
        ):
            x = block(x, skip)
            if self.deep_sup and self.training:
                aux_up = F.interpolate(aux(x), size=orig_size,
                                       mode="bilinear", align_corners=False)
                aux_logits.append(aux_up)

        x = self.final_up(x)
        x = F.interpolate(x, size=orig_size, mode="bilinear", align_corners=False)
        x = self.final_conv(x)
        main = self.head(x)

        if self.deep_sup and self.training:
            return [main] + aux_logits
        return main


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class Loss(nn.Module):
    """
    Dice + BCE blend for the main output, with auxiliary deep-supervision losses.

    Args:
        dice_weight: weight of Dice loss vs BCE (0 = pure BCE, 1 = pure Dice).
        aux_weight:  weight applied to each auxiliary loss.
        pos_weight:  BCEWithLogitsLoss positive class weight (handles imbalance).
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

    def _single_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, target)

        pred  = torch.sigmoid(logits)
        inter = (pred * target).sum()
        dice_loss = 1 - (2 * inter + self.SMOOTH) / (
            pred.sum() + target.sum() + self.SMOOTH
        )

        return (1 - self.dice_weight) * bce_loss + self.dice_weight * dice_loss

    def forward(self, output, target: torch.Tensor) -> torch.Tensor:
        if isinstance(output, (list, tuple)):
            main_loss = self._single_loss(output[0], target)
            aux_loss  = sum(self._single_loss(o, target) for o in output[1:])
            return main_loss + self.aux_weight * aux_loss
        return self._single_loss(output, target)


# ---------------------------------------------------------------------------
# Public factory (called by the notebook)
# ---------------------------------------------------------------------------

def build_model(
    in_channels:   int   = 3,
    img_size:      int   = 256,
    cnn_channels:  int   = 32,
    swin_channels: int   = 24,
    num_layers:    tuple = (4, 4, 4, 4),
    window_size:   int   = 8,
    mlp_ratio:     float = 4.0,
    drop_rate:     float = 0.1,
    deep_sup:      bool  = True,
    **_,           # absorb unknown kwargs for forward-compatibility
) -> nn.Module:
    """Instantiate and return the D0 model."""
    return CaT_Net_with_Decoder_DeepSup(
        in_channels   = in_channels,
        num_classes   = 1,
        img_size      = img_size,
        cnn_channels  = cnn_channels,
        swin_channels = swin_channels,
        num_layers    = num_layers,
        window_size   = window_size,
        mlp_ratio     = mlp_ratio,
        drop_rate     = drop_rate,
        deep_sup      = deep_sup,
    )
