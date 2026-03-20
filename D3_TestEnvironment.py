"""
DECTNet D3 — EfficientNet-B4 + Swin Transformer Dual Encoder
=============================================================
Architecture:
  - CNN Encoder  : EfficientNet-B4 (timm), stages 1-4 as skip sources
  - Transformer  : Swin-Tiny (timm), hierarchical window attention
  - Fusion       : Channel-wise concatenation at each scale
  - Decoder      : Progressive upsampling with conv refinement blocks
  - Head         : 1x1 conv → sigmoid (binary liver segmentation)

MODEL_ID: DN3
Compatible with JSV unified CSV logging schema.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, pad=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    """Upsample → concat skip → two conv-bn-relu refinement convs."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            # Pad if spatial dims differ by 1 pixel (odd input sizes)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# CNN Encoder — EfficientNet-B4
# ---------------------------------------------------------------------------

class EfficientNetB4Encoder(nn.Module):
    """
    Extracts multi-scale features from EfficientNet-B4.
    Returns 4 feature maps at strides 4, 8, 16, 32 relative to input.

    Channel dims (EfficientNet-B4 defaults):
        s1 →  32  (stride 4)
        s2 →  56  (stride 8)
        s3 → 160  (stride 16)
        s4 → 448  (stride 32)
    """
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4),   # strides 4, 8, 16, 32
        )
        self.backbone = backbone
        self.out_channels = backbone.feature_info.channels()  # [32, 56, 160, 448]

    def forward(self, x):
        return self.backbone(x)   # list of 4 tensors


# ---------------------------------------------------------------------------
# Transformer Encoder — Swin-Tiny
# ---------------------------------------------------------------------------

class SwinEncoder(nn.Module):
    """
    Swin-Tiny hierarchical encoder.
    Returns 4 feature maps matching the EfficientNet stride schedule.

    Channel dims (Swin-Tiny):
        s1 →  96  (stride 4)
        s2 → 192  (stride 8)
        s3 → 384  (stride 16)
        s4 → 768  (stride 32)
    """
    def __init__(self, pretrained=True, img_size=224):
        super().__init__()
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=img_size,
        )
        self.out_channels = self.backbone.feature_info.channels()  # [96, 192, 384, 768]

    def forward(self, x):
        feats = self.backbone(x)
        # Swin outputs (B, H, W, C) at each stage — permute to (B, C, H, W)
        out = []
        for f in feats:
            if f.dim() == 4 and f.shape[-1] != f.shape[1]:
                f = f.permute(0, 3, 1, 2).contiguous()
            out.append(f)
        return out


# ---------------------------------------------------------------------------
# Fusion module — channel concat + 1x1 projection
# ---------------------------------------------------------------------------

class FusionBlock(nn.Module):
    def __init__(self, cnn_ch, tr_ch, out_ch):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(cnn_ch + tr_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, cnn_feat, tr_feat):
        # Align spatial dims if needed
        if cnn_feat.shape[2:] != tr_feat.shape[2:]:
            tr_feat = F.interpolate(tr_feat, size=cnn_feat.shape[2:],
                                    mode='bilinear', align_corners=False)
        return self.proj(torch.cat([cnn_feat, tr_feat], dim=1))


# ---------------------------------------------------------------------------
# DECTNet D3 — Full model
# ---------------------------------------------------------------------------

class DECTNetD3(nn.Module):
    """
    DECTNet D3: EfficientNet-B4 CNN encoder + Swin-Tiny transformer encoder.

    Dual-encoder fusion at 4 scales, followed by a 4-stage UNet-style decoder.

    Args:
        num_classes  : output channels (1 for binary segmentation)
        pretrained   : load ImageNet weights for both encoders
        img_size     : expected input spatial size (used by Swin)
        deep_sup     : if True, return list [main, *aux] for deep supervision
    """
    MODEL_ID = 'DN3'

    # Fused channel dims after projection at each scale
    FUSED_CHS = [128, 256, 256, 512]

    def __init__(self, num_classes=1, pretrained=True, img_size=224, deep_sup=False):
        super().__init__()
        self.deep_sup = deep_sup

        # --- Encoders ---
        self.cnn_enc = EfficientNetB4Encoder(pretrained=pretrained)
        self.tr_enc  = SwinEncoder(pretrained=pretrained, img_size=img_size)

        cnn_chs = self.cnn_enc.out_channels   # [32, 56, 160, 448]
        tr_chs  = self.tr_enc.out_channels    # [96, 192, 384, 768]
        fc      = self.FUSED_CHS              # [128, 256, 256, 512]

        # --- Fusion blocks (one per scale) ---
        self.fuse = nn.ModuleList([
            FusionBlock(cnn_chs[i], tr_chs[i], fc[i]) for i in range(4)
        ])

        # --- Decoder ---
        # Stage 4→3: in=fc[3], skip=fc[2]
        self.dec3 = DecoderBlock(fc[3], fc[2], 256)
        # Stage 3→2: in=256, skip=fc[1]
        self.dec2 = DecoderBlock(256,   fc[1], 128)
        # Stage 2→1: in=128, skip=fc[0]
        self.dec1 = DecoderBlock(128,   fc[0],  64)
        # Stage 1→input: no skip (above stride-4 features)
        self.dec0 = DecoderBlock(64,    0,       32)

        # --- Segmentation heads ---
        self.head = nn.Conv2d(32, num_classes, 1)

        if deep_sup:
            self.aux3 = nn.Conv2d(256, num_classes, 1)
            self.aux2 = nn.Conv2d(128, num_classes, 1)
            self.aux1 = nn.Conv2d(64,  num_classes, 1)

    def forward(self, x):
        # Dual encoding
        cnn_feats = self.cnn_enc(x)   # [s1, s2, s3, s4]
        tr_feats  = self.tr_enc(x)    # [s1, s2, s3, s4]

        # Scale-wise fusion
        f = [self.fuse[i](cnn_feats[i], tr_feats[i]) for i in range(4)]
        # f[0]=stride4, f[1]=stride8, f[2]=stride16, f[3]=stride32

        # Decoder (bottom-up)
        d3 = self.dec3(f[3], f[2])   # → stride16 space, 256 ch
        d2 = self.dec2(d3,   f[1])   # → stride8  space, 128 ch
        d1 = self.dec1(d2,   f[0])   # → stride4  space,  64 ch
        d0 = self.dec0(d1)            # → stride2  space,  32 ch

        # Final upsample to input resolution
        out = F.interpolate(self.head(d0), size=x.shape[2:],
                            mode='bilinear', align_corners=False)

        if self.deep_sup and self.training:
            a3 = F.interpolate(self.aux3(d3), size=x.shape[2:],
                               mode='bilinear', align_corners=False)
            a2 = F.interpolate(self.aux2(d2), size=x.shape[2:],
                               mode='bilinear', align_corners=False)
            a1 = F.interpolate(self.aux1(d1), size=x.shape[2:],
                               mode='bilinear', align_corners=False)
            return [out, a3, a2, a1]

        return out


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = DECTNetD3(num_classes=1, pretrained=False, img_size=224).to(device)
    x      = torch.randn(2, 3, 224, 224).to(device)
    out    = model(x)
    print(f'[{DECTNetD3.MODEL_ID}] Input: {tuple(x.shape)}  Output: {tuple(out.shape)}')
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[{DECTNetD3.MODEL_ID}] Trainable params: {params:,}')
