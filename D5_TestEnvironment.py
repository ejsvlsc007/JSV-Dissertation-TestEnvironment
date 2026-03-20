"""
DECTNet D5 — EfficientNet-Lite4 + Swin Transformer Dual Encoder
================================================================
Architecture:
  - CNN Encoder  : EfficientNet-Lite4 (timm), no SE blocks, ReLU6 activation
  - Transformer  : Swin-Tiny (timm), hierarchical window attention
  - Fusion       : Channel-wise concatenation + 1x1 projection at each scale
  - Decoder      : Progressive upsampling with conv refinement blocks
  - Head         : 1x1 conv → sigmoid (binary liver segmentation)

Key difference from D3/D4:
  EfficientNet-Lite4 removes Squeeze-and-Excitation (SE) blocks and replaces
  Swish activation with ReLU6. This significantly reduces parameter count and
  removes channel recalibration — useful as a lightweight ablation to isolate
  the contribution of SE-based channel attention in the CNN encoder branch.

MODEL_ID: DN5
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
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class FusionBlock(nn.Module):
    """Align spatial dims, concat CNN + transformer features, project to out_ch."""
    def __init__(self, cnn_ch, tr_ch, out_ch):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(cnn_ch + tr_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, cnn_feat, tr_feat):
        if cnn_feat.shape[2:] != tr_feat.shape[2:]:
            tr_feat = F.interpolate(tr_feat, size=cnn_feat.shape[2:],
                                    mode='bilinear', align_corners=False)
        return self.proj(torch.cat([cnn_feat, tr_feat], dim=1))


# ---------------------------------------------------------------------------
# CNN Encoder — EfficientNet-Lite4
# ---------------------------------------------------------------------------

class EfficientNetLite4Encoder(nn.Module):
    """
    EfficientNet-Lite4 multi-scale feature extractor.
    Returns 4 feature maps at strides 4, 8, 16, 32 relative to input.

    No Squeeze-and-Excitation blocks. ReLU6 activation throughout.

    Channel dims (EfficientNet-Lite4):
        s1 →  32  (stride 4)
        s2 →  56  (stride 8)
        s3 → 160  (stride 16)
        s4 → 448  (stride 32)

    Note: channel dims identical to B4 but SE is absent — the encoder
    has no internal channel recalibration, making it a clean ablation.

    Pretrained weights for efficientnet_lite4 were removed from the timm
    registry in newer releases. If pretrained=True and loading fails, this
    encoder falls back to random initialisation and emits a warning so the
    caller is aware. All other architecture details remain unchanged.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            try:
                self.backbone = timm.create_model(
                    'efficientnet_lite4',
                    pretrained=True,
                    features_only=True,
                    out_indices=(1, 2, 3, 4),
                )
                print('[D5] EfficientNet-Lite4: pretrained weights loaded.')
            except RuntimeError:
                import warnings
                warnings.warn(
                    '[D5] efficientnet_lite4 pretrained weights are not available in this '
                    'timm version. Falling back to random initialisation. '
                    'Set pretrained=False to suppress this warning.',
                    UserWarning, stacklevel=2,
                )
                self.backbone = timm.create_model(
                    'efficientnet_lite4',
                    pretrained=False,
                    features_only=True,
                    out_indices=(1, 2, 3, 4),
                )
        else:
            self.backbone = timm.create_model(
                'efficientnet_lite4',
                pretrained=False,
                features_only=True,
                out_indices=(1, 2, 3, 4),
            )
        self.out_channels = self.backbone.feature_info.channels()

    def forward(self, x):
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Transformer Encoder — Swin-Tiny
# ---------------------------------------------------------------------------

class SwinEncoder(nn.Module):
    """
    Swin-Tiny hierarchical encoder.
    Channel dims: [96, 192, 384, 768] at strides [4, 8, 16, 32].
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
        self.out_channels = self.backbone.feature_info.channels()

    def forward(self, x):
        feats = self.backbone(x)
        out = []
        for f in feats:
            if f.dim() == 4 and f.shape[-1] != f.shape[1]:
                f = f.permute(0, 3, 1, 2).contiguous()
            out.append(f)
        return out


# ---------------------------------------------------------------------------
# DECTNet D5 — Full model
# ---------------------------------------------------------------------------

class DECTNetD5(nn.Module):
    """
    DECTNet D5: EfficientNet-Lite4 CNN encoder + Swin-Tiny transformer encoder.

    Lite4 removes SE channel attention and Swish activations compared to B4 (D3),
    providing a lightweight ablation of the CNN encoder branch. Lower total param
    count makes this useful for compute-constrained benchmarking comparisons.

    Args:
        num_classes : output channels (1 for binary segmentation)
        pretrained  : load ImageNet weights for both encoders
        img_size    : expected input spatial size (used by Swin)
        deep_sup    : if True, return list [main, *aux] during training
    """
    MODEL_ID = 'DN5'

    FUSED_CHS = [128, 256, 256, 512]

    def __init__(self, num_classes=1, pretrained=True, img_size=224, deep_sup=False):
        super().__init__()
        self.deep_sup = deep_sup

        # --- Encoders ---
        self.cnn_enc = EfficientNetLite4Encoder(pretrained=pretrained)
        self.tr_enc  = SwinEncoder(pretrained=pretrained, img_size=img_size)

        cnn_chs = self.cnn_enc.out_channels   # [32, 56, 160, 448]
        tr_chs  = self.tr_enc.out_channels    # [96, 192, 384, 768]
        fc      = self.FUSED_CHS

        # --- Fusion ---
        self.fuse = nn.ModuleList([
            FusionBlock(cnn_chs[i], tr_chs[i], fc[i]) for i in range(4)
        ])

        # --- Decoder ---
        self.dec3 = DecoderBlock(fc[3], fc[2], 256)
        self.dec2 = DecoderBlock(256,   fc[1], 128)
        self.dec1 = DecoderBlock(128,   fc[0],  64)
        self.dec0 = DecoderBlock(64,    0,       32)

        # --- Heads ---
        self.head = nn.Conv2d(32, num_classes, 1)

        if deep_sup:
            self.aux3 = nn.Conv2d(256, num_classes, 1)
            self.aux2 = nn.Conv2d(128, num_classes, 1)
            self.aux1 = nn.Conv2d(64,  num_classes, 1)

    def forward(self, x):
        cnn_feats = self.cnn_enc(x)
        tr_feats  = self.tr_enc(x)

        f = [self.fuse[i](cnn_feats[i], tr_feats[i]) for i in range(4)]

        d3 = self.dec3(f[3], f[2])
        d2 = self.dec2(d3,   f[1])
        d1 = self.dec1(d2,   f[0])
        d0 = self.dec0(d1)

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
    model  = DECTNetD5(num_classes=1, pretrained=False, img_size=224).to(device)
    x      = torch.randn(2, 3, 224, 224).to(device)
    out    = model(x)
    print(f'[{DECTNetD5.MODEL_ID}] Input: {tuple(x.shape)}  Output: {tuple(out.shape)}')
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[{DECTNetD5.MODEL_ID}] Trainable params: {params:,}')
