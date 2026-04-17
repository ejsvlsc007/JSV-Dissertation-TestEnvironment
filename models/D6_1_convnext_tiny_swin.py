"""
models/D6_1_convnext_tiny_swin.py
===================================
D6.1 — ConvNeXt-Tiny (pretrained) + Swin Transformer (from scratch)

CNN encoder:  ConvNeXt-Tiny via timm, ImageNet pretrained
              out_channels = [96, 192, 384, 768]
Transformer:  SwinEncoder from scratch, embed_dim=96
"""

import torch.nn as nn
import timm
from shared.swin_encoder import SwinEncoder
from shared.model_base import DualEncoderBase, Loss

MODEL_ID: str = "D6_1"


class _ConvNeXtTinyEncoder(nn.Module):
    out_channels = [96, 192, 384, 768]

    def __init__(self, in_channels: int):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_tiny", pretrained=True, features_only=True,
            out_indices=(0, 1, 2, 3), in_chans=in_channels,
        )

    def forward(self, x):
        return self.backbone(x)


class D6_1(DualEncoderBase):
    def __init__(self, in_channels=3, img_size=256, **cfg):
        self.cnn_encoder = _ConvNeXtTinyEncoder(in_channels)
        self.transformer = SwinEncoder(
            in_channels=in_channels,
            embed_dim=96,
            window_size=cfg.get("window_size", 8),
            mlp_ratio=cfg.get("mlp_ratio", 4.0),
            drop_rate=cfg.get("drop_rate", 0.0),
        )
        super().__init__(in_channels, img_size, **cfg)


def build_model(in_channels=3, img_size=256, **cfg) -> nn.Module:
    return D6_1(in_channels=in_channels, img_size=img_size, **cfg)
