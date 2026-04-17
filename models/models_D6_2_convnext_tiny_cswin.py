"""
models/D6_2_convnext_tiny_cswin.py
====================================
D6.2 — ConvNeXt-Tiny (pretrained) + CSwin Transformer (from scratch)
"""

import torch.nn as nn
import timm
from shared.cswin_encoder import CSwinEncoder
from shared.model_base import DualEncoderBase, Loss

MODEL_ID: str = "D6_2"


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


class D6_2(DualEncoderBase):
    def __init__(self, in_channels=3, img_size=256, **cfg):
        self.cnn_encoder = _ConvNeXtTinyEncoder(in_channels)
        self.transformer = CSwinEncoder(
            in_channels=in_channels,
            embed_dim=32,
            drop_rate=cfg.get("drop_rate", 0.0),
            mlp_ratio=cfg.get("mlp_ratio", 4.0),
        )
        super().__init__(in_channels, img_size, **cfg)


def build_model(in_channels=3, img_size=256, **cfg) -> nn.Module:
    return D6_2(in_channels=in_channels, img_size=img_size, **cfg)
