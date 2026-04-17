"""
models/D1_1_resnet_swin.py
===========================
D1.1 — ResNet-50 (pretrained) + Swin Transformer (from scratch)

CNN encoder:  ResNet-50 via timm, ImageNet pretrained
              out_channels = [256, 512, 1024, 2048]
Transformer:  SwinEncoder from scratch, embed_dim=96
              out_channels = [96, 192, 384, 768]
"""

import torch.nn as nn
import timm
from shared.swin_encoder import SwinEncoder
from shared.model_base import DualEncoderBase, Loss

MODEL_ID: str = "D1_1"


class _ResNet50Encoder(nn.Module):
    """Wrap timm ResNet-50 to expose a 4-stage feature pyramid."""

    out_channels = [256, 512, 1024, 2048]

    def __init__(self, in_channels: int):
        super().__init__()
        self.backbone = timm.create_model(
            "resnet50", pretrained=True, features_only=True,
            out_indices=(1, 2, 3, 4),
            in_chans=in_channels,
        )

    def forward(self, x):
        return self.backbone(x)   # list of 4 feature maps


class D1_1(DualEncoderBase):
    def __init__(self, in_channels=3, img_size=256, **cfg):
        self.cnn_encoder = _ResNet50Encoder(in_channels)
        self.transformer = SwinEncoder(
            in_channels=in_channels,
            embed_dim=96,
            window_size=cfg.get("window_size", 8),
            mlp_ratio=cfg.get("mlp_ratio", 4.0),
            drop_rate=cfg.get("drop_rate", 0.0),
        )
        super().__init__(in_channels, img_size, **cfg)


def build_model(in_channels=3, img_size=256, **cfg) -> nn.Module:
    return D1_1(in_channels=in_channels, img_size=img_size, **cfg)
