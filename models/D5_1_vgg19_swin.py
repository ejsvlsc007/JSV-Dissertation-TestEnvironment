"""
models/D5_1_vgg19_swin.py
==========================
D5.1 — VGG-19 (pretrained) + Swin Transformer (from scratch)

CNN encoder:  VGG-19 via timm, ImageNet pretrained
              out_channels = [64, 128, 256, 512]
Transformer:  SwinEncoder from scratch, embed_dim=64
"""

import torch.nn as nn
import timm
from shared.swin_encoder import SwinEncoder
from shared.model_base import DualEncoderBase, Loss

MODEL_ID: str = "D5_1"


class _VGG19Encoder(nn.Module):
    out_channels = [64, 128, 256, 512]

    def __init__(self, in_channels: int):
        super().__init__()
        self.backbone = timm.create_model(
            "vgg19", pretrained=True, features_only=True,
            out_indices=(1, 2, 3, 4), in_chans=in_channels,
        )

    def forward(self, x):
        return self.backbone(x)


class D5_1(DualEncoderBase):
    def __init__(self, in_channels=3, img_size=256, **cfg):
        self.cnn_encoder = _VGG19Encoder(in_channels)
        self.transformer = SwinEncoder(
            in_channels=in_channels,
            embed_dim=64,
            window_size=cfg.get("window_size", 8),
            mlp_ratio=cfg.get("mlp_ratio", 4.0),
            drop_rate=cfg.get("drop_rate", 0.0),
        )
        super().__init__(in_channels, img_size, **cfg)


def build_model(in_channels=3, img_size=256, **cfg) -> nn.Module:
    return D5_1(in_channels=in_channels, img_size=img_size, **cfg)
