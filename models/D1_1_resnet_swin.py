"""
models/D1_1_resnet_swin.py
D1_1 — resnet50 (pretrained) + Swin Transformer (from scratch)
"""

import torch.nn as nn
import timm
from shared.swin_encoder import SwinEncoder
from shared.model_base import DualEncoderBase, Loss

MODEL_ID: str = "D1_1"


class _CNNEncoder(nn.Module):
    out_channels = [256, 512, 1024, 2048]

    def __init__(self, in_channels: int):
        super().__init__()
        self.backbone = timm.create_model(
            "resnet50", pretrained=True, features_only=True,
            out_indices=(1,2,3,4), in_chans=in_channels,
        )

    def forward(self, x):
        return self.backbone(x)


class D1_1(DualEncoderBase):
    def __init__(self, in_channels=3, img_size=256, **cfg):
        nn.Module.__init__(self)
        self.cnn_encoder = _CNNEncoder(in_channels)
        self.transformer = SwinEncoder(
            in_channels=in_channels,
            embed_dim=96,
            window_size=cfg.get('window_size', 8),
            mlp_ratio=cfg.get('mlp_ratio', 4.0),
            drop_rate=cfg.get('drop_rate', 0.0),
        )
        DualEncoderBase.__init__(self, in_channels, img_size, **cfg)


def build_model(in_channels=3, img_size=256, **cfg) -> nn.Module:
    return D1_1(in_channels=in_channels, img_size=img_size, **cfg)
