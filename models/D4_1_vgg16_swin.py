"""
models/D4_1_vgg16_swin.py
D4_1 — vgg16 (pretrained) + Swin Transformer (from scratch)

timm vgg16 out_indices=(1,2,3,4) actual channels: [128, 256, 512, 512]
Swin embed_dim=64 → dims [64,128,256,512]
num_heads=(8,16,32,32): 64/8=8 ✓ 128/16=8 ✓ 256/32=8 ✓ 512/32=16 ✓
"""

import torch.nn as nn
import timm
from shared.swin_encoder import SwinEncoder
from shared.model_base import DualEncoderBase, Loss

MODEL_ID: str = "D4_1"


class _CNNEncoder(nn.Module):
    out_channels = [128, 256, 512, 512]   # actual timm vgg16 outputs

    def __init__(self, in_channels: int):
        super().__init__()
        self.backbone = timm.create_model(
            "vgg16", pretrained=True, features_only=True,
            out_indices=(1, 2, 3, 4), in_chans=in_channels,
        )

    def forward(self, x):
        return self.backbone(x)


class D4_1(DualEncoderBase):
    def __init__(self, in_channels=3, img_size=256, **cfg):
        nn.Module.__init__(self)
        self.cnn_encoder = _CNNEncoder(in_channels)
        self.transformer = SwinEncoder(
            in_channels=in_channels,
            embed_dim=64,
            window_size=cfg.get("window_size", 8),
            mlp_ratio=cfg.get("mlp_ratio", 4.0),
            drop_rate=cfg.get("drop_rate", 0.0),
            num_heads=(8, 16, 32, 32),
        )
        DualEncoderBase.__init__(self, in_channels, img_size, **cfg)


def build_model(in_channels=3, img_size=256, **cfg) -> nn.Module:
    return D4_1(in_channels=in_channels, img_size=img_size, **cfg)
