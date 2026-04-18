"""
models/D4_2_vgg16_cswin.py
D4_2 — vgg16 (pretrained) + CSwin Transformer (from scratch)

timm vgg16 out_indices=(1,2,3,4) actual channels: [128, 256, 512, 512]
CSwin embed_dim=32 → dims [32,64,128,256]
num_heads=(2,4,8,16): all even, all divide dims ✓
"""

import torch.nn as nn
import timm
from shared.cswin_encoder import CSwinEncoder
from shared.model_base import DualEncoderBase, Loss

MODEL_ID: str = "D4_2"


class _CNNEncoder(nn.Module):
    out_channels = [128, 256, 512, 512]

    def __init__(self, in_channels: int):
        super().__init__()
        self.backbone = timm.create_model(
            "vgg16", pretrained=True, features_only=True,
            out_indices=(1, 2, 3, 4), in_chans=in_channels,
        )

    def forward(self, x):
        return self.backbone(x)


class D4_2(DualEncoderBase):
    def __init__(self, in_channels=3, img_size=256, **cfg):
        nn.Module.__init__(self)
        self.cnn_encoder = _CNNEncoder(in_channels)
        self.transformer = CSwinEncoder(
            in_channels=in_channels,
            embed_dim=32,
            drop_rate=cfg.get("drop_rate", 0.0),
            mlp_ratio=cfg.get("mlp_ratio", 4.0),
            num_heads=(2, 4, 8, 16),
        )
        DualEncoderBase.__init__(self, in_channels, img_size, **cfg)


def build_model(in_channels=3, img_size=256, **cfg) -> nn.Module:
    return D4_2(in_channels=in_channels, img_size=img_size, **cfg)
