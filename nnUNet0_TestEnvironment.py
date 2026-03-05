"""
nnUNet_TestEnvironment.py
=========================
Standard nn-UNet (no-new-UNet) for 2-D medical image segmentation.
Isensee et al., "No New-Net", MICCAI 2018.

Usage in the notebook:

    Cell 8  : MODEL_FILE = "nnUNet_TestEnvironment.py"

    Cell 9  : from nnUNet_TestEnvironment import UNet, MODEL_CONFIGS
              MODEL_ID = "NN"
              model = UNet().to(device)
              print("✅ nn-UNet model built.")

    Cell 11 : from nnUNet_TestEnvironment import DeepSupervisionLoss
              criterion = DeepSupervisionLoss()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Channel sizes at each level: 32 → 64 → 128 → 256 → 320 → 320
# Index:                         0     1     2     3     4     5
CHANNELS = [min(32 * (2 ** i), 320) for i in range(6)]


# ── Building blocks ───────────────────────────────────────────────────────────

class ConvNormAct(nn.Module):
    """Conv2d → InstanceNorm2d → LeakyReLU(0.01)."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        nn.init.kaiming_normal_(self.block[0].weight, a=0.01,
                                nonlinearity='leaky_relu')

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """Two ConvNormAct layers with a residual shortcut."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = ConvNormAct(in_ch, out_ch, stride=stride)
        self.conv2 = ConvNormAct(out_ch, out_ch)
        self.act   = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        if stride != 1 or in_ch != out_ch:
            proj = nn.Conv2d(in_ch, out_ch, kernel_size=1,
                             stride=stride, bias=False)
            nn.init.kaiming_normal_(proj.weight, a=0.01,
                                    nonlinearity='leaky_relu')
            self.shortcut = nn.Sequential(
                proj, nn.InstanceNorm2d(out_ch, affine=True))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.act(self.conv2(self.conv1(x)) + self.shortcut(x))


class EncoderBlock(nn.Module):
    """Strided residual block (stride=2) — replaces max-pool."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = ResidualBlock(in_ch, out_ch, stride=2)

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """Bilinear upsample + 1x1 conv, then residual block with skip."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        up_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(up_conv.weight, a=0.01,
                                nonlinearity='leaky_relu')
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            up_conv,
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.block = ResidualBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x  = self.up(x)
        dh = skip.shape[2] - x.shape[2]
        dw = skip.shape[3] - x.shape[3]
        if dh > 0 or dw > 0:
            x = F.pad(x, (0, dw, 0, dh))
        return self.block(torch.cat([x, skip], dim=1))


# ── nn-UNet ───────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Standard nn-UNet (no-new-UNet).

    Full channel flow through forward pass:
      skips stack built as:  [32, 64, 128, 256, 320, 320]
                        idx:   0    1    2    3    4    5

      Bottleneck = skips.pop()  → ch=320 (idx 5)

      Decoder (each step pops next skip):
        Dec 0: in=320, skip=320(idx4), out=320  → decoder_outs[0]
        Dec 1: in=320, skip=256(idx3), out=256  → decoder_outs[1]
        Dec 2: in=256, skip=128(idx2), out=128  → decoder_outs[2]
        Dec 3: in=128, skip=64 (idx1), out=64   → decoder_outs[3]
        Dec 4: in=64,  skip=32 (idx0), out=32   → decoder_outs[4]

      Seg head:   decoder_outs[4] ch=32  → 1
      Aux heads:  decoder_outs[0] ch=320 → 1
                  decoder_outs[1] ch=256 → 1
                  decoder_outs[2] ch=128 → 1
                  decoder_outs[3] ch=64  → 1
    """

    def __init__(self):
        super().__init__()

        # Stem: 1 → 32, full resolution, no downsampling
        self.stem = ResidualBlock(1, CHANNELS[0])

        # Encoder: 5 blocks, each halves H and W
        self.encoders = nn.ModuleList([
            EncoderBlock(CHANNELS[i], CHANNELS[i + 1])
            for i in range(5)
        ])

        # Decoder: 5 blocks, each doubles H and W
        self.decoders = nn.ModuleList([
            DecoderBlock(
                in_ch   = CHANNELS[5 - i],
                skip_ch = CHANNELS[4 - i],
                out_ch  = CHANNELS[4 - i],
            )
            for i in range(5)
        ])

        # Primary segmentation head (full resolution, decoder_outs[4], ch=32)
        self.seg_head = nn.Conv2d(CHANNELS[0], 1, kernel_size=1)
        nn.init.kaiming_normal_(self.seg_head.weight, a=0.01,
                                nonlinearity='leaky_relu')

        # Auxiliary heads — channels must match decoder_outs[0..3] exactly:
        #   decoder_outs[0]=320, [1]=256, [2]=128, [3]=64
        aux_ch = [CHANNELS[4], CHANNELS[3], CHANNELS[2], CHANNELS[1]]
        #        = [320,         256,         128,         64]
        self.aux_heads = nn.ModuleList()
        for ch in aux_ch:
            head = nn.Conv2d(ch, 1, kernel_size=1)
            nn.init.kaiming_normal_(head.weight, a=0.01,
                                    nonlinearity='leaky_relu')
            self.aux_heads.append(head)

    def forward(self, x):
        # Encoder
        skips = []
        out = self.stem(x)              # ch=32
        skips.append(out)
        for enc in self.encoders:
            out = enc(out)
            skips.append(out)
        # skips = [ch=32, ch=64, ch=128, ch=256, ch=320, ch=320]

        # Bottleneck
        out = skips.pop()              # ch=320

        # Decoder
        decoder_outs = []
        for dec in self.decoders:
            skip = skips.pop()
            out  = dec(out, skip)
            decoder_outs.append(out)
        # decoder_outs[0]=ch=320, [1]=256, [2]=128, [3]=64, [4]=32

        # Primary prediction (full resolution)
        primary = torch.sigmoid(self.seg_head(decoder_outs[4]))

        # Eval: return plain Tensor — notebook metrics work unchanged
        if not self.training:
            return primary

        # Train: return list for deep supervision
        aux_preds = [
            torch.sigmoid(self.aux_heads[i](decoder_outs[i]))
            for i in range(4)
        ]
        return [primary] + aux_preds


# ── Deep Supervision Loss ─────────────────────────────────────────────────────

class _DiceLoss(nn.Module):
    _SMOOTH = 1e-5

    def forward(self, pred, target):
        p = pred.view(-1)
        t = target.view(-1)
        inter = (p * t).sum()
        return 1 - (2 * inter + self._SMOOTH) / (p.sum() + t.sum() + self._SMOOTH)


class DeepSupervisionLoss(nn.Module):
    """
    BCE + Dice with deep-supervision weighting.
    Replaces CombinedLoss in notebook cell 11.

    Training : weighted sum over scales (w=1.0, 0.5, 0.25, 0.125, 0.0625)
    Eval     : plain BCE + Dice on the single Tensor the model returns
    """

    def __init__(self):
        super().__init__()
        self._bce  = nn.BCELoss()
        self._dice = _DiceLoss()

    def _single(self, pred, target):
        return self._bce(pred, target) + self._dice(pred, target)

    def forward(self, pred, target):
        if isinstance(pred, torch.Tensor):
            return self._single(pred, target)

        total, weight = torch.tensor(0.0, device=target.device), 1.0
        for p in pred:
            t = (F.interpolate(target, size=p.shape[2:], mode='nearest')
                 if p.shape[2:] != target.shape[2:] else target)
            total  = total + weight * self._single(p, t)
            weight *= 0.5
        return total


# ── MODEL_CONFIGS (notebook API parity) ──────────────────────────────────────
MODEL_CONFIGS = {"NN": {}}
