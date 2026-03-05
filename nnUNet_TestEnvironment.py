"""
nnUNet_TestEnvironment.py
=========================
Standard nn-UNet (no-new-UNet) for 2-D medical image segmentation.
Isensee et al., "No New-Net", MICCAI 2018.

Usage in the notebook (cells 8, 9, 11):

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

# Feature channels at each of the 6 levels (stem=0 … deepest=5)
# 32 → 64 → 128 → 256 → 320 → 320  (capped at 320)
CHANNELS = [min(32 * (2 ** i), 320) for i in range(6)]
# CHANNELS = [32, 64, 128, 256, 320, 320]


class UNet(nn.Module):
    """
    Standard nn-UNet (no-new-UNet).
      - depth=5, base_features=32 (doubles per level, capped at 320)
      - Instance Norm, Leaky ReLU, strided conv down-sampling
      - Bilinear up-sampling, residual blocks throughout
      - Deep supervision during training (aux heads at each decoder scale)
      - Returns a plain Tensor during eval — compatible with the notebook's
        validate(), dice_coefficient(), and all other metric functions.
    """

    def __init__(self):
        super().__init__()

        # Stem: full resolution, no down-sampling
        # in=1 (greyscale), out=32
        self.stem = ResidualBlock(1, CHANNELS[0])

        # Encoder: 5 blocks, each halves H and W
        # E0: 32→64  E1: 64→128  E2: 128→256  E3: 256→320  E4: 320→320
        self.encoders = nn.ModuleList([
            EncoderBlock(CHANNELS[i], CHANNELS[i + 1])
            for i in range(5)
        ])

        # Decoder: 5 blocks, each doubles H and W
        # D0: in=320, skip=320, out=256
        # D1: in=256, skip=256, out=128
        # D2: in=128, skip=128, out=64
        # D3: in=64,  skip=64,  out=32
        # D4: in=32,  skip=32,  out=32
        self.decoders = nn.ModuleList([
            DecoderBlock(
                in_ch   = CHANNELS[5 - i],       # bottleneck / previous decoder output
                skip_ch = CHANNELS[4 - i],        # matching encoder skip
                out_ch  = CHANNELS[4 - i],        # output channels
            )
            for i in range(5)
        ])

        # Primary segmentation head at full resolution
        self.seg_head = nn.Conv2d(CHANNELS[0], 1, kernel_size=1)
        nn.init.kaiming_normal_(self.seg_head.weight, a=0.01,
                                nonlinearity='leaky_relu')

        # Auxiliary deep-supervision heads (4 coarser decoder outputs)
        # D0 out=256, D1 out=128, D2 out=64, D3 out=32
        self.aux_heads = nn.ModuleList()
        for i in range(4):
            head = nn.Conv2d(CHANNELS[4 - i], 1, kernel_size=1)
            nn.init.kaiming_normal_(head.weight, a=0.01,
                                    nonlinearity='leaky_relu')
            self.aux_heads.append(head)

    def forward(self, x):
        # ── Encoder ──────────────────────────────────────────────────────────
        skips = []
        out = self.stem(x)          # (B, 32, H, W)
        skips.append(out)
        for enc in self.encoders:
            out = enc(out)
            skips.append(out)
        # skips = [stem_out, E0_out, E1_out, E2_out, E3_out, E4_out]
        #          ch:  32      64      128     256     320     320

        # Bottleneck = deepest encoder output (E4_out, ch=320)
        out = skips.pop()           # (B, 320, H/32, W/32)

        # ── Decoder ──────────────────────────────────────────────────────────
        decoder_outs = []
        for dec in self.decoders:
            skip = skips.pop()      # pop from deepest remaining skip
            out  = dec(out, skip)
            decoder_outs.append(out)
        # decoder_outs[0]: ch=320→256  (coarsest)
        # decoder_outs[1]: ch=256→128
        # decoder_outs[2]: ch=128→64
        # decoder_outs[3]: ch=64→32
        # decoder_outs[4]: ch=32→32   (full resolution)

        # ── Primary prediction ────────────────────────────────────────────────
        primary = torch.sigmoid(self.seg_head(decoder_outs[4]))

        # Eval: plain Tensor — compatible with notebook's validate() and metrics
        if not self.training:
            return primary

        # Train: list for deep supervision [full_res, coarse_0, …, coarse_3]
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

    Training : weighted sum over all scales (w=1, 0.5, 0.25, 0.125, 0.0625)
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
# UNet() takes no arguments; empty dict unpacks cleanly.
MODEL_CONFIGS = {"NN": {}}
