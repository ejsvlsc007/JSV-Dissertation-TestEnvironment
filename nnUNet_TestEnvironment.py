"""
nnUNet_TestEnvironment.py
=========================
Standard nn-UNet (no-new-UNet) for 2-D medical image segmentation.
Isensee et al., "No New-Net", MICCAI 2018.

Usage in the notebook (cells 8, 9, 11):

    Cell 8  : MODEL_FILE = "nnUNet_TestEnvironment.py"

    Cell 9  : from nnUNet_TestEnvironment import UNet, MODEL_CONFIGS
              model = UNet(**MODEL_CONFIGS[MODEL_ID]).to(device)

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

        def nf(lvl):
            return min(32 * (2 ** lvl), 320)

        # Stem (full resolution, no down-sampling)
        self.stem = ResidualBlock(1, nf(0))

        # Encoder
        self.encoders = nn.ModuleList([
            EncoderBlock(nf(i), nf(i + 1)) for i in range(5)
        ])

        # Decoder
        self.decoders = nn.ModuleList([
            DecoderBlock(nf(5 - i), nf(5 - i - 1), nf(5 - i - 1))
            for i in range(5)
        ])

        # Primary segmentation head (full resolution)
        self.seg_head = nn.Conv2d(nf(0), 1, kernel_size=1)
        nn.init.kaiming_normal_(self.seg_head.weight, a=0.01,
                                nonlinearity='leaky_relu')

        # Auxiliary deep-supervision heads (4 coarser scales)
        self.aux_heads = nn.ModuleList()
        for i in range(1, 5):
            head = nn.Conv2d(nf(i), 1, kernel_size=1)
            nn.init.kaiming_normal_(head.weight, a=0.01,
                                    nonlinearity='leaky_relu')
            self.aux_heads.append(head)

    def forward(self, x):
        # Encoder
        skips = []
        out = self.stem(x)
        skips.append(out)
        for enc in self.encoders:
            out = enc(out)
            skips.append(out)

        # Bottleneck = deepest encoder output
        out = skips.pop()

        # Decoder
        decoder_outs = []
        for dec in self.decoders:
            out = dec(out, skips.pop())
            decoder_outs.append(out)
        # decoder_outs[-1] = full resolution

        # Primary prediction
        primary = torch.sigmoid(self.seg_head(decoder_outs[-1]))

        # Eval: return plain Tensor (notebook-compatible)
        if not self.training:
            return primary

        # Train: return list for deep supervision
        # decoder_outs[0]=coarsest … decoder_outs[-2]=second-finest
        aux_preds = [
            torch.sigmoid(head(decoder_outs[i]))
            for i, head in enumerate(self.aux_heads)
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


# ── MODEL_CONFIGS (kept for notebook API parity) ──────────────────────────────
# The notebook calls: model = UNet(**MODEL_CONFIGS[MODEL_ID])
# UNet takes no arguments, so this just maps any MODEL_ID to an empty dict.

MODEL_CONFIGS = {
    "NN": {}
}
