"""
nnUNet0_TestEnvironment.py
==========================
nn-UNet v2 -- 2-D medical image segmentation baseline.

Architectural reimplementation of the nnU-Net v2 network topology
(Isensee et al., 2022/2024) as a standalone PyTorch module, designed
to plug directly into the JSV dissertation test-environment notebooks.

Key v2 upgrades over the original no-new-Net (N0 v1):
  * Residual encoder blocks  (stride=2 on first conv, projection shortcut)
  * Strided convolutions for downsampling  (no max-pool)
  * Transposed convolutions for upsampling  (no bilinear)
  * 2x ConvNormAct in each decoder stage   (matches nnUNet v2 decoder)
  * Normalised geometric deep-supervision loss weights (sum to 1)
  * InstanceNorm2d + LeakyReLU(0.01)  (same as v1, retained)
  * Channel schedule: 32->64->128->256->320->320  (same cap as v1, retained)

MCT_FUSION class-imbalance handling (tumour task):
  * DeepSupervisionLoss accepts pos_weight to upweight the minority class
  * Notebook Cell 5 sets pos_weight automatically when DATASET_SOURCE=='MCT_FUSION'
  * Slice filtering in MCTFusionDataset keeps only slices with tumour pixels

Usage in the notebook
---------------------
Cell 4  (pull & load):
    MODEL_FILE = "nnUNet0_TestEnvironment.py"

Cell 4b  (build model):
    from nnUNet0_TestEnvironment import UNet, MODEL_CONFIGS

    model_cfg = dict(MODEL_CONFIGS[MODEL_ID])
    if DATASET_SOURCE == 'MCT_FUSION':
        model_cfg['in_ch'] = 3
        print("MCT_FUSION detected - overriding in_ch to 3")

    model = UNet(**model_cfg).to(device)
    print("nn-UNet v2 model built.")

Cell 5  (loss):
    from nnUNet0_TestEnvironment import DeepSupervisionLoss

    # For MCT_FUSION use pos_weight to counter class imbalance
    if DATASET_SOURCE == 'MCT_FUSION':
        criterion = DeepSupervisionLoss(pos_weight=100.0)
        print("MCT_FUSION: using pos_weight=100.0 in loss")
    else:
        criterion = DeepSupervisionLoss()

MODEL_CONFIGS
-------------
    "N0"  : standard single-channel baseline (in_ch=1)

    The notebook's Cell 4b injects in_ch=3 at runtime when
    DATASET_SOURCE == 'MCT_FUSION', so MODEL_CONFIGS itself does not need
    a separate MCT_FUSION entry -- the override pattern handles it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Channel schedule  (mirrors nnUNet v2 defaults for 2-D)
# 32 -> 64 -> 128 -> 256 -> 320 -> 320   (cap at 320)
# ---------------------------------------------------------------------------
CHANNELS = [min(32 * (2 ** i), 320) for i in range(6)]

# MODEL_CONFIGS -- values are passed as **kwargs to UNet.__init__
# in_ch defaults to 1; notebook overrides to 3 for MCT_FUSION at runtime.
MODEL_CONFIGS = {
    "N0": {"channels": CHANNELS},
}


# ===========================================================================
# Building blocks
# ===========================================================================

class ConvNormAct(nn.Module):
    """Conv2d -> InstanceNorm2d -> LeakyReLU(0.01)."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        nn.init.kaiming_normal_(self.block[0].weight, a=0.01,
                                nonlinearity='leaky_relu')

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    nnUNet v2 residual encoder block.

    Two ConvNormAct layers with an identity (or projection) shortcut.
    The first conv may use stride > 1 for downsampling; the shortcut then
    uses a matching 1x1 strided conv so dimensions align.
    """

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = ConvNormAct(in_ch, out_ch, stride=stride)
        self.conv2 = ConvNormAct(out_ch, out_ch)
        self.act   = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        # Projection shortcut when dimensions change
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride,
                          padding=0, bias=False),
                nn.InstanceNorm2d(out_ch, affine=True),
            )
            nn.init.kaiming_normal_(self.shortcut[0].weight, a=0.01,
                                    nonlinearity='leaky_relu')
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return self.act(out + residual)


class UpsampleBlock(nn.Module):
    """
    Transposed-convolution upsampling then concat skip + 2x ConvNormAct.

    nnUNet v2 uses transposed convolutions (not bilinear) for upsampling
    and concatenates the matching encoder feature map before applying two
    conv layers in the decoder.
    """

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        # Halve channels, double spatial resolution
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2,
                                     bias=False)
        nn.init.kaiming_normal_(self.up.weight, a=0.01,
                                nonlinearity='leaky_relu')

        # After concat with skip: (out_ch + skip_ch) channels -> out_ch
        self.conv = nn.Sequential(
            ConvNormAct(out_ch + skip_ch, out_ch),
            ConvNormAct(out_ch, out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Handle any +-1-pixel size mismatches from integer downsampling
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ===========================================================================
# Full U-Net (v2)
# ===========================================================================

class UNet(nn.Module):
    """
    2-D nnU-Net v2 architecture.

    Encoder   : 5 residual blocks with strided-conv downsampling
    Bottleneck: one residual block at the lowest resolution
    Decoder   : 5 transposed-conv + concat + 2xconv stages
    Heads     : 1 primary output head + 4 auxiliary heads (deep supervision)

    Parameters
    ----------
    in_ch : int
        Number of input channels.
          1  for single-phase MCT / LiTS (default)
          3  for MCT_FUSION 3-channel early fusion
        The notebook's Cell 4b injects in_ch=3 at runtime when
        DATASET_SOURCE == 'MCT_FUSION'.
    out_ch : int
        Number of output channels (1 for binary segmentation).
    channels : list[int]
        Per-stage channel counts. Defaults to [32,64,128,256,320,320].

    Output (training)
    -----------------
    list of 5 Tensors: [primary, aux_coarse, ..., aux_fine]
    Training loop must use out[0] for per-batch metrics.

    Output (eval)
    -------------
    Single Tensor (primary prediction only).
    """

    def __init__(self, in_ch=1, out_ch=1, channels=None):
        super().__init__()
        if channels is None:
            channels = CHANNELS          # [32, 64, 128, 256, 320, 320]

        # ------------------------------------------------------------------
        # Encoder  (6 stages: 0 to 5)
        # Stage 0 : stride=1  (no downsampling at the very first stage)
        # Stages 1-5: stride=2 inside ResidualBlock
        # ------------------------------------------------------------------
        self.enc0 = ResidualBlock(in_ch,         channels[0], stride=1)
        self.enc1 = ResidualBlock(channels[0],   channels[1], stride=2)
        self.enc2 = ResidualBlock(channels[1],   channels[2], stride=2)
        self.enc3 = ResidualBlock(channels[2],   channels[3], stride=2)
        self.enc4 = ResidualBlock(channels[3],   channels[4], stride=2)

        # Bottleneck
        self.bottleneck = ResidualBlock(channels[4], channels[5], stride=2)

        # ------------------------------------------------------------------
        # Decoder  (5 stages, mirroring encoder)
        # UpsampleBlock(in_ch, skip_ch, out_ch)
        # ------------------------------------------------------------------
        self.dec4 = UpsampleBlock(channels[5], channels[4], channels[4])
        self.dec3 = UpsampleBlock(channels[4], channels[3], channels[3])
        self.dec2 = UpsampleBlock(channels[3], channels[2], channels[2])
        self.dec1 = UpsampleBlock(channels[2], channels[1], channels[1])
        self.dec0 = UpsampleBlock(channels[1], channels[0], channels[0])

        # ------------------------------------------------------------------
        # Output heads
        # Primary head : after dec0  (full resolution, channels[0]=32)
        # Aux heads    : dec4(320), dec3(256), dec2(128), dec1(64)
        #                index 0 = coarsest, index 3 = finest aux
        # ------------------------------------------------------------------
        self.primary_head = nn.Conv2d(channels[0], out_ch, kernel_size=1)

        self.aux_heads = nn.ModuleList([
            nn.Conv2d(channels[4], out_ch, kernel_size=1),  # dec4  320ch
            nn.Conv2d(channels[3], out_ch, kernel_size=1),  # dec3  256ch
            nn.Conv2d(channels[2], out_ch, kernel_size=1),  # dec2  128ch
            nn.Conv2d(channels[1], out_ch, kernel_size=1),  # dec1   64ch
        ])

        # Weight init for all output heads
        for head in [self.primary_head] + list(self.aux_heads):
            nn.init.kaiming_normal_(head.weight, a=0.01,
                                    nonlinearity='leaky_relu')
            nn.init.zeros_(head.bias)

    def forward(self, x):
        # --- Encoder ---
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # --- Bottleneck ---
        b = self.bottleneck(e4)

        # --- Decoder ---
        d4 = self.dec4(b,  e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        d0 = self.dec0(d1, e0)

        # --- Output heads ---
        primary = torch.sigmoid(self.primary_head(d0))

        if self.training:
            aux = [
                torch.sigmoid(self.aux_heads[0](d4)),
                torch.sigmoid(self.aux_heads[1](d3)),
                torch.sigmoid(self.aux_heads[2](d2)),
                torch.sigmoid(self.aux_heads[3](d1)),
            ]
            # Training loop must use out[0] for per-batch metrics
            return [primary] + aux

        return primary


# ===========================================================================
# Deep-Supervision Loss  (Dice + weighted BCE, geometrically weighted)
# ===========================================================================

class DeepSupervisionLoss(nn.Module):
    """
    Weighted combination of Dice + BCE applied at each decoder output.

    Weights follow the nnUNet v2 default geometric schedule:
      w_i = (1/2)^i  then re-normalised to sum to 1.
      Index 0 = primary (full-res, ~51.6%), indices 1-4 = aux (coarse->fine).

    Parameters
    ----------
    smooth : float
        Laplace smoothing for Dice numerator/denominator.
    pos_weight : float or None
        If set, multiplies the BCE loss for positive pixels by this factor.
        Use for class-imbalanced tasks such as MCT_FUSION tumour segmentation.
        Typical value: ~100.0 (ratio of background to foreground pixels).
        None (default) = unweighted BCE, suitable for MCT whole-liver.

    Usage in Cell 5:
        # MCT whole-liver (balanced enough -- no weighting needed)
        criterion = DeepSupervisionLoss()

        # MCT_FUSION tumour (~0.01% positive voxels -- must weight)
        criterion = DeepSupervisionLoss(pos_weight=100.0)
    """

    def __init__(self, smooth=1e-5, pos_weight=None):
        super().__init__()
        self.smooth     = smooth
        self.pos_weight = pos_weight  # scalar float or None
        # 5 outputs: primary + 4 aux
        raw = [0.5 ** i for i in range(5)]
        total = sum(raw)
        self.weights = [w / total for w in raw]

    def _dice_bce(self, pred, target):
        # ── Dice loss ───────────────────────────────────────────────────────
        p = pred.view(-1)
        t = target.view(-1)
        intersection = (p * t).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / \
                          (p.sum() + t.sum() + self.smooth)

        # ── Weighted BCE ─────────────────────────────────────────────────────
        if self.pos_weight is not None:
            # pw tensor on same device as pred
            pw = torch.tensor([self.pos_weight], dtype=pred.dtype,
                              device=pred.device)
            bce_loss = F.binary_cross_entropy_with_logits(
                # BCE expects logits when using pos_weight; convert from sigmoid
                torch.logit(pred.clamp(1e-6, 1 - 1e-6)),
                target,
                pos_weight=pw,
                reduction='mean',
            )
        else:
            bce_loss = F.binary_cross_entropy(pred, target, reduction='mean')

        return dice_loss + bce_loss

    def forward(self, outputs, target):
        if isinstance(outputs, list):
            loss = 0.0
            for w, out in zip(self.weights, outputs):
                # Downsample target to match aux output resolution if needed
                if out.shape != target.shape:
                    t = F.interpolate(target, size=out.shape[2:],
                                      mode='nearest')
                else:
                    t = target
                loss += w * self._dice_bce(out, t)
            return loss
        else:
            return self._dice_bce(outputs, target)
