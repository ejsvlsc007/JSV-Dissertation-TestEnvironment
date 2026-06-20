"""
fusion_models.py — three phase-fusion variants on ONE shared U-Net backbone.

A controlled comparison of FUSION STAGE (the variable your first comparison
couldn't isolate). All three share the identical U-Net backbone; only where the
NC/ART/PVP phases are combined changes:

  fusion_stage = "early"        F0 naive stack → 3-ch input → single encoder.
  fusion_stage = "intermediate" per-phase SHARED encoder → cross-phase attention
                                at the bottleneck → shared decoder.
  fusion_stage = "late"         per-phase SHARED U-Net → learned softmax-weighted
                                fusion of the per-phase logits.

Weight sharing: intermediate/late run ONE encoder (or one full U-Net) over each
phase in a loop, so the parameter count stays ~1× the backbone (+ small fusion
module), NOT 3×. That keeps the comparison about fusion stage, not capacity.

Data pipeline is unchanged: the model always receives the F0-stacked
(B, 3, H, W) tensor from the slice cache; intermediate/late split it internally.
Output is a single logits map (no deep supervision), matching the U-Net F0 run,
so the loss / metrics / k-fold are identical across all three.
"""
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================== #
#  Phase-fusion mechanisms (standalone copy of shared/phase_fusion.py)
# =========================================================================== #
def split_phases(x, num_phases=3):
    assert x.shape[1] == num_phases, f"expected {num_phases} channels, got {x.shape[1]}"
    return [x[:, i:i + 1] for i in range(num_phases)]


class CrossPhaseAttentionFusion(nn.Module):
    """Bidirectional cross-phase attention over per-phase bottleneck features.
    Input: list of K (B,C,H,W). Output: fused (B,C,H,W)."""
    def __init__(self, channels, num_phases=3, num_heads=4, mlp_ratio=2.0, drop=0.0):
        super().__init__()
        if channels % num_heads != 0:
            num_heads = next((h for h in (8, 4, 2, 1) if channels % h == 0), 1)
        self.num_phases = num_phases
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(),
                                 nn.Dropout(drop), nn.Linear(hidden, channels),
                                 nn.Dropout(drop))
        self.phase_logits = nn.Parameter(torch.zeros(num_phases))

    def _attn(self, t):
        N, K, C = t.shape
        qkv = self.qkv(t).reshape(N, K, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(N, K, C)
        return self.proj(out)

    def forward(self, phases: List[torch.Tensor]) -> torch.Tensor:
        B, C, H, W = phases[0].shape
        t = torch.stack(phases, 1).permute(0, 3, 4, 1, 2).reshape(B * H * W, self.num_phases, C)
        t = t + self._attn(self.norm1(t))
        t = t + self.mlp(self.norm2(t))
        t = t.reshape(B, H, W, self.num_phases, C)
        w = self.phase_logits.softmax(0).view(1, 1, 1, self.num_phases, 1)
        return (t * w).sum(3).permute(0, 3, 1, 2).contiguous()


def fuse_skip_features(skip_lists):
    K, L = len(skip_lists), len(skip_lists[0])
    return [torch.stack([skip_lists[k][i] for k in range(K)], 0).mean(0) for i in range(L)]


class LearnedPhaseWeightFusion(nn.Module):
    """Late fusion: softmax-normalised per-phase weighting (no phase can be zeroed)."""
    def __init__(self, channels=1, num_phases=3, mode="spatial"):
        super().__init__()
        assert mode in ("scalar", "spatial")
        self.mode = mode
        self.num_phases = num_phases
        if mode == "scalar":
            self.phase_logits = nn.Parameter(torch.zeros(num_phases))
        else:
            self.weight_conv = nn.Conv2d(num_phases * channels, num_phases, 1)

    def forward(self, phases: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(phases, 1)                    # (B,K,C,H,W)
        if self.mode == "scalar":
            w = self.phase_logits.softmax(0).view(1, self.num_phases, 1, 1, 1)
            return (stacked * w).sum(1)
        B, K, C, H, W = stacked.shape
        w = self.weight_conv(stacked.reshape(B, K * C, H, W)).softmax(1).unsqueeze(2)
        return (stacked * w).sum(1)


# =========================================================================== #
#  Shared U-Net backbone (encoder / decoder separable)
# =========================================================================== #
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, convs=2, use_bn=True):
        super().__init__()
        layers = []
        for i in range(convs):
            layers.append(nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1,
                                    bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetEncoder(nn.Module):
    def __init__(self, in_ch, base_ch=64, depth=4, convs_per_block=2, use_bn=True):
        super().__init__()
        self.depth = depth
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = in_ch
        for d in range(depth):
            out_ch = base_ch * (2 ** d)
            self.encoders.append(ConvBlock(prev, out_ch, convs_per_block, use_bn))
            self.pools.append(nn.MaxPool2d(2))
            prev = out_ch
        self.bottleneck = ConvBlock(prev, prev * 2, convs_per_block, use_bn)
        self.bottleneck_ch = prev * 2

    def forward(self, x):
        skips = []
        for i in range(self.depth):
            x = self.encoders[i](x)
            skips.append(x)
            x = self.pools[i](x)
        return self.bottleneck(x), skips      # bott, [s0..s_{depth-1}]


class UNetDecoder(nn.Module):
    def __init__(self, base_ch=64, depth=4, convs_per_block=2, use_bn=True, out_ch=1):
        super().__init__()
        self.depth = depth
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for d in reversed(range(depth)):
            in_chs = base_ch * (2 ** (d + 1))
            out_chs = base_ch * (2 ** d)
            self.upconvs.append(nn.ConvTranspose2d(in_chs, out_chs, 2, stride=2))
            self.decoders.append(ConvBlock(in_chs, out_chs, convs_per_block, use_bn))
        self.out = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, bott, skips):
        x = bott
        for i in range(self.depth):
            x = self.upconvs[i](x)
            skip = skips[-(i + 1)]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = self.decoders[i](torch.cat([x, skip], 1))
        return self.out(x)                     # logits


# =========================================================================== #
#  Three fusion variants on the shared backbone
# =========================================================================== #
class UNetEarlyF0(nn.Module):
    """F0: naive stack (3-ch input) → single encoder/decoder."""
    def __init__(self, base_ch=64, depth=4, convs_per_block=2, use_bn=True, dropout_rate=None):
        super().__init__()
        self.enc = UNetEncoder(3, base_ch, depth, convs_per_block, use_bn)
        self.dec = UNetDecoder(base_ch, depth, convs_per_block, use_bn)
        self.drop = nn.Dropout2d(dropout_rate) if dropout_rate else None

    def forward(self, x):
        bott, skips = self.enc(x)
        if self.drop: bott = self.drop(bott)
        return self.dec(bott, skips)


class UNetIntermediate(nn.Module):
    """Intermediate: shared per-phase encoder → cross-phase attention at bottleneck
    → shared decoder."""
    def __init__(self, base_ch=64, depth=4, convs_per_block=2, use_bn=True,
                 dropout_rate=None, num_phases=3, attn_heads=4):
        super().__init__()
        self.num_phases = num_phases
        self.enc = UNetEncoder(1, base_ch, depth, convs_per_block, use_bn)   # SHARED
        self.dec = UNetDecoder(base_ch, depth, convs_per_block, use_bn)      # SHARED
        self.cpa = CrossPhaseAttentionFusion(self.enc.bottleneck_ch,
                                             num_phases=num_phases, num_heads=attn_heads)
        self.drop = nn.Dropout2d(dropout_rate) if dropout_rate else None

    def forward(self, x):
        phases = split_phases(x, self.num_phases)
        enc_out = [self.enc(p) for p in phases]            # shared weights, run per phase
        botts = [b for b, _ in enc_out]
        skips = [s for _, s in enc_out]
        fused_bott = self.cpa(botts)
        if self.drop: fused_bott = self.drop(fused_bott)
        fused_skips = fuse_skip_features(skips)
        return self.dec(fused_bott, fused_skips)


class UNetLate(nn.Module):
    """Late: shared per-phase U-Net → learned softmax-weighted fusion of logits."""
    def __init__(self, base_ch=64, depth=4, convs_per_block=2, use_bn=True,
                 dropout_rate=None, num_phases=3, late_mode="spatial"):
        super().__init__()
        self.num_phases = num_phases
        self.enc = UNetEncoder(1, base_ch, depth, convs_per_block, use_bn)   # SHARED
        self.dec = UNetDecoder(base_ch, depth, convs_per_block, use_bn)      # SHARED
        self.late = LearnedPhaseWeightFusion(channels=1, num_phases=num_phases, mode=late_mode)
        self.drop = nn.Dropout2d(dropout_rate) if dropout_rate else None

    def _phase_net(self, p):
        bott, skips = self.enc(p)
        if self.drop: bott = self.drop(bott)
        return self.dec(bott, skips)

    def forward(self, x):
        phases = split_phases(x, self.num_phases)
        per_phase_logits = [self._phase_net(p) for p in phases]   # shared weights
        return self.late(per_phase_logits)


# =========================================================================== #
#  Factory + loss (drop-in for the k-fold notebook)
# =========================================================================== #
def build_model(in_channels=3, img_size=256, fusion_stage="early",
                base_ch=64, depth=4, convs_per_block=2, use_bn=True,
                dropout_rate=None, **_ignored):
    common = dict(base_ch=base_ch, depth=depth, convs_per_block=convs_per_block,
                  use_bn=use_bn, dropout_rate=dropout_rate)
    if fusion_stage == "early":
        return UNetEarlyF0(**common)
    if fusion_stage == "intermediate":
        return UNetIntermediate(**common)
    if fusion_stage == "late":
        return UNetLate(**common)
    raise ValueError(f"fusion_stage must be early|intermediate|late, got {fusion_stage}")


class DECTNetLoss(nn.Module):
    def __init__(self, dice_weight=0.5, aux_weight=0.4, pos_weight=500.0, smooth=1e-5):
        super().__init__()
        self.dice_weight = dice_weight; self.aux_weight = aux_weight; self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def _dice(self, logits, target):
        prob = torch.sigmoid(logits)
        num = 2 * (prob * target).sum((1, 2, 3)) + self.smooth
        den = prob.sum((1, 2, 3)) + target.sum((1, 2, 3)) + self.smooth
        return (1 - num / den).mean()

    def _single(self, logits, target):
        return self.bce(logits, target) + self.dice_weight * self._dice(logits, target)

    def forward(self, outputs, target):
        if isinstance(outputs, (list, tuple)):
            loss = self._single(outputs[0], target)
            for a in outputs[1:]:
                loss = loss + self.aux_weight * self._single(a, target)
            return loss
        return self._single(outputs, target)


Loss = DECTNetLoss
