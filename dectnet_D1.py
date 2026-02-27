"""
DECTNet D1: Dual Encoder Network Combined Convolution and CSwin Transformer
for Medical Image Segmentation

Architecture change from DECTNet (dectnet.py):
  Section 2 REPLACED: Swin Transformer Encoder → CSwin Transformer Encoder

CSwin key differences vs Swin:
  - Cross-shaped window attention (horizontal + vertical stripes in parallel)
    instead of square local windows with cyclic shifting
  - Locally-Enhanced Positional Encoding (LePE): depth-wise conv on V,
    better for dense prediction than Swin's relative position bias table
  - No SW-MSA shift → no create_attn_mask, no cyclic roll
  - stripe_size replaces window_size (set to 8 for clean fit at 128×128)

Everything else is identical to dectnet.py:
  Section 1  — CNN Encoder         (unchanged)
  Section 3  — Feature Fusion      (unchanged, swin_ch=384 still matches)
  Section 4  — Decoder             (unchanged)
  Section 5  — Deep Supervision    (unchanged)
  Section 6  — DECTNet model       (unchanged, forward references swin_feats[2])
  Section 7  — Loss functions      (unchanged)
  Section 8  — Training utilities  (unchanged)
  Section 9  — Sanity check        (updated label only)

Exact Architecture:
┌──────────────────────────────────────────────────────────────────┐
│                        INPUT IMAGE                               │
│                       (B, C, H, W)                               │
└──────────────────┬───────────────────────┬───────────────────────┘
                   │                       │
      ┌────────────▼────────────┐ ┌────────▼─────────────────────┐
      │     CNN Encoder         │ │   CSwin Transformer Enc      │
      │  (ResNet-like, 4 stage) │ │  (Cross-stripe Attn + LePE,  │
      │  s1/2, s2/4, s3/8,     │ │   PatchMerging, 4 stage)     │
      │  s4/16                  │ │  t0/4, t1/8, t2/16, t3/32   │
      └────────────┬────────────┘ └────────┬─────────────────────-┘
                   │                       │
      ┌────────────▼───────────────────────▼───────────────────────┐
      │              Feature Fusion Module                         │
      │   (align spatially → concat → SE channel attention)       │
      └───────────────────────────┬────────────────────────────────┘
                                  │
      ┌───────────────────────────▼─────────────────────────────────┐
      │                U-Net Decoder                                │
      │         (skip connections from CNN encoder)                 │
      └───────────────────────────┬─────────────────────────────────┘
                                  │
      ┌───────────────────────────▼─────────────────────────────────┐
      │             Segmentation Map                                │
      │              (B, num_classes, H, W)                        │
      └─────────────────────────────────────────────────────────────┘

Dependencies:
    pip install torch torchvision
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════
# 1.  CNN ENCODER  (ResNet-like, 4 stages)  ── UNCHANGED
# ══════════════════════════════════════════════════════════════════

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = ConvBnRelu(in_ch, out_ch, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            ) if (in_ch != out_ch or stride != 1) else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv2(self.conv1(x)) + self.skip(x))


class CNNEncoder(nn.Module):
    """
    ResNet-style 4-stage encoder.
    Outputs: s1 (/2, 64ch), s2 (/4, 128ch), s3 (/8, 256ch), s4 (/16, 512ch)
    """
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.stem   = nn.Sequential(
            ConvBnRelu(in_channels, 32, stride=1),
            ConvBnRelu(32, 64, stride=2),           # /2
        )
        self.stage1 = nn.Sequential(ResidualBlock(64, 64),
                                    ResidualBlock(64, 64))
        self.stage2 = nn.Sequential(ResidualBlock(64, 128, stride=2),
                                    ResidualBlock(128, 128),
                                    ResidualBlock(128, 128))
        self.stage3 = nn.Sequential(ResidualBlock(128, 256, stride=2),
                                    ResidualBlock(256, 256),
                                    ResidualBlock(256, 256),
                                    ResidualBlock(256, 256))
        self.stage4 = nn.Sequential(ResidualBlock(256, 512, stride=2),
                                    ResidualBlock(512, 512),
                                    ResidualBlock(512, 512))

    def forward(self, x):
        x  = self.stem(x)
        s1 = self.stage1(x)    # (B,  64, H/2,  W/2)
        s2 = self.stage2(s1)   # (B, 128, H/4,  W/4)
        s3 = self.stage3(s2)   # (B, 256, H/8,  W/8)
        s4 = self.stage4(s3)   # (B, 512, H/16, W/16)
        return s1, s2, s3, s4


# ══════════════════════════════════════════════════════════════════
# 2.  CSWIN TRANSFORMER ENCODER  ── REPLACES Swin Transformer
# ══════════════════════════════════════════════════════════════════

# ── 2a. Shared helpers ───────────────────────────────────────────

class PatchEmbed(nn.Module):
    """
    Patch Partition + Linear Embedding.
    Unchanged from dectnet.py — CSwin uses the same first step.
    Output: tokens (B, H/p * W/p, embed_dim), H/p, W/p
    """
    def __init__(self, in_channels: int = 1, patch_size: int = 4,
                 embed_dim: int = 96):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)                        # (B, C, H/p, W/p)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)        # (B, H/p*W/p, C)
        return self.norm(x), H, W


class PatchMerging(nn.Module):
    """
    Downsample tokens 2× via concatenating 2×2 neighbour patches.
    Unchanged from dectnet.py.
    Input:  (B, H*W, C)
    Output: (B, H/2*W/2, 2C)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.norm      = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x  = torch.cat([x0, x1, x2, x3], dim=-1)   # (B, H/2, W/2, 4C)
        x  = x.view(B, -1, 4 * C)
        return self.reduction(self.norm(x)), H // 2, W // 2


# ── 2b. CSwin core: cross-shaped window attention with LePE ──────

class CSwinAttention(nn.Module):
    """
    Cross-Shaped Window Self-Attention with Locally-Enhanced Positional
    Encoding (LePE).

    Each head group attends along either horizontal or vertical stripes:
      - First  n_heads//2 heads  → horizontal stripes of height stripe_size
      - Second n_heads//2 heads  → vertical   stripes of width  stripe_size

    LePE adds a depth-wise conv on V (value) before the attention output,
    providing local positional cues without a separate positional embedding.

    Args:
        dim         : Total embedding dimension.
        n_heads     : Number of attention heads (must be even).
        stripe_size : Height/width of each stripe window. Should divide
                      evenly into the spatial resolution at this stage.
                      Recommended: 8 for 128×128 inputs (token grid 32×32).
        dropout     : Attention and projection dropout rate.
    """
    def __init__(self, dim: int, n_heads: int, stripe_size: int = 8,
                 dropout: float = 0.0):
        super().__init__()
        assert n_heads % 2 == 0, "n_heads must be even for CSwin split"
        self.dim         = dim
        self.n_heads     = n_heads
        self.stripe_size = stripe_size
        self.head_dim    = dim // n_heads
        self.scale       = math.sqrt(self.head_dim)

        self.qkv      = nn.Linear(dim, dim * 3, bias=False)
        self.proj     = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # LePE: one depth-wise conv per axis (horizontal / vertical)
        # Applied to V before combining with attention weights.
        half = dim // 2
        self.lepe_h = nn.Conv2d(half, half, kernel_size=3, padding=1,
                                groups=half, bias=False)
        self.lepe_v = nn.Conv2d(half, half, kernel_size=3, padding=1,
                                groups=half, bias=False)

    def _stripe_attn(self, q, k, v, lepe_fn,
                     B, H, W, n_heads, stripe_size, axis):
        """
        Compute attention along one axis (0=horizontal, 1=vertical).

        q/k/v : (B, H, W, n_heads * head_dim)  — already split to half heads
        Returns attended output: (B, H*W, n_heads * head_dim)
        """
        hd   = self.head_dim
        # ── Pad so spatial dim is divisible by stripe_size ──────────
        if axis == 0:          # horizontal: stripes along W, height = stripe_size
            pad_h = (stripe_size - H % stripe_size) % stripe_size
            pad_w = 0
        else:                  # vertical:   stripes along H, width  = stripe_size
            pad_h = 0
            pad_w = (stripe_size - W % stripe_size) % stripe_size

        if pad_h > 0 or pad_w > 0:
            q = F.pad(q, (0, 0, 0, pad_w, 0, pad_h))
            k = F.pad(k, (0, 0, 0, pad_w, 0, pad_h))
            v = F.pad(v, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = q.shape[1], q.shape[2]

        # ── LePE on V (spatial conv needs BCHW layout) ───────────────
        v_spatial = v.permute(0, 3, 1, 2).contiguous()   # (B, C_half, Hp, Wp)
        lepe      = lepe_fn(v_spatial)                    # (B, C_half, Hp, Wp)
        lepe      = lepe.permute(0, 2, 3, 1).contiguous() # (B, Hp, Wp, C_half)

        # ── Reshape into non-overlapping stripe windows ───────────────
        if axis == 0:
            # horizontal: group rows into stripes of height=stripe_size
            # each stripe covers full width Wp
            # shape: (B * (Hp/ss) * Wp, ss, n_heads, head_dim)
            ss   = stripe_size
            nH   = Hp // ss
            # (B, nH, ss, Wp, C) → (B*nH*Wp, ss, C)
            q    = q.view(B, nH, ss, Wp, n_heads * hd)
            k    = k.view(B, nH, ss, Wp, n_heads * hd)
            v    = v.view(B, nH, ss, Wp, n_heads * hd)
            lepe = lepe.view(B, nH, ss, Wp, n_heads * hd)
            # merge batch dims: (B*nH*Wp, ss, n_heads, hd)
            q    = q.permute(0,1,3,2,4).reshape(B*nH*Wp, ss, n_heads, hd)
            k    = k.permute(0,1,3,2,4).reshape(B*nH*Wp, ss, n_heads, hd)
            v    = v.permute(0,1,3,2,4).reshape(B*nH*Wp, ss, n_heads, hd)
            lepe = lepe.permute(0,1,3,2,4).reshape(B*nH*Wp, ss, n_heads, hd)
            N    = ss
            num_windows = B * nH * Wp
        else:
            # vertical: group cols into stripes of width=stripe_size
            ss   = stripe_size
            nW   = Wp // ss
            q    = q.view(B, Hp, nW, ss, n_heads * hd)
            k    = k.view(B, Hp, nW, ss, n_heads * hd)
            v    = v.view(B, Hp, nW, ss, n_heads * hd)
            lepe = lepe.view(B, Hp, nW, ss, n_heads * hd)
            # (B*Hp*nW, ss, n_heads, hd)
            q    = q.permute(0,1,2,3,4).reshape(B*Hp*nW, ss, n_heads, hd)
            k    = k.permute(0,1,2,3,4).reshape(B*Hp*nW, ss, n_heads, hd)
            v    = v.permute(0,1,2,3,4).reshape(B*Hp*nW, ss, n_heads, hd)
            lepe = lepe.permute(0,1,2,3,4).reshape(B*Hp*nW, ss, n_heads, hd)
            N    = ss
            num_windows = B * Hp * nW

        # ── Attention: (num_windows, n_heads, N, N) ──────────────────
        q = q.permute(0, 2, 1, 3)   # (nW, nh, N, hd)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) / self.scale   # (nW, nh, N, N)
        attn = self.attn_drop(attn.softmax(dim=-1))
        out  = attn @ v                                  # (nW, nh, N, hd)

        # ── Add LePE ─────────────────────────────────────────────────
        lepe = lepe.permute(0, 2, 1, 3)   # (nW, nh, N, hd)
        out  = out + lepe

        # ── Reassemble to (B, Hp, Wp, C_half) ────────────────────────
        out = out.permute(0, 2, 1, 3).reshape(num_windows, N, n_heads * hd)

        if axis == 0:
            out = out.view(B, nH, Wp, ss, n_heads * hd)
            out = out.permute(0,1,3,2,4).reshape(B, Hp, Wp, n_heads * hd)
        else:
            out = out.view(B, Hp, nW, ss, n_heads * hd)
            out = out.reshape(B, Hp, Wp, n_heads * hd)

        # ── Remove padding ────────────────────────────────────────────
        out = out[:, :H, :W, :].contiguous()
        return out.view(B, H * W, n_heads * hd)

    def forward(self, x: torch.Tensor, H: int, W: int):
        B, L, C = x.shape
        assert L == H * W

        qkv = self.qkv(x)                          # (B, L, 3C)
        q, k, v = qkv.chunk(3, dim=-1)             # each (B, L, C)

        # Spatial layout for stripe ops
        q = q.view(B, H, W, C)
        k = k.view(B, H, W, C)
        v = v.view(B, H, W, C)

        # Split channels: first half → horizontal, second half → vertical
        half = C // 2
        nh_h = self.n_heads // 2   # heads for horizontal
        nh_v = self.n_heads - nh_h # heads for vertical

        out_h = self._stripe_attn(
            q[..., :half], k[..., :half], v[..., :half],
            self.lepe_h, B, H, W, nh_h, self.stripe_size, axis=0
        )  # (B, H*W, half)
        out_v = self._stripe_attn(
            q[..., half:], k[..., half:], v[..., half:],
            self.lepe_v, B, H, W, nh_v, self.stripe_size, axis=1
        )  # (B, H*W, half)

        out = torch.cat([out_h, out_v], dim=-1)    # (B, H*W, C)
        return self.proj_drop(self.proj(out))


# ── 2c. CSwin block ──────────────────────────────────────────────

class CSwinTransformerBlock(nn.Module):
    """
    One CSwin Transformer block.
    Pre-norm → CSwinAttention → residual
    Pre-norm → FFN            → residual

    No alternating shift variants needed (cross attention handles both axes
    in a single pass).
    """
    def __init__(self, dim: int, n_heads: int, stripe_size: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = CSwinAttention(dim, n_heads, stripe_size, dropout)
        self.norm2 = nn.LayerNorm(dim)
        dim_ff     = int(dim * mlp_ratio)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, H: int, W: int):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.ffn(self.norm2(x))
        return x


# ── 2d. CSwin stage ──────────────────────────────────────────────

class CSwinStage(nn.Module):
    """
    One CSwin stage = `depth` CSwin blocks followed by optional
    PatchMerging downsampling.

    Returns:
        tokens : downsampled token sequence (or same if no downsample)
        H, W   : spatial dims after optional downsampling
        feat   : spatial feature map BEFORE downsampling, for skip connections
    """
    def __init__(self, dim: int, depth: int, n_heads: int,
                 stripe_size: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, downsample: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList([
            CSwinTransformerBlock(
                dim=dim, n_heads=n_heads, stripe_size=stripe_size,
                mlp_ratio=mlp_ratio, dropout=dropout,
            )
            for _ in range(depth)
        ])
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        # Spatial feature map for skip connections / fusion
        feat = x.transpose(1, 2).view(x.shape[0], -1, H, W)
        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)
        return x, H, W, feat


# ── 2e. Full CSwin encoder ───────────────────────────────────────

class CSwinTransformerEncoder(nn.Module):
    """
    4-stage CSwin Transformer encoder.

    Stage output channels (embed_dim=96, default):
      stage0:  96ch @ H/4  × W/4   (no downsample at end)
      stage1: 192ch @ H/8  × W/8
      stage2: 384ch @ H/16 × W/16
      stage3: 768ch @ H/32 × W/32  (no downsample at end)

    Returns spatial feature maps [t0, t1, t2, t3] — same interface as
    SwinTransformerEncoder in dectnet.py, so DECTNet.forward() is unchanged.

    Args:
        in_channels  : Input image channels (1 for grayscale CT/MRI).
        patch_size   : Patch size for PatchEmbed. Default 4.
        embed_dim    : Base embedding dimension. Default 96.
        depths       : Number of CSwin blocks per stage.
                       CSwin-Tiny equivalent: (1, 2, 21, 1).
                       Lightweight option:    (1, 2,  6, 1).
        n_heads      : Attention heads per stage (must be even).
                       CSwin-Tiny: (2, 4, 8, 16).
        stripe_size  : Cross-stripe width/height per stage.
                       Single int → same for all stages.
                       Tuple → per-stage (e.g. (1, 2, 7, 7) for CSwin paper).
                       Recommended for 128×128 inputs: 8 (divides 32 cleanly).
        mlp_ratio    : FFN hidden dim multiplier. Default 4.0.
        dropout      : Dropout rate. Default 0.1.
    """
    def __init__(self, in_channels: int = 1, patch_size: int = 4,
                 embed_dim: int = 96,
                 depths=(1, 2, 6, 1),
                 n_heads=(2, 4, 8, 16),
                 stripe_size=8,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(in_channels, patch_size, embed_dim)
        self.pos_drop    = nn.Dropout(dropout)

        dims = [embed_dim * (2 ** i) for i in range(4)]   # [96,192,384,768]

        # stripe_size can be a single int or a tuple of 4
        if isinstance(stripe_size, int):
            stripe_sizes = [stripe_size] * 4
        else:
            stripe_sizes = list(stripe_size)

        self.stages = nn.ModuleList([
            CSwinStage(
                dim=dims[i], depth=depths[i], n_heads=n_heads[i],
                stripe_size=stripe_sizes[i], mlp_ratio=mlp_ratio,
                dropout=dropout, downsample=(i < 3),
            )
            for i in range(4)
        ])

    def forward(self, x):
        tokens, H, W = self.patch_embed(x)
        tokens = self.pos_drop(tokens)

        features = []
        for stage in self.stages:
            tokens, H, W, feat = stage(tokens, H, W)
            features.append(feat)
        # t0: (B,  96, H/4,  W/4)
        # t1: (B, 192, H/8,  W/8)
        # t2: (B, 384, H/16, W/16)   ← used by FeatureFusionModule
        # t3: (B, 768, H/32, W/32)
        return features   # [t0, t1, t2, t3]


# ══════════════════════════════════════════════════════════════════
# 3.  FEATURE FUSION MODULE  ── UNCHANGED
# ══════════════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        r = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, r, bias=False), nn.ReLU(inplace=True),
            nn.Linear(r, channels, bias=False), nn.Sigmoid(),
        )
    def forward(self, x):
        B, C, _, _ = x.shape
        w = self.fc(self.pool(x).view(B, C)).view(B, C, 1, 1)
        return x * w


class FeatureFusionModule(nn.Module):
    """
    Fuses CNN s4 (512ch @ H/16) with CSwin t2 (384ch @ H/16).
    Interface and channel counts identical to dectnet.py.
    """
    def __init__(self, cnn_ch: int = 512, swin_ch: int = 384, out_ch: int = 512):
        super().__init__()
        self.swin_proj = nn.Sequential(
            nn.Conv2d(swin_ch, cnn_ch, 1, bias=False),
            nn.BatchNorm2d(cnn_ch),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            ConvBnRelu(cnn_ch * 2, out_ch, kernel=1, padding=0),
            ConvBnRelu(out_ch, out_ch),
        )
        self.se = SEBlock(out_ch)

    def forward(self, cnn_feat: torch.Tensor,
                swin_feat: torch.Tensor) -> torch.Tensor:
        swin_feat = F.interpolate(swin_feat, size=cnn_feat.shape[2:],
                                  mode='bilinear', align_corners=False)
        swin_feat = self.swin_proj(swin_feat)
        fused     = self.fuse(torch.cat([cnn_feat, swin_feat], dim=1))
        return self.se(fused)


# ══════════════════════════════════════════════════════════════════
# 4.  DECODER  (U-Net-style)  ── UNCHANGED
# ══════════════════════════════════════════════════════════════════

class DecoderBlock(nn.Module):
    """Upsample 2× → cat skip → 2× ConvBnRelu."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )
    def forward(self, x, skip):
        return self.conv(torch.cat([self.up(x), skip], dim=1))


class Decoder(nn.Module):
    """
    Progressive upsampling decoder.
    fused(512,/16) → dec3(256,/8) → dec2(128,/4) → dec1(64,/2) → head(,/1)
    Skip connections from CNN encoder (s3, s2, s1).
    """
    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.dec3     = DecoderBlock(512, 256, 256)
        self.dec2     = DecoderBlock(256, 128, 128)
        self.dec1     = DecoderBlock(128,  64,  64)
        self.up_final = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.head     = nn.Sequential(
            ConvBnRelu(64, 32),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, fused, s3, s2, s1):
        x = self.dec3(fused, s3)
        x = self.dec2(x,     s2)
        x = self.dec1(x,     s1)
        x = self.up_final(x)
        return self.head(x)


# ══════════════════════════════════════════════════════════════════
# 5.  DEEP SUPERVISION  ── UNCHANGED
# ══════════════════════════════════════════════════════════════════

class DeepSupHead(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            ConvBnRelu(in_ch, max(in_ch // 2, 1), kernel=1, padding=0),
            nn.Conv2d(max(in_ch // 2, 1), num_classes, 1),
        )
    def forward(self, x, target_size):
        return F.interpolate(self.head(x), size=target_size,
                             mode='bilinear', align_corners=False)


# ══════════════════════════════════════════════════════════════════
# 6.  DECTNet D1  (Full Model)
# ══════════════════════════════════════════════════════════════════

class DECTNet(nn.Module):
    """
    DECTNet D1: Dual Encoder Network (CNN + CSwin Transformer)
    for Medical Image Segmentation.

    Drop-in replacement for DECTNet (dectnet.py).
    Only change: SwinTransformerEncoder → CSwinTransformerEncoder.

    Args:
        in_channels  (int)  : Input channels. Default 1 (grayscale MRI/CT).
        num_classes  (int)  : Output classes.  Default 1 (binary).
        embed_dim    (int)  : CSwin base embed dim. Default 96.
        depths       (tuple): CSwin blocks per stage. Default (1, 2, 6, 1).
        n_heads      (tuple): CSwin attention heads per stage. Default (2,4,8,16).
        stripe_size  (int)  : Cross-stripe size. Default 8 (suits 128×128).
        mlp_ratio   (float) : FFN expansion. Default 4.0.
        dropout     (float) : Dropout rate. Default 0.1.
        deep_sup    (bool)  : Deep supervision during training. Default True.

    Training returns : (logits, aux_fused, aux_s3)   [tuple]
    Inference returns: logits  (B, num_classes, H, W)
    """

    def __init__(
        self,
        in_channels : int   = 1,
        num_classes : int   = 1,
        embed_dim   : int   = 96,
        depths      : tuple = (1, 2, 6, 1),
        n_heads     : tuple = (2, 4, 8, 16),
        stripe_size : int   = 8,
        mlp_ratio   : float = 4.0,
        dropout     : float = 0.1,
        deep_sup    : bool  = True,
    ):
        super().__init__()
        self.deep_sup = deep_sup

        # ── Branch 1: CNN Encoder ─────────────────────────────────────
        self.cnn_encoder = CNNEncoder(in_channels)

        # ── Branch 2: CSwin Transformer Encoder ──────────────────────
        self.swin_encoder = CSwinTransformerEncoder(
            in_channels=in_channels,
            patch_size=4,
            embed_dim=embed_dim,
            depths=depths,
            n_heads=n_heads,
            stripe_size=stripe_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # ── Feature Fusion Module ─────────────────────────────────────
        # CNN s4: 512ch @ H/16  |  CSwin t2: 384ch @ H/16  → fused: 512ch
        self.fusion = FeatureFusionModule(cnn_ch=512, swin_ch=384, out_ch=512)

        # ── Decoder ───────────────────────────────────────────────────
        self.decoder = Decoder(num_classes)

        # ── Deep Supervision ──────────────────────────────────────────
        if deep_sup:
            self.aux_fused = DeepSupHead(512, num_classes)
            self.aux_s3    = DeepSupHead(256, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape

        # ① CNN encoder — local features + skip connections
        s1, s2, s3, s4 = self.cnn_encoder(x)
        # s1: (B,  64, H/2,  W/2)
        # s2: (B, 128, H/4,  W/4)
        # s3: (B, 256, H/8,  W/8)
        # s4: (B, 512, H/16, W/16)

        # ② CSwin Transformer encoder — global cross-stripe context
        swin_feats = self.swin_encoder(x)
        # t2 @ H/16 matches CNN s4 spatially → use for fusion
        t2 = swin_feats[2]   # (B, 384, H/16, W/16)

        # ③ Feature Fusion
        fused = self.fusion(s4, t2)   # (B, 512, H/16, W/16)

        # ④ Decoder with CNN skip connections
        logits = self.decoder(fused, s3, s2, s1)   # (B, num_classes, H, W)

        if self.deep_sup and self.training:
            aux1 = self.aux_fused(fused, (H, W))
            aux2 = self.aux_s3(s3,       (H, W))
            return logits, aux1, aux2

        return logits


# ══════════════════════════════════════════════════════════════════
# 7.  LOSS FUNCTION  ── UNCHANGED
# ══════════════════════════════════════════════════════════════════

class DiceBCELoss(nn.Module):
    """Dice + Binary Cross-Entropy for binary segmentation."""
    def __init__(self, smooth=1.0, dice_weight=0.5):
        super().__init__()
        self.smooth = smooth
        self.dw     = dice_weight
        self.bce    = nn.BCEWithLogitsLoss()

    def dice(self, logits, targets):
        p = torch.sigmoid(logits).view(logits.shape[0], -1)
        t = targets.float().view(targets.shape[0], -1)
        return (1 - (2 * (p * t).sum(1) + self.smooth) /
                    (p.sum(1) + t.sum(1) + self.smooth)).mean()

    def forward(self, logits, targets):
        return self.dw * self.dice(logits, targets) + \
               (1 - self.dw) * self.bce(logits, targets.float())


class DECTNetLoss(nn.Module):
    """Total = main + aux_weight*(aux1 + aux2). Handles tuple/tensor outputs."""
    def __init__(self, smooth=1.0, dice_weight=0.5, aux_weight=0.4):
        super().__init__()
        self.crit = DiceBCELoss(smooth, dice_weight)
        self.aw   = aux_weight

    def forward(self, outputs, targets):
        if isinstance(outputs, (list, tuple)):
            logits, a1, a2 = outputs
            return (self.crit(logits, targets) +
                    self.aw * self.crit(a1, targets) +
                    self.aw * self.crit(a2, targets))
        return self.crit(outputs, targets)


# ══════════════════════════════════════════════════════════════════
# 8.  TRAINING UTILITIES  ── UNCHANGED
# ══════════════════════════════════════════════════════════════════

def build_optimizer(model: nn.Module, lr=1e-4, weight_decay=1e-4):
    """AdamW — lower LR for CSwin encoder (pre-trained style)."""
    swin_ids     = {id(p) for p in model.swin_encoder.parameters()}
    swin_params  = [p for p in model.parameters() if id(p) in swin_ids]
    other_params = [p for p in model.parameters() if id(p) not in swin_ids]
    return torch.optim.AdamW([
        {'params': swin_params,  'lr': lr * 0.1},
        {'params': other_params, 'lr': lr},
    ], weight_decay=weight_decay)


def build_scheduler(optimizer, total_steps: int):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6)


def count_parameters(model: nn.Module) -> str:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return (f"Total params    : {total:,}\n"
            f"Trainable params: {trainable:,}")


# ══════════════════════════════════════════════════════════════════
# 9.  SANITY CHECK
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}\n")

    model = DECTNet(
        in_channels=1,          # grayscale MRI / CT
        num_classes=1,          # binary segmentation
        embed_dim=96,
        depths=(1, 2, 6, 1),    # CSwin-Tiny-equivalent
        n_heads=(2, 4, 8, 16),
        stripe_size=8,          # clean divisor for 128×128 token grid (32×32)
        mlp_ratio=4.0,
        dropout=0.1,
        deep_sup=True,
    ).to(device)

    print(count_parameters(model), "\n")

    # ── Training ──────────────────────────────────────────────────
    model.train()
    x      = torch.randn(2, 1, 128, 128, device=device)
    target = torch.randint(0, 2, (2, 1, 128, 128), dtype=torch.float32, device=device)

    outputs = model(x)        # (logits, aux1, aux2)
    logits  = outputs[0]

    criterion = DECTNetLoss(dice_weight=0.5, aux_weight=0.4)
    loss      = criterion(outputs, target)
    loss.backward()

    print(f"Input shape   : {x.shape}")
    print(f"Output shape  : {logits.shape}")
    print(f"Training loss : {loss.item():.4f}")
    print("Backward pass : OK")

    # ── Inference ────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        pred = model(x)
        prob = torch.sigmoid(pred)
        mask = (prob > 0.5).float()

    print(f"\nInference output : {pred.shape}")
    print(f"Binary mask      : {mask.shape}")
    print("\nDECTNet D1 (CSwin) sanity check passed ✓")
