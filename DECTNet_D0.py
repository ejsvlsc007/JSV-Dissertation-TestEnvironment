"""
DECTNet: Dual Encoder Network Combined Convolution and Transformer
for Medical Image Segmentation

Exact Architecture:
┌──────────────────────────────────────────────────────────────────┐
│                        INPUT IMAGE                               │
│                       (B, C, H, W)                               │
└──────────────────┬───────────────────────┬───────────────────────┘
                   │                       │
      ┌────────────▼────────────┐ ┌────────▼─────────────────┐
      │     CNN Encoder         │ │   Swin Transformer Enc    │
      │  (ResNet-like, 4 stage) │ │  (Window Attn + Shift,    │
      │  s1/2, s2/4, s3/8,     │ │   Patch Merging, 4 stage) │
      │  s4/16                  │ │  t1/4, t2/8, t3/16, t4/32│
      └────────────┬────────────┘ └────────┬─────────────────-┘
                   │                       │
      ┌────────────▼───────────────────────▼───────────────────┐
      │              Feature Fusion Module                      │
      │   (align spatially → concat → SE channel attention)    │
      └───────────────────────────┬─────────────────────────────┘
                                  │
      ┌───────────────────────────▼─────────────────────────────┐
      │                U-Net Decoder                            │
      │         (skip connections from CNN encoder)             │
      └───────────────────────────┬─────────────────────────────┘
                                  │
      ┌───────────────────────────▼─────────────────────────────┐
      │             Segmentation Map                            │
      │              (B, num_classes, H, W)                     │
      └─────────────────────────────────────────────────────────┘

Dependencies:
    pip install torch torchvision
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════
# UTILITY HELPERS
# ══════════════════════════════════════════════════════════════════

def window_partition(x: torch.Tensor, window_size: int):
    """
    Partition feature map into non-overlapping windows.
    x   : (B, H, W, C)
    Returns: windows (B*nW, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
                   W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, C)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int):
    """
    Reverse window_partition.
    windows : (B*nW, window_size, window_size, C)
    Returns : (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


def create_attn_mask(window_size: int, shift_size: int,
                     H: int, W: int, device: torch.device):
    """Cyclic-shift attention mask for SW-MSA."""
    if shift_size == 0:
        return None
    img_mask = torch.zeros(1, H, W, 1, device=device)
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    cnt = 0
    for hs in h_slices:
        for ws in w_slices:
            img_mask[:, hs, ws, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, window_size)       # (nW, ws, ws, 1)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask    = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)


# ══════════════════════════════════════════════════════════════════
# 1.  CNN ENCODER  (ResNet-like, 4 stages)
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
# 2.  SWIN TRANSFORMER ENCODER
# ══════════════════════════════════════════════════════════════════

class PatchEmbed(nn.Module):
    """
    Patch Partition + Linear Embedding (first step of Swin).
    Splits image into patch_size×patch_size tokens → projects to embed_dim.
    Output: (B, H/p * W/p, embed_dim)
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
    Downsample tokens by 2× via concatenating 2×2 neighbour patches
    then projecting. (Swin's "strided conv" equivalent.)
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


class WindowAttention(nn.Module):
    """
    Window-based Multi-head Self-Attention (W-MSA / SW-MSA).
    Supports relative position bias.
    """
    def __init__(self, dim: int, window_size: int, n_heads: int,
                 dropout: float = 0.0):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size
        self.n_heads     = n_heads
        self.head_dim    = dim // n_heads
        self.scale       = math.sqrt(self.head_dim)

        # Relative position bias table
        self.rel_pos_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, n_heads))
        nn.init.trunc_normal_(self.rel_pos_bias_table, std=0.02)

        # Precompute relative position indices
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords   = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flat = torch.flatten(coords, 1)                    # (2, ws*ws)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, ws^2, ws^2)
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer('rel_pos_idx', rel.sum(-1))          # (ws^2, ws^2)

        self.qkv      = nn.Linear(dim, dim * 3, bias=False)
        self.proj     = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.softmax   = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) / self.scale

        # Add relative position bias
        bias = self.rel_pos_bias_table[self.rel_pos_idx.view(-1)]
        bias = bias.view(self.window_size ** 2, self.window_size ** 2,
                         self.n_heads).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)

        # Apply cyclic-shift mask for SW-MSA
        if mask is not None:
            nW  = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.n_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.n_heads, N, N)

        attn = self.attn_drop(self.softmax(attn))
        x    = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class SwinTransformerBlock(nn.Module):
    """
    One Swin Transformer block.
    Alternates between W-MSA (shift_size=0) and SW-MSA (shift_size=ws//2).
    """
    def __init__(self, dim: int, n_heads: int, window_size: int = 7,
                 shift_size: int = 0, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.window_size = window_size
        self.shift_size  = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowAttention(dim, window_size, n_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        dim_ff     = int(dim * mlp_ratio)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, H: int, W: int):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        # Pad if needed
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        if pad_b > 0 or pad_r > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = x.shape[1], x.shape[2]

        # Cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size),
                           dims=(1, 2))

        # Partition → attention → reverse
        windows  = window_partition(x, self.window_size)          # (nW*B, ws, ws, C)
        windows  = windows.view(-1, self.window_size ** 2, C)
        mask     = create_attn_mask(self.window_size, self.shift_size,
                                    Hp, Wp, x.device)
        attn_out = self.attn(windows, mask=mask)
        attn_out = attn_out.view(-1, self.window_size, self.window_size, C)
        x        = window_reverse(attn_out, self.window_size, Hp, Wp)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size),
                           dims=(1, 2))

        # Remove padding
        if pad_b > 0 or pad_r > 0:
            x = x[:, :H, :W, :].contiguous()

        x = shortcut + x.view(B, H * W, C)
        x = x + self.ffn(self.norm2(x))
        return x


class SwinStage(nn.Module):
    """
    One Swin Transformer stage = depth blocks (alternating W/SW-MSA)
    followed by optional PatchMerging downsampling.
    """
    def __init__(self, dim: int, depth: int, n_heads: int,
                 window_size: int = 7, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, downsample: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, n_heads=n_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, dropout=dropout,
            )
            for i in range(depth)
        ])
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        feat = x.transpose(1, 2).view(x.shape[0], -1, H, W)  # spatial for skip
        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)
        return x, H, W, feat


class SwinTransformerEncoder(nn.Module):
    """
    Full Swin Transformer encoder with 4 stages.

    Stage output channels (embed_dim=96):
      stage0: 96  @ H/4  × W/4   (no downsample)
      stage1: 192 @ H/8  × W/8
      stage2: 384 @ H/16 × W/16
      stage3: 768 @ H/32 × W/32  (no further downsample)

    Returns spatial feature maps for each stage (t0..t3).
    """
    def __init__(self, in_channels: int = 1, img_size: int = 224,
                 patch_size: int = 4, embed_dim: int = 96,
                 depths=(2, 2, 6, 2), n_heads=(3, 6, 12, 24),
                 window_size: int = 7, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(in_channels, patch_size, embed_dim)
        self.pos_drop    = nn.Dropout(dropout)

        dims = [embed_dim * (2 ** i) for i in range(4)]   # [96, 192, 384, 768]

        self.stages = nn.ModuleList([
            SwinStage(
                dim=dims[i], depth=depths[i], n_heads=n_heads[i],
                window_size=window_size, mlp_ratio=mlp_ratio,
                dropout=dropout, downsample=(i < 3),   # no downsample after last stage
            )
            for i in range(4)
        ])

    def forward(self, x):
        tokens, H, W = self.patch_embed(x)
        tokens = self.pos_drop(tokens)

        features = []
        for stage in self.stages:
            tokens, H, W, feat = stage(tokens, H, W)
            features.append(feat)   # spatial maps at each resolution
        # t0: (B,  96, H/4,  W/4)
        # t1: (B, 192, H/8,  W/8)
        # t2: (B, 384, H/16, W/16)
        # t3: (B, 768, H/32, W/32)
        return features   # [t0, t1, t2, t3]


# ══════════════════════════════════════════════════════════════════
# 3.  FEATURE FUSION MODULE
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
    Fuses the deepest CNN feature (s4) with the matching Swin feature (t2)
    by:
      1. Projecting Swin features to the same channel count as CNN
      2. Aligning spatial resolution
      3. Concatenating and blending
      4. SE channel attention refinement
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
# 4.  DECODER  (U-Net-style)
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
    Skip connections come from CNN encoder (s3, s2, s1).
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
# 5.  DEEP SUPERVISION
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
# 6.  DECTNet  (Full Model)
# ══════════════════════════════════════════════════════════════════

class DECTNet(nn.Module):
    """
    DECTNet: Dual Encoder Network (CNN + Swin Transformer)
    for Medical Image Segmentation.

    Args:
        in_channels (int)  : Input channels. Default 1 (grayscale MRI/CT).
        num_classes (int)  : Output classes.  Default 1 (binary).
        img_size    (int)  : Input H = W.     Default 224.
        embed_dim   (int)  : Swin base embed dim. Default 96.
        depths      (tuple): Swin blocks per stage. Default (2,2,6,2).
        n_heads     (tuple): Swin attention heads per stage. Default (3,6,12,24).
        window_size (int)  : Swin attention window size. Default 7.
        mlp_ratio  (float) : FFN expansion. Default 4.0.
        dropout    (float) : Dropout rate. Default 0.1.
        deep_sup   (bool)  : Deep supervision during training. Default True.

    Training returns : (logits, aux_fused, aux_s3)   [tuple]
    Inference returns: logits  (B, num_classes, H, W)
    """

    def __init__(
        self,
        in_channels : int   = 1,
        num_classes : int   = 1,
        img_size    : int   = 224,
        embed_dim   : int   = 96,
        depths      : tuple = (2, 2, 6, 2),
        n_heads     : tuple = (3, 6, 12, 24),
        window_size : int   = 7,
        mlp_ratio   : float = 4.0,
        dropout     : float = 0.1,
        deep_sup    : bool  = True,
    ):
        super().__init__()
        self.deep_sup = deep_sup

        # ── Branch 1: CNN Encoder ─────────────────────────────────
        self.cnn_encoder = CNNEncoder(in_channels)

        # ── Branch 2: Swin Transformer Encoder ───────────────────
        self.swin_encoder = SwinTransformerEncoder(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=4,               # standard Swin patch size
            embed_dim=embed_dim,
            depths=depths,
            n_heads=n_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # ── Feature Fusion Module ─────────────────────────────────
        # CNN s4: 512ch @ H/16  |  Swin t2: 384ch @ H/16  → fused: 512ch
        self.fusion = FeatureFusionModule(cnn_ch=512, swin_ch=384, out_ch=512)

        # ── Decoder ───────────────────────────────────────────────
        self.decoder = Decoder(num_classes)

        # ── Deep Supervision ──────────────────────────────────────
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

        # ② Swin Transformer encoder — global context
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
# 7.  LOSS FUNCTION
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
# 8.  TRAINING UTILITIES
# ══════════════════════════════════════════════════════════════════

def build_optimizer(model: nn.Module, lr=1e-4, weight_decay=1e-4):
    """AdamW — lower LR for Swin encoder (pre-trained style)."""
    swin_ids   = {id(p) for p in model.swin_encoder.parameters()}
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
        img_size=224,
        embed_dim=96,           # Swin-Tiny config
        depths=(2, 2, 6, 2),
        n_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.0,
        dropout=0.1,
        deep_sup=True,
    ).to(device)

    print(count_parameters(model), "\n")

    # ── Training ─────────────────────────────────────────────────
    model.train()
    x      = torch.randn(2, 1, 224, 224, device=device)
    target = torch.randint(0, 2, (2, 1, 224, 224), dtype=torch.float32, device=device)

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
    print("\nDECTNet (Swin) sanity check passed ✓")
