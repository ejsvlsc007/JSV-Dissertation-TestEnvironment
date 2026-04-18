"""
shared/swin_encoder.py
======================
From-scratch Swin Transformer encoder — fixed for all input sizes.

Key fixes vs previous versions:
  - Window size clamped to actual feature map size per forward call
  - Cyclic shift mask rebuilt per call based on padded (Hp, Wp)
  - No mask caching — mask is cheap to build and caching caused stale shapes
  - RPB interpolated when clamped ws < configured ws
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Relative position bias
# ---------------------------------------------------------------------------

class RelativePositionBias(nn.Module):
    def __init__(self, window_size: int, num_heads: int):
        super().__init__()
        self.ws        = window_size
        self.num_heads = num_heads
        self.table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.table, std=0.02)
        coords      = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing="ij"
        ))
        coords_flat = coords.flatten(1)
        rel         = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel         = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("index", rel.sum(-1))

    def forward(self, ws_actual: int = None):
        """Return bias table for ws_actual tokens. Interpolates if needed."""
        bias = self.table[self.index].permute(2, 0, 1).unsqueeze(0)
        # bias: (1, heads, ws², ws²)
        if ws_actual is not None and ws_actual != self.ws:
            N = ws_actual * ws_actual
            bias = F.interpolate(
                bias.reshape(1, self.num_heads,
                             self.ws * self.ws, self.ws * self.ws),
                size=(N, N), mode="bilinear", align_corners=False,
            )
        return bias


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------

def window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    """(B, H, W, C) -> (B*nH*nW, ws, ws, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def window_reverse(windows: torch.Tensor, ws: int, H: int, W: int) -> torch.Tensor:
    """(B*nH*nW, ws, ws, C) -> (B, H, W, C)"""
    B = int(windows.shape[0] / (H * W / ws / ws))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


def build_shift_mask(Hp: int, Wp: int, ws: int, device: torch.device) -> torch.Tensor:
    """Build cyclic-shift attention mask. Returns (nW, ws², ws²)."""
    shift    = ws // 2
    img_mask = torch.zeros(1, Hp, Wp, 1, device=device)
    h_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
    w_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
    cnt = 0
    for sh in h_slices:
        for sw in w_slices:
            img_mask[:, sh, sw, :] = cnt
            cnt += 1

    # window_partition -> (nW, ws, ws, 1)
    mask_windows = window_partition(img_mask, ws)  # (nW, ws, ws, 1)
    mask_windows = mask_windows.view(-1, ws * ws)  # (nW, ws²)  ← explicit flatten

    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (nW, ws², ws²)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0) \
                         .masked_fill(attn_mask == 0,   0.0)
    return attn_mask  # (nW, ws², ws²)


# ---------------------------------------------------------------------------
# Window Multi-head Self-Attention
# ---------------------------------------------------------------------------

class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int,
                 qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.ws        = window_size
        self.num_heads = num_heads
        self.scale     = (dim // num_heads) ** -0.5
        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj      = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rpb       = RelativePositionBias(window_size, num_heads)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        x:    (B*nW, ws_actual², dim)
        mask: (nW, ws_actual², ws_actual²) or None
        """
        Bw, N, C  = x.shape
        ws_actual = int(N ** 0.5)
        head_dim  = C // self.num_heads

        qkv = self.qkv(x).reshape(Bw, N, 3, self.num_heads, head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # q, k, v: (Bw, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn: (Bw, num_heads, N, N)

        attn = attn + self.rpb(ws_actual)

        if mask is not None:
            # mask: (nW, N, N)
            # Expand mask to (Bw, num_heads, N, N) by repeating over batch and heads
            nW         = mask.shape[0]
            B          = Bw // nW
            # repeat mask for each sample in batch, then for each head
            mask_exp   = mask.unsqueeze(0).unsqueeze(1)   # (1, 1, nW, N, N)
            mask_exp   = mask_exp.expand(B, self.num_heads, nW, N, N)
            # reshape to match attn: (B*nW, num_heads, N, N)
            mask_exp   = mask_exp.permute(0, 2, 1, 3, 4).contiguous()
            mask_exp   = mask_exp.view(Bw, self.num_heads, N, N)
            attn       = attn + mask_exp

        attn = self.attn_drop(F.softmax(attn, dim=-1))
        x    = (attn @ v).transpose(1, 2).reshape(Bw, N, C)
        return self.proj_drop(self.proj(x))


# ---------------------------------------------------------------------------
# Swin Block
# ---------------------------------------------------------------------------

class SwinBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int,
                 shift: bool, mlp_ratio: float, drop: float):
        super().__init__()
        self.ws    = window_size
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn  = WindowAttention(dim, window_size, num_heads,
                                     attn_drop=drop, proj_drop=drop)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, _, C  = x.shape
        shortcut = x

        # Clamp window size to actual feature map size
        ws = min(self.ws, H, W)

        x = self.norm1(x).view(B, H, W, C)

        # Pad so H, W are divisible by ws
        pad_b = (ws - H % ws) % ws
        pad_r = (ws - W % ws) % ws
        if pad_b or pad_r:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        # Cyclic shift — only when there are multiple windows
        n_windows = (Hp // ws) * (Wp // ws)
        do_shift  = self.shift and n_windows > 1

        if do_shift:
            shift = ws // 2
            x     = torch.roll(x, shifts=(-shift, -shift), dims=(1, 2))
            mask  = build_shift_mask(Hp, Wp, ws, x.device)
        else:
            mask = None

        # Window partition → attention → reverse
        windows = window_partition(x, ws).view(-1, ws * ws, C)
        windows = self.attn(windows, mask=mask)
        x       = window_reverse(windows.view(-1, ws, ws, C), ws, Hp, Wp)

        # Reverse cyclic shift
        if do_shift:
            x = torch.roll(x, shifts=(shift, shift), dims=(1, 2))

        # Remove padding
        if pad_b or pad_r:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C) + shortcut
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Swin Stage
# ---------------------------------------------------------------------------

class SwinStage(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int,
                 depth: int, mlp_ratio: float, drop: float, downsample: bool):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(dim, num_heads, window_size,
                      shift=(i % 2 == 1), mlp_ratio=mlp_ratio, drop=drop)
            for i in range(depth)
        ])
        self.downsample = nn.Sequential(
            nn.LayerNorm(4 * dim),
            nn.Linear(4 * dim, 2 * dim),
        ) if downsample else None

    def forward(self, x: torch.Tensor, H: int, W: int):
        for blk in self.blocks:
            x = blk(x, H, W)

        B, _, C = x.shape
        feat    = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        if self.downsample is not None:
            x2d    = x.view(B, H, W, C)
            x_merge = torch.cat([
                x2d[:, 0::2, 0::2, :],
                x2d[:, 1::2, 0::2, :],
                x2d[:, 0::2, 1::2, :],
                x2d[:, 1::2, 1::2, :],
            ], dim=-1)
            x_down = self.downsample(x_merge.view(B, -1, 4 * C))
            return feat, x_down, H // 2, W // 2

        return feat, x, H, W


# ---------------------------------------------------------------------------
# Full Swin Encoder
# ---------------------------------------------------------------------------

class SwinEncoder(nn.Module):
    """
    4-stage Swin Transformer encoder, from scratch.
    Accepts any in_channels. Returns 4 feature maps for decoder fusion.
    """

    def __init__(
        self,
        in_channels: int   = 3,
        embed_dim:   int   = 24,
        window_size: int   = 8,
        depths:      tuple = (2, 2, 6, 2),
        num_heads:   tuple = None,    # default computed from embed_dim below
        mlp_ratio:   float = 4.0,
        drop_rate:   float = 0.0,
    ):
        super().__init__()

        # Auto-compute num_heads if not provided:
        # pick the largest power-of-2 that divides embed_dim, up to 8
        if num_heads is None:
            base = min(8, embed_dim)
            while embed_dim % base != 0:
                base //= 2
            num_heads = tuple(
                min(base * (2 ** i), embed_dim * (2 ** i) // base)
                for i in range(4)
            )
            # Simpler: just use base heads, doubling with dims
            base_h = base
            num_heads = (base_h, base_h * 2, base_h * 4, base_h * 8)
            # Clamp so head_dim >= 4
            num_heads = tuple(
                max(1, min(nh, dims // 4))
                for nh, dims in zip(num_heads,
                                    [embed_dim * (2**i) for i in range(4)])
            )
        self.patch_conv = nn.Conv2d(in_channels, embed_dim, kernel_size=4, stride=4)
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.pos_drop   = nn.Dropout(drop_rate)

        dims = [embed_dim * (2 ** i) for i in range(4)]
        self.stages = nn.ModuleList([
            SwinStage(
                dim=dims[i], num_heads=num_heads[i],
                window_size=window_size, depth=depths[i],
                mlp_ratio=mlp_ratio, drop=drop_rate,
                downsample=(i < 3),
            )
            for i in range(4)
        ])
        self.out_channels = dims   # [embed_dim, 2e, 4e, 8e]

    def forward(self, x: torch.Tensor):
        x = self.patch_conv(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(self.patch_norm(x))

        feats = []
        for stage in self.stages:
            feat, x, H, W = stage(x, H, W)
            feats.append(feat)
        return feats
