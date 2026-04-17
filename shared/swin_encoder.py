"""
shared/swin_encoder.py
======================
From-scratch Swin Transformer encoder used by all D*.1 models.

Accepts any number of input channels (no pretrained weight constraint),
outputs a 4-stage feature pyramid matching the CNN encoder's spatial
resolution at each stage.

Architecture
------------
  Input (B, C, H, W)
  → Patch embedding (Conv2d stride 4) → (B, embed_dim, H/4, W/4)
  → Stage 1: SwinStage → feat_1 (B, embed_dim,    H/4,  W/4)  + downsample
  → Stage 2: SwinStage → feat_2 (B, embed_dim*2,  H/8,  W/8)  + downsample
  → Stage 3: SwinStage → feat_3 (B, embed_dim*4,  H/16, W/16) + downsample
  → Stage 4: SwinStage → feat_4 (B, embed_dim*8,  H/32, W/32)

Returns [feat_1, feat_2, feat_3, feat_4] for skip connection fusion
with the CNN encoder.

Key design decisions
--------------------
- Window attention with cyclic shift (standard Swin).
- From scratch: in_channels is unrestricted.
- Spatial size must be divisible by (patch_size × window_size).
  Default: patch_size=4, window_size=8 → divisible by 32.
  IMAGE_SIZE=256 satisfies this.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Relative position bias table
# ---------------------------------------------------------------------------

class RelativePositionBias(nn.Module):
    def __init__(self, window_size: int, num_heads: int):
        super().__init__()
        self.ws  = window_size
        self.num_heads = num_heads
        # Table: (2W-1) × (2W-1) × num_heads
        self.table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.table, std=0.02)

        # Precompute relative position index
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(window_size),
                torch.arange(window_size),
                indexing="ij",
            )
        )                                           # (2, W, W)
        coords_flat = coords.flatten(1)             # (2, W²)
        rel         = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, W², W²)
        rel         = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        index = rel.sum(-1)                         # (W², W²)
        self.register_buffer("index", index)

    def forward(self) -> torch.Tensor:
        return self.table[self.index].permute(2, 0, 1).unsqueeze(0)
        # (1, num_heads, W², W²)


# ---------------------------------------------------------------------------
# Window partition helpers
# ---------------------------------------------------------------------------

def window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    """(B, H, W, C) → (B*nH*nW, ws, ws, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def window_reverse(windows: torch.Tensor, ws: int, H: int, W: int) -> torch.Tensor:
    """(B*nH*nW, ws, ws, C) → (B, H, W, C)"""
    B = int(windows.shape[0] / (H * W / ws / ws))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


# ---------------------------------------------------------------------------
# Window Multi-head Self-Attention (W-MSA / SW-MSA)
# ---------------------------------------------------------------------------

class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int,
                 qkv_bias: bool = True, attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        self.ws        = window_size
        self.num_heads = num_heads
        self.scale     = (dim // num_heads) ** -0.5

        self.qkv      = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj     = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rpb       = RelativePositionBias(window_size, num_heads)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """x: (B*nW, ws*ws, dim)"""
        Bw, N, C = x.shape
        qkv = self.qkv(x).reshape(Bw, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.rpb()

        if mask is not None:
            nW   = mask.shape[0]
            attn = attn.view(Bw // nW, nW, self.num_heads, N, N) + \
                   mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(F.softmax(attn, dim=-1))
        x    = (attn @ v).transpose(1, 2).reshape(Bw, N, C)
        return self.proj_drop(self.proj(x))


# ---------------------------------------------------------------------------
# Swin Transformer Block
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
        self.attn_mask    = None   # built lazily on first forward
        self._mask_hw     = (-1, -1)  # (Hp, Wp) used to build current mask

    def _build_mask(self, H: int, W: int, device: torch.device):
        ws = self.ws
        shift = ws // 2
        img_mask = torch.zeros(1, H, W, 1, device=device)
        slices_h = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        slices_w = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        cnt = 0
        for sh in slices_h:
            for sw in slices_w:
                img_mask[:, sh, sw, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, ws).squeeze(-1)  # (nW, ws*ws)
        attn_mask    = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask    = attn_mask.masked_fill(attn_mask != 0, -100.0) \
                                .masked_fill(attn_mask == 0, 0.0)
        return attn_mask   # (nW, ws², ws²)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """x: (B, H*W, C)"""
        B, _, C = x.shape
        ws      = self.ws

        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        # Pad to window size
        pad_b = (ws - H % ws) % ws
        pad_r = (ws - W % ws) % ws
        if pad_b or pad_r:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        # Cyclic shift
        if self.shift:
            shift = ws // 2
            x = torch.roll(x, shifts=(-shift, -shift), dims=(1, 2))
            if self.attn_mask is None or self._mask_hw != (Hp, Wp):
                self.attn_mask = self._build_mask(Hp, Wp, x.device)
                self._mask_hw  = (Hp, Wp)
            mask = self.attn_mask
        else:
            mask = None

        # Window partition → attention → reverse
        windows = window_partition(x, ws).view(-1, ws * ws, C)
        windows = self.attn(windows, mask=mask)
        x       = window_reverse(windows.view(-1, ws, ws, C), ws, Hp, Wp)

        # Reverse cyclic shift
        if self.shift:
            shift = ws // 2
            x = torch.roll(x, shifts=(shift, shift), dims=(1, 2))

        # Remove padding
        if pad_b or pad_r:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C) + shortcut
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Swin Stage (2 blocks: W-MSA + SW-MSA, then patch merging)
# ---------------------------------------------------------------------------

class SwinStage(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int,
                 depth: int, mlp_ratio: float, drop: float,
                 downsample: bool):
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
        self.out_dim = 2 * dim if downsample else dim

    def forward(self, x: torch.Tensor, H: int, W: int):
        """
        x: (B, H*W, dim)
        Returns: feat (B, dim, H, W),  x_down (B, (H/2)*(W/2), 2*dim)
        """
        for blk in self.blocks:
            x = blk(x, H, W)

        B, _, C = x.shape
        feat = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B,C,H,W)

        if self.downsample is not None:
            x2d    = x.view(B, H, W, C)
            # Patch merging: 2×2 → 4C → 2C
            x_merge = torch.cat([
                x2d[:, 0::2, 0::2, :],
                x2d[:, 1::2, 0::2, :],
                x2d[:, 0::2, 1::2, :],
                x2d[:, 1::2, 1::2, :],
            ], dim=-1)                              # (B, H/2, W/2, 4C)
            x_down = self.downsample(x_merge.view(B, -1, 4 * C))
            return feat, x_down, H // 2, W // 2
        return feat, x, H, W


# ---------------------------------------------------------------------------
# Full Swin Encoder
# ---------------------------------------------------------------------------

class SwinEncoder(nn.Module):
    """
    4-stage Swin Transformer encoder.

    Returns a list of 4 feature maps (one per stage) for fusion with
    the CNN encoder.  The spatial resolution halves each stage.

    Args:
        in_channels:  input channels (unrestricted — no pretrained weights).
        embed_dim:    base embedding dimension (default 24 keeps param count low).
        window_size:  attention window (8 for 256px input).
        depths:       number of Swin blocks per stage.
        num_heads:    attention heads per stage.
        mlp_ratio:    FFN expansion ratio.
        drop_rate:    dropout rate.
    """

    def __init__(
        self,
        in_channels: int  = 3,
        embed_dim:   int  = 24,
        window_size: int  = 8,
        depths:      tuple = (2, 2, 6, 2),
        num_heads:   tuple = (3, 6, 12, 24),
        mlp_ratio:   float = 4.0,
        drop_rate:   float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Patch embedding: stride-4 conv → H/4 × W/4
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=4, stride=4),
            nn.LayerNorm([embed_dim, 1, 1]),   # dummy shape — applied below
        )
        # Simpler: just a conv + separate norm
        self.patch_conv = nn.Conv2d(in_channels, embed_dim, kernel_size=4, stride=4)
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.pos_drop   = nn.Dropout(drop_rate)

        dims = [embed_dim * (2 ** i) for i in range(4)]
        self.stages = nn.ModuleList()
        for i in range(4):
            self.stages.append(SwinStage(
                dim        = dims[i],
                num_heads  = num_heads[i],
                window_size= window_size,
                depth      = depths[i],
                mlp_ratio  = mlp_ratio,
                drop       = drop_rate,
                downsample = (i < 3),   # no downsample after last stage
            ))

        self.out_channels = dims   # [embed_dim, 2e, 4e, 8e]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: (B, in_channels, H, W)

        Returns:
            list of 4 feature maps:
                [feat1 (B, embed_dim,   H/4,  W/4),
                 feat2 (B, embed_dim*2, H/8,  W/8),
                 feat3 (B, embed_dim*4, H/16, W/16),
                 feat4 (B, embed_dim*8, H/32, W/32)]
        """
        # Patch embedding
        x  = self.patch_conv(x)                          # (B, embed_dim, H/4, W/4)
        B, C, H, W = x.shape
        x  = x.flatten(2).transpose(1, 2)                # (B, H/4*W/4, embed_dim)
        x  = self.pos_drop(self.patch_norm(x))

        feats = []
        for stage in self.stages:
            result = stage(x, H, W)
            feat, x, H, W = result
            feats.append(feat)

        return feats   # 4 × (B, dim_i, H_i, W_i)
