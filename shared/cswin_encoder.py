"""
shared/cswin_encoder.py
========================
From-scratch CSwin Transformer encoder used by all D*.2 models.

CSwin (Cross-Shaped Window) Transformer replaces the square window
attention of standard Swin with horizontal and vertical stripe attention
computed in parallel, then merged.  This gives each token a cross-shaped
receptive field that covers the full height OR full width of the feature
map without the quadratic cost of global attention.

Reference: "CSWin Transformer: A General Vision Transformer Backbone
with Cross-Shaped Windows" (Dong et al., CVPR 2022)

This implementation is from scratch (no pretrained weights) so
in_channels is unrestricted — compatible with any fusion strategy.

Architecture
------------
  Input (B, C, H, W)
  → Patch embedding (Conv2d stride 4) → (B, embed_dim, H/4, W/4)
  → CSwin Stage 1: LW-MSA → feat_1 + patch merging
  → CSwin Stage 2: LW-MSA → feat_2 + patch merging
  → CSwin Stage 3: LW-MSA → feat_3 + patch merging
  → CSwin Stage 4: LW-MSA → feat_4

Returns [feat_1, feat_2, feat_3, feat_4].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Locally-enhanced positional encoding (LePE)
# ---------------------------------------------------------------------------

class LePE(nn.Module):
    """Depthwise conv adds local positional encoding to value tokens."""

    def __init__(self, dim: int):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, v: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        v:   (B_total, N, head_dim)  where N = H * W
        H,W: spatial dims of the stripe (sp×W for horizontal, H×sp for vertical)
        """
        B_total, N, C = v.shape
        assert N == H * W, f"LePE: N={N} != H*W={H*W}"
        v2d  = v.transpose(1, 2).reshape(B_total, C, H, W)
        lepe = self.dw(v2d).reshape(B_total, C, N).transpose(1, 2)
        return v + lepe


# ---------------------------------------------------------------------------
# Cross-shaped window attention (horizontal + vertical stripes)
# ---------------------------------------------------------------------------

class CSWinAttention(nn.Module):
    """
    Cross-shaped window attention.

    Splits heads equally between horizontal and vertical stripe attention,
    then concatenates the outputs.  Each head in the H-group attends
    across the full width within a horizontal stripe of height `split_size`.
    V-group: full height within vertical stripe of width `split_size`.

    Args:
        dim:        embedding dimension.
        num_heads:  total heads (split evenly H/V).
        split_size: stripe width/height (analogous to window size in Swin).
        attn_drop:  attention dropout.
        proj_drop:  output projection dropout.
    """

    def __init__(self, dim: int, num_heads: int, split_size: int,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert num_heads % 2 == 0, "num_heads must be even for H/V split"
        self.dim        = dim
        self.num_heads  = num_heads
        self.split_size = split_size
        self.head_dim   = dim // num_heads
        self.scale      = self.head_dim ** -0.5
        self.half_heads = num_heads // 2

        self.qkv      = nn.Linear(dim, dim * 3)
        self.proj     = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # LePE for H and V groups — built with head_dim since each head
        # is processed independently (heads are merged into batch dim)
        self.lepe_h = LePE(self.head_dim)
        self.lepe_v = LePE(self.head_dim)

    def _stripe_attn(
        self,
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        lepe: LePE, H: int, W: int, sp: int, horizontal: bool,
    ) -> torch.Tensor:
        """
        Attention within stripes.

        q/k/v: (B, half_heads, H*W, head_dim)
        sp:    stripe size (clamped to feature map size by caller)
        Returns: (B, H*W, half_heads * head_dim)
        """
        B, nh, N, d = q.shape

        if horizontal:
            n_stripes = H // sp
            q = q.view(B, nh, n_stripes, sp, W, d).permute(0, 2, 1, 3, 4, 5)
            k = k.view(B, nh, n_stripes, sp, W, d).permute(0, 2, 1, 3, 4, 5)
            v = v.view(B, nh, n_stripes, sp, W, d).permute(0, 2, 1, 3, 4, 5)
            q = q.reshape(B * n_stripes, nh, sp * W, d)
            k = k.reshape(B * n_stripes, nh, sp * W, d)
            v = v.reshape(B * n_stripes, nh, sp * W, d)
        else:
            n_stripes = W // sp
            q = q.view(B, nh, H, n_stripes, sp, d).permute(0, 3, 1, 2, 4, 5)
            k = k.view(B, nh, H, n_stripes, sp, d).permute(0, 3, 1, 2, 4, 5)
            v = v.view(B, nh, H, n_stripes, sp, d).permute(0, 3, 1, 2, 4, 5)
            q = q.reshape(B * n_stripes, nh, H * sp, d)
            k = k.reshape(B * n_stripes, nh, H * sp, d)
            v = v.reshape(B * n_stripes, nh, H * sp, d)

        # LePE: merge batch+stripes+heads so LePE sees (total, Ns, head_dim)
        Bw, nh2, Ns, d2 = v.shape
        v_flat = v.reshape(Bw * nh2, Ns, d2)
        if horizontal:
            v_flat = lepe(v_flat, sp, W)
        else:
            v_flat = lepe(v_flat, H, sp)
        v_lepe = v_flat.reshape(Bw, nh2, Ns, d2)

        attn = self.attn_drop(
            F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        )
        out = (attn @ v_lepe)   # (B*n_stripes, nh, Ns, d)

        # Reconstruct (B, H*W, nh*d)
        if horizontal:
            out = out.view(B, n_stripes, nh, sp, W, d)
            out = out.permute(0, 2, 1, 3, 4, 5).reshape(B, nh, H, W, d)
        else:
            out = out.view(B, n_stripes, nh, H, sp, d)
            out = out.permute(0, 2, 3, 1, 4, 5).reshape(B, nh, H, W, d)

        return out.permute(0, 2, 3, 1, 4).reshape(B, H * W, nh * d)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """x: (B, H*W, dim)"""
        B, N, C = x.shape
        H_orig, W_orig = H, W

        # Clamp split_size to actual feature map dims
        sp = min(self.split_size, H, W)

        # Ensure H and W are divisible by sp — pad if needed
        pad_h = (sp - H % sp) % sp
        pad_w = (sp - W % sp) % sp
        if pad_h or pad_w:
            x = x.view(B, H, W, C)
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H = H + pad_h
            W = W + pad_w
            x = x.view(B, H * W, C)

        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # q/k/v: (B, num_heads, H*W, head_dim)

        # Split heads: first half → horizontal, second half → vertical
        q_h, q_v = q[:, :self.half_heads], q[:, self.half_heads:]
        k_h, k_v = k[:, :self.half_heads], k[:, self.half_heads:]
        v_h, v_v = v[:, :self.half_heads], v[:, self.half_heads:]

        out_h = self._stripe_attn(q_h, k_h, v_h, self.lepe_h, H, W, sp, horizontal=True)
        out_v = self._stripe_attn(q_v, k_v, v_v, self.lepe_v, H, W, sp, horizontal=False)

        out = torch.cat([out_h, out_v], dim=-1)    # (B, H*W, dim)

        # Crop back to original size if we padded
        if pad_h or pad_w:
            out = out.view(B, H, W, C)
            out = out[:, :H_orig, :W_orig, :].contiguous()
            out = out.view(B, H_orig * W_orig, C)

        return self.proj_drop(self.proj(out))


# ---------------------------------------------------------------------------
# CSwin Block
# ---------------------------------------------------------------------------

class CSwinBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, split_size: int,
                 mlp_ratio: float, drop: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn  = CSWinAttention(dim, num_heads, split_size,
                                    attn_drop=drop, proj_drop=drop)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# CSwin Stage
# ---------------------------------------------------------------------------

class CSwinStage(nn.Module):
    def __init__(self, dim: int, num_heads: int, split_size: int,
                 depth: int, mlp_ratio: float, drop: float,
                 downsample: bool):
        super().__init__()
        self.blocks = nn.ModuleList([
            CSwinBlock(dim, num_heads, split_size, mlp_ratio, drop)
            for _ in range(depth)
        ])
        # Patch merging: 2×2 neighbourhood → 4C → 2C
        self.merge = nn.Sequential(
            nn.LayerNorm(4 * dim),
            nn.Linear(4 * dim, 2 * dim),
        ) if downsample else None

    def forward(self, x: torch.Tensor, H: int, W: int):
        for blk in self.blocks:
            x = blk(x, H, W)

        B, _, C = x.shape
        feat = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        if self.merge is not None:
            x2d = x.view(B, H, W, C)
            x_m = torch.cat([
                x2d[:, 0::2, 0::2, :],
                x2d[:, 1::2, 0::2, :],
                x2d[:, 0::2, 1::2, :],
                x2d[:, 1::2, 1::2, :],
            ], dim=-1).view(B, -1, 4 * C)
            x_down = self.merge(x_m)
            return feat, x_down, H // 2, W // 2
        return feat, x, H, W


# ---------------------------------------------------------------------------
# Full CSwin Encoder
# ---------------------------------------------------------------------------

class CSwinEncoder(nn.Module):
    """
    4-stage CSwin Transformer encoder.

    Returns a list of 4 feature maps matching SwinEncoder's interface
    so all model files can swap between them with one import change.

    Args:
        in_channels: unrestricted (no pretrained weights).
        embed_dim:   base embedding dimension.
        split_sizes: stripe size per stage (decreasing is standard).
        depths:      number of CSwin blocks per stage.
        num_heads:   attention heads per stage (must be even for H/V split).
        mlp_ratio:   FFN expansion ratio.
        drop_rate:   dropout.
    """

    def __init__(
        self,
        in_channels:  int   = 3,
        embed_dim:    int   = 32,
        split_sizes:  tuple = (1, 2, 7, 7),
        depths:       tuple = (1, 2, 21, 1),
        num_heads:    tuple = (2, 4, 8, 16),
        mlp_ratio:    float = 4.0,
        drop_rate:    float = 0.0,
    ):
        super().__init__()
        # Patch embedding
        self.patch_conv = nn.Conv2d(in_channels, embed_dim, kernel_size=4, stride=4)
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.pos_drop   = nn.Dropout(drop_rate)

        dims = [embed_dim * (2 ** i) for i in range(4)]
        self.stages = nn.ModuleList()
        for i in range(4):
            self.stages.append(CSwinStage(
                dim        = dims[i],
                num_heads  = num_heads[i],
                split_size = split_sizes[i],
                depth      = depths[i],
                mlp_ratio  = mlp_ratio,
                drop       = drop_rate,
                downsample = (i < 3),
            ))

        self.out_channels = dims   # [embed_dim, 2e, 4e, 8e]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.patch_conv(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(self.patch_norm(x))

        feats = []
        for stage in self.stages:
            feat, x, H, W = stage(x, H, W)
            feats.append(feat)
        return feats
