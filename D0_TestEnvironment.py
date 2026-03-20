# ============================================================
# CaTNet_JSV.py
# CaT-Net adapted for JSV Dissertation Test Environment
#
# Drop this file into your GitHub repo and point
# MODEL_FILE = "CaTNet_JSV.py" in Cell 8.
#
# Key changes vs. original CaTNet.py
# ───────────────────────────────────
# 1. CaT_Net_with_Decoder_DeepSup now accepts:
#      in_channels   (1 for grayscale LiTS/MCT, 3 for RGB)
#      num_classes   (1 for binary liver segmentation)
#      img_size      (driven by IMAGE_SIZE param in notebook)
#      cnn_channels  (driven by CNN_CHANNELS param)
#      swin_channels (driven by SWIN_CHANNELS param)
#      num_layers    (driven by CNN_NUM_LAYERS param)
#      drop_rate     (driven by DROPOUT param)
# 2. Decoder_Deep_Supervision interpolates to img_size, not
#    hard-coded 224.
# 3. CaTNetLoss mirrors DECTNetLoss API:
#      criterion = CaTNetLoss(dice_weight, aux_weight)
#      loss = criterion(out, masks)   # out may be (pred, aux_tuple)
# 4. Model name exposed as CaT_Net_with_Decoder_DeepSup.__name__
#    so MODEL_ID tagging works identically to other variants.
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# ─────────────────────────────────────────────────────────────
# Core building blocks  (unchanged from original CaTNet.py)
# ─────────────────────────────────────────────────────────────

class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                               padding, dilation, groups=in_channels, bias=bias)
        self.point_wise_conv = nn.Conv2d(in_channels, out_channels,
                                         kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.point_wise_conv(x)
        return x


class _DenseLayer(nn.Module):
    def __init__(self, inplace, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate
        self.dense_layer = nn.ModuleList([
            nn.BatchNorm2d(inplace),
            nn.GELU(),
            SeparableConv(in_channels=inplace,
                          out_channels=bn_size * growth_rate,
                          kernel_size=1, bias=False),
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.GELU(),
            SeparableConv(in_channels=bn_size * growth_rate,
                          out_channels=growth_rate,
                          kernel_size=3, padding=1, bias=False),
        ])
        self.dropout = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        skip_x = x
        for blk in self.dense_layer:
            x = blk(x)
        if self.drop_rate > 0:
            x = self.dropout(x)
        return torch.cat([x, skip_x], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, inplances, growth_rate, bn_size, drop_rate=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            _DenseLayer(inplances + i * growth_rate, growth_rate, bn_size, drop_rate)
            for i in range(num_layers)
        ])

    def forward(self, x):
        for blk in self.layers:
            x = blk(x)
        return x


class _CBAMLayer(nn.Module):
    def __init__(self, channel, ratio=16):
        super().__init__()
        self.squeeze_avg = nn.AdaptiveAvgPool2d(1)
        self.squeeze_max = nn.AdaptiveMaxPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channel, int(channel // ratio)),
            nn.GELU(),
            nn.Linear(int(channel // ratio), channel),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
        self.conv = SeparableConv(in_channels=2, out_channels=1, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        y_avg = self.squeeze_avg(x).contiguous().view(b, c)
        y_max = self.squeeze_max(x).contiguous().view(b, c)
        z_avg = self.excitation(y_avg).contiguous().view(b, c, 1, 1)
        z_max = self.excitation(y_max).contiguous().view(b, c, 1, 1)
        z = self.sigmoid(z_avg + z_max)
        w = x * z.expand_as(x)
        s_avg = torch.mean(w, dim=1, keepdim=True)
        s_max, _ = torch.max(w, dim=1, keepdim=True)
        s = self.sigmoid(self.conv(torch.cat((s_avg, s_max), dim=1)))
        return w * s.expand_as(w)


class DenseCBAMBlock(nn.Module):
    def __init__(self, num_layers, inplances, channel, growth_rate,
                 bn_size, drop_rate=0., ratio=16):
        super().__init__()
        self.dense_layer = DenseBlock(num_layers, inplances, growth_rate, bn_size, drop_rate)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(inplances + num_layers * growth_rate),
            nn.GELU(),
            SeparableConv(inplances + num_layers * growth_rate, channel,
                          kernel_size=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.GELU()
        )
        self.cbam_layer = _CBAMLayer(channel, ratio)

    def forward(self, x):
        x = self.dense_layer(x)
        x = self.conv(x)
        x = self.cbam_layer(x)
        return x


class CNN_Encoder(nn.Module):
    """
    Dense-CBAM CNN encoder.

    Parameters
    ----------
    encoder_depth : int          — number of encoder stages (default 4)
    in_channels   : int          — image channels fed in (1 = grayscale; handled by
                                   the top-level model's channel-repeat logic)
    channels      : int          — base channel count (= cnn_channels // 2)
    num_layers    : list[int]    — dense layers per stage  (len == encoder_depth)
    growth_rate   : list[int]    — growth rate per stage   (len == encoder_depth)
    drop_rate     : float
    """
    def __init__(self, encoder_depth, in_channels, channels, num_layers,
                 growth_rate, drop_rate=0.):
        super().__init__()
        self.init_conv = SeparableConv(in_channels, channels, kernel_size=1, bias=False)
        self.layers = nn.ModuleList()
        for i in range(encoder_depth):
            layer = DenseCBAMBlock(
                num_layers=num_layers[i],
                inplances=channels * (2 ** i),
                channel=channels * (2 ** (i + 1)),
                growth_rate=growth_rate[i],
                bn_size=4,
                drop_rate=drop_rate
            )
            self.layers.append(layer)
            if i < encoder_depth - 1:
                down = SeparableConv(channels * (2 ** (i + 1)),
                                     channels * (2 ** (i + 1)),
                                     kernel_size=2, stride=2, bias=False)
                self.layers.append(down)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight, gain=1.0)

    def forward(self, x):
        res = []
        x = self.init_conv(x)
        for blk in self.layers:
            x = blk(x)
            if blk._get_name() == "DenseCBAMBlock":
                res.append(x)
        return res


# ─────────────────────────────────────────────────────────────
# Swin Transformer encoder  (unchanged from original)
# ─────────────────────────────────────────────────────────────

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop_rate=0.):
        super().__init__()
        out_features     = out_features or in_features
        hidden_features  = hidden_features or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
                   W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                        window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size
        self.num_heads   = num_heads
        head_dim         = dim // num_heads
        self.scale       = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords   = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten   = torch.flatten(coords, 1)
        relative_coords  = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords  = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax   = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn    = (q * self.scale) @ k.transpose(-2, -1)

        rpb = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1).permute(2, 0, 1).contiguous()
        attn = attn + rpb.unsqueeze(0)

        if mask is not None:
            nW   = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) \
                   + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim              = dim
        self.input_resolution = input_resolution
        self.num_heads        = num_heads
        self.window_size      = window_size
        self.shift_size       = shift_size
        self.mlp_ratio        = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size  = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.norm1     = norm_layer(dim)
        self.attn      = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2     = norm_layer(dim)
        self.mlp       = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop_rate=drop)

        if self.shift_size > 0:
            H, W      = self.input_resolution
            img_mask  = torch.zeros((1, H, W, 1))
            h_slices  = (slice(0, -self.window_size),
                         slice(-self.window_size, -self.shift_size),
                         slice(-self.shift_size, None))
            w_slices  = (slice(0, -self.window_size),
                         slice(-self.window_size, -self.shift_size),
                         slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask    = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask    = attn_mask.masked_fill(attn_mask != 0, -100.).masked_fill(attn_mask == 0, 0.)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W           = self.input_resolution
        B, H_x, W_x, C = x.shape
        assert H_x == H and W_x == W

        shortcut = x.view(B, -1, C)
        x        = self.norm1(shortcut).view(B, H, W, C)

        shifted_x = torch.roll(x, (-self.shift_size, -self.shift_size), (1, 2)) \
                    if self.shift_size > 0 else x
        x_windows  = window_partition(shifted_x, self.window_size)
        x_windows  = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_win   = self.attn(x_windows, mask=self.attn_mask)
        attn_win   = attn_win.view(-1, self.window_size, self.window_size, C)
        shifted_x  = window_reverse(attn_win, self.window_size, H, W)
        x          = torch.roll(shifted_x, (self.shift_size, self.shift_size), (1, 2)) \
                     if self.shift_size > 0 else shifted_x
        x          = x.view(B, H * W, C)
        x          = shortcut + self.drop_path(x)
        x          = x + self.drop_path(self.mlp(self.norm2(x)))
        return x.view(B, H, W, C)


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim       = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm      = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, H_x, W_x, C = x.shape
        assert H_x == H and W_x == W and H % 2 == 0 and W % 2 == 0
        x0, x1 = x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :]
        x2, x3 = x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x.view(B, H // 2, W // 2, 2 * C)


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 use_checkpoint=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class PatchEmbed(nn.Module):
    """
    Patch embedding.

    Parameters
    ----------
    img_size   : int   — spatial size of input image (from IMAGE_SIZE)
    patch_size : int   — patch stride (default 2)
    in_chans   : int   — channels *after* the grayscale→RGB repeat in the
                         top-level forward(); always 3 here.
    embed_dim  : int   — driven by SWIN_CHANNELS param
    """
    def __init__(self, img_size=224, patch_size=2, in_chans=3, embed_dim=48):
        super().__init__()
        img_size   = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0],
                               img_size[1] // patch_size[1]]
        self.img_size          = img_size
        self.patch_size        = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches       = patches_resolution[0] * patches_resolution[1]
        self.in_chans  = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            SeparableConv(in_chans,   embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim), nn.GELU(),
            SeparableConv(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim), nn.GELU(),
            SeparableConv(embed_dim, embed_dim, kernel_size=patch_size[0], stride=patch_size[0]),
            nn.BatchNorm2d(embed_dim), nn.GELU()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input size {H}×{W} ≠ model size {self.img_size[0]}×{self.img_size[1]}"
        return self.proj(x).permute(0, 2, 3, 1).contiguous()


class SwinTransformerEncoder(nn.Module):
    """
    Swin encoder whose spatial resolution is driven by img_size.

    Parameters
    ----------
    img_size        : int   — from IMAGE_SIZE notebook param
    patch_size      : int   — default 2
    in_chans        : int   — always 3 (grayscale repeated upstream)
    embed_dim       : int   — from SWIN_CHANNELS notebook param
    depths          : list  — blocks per stage  (default [2,2,6,2])
    num_heads       : list  — heads per stage   (default [3,6,12,24])
    window_size     : int   — from WINDOW_SIZE notebook param (7 for 224px)
    mlp_ratio       : float — from MLP_RATIO notebook param
    drop_rate       : float — from DROPOUT notebook param
    attn_drop_rate  : float — from DROPOUT notebook param
    drop_path_rate  : float — stochastic depth rate (default 0.1)
    """
    def __init__(self, img_size=224, patch_size=2, in_chans=3,
                 embed_dim=48, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False):
        super().__init__()
        self.num_layers  = len(depths)
        self.embed_dim   = embed_dim
        self.mlp_ratio   = mlp_ratio
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, self.patch_embed.num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(BasicLayer(
                dim=int(embed_dim * 2 ** i),
                input_resolution=(patches_resolution[0] // (2 ** i),
                                   patches_resolution[1] // (2 ** i)),
                depth=depths[i], num_heads=num_heads[i],
                window_size=window_size, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer, use_checkpoint=use_checkpoint))
            if i < self.num_layers - 1:
                self.layers.append(PatchMerging(
                    input_resolution=(patches_resolution[0] // (2 ** i),
                                      patches_resolution[1] // (2 ** i)),
                    dim=int(embed_dim * 2 ** i), norm_layer=norm_layer))

        self.norm = norm_layer(int(embed_dim * 2 ** (self.num_layers - 1)))

    def forward(self, x):
        res = []
        x   = self.pos_drop(self.patch_embed(x))
        for layer in self.layers:
            x = layer(x)
            if layer._get_name() == "BasicLayer":
                res.append(x.permute(0, 3, 1, 2))
        return res


# ─────────────────────────────────────────────────────────────
# Decoder
# ─────────────────────────────────────────────────────────────

class FeatureFuse(nn.Module):
    def __init__(self, input_channels, vit_channels, cnn_channels, out_channels):
        super().__init__()
        if input_channels is not None:
            self.signal = 0
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(input_channels + vit_channels,
                                   input_channels + vit_channels,
                                   kernel_size=2, stride=2),
                nn.BatchNorm2d(input_channels + vit_channels), nn.GELU())
            self.channel_fuse = nn.Sequential(
                SeparableConv(input_channels + vit_channels + cnn_channels,
                              out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels), nn.GELU())
        else:
            self.signal = 1
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(vit_channels, vit_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(vit_channels), nn.GELU())
            self.channel_fuse = nn.Sequential(
                SeparableConv(vit_channels + cnn_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels), nn.GELU())

    def forward(self, input_x, vit_x, cnn_x):
        if self.signal == 1:
            x = torch.cat([self.upsample(vit_x), cnn_x], dim=1)
        else:
            x = torch.cat([self.upsample(torch.cat([input_x, vit_x], dim=1)), cnn_x], dim=1)
        return self.channel_fuse(x)


class SEModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super().__init__()
        self.squeeze    = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channel, int(channel // ratio)),
            nn.GELU(),
            nn.Linear(int(channel // ratio), channel),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        z = self.excitation(self.squeeze(x).view(b, c)).view(b, c, 1, 1)
        return x * z.expand_as(x)


def Conv3X3BNGELU(in_channels, out_channels):
    return nn.Sequential(
        SeparableConv(in_channels,  out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels), nn.GELU(),
        SeparableConv(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels))


class ConvPipeline(nn.Module):
    def __init__(self, in_channels, out_channels, attn_ratio=16):
        super().__init__()
        self.conv_layer = Conv3X3BNGELU(in_channels, out_channels)
        self.se_layer   = SEModule(out_channels, attn_ratio)
        self.gelu       = nn.GELU()

    def forward(self, x):
        return self.se_layer(self.gelu(x + self.conv_layer(x)))


class DecoderLayer(nn.Module):
    def __init__(self, input_channels, vit_channels, cnn_channels, out_channels):
        super().__init__()
        self.feature_fuse    = FeatureFuse(input_channels, vit_channels, cnn_channels, out_channels)
        self.channel_pipeline = ConvPipeline(out_channels, out_channels)

    def forward(self, input_x, vit_x, cnn_x):
        return self.channel_pipeline(self.feature_fuse(input_x, vit_x, cnn_x))


class Decoder(nn.Module):
    def __init__(self, depths, num_classes, input_channels,
                 vit_channels, cnn_channels, out_channels):
        super().__init__()
        self.depths = depths
        self.decoder_layers = nn.ModuleList()
        for i in range(depths):
            self.decoder_layers.append(DecoderLayer(
                input_channels=None if i == 0 else input_channels * (2 ** (depths - i)),
                cnn_channels=cnn_channels * (2 ** (depths - (i + 1))),
                vit_channels=vit_channels * (2 ** (depths - (i + 1))),
                out_channels=out_channels * (2 ** (depths - (i + 1)))))
        self.classifier = SeparableConv(out_channels, num_classes, kernel_size=1, bias=True)

    def forward(self, vit_features, cnn_features):
        res = []
        x   = None
        for i in range(self.depths):
            x = self.decoder_layers[i](
                None if i == 0 else x,
                vit_features[-(i + 1)],
                cnn_features[-(i + 1)])
            res.append(x)
        return self.classifier(x), res


# ─────────────────────────────────────────────────────────────
# Deep Supervision head
# ─────────────────────────────────────────────────────────────

class Decoder_Deep_Supervision(nn.Module):
    """
    Produces three auxiliary logits from decoder intermediate feature maps.

    The output size is interpolated to `img_size` (not hard-coded 224)
    so the module works with any IMAGE_SIZE.

    Auxiliary output order (returned as a 3-tuple):
        out1  ← decoder stage 3  (cnn_channels * 2)
        out2  ← decoder stage 2  (cnn_channels * 4)
        out3  ← decoder stage 1  (cnn_channels * 8)

    This ordering is *descending channel count* (matches decoder res list),
    consistent with the aux_head ordering rule in the notebook.
    """
    def __init__(self, num_classes, cnn_channels, img_size=224):
        super().__init__()
        self.img_size = img_size

        # Stage 1 of decoder  →  cnn_channels * 8  (deepest, fewest spatial tokens)
        self.fm3_conv = nn.Sequential(
            SeparableConv(cnn_channels * (2 ** 3), cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels), nn.ReLU(inplace=True))
        self.fm3_cls  = SeparableConv(cnn_channels, num_classes, kernel_size=1)

        # Stage 2 of decoder  →  cnn_channels * 4
        self.fm2_conv = nn.Sequential(
            SeparableConv(cnn_channels * (2 ** 2), cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels), nn.ReLU(inplace=True))
        self.fm2_cls  = SeparableConv(cnn_channels, num_classes, kernel_size=1)

        # Stage 3 of decoder  →  cnn_channels * 2
        self.fm1_conv = nn.Sequential(
            SeparableConv(cnn_channels * (2 ** 1), cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels), nn.ReLU(inplace=True))
        self.fm1_cls  = SeparableConv(cnn_channels, num_classes, kernel_size=1)

    def _up(self, x):
        return F.interpolate(x, size=(self.img_size, self.img_size),
                             mode='bilinear', align_corners=False)

    def forward(self, res):
        # res[0] = deepest (stage 1), res[1] = stage 2, res[2] = stage 3
        out3 = self.fm3_cls(self._up(self.fm3_conv(res[0])))
        out2 = self.fm2_cls(self._up(self.fm2_conv(res[1])))
        out1 = self.fm1_cls(self._up(self.fm1_conv(res[2])))
        # Return descending channel order (deepest first) — consistent with
        # the aux_head ordering convention in the training loop
        return out3, out2, out1


# ─────────────────────────────────────────────────────────────
# Top-level model
# ─────────────────────────────────────────────────────────────

class CaT_Net_with_Decoder_DeepSup(nn.Module):
    """
    CaT-Net for liver segmentation — JSV dissertation variant.

    Notebook parameters → constructor arguments
    ───────────────────────────────────────────
    in_channels   ← 1  (LiTS / MCT are grayscale; channels are repeated to 3 internally)
    num_classes   ← 1  (binary: liver vs. background)
    img_size      ← IMAGE_SIZE
    cnn_channels  ← CNN_CHANNELS   (default 32)
    swin_channels ← SWIN_CHANNELS  (default 24)
    num_layers    ← CNN_NUM_LAYERS (default [4,4,4,4])
    window_size   ← WINDOW_SIZE    (7 for 224 px, 8 for 256 px)
    mlp_ratio     ← MLP_RATIO
    drop_rate     ← DROPOUT
    """
    def __init__(self,
                 in_channels=1,
                 num_classes=1,
                 img_size=224,
                 cnn_channels=32,
                 swin_channels=24,
                 num_layers=(4, 4, 4, 4),
                 window_size=7,
                 mlp_ratio=4.,
                 drop_rate=0.1):
        super().__init__()
        self.img_size = img_size

        # CNN encoder — accepts 3-channel input (grayscale is repeated below)
        self.cnn_encoder = CNN_Encoder(
            encoder_depth=4,
            in_channels=3,
            channels=cnn_channels // 2,
            num_layers=list(num_layers),
            growth_rate=[cnn_channels // 2] * 4,
            drop_rate=drop_rate)

        # Swin encoder
        self.swin_encoder = SwinTransformerEncoder(
            img_size=img_size,
            patch_size=2,
            in_chans=3,
            embed_dim=swin_channels,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=drop_rate,
            drop_path_rate=0.1)

        # Dual-stream decoder
        self.decoder = Decoder(
            depths=4,
            num_classes=num_classes,
            input_channels=cnn_channels,
            vit_channels=swin_channels,
            cnn_channels=cnn_channels,
            out_channels=cnn_channels)

        # Deep supervision heads
        self.deep_sup = Decoder_Deep_Supervision(
            num_classes=num_classes,
            cnn_channels=cnn_channels,
            img_size=img_size)

    def forward(self, x):
        # Grayscale → pseudo-RGB (matches LiTS / MCT pipeline)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        cnn_feats  = self.cnn_encoder(x)
        swin_feats = self.swin_encoder(x)
        pred, res  = self.decoder(swin_feats, cnn_feats)
        aux_outs   = self.deep_sup(res)         # (out3, out2, out1)
        return pred, aux_outs


# ─────────────────────────────────────────────────────────────
# Loss function — mirrors DECTNetLoss API exactly
# ─────────────────────────────────────────────────────────────

class CaTNetLoss(nn.Module):
    """
    Combined Dice + BCE loss with optional deep supervision.

    Parameters
    ----------
    dice_weight : float — proportion of Dice loss in the main term
                          (1 − dice_weight) goes to BCE.
                          Driven by DICE_WEIGHT notebook param.
    aux_weight  : float — total weight of auxiliary outputs.
                          Driven by AUX_WEIGHT notebook param.
    smooth      : float — Dice smoothing constant

    Usage (identical to DECTNetLoss)
    ──────────────────────────────────
        criterion = CaTNetLoss(dice_weight=DICE_WEIGHT, aux_weight=AUX_WEIGHT)
        loss      = criterion(out, masks)   # out = (pred, (aux3, aux2, aux1))
    """
    def __init__(self, dice_weight=0.5, aux_weight=0.4, smooth=1e-5):
        super().__init__()
        self.dice_weight = dice_weight
        self.aux_weight  = aux_weight
        self.smooth      = smooth
        self.bce         = nn.BCEWithLogitsLoss()

    def _dice_loss(self, logits, target):
        probs        = torch.sigmoid(logits)
        probs_flat   = probs.view(-1)
        target_flat  = target.view(-1)
        intersection = (probs_flat * target_flat).sum()
        return 1. - (2. * intersection + self.smooth) / \
                    (probs_flat.sum() + target_flat.sum() + self.smooth)

    def _combined(self, logits, target):
        return (self.dice_weight       * self._dice_loss(logits, target) +
                (1 - self.dice_weight) * self.bce(logits, target))

    def forward(self, output, target):
        # output may be a raw tensor (eval) or (pred, aux_tuple) (train)
        if isinstance(output, tuple):
            pred, aux_outs = output[0], output[1]
        else:
            return self._combined(output, target)

        main_loss = self._combined(pred, target)

        # aux_outs is a tuple of 3 logit tensors at full resolution
        n_aux     = len(aux_outs)
        aux_loss  = sum(self._combined(a, target) for a in aux_outs) / max(n_aux, 1)

        return (1 - self.aux_weight) * main_loss + self.aux_weight * aux_loss


# ─────────────────────────────────────────────────────────────
# Quick smoke-test (run as __main__)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = CaT_Net_with_Decoder_DeepSup(
        in_channels=1,
        num_classes=1,
        img_size=224,
        cnn_channels=32,
        swin_channels=24,
        num_layers=(4, 4, 4, 4),
        window_size=7,
        mlp_ratio=4.,
        drop_rate=0.1)

    x             = torch.randn(2, 1, 224, 224)
    pred, aux_outs = model(x)
    criterion     = CaTNetLoss(dice_weight=0.5, aux_weight=0.4)
    target        = torch.zeros(2, 1, 224, 224)
    loss          = criterion((pred, aux_outs), target)

    print(f"pred shape  : {pred.shape}")
    print(f"aux shapes  : {[a.shape for a in aux_outs]}")
    print(f"loss        : {loss.item():.4f}")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total:,}")
    print(f"Trainable params: {trainable:,}")
