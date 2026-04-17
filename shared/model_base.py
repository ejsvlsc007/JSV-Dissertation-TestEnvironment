"""
shared/model_base.py
====================
Shared decoder and DualEncoderBase class.

All D1-D7 model files inherit from DualEncoderBase and only need to:
  1. Call nn.Module.__init__(self) first
  2. Define self.cnn_encoder and self.transformer
  3. Call DualEncoderBase.__init__(self, ...) to build the decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Decoder building block
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class DualEncoderLoss(nn.Module):
    SMOOTH = 1e-5

    def __init__(self, dice_weight=0.5, aux_weight=0.4, pos_weight=500.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.aux_weight  = aux_weight
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

    def _single(self, logits, target):
        bce  = self.bce(logits, target)
        pred = torch.sigmoid(logits)
        inter = (pred * target).sum()
        dice = 1 - (2 * inter + self.SMOOTH) / (
            pred.sum() + target.sum() + self.SMOOTH
        )
        return (1 - self.dice_weight) * bce + self.dice_weight * dice

    def forward(self, output, target):
        if isinstance(output, (list, tuple)):
            main = self._single(output[0], target)
            aux  = sum(self._single(o, target) for o in output[1:])
            return main + self.aux_weight * aux
        return self._single(output, target)


Loss = DualEncoderLoss


# ---------------------------------------------------------------------------
# DualEncoderBase
# ---------------------------------------------------------------------------

class DualEncoderBase(nn.Module):
    """
    Base class for all D1-D7 dual-encoder segmentation models.

    Correct subclass pattern
    ------------------------
        class MyModel(DualEncoderBase):
            def __init__(self, in_channels=3, img_size=256, **cfg):
                nn.Module.__init__(self)               # MUST be first
                self.cnn_encoder = ...                 # then assign encoders
                self.transformer = ...
                DualEncoderBase.__init__(self,         # then build decoder
                    in_channels, img_size, **cfg)

    This explicit two-step init is required because DualEncoderBase needs
    self.cnn_encoder and self.transformer to already exist when it builds
    the alignment convs and decoder blocks.
    """

    def __init__(self, in_channels, img_size, deep_sup=True, **_):
        # NOTE: do NOT call super().__init__() here — nn.Module.__init__
        # must have already been called by the subclass before assigning
        # self.cnn_encoder and self.transformer.
        # We just build the decoder components using those existing attrs.

        self.deep_sup = deep_sup

        cnn_ch   = self.cnn_encoder.out_channels   # [c1, c2, c3, c4]
        trans_ch = self.transformer.out_channels   # [t1, t2, t3, t4]
        n_stages = len(cnn_ch)

        # 1x1 convs to align transformer channels to CNN channels
        self.align = nn.ModuleList([
            nn.Conv2d(tc, cc, 1)
            for tc, cc in zip(trans_ch, cnn_ch)
        ])

        # Decoder — coarse to fine
        dec_in = cnn_ch[-1]
        self.decoder_blocks = nn.ModuleList()
        self.aux_heads      = nn.ModuleList()
        for i in range(n_stages - 1, 0, -1):
            skip_ch = cnn_ch[i - 1]
            out_ch  = skip_ch
            self.decoder_blocks.append(DecoderBlock(dec_in, skip_ch, out_ch))
            self.aux_heads.append(nn.Conv2d(out_ch, 1, 1))
            dec_in = out_ch

        self.final_up   = nn.ConvTranspose2d(dec_in, dec_in // 2, 2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(dec_in // 2, dec_in // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(dec_in // 2),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(dec_in // 2, 1, 1)

    def forward(self, x):
        orig_size = x.shape[2:]

        cnn_feats   = self.cnn_encoder(x)
        trans_feats = self.transformer(x)

        # Fuse: align transformer to CNN resolution then add
        fused = []
        for cf, tf, align in zip(cnn_feats, trans_feats, self.align):
            tf_r = F.interpolate(align(tf), size=cf.shape[2:],
                                 mode="bilinear", align_corners=False)
            fused.append(cf + tf_r)

        x = fused[-1]
        aux_logits = []
        for block, aux_head, skip in zip(
            self.decoder_blocks,
            self.aux_heads,
            reversed(fused[:-1]),
        ):
            x = block(x, skip)
            if self.deep_sup and self.training:
                aux_logits.append(
                    F.interpolate(aux_head(x), size=orig_size,
                                  mode="bilinear", align_corners=False)
                )

        x    = self.final_up(x)
        x    = F.interpolate(x, size=orig_size,
                              mode="bilinear", align_corners=False)
        x    = self.final_conv(x)
        main = self.head(x)

        if self.deep_sup and self.training:
            return [main] + aux_logits
        return main
