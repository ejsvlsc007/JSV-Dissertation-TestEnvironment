"""
shared/phase_fusion.py

Two genuine phase-fusion mechanisms to challenge F0 (early / naive stacking),
each operating at a DIFFERENT stage of the network:

  • CrossPhaseAttentionFusion  — INTERMEDIATE (feature-level) fusion.
        Fuses per-phase BOTTLENECK feature maps with bidirectional cross-phase
        attention. Use when phases still share spatial correspondence (they do
        at the bottleneck), so each phase can attend to the others to model the
        NC→ART→PVP enhancement dynamics explicitly.

  • LearnedPhaseWeightFusion    — LATE (decision-level) fusion.
        Combines per-phase OUTPUT maps (logits or final features) with
        softmax-normalised per-phase weights. Correct for the decision stage,
        where dense cross-phase spatial correspondence has mostly dissolved and
        a lightweight weighting is the right tool.

------------------------------------------------------------------------------
DESIGN CONSTRAINTS baked in (these are what make the comparison fair + safe):

  1. WEIGHT SHARING is the model's responsibility, not the fusion module's.
     Run ONE shared encoder over each phase (a loop), do NOT instantiate three
     separate encoders — otherwise the intermediate/late models get ~3x the
     encoder parameters of F0 and any measured difference is capacity, not
     fusion stage. `split_phases()` + a Python loop over the SAME module is the
     intended pattern (see the integration sketch at the bottom).

  2. NO MULTIPLICATIVE GATES THAT CAN ZERO A PHASE. The late module normalises
     weights with a softmax ACROSS phases (sum to 1, all > 0), so no phase can
     be driven to an exact zero — this avoids the collapse seen with the
     independent sigmoid gates of F1.

  3. SAME DATA PIPELINE AS F0. The model still receives the F0-stacked
     (B, 3, H, W) tensor from the slice cache; `split_phases()` slices it back
     into three (B, 1, H, W) phase tensors inside the model. So the dataloader,
     cache, k-fold splits and metrics are identical across F0 / intermediate /
     late — fusion stage is the only moving variable.
------------------------------------------------------------------------------
"""
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Helper: split the F0-stacked input back into per-phase tensors
# --------------------------------------------------------------------------- #
def split_phases(x: torch.Tensor, num_phases: int = 3) -> List[torch.Tensor]:
    """(B, num_phases, H, W) -> list of num_phases tensors (B, 1, H, W).

    For per-phase encoding through a 3-channel pretrained backbone, replicate
    each single channel to 3 with `.repeat(1, 3, 1, 1)`, or build the shared
    encoder with in_channels=1.
    """
    assert x.shape[1] == num_phases, (
        f"expected {num_phases} channels (NC/ART/PVP), got {x.shape[1]}")
    return [x[:, i:i + 1] for i in range(num_phases)]


# --------------------------------------------------------------------------- #
#  INTERMEDIATE — bidirectional cross-phase attention at the bottleneck
# --------------------------------------------------------------------------- #
STAGE_INTERMEDIATE = "intermediate"


class CrossPhaseAttentionFusion(nn.Module):
    """Bidirectional cross-phase attention over per-phase feature maps.

    Input  : list of K tensors (B, C, H, W), one per phase, from a SHARED encoder.
    Output : fused (B, C, H, W).

    At every spatial location the K phase-vectors attend to one another
    (full K×K multi-head attention → bidirectional), followed by a residual
    channel MLP. The refined phases are then aggregated with learned,
    softmax-normalised phase weights. Compute is O(B·H·W·K²·C); cheap because
    it is applied at the bottleneck where H·W is small.
    """

    def __init__(self, channels: int, num_phases: int = 3, num_heads: int = 4,
                 mlp_ratio: float = 2.0, drop: float = 0.0):
        super().__init__()
        if channels % num_heads != 0:
            # fall back to the largest head count that divides `channels`
            num_heads = next((h for h in (8, 4, 2, 1) if channels % h == 0), 1)
        self.channels = channels
        self.num_phases = num_phases
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3, bias=True)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(drop)

        self.norm2 = nn.LayerNorm(channels)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, channels), nn.Dropout(drop),
        )
        # learned aggregation weights across phases (softmax-normalised)
        self.phase_logits = nn.Parameter(torch.zeros(num_phases))

    def _attn(self, t: torch.Tensor) -> torch.Tensor:
        # t: (N, K, C)
        N, K, C = t.shape
        qkv = self.qkv(t).reshape(N, K, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # (3, N, heads, K, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale   # (N, heads, K, K)
        attn = attn.softmax(dim=-1)
        out = attn @ v                            # (N, heads, K, head_dim)
        out = out.transpose(1, 2).reshape(N, K, C)
        return self.proj_drop(self.proj(out))

    def forward(self, phases: List[torch.Tensor]) -> torch.Tensor:
        assert len(phases) == self.num_phases
        B, C, H, W = phases[0].shape
        # (B, C, H, W) × K -> (B, H, W, K, C) -> (N=B*H*W, K, C)
        t = torch.stack(phases, dim=1)            # (B, K, C, H, W)
        t = t.permute(0, 3, 4, 1, 2).reshape(B * H * W, self.num_phases, C)

        t = t + self._attn(self.norm1(t))         # cross-phase attention + residual
        t = t + self.mlp(self.norm2(t))           # channel MLP + residual

        t = t.reshape(B, H, W, self.num_phases, C)
        w = self.phase_logits.softmax(dim=0).view(1, 1, 1, self.num_phases, 1)
        fused = (t * w).sum(dim=3)                # (B, H, W, C)
        return fused.permute(0, 3, 1, 2).contiguous()   # (B, C, H, W)


def fuse_skip_features(skip_lists: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """Fuse per-phase skip connections for the shared decoder.

    skip_lists: K lists, each with the same L encoder skip tensors.
    Returns L fused skips (mean across phases — cheap and stable; swap for a
    per-level CrossPhaseAttentionFusion if you want attention on skips too).
    """
    K = len(skip_lists)
    L = len(skip_lists[0])
    return [torch.stack([skip_lists[k][i] for k in range(K)], 0).mean(0)
            for i in range(L)]


# --------------------------------------------------------------------------- #
#  LATE — learned, softmax-normalised per-phase weighting of outputs
# --------------------------------------------------------------------------- #
STAGE_LATE = "late"


class LearnedPhaseWeightFusion(nn.Module):
    """Decision-level fusion of per-phase output maps.

    Input  : list of K tensors (B, C, H, W), one per phase (logits or final feats).
    Output : fused (B, C, H, W).

    mode='scalar'  : K learnable scalars, softmax across phases (global per-phase
                     importance). Fewest parameters, the safest baseline.
    mode='spatial' : content-adaptive per-pixel weights from a 1×1 conv over the
                     concatenated phase maps, softmax across phases. More
                     expressive, still normalised so no phase can be zeroed.

    Softmax-over-phases is the key safety property: weights are positive and sum
    to 1, so unlike F1's independent sigmoid gates a phase cannot be suppressed
    to exactly zero → no fusion collapse.
    """

    def __init__(self, channels: int = 1, num_phases: int = 3, mode: str = "spatial"):
        super().__init__()
        assert mode in ("scalar", "spatial")
        self.mode = mode
        self.num_phases = num_phases
        if mode == "scalar":
            self.phase_logits = nn.Parameter(torch.zeros(num_phases))
        else:
            self.weight_conv = nn.Conv2d(num_phases * channels, num_phases,
                                         kernel_size=1)

    def forward(self, phases: List[torch.Tensor]) -> torch.Tensor:
        assert len(phases) == self.num_phases
        stacked = torch.stack(phases, dim=1)              # (B, K, C, H, W)
        if self.mode == "scalar":
            w = self.phase_logits.softmax(dim=0).view(1, self.num_phases, 1, 1, 1)
            return (stacked * w).sum(dim=1)               # (B, C, H, W)
        # spatial: per-pixel weights
        B, K, C, H, W = stacked.shape
        cat = stacked.reshape(B, K * C, H, W)
        w = self.weight_conv(cat).softmax(dim=1)          # (B, K, H, W)
        w = w.unsqueeze(2)                                # (B, K, 1, H, W)
        return (stacked * w).sum(dim=1)                   # (B, C, H, W)

    def phase_importance(self) -> torch.Tensor:
        """Diagnostic: current global phase weights (scalar mode only)."""
        if self.mode == "scalar":
            return self.phase_logits.softmax(dim=0).detach()
        raise RuntimeError("phase_importance() is only defined for mode='scalar'")


# --------------------------------------------------------------------------- #
#  Integration sketch (DualEncoderBase models)
# --------------------------------------------------------------------------- #
"""
INTERMEDIATE model forward (one SHARED dual-encoder, fuse at bottleneck):

    phases = split_phases(x)                      # 3 × (B,1,H,W)  from F0 cache
    enc_out = [self.shared_encoder(p) for p in phases]   # SAME module, shared weights
    bott  = self.cpa([e.bottleneck for e in enc_out])    # CrossPhaseAttentionFusion
    skips = fuse_skip_features([e.skips for e in enc_out])
    logits = self.shared_decoder(bott, skips)

LATE model forward (one SHARED encoder-decoder, fuse at output):

    phases = split_phases(x)
    per_phase_logits = [self.shared_net(p) for p in phases]   # 3 × (B,1,H,W)
    logits = self.late(per_phase_logits)          # LearnedPhaseWeightFusion(channels=1)

Both keep the F0 (B,3,H,W) input, so dataloader / cache / k-fold / metrics are
unchanged; only the fusion stage differs.
"""
