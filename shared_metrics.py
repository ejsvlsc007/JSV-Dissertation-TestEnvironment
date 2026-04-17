"""
shared/metrics.py
=================
All evaluation metrics used across every model × fusion experiment.

Exports
-------
dice_coefficient(pred, target)          → float
iou_score(pred, target)                 → float   (alias: jaccard_score)
pixel_accuracy(pred, target)            → float
hausdorff_distance_batch(preds_np, masks_np, percentile) → float
"""

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

SMOOTH = 1e-5


# ---------------------------------------------------------------------------
# Tensor metrics  (called inside train / val loops, batch-level)
# ---------------------------------------------------------------------------

def dice_coefficient(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Soft Dice on a batch.  Threshold: 0.5 on sigmoid output.

    Args:
        pred:   FloatTensor (B, 1, H, W)  — sigmoid probabilities.
        target: FloatTensor (B, 1, H, W)  — binary ground truth.

    Returns:
        Scalar float.
    """
    pred   = (pred > 0.5).float().view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return ((2 * intersection + SMOOTH) / (pred.sum() + target.sum() + SMOOTH)).item()


def iou_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Intersection-over-Union (Jaccard) on a batch.

    Args:
        pred:   FloatTensor (B, 1, H, W)  — sigmoid probabilities.
        target: FloatTensor (B, 1, H, W)  — binary ground truth.

    Returns:
        Scalar float.
    """
    pred   = (pred > 0.5).float().view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return ((intersection + SMOOTH) / (union + SMOOTH)).item()


# Alias — IoU and Jaccard are identical for binary segmentation
jaccard_score = iou_score


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Pixel-level accuracy on a batch.

    Args:
        pred:   FloatTensor (B, 1, H, W)  — sigmoid probabilities.
        target: FloatTensor (B, 1, H, W)  — binary ground truth.

    Returns:
        Scalar float.
    """
    pred    = (pred > 0.5).float()
    correct = (pred == target).float().sum()
    return (correct / torch.numel(target)).item()


# ---------------------------------------------------------------------------
# Hausdorff Distance  (numpy, called post-batch on CPU arrays)
# ---------------------------------------------------------------------------

def _surface_distances(pred_bin: np.ndarray, true_bin: np.ndarray) -> np.ndarray:
    """
    Approximate symmetric surface distances between two binary masks.

    Both inputs are 2-D boolean arrays (H, W).
    Returns a 1-D array of distances; [inf] when one mask is empty and the
    other is not; [0.0] when both are empty.
    """
    if pred_bin.sum() == 0 and true_bin.sum() == 0:
        return np.array([0.0])
    if pred_bin.sum() == 0 or true_bin.sum() == 0:
        return np.array([np.inf])

    dist_pred = distance_transform_edt(~pred_bin)
    dist_true = distance_transform_edt(~true_bin)

    # Approximate surface by checking border pixels
    surf_pred = pred_bin & ~np.pad(pred_bin, 1, mode="edge")[1:-1, 1:-1]
    surf_true = true_bin & ~np.pad(true_bin, 1, mode="edge")[1:-1, 1:-1]

    d1 = dist_true[surf_pred > 0]
    d2 = dist_pred[surf_true > 0]

    if d1.size and d2.size:
        return np.concatenate([d1, d2])
    return np.array([np.inf])


def hausdorff_distance_batch(
    preds_np: np.ndarray,
    masks_np: np.ndarray,
    percentile: int = 95,
) -> float:
    """
    Mean Hausdorff distance over a batch of 2-D slices.

    Args:
        preds_np:   float32 ndarray (N, H, W) — raw sigmoid probabilities.
        masks_np:   float32 ndarray (N, H, W) — binary ground truth.
        percentile: 95 for HD95, 100 for full Hausdorff.

    Returns:
        Mean HD across the batch as a scalar float.
    """
    hd_vals = [
        float(
            np.percentile(
                _surface_distances(
                    (p > 0.5).astype(bool),
                    m.astype(bool),
                ),
                percentile,
            )
        )
        for p, m in zip(preds_np, masks_np)
    ]
    return float(np.mean(hd_vals))
