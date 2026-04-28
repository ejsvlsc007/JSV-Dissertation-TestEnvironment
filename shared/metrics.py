"""
shared/metrics.py
=================
All evaluation metrics used across every model x fusion experiment.

Exports
-------
dice_coefficient(pred, target)                           -> float
sensitivity(pred, target)                                -> float  (recall / TPR)
specificity(pred, target)                                -> float  (TNR)
precision(pred, target)                                  -> float  (PPV)
pixel_accuracy(pred, target)                             -> float
hausdorff_distance_batch(preds_np, masks_np, percentile) -> float  (HD95, no inf)
nsd_batch(preds_np, masks_np, tolerance)                 -> float  (NSD)
get_model_size_mb(filepath)                              -> float
"""

import os
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

SMOOTH = 1e-5


# ---------------------------------------------------------------------------
# Core segmentation metrics
# ---------------------------------------------------------------------------

def dice_coefficient(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Dice similarity coefficient. Threshold: 0.5.
    pred/target: FloatTensor (B, 1, H, W)
    """
    pred   = (pred > 0.5).float().view(-1)
    target = target.view(-1)
    inter  = (pred * target).sum()
    return ((2 * inter + SMOOTH) / (pred.sum() + target.sum() + SMOOTH)).item()


    """
    Jaccard index (IoU). Threshold: 0.5.
    Identical to IoU for binary segmentation.
    pred/target: FloatTensor (B, 1, H, W)
    """
    pred   = (pred > 0.5).float().view(-1)
    target = target.view(-1)
    inter  = (pred * target).sum()
    union  = pred.sum() + target.sum() - inter
    return ((inter + SMOOTH) / (union + SMOOTH)).item()


def sensitivity(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Sensitivity (Recall / True Positive Rate).
    Measures how much of the tumour the model finds.
    TPR = TP / (TP + FN)
    """
    pred   = (pred > 0.5).float().view(-1)
    target = target.view(-1)
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    return ((tp + SMOOTH) / (tp + fn + SMOOTH)).item()


def specificity(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Specificity (True Negative Rate).
    Measures how well background pixels are rejected.
    TNR = TN / (TN + FP)
    """
    pred   = (pred > 0.5).float().view(-1)
    target = target.view(-1)
    tn = ((1 - pred) * (1 - target)).sum()
    fp = (pred * (1 - target)).sum()
    return ((tn + SMOOTH) / (tn + fp + SMOOTH)).item()


def precision(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Precision (Positive Predictive Value).
    Measures quality of positive predictions.
    PPV = TP / (TP + FP)
    """
    pred   = (pred > 0.5).float().view(-1)
    target = target.view(-1)
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    return ((tp + SMOOTH) / (tp + fp + SMOOTH)).item()


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Pixel-level accuracy.
    Note: misleading on imbalanced datasets — use sensitivity/specificity
    for a more meaningful breakdown per class.
    """
    pred    = (pred > 0.5).float()
    correct = (pred == target).float().sum()
    return (correct / torch.numel(target)).item()


# ---------------------------------------------------------------------------
# Hausdorff Distance HD95  (inf-free)
# ---------------------------------------------------------------------------

def _surface_distances(pred_bin: np.ndarray, true_bin: np.ndarray) -> np.ndarray:
    """
    Symmetric surface distances between two binary 2-D masks.

    Empty mask handling (no inf returned):
        both empty  -> [0.0]       perfect agreement
        one empty   -> [diagonal]  worst-case finite penalty
    """
    H, W = pred_bin.shape
    diag = float(np.sqrt(H ** 2 + W ** 2))

    if pred_bin.sum() == 0 and true_bin.sum() == 0:
        return np.array([0.0])
    if pred_bin.sum() == 0 or true_bin.sum() == 0:
        return np.array([diag])   # capped at diagonal, never inf

    dist_pred = distance_transform_edt(~pred_bin)
    dist_true = distance_transform_edt(~true_bin)

    surf_pred = pred_bin & ~np.pad(pred_bin, 1, mode="edge")[1:-1, 1:-1]
    surf_true = true_bin & ~np.pad(true_bin, 1, mode="edge")[1:-1, 1:-1]

    d1 = dist_true[surf_pred > 0]
    d2 = dist_pred[surf_true > 0]

    if d1.size and d2.size:
        return np.concatenate([d1, d2])
    return np.array([diag])


def hausdorff_distance_batch(
    preds_np: np.ndarray,
    masks_np: np.ndarray,
    percentile: int = 95,
) -> float:
    """
    Mean HD95 over a batch of 2-D slices. Never returns inf or nan.

    Args:
        preds_np:   float32 ndarray (N, H, W) — sigmoid probabilities.
        masks_np:   float32 ndarray (N, H, W) — binary ground truth.
        percentile: 95 for HD95, 100 for full Hausdorff.
    """
    hd_vals = [
        float(np.percentile(
            _surface_distances(
                (p > 0.5).astype(bool),
                m.astype(bool),
            ),
            percentile,
        ))
        for p, m in zip(preds_np, masks_np)
    ]
    return float(np.mean(hd_vals))


# ---------------------------------------------------------------------------
# Normalised Surface Distance (NSD)
# ---------------------------------------------------------------------------

def _nsd_single(
    pred_bin: np.ndarray,
    true_bin: np.ndarray,
    tolerance: float = 2.0,
) -> float:
    """
    NSD for a single 2-D slice.

    NSD = (surface points of pred within tau of true +
           surface points of true within tau of pred)
          / (total surface points of pred + true)

    Returns value in [0, 1]. Higher is better.
    """
    if pred_bin.sum() == 0 and true_bin.sum() == 0:
        return 1.0
    if pred_bin.sum() == 0 or true_bin.sum() == 0:
        return 0.0

    dist_pred = distance_transform_edt(~pred_bin)
    dist_true = distance_transform_edt(~true_bin)

    surf_pred = pred_bin & ~np.pad(pred_bin, 1, mode="edge")[1:-1, 1:-1]
    surf_true = true_bin & ~np.pad(true_bin, 1, mode="edge")[1:-1, 1:-1]

    n_pred = surf_pred.sum()
    n_true = surf_true.sum()

    if n_pred == 0 or n_true == 0:
        return 0.0

    pred_within = (dist_true[surf_pred] <= tolerance).sum()
    true_within = (dist_pred[surf_true] <= tolerance).sum()

    return float((pred_within + true_within) / (n_pred + n_true))


def nsd_batch(
    preds_np: np.ndarray,
    masks_np: np.ndarray,
    tolerance: float = 2.0,
) -> float:
    """
    Mean NSD over a batch of 2-D slices.

    Args:
        preds_np:  float32 ndarray (N, H, W) — sigmoid probabilities.
        masks_np:  float32 ndarray (N, H, W) — binary ground truth.
        tolerance: surface tolerance in pixels (default 2px ≈ 2mm at 1mm/px).

    Returns:
        Mean NSD in [0, 1]. Higher is better.
    """
    return float(np.mean([
        _nsd_single((p > 0.5).astype(bool), m.astype(bool), tolerance)
        for p, m in zip(preds_np, masks_np)
    ]))


# ---------------------------------------------------------------------------
# Model size
# ---------------------------------------------------------------------------

def get_model_size_mb(filepath: str) -> float:
    """
    Return the size of a saved .pth checkpoint in megabytes.
    Returns float('nan') if the file does not exist yet.
    """
    if not os.path.exists(filepath):
        return float("nan")
    return os.path.getsize(filepath) / (1024 ** 2)
