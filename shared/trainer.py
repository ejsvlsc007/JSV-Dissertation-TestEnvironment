"""
shared/trainer.py
=================
Reusable training loop, validation loop, and CSV logger.
Used identically by every model x fusion combination.

validate / validate_timed return order:
    loss, dice, iou, sensitivity, specificity, precision,
    accuracy, hausdorff, nsd
    [+ elapsed, ms_per_image for timed variant]
"""

import csv
import os
import time
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from shared.metrics import (
    dice_coefficient,
    iou_score,
    sensitivity,
    specificity,
    precision,
    pixel_accuracy,
    hausdorff_distance_batch,
    nsd_batch,
    get_model_size_mb,
)


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device):
    """Run one full training epoch. Returns (mean_loss, mean_dice)."""
    model.train()
    loss_sum, dice_sum = 0.0, 0.0

    for imgs, masks in tqdm(loader, desc="Training", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, masks)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        logits    = out[0] if isinstance(out, (list, tuple)) else out
        pred      = torch.sigmoid(logits)
        dice_sum += dice_coefficient(pred, masks)

    n = len(loader)
    return loss_sum / n, dice_sum / n


# ---------------------------------------------------------------------------
# Validation / evaluation
# ---------------------------------------------------------------------------

def validate(
    model, loader, criterion, device,
    compute_hd:    bool  = False,
    compute_nsd:   bool  = False,
    hd_percentile: int   = 95,
    nsd_tolerance: float = 2.0,
):
    """
    Run inference and return all evaluation metrics.

    Returns:
        loss, dice, iou, sensitivity, specificity, precision,
        accuracy, hausdorff, nsd

        hausdorff / nsd are float('nan') when not computed.
    """
    model.eval()
    loss_sum = dice_sum = iou_sum = 0.0
    sen_sum  = spe_sum = pre_sum = acc_sum = 0.0
    hd_vals:  list[float] = []
    nsd_vals: list[float] = []

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            out  = model(imgs)
            loss = criterion(out, masks)
            loss_sum += loss.item()

            logits = out[0] if isinstance(out, (list, tuple)) else out
            pred   = torch.sigmoid(logits)

            dice_sum += dice_coefficient(pred, masks)
            iou_sum  += iou_score(pred, masks)
            sen_sum  += sensitivity(pred, masks)
            spe_sum  += specificity(pred, masks)
            pre_sum  += precision(pred, masks)
            acc_sum  += pixel_accuracy(pred, masks)

            if compute_hd or compute_nsd:
                preds_np = pred.cpu().numpy().squeeze(1)
                masks_np = masks.cpu().numpy().squeeze(1)
                if compute_hd:
                    hd_vals.append(hausdorff_distance_batch(
                        preds_np, masks_np, percentile=hd_percentile))
                if compute_nsd:
                    nsd_vals.append(nsd_batch(
                        preds_np, masks_np, tolerance=nsd_tolerance))

    n   = len(loader)
    hd  = float(np.mean(hd_vals))  if hd_vals  else float("nan")
    nsd = float(np.mean(nsd_vals)) if nsd_vals else float("nan")

    return (
        loss_sum / n,   # loss
        dice_sum / n,   # dice
        iou_sum  / n,   # iou
        sen_sum  / n,   # sensitivity
        spe_sum  / n,   # specificity
        pre_sum  / n,   # precision
        acc_sum  / n,   # accuracy
        hd,             # hausdorff
        nsd,            # nsd
    )


# ---------------------------------------------------------------------------
# Timed wrappers
# ---------------------------------------------------------------------------

def train_epoch_timed(model, loader, criterion, optimizer, device):
    """Returns (loss, dice, elapsed_s)."""
    t0 = time.perf_counter()
    loss, dice = train_epoch(model, loader, criterion, optimizer, device)
    return loss, dice, time.perf_counter() - t0


def validate_timed(
    model, loader, criterion, device, dataset_len,
    compute_hd=False, compute_nsd=False,
    hd_percentile=95, nsd_tolerance=2.0,
):
    """
    validate + wall-clock timing.

    Returns:
        loss, dice, iou, sensitivity, specificity, precision,
        accuracy, hausdorff, nsd,
        total_inference_s, ms_per_image
    """
    t0      = time.perf_counter()
    results = validate(
        model, loader, criterion, device,
        compute_hd=compute_hd, compute_nsd=compute_nsd,
        hd_percentile=hd_percentile, nsd_tolerance=nsd_tolerance,
    )
    elapsed = time.perf_counter() - t0
    ms_per  = (elapsed / dataset_len) * 1000 if dataset_len > 0 else float("nan")
    return (*results, elapsed, ms_per)


# ---------------------------------------------------------------------------
# CSV Logger
# ---------------------------------------------------------------------------

_EPOCH_FIELDS = [
    "type", "timestamp", "epoch",
    "train_loss", "train_dice",
    "val_loss", "val_dice", "val_iou",
    "val_sensitivity", "val_specificity", "val_precision",
    "val_accuracy", "val_hausdorff", "val_nsd",
    "val_inference_time_s", "inference_ms_per_image",
    "train_epoch_time_s", "total_training_time_s",
    "lr",
    "model_id", "fusion_id", "image_size", "dataset_fraction",
    "batch_size", "total_epochs",
    "model_size_mb",
]


class CSVLogger:
    """Persistent per-run CSV logger."""

    def __init__(self, log_dir: str, run_id: str):
        os.makedirs(log_dir, exist_ok=True)
        self.path   = os.path.join(log_dir, f"{run_id}.csv")
        self.run_id = run_id

    @staticmethod
    def _fmt(v) -> str:
        if isinstance(v, float) and v != v: return "-"
        if isinstance(v, float): return f"{v:.6f}"
        return str(v)

    @staticmethod
    def _fmt4(v) -> str:
        if isinstance(v, float) and v != v: return "-"
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    def init(self):
        with open(self.path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=_EPOCH_FIELDS).writeheader()

    def log_epoch(
        self, *,
        epoch: int,
        tr_loss: float, tr_dice: float,
        va_loss: float, va_dice: float, va_iou: float,
        va_sensitivity: float, va_specificity: float, va_precision: float,
        va_accuracy: float, va_hausdorff: float, va_nsd: float,
        val_inference_time_s: float, inference_ms_per_image: float,
        train_epoch_time_s: float, total_training_time_s: float,
        current_lr: float,
        model_id: str, fusion_id: str,
        image_size: int, dataset_fraction: float,
        batch_size: int, total_epochs: int,
        model_size_mb: float = float("nan"),
    ):
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=_EPOCH_FIELDS).writerow({
                "type":                   "epoch",
                "timestamp":              datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "epoch":                  epoch,
                "train_loss":             self._fmt(tr_loss),
                "train_dice":             self._fmt(tr_dice),
                "val_loss":               self._fmt(va_loss),
                "val_dice":               self._fmt(va_dice),
                "val_iou":                self._fmt(va_iou),
                "val_sensitivity":        self._fmt(va_sensitivity),
                "val_specificity":        self._fmt(va_specificity),
                "val_precision":          self._fmt(va_precision),
                "val_accuracy":           self._fmt(va_accuracy),
                "val_hausdorff":          self._fmt4(va_hausdorff),
                "val_nsd":                self._fmt4(va_nsd),
                "val_inference_time_s":   self._fmt4(val_inference_time_s),
                "inference_ms_per_image": self._fmt4(inference_ms_per_image),
                "train_epoch_time_s":     self._fmt4(train_epoch_time_s),
                "total_training_time_s":  self._fmt4(total_training_time_s),
                "lr":                     f"{current_lr:.2e}",
                "model_id":               model_id,
                "fusion_id":              fusion_id,
                "image_size":             image_size,
                "dataset_fraction":       dataset_fraction,
                "batch_size":             batch_size,
                "total_epochs":           total_epochs,
                "model_size_mb":          self._fmt4(model_size_mb),
            })

    def log_summary(
        self, *,
        best_val_dice: float, best_val_iou: float,
        best_val_sensitivity: float, best_val_specificity: float,
        best_val_precision: float,
        best_val_accuracy: float, best_val_hausdorff: float,
        best_val_nsd: float,
        mean_val_dice: float, mean_val_iou: float,
        mean_val_sensitivity: float, mean_val_specificity: float,
        mean_val_precision: float,
        mean_val_accuracy: float, mean_val_hausdorff: float,
        mean_val_nsd: float,
        test_dice: float, test_iou: float,
        test_sensitivity: float, test_specificity: float,
        test_precision: float,
        test_accuracy: float, test_hausdorff: float,
        test_nsd: float,
        test_inference_time_s: float = float("nan"),
        test_inference_ms_per_image: float = float("nan"),
        total_training_time_s: float = float("nan"),
        model_size_mb: float = float("nan"),
        model_id: str = "", fusion_id: str = "",
        image_size: int = 0, dataset_fraction: float = 0.0,
        batch_size: int = 0, total_epochs: int = 0,
    ):
        ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        meta = dict(model_id=model_id, fusion_id=fusion_id,
                    image_size=image_size, dataset_fraction=dataset_fraction,
                    batch_size=batch_size, total_epochs=total_epochs)
        nan  = float("nan")

        def _row(label, dice, iou, sen, spe, pre, acc, hd, nsd,
                 inf_s, inf_ms, tot_s, sz):
            return {
                "type": "summary", "timestamp": ts, "epoch": label,
                "train_loss": "-", "train_dice": self._fmt(dice),
                "val_loss": "-",
                "val_dice":        self._fmt(dice),
                "val_iou":         self._fmt(iou),
                "val_sensitivity": self._fmt(sen),
                "val_specificity": self._fmt(spe),
                "val_precision":   self._fmt(pre),
                "val_accuracy":    self._fmt(acc),
                "val_hausdorff":   self._fmt4(hd),
                "val_nsd":         self._fmt4(nsd),
                "val_inference_time_s":   self._fmt4(inf_s),
                "inference_ms_per_image": self._fmt4(inf_ms),
                "train_epoch_time_s":     "-",
                "total_training_time_s":  self._fmt4(tot_s),
                "lr": "-",
                "model_size_mb": self._fmt4(sz),
                **meta,
            }

        rows = [
            _row("best_val",
                 best_val_dice, best_val_iou,
                 best_val_sensitivity, best_val_specificity, best_val_precision,
                 best_val_accuracy, best_val_hausdorff, best_val_nsd,
                 nan, nan, nan, model_size_mb),
            _row("mean_val",
                 mean_val_dice, mean_val_iou,
                 mean_val_sensitivity, mean_val_specificity, mean_val_precision,
                 mean_val_accuracy, mean_val_hausdorff, mean_val_nsd,
                 nan, nan, nan, model_size_mb),
            _row("test_final",
                 test_dice, test_iou,
                 test_sensitivity, test_specificity, test_precision,
                 test_accuracy, test_hausdorff, test_nsd,
                 test_inference_time_s, test_inference_ms_per_image,
                 total_training_time_s, model_size_mb),
        ]

        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_EPOCH_FIELDS)
            for row in rows:
                w.writerow(row)
