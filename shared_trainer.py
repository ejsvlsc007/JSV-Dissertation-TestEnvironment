"""
shared/trainer.py
=================
Reusable training loop, validation loop, and CSV logger.
Used identically by every model × fusion combination so that
results are always directly comparable.

Exports
-------
train_epoch(model, loader, criterion, optimizer, device) → (loss, dice)
validate(model, loader, criterion, device, compute_hd, hd_percentile) → (loss, dice, iou, jaccard, acc, hd)
CSVLogger   — init / log_epoch / log_summary
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
    jaccard_score,
    pixel_accuracy,
    hausdorff_distance_batch,
)


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    Run one full training epoch.

    Handles both single-output models and deep-supervision models that
    return a list/tuple of outputs (first element is the main output).

    Returns:
        mean_loss, mean_dice  over all batches.
    """
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
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    compute_hd: bool = False,
    hd_percentile: int = 95,
) -> tuple[float, float, float, float, float, float]:
    """
    Run inference on a DataLoader and return all evaluation metrics.

    Args:
        model:         trained (or in-training) model.
        loader:        DataLoader for val or test split.
        criterion:     loss function (same as training).
        device:        cuda / cpu.
        compute_hd:    whether to compute Hausdorff (expensive — skip most epochs).
        hd_percentile: 95 for HD95, 100 for full Hausdorff.

    Returns:
        loss, dice, iou, jaccard, accuracy, hausdorff
        hausdorff is float('nan') when compute_hd=False.
    """
    model.eval()
    loss_sum = dice_sum = iou_sum = jac_sum = acc_sum = 0.0
    hd_vals: list[float] = []

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
            jac_sum  += jaccard_score(pred, masks)
            acc_sum  += pixel_accuracy(pred, masks)

            if compute_hd:
                hd_vals.append(
                    hausdorff_distance_batch(
                        pred.cpu().numpy().squeeze(1),
                        masks.cpu().numpy().squeeze(1),
                        percentile=hd_percentile,
                    )
                )

    n  = len(loader)
    hd = float(np.mean(hd_vals)) if hd_vals else float("nan")
    return loss_sum / n, dice_sum / n, iou_sum / n, jac_sum / n, acc_sum / n, hd


# ---------------------------------------------------------------------------
# Timed wrappers  (return wall-clock seconds alongside the metrics)
# ---------------------------------------------------------------------------

def train_epoch_timed(model, loader, criterion, optimizer, device):
    """train_epoch + wall-clock timing.  Returns (loss, dice, elapsed_s)."""
    t0 = time.perf_counter()
    loss, dice = train_epoch(model, loader, criterion, optimizer, device)
    return loss, dice, time.perf_counter() - t0


def validate_timed(model, loader, criterion, device, dataset_len,
                   compute_hd=False, hd_percentile=95):
    """
    validate + wall-clock timing.

    Returns:
        loss, dice, iou, jaccard, acc, hd,
        total_inference_s, ms_per_image
    """
    t0 = time.perf_counter()
    loss, dice, iou, jac, acc, hd = validate(
        model, loader, criterion, device, compute_hd, hd_percentile
    )
    elapsed = time.perf_counter() - t0
    ms_per  = (elapsed / dataset_len) * 1000 if dataset_len > 0 else float("nan")
    return loss, dice, iou, jac, acc, hd, elapsed, ms_per


# ---------------------------------------------------------------------------
# CSV Logger
# ---------------------------------------------------------------------------

_EPOCH_FIELDS = [
    "type", "timestamp", "epoch",
    "train_loss", "train_dice",
    "val_loss", "val_dice", "val_iou", "val_jaccard", "val_accuracy", "val_hausdorff",
    "val_inference_time_s", "inference_ms_per_image",
    "train_epoch_time_s", "total_training_time_s",
    "lr",
    # run metadata (repeated every row for easy filtering in pandas)
    "model_id", "fusion_id", "image_size", "dataset_fraction",
    "batch_size", "total_epochs",
]


class CSVLogger:
    """
    Persistent per-run CSV logger.

    Usage
    -----
        logger = CSVLogger(log_dir, run_id)
        logger.init()                   # create file + header (call once per run)
        logger.log_epoch(epoch, ...)    # call after every validation step
        logger.log_summary(...)         # call once after training + test eval
    """

    def __init__(self, log_dir: str, run_id: str):
        os.makedirs(log_dir, exist_ok=True)
        self.path    = os.path.join(log_dir, f"{run_id}.csv")
        self.run_id  = run_id
        self._fields = _EPOCH_FIELDS

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _fmt(v) -> str:
        """Format a float; return '-' for NaN."""
        if isinstance(v, float) and v != v:   # NaN check
            return "-"
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v)

    @staticmethod
    def _fmt4(v) -> str:
        if isinstance(v, float) and v != v:
            return "-"
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    # ── Public API ────────────────────────────────────────────────────────────

    def init(self):
        """Create (or overwrite) the CSV file and write the header row."""
        with open(self.path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self._fields).writeheader()

    def log_epoch(
        self, *,
        epoch: int,
        tr_loss: float, tr_dice: float,
        va_loss: float, va_dice: float, va_iou: float,
        va_jaccard: float, va_accuracy: float, va_hausdorff: float,
        val_inference_time_s: float, inference_ms_per_image: float,
        train_epoch_time_s: float, total_training_time_s: float,
        current_lr: float,
        # run metadata
        model_id: str, fusion_id: str,
        image_size: int, dataset_fraction: float,
        batch_size: int, total_epochs: int,
    ):
        """Append one epoch row."""
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._fields).writerow({
                "type":                   "epoch",
                "timestamp":              datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "epoch":                  epoch,
                "train_loss":             self._fmt(tr_loss),
                "train_dice":             self._fmt(tr_dice),
                "val_loss":               self._fmt(va_loss),
                "val_dice":               self._fmt(va_dice),
                "val_iou":                self._fmt(va_iou),
                "val_jaccard":            self._fmt(va_jaccard),
                "val_accuracy":           self._fmt(va_accuracy),
                "val_hausdorff":          self._fmt4(va_hausdorff),
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
            })

    def log_summary(
        self, *,
        # best-val row
        best_val_dice: float, best_val_iou: float,
        best_val_jaccard: float, best_val_accuracy: float,
        best_val_hausdorff: float,
        # mean-val row
        mean_val_dice: float, mean_val_iou: float,
        mean_val_jaccard: float, mean_val_accuracy: float,
        mean_val_hausdorff: float,
        # test row
        test_dice: float, test_iou: float,
        test_jaccard: float, test_accuracy: float,
        test_hausdorff: float,
        test_inference_time_s: float = float("nan"),
        test_inference_ms_per_image: float = float("nan"),
        total_training_time_s: float = float("nan"),
        # run metadata
        model_id: str = "", fusion_id: str = "",
        image_size: int = 0, dataset_fraction: float = 0.0,
        batch_size: int = 0, total_epochs: int = 0,
    ):
        """Append best_val, mean_val, and test_final summary rows."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        meta = dict(
            model_id=model_id, fusion_id=fusion_id,
            image_size=image_size, dataset_fraction=dataset_fraction,
            batch_size=batch_size, total_epochs=total_epochs,
        )
        rows = [
            ("best_val",   best_val_dice,  best_val_iou,  best_val_jaccard,  best_val_accuracy,  best_val_hausdorff,  float("nan"), float("nan"), float("nan"), float("nan")),
            ("mean_val",   mean_val_dice,  mean_val_iou,  mean_val_jaccard,  mean_val_accuracy,  mean_val_hausdorff,  float("nan"), float("nan"), float("nan"), float("nan")),
            ("test_final", test_dice,      test_iou,      test_jaccard,      test_accuracy,      test_hausdorff,      test_inference_time_s, test_inference_ms_per_image, float("nan"), total_training_time_s),
        ]
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._fields)
            for label, dice, iou, jac, acc, hd, inf_s, inf_ms, ep_s, tot_s in rows:
                w.writerow({
                    "type":                   "summary",
                    "timestamp":              ts,
                    "epoch":                  label,
                    "train_loss":             "-",
                    "train_dice":             self._fmt(dice),
                    "val_loss":               "-",
                    "val_dice":               self._fmt(dice),
                    "val_iou":                self._fmt(iou),
                    "val_jaccard":            self._fmt(jac),
                    "val_accuracy":           self._fmt(acc),
                    "val_hausdorff":          self._fmt4(hd),
                    "val_inference_time_s":   self._fmt4(inf_s),
                    "inference_ms_per_image": self._fmt4(inf_ms),
                    "train_epoch_time_s":     self._fmt4(ep_s),
                    "total_training_time_s":  self._fmt4(tot_s),
                    "lr":                     "-",
                    **meta,
                })
