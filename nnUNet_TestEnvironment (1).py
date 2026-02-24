# =============================================================================
# JSV nn-UNet TestTrainer
# Upload this file to your GitHub repo.
# Section 4 of the notebook will clone the repo, stamp in the {PLACEHOLDERS},
# and install the result as TestTrainer.py into the nnUNet package directory.
#
# Placeholders stamped at runtime (do NOT edit these manually):
#   {EPOCHS}            ← EPOCHS from Section 2
#   {LR}                ← LEARNING_RATE_NNUNET from Section 2
#   {BATCH_SIZE_NNUNET} ← BATCH_SIZE_NNUNET from Section 2
#   {LOG_DIR}           ← LOG_DIR from Section 2
#   {IMAGE_SIZE}        ← IMAGE_SIZE from Section 2
#   {DATASET_FRACTION}  ← DATASET_FRACTION from Section 2
#   {DATASET_SOURCE}    ← DATASET_SOURCE from Section 0
#   {HD_PERCENTILE}     ← HD_PERCENTILE from Section 2
#   {TRAIN_KEYS}        ← list of training case keys from Section 3
#   {VAL_KEYS}          ← list of validation case keys from Section 3
# =============================================================================

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
import csv
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy.ndimage import distance_transform_edt

SMOOTH = 1e-5


# =============================================================================
# Metric helpers
# =============================================================================

def _dice(p, t):
    i = (p & t).sum()
    d = p.sum() + t.sum()
    return float((2 * i + SMOOTH) / (d + SMOOTH))


def _iou(p, t):
    i = (p & t).sum()
    u = p.sum() + t.sum() - i
    return float((i + SMOOTH) / (u + SMOOTH))


def _accuracy(p, t):
    return float((p == t).sum() / p.size)


def _surface_distances(p, t):
    """Bidirectional surface distances between binary masks p and t."""
    if not p.any() and not t.any():
        return np.array([0.0])
    if not p.any() or not t.any():
        return np.array([np.inf])
    dt_t = distance_transform_edt(~t)
    dt_p = distance_transform_edt(~p)
    # Surface = pixels that are foreground but whose eroded version is not
    sp = p & ~np.pad(p, 1, mode="edge")[1:-1, 1:-1]
    st = t & ~np.pad(t, 1, mode="edge")[1:-1, 1:-1]
    d1 = dt_t[sp]
    d2 = dt_p[st]
    if d1.size and d2.size:
        return np.concatenate([d1, d2])
    return np.array([np.inf])


def _hd_batch(preds, masks, pct):
    """Mean HD at percentile pct over a batch of 2D binary arrays."""
    vals = []
    for p, m in zip(preds, masks):
        dists = _surface_distances((p > 0.5), m.astype(bool))
        finite = dists[np.isfinite(dists)]
        vals.append(
            float(np.percentile(finite, pct)) if finite.size > 0 else float("nan")
        )
    finite_vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(finite_vals)) if finite_vals else float("nan")


# =============================================================================
# TestTrainer
# =============================================================================

class TestTrainer(nnUNetTrainer):
    # ── Stamped in from Section 2 at install time ─────────────────────────────
    _EPOCHS              = {EPOCHS}
    _LR                  = {LR}            # None → nnUNet default poly LR
    _BATCH_SIZE_OVERRIDE = {BATCH_SIZE_NNUNET}  # None → nnUNet auto
    _LOG_DIR             = r"{LOG_DIR}"
    _IMAGE_SIZE          = {IMAGE_SIZE}
    _DATASET_FRACTION    = {DATASET_FRACTION}
    _DATASET_SOURCE      = "{DATASET_SOURCE}"
    _HD_PERCENTILE       = {HD_PERCENTILE}
    # ── Stamped in from Section 3 at install time ─────────────────────────────
    _TRAIN_KEYS          = {TRAIN_KEYS}
    _VAL_KEYS            = {VAL_KEYS}

    # -------------------------------------------------------------------------
    def initialize(self):
        # Set epochs and LR BEFORE super().initialize() so nnUNet reads them
        self.num_epochs = self._EPOCHS
        if self._LR is not None:
            self.initial_lr = self._LR

        super().initialize()

        # Overwrite nnUNet's auto-generated 5-fold split with our exact split
        preprocessed_dir = Path(
            os.environ.get("nnUNet_preprocessed", "/content/nnUNet_preprocessed")
        )
        splits_file = (
            preprocessed_dir / self.plans_manager.dataset_name / "splits_final.json"
        )
        with open(splits_file, "w") as f:
            json.dump([{"train": self._TRAIN_KEYS, "val": self._VAL_KEYS}], f)
        print(
            f"[TestTrainer] Split written: "
            f"{len(self._TRAIN_KEYS)} train, {len(self._VAL_KEYS)} val"
        )

        # CSV log file
        os.makedirs(self._LOG_DIR, exist_ok=True)
        self.csv_log_file = os.path.join(
            self._LOG_DIR,
            f"nnUNet_E{self.num_epochs}_IMG{self._IMAGE_SIZE}"
            f"_F{self._DATASET_FRACTION}_{self._DATASET_SOURCE}.csv",
        )
        self._csv_cols = [
            "type", "timestamp", "epoch",
            "train_loss", "train_dice",
            "val_loss", "val_dice", "val_iou", "val_jaccard",
            "val_accuracy", "val_hausdorff",
            "lr", "image_size", "dataset_fraction",
            "dataset_source", "batch_size", "total_epochs",
        ]
        if not os.path.exists(self.csv_log_file):
            with open(self.csv_log_file, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self._csv_cols).writeheader()

        self._v_dice = []
        self._v_iou  = []
        self._v_jac  = []
        self._v_acc  = []
        self._v_hd   = []

    # -------------------------------------------------------------------------
    def configure_optimizers(self):
        lr  = self.initial_lr
        opt = torch.optim.SGD(
            self.network.parameters(),
            lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        if self._LR is not None:
            # Constant LR — better for short runs (20–50 epochs)
            sched = torch.optim.lr_scheduler.ConstantLR(
                opt, factor=1.0, total_iters=self.num_epochs
            )
        else:
            # nnUNet default poly LR — better for long runs (250+ epochs)
            sched = torch.optim.lr_scheduler.PolynomialLR(
                opt, total_iters=self.num_epochs, power=0.9
            )
        return opt, sched

    # -------------------------------------------------------------------------
    def _extended_val(self, compute_hd=False):
        """
        Run validation and compute Dice / IoU / Accuracy every epoch.
        Hausdorff distance is only computed on the final epoch (expensive).
        """
        self.network.eval()
        d_v = []; i_v = []; a_v = []; h_v = []

        with torch.no_grad():
            for batch in self.dataloader_val:
                data   = batch["data"].to(self.device)
                target = batch["target"]
                target = (
                    [t.to(self.device) for t in target]
                    if isinstance(target, list)
                    else target.to(self.device)
                )
                out = self.network(data)
                if isinstance(out, (list, tuple)):
                    out = out[0]

                pn = torch.sigmoid(out).cpu().numpy()
                p  = pn[:, 0] > 0.5
                t  = (
                    target[0].cpu().numpy()[:, 0] > 0.5
                    if isinstance(target, list)
                    else target.cpu().numpy()[:, 0] > 0.5
                )

                for pi, ti in zip(p, t):
                    d_v.append(_dice(pi, ti))
                    i_v.append(_iou(pi, ti))
                    a_v.append(_accuracy(pi, ti))

                if compute_hd:
                    h_v.append(_hd_batch(p, t, self._HD_PERCENTILE))

        hv = float(np.nanmean(h_v)) if h_v else float("nan")
        return (
            float(np.mean(d_v)),
            float(np.mean(i_v)),
            float(np.mean(i_v)),   # jaccard == iou
            float(np.mean(a_v)),
            hv,
        )

    # -------------------------------------------------------------------------
    def on_epoch_end(self):
        super().on_epoch_end()

        log   = self.logger.my_fantastic_logging
        epoch = self.current_epoch
        tl    = log["train_losses"][-1]
        vl    = log["val_losses"][-1]
        dl    = log["dice_per_class_or_region"][-1]
        td    = float(dl[0]) if dl else 0.0
        lr    = self.optimizer.param_groups[0]["lr"]
        ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # HD only on the final epoch to avoid multi-minute stalls mid-training
        is_last = epoch == self.num_epochs - 1
        vd, vi, vj, va, vh = self._extended_val(compute_hd=is_last)

        self._v_dice.append(vd)
        self._v_iou.append(vi)
        self._v_jac.append(vj)
        self._v_acc.append(va)
        self._v_hd.append(vh)

        actual_bs = self.configuration_manager.configuration.get("batch_size", 0)

        with open(self.csv_log_file, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._csv_cols).writerow({
                "type":             "epoch",
                "timestamp":        ts,
                "epoch":            epoch,
                "train_loss":       f"{tl:.6f}",
                "train_dice":       f"{td:.6f}",
                "val_loss":         f"{vl:.6f}",
                "val_dice":         f"{vd:.6f}",
                "val_iou":          f"{vi:.6f}",
                "val_jaccard":      f"{vj:.6f}",
                "val_accuracy":     f"{va:.6f}",
                "val_hausdorff":    f"{vh:.4f}",
                "lr":               f"{lr:.2e}",
                "image_size":       self._IMAGE_SIZE,
                "dataset_fraction": self._DATASET_FRACTION,
                "dataset_source":   self._DATASET_SOURCE,
                "batch_size":       actual_bs,
                "total_epochs":     self.num_epochs,
            })

    # -------------------------------------------------------------------------
    def on_train_end(self):
        super().on_train_end()

        d  = self._v_dice
        i  = self._v_iou
        j  = self._v_jac
        a  = self._v_acc
        hh = [x for x in self._v_hd if not np.isnan(x)]
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        actual_bs = self.configuration_manager.configuration.get("batch_size", 0)

        rows = [
            ("best_val",
             max(d) if d else 0, max(i) if i else 0,
             max(j) if j else 0, max(a) if a else 0,
             min(hh) if hh else float("nan")),
            ("mean_val",
             float(np.mean(d)) if d else 0, float(np.mean(i)) if i else 0,
             float(np.mean(j)) if j else 0, float(np.mean(a)) if a else 0,
             float(np.mean(hh)) if hh else float("nan")),
            ("test_final",
             d[-1] if d else 0, i[-1] if i else 0,
             j[-1] if j else 0, a[-1] if a else 0,
             hh[-1] if hh else float("nan")),
        ]

        with open(self.csv_log_file, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._csv_cols)
            for label, vd, vi, vj, va, vh in rows:
                w.writerow({
                    "type":             "summary",
                    "timestamp":        ts,
                    "epoch":            label,
                    "train_loss":       "-",
                    "train_dice":       f"{vd:.6f}",
                    "val_loss":         "-",
                    "val_dice":         f"{vd:.6f}",
                    "val_iou":          f"{vi:.6f}",
                    "val_jaccard":      f"{vj:.6f}",
                    "val_accuracy":     f"{va:.6f}",
                    "val_hausdorff":    f"{vh:.4f}",
                    "lr":               "-",
                    "image_size":       self._IMAGE_SIZE,
                    "dataset_fraction": self._DATASET_FRACTION,
                    "dataset_source":   self._DATASET_SOURCE,
                    "batch_size":       actual_bs,
                    "total_epochs":     self.num_epochs,
                })
