"""
nnUNet1_TestEnvironment.py
==========================
MIC-DKFZ nnU-Net (nn1) — trained via CLI (nnUNetv2_train).

Approach:
  1. PNG slices → NIfTI volumes (auto inside notebook)
  2. nnU-Net dataset fingerprinting + auto-configuration
  3. Training via subprocess nnUNetv2_train
  4. Inference on test set via nnUNetv2_predict
  5. Load predicted NIfTIs → compute your metrics → log to CSV

Key difference from nn2:
  • Architecture AND training pipeline are 100% MIC-DKFZ (no custom loop)
  • Per-epoch metrics are parsed from nnU-Net's progress.png / training_log
  • Early stopping uses nnU-Net's internal mechanism + subprocess monitor
  • Least controllable but most authentic MIC-DKFZ nnU-Net
"""

import os
import re
import csv
import json
import glob
import time
import shutil
import subprocess
import threading
import numpy as np
from datetime import datetime
from pathlib import Path

# Optional imports — available after nnunetv2 is installed
try:
    import nibabel as nib
except ImportError:
    nib = None

try:
    from PIL import Image
except ImportError:
    Image = None

from scipy.ndimage import distance_transform_edt


# ── NIfTI conversion utilities ────────────────────────────────────────────────

def png_slices_to_nifti(volume_ids, dataset_dir, out_dir,
                        image_size, file_ext=".png"):
    """
    Convert LiTS PNG slices for a list of volume IDs into NIfTI volumes.

    Each volume becomes:
      images/  liver_XXXX_0000.nii.gz   (greyscale CT)
      labels/  liver_XXXX.nii.gz        (binary liver mask)

    Parameters
    ----------
    volume_ids : list[str]  e.g. ["0","1","2",...]
    dataset_dir : str       folder containing volume-X_Y.png and segmentation files
    out_dir : str           nnU-Net raw dataset root
    image_size : int        resize target (slices are resized before stacking)
    """
    assert nib  is not None, "nibabel is required: pip install nibabel"
    assert Image is not None, "Pillow is required: pip install Pillow"

    import torchvision.transforms as T
    img_dir = os.path.join(out_dir, "imagesTr")
    lbl_dir = os.path.join(out_dir, "labelsTr")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    resize = T.Resize((image_size, image_size))

    for vol_id in sorted(volume_ids, key=lambda x: int(x)):
        slices_img, slices_lbl = [], []
        pattern = os.path.join(dataset_dir, f"volume-{vol_id}_*.png")
        slice_files = sorted(glob.glob(pattern),
                             key=lambda p: int(p.split("_")[-1].replace(".png","")))
        if not slice_files:
            continue
        for sl_path in slice_files:
            sl_idx = sl_path.split("_")[-1].replace(".png","")
            mask_path = os.path.join(dataset_dir,
                         f"segmentation-{vol_id}_livermask_{sl_idx}.png")
            img = Image.open(sl_path).convert("L")
            img = img.resize((image_size, image_size), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            slices_img.append(arr)

            if os.path.exists(mask_path):
                msk = Image.open(mask_path).convert("L")
                msk = msk.resize((image_size, image_size), Image.NEAREST)
                marr = (np.array(msk) > 0).astype(np.uint8)
            else:
                marr = np.zeros((image_size, image_size), dtype=np.uint8)
            slices_lbl.append(marr)

        vol_np  = np.stack(slices_img, axis=2)   # H x W x D
        lbl_np  = np.stack(slices_lbl, axis=2)

        case_id = f"liver_{int(vol_id):04d}"
        nib.save(nib.Nifti1Image(vol_np, np.eye(4)),
                 os.path.join(img_dir, f"{case_id}_0000.nii.gz"))
        nib.save(nib.Nifti1Image(lbl_np.astype(np.int16), np.eye(4)),
                 os.path.join(lbl_dir, f"{case_id}.nii.gz"))

    print(f"✓ Converted {len(volume_ids)} volumes → {out_dir}")


def write_dataset_json(out_dir, train_ids, val_ids, channel_names=None,
                       labels=None, dataset_name="LiTS_NN1"):
    """Write nnU-Net v2 dataset.json."""
    channel_names = channel_names or {"0": "CT"}
    labels        = labels        or {"background": 0, "liver": 1}

    training = []
    for vid in train_ids + val_ids:
        case_id = f"liver_{int(vid):04d}"
        training.append({
            "image": f"./imagesTr/{case_id}_0000.nii.gz",
            "label": f"./labelsTr/{case_id}.nii.gz",
        })

    ds = {
        "channel_names": channel_names,
        "labels": labels,
        "numTraining": len(training),
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "training": training,
    }
    with open(os.path.join(out_dir, "dataset.json"), "w") as f:
        json.dump(ds, f, indent=2)
    print(f"✓ dataset.json written ({len(training)} cases)")


# ── Metric functions (identical to UNet notebook) ────────────────────────────

SMOOTH = 1e-5


def dice_coefficient_np(pred_bin, true_bin):
    p = pred_bin.flatten().astype(float)
    t = true_bin.flatten().astype(float)
    inter = (p * t).sum()
    return float((2 * inter + SMOOTH) / (p.sum() + t.sum() + SMOOTH))


def iou_score_np(pred_bin, true_bin):
    p = pred_bin.flatten().astype(float)
    t = true_bin.flatten().astype(float)
    inter = (p * t).sum()
    union = p.sum() + t.sum() - inter
    return float((inter + SMOOTH) / (union + SMOOTH))


def pixel_accuracy_np(pred_bin, true_bin):
    return float((pred_bin == true_bin).mean())


def _surface_distances(pred_bin, true_bin):
    if pred_bin.sum() == 0 and true_bin.sum() == 0:
        return np.array([0.0])
    if pred_bin.sum() == 0 or true_bin.sum() == 0:
        return np.array([np.inf])
    dist_pred = distance_transform_edt(~pred_bin)
    dist_true = distance_transform_edt(~true_bin)
    surf_pred = pred_bin & ~np.pad(pred_bin, 1, mode='edge')[1:-1, 1:-1]
    surf_true = true_bin & ~np.pad(true_bin, 1, mode='edge')[1:-1, 1:-1]
    d1 = dist_true[surf_pred > 0]
    d2 = dist_pred[surf_true > 0]
    if d1.size and d2.size:
        return np.concatenate([d1, d2])
    return np.array([np.inf])


def hausdorff_volume(pred_vol, true_vol, percentile=95):
    """Compute HD over all slices of a 3-D volume (H x W x D)."""
    hd_vals = []
    for z in range(pred_vol.shape[2]):
        dists = _surface_distances(
            pred_vol[:, :, z].astype(bool),
            true_vol[:, :, z].astype(bool))
        hd_vals.append(float(np.percentile(dists, percentile)))
    return float(np.mean(hd_vals))


def compute_metrics_from_niftis(pred_dir, label_dir, case_ids,
                                 hd_percentile=95):
    """
    Load predicted and ground-truth NIfTIs for each case and compute metrics.
    Returns dict of mean metrics over all cases.
    """
    assert nib is not None, "nibabel required"
    dice_l, iou_l, acc_l, hd_l = [], [], [], []

    for vid in case_ids:
        case_id  = f"liver_{int(vid):04d}"
        pred_path = os.path.join(pred_dir, f"{case_id}.nii.gz")
        lbl_path  = os.path.join(label_dir, f"{case_id}.nii.gz")
        if not os.path.exists(pred_path) or not os.path.exists(lbl_path):
            continue
        pred = (nib.load(pred_path).get_fdata() > 0.5).astype(bool)
        true = (nib.load(lbl_path).get_fdata()  > 0).astype(bool)
        # Flatten to 2-D slices for per-slice metrics then average
        dice_l.append(dice_coefficient_np(pred, true))
        iou_l.append(iou_score_np(pred, true))
        acc_l.append(pixel_accuracy_np(pred, true))
        hd_l.append(hausdorff_volume(pred, true, hd_percentile))

    return {
        "dice":      float(np.mean(dice_l))     if dice_l else float("nan"),
        "iou":       float(np.mean(iou_l))      if iou_l  else float("nan"),
        "jaccard":   float(np.mean(iou_l))      if iou_l  else float("nan"),
        "accuracy":  float(np.mean(acc_l))      if acc_l  else float("nan"),
        "hausdorff": float(np.nanmean(hd_l))    if hd_l   else float("nan"),
    }


# ── Training log parser ───────────────────────────────────────────────────────

def parse_nnunet_training_log(log_path):
    """
    Parse nnU-Net v2 training_log.txt for per-epoch train/val loss and dice.
    Returns list of dicts with keys: epoch, train_loss, val_loss, val_dice.
    """
    epochs = []
    if not os.path.exists(log_path):
        return epochs

    epoch_re    = re.compile(r"Epoch\s+(\d+)")
    tr_loss_re  = re.compile(r"train loss\s*:\s*([\d.eE+\-]+)")
    val_loss_re = re.compile(r"val loss\s*:\s*([\d.eE+\-]+)")
    val_dice_re = re.compile(r"Pseudo dice\s*\[([^\]]+)\]")

    current = {}
    with open(log_path) as f:
        for line in f:
            m = epoch_re.search(line)
            if m:
                if current:
                    epochs.append(current)
                current = {"epoch": int(m.group(1))}
            m = tr_loss_re.search(line)
            if m:
                current["train_loss"] = float(m.group(1))
            m = val_loss_re.search(line)
            if m:
                current["val_loss"] = float(m.group(1))
            m = val_dice_re.search(line)
            if m:
                vals = [float(v.strip()) for v in m.group(1).split(",")]
                current["val_dice"] = float(np.mean(vals))
    if current:
        epochs.append(current)
    return epochs


# ── CSV logger (same schema as UNet notebook) ─────────────────────────────────

CSV_COLS = [
    'type', 'timestamp', 'epoch',
    'train_loss', 'train_dice',
    'val_loss', 'val_dice', 'val_iou', 'val_jaccard',
    'val_accuracy', 'val_hausdorff', 'lr',
    'image_size', 'dataset_fraction', 'dataset_source',
    'batch_size', 'total_epochs', 'model_id',
]


def init_csv(log_file):
    with open(log_file, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=CSV_COLS).writeheader()


def log_epoch_row(log_file, epoch, tr_loss, tr_dice,
                  va_loss, va_dice, va_iou, va_jaccard,
                  va_accuracy, va_hausdorff, lr,
                  image_size, dataset_fraction, dataset_source,
                  batch_size, total_epochs, model_id):
    with open(log_file, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writerow({
            'type': 'epoch',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'epoch': epoch,
            'train_loss':    f'{tr_loss:.6f}',
            'train_dice':    f'{tr_dice:.6f}',
            'val_loss':      f'{va_loss:.6f}',
            'val_dice':      f'{va_dice:.6f}',
            'val_iou':       f'{va_iou:.6f}',
            'val_jaccard':   f'{va_jaccard:.6f}',
            'val_accuracy':  f'{va_accuracy:.6f}',
            'val_hausdorff': f'{va_hausdorff:.4f}',
            'lr':            f'{lr:.2e}',
            'image_size': image_size, 'dataset_fraction': dataset_fraction,
            'dataset_source': dataset_source, 'batch_size': batch_size,
            'total_epochs': total_epochs, 'model_id': model_id,
        })


def log_summary_rows(log_file,
                     best_val_dice, best_val_iou, best_val_jaccard,
                     best_val_accuracy, best_val_hausdorff,
                     mean_val_dice, mean_val_iou, mean_val_jaccard,
                     mean_val_accuracy, mean_val_hausdorff,
                     test_dice, test_iou, test_jaccard,
                     test_accuracy, test_hausdorff,
                     image_size, dataset_fraction, dataset_source,
                     batch_size, total_epochs, model_id):
    ts   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    rows = [
        ('best_val',   best_val_dice, best_val_iou, best_val_jaccard,
                       best_val_accuracy, best_val_hausdorff),
        ('mean_val',   mean_val_dice, mean_val_iou, mean_val_jaccard,
                       mean_val_accuracy, mean_val_hausdorff),
        ('test_final', test_dice, test_iou, test_jaccard,
                       test_accuracy, test_hausdorff),
    ]
    with open(log_file, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        for label, dice, iou, jaccard, accuracy, hd in rows:
            w.writerow({
                'type': 'summary', 'timestamp': ts, 'epoch': label,
                'train_loss': '-', 'train_dice': f'{dice:.6f}',
                'val_loss':   '-', 'val_dice':   f'{dice:.6f}',
                'val_iou':       f'{iou:.6f}',
                'val_jaccard':   f'{jaccard:.6f}',
                'val_accuracy':  f'{accuracy:.6f}',
                'val_hausdorff': f'{hd:.4f}',
                'lr': '-',
                'image_size': image_size, 'dataset_fraction': dataset_fraction,
                'dataset_source': dataset_source, 'batch_size': batch_size,
                'total_epochs': total_epochs, 'model_id': model_id,
            })


# ── CLI runner ────────────────────────────────────────────────────────────────

class NNUNetCLIRunner:
    """
    Manages the full MIC-DKFZ nnU-Net CLI pipeline inside Colab.

    Steps:
      plan_and_preprocess → train → predict → evaluate
    """

    def __init__(self, nnunet_raw, nnunet_preprocessed, nnunet_results,
                 dataset_id=1, config="2d", fold=0,
                 trainer="nnUNetTrainer", epochs=50,
                 es_patience=10, log_file=None,
                 image_size=128, dataset_fraction=0.25,
                 dataset_source="LiTS", batch_size=8, model_id="NN1"):

        self.raw          = nnunet_raw
        self.preprocessed = nnunet_preprocessed
        self.results      = nnunet_results
        self.dataset_id   = dataset_id
        self.dataset_name = f"Dataset{dataset_id:03d}_LiTS"
        self.config       = config
        self.fold         = fold
        self.trainer      = trainer
        self.epochs       = epochs
        self.es_patience  = es_patience
        self.log_file     = log_file
        self.image_size   = image_size
        self.dataset_fraction = dataset_fraction
        self.dataset_source   = dataset_source
        self.batch_size   = batch_size
        self.model_id     = model_id

        # Set nnU-Net env vars
        os.environ["nnUNet_raw"]          = nnunet_raw
        os.environ["nnUNet_preprocessed"] = nnunet_preprocessed
        os.environ["nnUNet_results"]      = nnunet_results

    def plan_and_preprocess(self):
        print("── Step 1: Plan & Preprocess ──")
        cmd = [
            "nnUNetv2_plan_and_preprocess",
            "-d", str(self.dataset_id),
            "--verify_dataset_integrity",
        ]
        self._run(cmd)

    def train(self):
        print(f"── Step 2: Train (fold={self.fold}, config={self.config}) ──")
        # Override num_epochs via nnU-Net trainer override file
        self._write_trainer_override()

        cmd = [
            "nnUNetv2_train",
            str(self.dataset_id), self.config, str(self.fold),
            "--npz",
        ]
        # Run training in a thread so we can monitor for early stopping
        self._train_with_monitor(cmd)

    def _write_trainer_override(self):
        """Write a plans override to set num_epochs."""
        plans_dir = os.path.join(
            self.preprocessed, self.dataset_name)
        os.makedirs(plans_dir, exist_ok=True)
        override = {"num_epochs": self.epochs}
        # nnU-Net v2 respects nnUNetPlans.json overrides for num_epochs
        override_path = os.path.join(plans_dir, "nnUNetPlans_override.json")
        with open(override_path, "w") as f:
            json.dump(override, f)

    def _train_with_monitor(self, cmd):
        """Run training subprocess while monitoring log for early stopping."""
        log_path = self._training_log_path()
        proc     = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)

        best_val_loss  = float("inf")
        es_wait        = 0
        stop_requested = False

        def stream_output():
            for line in proc.stdout:
                print(line, end="")

        t = threading.Thread(target=stream_output, daemon=True)
        t.start()

        while proc.poll() is None:
            time.sleep(30)  # check every 30 s
            epochs = parse_nnunet_training_log(log_path)
            if epochs:
                last = epochs[-1]
                val_loss = last.get("val_loss", float("inf"))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    es_wait = 0
                else:
                    es_wait += 1
                if es_wait >= self.es_patience and not stop_requested:
                    print(f"\n⏹  Early stopping: val_loss no improvement "
                          f"for {self.es_patience} checks. Sending SIGTERM.")
                    proc.terminate()
                    stop_requested = True

        t.join(timeout=5)
        rc = proc.wait()
        if rc not in (0, -15):   # -15 = SIGTERM (our early stop)
            raise RuntimeError(f"nnUNetv2_train exited with code {rc}")
        print("✓ Training complete.")

    def predict(self, input_dir, output_dir):
        print("── Step 3: Predict ──")
        os.makedirs(output_dir, exist_ok=True)
        model_dir = os.path.join(
            self.results, self.dataset_name,
            f"{self.trainer}__{self.dataset_name}__"
            f"nnUNetPlans__{self.config}",
        )
        cmd = [
            "nnUNetv2_predict",
            "-i", input_dir,
            "-o", output_dir,
            "-d", str(self.dataset_id),
            "-c", self.config,
            "-f", str(self.fold),
        ]
        self._run(cmd)

    def build_epoch_csv(self):
        """Parse training log and write per-epoch rows to CSV."""
        if not self.log_file:
            return
        log_path = self._training_log_path()
        epochs   = parse_nnunet_training_log(log_path)
        init_csv(self.log_file)
        for row in epochs:
            ep       = row.get("epoch", 0)
            tr_loss  = row.get("train_loss", float("nan"))
            va_loss  = row.get("val_loss",   float("nan"))
            va_dice  = row.get("val_dice",   float("nan"))
            # IoU / accuracy / HD not available per-epoch from log
            log_epoch_row(
                self.log_file, ep,
                tr_loss, float("nan"),
                va_loss, va_dice,
                float("nan"), float("nan"), float("nan"), float("nan"),
                lr=float("nan"),
                image_size=self.image_size,
                dataset_fraction=self.dataset_fraction,
                dataset_source=self.dataset_source,
                batch_size=self.batch_size,
                total_epochs=self.epochs,
                model_id=self.model_id,
            )
        print(f"✓ {len(epochs)} epoch rows written to CSV.")
        return epochs

    def _training_log_path(self):
        return os.path.join(
            self.results, self.dataset_name,
            f"{self.trainer}__{self.dataset_name}__"
            f"nnUNetPlans__{self.config}",
            f"fold_{self.fold}", "training_log.txt",
        )

    @staticmethod
    def _run(cmd):
        print("  $", " ".join(cmd))
        r = subprocess.run(cmd, check=True)
        return r
