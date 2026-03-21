"""
nnUNet2_TestEnvironment.py
==========================
MIC-DKFZ nnU-Net (nn2) — trained via Python trainer class directly.

Approach:
  1. PNG slices → NIfTI volumes (auto inside notebook)
  2. nnU-Net auto-configuration (fingerprint + plans)
  3. Training via nnUNetTrainer Python API with per-epoch hooks
  4. Early stopping + ReduceLROnPlateau controlled from Python
  5. Per-epoch metrics logged to CSV in real time
  6. Inference on test NIfTIs → your metric functions → CSV summary

Key difference from nn1:
  • Training is driven from Python (not CLI subprocess)
  • Full per-epoch CSV logging (loss, dice, iou, accuracy, hausdorff)
  • Configurable ES_PATIENCE, LR_PATIENCE, LR_FACTOR identical to notebook
  • Most compatible with existing notebook structure
"""

import os
import csv
import glob
import json
import shutil
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy.ndimage import distance_transform_edt

try:
    import nibabel as nib
except ImportError:
    nib = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None


# ── NIfTI conversion (shared with nn1) ───────────────────────────────────────

def png_slices_to_nifti(volume_ids, dataset_dir, out_dir,
                        image_size, file_ext=".png"):
    """
    Convert LiTS PNG slices for a list of volume IDs into NIfTI volumes.

    Each volume becomes:
      imagesTr/  liver_XXXX_0000.nii.gz
      labelsTr/  liver_XXXX.nii.gz
    """
    assert nib   is not None, "nibabel required: pip install nibabel"
    assert Image is not None, "Pillow required: pip install Pillow"

    img_dir = os.path.join(out_dir, "imagesTr")
    lbl_dir = os.path.join(out_dir, "labelsTr")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    converted = 0
    for vol_id in sorted(volume_ids, key=lambda x: int(x)):
        slices_img, slices_lbl = [], []
        pattern     = os.path.join(dataset_dir, f"volume-{vol_id}_*.png")
        slice_files = sorted(glob.glob(pattern),
                             key=lambda p: int(p.split("_")[-1].replace(".png","")))
        if not slice_files:
            continue
        for sl_path in slice_files:
            sl_idx    = sl_path.split("_")[-1].replace(".png","")
            mask_path = os.path.join(dataset_dir,
                          f"segmentation-{vol_id}_livermask_{sl_idx}.png")
            img = Image.open(sl_path).convert("L")
            img = img.resize((image_size, image_size), Image.BILINEAR)
            slices_img.append(np.array(img, dtype=np.float32) / 255.0)

            if os.path.exists(mask_path):
                msk = Image.open(mask_path).convert("L")
                msk = msk.resize((image_size, image_size), Image.NEAREST)
                slices_lbl.append((np.array(msk) > 0).astype(np.uint8))
            else:
                slices_lbl.append(np.zeros((image_size, image_size), dtype=np.uint8))

        vol_np  = np.stack(slices_img, axis=2)   # H x W x D
        lbl_np  = np.stack(slices_lbl, axis=2)
        case_id = f"liver_{int(vol_id):04d}"

        nib.save(nib.Nifti1Image(vol_np,  np.eye(4)),
                 os.path.join(img_dir, f"{case_id}_0000.nii.gz"))
        nib.save(nib.Nifti1Image(lbl_np.astype(np.int16), np.eye(4)),
                 os.path.join(lbl_dir, f"{case_id}.nii.gz"))
        converted += 1

    print(f"✓ Converted {converted} volumes → {out_dir}")
    return converted


def write_dataset_json(out_dir, train_ids, val_ids,
                       channel_names=None, labels=None,
                       dataset_name="LiTS_NN2"):
    channel_names = channel_names or {"0": "CT"}
    labels        = labels        or {"background": 0, "liver": 1}
    training = []
    for vid in list(train_ids) + list(val_ids):
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


# ── Metric functions ──────────────────────────────────────────────────────────

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
    hd_vals = []
    for z in range(pred_vol.shape[2]):
        dists = _surface_distances(
            pred_vol[:,:,z].astype(bool),
            true_vol[:,:,z].astype(bool))
        hd_vals.append(float(np.percentile(dists, percentile)))
    return float(np.mean(hd_vals))


def compute_metrics_from_niftis(pred_dir, label_dir, case_ids,
                                 hd_percentile=95):
    assert nib is not None, "nibabel required"
    dice_l, iou_l, acc_l, hd_l = [], [], [], []
    for vid in case_ids:
        case_id   = f"liver_{int(vid):04d}"
        pred_path = os.path.join(pred_dir,  f"{case_id}.nii.gz")
        lbl_path  = os.path.join(label_dir, f"{case_id}.nii.gz")
        if not os.path.exists(pred_path) or not os.path.exists(lbl_path):
            continue
        pred = (nib.load(pred_path).get_fdata() > 0.5).astype(bool)
        true = (nib.load(lbl_path).get_fdata()  > 0  ).astype(bool)
        dice_l.append(dice_coefficient_np(pred, true))
        iou_l.append(iou_score_np(pred, true))
        acc_l.append(pixel_accuracy_np(pred, true))
        hd_l.append(hausdorff_volume(pred, true, hd_percentile))
    return {
        "dice":      float(np.mean(dice_l))  if dice_l else float("nan"),
        "iou":       float(np.mean(iou_l))   if iou_l  else float("nan"),
        "jaccard":   float(np.mean(iou_l))   if iou_l  else float("nan"),
        "accuracy":  float(np.mean(acc_l))   if acc_l  else float("nan"),
        "hausdorff": float(np.nanmean(hd_l)) if hd_l   else float("nan"),
    }


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


# ── NNUNet2Trainer — Python-controlled training ───────────────────────────────

class NNUNet2Trainer:
    """
    Wraps nnUNetTrainer for Python-controlled training with:
      - Per-epoch CSV logging (all metrics)
      - Early stopping (ES_PATIENCE)
      - ReduceLROnPlateau (LR_PATIENCE, LR_FACTOR, LR_MIN)
      - Hausdorff computed every HD_EVERY epochs (expensive)

    Usage
    -----
    trainer = NNUNet2Trainer(
        dataset_id=1,
        nnunet_raw=..., nnunet_preprocessed=..., nnunet_results=...,
        epochs=50, batch_size=8, learning_rate=1e-4,
        es_patience=10, lr_patience=5, lr_factor=0.5, lr_min=1e-7,
        hd_percentile=95, log_file=..., image_size=128,
        dataset_fraction=0.25, dataset_source="LiTS", model_id="NN2"
    )
    trainer.plan_and_preprocess()
    trainer.train(train_ids, val_ids)
    metrics = trainer.evaluate(test_ids, label_dir)
    """

    def __init__(self,
                 dataset_id=1,
                 nnunet_raw="",
                 nnunet_preprocessed="",
                 nnunet_results="",
                 config="2d",
                 fold=0,
                 epochs=50,
                 batch_size=8,
                 learning_rate=1e-4,
                 es_patience=10,
                 lr_patience=5,
                 lr_factor=0.5,
                 lr_min=1e-7,
                 hd_percentile=95,
                 hd_every=5,
                 log_file=None,
                 image_size=128,
                 dataset_fraction=0.25,
                 dataset_source="LiTS",
                 model_id="NN2"):

        self.dataset_id   = dataset_id
        self.raw          = nnunet_raw
        self.preprocessed = nnunet_preprocessed
        self.results      = nnunet_results
        self.config       = config
        self.fold         = fold
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = learning_rate
        self.es_patience  = es_patience
        self.lr_patience  = lr_patience
        self.lr_factor    = lr_factor
        self.lr_min       = lr_min
        self.hd_percentile= hd_percentile
        self.hd_every     = hd_every
        self.log_file     = log_file
        self.image_size   = image_size
        self.dataset_fraction = dataset_fraction
        self.dataset_source   = dataset_source
        self.model_id     = model_id

        self.dataset_name = f"Dataset{dataset_id:03d}_LiTS"

        # Set nnU-Net env vars
        os.environ["nnUNet_raw"]          = nnunet_raw
        os.environ["nnUNet_preprocessed"] = nnunet_preprocessed
        os.environ["nnUNet_results"]      = nnunet_results

        self._nnunet_trainer = None

    def plan_and_preprocess(self):
        """Run nnU-Net fingerprinting and preprocessing."""
        print("── Plan & Preprocess ──")
        import subprocess
        cmd = [
            "nnUNetv2_plan_and_preprocess",
            "-d", str(self.dataset_id),
            "--verify_dataset_integrity",
        ]
        subprocess.run(cmd, check=True)
        print("✓ Plan & preprocess complete.")

    def _build_nnunet_trainer(self):
        """Instantiate nnUNetTrainer with auto-configured plans."""
        from nnunetv2.run.run_training import get_trainer_from_args
        from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

        nnUNetTrainer = recursive_find_python_class(
            [os.path.join(os.path.dirname(__import__("nnunetv2").__file__),
                          "training", "nnUNetTrainer")],
            "nnUNetTrainer",
            current_module="nnunetv2.training.nnUNetTrainer",
        )

        trainer = nnUNetTrainer(
            plans          = self._load_plans(),
            configuration  = self.config,
            fold           = self.fold,
            dataset_json   = self._load_dataset_json(),
            unpack_dataset = True,
            device         = __import__("torch").device(
                                "cuda" if __import__("torch").cuda.is_available()
                                else "cpu"),
        )
        # Override hyperparameters
        trainer.num_epochs              = self.epochs
        trainer.initial_lr              = self.lr
        trainer.oversample_foreground_percent = 0.33

        return trainer

    def _load_plans(self):
        import json
        plans_path = os.path.join(
            self.preprocessed, self.dataset_name, "nnUNetPlans.json")
        with open(plans_path) as f:
            return json.load(f)

    def _load_dataset_json(self):
        import json
        ds_path = os.path.join(self.raw, self.dataset_name, "dataset.json")
        with open(ds_path) as f:
            return json.load(f)

    def train(self, train_ids, val_ids):
        """
        Run per-epoch training loop with full metric logging,
        early stopping, and LR scheduling.
        """
        import torch
        import torch.optim as optim

        print("── Building nnUNetTrainer ──")
        tr = self._build_nnunet_trainer()
        tr.initialize()
        self._nnunet_trainer = tr

        optimizer = tr.optimizer
        # Replace nnU-Net's scheduler with ReduceLROnPlateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=self.lr_factor,
            patience=self.lr_patience, min_lr=self.lr_min)

        if self.log_file:
            init_csv(self.log_file)

        best_val_loss  = float("inf")
        best_val_dice  = 0.0
        best_val_iou   = best_val_jaccard = best_val_acc = 0.0
        best_val_hd    = float("nan")
        es_wait        = 0
        all_val_dice, all_val_iou   = [], []
        all_val_jaccard, all_val_acc = [], []
        all_val_hd     = []

        print("=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)

        for epoch in range(1, self.epochs + 1):

            # ── Train one epoch ──────────────────────────────────────────────
            tr.on_epoch_start()
            tr.on_train_epoch_start()
            train_outputs = []
            for batch in tr.dataloader_train:
                train_outputs.append(tr.train_step(batch))
            tr.on_train_epoch_end(train_outputs)

            # ── Validate ─────────────────────────────────────────────────────
            tr.on_validation_epoch_start()
            val_outputs = []
            for batch in tr.dataloader_val:
                val_outputs.append(tr.validation_step(batch))
            tr.on_validation_epoch_end(val_outputs)

            tr_loss = float(np.mean([o["loss"] for o in train_outputs]))
            va_loss = float(np.mean([o["loss"] for o in val_outputs]))

            # Dice from nnU-Net's own pseudo-dice
            va_dice = float(tr.logger.my_fantastic_logging.get(
                "mean_fg_dice", [float("nan")])[-1])
            tr_dice = va_dice  # nnU-Net doesn't log train dice separately

            # Full metrics on val NIfTIs every hd_every epochs
            compute_hd = (epoch % self.hd_every == 0) or (epoch == self.epochs)
            va_iou = va_jaccard = va_acc = float("nan")
            va_hd  = float("nan")

            if compute_hd and val_ids is not None:
                # Run quick inference on val set for full metrics
                val_metrics = self._quick_eval(val_ids, compute_hd=True)
                va_iou      = val_metrics["iou"]
                va_jaccard  = val_metrics["jaccard"]
                va_acc      = val_metrics["accuracy"]
                va_hd       = val_metrics["hausdorff"]
                va_dice     = val_metrics["dice"]

            current_lr = optimizer.param_groups[0]["lr"]
            all_val_dice.append(va_dice)
            all_val_iou.append(va_iou)
            all_val_jaccard.append(va_jaccard)
            all_val_acc.append(va_acc)
            all_val_hd.append(va_hd)

            if self.log_file:
                log_epoch_row(
                    self.log_file, epoch,
                    tr_loss, tr_dice,
                    va_loss, va_dice, va_iou, va_jaccard, va_acc, va_hd,
                    current_lr,
                    self.image_size, self.dataset_fraction,
                    self.dataset_source, self.batch_size,
                    self.epochs, self.model_id,
                )

            hd_str = f"{va_hd:.2f}" if not np.isnan(va_hd) else "n/a"
            print(f"Epoch {epoch:>3}/{self.epochs}  "
                  f"Loss: {tr_loss:.4f}/{va_loss:.4f}  "
                  f"Dice: {tr_dice:.4f}/{va_dice:.4f}  "
                  f"IoU: {va_iou:.4f}  Acc: {va_acc:.4f}  "
                  f"HD{self.hd_percentile}: {hd_str}  LR: {current_lr:.2e}")

            # ── Save best ─────────────────────────────────────────────────────
            if va_dice > best_val_dice:
                best_val_dice    = va_dice
                best_val_iou     = va_iou
                best_val_jaccard = va_jaccard
                best_val_acc     = va_acc
                best_val_hd      = va_hd
                tr.save_checkpoint(
                    os.path.join(tr.output_folder, "checkpoint_best.pth"))
                print(f"  ✅ New best saved! (Dice={best_val_dice:.4f})")

            # ── LR scheduler ─────────────────────────────────────────────────
            scheduler.step(va_loss)

            # ── Early stopping ────────────────────────────────────────────────
            if va_loss < best_val_loss:
                best_val_loss = va_loss
                es_wait = 0
            else:
                es_wait += 1
                if es_wait >= self.es_patience:
                    print(f"\n⏹  Early stopping at epoch {epoch}.")
                    break

            tr.on_epoch_end()

        # Store summary stats for CSV
        self._best  = (best_val_dice, best_val_iou, best_val_jaccard,
                       best_val_acc, best_val_hd)
        self._means = (
            float(np.nanmean(all_val_dice)),
            float(np.nanmean(all_val_iou)),
            float(np.nanmean(all_val_jaccard)),
            float(np.nanmean(all_val_acc)),
            float(np.nanmean(all_val_hd)),
        )

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Best Val  Dice: {best_val_dice:.4f}  IoU: {best_val_iou:.4f}")

    def _quick_eval(self, vol_ids, compute_hd=False):
        """
        Run nnU-Net inference on a set of NIfTI volumes and return metrics.
        Uses the current (best) checkpoint.
        """
        import subprocess, tempfile
        img_dir  = os.path.join(self.raw, self.dataset_name, "imagesTr")
        lbl_dir  = os.path.join(self.raw, self.dataset_name, "labelsTr")
        pred_dir = os.path.join(self.results, "tmp_eval_preds")
        os.makedirs(pred_dir, exist_ok=True)

        # Write a temp input folder with only the requested volumes
        tmp_in = os.path.join(self.results, "tmp_eval_in")
        os.makedirs(tmp_in, exist_ok=True)
        for vid in vol_ids:
            case_id = f"liver_{int(vid):04d}"
            src = os.path.join(img_dir, f"{case_id}_0000.nii.gz")
            dst = os.path.join(tmp_in,  f"{case_id}_0000.nii.gz")
            if os.path.exists(src) and not os.path.exists(dst):
                os.symlink(src, dst)

        subprocess.run([
            "nnUNetv2_predict",
            "-i", tmp_in, "-o", pred_dir,
            "-d", str(self.dataset_id),
            "-c", self.config, "-f", str(self.fold),
            "--save_probabilities",
        ], check=True, capture_output=True)

        metrics = compute_metrics_from_niftis(
            pred_dir, lbl_dir, vol_ids, self.hd_percentile)
        shutil.rmtree(tmp_in,  ignore_errors=True)
        shutil.rmtree(pred_dir, ignore_errors=True)
        return metrics

    def evaluate(self, test_ids, label_dir=None):
        """
        Run inference on test set and compute full metrics.
        Returns dict with dice, iou, jaccard, accuracy, hausdorff.
        """
        import subprocess
        img_dir  = os.path.join(self.raw, self.dataset_name, "imagesTr")
        lbl_dir  = label_dir or os.path.join(
            self.raw, self.dataset_name, "labelsTr")
        pred_dir = os.path.join(self.results, "test_predictions")
        os.makedirs(pred_dir, exist_ok=True)

        # Build test input folder
        test_in = os.path.join(self.results, "test_in")
        os.makedirs(test_in, exist_ok=True)
        for vid in test_ids:
            case_id = f"liver_{int(vid):04d}"
            src = os.path.join(img_dir, f"{case_id}_0000.nii.gz")
            dst = os.path.join(test_in, f"{case_id}_0000.nii.gz")
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

        print("── Test Set Inference ──")
        subprocess.run([
            "nnUNetv2_predict",
            "-i", test_in, "-o", pred_dir,
            "-d", str(self.dataset_id),
            "-c", self.config, "-f", str(self.fold),
        ], check=True)

        print("── Computing Test Metrics ──")
        metrics = compute_metrics_from_niftis(
            pred_dir, lbl_dir, test_ids, self.hd_percentile)

        print("=" * 60)
        print("TEST SET RESULTS")
        print("=" * 60)
        print(f"Test Dice:      {metrics['dice']:.4f}")
        print(f"Test IoU:       {metrics['iou']:.4f}")
        print(f"Test Jaccard:   {metrics['jaccard']:.4f}")
        print(f"Test Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Test HD{self.hd_percentile}:      {metrics['hausdorff']:.4f}")
        print("=" * 60)

        return metrics

    def write_summary_to_csv(self, test_metrics):
        if not self.log_file:
            return
        best = self._best
        means = self._means
        log_summary_rows(
            self.log_file,
            best[0], best[1], best[2], best[3], best[4],
            means[0], means[1], means[2], means[3], means[4],
            test_metrics["dice"],    test_metrics["iou"],
            test_metrics["jaccard"], test_metrics["accuracy"],
            test_metrics["hausdorff"],
            self.image_size, self.dataset_fraction, self.dataset_source,
            self.batch_size, self.epochs, self.model_id,
        )
        print(f"✓ Summary written to: {self.log_file}")
