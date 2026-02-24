# =========================
# 4b️⃣  Inline TestTrainer
# =========================

_TRAINER_CODE = '''
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch, csv, os, json, numpy as np
from datetime import datetime
from pathlib import Path
from scipy.ndimage import distance_transform_edt

SMOOTH = 1e-5

def _dice(p, t):
    i = (p & t).sum(); d = p.sum() + t.sum()
    return float((2*i + SMOOTH) / (d + SMOOTH))

def _iou(p, t):
    i = (p & t).sum(); u = p.sum() + t.sum() - i
    return float((i + SMOOTH) / (u + SMOOTH))

def _accuracy(p, t):
    return float((p == t).sum() / p.size)

def _surface_distances(p, t):
    if not p.any() and not t.any(): return np.array([0.0])
    if not p.any() or  not t.any(): return np.array([np.inf])
    dt_t = distance_transform_edt(~t); dt_p = distance_transform_edt(~p)
    sp = p & ~np.pad(p, 1, mode="edge")[1:-1, 1:-1]
    st = t & ~np.pad(t, 1, mode="edge")[1:-1, 1:-1]
    d1 = dt_t[sp]; d2 = dt_p[st]
    if d1.size and d2.size: return np.concatenate([d1, d2])
    return np.array([np.inf])

def _hd_batch(preds, masks, pct):
    vals = []
    for p, m in zip(preds, masks):
        dists = _surface_distances((p > 0.5), m.astype(bool))
        finite = dists[np.isfinite(dists)]
        vals.append(float(np.percentile(finite, pct)) if finite.size > 0 else float("nan"))
    finite_vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(finite_vals)) if finite_vals else float("nan")

class TestTrainer(nnUNetTrainer):
    _EPOCHS              = {EPOCHS}
    _LR                  = {LR}
    _BATCH_SIZE_OVERRIDE = {BATCH_SIZE_NNUNET}
    _LOG_DIR             = r"{LOG_DIR}"
    _IMAGE_SIZE          = {IMAGE_SIZE}
    _DATASET_FRACTION    = {DATASET_FRACTION}
    _DATASET_SOURCE      = "{DATASET_SOURCE}"
    _HD_PERCENTILE       = {HD_PERCENTILE}
    _TRAIN_KEYS          = {TRAIN_KEYS}
    _VAL_KEYS            = {VAL_KEYS}

    def initialize(self):
        self.num_epochs = self._EPOCHS
        if self._LR is not None:
            self.initial_lr = self._LR
        super().initialize()

        # Overwrite nnUNet auto-split with our exact train/val split
        preprocessed_dir = Path(os.environ.get("nnUNet_preprocessed", "/content/nnUNet_preprocessed"))
        splits_file = preprocessed_dir / self.plans_manager.dataset_name / "splits_final.json"
        with open(splits_file, "w") as f:
            json.dump([{"train": self._TRAIN_KEYS, "val": self._VAL_KEYS}], f)
        print(f"[TestTrainer] Split written: {len(self._TRAIN_KEYS)} train, {len(self._VAL_KEYS)} val")

        os.makedirs(self._LOG_DIR, exist_ok=True)
        self.csv_log_file = os.path.join(
            self._LOG_DIR,
            f"nnUNet_E{self.num_epochs}_IMG{self._IMAGE_SIZE}"
            f"_F{self._DATASET_FRACTION}_{self._DATASET_SOURCE}.csv"
        )
        self._csv_cols = [
            "type", "timestamp", "epoch",
            "train_loss", "train_dice",
            "val_loss", "val_dice", "val_iou", "val_jaccard", "val_accuracy", "val_hausdorff",
            "lr", "image_size", "dataset_fraction", "dataset_source", "batch_size", "total_epochs",
        ]
        if not os.path.exists(self.csv_log_file):
            with open(self.csv_log_file, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self._csv_cols).writeheader()
        self._v_dice=[]; self._v_iou=[]; self._v_jac=[]; self._v_acc=[]; self._v_hd=[]

    def configure_optimizers(self):
        lr = self.initial_lr
        opt = torch.optim.SGD(
            self.network.parameters(), lr,
            weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        if self._LR is not None:
            sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=self.num_epochs)
        else:
            sched = torch.optim.lr_scheduler.PolynomialLR(opt, total_iters=self.num_epochs, power=0.9)
        return opt, sched

    def _extended_val(self, compute_hd=False):
        """Dice/IoU/Acc every epoch. HD only on final epoch (compute_hd=True)."""
        self.network.eval()
        d_v=[]; i_v=[]; a_v=[]; h_v=[]
        with torch.no_grad():
            for batch in self.dataloader_val:
                data   = batch["data"].to(self.device)
                target = batch["target"]
                target = [t.to(self.device) for t in target] if isinstance(target, list) else target.to(self.device)
                out = self.network(data)
                if isinstance(out, (list, tuple)): out = out[0]
                pn = torch.sigmoid(out).cpu().numpy()
                p  = (pn[:, 0] > 0.5)
                t  = (target[0].cpu().numpy()[:, 0] > 0.5) if isinstance(target, list) else (target.cpu().numpy()[:, 0] > 0.5)
                for pi, ti in zip(p, t):
                    d_v.append(_dice(pi, ti))
                    i_v.append(_iou(pi, ti))
                    a_v.append(_accuracy(pi, ti))
                if compute_hd:
                    h_v.append(_hd_batch(p, t, self._HD_PERCENTILE))
        hv = float(np.nanmean(h_v)) if h_v else float("nan")
        return float(np.mean(d_v)), float(np.mean(i_v)), float(np.mean(i_v)), float(np.mean(a_v)), hv

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
        is_last = (epoch == self.num_epochs - 1)
        vd, vi, vj, va, vh = self._extended_val(compute_hd=is_last)
        self._v_dice.append(vd); self._v_iou.append(vi)
        self._v_jac.append(vj);  self._v_acc.append(va); self._v_hd.append(vh)
        actual_bs = self.configuration_manager.configuration.get("batch_size", 0)
        with open(self.csv_log_file, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._csv_cols).writerow({
                "type":"epoch","timestamp":ts,"epoch":epoch,
                "train_loss":f"{tl:.6f}","train_dice":f"{td:.6f}",
                "val_loss":f"{vl:.6f}","val_dice":f"{vd:.6f}",
                "val_iou":f"{vi:.6f}","val_jaccard":f"{vj:.6f}",
                "val_accuracy":f"{va:.6f}","val_hausdorff":f"{vh:.4f}",
                "lr":f"{lr:.2e}",
                "image_size":self._IMAGE_SIZE,"dataset_fraction":self._DATASET_FRACTION,
                "dataset_source":self._DATASET_SOURCE,
                "batch_size":actual_bs,
                "total_epochs":self.num_epochs,
            })

    def on_train_end(self):
        super().on_train_end()
        d=self._v_dice; i=self._v_iou; j=self._v_jac
        a=self._v_acc;  hh=[x for x in self._v_hd if not np.isnan(x)]
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        actual_bs = self.configuration_manager.configuration.get("batch_size", 0)
        rows = [
            ("best_val",  max(d) if d else 0, max(i) if i else 0, max(j) if j else 0, max(a) if a else 0, min(hh) if hh else float("nan")),
            ("mean_val",  float(np.mean(d)) if d else 0, float(np.mean(i)) if i else 0, float(np.mean(j)) if j else 0, float(np.mean(a)) if a else 0, float(np.mean(hh)) if hh else float("nan")),
            ("test_final",d[-1] if d else 0, i[-1] if i else 0, j[-1] if j else 0, a[-1] if a else 0, hh[-1] if hh else float("nan")),
        ]
        with open(self.csv_log_file, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._csv_cols)
            for label, vd, vi, vj, va, vh in rows:
                w.writerow({
                    "type":"summary","timestamp":ts,"epoch":label,
                    "train_loss":"-","train_dice":f"{vd:.6f}",
                    "val_loss":"-","val_dice":f"{vd:.6f}",
                    "val_iou":f"{vi:.6f}","val_jaccard":f"{vj:.6f}",
                    "val_accuracy":f"{va:.6f}","val_hausdorff":f"{vh:.4f}",
                    "lr":"-",
                    "image_size":self._IMAGE_SIZE,"dataset_fraction":self._DATASET_FRACTION,
                    "dataset_source":self._DATASET_SOURCE,
                    "batch_size":actual_bs,
                    "total_epochs":self.num_epochs,
                })
'''

# ── Find nnUNet package location dynamically ──────────────────────────────────
import importlib, nnunetv2
_trainer_install_dir  = Path(nnunetv2.__file__).parent / "training" / "nnUNetTrainer"
_trainer_install_path = _trainer_install_dir / "TestTrainer.py"
print(f"Installing TestTrainer → {_trainer_install_path}")

# ── Keys from Section 3 ───────────────────────────────────────────────────────
_train_keys = _nnunet_train_keys
_val_keys   = _nnunet_val_keys

# ── Stamp Section 2 values into trainer code ──────────────────────────────────
_trainer_code = (
    _TRAINER_CODE
    .replace("{EPOCHS}",            str(EPOCHS))
    .replace("{LR}",                str(LEARNING_RATE_NNUNET))
    .replace("{BATCH_SIZE_NNUNET}", str(BATCH_SIZE_NNUNET))
    .replace("{LOG_DIR}",           LOG_DIR)
    .replace("{IMAGE_SIZE}",        str(IMAGE_SIZE))
    .replace("{DATASET_FRACTION}",  str(DATASET_FRACTION))
    .replace("{DATASET_SOURCE}",    DATASET_SOURCE)
    .replace("{HD_PERCENTILE}",     str(HD_PERCENTILE))
    .replace("{TRAIN_KEYS}",        repr(_train_keys))
    .replace("{VAL_KEYS}",          repr(_val_keys))
)

with open(_trainer_install_path, "w") as f:
    f.write(_trainer_code)

# ── Verify nnUNet can find it ─────────────────────────────────────────────────
import importlib
_mod = importlib.import_module("nnunetv2.training.nnUNetTrainer.TestTrainer")
assert hasattr(_mod, "TestTrainer"), "TestTrainer class not found after install!"
print("✅ Import check passed — nnUNet can find TestTrainer")

_trainer_drive_copy = os.path.join(LOG_DIR, f"trainer_source_E{EPOCHS}_IMG{IMAGE_SIZE}_inline.py")
shutil.copy2(str(_trainer_install_path), _trainer_drive_copy)
print(f"✅ Drive copy saved → {_trainer_drive_copy}")
print(f"   LR: {LEARNING_RATE_NNUNET}  |  Batch: {BATCH_SIZE_NNUNET}  |  Epochs: {EPOCHS}")
print(f"   Split: {len(_train_keys)} train, {len(_val_keys)} val")
