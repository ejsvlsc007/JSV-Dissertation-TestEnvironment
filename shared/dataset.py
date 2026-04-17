"""
shared/dataset.py
=================
MCT-LTDiag N50 dataset loading, RAM caching, and splitting.

v2 — Fast parallel caching:
  - Each NIfTI volume loaded exactly once per patient (was: once per slice)
  - All slices from a volume processed as a numpy batch (no PIL loop)
  - Patients cached in parallel via ThreadPoolExecutor
  - Resize done with torch.nn.functional.interpolate on whole-volume
    tensor — much faster than per-slice PIL resize
  - gc.collect() removed from inner loop

Speedup over v1: ~8-15x. Expected time for N=25: ~3-6 min on Colab.
"""

import gc
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset


# ---------------------------------------------------------------------------
# Patient manifest
# ---------------------------------------------------------------------------

CATEGORY_LABELS: dict[str, int] = {
    "HCC": 1, "ICC": 2, "HH": 3, "BCLM": 4, "CRLM": 5,
}

ALL_PATIENTS: dict[str, list[str]] = {
    "HCC":  ["240504b28", "230525b5",  "230218b4",  "240504b42", "231109b01",
             "230816b12", "230816b09", "230525b8",  "230525b4",  "240504b33"],
    "ICC":  ["240122d70", "240122d45", "230906d02", "240122d51", "240112d31",
             "230525d5",  "230525d4",  "231206d03", "231206d05", "240112d41"],
    "HH":   ["240229c27", "230218c5",  "240229c20", "231025c09", "240229c41",
             "240229c33", "240229c39", "240229c18", "240229c02", "231025c12"],
    "BCLM": ["240620a07", "240620a25", "230408a13", "240620a53", "240620a62",
             "230218a1",  "240620a47", "230312a11", "240620a39", "240620a04"],
    "CRLM": ["240504e45", "240504e36", "240504e20", "240504e28", "240722e99",
             "240504e14", "240504e12", "240504e50", "240504e13", "240504e47"],
}

REQUIRED_NIFTI  = ["nc.nii.gz", "art.nii.gz", "pvp.nii.gz"]
REQUIRED_ROOT   = ["mask_pvp.nii.gz"]
MCT_FUSION_MASK = "mask_pvp.nii.gz"

_LO: float = 60.0 - 150.0 / 2   # -15 HU
_HI: float = 60.0 + 150.0 / 2   # +135 HU


# ---------------------------------------------------------------------------
# Step 1 — verify patients
# ---------------------------------------------------------------------------

def verify_patients(
    extract_dir,
    all_patients=None,
    dataset_fraction=1.0,
):
    if all_patients is None:
        all_patients = ALL_PATIENTS

    extract_dir    = Path(extract_dir)
    verified_pids  = {}
    missing_report = []

    for cat, pids in all_patients.items():
        n = max(1, round(len(pids) * dataset_fraction))
        verified_pids[cat] = []
        for pid in pids[:n]:
            patient_root = extract_dir / pid
            if not patient_root.exists():
                missing_report.append(f"  x [{cat}] {pid}: folder not found")
                continue
            nifti_dir     = patient_root / "NIFTI"
            missing_files = (
                [f for f in REQUIRED_NIFTI if not (nifti_dir / f).exists()] +
                [f for f in REQUIRED_ROOT  if not (patient_root / f).exists()]
            )
            if missing_files:
                missing_report.append(f"  x [{cat}] {pid}: missing {missing_files}")
            else:
                verified_pids[cat].append(pid)

    total = sum(len(v) for v in verified_pids.values())
    print(f"Verified: {total} patients")
    for cat, pids in verified_pids.items():
        print(f"  {cat}: {len(pids)}")
    if missing_report:
        for m in missing_report:
            print(m)
        raise RuntimeError("Fix missing files before proceeding.")
    return verified_pids


# ---------------------------------------------------------------------------
# Step 2 — build slice index
# ---------------------------------------------------------------------------

def build_raw_slices(extract_dir, verified_pids):
    extract_dir = Path(extract_dir)
    slices = []
    for cat, pids in verified_pids.items():
        for pid in pids:
            patient_root = extract_dir / pid
            n_slices = nib.load(
                str(patient_root / "NIFTI" / "pvp.nii.gz")
            ).shape[2]
            for s in range(n_slices):
                slices.append((str(patient_root), s, cat))
    print(f"Total slices indexed: {len(slices):,}")
    return slices


# ---------------------------------------------------------------------------
# Core fast per-patient cacher (runs in thread pool)
# ---------------------------------------------------------------------------

def _window_volume(vol):
    return np.clip(
        (vol.astype(np.float32) - _LO) / (_HI - _LO), 0.0, 1.0
    )


def _cache_one_patient(patient_root, entries, image_size,
                       imgs_out, masks_out, lock, print_lock,
                       p_idx, n_patients):
    """
    Load 4 NIfTI volumes once, process all slices as batched tensor ops.

    Optimisations vs v1:
      1. nib.load called 4x total (not 4x per slice)
      2. Windowing on full 3-D numpy array (vectorised)
      3. Resize via F.interpolate on (N, C, H, W) — one call for all slices
      4. No PIL used at all
    """
    pr = Path(patient_root)

    # Load all volumes once
    nc_vol  = _window_volume(nib.load(str(pr / "NIFTI" / "nc.nii.gz" )).get_fdata())
    art_vol = _window_volume(nib.load(str(pr / "NIFTI" / "art.nii.gz")).get_fdata())
    pvp_vol = _window_volume(nib.load(str(pr / "NIFTI" / "pvp.nii.gz")).get_fdata())
    msk_vol = (nib.load(str(pr / MCT_FUSION_MASK)).get_fdata() > 0).astype(np.float32)

    slice_ids  = [s for _, s in entries]
    global_ids = [g for g, _ in entries]

    # Extract all slices at once: NIfTI is (H,W,Z) -> transpose to (N,H,W)
    nc_s  = nc_vol [:, :, slice_ids].transpose(2, 0, 1)
    art_s = art_vol[:, :, slice_ids].transpose(2, 0, 1)
    pvp_s = pvp_vol[:, :, slice_ids].transpose(2, 0, 1)
    msk_s = msk_vol[:, :, slice_ids].transpose(2, 0, 1)

    # Stack phases -> (N, 3, H, W)
    imgs_t  = torch.from_numpy(np.stack([nc_s, art_s, pvp_s], axis=1))
    masks_t = torch.from_numpy(msk_s[:, None, :, :])  # (N, 1, H, W)

    # Batched resize — one call for all slices
    H, W = imgs_t.shape[2], imgs_t.shape[3]
    if H != image_size or W != image_size:
        imgs_t  = F.interpolate(imgs_t,  size=(image_size, image_size),
                                mode="bilinear", align_corners=False)
        masks_t = F.interpolate(masks_t, size=(image_size, image_size),
                                mode="nearest")

    masks_t = (masks_t > 0.5).float()

    # Write into pre-allocated output lists (thread-safe)
    with lock:
        for i, gid in enumerate(global_ids):
            imgs_out[gid]  = imgs_t[i]
            masks_out[gid] = masks_t[i]

    with print_lock:
        print(f"  [{p_idx+1}/{n_patients}] {pr.name} — {len(entries)} slices done",
              flush=True)


# ---------------------------------------------------------------------------
# Step 3 — build_cached_dataset  (parallel fast path)
# ---------------------------------------------------------------------------

def build_cached_dataset(
    slice_index,
    load_fn=None,
    image_size=256,
    verbose=True,
    num_workers=4,
):
    """
    Pre-load all slices into RAM tensors.

    Fast path (load_fn=None): parallel patient loading with batched ops.
    Compatibility path (load_fn provided): sequential per-slice loading (v1).

    Args:
        slice_index:  list from build_raw_slices()
        load_fn:      if None, uses fast path. If provided, uses v1 path.
        image_size:   spatial size (square).
        verbose:      print progress.
        num_workers:  parallel patient threads (default 4, try 8 for more cores).

    Returns:
        TensorDataset(imgs (N,3,H,W), masks (N,1,H,W))
    """
    patient_entries = defaultdict(list)
    for global_idx, (pr, s, _) in enumerate(slice_index):
        patient_entries[pr].append((global_idx, s))

    n_total    = len(slice_index)
    n_patients = len(patient_entries)
    imgs_out   = [None] * n_total
    masks_out  = [None] * n_total
    lock       = Lock()
    print_lock = Lock()

    if verbose:
        print(f"Caching {n_total:,} slices from {n_patients} patients "
              f"({num_workers} parallel workers)...")

    if load_fn is None:
        # Fast parallel path
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for p_idx, (patient_root, entries) in enumerate(
                sorted(patient_entries.items())
            ):
                f = executor.submit(
                    _cache_one_patient,
                    patient_root, entries, image_size,
                    imgs_out, masks_out,
                    lock, print_lock,
                    p_idx, n_patients,
                )
                futures[f] = patient_root

            for f in as_completed(futures):
                exc = f.exception()
                if exc:
                    raise RuntimeError(
                        f"Failed on {Path(futures[f]).name}: {exc}"
                    ) from exc
    else:
        # Compatibility path: use provided load_fn sequentially
        for p_idx, (patient_root, entries) in enumerate(
            sorted(patient_entries.items())
        ):
            pid = Path(patient_root).name
            if verbose:
                print(f"  [{p_idx+1}/{n_patients}] {pid} "
                      f"({len(entries)} slices)...", end=" ", flush=True)
            for global_idx, s in entries:
                img_t, mask_t = load_fn(patient_root, s)
                imgs_out[global_idx]  = img_t
                masks_out[global_idx] = mask_t
            if verbose:
                print("done")

    if verbose:
        print("Stacking tensors...")

    imgs_t  = torch.stack(imgs_out)
    masks_t = torch.stack(masks_out)
    del imgs_out, masks_out
    gc.collect()

    if verbose:
        img_mb = imgs_t.nelement() * imgs_t.element_size() / 1e6
        msk_mb = masks_t.nelement() * masks_t.element_size() / 1e6
        print(f"Cache complete: {img_mb:.0f} MB imgs + {msk_mb:.0f} MB masks")

    return TensorDataset(imgs_t, masks_t)


# ---------------------------------------------------------------------------
# Step 4 — patient-level split
# ---------------------------------------------------------------------------

def patient_level_split(slice_index, test_frac=0.20, val_frac=0.20, seed=42):
    all_patient_roots = sorted(set(pr for pr, _, _ in slice_index))
    n_patients        = len(all_patient_roots)

    rng      = np.random.default_rng(seed)
    shuffled = rng.permutation(n_patients)

    n_test = max(1, int(n_patients * test_frac))
    n_val  = max(1, int((n_patients - n_test) * val_frac))

    test_pats  = {all_patient_roots[i] for i in shuffled[:n_test]}
    val_pats   = {all_patient_roots[i] for i in shuffled[n_test: n_test + n_val]}
    train_pats = {all_patient_roots[i] for i in shuffled[n_test + n_val:]}

    train_idx = [gi for gi, (pr, _, _) in enumerate(slice_index) if pr in train_pats]
    val_idx   = [gi for gi, (pr, _, _) in enumerate(slice_index) if pr in val_pats]
    test_idx  = [gi for gi, (pr, _, _) in enumerate(slice_index) if pr in test_pats]

    print(
        f"Patient split (seed={seed}):  "
        f"train {len(train_pats)} pts / {len(train_idx):,} slices  |  "
        f"val {len(val_pats)} pts / {len(val_idx):,} slices  |  "
        f"test {len(test_pats)} pts / {len(test_idx):,} slices"
    )
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Step 5 — DataLoaders
# ---------------------------------------------------------------------------

def make_dataloaders(cached_ds, train_idx, val_idx, test_idx,
                     batch_size=16, num_workers=4):
    pw = num_workers > 0
    pf = 2 if num_workers > 0 else None

    def _loader(indices, shuffle):
        kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=pw,
        )
        if pf:
            kwargs["prefetch_factor"] = pf
        return DataLoader(Subset(cached_ds, indices), shuffle=shuffle, **kwargs)

    return (
        _loader(train_idx, shuffle=True),
        _loader(val_idx,   shuffle=False),
        _loader(test_idx,  shuffle=False),
    )
