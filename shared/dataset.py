"""
shared/dataset.py
=================
MCT-LTDiag N50 dataset loading, RAM caching, and splitting.
Used by every fusion module — fusion modules call build_cached_dataset()
and receive a TensorDataset they can wrap however they need.

The full patient manifest and extraction pipeline (zip → tar → folders)
live in Cell 0 of the notebook.  By the time this module is imported,
EXTRACT_DIR already contains one sub-folder per patient:

    /content/mct_raw/
      <patient_id>/
        NIFTI/
          nc.nii.gz
          art.nii.gz
          pvp.nii.gz
        mask_pvp.nii.gz

Exports
-------
ALL_PATIENTS          dict[str, list[str]]   full 50-patient manifest
CATEGORY_LABELS       dict[str, int]         HCC→1 … CRLM→5

verify_patients(extract_dir, all_patients, dataset_fraction)
    → verified_pids: dict[str, list[str]]

build_raw_slices(extract_dir, verified_pids)
    → list of (patient_root: str, slice_idx: int, category: str)

build_cached_dataset(slice_index, load_fn, image_size, verbose)
    → TensorDataset  (imgs: N×C×H×W,  masks: N×1×H×W)

patient_level_split(slice_index, test_frac, val_frac, seed)
    → train_idx, val_idx, test_idx   (lists of global ints)

make_dataloaders(cached_ds, train_idx, val_idx, test_idx, batch_size, num_workers)
    → train_loader, val_loader, test_loader
"""

import gc
import os
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Patient manifest  (exactly as in nnUNet N0 notebook)
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

REQUIRED_NIFTI = ["nc.nii.gz", "art.nii.gz", "pvp.nii.gz"]
REQUIRED_ROOT  = ["mask_pvp.nii.gz"]
MCT_FUSION_MASK = "mask_pvp.nii.gz"


# ---------------------------------------------------------------------------
# Step 1 — verify patients on disk
# ---------------------------------------------------------------------------

def verify_patients(
    extract_dir: str,
    all_patients: dict[str, list[str]] | None = None,
    dataset_fraction: float = 1.0,
) -> dict[str, list[str]]:
    """
    Walk extract_dir, confirm required files exist for each patient,
    and apply dataset_fraction (takes first N patients per category).

    Args:
        extract_dir:      root of extracted patient folders.
        all_patients:     optional override; defaults to ALL_PATIENTS.
        dataset_fraction: 1.0 = all 50, 0.5 = 25, etc.

    Returns:
        verified_pids:  dict[category → list[patient_id]]
                        only patients whose files are all present.

    Raises:
        RuntimeError if any selected patient is missing required files.
    """
    if all_patients is None:
        all_patients = ALL_PATIENTS

    extract_dir    = Path(extract_dir)
    verified_pids  = {}
    missing_report = []

    for cat, pids in all_patients.items():
        n = max(1, round(len(pids) * dataset_fraction))
        selected = pids[:n]
        verified_pids[cat] = []

        for pid in selected:
            patient_root = extract_dir / pid
            if not patient_root.exists():
                missing_report.append(f"  ✗ [{cat}] {pid}: folder not found")
                continue
            nifti_dir     = patient_root / "NIFTI"
            missing_files = (
                [f for f in REQUIRED_NIFTI if not (nifti_dir / f).exists()] +
                [f for f in REQUIRED_ROOT  if not (patient_root / f).exists()]
            )
            if missing_files:
                missing_report.append(f"  ✗ [{cat}] {pid}: missing {missing_files}")
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

def build_raw_slices(
    extract_dir: str,
    verified_pids: dict[str, list[str]],
) -> list[tuple[str, int, str]]:
    """
    Enumerate all axial slices across all verified patients.

    Returns:
        List of (patient_root: str, slice_idx: int, category: str)
        The length of this list is the total number of slices.
    """
    extract_dir = Path(extract_dir)
    slices: list[tuple[str, int, str]] = []

    for cat, pids in verified_pids.items():
        for pid in pids:
            patient_root = extract_dir / pid
            pvp_path     = patient_root / "NIFTI" / "pvp.nii.gz"
            n_slices     = nib.load(str(pvp_path)).shape[2]
            for s in range(n_slices):
                slices.append((str(patient_root), s, cat))

    print(f"Total slices indexed: {len(slices):,}")
    return slices


# ---------------------------------------------------------------------------
# Step 3 — RAM-cache all slices
# ---------------------------------------------------------------------------

def build_cached_dataset(
    slice_index: list[tuple[str, int, str]],
    load_fn,
    image_size: int = 256,
    verbose: bool = True,
) -> TensorDataset:
    """
    Pre-load every slice into RAM as tensors.

    Each NIfTI volume is loaded exactly once per patient, keeping I/O minimal.

    Args:
        slice_index: list from build_raw_slices().
        load_fn:     callable(patient_root, slice_idx) → (img_tensor, mask_tensor)
                     This is supplied by the fusion module so each fusion
                     strategy controls exactly what goes into the cache.
        image_size:  spatial size (square) to resize each slice.
        verbose:     print patient-level progress.

    Returns:
        TensorDataset(imgs, masks)
            imgs  shape: (N, C, image_size, image_size)   float32
            masks shape: (N, 1, image_size, image_size)   float32 {0,1}
        where C is determined by the fusion load_fn.
    """
    # Group slices by patient so each volume is opened only once
    patient_entries: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for global_idx, (pr, s, cat) in enumerate(slice_index):
        patient_entries[pr].append((global_idx, s, cat))

    n_total     = len(slice_index)
    imgs_list:  list[torch.Tensor | None] = [None] * n_total
    masks_list: list[torch.Tensor | None] = [None] * n_total

    for p_idx, (patient_root, entries) in enumerate(
        sorted(patient_entries.items())
    ):
        pid = Path(patient_root).name
        if verbose:
            print(
                f"  [{p_idx + 1}/{len(patient_entries)}] {pid} "
                f"({len(entries)} slices)...",
                end=" ", flush=True,
            )

        for global_idx, s, _ in entries:
            img_t, mask_t = load_fn(patient_root, s)
            imgs_list[global_idx]  = img_t
            masks_list[global_idx] = mask_t

        gc.collect()
        if verbose:
            print("done")

    imgs_t  = torch.stack(imgs_list)   # type: ignore[arg-type]
    masks_t = torch.stack(masks_list)  # type: ignore[arg-type]
    del imgs_list, masks_list
    gc.collect()

    if verbose:
        img_mb  = imgs_t.nelement()  * imgs_t.element_size()  / 1e6
        msk_mb  = masks_t.nelement() * masks_t.element_size() / 1e6
        print(f"Cache complete: {img_mb:.0f} MB imgs + {msk_mb:.0f} MB masks")

    return TensorDataset(imgs_t, masks_t)


# ---------------------------------------------------------------------------
# Step 4 — patient-level split
# ---------------------------------------------------------------------------

def patient_level_split(
    slice_index: list[tuple[str, int, str]],
    test_frac:  float = 0.20,
    val_frac:   float = 0.20,
    seed:       int   = 42,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split by patient so no patient's slices appear in more than one partition.

    The split order mirrors nn-UNet: all patients → shuffle → assign test
    first, then val, then train.

    Args:
        slice_index: list from build_raw_slices().
        test_frac:   fraction of patients held out for testing.
        val_frac:    fraction of remaining patients for validation.
        seed:        RNG seed for reproducibility.

    Returns:
        train_idx, val_idx, test_idx  — global indices into slice_index
        (and therefore into the TensorDataset returned by build_cached_dataset).
    """
    all_patient_roots = sorted(set(pr for pr, _, _ in slice_index))
    n_patients        = len(all_patient_roots)

    rng              = np.random.default_rng(seed)
    shuffled         = rng.permutation(n_patients)

    n_test  = max(1, int(n_patients * test_frac))
    n_val   = max(1, int((n_patients - n_test) * val_frac))

    test_pats  = {all_patient_roots[i] for i in shuffled[:n_test]}
    val_pats   = {all_patient_roots[i] for i in shuffled[n_test: n_test + n_val]}
    train_pats = {all_patient_roots[i] for i in shuffled[n_test + n_val:]}

    train_idx = [gi for gi, (pr, _, _) in enumerate(slice_index) if pr in train_pats]
    val_idx   = [gi for gi, (pr, _, _) in enumerate(slice_index) if pr in val_pats]
    test_idx  = [gi for gi, (pr, _, _) in enumerate(slice_index) if pr in test_pats]

    print(
        f"Patient split (seed={seed}): "
        f"train={len(train_pats)} patients / {len(train_idx):,} slices | "
        f"val={len(val_pats)} patients / {len(val_idx):,} slices | "
        f"test={len(test_pats)} patients / {len(test_idx):,} slices"
    )
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Step 5 — DataLoaders
# ---------------------------------------------------------------------------

def make_dataloaders(
    cached_ds: TensorDataset,
    train_idx: list[int],
    val_idx:   list[int],
    test_idx:  list[int],
    batch_size:  int = 16,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Wrap split indices in DataLoaders with persistent workers and prefetch.

    Args:
        cached_ds:   TensorDataset from build_cached_dataset().
        train/val/test_idx: from patient_level_split().
        batch_size:  samples per batch.
        num_workers: 4 recommended when data is already in RAM.
                     Set 0 to disable multiprocessing (Windows / debug).

    Returns:
        train_loader, val_loader, test_loader
    """
    pw = num_workers > 0
    pf = 2 if num_workers > 0 else None

    def _loader(indices, shuffle):
        kwargs: dict = dict(
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
