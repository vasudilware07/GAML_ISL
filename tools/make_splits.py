#!/usr/bin/env python3
"""
Create deterministic, stratified per-class train/test splits and save as JSON.

Produces files in ``splits/`` that the dataset loader and evaluation scripts
consume via ``--split train|test``.

JSON format
-----------
    {
      "class_name_0": ["path/to/a.npy", "path/to/b.npy", ...],
      "class_name_1": ["path/to/c.npy", ...]
    }

All paths are relative to the *dataset root* (``data/processed/<name>``).

Example
-------
    python tools/make_splits.py --dataset asl_alphabet --seed 42 --ratio 0.7
    python tools/make_splits.py --dataset libras_alphabet --seed 42 --ratio 0.7
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SPLITS_DIR = REPO_ROOT / "splits"

DEFAULT_DATASETS = [
    "asl_alphabet",
    "libras_alphabet",
    "arabic_sign_alphabet",
    "thai_fingerspelling",
]


# ═════════════════════════════════════════════════════════════════════════════
#  Core split logic
# ═════════════════════════════════════════════════════════════════════════════

def _discover_classes(dataset_root: Path) -> Dict[str, List[str]]:
    """Return {class_name: sorted list of relative .npy paths}."""
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    class_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class sub-directories in {dataset_root}")
    result: Dict[str, List[str]] = {}
    for cls_dir in class_dirs:
        # Sorted deterministically by filename
        files = sorted(cls_dir.glob("*.npy"), key=lambda p: p.name)
        if files:
            # Store paths relative to dataset_root
            result[cls_dir.name] = [str(f.relative_to(dataset_root)) for f in files]
    return result


def make_stratified_split(
    class_files: Dict[str, List[str]],
    train_ratio: float = 0.7,
    seed: int = 42,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Deterministic stratified per-class split.

    For each class *c* with *n_c* samples:
        n_train = max(1, floor(train_ratio * n_c))
        n_test  = n_c - n_train
        if n_test == 0: move 1 sample from train to test

    Args:
        class_files: {class_name: sorted list of relative paths}.
        train_ratio: Fraction for train (default 0.7).
        seed: Numpy RandomState seed.

    Returns:
        (train_dict, test_dict) each mapping class_name -> list of paths.
    """
    rng = np.random.RandomState(seed)
    train_dict: Dict[str, List[str]] = {}
    test_dict: Dict[str, List[str]] = {}

    for cls_name in sorted(class_files.keys()):
        paths = list(class_files[cls_name])  # already sorted
        n_c = len(paths)

        # Deterministic permutation
        perm = rng.permutation(n_c)
        shuffled = [paths[i] for i in perm]

        n_train = max(1, math.floor(train_ratio * n_c))
        n_test = n_c - n_train
        if n_test == 0:
            # Move 1 from train to test
            n_train -= 1
            n_test = 1

        train_dict[cls_name] = shuffled[:n_train]
        test_dict[cls_name] = shuffled[n_train:]

    return train_dict, test_dict


def validate_split(
    train_dict: Dict[str, List[str]],
    test_dict: Dict[str, List[str]],
    dataset_name: str,
) -> None:
    """Assert no overlap and no duplicates within or across splits."""
    train_all: List[str] = []
    test_all: List[str] = []
    for cls in train_dict:
        train_all.extend(train_dict[cls])
    for cls in test_dict:
        test_all.extend(test_dict[cls])

    # Check duplicates within train
    if len(train_all) != len(set(train_all)):
        raise RuntimeError(f"[{dataset_name}] Duplicate paths in train split!")
    # Check duplicates within test
    if len(test_all) != len(set(test_all)):
        raise RuntimeError(f"[{dataset_name}] Duplicate paths in test split!")
    # Check overlap
    overlap = set(train_all) & set(test_all)
    if overlap:
        raise RuntimeError(
            f"[{dataset_name}] {len(overlap)} paths appear in BOTH train and test! "
            f"First: {list(overlap)[:3]}"
        )


def print_summary(
    split_dict: Dict[str, List[str]],
    label: str,
) -> None:
    """Print per-class count statistics for a split."""
    counts = [len(v) for v in split_dict.values()]
    if not counts:
        print(f"  {label}: EMPTY")
        return
    arr = np.array(counts)
    total = int(arr.sum())
    print(
        f"  {label}: {total} samples, {len(counts)} classes  "
        f"(min={arr.min()}, median={int(np.median(arr))}, max={arr.max()})"
    )


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Create deterministic stratified train/test split JSONs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=str, nargs="+", default=DEFAULT_DATASETS,
        help="Dataset name(s) under data/processed/ (default: all four)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for numpy.RandomState (default: 42)",
    )
    parser.add_argument(
        "--ratio", type=float, default=0.7,
        help="Train ratio (default: 0.7, test = 1 - ratio = 0.3)",
    )
    parser.add_argument(
        "--data_root", type=str, default=None,
        help="Override dataset root (default: $DATA_ROOT or data/processed/)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else (
        Path(os.environ["DATA_ROOT"]) if "DATA_ROOT" in os.environ
        else REPO_ROOT / "data" / "processed"
    )
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    for ds_name in args.dataset:
        # Try <name> first, then <name>_split (use flat source for splitting)
        ds_dir = data_root / ds_name
        if not ds_dir.exists():
            # Maybe there's already a _split dir; we split from the flat root
            alt = data_root / f"{ds_name}_split"
            if alt.exists():
                # Use the flat source if it exists alongside
                print(f"NOTE: {ds_dir} not found, but {alt} exists.")
                print(f"  Scanning flat source in {ds_dir}... ", end="")
                if not ds_dir.exists():
                    print(f"SKIP: no flat source for {ds_name}")
                    continue
            else:
                print(f"SKIP: {ds_dir} not found")
                continue

        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"Source:  {ds_dir}")
        print(f"Seed:    {args.seed}   Ratio: {args.ratio}")
        print(f"{'='*60}")

        class_files = _discover_classes(ds_dir)
        train_dict, test_dict = make_stratified_split(class_files, args.ratio, args.seed)
        validate_split(train_dict, test_dict, ds_name)

        print_summary(train_dict, "Train")
        print_summary(test_dict, "Test")

        # Save JSONs
        train_path = SPLITS_DIR / f"{ds_name}_train.json"
        test_path = SPLITS_DIR / f"{ds_name}_test.json"

        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(train_dict, f, indent=2, sort_keys=True)
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(test_dict, f, indent=2, sort_keys=True)

        print(f"  Saved: {train_path}")
        print(f"  Saved: {test_path}")

    print(f"\nDone. Split files are in {SPLITS_DIR}/")


if __name__ == "__main__":
    main()
