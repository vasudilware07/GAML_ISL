#!/usr/bin/env python3
"""
Log eligible class counts per dataset for each (k_shot, q_query) setting.

A class is eligible for an episode if its test-split sample count >= k + q.
Saves results/eligibility.json with full details.

Usage:
    python tools/log_eligibility.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.datasets import SplitLandmarkDataset


DATASETS = {
    "asl_alphabet":         "data/processed/asl_alphabet",
    "libras_alphabet":      "data/processed/libras_alphabet",
    "arabic_sign_alphabet": "data/processed/arabic_sign_alphabet",
    "thai_fingerspelling":  "data/processed/thai_fingerspelling",
}

SHOTS = [1, 3, 5]
Q_QUERY = 15
N_WAY = 5


def compute_eligibility(
    ds_name: str,
    data_root: str,
    split: str = "test",
    representations: List[str] = ["raw"],
) -> Dict:
    """Compute eligible class counts for a dataset.

    Returns a dict with per-class sample counts and eligibility for each k.
    """
    ds = SplitLandmarkDataset(ds_name, split, data_root, representations[0])
    labels = [ds[i][1] for i in range(len(ds))]
    class_counts = Counter(labels)
    total_classes = len(class_counts)
    total_samples = len(labels)

    per_class = {}
    for cls_id, count in sorted(class_counts.items()):
        per_class[str(cls_id)] = count

    eligibility = {}
    for k in SHOTS:
        threshold = k + Q_QUERY
        eligible = sum(1 for c in class_counts.values() if c >= threshold)
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        eligibility[f"k={k}_q={Q_QUERY}"] = {
            "threshold": threshold,
            "eligible_classes": eligible,
            "total_classes": total_classes,
            "eligible_fraction": round(eligible / total_classes, 3),
            "can_run_5way": eligible >= N_WAY,
            "min_class_count": min_count,
            "max_class_count": max_count,
        }

    return {
        "dataset": ds_name,
        "split": split,
        "total_classes": total_classes,
        "total_samples": total_samples,
        "per_class_counts": per_class,
        "eligibility": eligibility,
    }


def main():
    results = {}
    for ds_name, data_root in DATASETS.items():
        root = REPO_ROOT / data_root
        if not root.exists():
            print(f"SKIP {ds_name}: {root} not found")
            continue
        print(f"Processing {ds_name}...")
        info = compute_eligibility(ds_name, str(root))
        results[ds_name] = info

        # Print summary
        for setting, elig in info["eligibility"].items():
            status = "✓" if elig["can_run_5way"] else "✗"
            print(f"  {setting}: {elig['eligible_classes']}/{elig['total_classes']} "
                  f"eligible ({elig['eligible_fraction']:.0%}) {status}")

    # Save
    out_path = REPO_ROOT / "results" / "eligibility.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print summary table
    print("\n| Dataset | Classes | K=1 | K=3 | K=5 |")
    print("|---------|---------|-----|-----|-----|")
    for ds_name, info in results.items():
        e = info["eligibility"]
        row = (
            f"| {ds_name:24s} | {info['total_classes']:>7d} "
            f"| {e['k=1_q=15']['eligible_classes']:>3d} "
            f"| {e['k=3_q=15']['eligible_classes']:>3d} "
            f"| {e['k=5_q=15']['eligible_classes']:>3d} |"
        )
        print(row)


if __name__ == "__main__":
    main()
