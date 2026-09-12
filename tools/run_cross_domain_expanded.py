"""
Run cross-domain (cross-lingual) evaluation matrix.

Automates all source→target transfer pairs across representations
and adaptation modes. Reproduces Tables 4 and 6 from the paper.

Usage:
    python tools/run_cross_domain_expanded.py \
        [--sources asl libras arabic thai] \
        [--targets asl libras arabic thai] \
        [--reprs angle raw raw_angle] \
        [--modes frozen target_supervised]
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path


DATASETS = ["asl", "libras", "arabic", "thai"]
REPRESENTATIONS = ["raw", "angle", "raw_angle"]
MODES = ["frozen", "target_supervised"]


def run_transfer(
    source: str,
    target: str,
    representation: str,
    encoder: str,
    mode: str,
    seed: int = 42,
    config: str = "configs/default.yaml",
) -> dict:
    """Run a single cross-lingual transfer experiment."""
    print(f"\n  {source} → {target} ({representation}, {encoder}, {mode})")

    cmd = [
        sys.executable, "adapt.py",
        "--config", config,
        "--source", source,
        "--target", target,
        "--repr", representation,
        "--encoder", encoder,
        "--mode", mode,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"    FAILED: {result.stderr[:300]}")
        return None

    # Parse output for accuracy
    for line in result.stdout.split("\n"):
        if "Accuracy:" in line:
            print(f"    {line.strip()}")

    # Load saved results
    result_file = (
        f"results/cross_lingual/{source}_to_{target}_{encoder}_{representation}_{mode}.json"
    )
    if os.path.exists(result_file):
        with open(result_file) as f:
            return json.load(f)

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Run cross-domain evaluation matrix (Tables 4, 6)."
    )
    parser.add_argument(
        "--sources", nargs="+", default=DATASETS,
        help="Source languages.",
    )
    parser.add_argument(
        "--targets", nargs="+", default=DATASETS,
        help="Target languages.",
    )
    parser.add_argument(
        "--reprs", nargs="+", default=REPRESENTATIONS,
        help="Representations.",
    )
    parser.add_argument(
        "--encoder", type=str, default="mlp",
        help="Encoder type.",
    )
    parser.add_argument(
        "--modes", nargs="+", default=MODES,
        help="Adaptation modes.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
    )

    args = parser.parse_args()

    all_results = {}
    total_pairs = len(args.sources) * len(args.targets)
    completed = 0

    for source, target in itertools.product(args.sources, args.targets):
        if source == target:
            continue  # Skip within-domain (handled by run_full_matrix.py)

        print(f"\n{'='*50}")
        print(f"  Transfer: {source} → {target}")
        print(f"{'='*50}")

        for repr_name, mode in itertools.product(args.reprs, args.modes):
            result = run_transfer(
                source=source,
                target=target,
                representation=repr_name,
                encoder=args.encoder,
                mode=mode,
                seed=args.seed,
                config=args.config,
            )

            key = f"{source}->{target}/{repr_name}/{mode}"
            all_results[key] = result

        completed += 1
        print(f"\nProgress: {completed}/{total_pairs - len(args.sources)}")

    # Save combined results
    os.makedirs("results/cross_lingual", exist_ok=True)
    with open("results/cross_lingual/full_cross_domain.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("Cross-domain evaluation complete.")
    print("Results: results/cross_lingual/full_cross_domain.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
