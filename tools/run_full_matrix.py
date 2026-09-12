"""
Run the full experimental matrix.

Automates within-domain training and evaluation across all combinations of:
    - Datasets: ASL, LIBRAS, Arabic, Thai
    - Representations: raw, angle, raw_angle
    - Encoders: mlp, transformer
    - K-shots: 1, 3, 5

This reproduces Table 3 from the paper (within-domain few-shot accuracy).

Usage:
    python tools/run_full_matrix.py [--datasets asl libras] [--reprs angle raw_angle]
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
ENCODERS = ["mlp", "transformer"]
K_SHOTS = [1, 3, 5]


def run_experiment(
    dataset: str,
    representation: str,
    encoder: str,
    k_shot: int,
    seed: int = 42,
    config: str = "configs/default.yaml",
    skip_train: bool = False,
) -> dict:
    """
    Run a single training + evaluation experiment.

    Returns:
        Dictionary with results or None if failed.
    """
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset} | Repr: {representation} | "
          f"Encoder: {encoder} | K={k_shot}")
    print(f"{'='*60}")

    # Train
    if not skip_train:
        train_cmd = [
            sys.executable, "train.py",
            "--config", config,
            "--dataset", dataset,
            "--repr", representation,
            "--encoder", encoder,
            "--seed", str(seed),
        ]
        print(f"Training: {' '.join(train_cmd)}")
        result = subprocess.run(train_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  FAILED: {result.stderr[:500]}")
            return None

    # Evaluate
    checkpoint = f"checkpoints/{dataset}_{encoder}_{representation}_best.pt"
    eval_cmd = [
        sys.executable, "evaluate.py",
        "--config", config,
        "--dataset", dataset,
        "--repr", representation,
        "--encoder", encoder,
        "--k_shot", str(k_shot),
        "--checkpoint", checkpoint,
        "--seed", str(seed),
        "--no_tsne", "--no_confusion",
    ]
    print(f"Evaluating: {' '.join(eval_cmd)}")
    result = subprocess.run(eval_cmd, capture_output=True, text=True)
    print(result.stdout[-500:] if result.stdout else "")

    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[:500]}")
        return None

    # Load results
    result_file = f"results/{dataset}_{encoder}_{representation}_{k_shot}shot.json"
    if os.path.exists(result_file):
        with open(result_file) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Run the full experimental matrix (Table 3)."
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DATASETS,
        help="Datasets to evaluate.",
    )
    parser.add_argument(
        "--reprs", nargs="+", default=REPRESENTATIONS,
        help="Representations to evaluate.",
    )
    parser.add_argument(
        "--encoders", nargs="+", default=ENCODERS,
        help="Encoders to evaluate.",
    )
    parser.add_argument(
        "--k_shots", nargs="+", type=int, default=K_SHOTS,
        help="K-shot values.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--skip_train", action="store_true",
        help="Skip training, only evaluate.",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Base configuration file.",
    )

    args = parser.parse_args()

    all_results = {}
    total = len(args.datasets) * len(args.reprs) * len(args.encoders)
    completed = 0

    for dataset, encoder in itertools.product(args.datasets, args.encoders):
        for repr_name in args.reprs:
            # Train once (with default k_shot)
            result = run_experiment(
                dataset=dataset,
                representation=repr_name,
                encoder=encoder,
                k_shot=5,  # Default K for training
                seed=args.seed,
                config=args.config,
                skip_train=args.skip_train,
            )
            completed += 1

            # Evaluate at different K-shots
            for k in args.k_shots:
                result = run_experiment(
                    dataset=dataset,
                    representation=repr_name,
                    encoder=encoder,
                    k_shot=k,
                    seed=args.seed,
                    config=args.config,
                    skip_train=True,  # Already trained
                )

                key = f"{dataset}/{encoder}/{repr_name}/{k}shot"
                all_results[key] = result

            print(f"\nProgress: {completed}/{total}")

    # Save combined results
    os.makedirs("results", exist_ok=True)
    with open("results/full_matrix.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Full matrix complete. Results saved to results/full_matrix.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
