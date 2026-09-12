"""
Ablation study framework.

Systematically varies representation, loss, model, normalisation, and
adaptation settings to produce a comprehensive ablation table.

Usage:
    python ablation.py
    python ablation.py --fast          # fewer episodes for quick sanity check
    python ablation.py --device cpu
"""

import argparse
import copy
import csv
import itertools
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import SyntheticLandmarkDataset
from data.episodes import EpisodicSampler, split_support_query, collate_episode
from losses.supcon import build_loss
from models import build_encoder, build_few_shot_model
from utils.logger import get_logger
from utils.metrics import accuracy, few_shot_accuracy_with_ci
from utils.seed import set_seed
from train import get_dataset, get_device, load_config, train_one_epoch, evaluate_episodes


# ── Ablation axes ────────────────────────────────────────────────────────────

ABLATION_AXES: Dict[str, List[Dict[str, Any]]] = {
    "representation": [
        {"name": "raw", "representation": "raw"},
        {"name": "pairwise", "representation": "pairwise"},
        {"name": "graph", "representation": "graph"},
    ],
    "loss": [
        {"name": "triplet", "loss": {"name": "triplet", "triplet": {"margin": 0.5, "mining": "semi_hard"}}},
        {"name": "supcon", "loss": {"name": "supcon", "supcon": {"temperature": 0.07}}},
    ],
    "model": [
        {"name": "mlp", "model_encoder": "mlp"},
        {"name": "transformer", "model_encoder": "transformer"},
        {"name": "gcn", "model_encoder": "gcn"},
    ],
    "normalisation": [
        {"name": "with_norm", "normalize_translation": True, "normalize_scale": True},
        {"name": "no_scale", "normalize_translation": True, "normalize_scale": False},
        {"name": "no_norm", "normalize_translation": False, "normalize_scale": False},
    ],
    "adaptation": [
        {"name": "zero_shot", "shots": 0},
        {"name": "1_shot", "shots": 1},
        {"name": "5_shot", "shots": 5},
    ],
}


def apply_ablation_setting(cfg: dict, axis: str, setting: Dict[str, Any]) -> dict:
    """Apply an ablation setting to a config copy.

    Args:
        cfg: Base config dictionary.
        axis: Ablation axis name.
        setting: Setting dict from ``ABLATION_AXES``.

    Returns:
        Modified config copy.
    """
    c = copy.deepcopy(cfg)

    if axis == "representation":
        c["representation"] = setting["representation"]
        if setting["representation"] == "gcn":
            c["model"]["encoder"] = "gcn"
        elif c["model"]["encoder"] == "gcn" and setting["representation"] != "graph":
            c["model"]["encoder"] = "transformer"

    elif axis == "loss":
        c["loss"] = {**c.get("loss", {}), **setting["loss"]}

    elif axis == "model":
        c["model"]["encoder"] = setting["model_encoder"]
        # Force compatible representation
        if setting["model_encoder"] == "gcn":
            c["representation"] = "graph"
        elif c.get("representation") == "graph" and setting["model_encoder"] != "gcn":
            c["representation"] = "raw"

    elif axis == "normalisation":
        c["dataset"]["normalize_translation"] = setting["normalize_translation"]
        c["dataset"]["normalize_scale"] = setting["normalize_scale"]

    elif axis == "adaptation":
        if setting["shots"] == 0:
            c["few_shot"]["k_shot"] = 5  # use default for prototypes
        else:
            c["few_shot"]["k_shot"] = setting["shots"]

    return c


def run_single_ablation(
    cfg: dict,
    device: torch.device,
    train_episodes: int = 200,
    eval_episodes: int = 500,
) -> Dict[str, float]:
    """Train and evaluate one ablation configuration.

    Args:
        cfg: Full config dict (already modified for this ablation).
        device: Torch device.
        train_episodes: Number of training episodes.
        eval_episodes: Number of evaluation episodes.

    Returns:
        Dict with accuracy and CI.
    """
    representation = cfg.get("representation", "raw")
    n_way = cfg["few_shot"]["n_way"]
    k_shot = cfg["few_shot"]["k_shot"]
    q_query = cfg["few_shot"]["q_query"]

    # Use real data if available, else fall back to synthetic
    train_ds = get_dataset(cfg, "train")
    test_ds = get_dataset(cfg, "test")

    # Override representation if the dataset is LandmarkDataset
    from data.datasets import LandmarkDataset
    if isinstance(train_ds, LandmarkDataset):
        train_ds.representation = representation
    if isinstance(test_ds, LandmarkDataset):
        test_ds.representation = representation

    train_labels = [train_ds[i][1] for i in range(len(train_ds))]
    test_labels = [test_ds[i][1] for i in range(len(test_ds))]

    train_sampler = EpisodicSampler(
        train_labels, n_way=n_way, k_shot=k_shot, q_query=q_query,
        episodes=train_episodes,
    )
    test_sampler = EpisodicSampler(
        test_labels, n_way=n_way, k_shot=k_shot, q_query=q_query,
        episodes=eval_episodes,
    )

    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=collate_episode, num_workers=0)
    test_loader = DataLoader(test_ds, batch_sampler=test_sampler, collate_fn=collate_episode, num_workers=0)

    # Build model
    encoder = build_encoder(cfg, representation)
    model = build_few_shot_model(cfg, encoder).to(device)
    loss_fn = build_loss(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=1e-4)

    # Quick training
    n_epochs = min(cfg["training"].get("epochs", 10), 10)
    for epoch in range(n_epochs):
        train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            n_way, k_shot, q_query, cfg["training"].get("grad_clip", 1.0),
        )

    # Evaluate
    results = evaluate_episodes(model, test_loader, device, n_way, k_shot, q_query)
    return results


def main():
    parser = argparse.ArgumentParser(description="Ablation study")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--fast", action="store_true", help="Use fewer episodes")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--axes", type=str, nargs="+", default=None,
                        help="Which axes to ablate (default: all)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg)

    os.makedirs(cfg.get("output_dir", "results"), exist_ok=True)
    logger = get_logger("ablation", "results/ablation.log")

    train_eps = 50 if args.fast else 200
    eval_eps = 100 if args.fast else 500

    axes = args.axes or list(ABLATION_AXES.keys())
    all_results: List[Dict[str, str]] = []

    for axis in axes:
        if axis not in ABLATION_AXES:
            logger.warning(f"Unknown axis: {axis}, skipping")
            continue

        settings = ABLATION_AXES[axis]
        logger.info(f"\n{'='*60}")
        logger.info(f"Ablation axis: {axis}")
        logger.info(f"{'='*60}")

        for setting in settings:
            name = setting["name"]
            logger.info(f"  Running: {axis}/{name} ...")

            try:
                mod_cfg = apply_ablation_setting(cfg, axis, setting)
                results = run_single_ablation(mod_cfg, device, train_eps, eval_eps)
                row = {
                    "axis": axis,
                    "setting": name,
                    "accuracy": f"{results['accuracy']:.4f}",
                    "ci": f"{results['ci']:.4f}",
                    "loss": f"{results['loss']:.4f}",
                }
                all_results.append(row)
                logger.info(
                    f"    {name}: acc={results['accuracy']:.4f} ± {results['ci']:.4f}"
                )
            except Exception as e:
                logger.error(f"    {name}: FAILED — {e}")
                all_results.append({
                    "axis": axis,
                    "setting": name,
                    "accuracy": "FAILED",
                    "ci": "N/A",
                    "loss": "N/A",
                })

    # Save ablation table
    csv_path = os.path.join(cfg.get("output_dir", "results"), "ablation.csv")
    fieldnames = ["axis", "setting", "accuracy", "ci", "loss"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    logger.info(f"\nAblation results saved to {csv_path}")

    # Print summary table
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Axis':<18} {'Setting':<15} {'Accuracy':<12} {'CI':<10}")
    logger.info("-" * 60)
    for r in all_results:
        logger.info(f"{r['axis']:<18} {r['setting']:<15} {r['accuracy']:<12} {r['ci']:<10}")


if __name__ == "__main__":
    main()
