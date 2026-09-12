#!/usr/bin/env python3
"""
Baseline experiments for paper comparison:
  (A) Linear classifier on raw embeddings (sklearn LogReg on ProtoNet embeddings)
  (B) Robustness check with multiple seeds
  (C) Vanilla ProtoNet (no normalization) on nonorm data

Usage:
    python tools/run_baselines.py --experiment linear_classifier
    python tools/run_baselines.py --experiment robustness --seeds 42 1337 2024
    python tools/run_baselines.py --experiment all
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.datasets import SplitLandmarkDataset
from data.episodes import EpisodicSampler, split_support_query, collate_episode
from models import build_encoder, build_few_shot_model
from utils.metrics import accuracy, few_shot_accuracy_with_ci
from utils.seed import set_seed


# ═════════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ═════════════════════════════════════════════════════════════════════════════

DATASETS = {
    "asl_alphabet":         "data/processed/asl_alphabet",
    "libras_alphabet":      "data/processed/libras_alphabet",
    "arabic_sign_alphabet": "data/processed/arabic_sign_alphabet",
    "thai_fingerspelling":  "data/processed/thai_fingerspelling",
}

def build_cfg(encoder: str, representation: str, distance: str = "euclidean") -> dict:
    return {
        "representation": representation,
        "distance": distance,
        "model": {
            "encoder": encoder,
            "embedding_dim": 128,
            "mlp":         {"hidden_dims": [256, 256], "dropout": 0.3},
            "transformer": {"num_heads": 4, "num_layers": 2,
                            "dim_feedforward": 256, "dropout": 0.1},
            "gcn":         {"hidden_dim": 128, "num_layers": 3, "dropout": 0.2},
        },
        "few_shot": {"method": "prototypical"},
        "dataset": {"num_landmarks": 21, "landmark_dim": 3},
    }


@torch.no_grad()
def run_episodic_eval(
    ds: SplitLandmarkDataset,
    encoder_name: str,
    representation: str,
    n_way: int,
    k_shot: int,
    q_query: int,
    episodes: int,
    seed: int,
    device: torch.device,
    ds_name: str = "unknown",
    auto_adjust_q: bool = False,
) -> Dict[str, float]:
    """Run episodic evaluation for one (encoder, repr, k_shot) setting."""
    set_seed(seed + k_shot * 1000, deterministic=True)
    labels = [ds[i][1] for i in range(len(ds))]
    try:
        sampler = EpisodicSampler(
            labels, n_way=n_way, k_shot=k_shot, q_query=q_query,
            episodes=episodes, seed=seed,
            auto_adjust_q=auto_adjust_q,
            dataset_name=ds_name, split_name="test",
        )
    except ValueError as e:
        print(f"  SKIP {ds_name} k={k_shot}: {e}")
        return {"accuracy": float("nan"), "ci": float("nan"), "actual_q": q_query}

    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_episode)
    cfg = build_cfg(encoder_name, representation)
    encoder = build_encoder(cfg, representation)
    model = build_few_shot_model(cfg, encoder).to(device)
    model.eval()

    accs = []
    for batch in loader:
        data, lbls = batch
        sx, sy, qx, qy = split_support_query(
            (data.to(device), lbls.to(device)),
            n_way, k_shot, sampler.q_query,
        )
        log_probs = model(sx, sy, qx, n_way)
        accs.append(accuracy(log_probs, qy))

    mean_acc, ci = few_shot_accuracy_with_ci(accs)
    return {"accuracy": mean_acc, "ci": ci, "actual_q": sampler.q_query}


# ═════════════════════════════════════════════════════════════════════════════
#  (A) Linear classifier baseline
# ═════════════════════════════════════════════════════════════════════════════

def run_linear_classifier(args):
    """Train a linear classifier on embeddings and compare vs ProtoNet."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    device = torch.device("cpu")
    results = []

    for ds_name, data_root in DATASETS.items():
        for enc_name in ["mlp"]:
            for rep in ["raw", "angle"]:
                print(f"\n=== Linear Classifier: {ds_name} | {enc_name} | {rep} ===")
                set_seed(42, deterministic=True)

                # Load train and test
                train_ds = SplitLandmarkDataset(ds_name, "train", data_root, rep)
                test_ds = SplitLandmarkDataset(ds_name, "test", data_root, rep)

                # Build model (random init — same as ProtoNet baseline)
                cfg = build_cfg(enc_name, rep)
                encoder = build_encoder(cfg, rep)
                model = build_few_shot_model(cfg, encoder).to(device)
                model.eval()

                # Extract embeddings
                def get_embeddings(ds, model):
                    loader = DataLoader(ds, batch_size=256, shuffle=False)
                    embs, lbls = [], []
                    with torch.no_grad():
                        for data, labels in loader:
                            emb = model.get_embeddings(data.to(device))
                            embs.append(emb.cpu().numpy())
                            lbls.append(labels.numpy())
                    return np.concatenate(embs), np.concatenate(lbls)

                train_emb, train_lbl = get_embeddings(train_ds, model)
                test_emb, test_lbl = get_embeddings(test_ds, model)

                # Scale features
                scaler = StandardScaler()
                train_scaled = scaler.fit_transform(train_emb)
                test_scaled = scaler.transform(test_emb)

                # Train logistic regression
                clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
                clf.fit(train_scaled, train_lbl)
                train_acc = clf.score(train_scaled, train_lbl)
                test_acc = clf.score(test_scaled, test_lbl)

                print(f"  Linear: train={train_acc:.4f}, test={test_acc:.4f}")
                results.append({
                    "dataset": ds_name, "encoder": enc_name, "representation": rep,
                    "method": "linear_classifier",
                    "train_acc": round(train_acc, 4),
                    "test_acc": round(test_acc, 4),
                })

    # Save
    csv_path = REPO_ROOT / "results" / "baseline_linear.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "encoder", "representation", "method", "train_acc", "test_acc"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {csv_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  (B) Robustness check (multiple seeds)
# ═════════════════════════════════════════════════════════════════════════════

def run_robustness(args):
    """Run 5-shot eval with 3 seeds and report mean ± std."""
    seeds = args.seeds
    device = torch.device("cpu")
    n_way, k_shot, q_query, episodes = 5, 5, 15, 600

    results = []

    for ds_name, data_root in DATASETS.items():
        # Best representation per dataset: angle (generally best)
        rep = "angle"
        enc = "mlp"

        print(f"\n=== Robustness: {ds_name} | {enc}/{rep} | seeds={seeds} ===")

        seed_accs = []
        for seed in seeds:
            ds = SplitLandmarkDataset(ds_name, "test", data_root, rep)
            r = run_episodic_eval(
                ds, enc, rep, n_way, k_shot, q_query, episodes, seed,
                device, ds_name=ds_name, auto_adjust_q=True,
            )
            print(f"  seed={seed}: {r['accuracy']:.4f} ± {r['ci']:.4f}")
            seed_accs.append(r["accuracy"])

        mean_across = np.mean(seed_accs)
        std_across = np.std(seed_accs, ddof=1) if len(seed_accs) > 1 else 0.0
        print(f"  → Mean across seeds: {mean_across:.4f} ± {std_across:.4f}")

        results.append({
            "dataset": ds_name, "encoder": enc, "representation": rep,
            "seeds": str(seeds), "k_shot": k_shot,
            **{f"acc_seed_{s}": round(a, 4) for s, a in zip(seeds, seed_accs)},
            "mean": round(mean_across, 4),
            "std": round(std_across, 4),
        })

    # Save
    csv_path = REPO_ROOT / "results" / "robustness_seeds.csv"
    fieldnames = ["dataset", "encoder", "representation", "seeds", "k_shot"]
    fieldnames += [f"acc_seed_{s}" for s in seeds]
    fieldnames += ["mean", "std"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {csv_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  (C) Episode-wise linear head baseline
# ═════════════════════════════════════════════════════════════════════════════

def run_episode_linear(args):
    """Per-episode logistic regression on support embeddings, eval on query.

    This baseline replaces the ProtoNet nearest-prototype head with a
    per-episode supervised classifier fitted on the K*N support embeddings.
    """
    from sklearn.linear_model import LogisticRegression

    device = torch.device("cpu")
    n_way, k_shot, q_query, episodes = 5, 5, 15, 600
    seed = 42
    results = []

    for ds_name, data_root in DATASETS.items():
        for rep in ["raw", "angle"]:
            print(f"\n=== Episode-Linear: {ds_name} | mlp/{rep} ===")
            set_seed(seed + k_shot * 1000, deterministic=True)

            ds = SplitLandmarkDataset(ds_name, "test", data_root, rep)
            labels = [ds[i][1] for i in range(len(ds))]

            try:
                sampler = EpisodicSampler(
                    labels, n_way=n_way, k_shot=k_shot, q_query=q_query,
                    episodes=episodes, seed=seed,
                    auto_adjust_q=True,
                    dataset_name=ds_name, split_name="test",
                )
            except ValueError as e:
                print(f"  SKIP: {e}")
                continue

            loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_episode)
            cfg = build_cfg("mlp", rep)
            encoder = build_encoder(cfg, rep)
            model = build_few_shot_model(cfg, encoder).to(device)
            model.eval()

            actual_q = sampler.q_query
            accs = []
            for batch in loader:
                data, lbls = batch
                sx, sy, qx, qy = split_support_query(
                    (data.to(device), lbls.to(device)),
                    n_way, k_shot, actual_q,
                )
                with torch.no_grad():
                    s_emb = model.get_embeddings(sx).cpu().numpy()
                    q_emb = model.get_embeddings(qx).cpu().numpy()
                sy_np = sy.cpu().numpy()
                qy_np = qy.cpu().numpy()

                # Fit per-episode logistic regression
                clf = LogisticRegression(max_iter=200, C=1.0, solver="lbfgs")
                clf.fit(s_emb, sy_np)
                preds = clf.predict(q_emb)
                acc = float((preds == qy_np).mean())
                accs.append(acc)

            mean_acc, ci = few_shot_accuracy_with_ci(accs)
            print(f"  Episode-linear: {mean_acc:.4f} ± {ci:.4f}")
            results.append({
                "dataset": ds_name, "encoder": "mlp", "representation": rep,
                "method": "episode_linear", "k_shot": k_shot,
                "accuracy": round(mean_acc, 4), "ci": round(ci, 4),
            })

    csv_path = REPO_ROOT / "results" / "baseline_episode_linear.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "encoder", "representation", "method", "k_shot",
            "accuracy", "ci"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {csv_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  (D) Input-space nearest-prototype baseline (no encoder)
# ═════════════════════════════════════════════════════════════════════════════

def run_input_space(args):
    """Nearest-prototype directly in the raw feature space (no learned encoder).

    Replaces the encoder with an identity function and computes ProtoNet
    classification in the original input space.
    """
    device = torch.device("cpu")
    n_way, k_shot, q_query, episodes = 5, 5, 15, 600
    seed = 42
    results = []

    for ds_name, data_root in DATASETS.items():
        for rep in ["raw", "angle"]:
            print(f"\n=== Input-Space ProtoNet: {ds_name} | {rep} ===")
            set_seed(seed + k_shot * 1000, deterministic=True)

            ds = SplitLandmarkDataset(ds_name, "test", data_root, rep)
            labels = [ds[i][1] for i in range(len(ds))]

            try:
                sampler = EpisodicSampler(
                    labels, n_way=n_way, k_shot=k_shot, q_query=q_query,
                    episodes=episodes, seed=seed,
                    auto_adjust_q=True,
                    dataset_name=ds_name, split_name="test",
                )
            except ValueError as e:
                print(f"  SKIP: {e}")
                continue

            loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_episode)
            actual_q = sampler.q_query
            accs = []

            for batch in loader:
                data, lbls = batch
                sx, sy, qx, qy = split_support_query(
                    (data.to(device), lbls.to(device)),
                    n_way, k_shot, actual_q,
                )
                # Flatten inputs if needed
                sx_flat = sx.reshape(sx.size(0), -1)
                qx_flat = qx.reshape(qx.size(0), -1)

                # Compute prototypes in input space
                prototypes = torch.zeros(n_way, sx_flat.size(-1), device=device)
                for c in range(n_way):
                    mask = sy == c
                    prototypes[c] = sx_flat[mask].mean(dim=0)

                # Nearest prototype
                dists = torch.cdist(qx_flat.float(), prototypes.float(), p=2)
                preds = dists.argmin(dim=1)
                acc = float((preds == qy).float().mean().item())
                accs.append(acc)

            mean_acc, ci = few_shot_accuracy_with_ci(accs)
            print(f"  Input-space: {mean_acc:.4f} ± {ci:.4f}")
            results.append({
                "dataset": ds_name, "representation": rep,
                "method": "input_space_proto", "k_shot": k_shot,
                "accuracy": round(mean_acc, 4), "ci": round(ci, 4),
            })

    csv_path = REPO_ROOT / "results" / "baseline_input_space.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "representation", "method", "k_shot",
            "accuracy", "ci"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {csv_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Run baseline experiments")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["linear_classifier", "robustness",
                                 "episode_linear", "input_space", "all"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024])
    args = parser.parse_args()

    if args.experiment in ("linear_classifier", "all"):
        run_linear_classifier(args)
    if args.experiment in ("robustness", "all"):
        run_robustness(args)
    if args.experiment in ("episode_linear", "all"):
        run_episode_linear(args)
    if args.experiment in ("input_space", "all"):
        run_input_space(args)


if __name__ == "__main__":
    main()
