#!/usr/bin/env python3
"""
Expanded cross-domain evaluation: all repr × encoder × mode combos.

Evaluates SOURCE→TARGET for every combination of:
  - Encoders:       mlp, transformer
  - Representations: raw, angle, raw_angle
  - Modes:          frozen (random init pretrain), adapted (finetune_last)

The "frozen" mode pretrains on SOURCE (train split) for a few epochs and then
evaluates on target test splits with a frozen encoder.
The "adapted" mode additionally fine-tunes the last encoder layer on the
target's train split before evaluating on the target's test split.

Output:  results/cross_domain_expanded.csv

Usage:
    python tools/run_cross_domain_expanded.py
    python tools/run_cross_domain_expanded.py --source libras_alphabet --targets arabic_sign_alphabet
    python tools/run_cross_domain_expanded.py --encoders mlp --reprs raw angle
    python tools/run_cross_domain_expanded.py --modes frozen   # skip adaptation
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List

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


DEFAULT_SOURCE = "asl_alphabet"
ALL_DATASETS = ["asl_alphabet", "libras_alphabet", "arabic_sign_alphabet", "thai_fingerspelling"]
DATA_ROOTS = {
    "asl_alphabet":         "data/processed/asl_alphabet",
    "libras_alphabet":      "data/processed/libras_alphabet",
    "arabic_sign_alphabet": "data/processed/arabic_sign_alphabet",
    "thai_fingerspelling":  "data/processed/thai_fingerspelling",
}


def resolve_device(device_arg: str | None = None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_cfg(encoder: str, representation: str) -> dict:
    return {
        "representation": representation,
        "distance": "euclidean",
        "model": {
            "encoder": encoder,
            "embedding_dim": 128,
            "mlp":         {"hidden_dims": [256, 256], "dropout": 0.3},
            "transformer": {"num_heads": 4, "num_layers": 2,
                            "dim_feedforward": 256, "dropout": 0.1},
        },
        "few_shot": {"method": "prototypical"},
        "dataset": {"num_landmarks": 21, "landmark_dim": 3},
    }


def pretrain_on_source(
    encoder_name: str,
    representation: str,
    source: str = "asl_alphabet",
    epochs: int = 3,
    seed: int = 42,
    device: torch.device | None = None,
) -> dict:
    """Quick episodic pretraining on source train split. Returns model state_dict."""
    from losses.supcon import build_loss

    set_seed(seed, deterministic=True)
    device = device or resolve_device()

    cfg = build_cfg(encoder_name, representation)
    cfg["loss"] = {"name": "supcon", "supcon": {"temperature": 0.07}}

    train_ds = SplitLandmarkDataset(source, "train", DATA_ROOTS[source], representation)
    labels = [train_ds[i][1] for i in range(len(train_ds))]

    n_way, k_shot, q_query = 5, 5, 15
    sampler = EpisodicSampler(
        labels, n_way=n_way, k_shot=k_shot, q_query=q_query,
        episodes=100, seed=seed, auto_adjust_q=True,
        dataset_name=source, split_name="train",
    )
    loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate_episode)

    encoder = build_encoder(cfg, representation)
    model = build_few_shot_model(cfg, encoder).to(device)
    loss_fn = build_loss(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        for batch in loader:
            data, lbls = batch
            sx, sy, qx, qy = split_support_query(
                (data.to(device), lbls.to(device)), n_way, k_shot, sampler.q_query)
            log_probs = model(sx, sy, qx, n_way)
            cls_loss = F.nll_loss(log_probs, qy)
            all_x = torch.cat([sx, qx])
            all_y = torch.cat([sy, qy])
            emb = model.get_embeddings(all_x)
            m_loss = loss_fn(emb, all_y)
            loss = cls_loss + 0.5 * m_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return model.state_dict()


def adapt_on_target(
    model,
    target_ds_train,
    n_way: int = 5,
    k_shot: int = 5,
    q_query: int = 15,
    adapt_epochs: int = 5,
    seed: int = 42,
    ds_name: str = "",
):
    """Fine-tune last layer on target train split."""
    from losses.supcon import build_loss

    device = torch.device("cpu")

    # Freeze all except last layer
    params = list(model.encoder.parameters())
    for p in params[:-2]:
        p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=1e-4)

    labels = [target_ds_train[i][1] for i in range(len(target_ds_train))]
    try:
        sampler = EpisodicSampler(
            labels, n_way=n_way, k_shot=k_shot, q_query=q_query,
            episodes=50, seed=seed, auto_adjust_q=True,
            dataset_name=ds_name, split_name="train",
        )
    except ValueError:
        return  # Not enough classes

    loader = DataLoader(target_ds_train, batch_sampler=sampler, collate_fn=collate_episode)
    cfg_loss = {"loss": {"name": "supcon", "supcon": {"temperature": 0.07}}}
    loss_fn = build_loss(cfg_loss).to(device)

    model.train()
    for epoch in range(adapt_epochs):
        for batch in loader:
            data, lbls = batch
            sx, sy, qx, qy = split_support_query(
                (data.to(device), lbls.to(device)),
                n_way, k_shot, sampler.q_query)
            log_probs = model(sx, sy, qx, n_way)
            cls_loss = F.nll_loss(log_probs, qy)
            emb = model.get_embeddings(torch.cat([sx, qx]))
            m_loss = loss_fn(emb, torch.cat([sy, qy]))
            loss = cls_loss + 0.5 * m_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

    # Unfreeze for clean state
    for p in model.parameters():
        p.requires_grad = True


@torch.no_grad()
def evaluate_cross(model, target_ds, n_way, k_shot, q_query, episodes, seed, ds_name):
    """Evaluate model on target test split."""
    device = torch.device("cpu")
    model.eval()
    labels = [target_ds[i][1] for i in range(len(target_ds))]
    try:
        sampler = EpisodicSampler(
            labels, n_way=n_way, k_shot=k_shot, q_query=q_query,
            episodes=episodes, seed=seed, auto_adjust_q=True,
            dataset_name=ds_name, split_name="test",
        )
    except ValueError as e:
        print(f"  SKIP {ds_name}: {e}")
        return float("nan"), float("nan")

    loader = DataLoader(target_ds, batch_sampler=sampler, collate_fn=collate_episode)
    accs = []
    for batch in loader:
        data, lbls = batch
        sx, sy, qx, qy = split_support_query(
            (data.to(device), lbls.to(device)),
            n_way, k_shot, sampler.q_query)
        log_probs = model(sx, sy, qx, n_way)
        accs.append(accuracy(log_probs, qy))
    mean_acc, ci = few_shot_accuracy_with_ci(accs)
    return mean_acc, ci


def main():
    parser = argparse.ArgumentParser(description="Expanded cross-domain evaluation")
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE,
                        help="Source dataset for pretraining (default: asl_alphabet)")
    parser.add_argument("--encoders", nargs="+", default=["mlp", "transformer"])
    parser.add_argument("--reprs", nargs="+", default=["raw", "angle", "raw_angle"])
    parser.add_argument("--modes", nargs="+", default=["frozen", "adapted"],
                        choices=["frozen", "adapted"])
    parser.add_argument("--targets", nargs="+", default=ALL_DATASETS)
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--pretrain_epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/cross_domain_expanded.csv")
    args = parser.parse_args()

    n_way, k_shot, q_query = 5, 5, 15
    results = []

    source = args.source
    print(f"\n  Source dataset: {source}")

    for enc in args.encoders:
        for rep in args.reprs:
            print(f"\n{'='*60}")
            print(f"  Pretraining: {enc} / {rep} on {source}")
            print(f"{'='*60}")
            t0 = time.time()
            state_dict = pretrain_on_source(enc, rep, source=source,
                                           epochs=args.pretrain_epochs, seed=args.seed)
            print(f"  Pretrained in {time.time()-t0:.1f}s")

            for mode in args.modes:
                for target in args.targets:
                    print(f"\n  → {source} → {target} | {enc}/{rep} | mode={mode}")

                    cfg = build_cfg(enc, rep)
                    encoder = build_encoder(cfg, rep)
                    model = build_few_shot_model(cfg, encoder)
                    model.load_state_dict(state_dict)

                    if mode == "adapted" and target != source:
                        # Fine-tune last layer on target train
                        target_train = SplitLandmarkDataset(
                            target, "train", DATA_ROOTS[target], rep)
                        adapt_on_target(model, target_train,
                                       n_way=n_way, k_shot=k_shot, q_query=q_query,
                                       adapt_epochs=5, seed=args.seed, ds_name=target)

                    target_test = SplitLandmarkDataset(
                        target, "test", DATA_ROOTS[target], rep)
                    mean_acc, ci = evaluate_cross(
                        model, target_test, n_way, k_shot, q_query,
                        args.episodes, args.seed, target)

                    print(f"    Acc: {mean_acc:.4f} ± {ci:.4f}")

                    results.append({
                        "source": source,
                        "target": target,
                        "encoder": enc,
                        "representation": rep,
                        "mode": mode,
                        "k_shot": k_shot,
                        "n_way": n_way,
                        "q_query": q_query,
                        "episodes": args.episodes,
                        "seed": args.seed,
                        "accuracy_mean": round(mean_acc, 4),
                        "ci95": round(ci, 4),
                    })

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n{'='*60}")
    print(f"  Saved {len(results)} rows to {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
