"""
Training script for episodic metric learning on sign language landmarks.

Usage:
    python train.py --config configs/base.yaml --dataset ASL
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import LandmarkDataset, SplitLandmarkDataset, SyntheticLandmarkDataset
from data.episodes import EpisodicSampler, split_support_query, collate_episode
from losses.supcon import build_loss
from models import build_encoder, build_few_shot_model
from utils.logger import get_logger
from utils.metrics import accuracy, few_shot_accuracy_with_ci
from utils.seed import set_seed


def load_config(path: str) -> dict:
    """Load a YAML configuration file.

    Args:
        path: Path to YAML config.

    Returns:
        Configuration dictionary.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_device(cfg: dict) -> torch.device:
    """Resolve device from config.

    Args:
        cfg: Config dict with ``device`` key.

    Returns:
        ``torch.device``
    """
    dev = cfg.get("device", "auto")
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def get_dataset(cfg: dict, split: str = "train", use_json_splits: bool = False):
    """Build dataset from config. Falls back to synthetic if real data unavailable.

    Respects the ``DATA_ROOT`` environment variable: if set, the dataset root
    path is resolved relative to ``$DATA_ROOT`` instead of the working dir.

    When *use_json_splits* is ``True`` the samples are controlled by the JSON
    split file in ``splits/<name>_<split>.json`` (produced by
    ``tools/make_splits.py``).  The data root is then the **flat** preprocessed
    directory (without ``/train`` or ``/test`` suffix).

    Args:
        cfg: Config dict.
        split: ``'train'`` or ``'test'``.
        use_json_splits: Use JSON-based split files instead of directory splits.

    Returns:
        Dataset instance.
    """
    ds_cfg = cfg["dataset"]
    representation = cfg.get("representation", "raw")
    dataset_name = ds_cfg.get("name", "unknown").lower()

    # ── JSON-based splits (new protocol) ─────────────────────────────────
    if use_json_splits:
        raw_root = ds_cfg["root"]
        if "DATA_ROOT" in os.environ and not Path(raw_root).is_absolute():
            flat_root = Path(os.environ["DATA_ROOT"]) / raw_root
        else:
            flat_root = Path(raw_root)
        # If root points at a *_split dir, strip the suffix to get the flat dir
        if flat_root.name.endswith("_split"):
            flat_root = flat_root.parent / flat_root.name[: -len("_split")]
        if flat_root.exists():
            return SplitLandmarkDataset(
                dataset_name=dataset_name,
                split=split,
                data_root=str(flat_root),
                representation=representation,
            )
        # fall through to synthetic

    # ── Directory-based splits (legacy) ──────────────────────────────────
    raw_root = ds_cfg["root"]
    if "DATA_ROOT" in os.environ and not Path(raw_root).is_absolute():
        root = Path(os.environ["DATA_ROOT"]) / raw_root / split
    else:
        root = Path(raw_root) / split

    if root.exists() and any(root.iterdir()):
        return LandmarkDataset(
            root=str(root),
            representation=representation,
        )
    else:
        # Fallback: synthetic dataset for demo / testing
        n_cls = ds_cfg.get("num_classes", 100) if split == "train" else min(ds_cfg.get("num_classes", 60), 60)
        return SyntheticLandmarkDataset(
            num_classes=n_cls,
            samples_per_class=30 if split == "train" else 20,
            representation=representation,
        )


def train_one_epoch(
    model,
    dataloader,
    loss_fn,
    optimizer,
    device,
    n_way: int,
    k_shot: int,
    q_query: int,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Run one training epoch over episodic batches.

    Args:
        model: Few-shot model.
        dataloader: Episodic DataLoader.
        loss_fn: Metric learning loss (used for embedding-level loss).
        optimizer: Optimiser.
        device: Torch device.
        n_way: N-way.
        k_shot: K-shot.
        q_query: Q-query.
        grad_clip: Gradient clipping norm.

    Returns:
        Dict with ``'loss'`` and ``'accuracy'`` averages.
    """
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_episodes = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        data, labels = batch
        data, labels = data.to(device), labels.to(device)
        support_x, support_y, query_x, query_y = split_support_query(
            (data, labels), n_way, k_shot, q_query,
        )
        support_x = support_x.to(device)
        support_y = support_y.to(device)
        query_x = query_x.to(device)
        query_y = query_y.to(device)

        # Episodic forward
        log_probs = model(support_x, support_y, query_x, n_way)
        cls_loss = F.nll_loss(log_probs, query_y)

        # Metric learning loss on support + query embeddings
        all_x = torch.cat([support_x, query_x], dim=0)
        all_y = torch.cat([support_y, query_y], dim=0)
        all_emb = model.get_embeddings(all_x)
        metric_loss = loss_fn(all_emb, all_y)

        loss = cls_loss + 0.5 * metric_loss

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        acc = accuracy(log_probs, query_y)
        total_loss += loss.item()
        total_acc += acc
        num_episodes += 1

    return {
        "loss": total_loss / max(num_episodes, 1),
        "accuracy": total_acc / max(num_episodes, 1),
    }


@torch.no_grad()
def evaluate_episodes(
    model,
    dataloader,
    device,
    n_way: int,
    k_shot: int,
    q_query: int,
) -> Dict[str, float]:
    """Evaluate over episodic batches and compute mean accuracy with CI.

    Args:
        model: Few-shot model.
        dataloader: Episodic DataLoader.
        device: Torch device.
        n_way: N-way.
        k_shot: K-shot.
        q_query: Q-query.

    Returns:
        Dict with ``'accuracy'``, ``'ci'``, and ``'loss'``.
    """
    model.eval()
    accs = []
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        data, labels = batch
        data, labels = data.to(device), labels.to(device)
        support_x, support_y, query_x, query_y = split_support_query(
            (data, labels), n_way, k_shot, q_query,
        )
        support_x = support_x.to(device)
        support_y = support_y.to(device)
        query_x = query_x.to(device)
        query_y = query_y.to(device)

        log_probs = model(support_x, support_y, query_x, n_way)
        loss = F.nll_loss(log_probs, query_y)
        acc = accuracy(log_probs, query_y)

        accs.append(acc)
        total_loss += loss.item()

    mean_acc, ci = few_shot_accuracy_with_ci(accs)
    return {
        "accuracy": mean_acc,
        "ci": ci,
        "loss": total_loss / max(len(accs), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train sign language metric learning")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--json_splits", action="store_true",
                        help="Use JSON-based splits from splits/ directory")
    parser.add_argument("--save", type=str, default=None,
                        help="Override checkpoint save path")
    parser.add_argument("--auto_adjust_q", action="store_true",
                        help="Auto-lower q_query when classes have too few samples")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset:
        cfg["dataset"]["name"] = args.dataset
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.device:
        cfg["device"] = args.device

    # Seed & device
    set_seed(cfg.get("seed", 42), cfg.get("deterministic", True))
    device = get_device(cfg)

    # Directories
    os.makedirs(cfg.get("output_dir", "results"), exist_ok=True)
    os.makedirs(cfg.get("checkpoint_dir", "results/checkpoints"), exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    logger = get_logger("train", cfg.get("log_file", "results/train.log"))
    logger.info(f"Config: {cfg}")
    logger.info(f"Device: {device}")

    # Data
    representation = cfg.get("representation", "raw")
    use_json = getattr(args, "json_splits", False)
    train_ds = get_dataset(cfg, split="train", use_json_splits=use_json)
    test_ds = get_dataset(cfg, split="test", use_json_splits=use_json)

    fs_cfg = cfg["few_shot"]
    n_way = fs_cfg["n_way"]
    k_shot = fs_cfg["k_shot"]
    q_query = fs_cfg["q_query"]

    train_labels = [s[1] if isinstance(s, tuple) else s for s in
                    [(train_ds[i][1]) for i in range(len(train_ds))]]
    test_labels = [test_ds[i][1] for i in range(len(test_ds))]

    seed = cfg.get("seed", 42)
    ds_name = cfg["dataset"].get("name", "unknown").lower()
    auto_q = getattr(args, "auto_adjust_q", False)
    train_sampler = EpisodicSampler(
        train_labels, n_way=n_way, k_shot=k_shot, q_query=q_query,
        episodes=fs_cfg.get("episodes_train", 1000),
        seed=seed, auto_adjust_q=auto_q,
        dataset_name=ds_name, split_name="train",
    )
    test_sampler = EpisodicSampler(
        test_labels, n_way=n_way, k_shot=k_shot, q_query=q_query,
        episodes=fs_cfg.get("episodes_eval", 1000),
        seed=seed + 10000, auto_adjust_q=auto_q,
        dataset_name=ds_name, split_name="test",
    )

    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=collate_episode,
                              num_workers=cfg["training"].get("num_workers", 0))
    test_loader = DataLoader(test_ds, batch_sampler=test_sampler, collate_fn=collate_episode,
                             num_workers=cfg["training"].get("num_workers", 0))

    # Model
    encoder = build_encoder(cfg, representation)
    model = build_few_shot_model(cfg, encoder).to(device)
    logger.info(f"Model: {cfg['few_shot']['method']} with {cfg['model']['encoder']} encoder")
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # Loss
    loss_fn = build_loss(cfg).to(device)

    # Optimiser
    t_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=t_cfg["lr"], weight_decay=t_cfg["weight_decay"],
    )

    # Scheduler
    scheduler = None
    if t_cfg.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_cfg["epochs"],
        )
    elif t_cfg.get("scheduler") == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Training loop
    best_acc = 0.0
    patience_counter = 0
    patience = t_cfg.get("patience", 15)

    for epoch in range(1, t_cfg["epochs"] + 1):
        logger.info(f"Epoch {epoch}/{t_cfg['epochs']}")

        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            n_way, k_shot, q_query, t_cfg.get("grad_clip", 1.0),
        )
        logger.info(f"  Train loss: {train_metrics['loss']:.4f}  acc: {train_metrics['accuracy']:.4f}")

        eval_metrics = evaluate_episodes(model, test_loader, device, n_way, k_shot, q_query)
        logger.info(
            f"  Eval  acc: {eval_metrics['accuracy']:.4f} ± {eval_metrics['ci']:.4f}  "
            f"loss: {eval_metrics['loss']:.4f}"
        )

        if scheduler:
            scheduler.step()

        # Checkpoint best
        if eval_metrics["accuracy"] > best_acc:
            best_acc = eval_metrics["accuracy"]
            patience_counter = 0
            ckpt_path = (
                args.save
                if getattr(args, "save", None)
                else os.path.join(
                    cfg.get("checkpoint_dir", "results/checkpoints"),
                    f"best_{cfg['dataset']['name'].lower()}.pt",
                )
            )
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_accuracy": best_acc,
                "config": cfg,
            }, ckpt_path)
            logger.info(f"  ★ New best model saved ({best_acc:.4f}) → {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    logger.info(f"Training complete. Best accuracy: {best_acc:.4f}")
    return best_acc


if __name__ == "__main__":
    main()
