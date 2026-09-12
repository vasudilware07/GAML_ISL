"""
Episodic training loop for Prototypical Networks.

Trains a Prototypical Network encoder using a combined loss:
    L_total = L_episode + λ * L_supcon

where L_episode is the negative log-likelihood on ProtoNet log-probabilities
and L_supcon is the supervised contrastive loss (λ=0.5, τ=0.07).

Optimisation: AdamW (lr=1e-4, weight_decay=1e-4), gradient clipping at 1.0,
cosine annealing schedule, early stopping with patience 15.

Usage:
    python train.py --config configs/default.yaml --dataset asl --repr angle --encoder mlp

References:
    Section 3.5 — Implementation Details (Training and losses)
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from data.dataset import SplitLandmarkDataset
from data.episodes import EpisodicSampler, split_support_query
from data.representation import get_input_dim
from losses.supcon import SupConLoss
from models import create_model, count_parameters


def set_seed(seed: int):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str, overrides: dict = None) -> dict:
    """Load YAML config with optional overrides."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if overrides:
        config.update(overrides)

    return config


def train_epoch(
    model: nn.Module,
    dataset: SplitLandmarkDataset,
    optimizer: torch.optim.Optimizer,
    scheduler,
    supcon_loss_fn: SupConLoss,
    config: dict,
    device: torch.device,
    epoch: int,
) -> dict:
    """
    Run one training epoch (multiple episodes).

    Returns:
        Dictionary with epoch metrics: loss, accuracy, episode_loss, supcon_loss.
    """
    model.train()

    n_way = config["n_way"]
    k_shot = config["k_shot"]
    q_query = config["q_query"]
    episodes_per_epoch = config["episodes_per_epoch"]
    supcon_weight = config["supcon_weight"]
    grad_clip = config["grad_clip"]
    seed = config["seed"]

    total_loss = 0.0
    total_episode_loss = 0.0
    total_supcon_loss = 0.0
    total_correct = 0
    total_queries = 0

    # Create episodic sampler for this epoch
    epoch_seed = seed + epoch * episodes_per_epoch

    for ep_idx in range(episodes_per_epoch):
        # Create episode
        from data.episodes import create_episode_batch

        support_features, support_labels, query_features, query_labels = (
            create_episode_batch(
                dataset, n_way, k_shot, q_query,
                seed=epoch_seed, episode_idx=ep_idx, device=device,
            )
        )

        # Forward pass through ProtoNet
        log_probs, predictions, distances = model(
            support_features, support_labels, query_features, n_way
        )

        # Episode classification loss (NLL)
        episode_loss = F.nll_loss(log_probs, query_labels)

        # Supervised contrastive loss on support+query embeddings
        all_features = torch.cat([support_features, query_features], dim=0)
        all_labels = torch.cat([support_labels, query_labels], dim=0)
        all_embeddings = model.encode(all_features)
        supcon_loss = supcon_loss_fn(all_embeddings, all_labels)

        # Combined loss
        loss = episode_loss + supcon_weight * supcon_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_episode_loss += episode_loss.item()
        total_supcon_loss += supcon_loss.item()
        total_correct += (predictions == query_labels).sum().item()
        total_queries += query_labels.size(0)

    # Update learning rate
    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / episodes_per_epoch
    avg_episode_loss = total_episode_loss / episodes_per_epoch
    avg_supcon_loss = total_supcon_loss / episodes_per_epoch
    accuracy = total_correct / total_queries * 100

    return {
        "loss": avg_loss,
        "episode_loss": avg_episode_loss,
        "supcon_loss": avg_supcon_loss,
        "accuracy": accuracy,
    }


def validate(
    model: nn.Module,
    dataset: SplitLandmarkDataset,
    config: dict,
    device: torch.device,
    num_episodes: int = 100,
    seed: int = 99999,
) -> dict:
    """
    Validate model on a set of episodes.

    Returns:
        Dictionary with validation accuracy and 95% CI.
    """
    model.eval()

    n_way = config["n_way"]
    k_shot = config["k_shot"]
    q_query = config["q_query"]

    accuracies = []

    with torch.no_grad():
        for ep_idx in range(num_episodes):
            from data.episodes import create_episode_batch

            support_features, support_labels, query_features, query_labels = (
                create_episode_batch(
                    dataset, n_way, k_shot, q_query,
                    seed=seed, episode_idx=ep_idx, device=device,
                )
            )

            log_probs, predictions, _ = model(
                support_features, support_labels, query_features, n_way
            )

            acc = (predictions == query_labels).float().mean().item() * 100
            accuracies.append(acc)

    accuracies = np.array(accuracies)
    mean_acc = accuracies.mean()
    ci_95 = 1.96 * accuracies.std() / np.sqrt(len(accuracies))

    return {
        "accuracy": mean_acc,
        "ci_95": ci_95,
        "accuracies": accuracies,
    }


def train(config: dict):
    """
    Full training pipeline.

    1. Set up data, model, optimizer
    2. Train episodically with early stopping
    3. Save best checkpoint
    """
    # Setup
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    dataset_name = config["dataset"]
    representation = config["representation"]
    splits_dir = config["splits_dir"]
    data_dir = config["data_dir"]

    train_split = os.path.join(splits_dir, f"{dataset_name}_train.json")
    test_split = os.path.join(splits_dir, f"{dataset_name}_test.json")

    # Validate no overlap
    if os.path.exists(train_split) and os.path.exists(test_split):
        SplitLandmarkDataset.validate_no_overlap(train_split, test_split)

    train_dataset = SplitLandmarkDataset(
        data_dir=data_dir,
        split_file=train_split,
        representation=representation,
        normalize=config.get("normalize", True),
    )

    test_dataset = SplitLandmarkDataset(
        data_dir=data_dir,
        split_file=test_split,
        representation=representation,
        normalize=config.get("normalize", True),
    )

    print(f"Train dataset: {train_dataset}")
    print(f"Test dataset:  {test_dataset}")

    # Model
    input_dim = get_input_dim(representation)
    encoder_name = config["encoder"]

    model = create_model(
        encoder_name=encoder_name,
        input_dim=input_dim,
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config.get("dropout", 0.3),
        n_heads=config.get("transformer_heads", 4),
        n_layers=config.get("transformer_layers", 2),
        transformer_dropout=config.get("transformer_dropout", 0.1),
        representation=representation,
        distance=config.get("distance", "euclidean"),
    )
    model = model.to(device)

    num_params = count_parameters(model)
    print(f"Model: {encoder_name} / {representation} — {num_params:,} parameters")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["max_epochs"],
    )

    # SupCon loss
    supcon_loss_fn = SupConLoss(temperature=config["supcon_temperature"])

    # Checkpoint directory
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = f"{dataset_name}_{encoder_name}_{representation}"
    best_checkpoint_path = checkpoint_dir / f"{checkpoint_name}_best.pt"

    # Training loop with early stopping
    best_val_acc = 0.0
    patience_counter = 0
    patience = config["patience"]

    print(f"\nTraining for up to {config['max_epochs']} epochs "
          f"(patience={patience})...\n")

    for epoch in range(1, config["max_epochs"] + 1):
        start_time = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_dataset, optimizer, scheduler,
            supcon_loss_fn, config, device, epoch,
        )

        # Validate
        val_metrics = validate(
            model, test_dataset, config, device,
            num_episodes=100, seed=99999,
        )

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_metrics['loss']:.4f} "
            f"(ep: {train_metrics['episode_loss']:.4f}, "
            f"sc: {train_metrics['supcon_loss']:.4f}) | "
            f"Train Acc: {train_metrics['accuracy']:.1f}% | "
            f"Val Acc: {val_metrics['accuracy']:.1f}±{val_metrics['ci_95']:.1f}% | "
            f"{elapsed:.1f}s"
        )

        # Early stopping check
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            patience_counter = 0

            # Save best checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": best_val_acc,
                "config": config,
            }, best_checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    print(f"\nBest validation accuracy: {best_val_acc:.1f}%")
    print(f"Checkpoint saved: {best_checkpoint_path}")

    return model, best_val_acc


def main():
    parser = argparse.ArgumentParser(
        description="Train Prototypical Network for few-shot SLR."
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument("--dataset", type=str, help="Dataset name override.")
    parser.add_argument("--repr", type=str, help="Representation override.")
    parser.add_argument("--encoder", type=str, help="Encoder override.")
    parser.add_argument("--k_shot", type=int, help="K-shot override.")
    parser.add_argument("--max_epochs", type=int, help="Max epochs override.")
    parser.add_argument("--seed", type=int, help="Seed override.")
    parser.add_argument("--data_dir", type=str, help="Data directory override.")
    parser.add_argument("--normalize", type=bool, default=None,
                        help="Normalization override (for ablation).")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply overrides
    if args.dataset:
        config["dataset"] = args.dataset
        # Also load dataset-specific config if available
        ds_config_path = f"configs/{args.dataset}.yaml"
        if os.path.exists(ds_config_path):
            with open(ds_config_path, "r") as f:
                ds_config = yaml.safe_load(f)
            config.update(ds_config)
    if args.repr:
        config["representation"] = args.repr
    if args.encoder:
        config["encoder"] = args.encoder
    if args.k_shot is not None:
        config["k_shot"] = args.k_shot
    if args.max_epochs is not None:
        config["max_epochs"] = args.max_epochs
    if args.seed is not None:
        config["seed"] = args.seed
    if args.data_dir:
        config["data_dir"] = args.data_dir
    if args.normalize is not None:
        config["normalize"] = args.normalize

    train(config)


if __name__ == "__main__":
    main()
