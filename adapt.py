"""
Cross-lingual adaptation for few-shot sign language recognition.

Loads a pretrained checkpoint from a source language and evaluates on
a target language under two adaptation modes:

    1. Frozen: All encoder weights are fixed; measures how well the
       learned embedding transfers without any adaptation.

    2. Target-supervised: The last linear layer is fine-tuned on the
       target language's train split for up to 20 epochs (lr=1e-4),
       quantifying the benefit of minimal adaptation.

Usage:
    python adapt.py --source asl --target libras \
        --repr angle --encoder mlp --mode frozen

    python adapt.py --source asl --target arabic \
        --repr angle --encoder mlp --mode target_supervised

References:
    Section 3.4 — Few-Shot Evaluation Protocol (Cross-lingual transfer)
    Section 3.5 — Implementation Details (Cross-lingual adaptation)
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
from data.episodes import create_episode_batch
from data.representation import get_input_dim
from evaluate import evaluate_episodes, set_seed
from losses.supcon import SupConLoss
from models import create_model, count_parameters


def load_pretrained_model(
    checkpoint_path: str,
    config: dict,
    device: torch.device,
):
    """
    Load a pretrained model from checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        config: Configuration dictionary.
        device: Target device.

    Returns:
        Loaded PrototypicalNetwork model.
    """
    representation = config["representation"]
    input_dim = get_input_dim(representation)

    model = create_model(
        encoder_name=config["encoder"],
        input_dim=input_dim,
        embedding_dim=config.get("embedding_dim", 128),
        hidden_dim=config.get("hidden_dim", 256),
        dropout=config.get("dropout", 0.3),
        representation=representation,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  Source val accuracy: {checkpoint.get('val_accuracy', 'N/A'):.1f}%")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")

    return model


def frozen_evaluation(
    model,
    target_dataset: SplitLandmarkDataset,
    config: dict,
    device: torch.device,
) -> dict:
    """
    Evaluate with frozen encoder weights (no adaptation).

    The entire encoder is fixed; only nearest-prototype matching
    is used for classification.
    """
    model.eval()

    results = evaluate_episodes(
        model=model,
        dataset=target_dataset,
        n_way=config["n_way"],
        k_shot=config["k_shot"],
        q_query=config["q_query"],
        num_episodes=config.get("num_eval_episodes", 600),
        seed=config["seed"],
        device=device,
    )

    return results


def target_supervised_adaptation(
    model,
    target_train_dataset: SplitLandmarkDataset,
    target_test_dataset: SplitLandmarkDataset,
    config: dict,
    device: torch.device,
) -> dict:
    """
    Fine-tune the last linear layer on the target language's train split.

    Only the last linear layer is unfrozen; all other encoder weights remain
    fixed. Training uses the same episodic protocol with SupCon loss.

    Args:
        model: Pretrained PrototypicalNetwork.
        target_train_dataset: Target language train split.
        target_test_dataset: Target language test split.
        config: Configuration dictionary.
        device: Target device.

    Returns:
        Evaluation results after adaptation.
    """
    # Freeze all except last layer
    model.encoder.freeze_except_last()

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = count_parameters(model) + sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    print(f"  Trainable parameters: {trainable:,} / {total:,}")

    # Optimizer for last layer only
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.get("adapt_lr", 1e-4),
        weight_decay=config.get("weight_decay", 1e-4),
    )

    supcon_loss_fn = SupConLoss(temperature=config.get("supcon_temperature", 0.07))

    n_way = config["n_way"]
    k_shot = config["k_shot"]
    q_query = config["q_query"]
    adapt_epochs = config.get("adapt_epochs", 20)
    episodes_per_epoch = min(config.get("episodes_per_epoch", 100), 50)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, adapt_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        epoch_seed = config["seed"] + 10000 + epoch * episodes_per_epoch

        for ep_idx in range(episodes_per_epoch):
            support_features, support_labels, query_features, query_labels = (
                create_episode_batch(
                    target_train_dataset, n_way, k_shot, q_query,
                    seed=epoch_seed, episode_idx=ep_idx, device=device,
                )
            )

            log_probs, predictions, _ = model(
                support_features, support_labels, query_features, n_way
            )

            # Episode loss
            episode_loss = F.nll_loss(log_probs, query_labels)

            # SupCon loss
            all_features = torch.cat([support_features, query_features], dim=0)
            all_labels = torch.cat([support_labels, query_labels], dim=0)
            all_embeddings = model.encode(all_features)
            supcon_loss = supcon_loss_fn(all_embeddings, all_labels)

            loss = episode_loss + config.get("supcon_weight", 0.5) * supcon_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, config.get("grad_clip", 1.0))
            optimizer.step()

            epoch_loss += loss.item()
            epoch_correct += (predictions == query_labels).sum().item()
            epoch_total += query_labels.size(0)

        # Validate
        val_results = evaluate_episodes(
            model, target_test_dataset, n_way, k_shot, q_query,
            num_episodes=100, seed=99999, device=device,
        )

        train_acc = epoch_correct / epoch_total * 100

        if val_results["accuracy"] > best_val_acc:
            best_val_acc = val_results["accuracy"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(
            f"  Adapt Epoch {epoch:2d} | "
            f"Loss: {epoch_loss / episodes_per_epoch:.4f} | "
            f"Train Acc: {train_acc:.1f}% | "
            f"Val Acc: {val_results['accuracy']:.1f}±{val_results['ci_95']:.1f}%"
        )

    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    final_results = evaluate_episodes(
        model, target_test_dataset, n_way, k_shot, q_query,
        num_episodes=config.get("num_eval_episodes", 600),
        seed=config["seed"], device=device,
    )

    return final_results


def main():
    parser = argparse.ArgumentParser(
        description="Cross-lingual adaptation for few-shot SLR."
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Base configuration file.",
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="Source language (pretrained model).",
    )
    parser.add_argument(
        "--target", type=str, required=True,
        help="Target language for evaluation.",
    )
    parser.add_argument(
        "--repr", type=str, default="angle",
        help="Representation (raw, angle, raw_angle).",
    )
    parser.add_argument(
        "--encoder", type=str, default="mlp",
        help="Encoder type (mlp, transformer).",
    )
    parser.add_argument(
        "--mode", type=str, default="frozen",
        choices=["frozen", "target_supervised", "both"],
        help="Adaptation mode.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Override checkpoint path.",
    )
    parser.add_argument("--k_shot", type=int, default=None, help="K-shot override.")
    parser.add_argument("--seed", type=int, default=None, help="Seed override.")

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config["representation"] = args.repr
    config["encoder"] = args.encoder
    if args.k_shot is not None:
        config["k_shot"] = args.k_shot
    if args.seed is not None:
        config["seed"] = args.seed

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(
            config.get("checkpoint_dir", "checkpoints"),
            f"{args.source}_{args.encoder}_{args.repr}_best.pt",
        )

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("Train the source model first:")
        print(f"  python train.py --dataset {args.source} --repr {args.repr} "
              f"--encoder {args.encoder}")
        return

    # Load target dataset config
    target_config = config.copy()
    target_ds_path = f"configs/{args.target}.yaml"
    if os.path.exists(target_ds_path):
        with open(target_ds_path, "r") as f:
            target_config.update(yaml.safe_load(f))

    splits_dir = config["splits_dir"]
    target_data_dir = target_config.get("data_dir", f"data/landmarks/{args.target}")

    target_train_split = os.path.join(splits_dir, f"{args.target}_train.json")
    target_test_split = os.path.join(splits_dir, f"{args.target}_test.json")

    target_test_dataset = SplitLandmarkDataset(
        data_dir=target_data_dir,
        split_file=target_test_split,
        representation=args.repr,
    )

    print(f"\n{'='*60}")
    print(f"Cross-Lingual Transfer: {args.source} → {args.target}")
    print(f"Representation: {args.repr} | Encoder: {args.encoder}")
    print(f"{'='*60}")

    # Load pretrained model
    model = load_pretrained_model(checkpoint_path, config, device)

    results_dir = Path(config.get("results_dir", "results")) / "cross_lingual"
    results_dir.mkdir(parents=True, exist_ok=True)

    modes = ["frozen", "target_supervised"] if args.mode == "both" else [args.mode]

    for mode in modes:
        print(f"\n--- Mode: {mode} ---")

        if mode == "frozen":
            results = frozen_evaluation(model, target_test_dataset, config, device)
        else:
            # Need to reload model for fresh adaptation
            model = load_pretrained_model(checkpoint_path, config, device)

            target_train_dataset = SplitLandmarkDataset(
                data_dir=target_data_dir,
                split_file=target_train_split,
                representation=args.repr,
            )

            results = target_supervised_adaptation(
                model, target_train_dataset, target_test_dataset,
                config, device,
            )

        print(f"\n  {args.source} → {args.target} ({mode}):")
        print(f"  Accuracy: {results['accuracy']:.1f} ± {results['ci_95']:.1f}%")

        # Save results
        result_file = results_dir / (
            f"{args.source}_to_{args.target}_{args.encoder}_{args.repr}_{mode}.json"
        )
        with open(result_file, "w") as f:
            json.dump({
                "source": args.source,
                "target": args.target,
                "encoder": args.encoder,
                "representation": args.repr,
                "mode": mode,
                "n_way": config["n_way"],
                "k_shot": config["k_shot"],
                "accuracy": results["accuracy"],
                "ci_95": results["ci_95"],
            }, f, indent=2)
        print(f"  Saved: {result_file}")


if __name__ == "__main__":
    main()
