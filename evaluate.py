"""
Evaluation script for few-shot sign language recognition.

Runs 600-episode evaluation with 95% confidence intervals, produces
t-SNE visualisations and confusion matrices.

Usage:
    python evaluate.py --config configs/default.yaml \
        --dataset asl --repr angle --encoder mlp \
        --checkpoint checkpoints/asl_mlp_angle_best.pt

References:
    Section 3.4 — Few-Shot Evaluation Protocol
    Section 3.5 — Implementation Details (Evaluation and reproducibility)
"""

import argparse
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from sklearn.manifold import TSNE
from tqdm import tqdm

from data.dataset import SplitLandmarkDataset
from data.episodes import create_episode_batch
from data.representation import get_input_dim
from models import create_model, count_parameters


def set_seed(seed: int):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_episodes(
    model,
    dataset: SplitLandmarkDataset,
    n_way: int,
    k_shot: int,
    q_query: int,
    num_episodes: int,
    seed: int,
    device: torch.device,
) -> dict:
    """
    Run N-way K-shot evaluation over multiple episodes.

    Each episode is seeded by (seed + episode_index) for exact reproducibility.

    Args:
        model: PrototypicalNetwork instance.
        dataset: Test dataset.
        n_way: Number of classes per episode (5).
        k_shot: Support examples per class.
        q_query: Query examples per class (15).
        num_episodes: Number of episodes (600).
        seed: Base seed (42).
        device: Compute device.

    Returns:
        Dictionary with:
            - accuracy: Mean accuracy across episodes (%)
            - ci_95: 95% confidence interval (1.96σ/√n)
            - per_episode: List of per-episode accuracies
            - all_predictions: All predictions for confusion analysis
            - all_labels: All true labels
    """
    model.eval()

    episode_accuracies = []
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for ep_idx in tqdm(range(num_episodes), desc="Evaluating", leave=False):
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
            episode_accuracies.append(acc)

            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_labels.extend(query_labels.cpu().numpy().tolist())

    accuracies = np.array(episode_accuracies)
    mean_acc = accuracies.mean()
    std_acc = accuracies.std()
    ci_95 = 1.96 * std_acc / np.sqrt(len(accuracies))

    return {
        "accuracy": mean_acc,
        "std": std_acc,
        "ci_95": ci_95,
        "per_episode": episode_accuracies,
        "all_predictions": all_predictions,
        "all_labels": all_labels,
    }


def compute_per_class_accuracy(
    predictions: list,
    labels: list,
    n_way: int,
) -> dict:
    """Compute per-class accuracy from episode predictions."""
    predictions = np.array(predictions)
    labels = np.array(labels)

    per_class = {}
    for c in range(n_way):
        mask = labels == c
        if mask.sum() > 0:
            per_class[c] = (predictions[mask] == c).mean() * 100

    return per_class


def plot_tsne(
    model,
    dataset: SplitLandmarkDataset,
    n_way: int,
    k_shot: int,
    q_query: int,
    seed: int,
    device: torch.device,
    save_path: str,
    perplexity: int = 30,
    n_iter: int = 1000,
):
    """
    Generate t-SNE visualisation of embeddings from a single episode.

    Saves the plot to save_path.
    """
    model.eval()

    with torch.no_grad():
        support_features, support_labels, query_features, query_labels = (
            create_episode_batch(
                dataset, n_way, k_shot, q_query,
                seed=seed, episode_idx=0, device=device,
            )
        )

        # Get embeddings
        all_features = torch.cat([support_features, query_features], dim=0)
        all_labels = torch.cat([support_labels, query_labels], dim=0)
        embeddings = model.encode(all_features).cpu().numpy()
        labels = all_labels.cpu().numpy()

    # Compute t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(embeddings) - 1),
        n_iter=n_iter,
        random_state=seed,
    )
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    palette = sns.color_palette("husl", n_way)

    n_support = n_way * k_shot

    for c in range(n_way):
        # Support points (larger markers)
        mask_s = (labels[:n_support] == c)
        ax.scatter(
            embeddings_2d[:n_support][mask_s, 0],
            embeddings_2d[:n_support][mask_s, 1],
            color=palette[c], marker="^", s=120, edgecolors="black",
            label=f"Class {c} (support)", zorder=3,
        )

        # Query points (smaller markers)
        mask_q = (labels[n_support:] == c)
        ax.scatter(
            embeddings_2d[n_support:][mask_q, 0],
            embeddings_2d[n_support:][mask_q, 1],
            color=palette[c], marker="o", s=40, alpha=0.6,
            label=f"Class {c} (query)",
        )

    ax.set_title("t-SNE Embedding Visualisation", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"t-SNE plot saved: {save_path}")


def plot_confusion_matrix(
    predictions: list,
    labels: list,
    n_way: int,
    save_path: str,
):
    """Generate and save a confusion matrix heatmap."""
    predictions = np.array(predictions)
    labels = np.array(labels)

    # Compute confusion matrix
    cm = np.zeros((n_way, n_way), dtype=int)
    for true, pred in zip(labels, predictions):
        cm[true][pred] += 1

    # Normalize rows
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=range(n_way), yticklabels=range(n_way), ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix (Normalised)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate few-shot SLR model."
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument("--dataset", type=str, help="Dataset name.")
    parser.add_argument("--repr", type=str, help="Representation name.")
    parser.add_argument("--encoder", type=str, help="Encoder name.")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path.")
    parser.add_argument("--k_shot", type=int, help="K-shot override.")
    parser.add_argument("--num_episodes", type=int, help="Number of episodes.")
    parser.add_argument("--seed", type=int, help="Seed override.")
    parser.add_argument("--data_dir", type=str, help="Data directory override.")
    parser.add_argument("--no_tsne", action="store_true", help="Skip t-SNE plot.")
    parser.add_argument("--no_confusion", action="store_true",
                        help="Skip confusion matrix.")

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Apply overrides
    if args.dataset:
        config["dataset"] = args.dataset
        ds_config_path = f"configs/{args.dataset}.yaml"
        if os.path.exists(ds_config_path):
            with open(ds_config_path, "r") as f:
                config.update(yaml.safe_load(f))
    if args.repr:
        config["representation"] = args.repr
    if args.encoder:
        config["encoder"] = args.encoder
    if args.k_shot is not None:
        config["k_shot"] = args.k_shot
    if args.num_episodes is not None:
        config["num_eval_episodes"] = args.num_episodes
    if args.seed is not None:
        config["seed"] = args.seed
    if args.data_dir:
        config["data_dir"] = args.data_dir

    # Setup
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset_name = config["dataset"]
    representation = config["representation"]
    splits_dir = config["splits_dir"]
    data_dir = config["data_dir"]

    test_split = os.path.join(splits_dir, f"{dataset_name}_test.json")
    test_dataset = SplitLandmarkDataset(
        data_dir=data_dir,
        split_file=test_split,
        representation=representation,
    )

    print(f"Test dataset: {test_dataset}")

    # Create model
    input_dim = get_input_dim(representation)
    model = create_model(
        encoder_name=config["encoder"],
        input_dim=input_dim,
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config.get("dropout", 0.3),
        representation=representation,
    )
    model = model.to(device)

    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint}")
        print(f"  Checkpoint val accuracy: {checkpoint.get('val_accuracy', 'N/A')}")

    print(f"Model parameters: {count_parameters(model):,}")

    # Evaluate
    n_way = config["n_way"]
    k_shot = config["k_shot"]
    q_query = config["q_query"]
    num_episodes = config["num_eval_episodes"]
    seed = config["seed"]

    print(f"\nEvaluating: {n_way}-way {k_shot}-shot, Q={q_query}, "
          f"{num_episodes} episodes, seed={seed}")

    results = evaluate_episodes(
        model, test_dataset, n_way, k_shot, q_query,
        num_episodes, seed, device,
    )

    print(f"\nResults:")
    print(f"  Accuracy: {results['accuracy']:.1f} ± {results['ci_95']:.1f}%")
    print(f"  Std:      {results['std']:.2f}")

    # Save results
    results_dir = Path(config.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    result_name = f"{dataset_name}_{config['encoder']}_{representation}_{k_shot}shot"
    result_file = results_dir / f"{result_name}.json"

    with open(result_file, "w") as f:
        json.dump({
            "dataset": dataset_name,
            "encoder": config["encoder"],
            "representation": representation,
            "n_way": n_way,
            "k_shot": k_shot,
            "q_query": q_query,
            "num_episodes": num_episodes,
            "seed": seed,
            "accuracy": results["accuracy"],
            "ci_95": results["ci_95"],
            "std": results["std"],
        }, f, indent=2)

    print(f"Results saved: {result_file}")

    # t-SNE visualisation
    if not args.no_tsne:
        tsne_path = results_dir / f"{result_name}_tsne.png"
        plot_tsne(
            model, test_dataset, n_way, k_shot, q_query,
            seed, device, str(tsne_path),
        )

    # Confusion matrix
    if not args.no_confusion:
        cm_path = results_dir / f"{result_name}_confusion.png"
        plot_confusion_matrix(
            results["all_predictions"], results["all_labels"],
            n_way, str(cm_path),
        )


if __name__ == "__main__":
    main()
