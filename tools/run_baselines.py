"""
Run baseline comparisons.

Compares three 5-way 5-shot classification methods:
    1. Input-space nearest prototype (no encoder)
    2. Episode-linear (per-episode logistic regression on MLP embeddings)
    3. ProtoNet with MLP encoder

Reproduces Table 9 from the paper.

Usage:
    python tools/run_baselines.py [--datasets asl libras arabic thai]
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from data.dataset import SplitLandmarkDataset
from data.episodes import create_episode_batch
from data.representation import get_input_dim
from evaluate import evaluate_episodes, set_seed
from models import create_model


def input_space_baseline(
    dataset: SplitLandmarkDataset,
    n_way: int,
    k_shot: int,
    q_query: int,
    num_episodes: int,
    seed: int,
) -> dict:
    """
    Input-space nearest prototype baseline (no encoder).

    Computes Euclidean distance directly in the raw/angle feature space.
    """
    accuracies = []

    for ep_idx in tqdm(range(num_episodes), desc="Input-space", leave=False):
        support_features, support_labels, query_features, query_labels = (
            create_episode_batch(
                dataset, n_way, k_shot, q_query,
                seed=seed, episode_idx=ep_idx,
            )
        )

        # Compute prototypes (class means in input space)
        prototypes = torch.zeros(n_way, support_features.size(-1))
        for c in range(n_way):
            mask = support_labels == c
            prototypes[c] = support_features[mask].mean(dim=0)

        # Classify by nearest prototype
        diff = query_features.unsqueeze(1) - prototypes.unsqueeze(0)
        distances = (diff ** 2).sum(dim=-1)
        predictions = distances.argmin(dim=-1)

        acc = (predictions == query_labels).float().mean().item() * 100
        accuracies.append(acc)

    accs = np.array(accuracies)
    return {
        "accuracy": accs.mean(),
        "ci_95": 1.96 * accs.std() / np.sqrt(len(accs)),
    }


def episode_linear_baseline(
    dataset: SplitLandmarkDataset,
    model,
    n_way: int,
    k_shot: int,
    q_query: int,
    num_episodes: int,
    seed: int,
    device: torch.device,
) -> dict:
    """
    Episode-linear baseline: fit per-episode logistic regression on MLP embeddings.
    """
    model.eval()
    accuracies = []

    with torch.no_grad():
        for ep_idx in tqdm(range(num_episodes), desc="Episode-linear", leave=False):
            support_features, support_labels, query_features, query_labels = (
                create_episode_batch(
                    dataset, n_way, k_shot, q_query,
                    seed=seed, episode_idx=ep_idx, device=device,
                )
            )

            # Encode with MLP
            support_emb = model.encode(support_features).cpu().numpy()
            query_emb = model.encode(query_features).cpu().numpy()
            s_labels = support_labels.cpu().numpy()
            q_labels = query_labels.cpu().numpy()

            # Fit logistic regression on support embeddings
            clf = LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                multi_class="multinomial",
                random_state=seed,
            )
            clf.fit(support_emb, s_labels)

            predictions = clf.predict(query_emb)
            acc = (predictions == q_labels).mean() * 100
            accuracies.append(acc)

    accs = np.array(accuracies)
    return {
        "accuracy": accs.mean(),
        "ci_95": 1.96 * accs.std() / np.sqrt(len(accs)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline comparisons (Table 9)."
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["asl", "libras", "arabic", "thai"],
    )
    parser.add_argument(
        "--reprs", nargs="+", default=["raw", "angle"],
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config["seed"] = args.seed
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}

    for dataset_name in args.datasets:
        for repr_name in args.reprs:
            print(f"\n{'='*50}")
            print(f"  {dataset_name} / {repr_name}")
            print(f"{'='*50}")

            # Load dataset config
            ds_config = config.copy()
            ds_config_path = f"configs/{dataset_name}.yaml"
            if os.path.exists(ds_config_path):
                with open(ds_config_path) as f:
                    ds_config.update(yaml.safe_load(f))

            data_dir = ds_config.get("data_dir", f"data/landmarks/{dataset_name}")
            test_split = os.path.join(
                config["splits_dir"], f"{dataset_name}_test.json"
            )

            test_dataset = SplitLandmarkDataset(
                data_dir=data_dir,
                split_file=test_split,
                representation=repr_name,
            )

            n_way = config["n_way"]
            k_shot = config["k_shot"]
            q_query = config["q_query"]
            num_episodes = config["num_eval_episodes"]

            # 1. Input-space baseline
            print("\n  [1] Input-space nearest prototype...")
            input_result = input_space_baseline(
                test_dataset, n_way, k_shot, q_query, num_episodes, args.seed,
            )
            print(f"      Accuracy: {input_result['accuracy']:.1f} ± {input_result['ci_95']:.1f}%")

            # 2. Load trained MLP for episode-linear and ProtoNet baselines
            checkpoint = f"checkpoints/{dataset_name}_mlp_{repr_name}_best.pt"
            input_dim = get_input_dim(repr_name)

            model = create_model(
                encoder_name="mlp",
                input_dim=input_dim,
                embedding_dim=config["embedding_dim"],
                hidden_dim=config["hidden_dim"],
                representation=repr_name,
            ).to(device)

            if os.path.exists(checkpoint):
                ckpt = torch.load(checkpoint, map_location=device)
                model.load_state_dict(ckpt["model_state_dict"])

                # 2. Episode-linear baseline
                print("  [2] Episode-linear (logistic regression)...")
                linear_result = episode_linear_baseline(
                    test_dataset, model, n_way, k_shot, q_query,
                    num_episodes, args.seed, device,
                )
                print(f"      Accuracy: {linear_result['accuracy']:.1f} ± {linear_result['ci_95']:.1f}%")

                # 3. ProtoNet baseline
                print("  [3] ProtoNet with MLP encoder...")
                proto_result = evaluate_episodes(
                    model, test_dataset, n_way, k_shot, q_query,
                    num_episodes, args.seed, device,
                )
                print(f"      Accuracy: {proto_result['accuracy']:.1f} ± {proto_result['ci_95']:.1f}%")
            else:
                print(f"      Checkpoint not found: {checkpoint}")
                linear_result = None
                proto_result = None

            key = f"{dataset_name}/{repr_name}"
            results[key] = {
                "input_space": input_result,
                "episode_linear": linear_result,
                "protonet": {
                    "accuracy": proto_result["accuracy"],
                    "ci_95": proto_result["ci_95"],
                } if proto_result else None,
            }

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/baselines.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBaseline results saved: results/baselines.json")


if __name__ == "__main__":
    main()
