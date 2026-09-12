"""
Episodic sampling for N-way K-shot few-shot learning.

Generates deterministic episodes for Prototypical Network training and
evaluation. Each episode samples N classes, draws K support and Q query
examples per class, and partitions them for metric-learning classification.

References:
    Section 3.4 — Few-Shot Evaluation Protocol
    Section 3.5 — Implementation Details (Episodic sampling)
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class EpisodicSampler(Sampler):
    """
    Sampler that generates N-way K-shot episodes for few-shot learning.

    Each episode:
        1. Samples N classes from eligible classes (those with ≥ K+Q test samples)
        2. For each class, draws K support + Q query samples without replacement
        3. Seeds each episode with (base_seed + episode_index) for reproducibility

    Args:
        dataset: A SplitLandmarkDataset instance with class_samples attribute.
        n_way: Number of classes per episode (N).
        k_shot: Number of support examples per class (K).
        q_query: Number of query examples per class (Q).
        num_episodes: Total number of episodes to generate.
        seed: Base random seed for reproducibility.
    """

    def __init__(
        self,
        dataset,
        n_way: int = 5,
        k_shot: int = 5,
        q_query: int = 15,
        num_episodes: int = 600,
        seed: int = 42,
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.num_episodes = num_episodes
        self.seed = seed

        # Get eligible classes (need at least K + Q samples)
        min_samples = k_shot + q_query
        self.eligible_classes = dataset.get_eligible_classes(min_samples)

        if len(self.eligible_classes) < n_way:
            raise ValueError(
                f"Not enough eligible classes for {n_way}-way episodes. "
                f"Found {len(self.eligible_classes)} classes with ≥ {min_samples} samples, "
                f"need at least {n_way}."
            )

    def __iter__(self):
        for episode_idx in range(self.num_episodes):
            # Deterministic seed per episode
            rng = random.Random(self.seed + episode_idx)

            # Step 1: Sample N classes
            episode_classes = rng.sample(self.eligible_classes, self.n_way)

            # Step 2: For each class, sample K + Q examples
            batch_indices = []
            for class_idx in episode_classes:
                class_samples = self.dataset.get_class_samples(class_idx)
                selected = rng.sample(class_samples, self.k_shot + self.q_query)
                batch_indices.extend(selected)

            yield batch_indices

    def __len__(self):
        return self.num_episodes


def split_support_query(
    features: torch.Tensor,
    labels: torch.Tensor,
    n_way: int,
    k_shot: int,
    q_query: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split a batch into support and query sets, re-labelling classes to {0,...,N-1}.

    Given a batch from EpisodicSampler (N*(K+Q) samples ordered by class),
    splits into K support and Q query samples per class.

    Args:
        features: Tensor of shape (N*(K+Q), D) — feature vectors.
        labels: Tensor of shape (N*(K+Q),) — original class labels.
        n_way: Number of classes in the episode.
        k_shot: Number of support examples per class.
        q_query: Number of query examples per class.

    Returns:
        support_features: Tensor of shape (N*K, D)
        support_labels: Tensor of shape (N*K,) — re-labelled to {0,...,N-1}
        query_features: Tensor of shape (N*Q, D)
        query_labels: Tensor of shape (N*Q,) — re-labelled to {0,...,N-1}
    """
    samples_per_class = k_shot + q_query

    support_features_list = []
    support_labels_list = []
    query_features_list = []
    query_labels_list = []

    for i in range(n_way):
        start = i * samples_per_class
        end = start + samples_per_class

        class_features = features[start:end]

        # Support: first K samples
        support_features_list.append(class_features[:k_shot])
        support_labels_list.append(
            torch.full((k_shot,), i, dtype=torch.long)
        )

        # Query: remaining Q samples
        query_features_list.append(class_features[k_shot:k_shot + q_query])
        query_labels_list.append(
            torch.full((q_query,), i, dtype=torch.long)
        )

    support_features = torch.cat(support_features_list, dim=0)
    support_labels = torch.cat(support_labels_list, dim=0)
    query_features = torch.cat(query_features_list, dim=0)
    query_labels = torch.cat(query_labels_list, dim=0)

    return support_features, support_labels, query_features, query_labels


def create_episode_batch(
    dataset,
    n_way: int,
    k_shot: int,
    q_query: int,
    seed: int,
    episode_idx: int,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a single episode batch directly from a dataset.

    Convenience function that combines sampling and splitting.

    Args:
        dataset: SplitLandmarkDataset instance.
        n_way: Number of classes per episode.
        k_shot: Support examples per class.
        q_query: Query examples per class.
        seed: Base seed.
        episode_idx: Episode index (combined with seed for reproducibility).
        device: Target device for tensors.

    Returns:
        support_features, support_labels, query_features, query_labels
    """
    rng = random.Random(seed + episode_idx)
    min_samples = k_shot + q_query
    eligible = dataset.get_eligible_classes(min_samples)

    episode_classes = rng.sample(eligible, n_way)

    features_list = []
    labels_list = []

    for class_idx in episode_classes:
        class_samples = dataset.get_class_samples(class_idx)
        selected_indices = rng.sample(class_samples, k_shot + q_query)

        for sample_idx in selected_indices:
            feat, _ = dataset[sample_idx]
            features_list.append(feat)
            labels_list.append(class_idx)

    features = torch.stack(features_list).to(device)
    labels = torch.tensor(labels_list, dtype=torch.long).to(device)

    return split_support_query(features, labels, n_way, k_shot, q_query)
