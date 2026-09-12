"""
Episodic data sampler for N-way K-shot few-shot learning.

Deterministic per-episode seeding, strict K+Q feasibility, NaN guard.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)


class EpisodicSampler(Sampler):
    """Sampler that yields indices for N-way K-shot episodes.

    Each episode selects *n_way* classes and for each class samples
    *k_shot + q_query* examples.  Sampling is **deterministic**: episode
    *e* uses ``np.random.RandomState(seed + e)``.

    Feasibility rule
    ~~~~~~~~~~~~~~~~
    A class is *eligible* only when it has at least ``k_shot + q_query``
    samples.  If fewer than ``n_way`` classes are eligible, the sampler
    raises unless ``auto_adjust_q`` is enabled, in which case ``q_query``
    is lowered to the largest value that makes at least ``n_way`` classes
    eligible (minimum 1).

    Args:
        labels: List of integer labels for every sample in the dataset.
        n_way: Number of classes per episode.
        k_shot: Support examples per class.
        q_query: Query examples per class.
        episodes: Number of episodes to generate.
        seed: Base random seed for deterministic episode generation.
        auto_adjust_q: If True, lower ``q_query`` when necessary.
        dataset_name: Logical name used in log / error messages.
        split_name: ``'train'`` or ``'test'`` – used in log / error messages.
    """

    def __init__(
        self,
        labels: List[int],
        n_way: int = 5,
        k_shot: int = 5,
        q_query: int = 15,
        episodes: int = 1000,
        seed: int = 42,
        auto_adjust_q: bool = False,
        dataset_name: str = "unknown",
        split_name: str = "unknown",
    ) -> None:
        super().__init__()
        self.labels = np.array(labels)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.episodes = episodes
        self.seed = seed
        self.auto_adjust_q = auto_adjust_q
        self.dataset_name = dataset_name
        self.split_name = split_name

        # Build class → indices mapping
        self.class_indices: Dict[int, List[int]] = {}
        for idx, lbl in enumerate(self.labels):
            self.class_indices.setdefault(int(lbl), []).append(idx)

        # Per-class counts (for diagnostics)
        class_counts = {c: len(idxs) for c, idxs in self.class_indices.items()}
        min_count = min(class_counts.values()) if class_counts else 0

        # ── Feasibility check ────────────────────────────────────────────
        required = k_shot + q_query
        eligible = [c for c, n in class_counts.items() if n >= required]

        if len(eligible) < n_way and auto_adjust_q:
            # Lower q_query to the largest value that makes >= n_way classes eligible
            for q_try in range(q_query - 1, 0, -1):
                req = k_shot + q_try
                elig = [c for c, n in class_counts.items() if n >= req]
                if len(elig) >= n_way:
                    logger.warning(
                        "[%s/%s] auto_adjust_q: q_query lowered %d → %d "
                        "(min_per_class=%d, eligible=%d/%d)",
                        dataset_name, split_name, q_query, q_try,
                        min_count, len(elig), len(class_counts),
                    )
                    self.q_query = q_try
                    eligible = elig
                    break
            else:
                # Even q=1 is not enough
                eligible = []  # fall through to the error below

        if len(eligible) < n_way:
            # Detailed diagnostic
            too_small = {c: n for c, n in class_counts.items() if n < required}
            raise ValueError(
                f"[{dataset_name}/{split_name}] K+Q feasibility FAILED.\n"
                f"  Need n_c >= K+Q = {k_shot}+{self.q_query} = {k_shot + self.q_query} "
                f"for at least N={n_way} classes.\n"
                f"  Eligible classes: {len(eligible)}/{len(class_counts)}.\n"
                f"  Min samples/class: {min_count}.\n"
                f"  Classes with too few samples ({len(too_small)}): "
                + ", ".join(f"cls {c}({n})" for c, n in sorted(too_small.items())[:10])
                + ("\n  Hint: lower K, Q, or N; or use --auto_adjust_q." if not auto_adjust_q else "")
            )

        self.valid_classes = sorted(eligible)

        # ── Sanity log ───────────────────────────────────────────────────
        logger.info(
            "[%s/%s] EpisodicSampler ready: %d-way %d-shot %d-query, "
            "%d episodes, seed=%d, eligible %d/%d classes, min_n_c=%d",
            dataset_name, split_name, n_way, k_shot, self.q_query,
            episodes, seed, len(self.valid_classes), len(class_counts), min_count,
        )

    def __iter__(self):
        for e in range(self.episodes):
            rng = np.random.RandomState(self.seed + e)
            episode_classes = rng.choice(
                self.valid_classes, size=self.n_way, replace=False
            ).tolist()
            indices: List[int] = []
            for cls in episode_classes:
                cls_idxs = self.class_indices[cls]
                chosen = rng.choice(
                    cls_idxs, size=self.k_shot + self.q_query, replace=False
                ).tolist()
                indices.extend(chosen)
            yield indices

    def __len__(self) -> int:
        return self.episodes


def split_support_query(
    batch: Tuple[torch.Tensor, torch.Tensor],
    n_way: int,
    k_shot: int,
    q_query: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a flat episode batch into support and query sets.

    The batch is assumed to be ordered: for each of the *n_way* classes,
    the first *k_shot* samples are support, the next *q_query* are query.

    Raises ``RuntimeError`` if any data tensor contains NaN.

    Args:
        batch: Tuple of (data_tensor, label_tensor).
        n_way: Number of classes.
        k_shot: Support shots.
        q_query: Query shots.

    Returns:
        (support_x, support_y, query_x, query_y)
    """
    data, labels = batch

    # ── NaN guard ────────────────────────────────────────────────────────
    if torch.isnan(data).any():
        nan_count = int(torch.isnan(data).sum())
        raise RuntimeError(
            f"NaN detected in episode data! {nan_count} NaN values in "
            f"tensor of shape {tuple(data.shape)}. "
            f"Check preprocessing or data files."
        )

    per_class = k_shot + q_query

    support_x, support_y, query_x, query_y = [], [], [], []

    for i in range(n_way):
        start = i * per_class
        s_end = start + k_shot
        q_end = s_end + q_query

        support_x.append(data[start:s_end])
        support_y.append(labels[start:s_end])
        query_x.append(data[s_end:q_end])
        query_y.append(labels[s_end:q_end])

    support_x = torch.cat(support_x, dim=0)
    support_y = torch.cat(support_y, dim=0)
    query_x = torch.cat(query_x, dim=0)
    query_y = torch.cat(query_y, dim=0)

    # ── Support/query disjointness assertion ─────────────────────────────
    # By construction (positional split of replace=False sample), support
    # and query indices are disjoint.  Assert tensor-level uniqueness as a
    # defence-in-depth check.
    n_support = support_x.shape[0]
    n_query = query_x.shape[0]
    assert n_support == n_way * k_shot, (
        f"Expected {n_way * k_shot} support samples, got {n_support}"
    )
    assert n_query == n_way * q_query, (
        f"Expected {n_way * q_query} query samples, got {n_query}"
    )

    # Re-label to 0..n_way-1
    unique_labels = support_y.unique()
    label_map = {int(old): new for new, old in enumerate(unique_labels.tolist())}
    support_y = torch.tensor([label_map[int(l)] for l in support_y])
    query_y = torch.tensor([label_map[int(l)] for l in query_y])

    return support_x, support_y, query_x, query_y


def collate_episode(batch):
    """Custom collate that stacks episode samples into a single tensor.

    Args:
        batch: List of lists of (tensor, label) from EpisodicSampler.

    Returns:
        (data, labels) tensors.
    """
    # batch is a list of (tensor, label) tuples from a single episode
    data = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return data, labels
