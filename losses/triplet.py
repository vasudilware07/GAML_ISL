"""
Triplet loss with online batch mining strategies.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_distances(embeddings: torch.Tensor, squared: bool = False) -> torch.Tensor:
    """Compute pairwise Euclidean distance matrix.

    Args:
        embeddings: ``(N, D)`` embeddings.
        squared: If True, return squared distances.

    Returns:
        ``(N, N)`` distance matrix.
    """
    dot = embeddings @ embeddings.t()
    sq_norms = torch.diag(dot)
    dists = sq_norms.unsqueeze(0) - 2.0 * dot + sq_norms.unsqueeze(1)
    dists = torch.clamp(dists, min=0.0)
    if not squared:
        # Avoid NaN grad at zero
        mask = (dists == 0.0).float()
        dists = torch.sqrt(dists + mask * 1e-16)
        dists = dists * (1.0 - mask)
    return dists


def get_valid_triplet_mask(labels: torch.Tensor) -> torch.Tensor:
    """Return boolean mask for valid (anchor, positive, negative) triplets.

    A valid triplet ``(i, j, k)`` satisfies:
    - ``i != j != k``
    - ``labels[i] == labels[j]``
    - ``labels[i] != labels[k]``

    Args:
        labels: ``(N,)`` integer labels.

    Returns:
        ``(N, N, N)`` boolean mask.
    """
    N = labels.size(0)
    indices_not_equal = ~torch.eye(N, dtype=torch.bool, device=labels.device)
    i_ne_j = indices_not_equal.unsqueeze(2)
    i_ne_k = indices_not_equal.unsqueeze(1)
    j_ne_k = indices_not_equal.unsqueeze(0)
    distinct = i_ne_j & i_ne_k & j_ne_k

    lbl_eq = labels.unsqueeze(0) == labels.unsqueeze(1)   # (N, N)
    lbl_ne = ~lbl_eq

    valid = distinct & lbl_eq.unsqueeze(2) & lbl_ne.unsqueeze(1)
    return valid


class TripletLoss(nn.Module):
    """Triplet loss with online mining.

    Supports three mining strategies:
    - ``'all'``: all valid triplets.
    - ``'hard'``: hardest positive + hardest negative per anchor.
    - ``'semi_hard'``: hardest negative that is farther than the positive
      but within the margin.

    Args:
        margin: Triplet margin.
        mining: Mining strategy.
    """

    def __init__(self, margin: float = 0.5, mining: str = "semi_hard") -> None:
        super().__init__()
        self.margin = margin
        self.mining = mining

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute triplet loss.

        Args:
            embeddings: ``(N, D)`` normalised or unnormalised embeddings.
            labels: ``(N,)`` integer labels.

        Returns:
            Scalar loss.
        """
        if self.mining == "all":
            return self._batch_all(embeddings, labels)
        elif self.mining == "hard":
            return self._batch_hard(embeddings, labels)
        elif self.mining == "semi_hard":
            return self._batch_semi_hard(embeddings, labels)
        else:
            raise ValueError(f"Unknown mining strategy: {self.mining}")

    def _batch_all(
        self, embeddings: torch.Tensor, labels: torch.Tensor,
    ) -> torch.Tensor:
        dists = pairwise_distances(embeddings)
        N = labels.size(0)

        ap = dists.unsqueeze(2)  # (N, N, 1)
        an = dists.unsqueeze(1)  # (N, 1, N)
        loss = torch.clamp(ap - an + self.margin, min=0.0)  # (N, N, N)

        mask = get_valid_triplet_mask(labels).float()
        loss = loss * mask
        num_valid = mask.sum()
        return loss.sum() / (num_valid + 1e-8)

    def _batch_hard(
        self, embeddings: torch.Tensor, labels: torch.Tensor,
    ) -> torch.Tensor:
        dists = pairwise_distances(embeddings)
        N = labels.size(0)

        lbl_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N, N)
        lbl_ne = ~lbl_eq

        # Hardest positive
        ap_dists = dists * lbl_eq.float()
        hardest_pos, _ = ap_dists.max(dim=1)

        # Hardest negative (set positives to large value)
        large = dists.max() + 1.0
        an_dists = dists + lbl_eq.float() * large
        hardest_neg, _ = an_dists.min(dim=1)

        loss = torch.clamp(hardest_pos - hardest_neg + self.margin, min=0.0)
        return loss.mean()

    def _batch_semi_hard(
        self, embeddings: torch.Tensor, labels: torch.Tensor,
    ) -> torch.Tensor:
        dists = pairwise_distances(embeddings)
        N = labels.size(0)

        lbl_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        lbl_ne = ~lbl_eq

        # For each anchor, get mean positive distance
        ap_dists = dists * lbl_eq.float()
        ap_mean = ap_dists.sum(dim=1) / (lbl_eq.float().sum(dim=1) + 1e-8)

        # Semi-hard negatives: negatives farther than positive but within margin
        large = dists.max() + 1.0
        an_dists = dists.clone()
        an_dists[lbl_eq] = large  # mask out positives

        # Choose negatives within [ap_mean, ap_mean + margin]
        semi_hard_mask = (an_dists > ap_mean.unsqueeze(1)) & (an_dists < ap_mean.unsqueeze(1) + self.margin)

        # If no semi-hard, fall back to hardest negative
        has_semi = semi_hard_mask.any(dim=1)

        # For semi-hard: pick the closest
        an_dists_sh = an_dists.clone()
        an_dists_sh[~semi_hard_mask] = large
        semi_hard_neg, _ = an_dists_sh.min(dim=1)

        # For hard fallback
        hard_neg, _ = an_dists.min(dim=1)

        neg_dists = torch.where(has_semi, semi_hard_neg, hard_neg)
        loss = torch.clamp(ap_mean - neg_dists + self.margin, min=0.0)
        return loss.mean()
