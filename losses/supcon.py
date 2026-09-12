"""
Supervised Contrastive Loss (SupCon).

Implementation of the supervised contrastive learning loss from
Khosla et al. (2020). Pulls together embeddings of the same class
while pushing apart embeddings of different classes.

Used as an auxiliary training objective combined with the Prototypical
Network episode classification loss:
    L_total = L_episode + λ * L_supcon
where λ = 0.5 and temperature τ = 0.07.

References:
    Section 3.5 — Implementation Details (Training and losses)
    Khosla et al. (2020) — Supervised Contrastive Learning (NeurIPS)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    For a batch of embeddings with known labels, this loss:
        1. Computes pairwise cosine similarities (scaled by temperature)
        2. For each anchor, treats same-class samples as positives
        3. Maximises agreement between positives while minimising agreement
           with negatives

    Args:
        temperature: Temperature scaling parameter τ (default: 0.07).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the supervised contrastive loss.

        Args:
            features: Tensor of shape (batch_size, embedding_dim) — L2-normalised
                      embeddings are computed internally.
            labels: Tensor of shape (batch_size,) — class labels.

        Returns:
            Scalar loss value.
        """
        device = features.device
        batch_size = features.shape[0]

        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # L2-normalise features
        features = F.normalize(features, p=2, dim=1)

        # Compute similarity matrix: (batch_size, batch_size)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Create mask for positive pairs (same class, different sample)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Remove self-similarity (diagonal)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        # Check if there are any positive pairs
        positives_per_row = mask.sum(dim=1)
        has_positives = positives_per_row > 0

        if not has_positives.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # For numerical stability, subtract the max from logits
        logits_max, _ = torch.max(similarity_matrix * logits_mask, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # Compute log-sum-exp of all negatives + positives (denominator)
        exp_logits = torch.exp(logits) * logits_mask
        log_sum_exp = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Compute mean of log-prob over positive pairs
        log_prob = logits - log_sum_exp

        # Average over positive pairs for each anchor
        mean_log_prob = (mask * log_prob).sum(dim=1) / (positives_per_row + 1e-8)

        # Only include anchors that have positive pairs
        loss = -mean_log_prob[has_positives].mean()

        return loss
