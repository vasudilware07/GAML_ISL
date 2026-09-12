"""
Prototypical Networks for few-shot sign language recognition.

Reference: Snell et al., "Prototypical Networks for Few-shot Learning", NeurIPS 2017.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypicalNetwork(nn.Module):
    """Prototypical Network wrapper around an arbitrary encoder backbone.

    During an episode the support set is encoded, class prototypes are
    computed as mean embeddings, and query samples are classified based on
    negative Euclidean distance to each prototype.

    Args:
        encoder: Backbone that maps input to ``(B, D)`` embedding.
        distance: Distance metric — ``'euclidean'`` or ``'cosine'``.
    """

    def __init__(
        self,
        encoder: nn.Module,
        distance: str = "euclidean",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.distance = distance

    def compute_prototypes(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        n_way: int,
    ) -> torch.Tensor:
        """Compute class prototypes from the support set.

        Args:
            support_x: Support embeddings ``(N_s, D)``.
            support_y: Support labels ``(N_s,)`` with values in ``[0, n_way)``.
            n_way: Number of classes.

        Returns:
            Prototypes ``(n_way, D)``.
        """
        D = support_x.size(-1)
        prototypes = torch.zeros(n_way, D, device=support_x.device)
        for c in range(n_way):
            mask = support_y == c
            prototypes[c] = support_x[mask].mean(dim=0)
        return prototypes

    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        n_way: int,
    ) -> torch.Tensor:
        """Forward pass for one episode.

        Args:
            support_x: Support inputs.
            support_y: Support labels in ``[0, n_way)``.
            query_x: Query inputs.
            n_way: Number of classes.

        Returns:
            Log-probabilities over *n_way* classes for each query, ``(N_q, n_way)``.
        """
        z_support = self.encoder(support_x)  # (N_s, D)
        z_query = self.encoder(query_x)      # (N_q, D)

        prototypes = self.compute_prototypes(z_support, support_y, n_way)  # (n_way, D)

        if self.distance == "euclidean":
            # Negative squared Euclidean distance
            dists = torch.cdist(z_query, prototypes, p=2)  # (N_q, n_way)
            logits = -dists
        elif self.distance == "cosine":
            z_query_n = F.normalize(z_query, dim=-1)
            prototypes_n = F.normalize(prototypes, dim=-1)
            logits = z_query_n @ prototypes_n.t()  # (N_q, n_way)
        else:
            raise ValueError(f"Unknown distance: {self.distance}")

        return F.log_softmax(logits, dim=-1)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings without episodic computation.

        Args:
            x: Input tensor.

        Returns:
            Embeddings ``(B, D)``.
        """
        return self.encoder(x)
