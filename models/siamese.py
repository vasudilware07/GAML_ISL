"""
Siamese Network for few-shot sign language recognition.

Supports pairwise similarity learning and episode-based evaluation.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    """Siamese network that learns a pairwise similarity function.

    The backbone encoder is shared between both branches. A learnable
    similarity head maps the element-wise absolute difference of two
    embeddings to a scalar similarity score.

    Args:
        encoder: Backbone encoder mapping input → ``(B, D)`` embedding.
        embedding_dim: Embedding dimensionality (must match encoder output).
    """

    def __init__(self, encoder: nn.Module, embedding_dim: int = 128) -> None:
        super().__init__()
        self.encoder = encoder
        self.similarity_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward_pair(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pairwise similarity scores.

        Args:
            x1: First input batch ``(B, ...)``.
            x2: Second input batch ``(B, ...)``.

        Returns:
            Similarity logits ``(B, 1)``.
        """
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        diff = torch.abs(z1 - z2)
        return self.similarity_head(diff)

    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        n_way: int,
    ) -> torch.Tensor:
        """Episode-based forward pass (compatible with Prototypical API).

        For each query, compute similarity to each support exemplar and
        aggregate per-class similarities.

        Args:
            support_x: Support inputs.
            support_y: Support labels ``[0, n_way)``.
            query_x: Query inputs.
            n_way: Number of classes.

        Returns:
            Log-probabilities ``(N_q, n_way)``.
        """
        z_support = self.encoder(support_x)  # (N_s, D)
        z_query = self.encoder(query_x)      # (N_q, D)

        N_q = z_query.size(0)
        N_s = z_support.size(0)

        # Compute all pairwise similarities
        # Expand: (N_q, N_s, D)
        z_q_exp = z_query.unsqueeze(1).expand(-1, N_s, -1)
        z_s_exp = z_support.unsqueeze(0).expand(N_q, -1, -1)
        diff = torch.abs(z_q_exp - z_s_exp)  # (N_q, N_s, D)
        sim = self.similarity_head(diff).squeeze(-1)  # (N_q, N_s)

        # Aggregate similarities per class
        logits = torch.zeros(N_q, n_way, device=sim.device)
        for c in range(n_way):
            mask = support_y == c
            logits[:, c] = sim[:, mask].mean(dim=-1)

        return F.log_softmax(logits, dim=-1)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings."""
        return self.encoder(x)


class MatchingNetwork(nn.Module):
    """Matching Network using cosine attention over support embeddings.

    Reference: Vinyals et al., "Matching Networks for One Shot Learning",
    NeurIPS 2016.

    Args:
        encoder: Backbone encoder.
    """

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        n_way: int,
    ) -> torch.Tensor:
        """Episode forward pass.

        Args:
            support_x: Support inputs.
            support_y: Support labels ``[0, n_way)``.
            query_x: Query inputs.
            n_way: Number of classes.

        Returns:
            Log-probabilities ``(N_q, n_way)``.
        """
        z_support = F.normalize(self.encoder(support_x), dim=-1)  # (N_s, D)
        z_query = F.normalize(self.encoder(query_x), dim=-1)      # (N_q, D)

        # Cosine similarity as attention
        attn = z_query @ z_support.t()  # (N_q, N_s)
        attn = F.softmax(attn, dim=-1)

        # One-hot support labels
        support_onehot = F.one_hot(support_y, num_classes=n_way).float()  # (N_s, n_way)
        logits = attn @ support_onehot  # (N_q, n_way)

        return torch.log(logits + 1e-8)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings."""
        return self.encoder(x)
