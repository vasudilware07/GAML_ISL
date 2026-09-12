"""
Prototypical Network for few-shot classification.

Classifies query samples by computing Euclidean distance to class prototypes
(centroids of support set embeddings). No learnable parameters are introduced
beyond those of the encoder.

References:
    Section 3.3 — Model Architecture (Prototypical Network head)
    Section 3.4 — Few-Shot Evaluation Protocol
    Snell et al. (2017) — Prototypical Networks for Few-Shot Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network wrapper around an encoder.

    Given an N-way K-shot episode:
        1. Encode all support and query samples using the encoder
        2. Compute class prototypes as mean of K support embeddings
        3. Classify queries by nearest prototype (Euclidean distance)
        4. Apply softmax over negative squared distances for probabilities

    Args:
        encoder: nn.Module that maps input features to embedding space.
        distance: Distance metric ('euclidean'). Only Euclidean is used in the paper.
    """

    def __init__(self, encoder: nn.Module, distance: str = "euclidean"):
        super().__init__()
        self.encoder = encoder
        self.distance = distance

    def compute_prototypes(
        self,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor,
        n_way: int,
    ) -> torch.Tensor:
        """
        Compute class prototypes as the mean of support embeddings.

        Args:
            support_embeddings: Tensor of shape (N*K, D) — encoded support samples.
            support_labels: Tensor of shape (N*K,) — labels in {0,...,N-1}.
            n_way: Number of classes (N).

        Returns:
            Prototypes tensor of shape (N, D).
        """
        prototypes = torch.zeros(
            n_way,
            support_embeddings.size(-1),
            device=support_embeddings.device,
            dtype=support_embeddings.dtype,
        )

        for i in range(n_way):
            mask = support_labels == i
            prototypes[i] = support_embeddings[mask].mean(dim=0)

        return prototypes

    def compute_distances(
        self,
        query_embeddings: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute squared Euclidean distances from queries to prototypes.

        Args:
            query_embeddings: Tensor of shape (N*Q, D).
            prototypes: Tensor of shape (N, D).

        Returns:
            Distance matrix of shape (N*Q, N) — squared Euclidean distances.
        """
        # Expand for broadcasting:
        # query: (N*Q, 1, D), prototypes: (1, N, D)
        # diff: (N*Q, N, D), distances: (N*Q, N)
        diff = query_embeddings.unsqueeze(1) - prototypes.unsqueeze(0)
        distances = (diff ** 2).sum(dim=-1)
        return distances

    def forward(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        query_features: torch.Tensor,
        n_way: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode, compute prototypes, classify queries.

        Args:
            support_features: Tensor of shape (N*K, input_dim) — support features.
            support_labels: Tensor of shape (N*K,) — support labels in {0,...,N-1}.
            query_features: Tensor of shape (N*Q, input_dim) — query features.
            n_way: Number of classes (N).

        Returns:
            log_probabilities: Tensor of shape (N*Q, N) — log-softmax over classes.
            predictions: Tensor of shape (N*Q,) — predicted class indices.
            distances: Tensor of shape (N*Q, N) — squared Euclidean distances.
        """
        # Encode support and query samples
        support_embeddings = self.encoder(support_features)
        query_embeddings = self.encoder(query_features)

        # Compute class prototypes (mean of support embeddings)
        prototypes = self.compute_prototypes(
            support_embeddings, support_labels, n_way
        )

        # Compute distances from queries to prototypes
        distances = self.compute_distances(query_embeddings, prototypes)

        # Negative distances → log-softmax for probabilities
        log_probs = F.log_softmax(-distances, dim=-1)

        # Predictions: nearest prototype
        predictions = distances.argmin(dim=-1)

        return log_probs, predictions, distances

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode features without classification (for SupCon loss).

        Args:
            features: Tensor of shape (batch_size, input_dim).

        Returns:
            Embeddings tensor of shape (batch_size, embedding_dim).
        """
        return self.encoder(features)

    def classify_with_prototypes(
        self,
        query_features: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classify queries using pre-computed prototypes.

        Useful for evaluation where prototypes are computed once from
        the support set and reused across queries.

        Args:
            query_features: Tensor of shape (Q, input_dim).
            prototypes: Tensor of shape (N, D).

        Returns:
            log_probabilities: Tensor of shape (Q, N).
            predictions: Tensor of shape (Q,).
        """
        query_embeddings = self.encoder(query_features)
        distances = self.compute_distances(query_embeddings, prototypes)
        log_probs = F.log_softmax(-distances, dim=-1)
        predictions = distances.argmin(dim=-1)
        return log_probs, predictions
