"""
MLP Encoder for hand keypoint embedding.

A lightweight multi-layer perceptron that maps hand-keypoint feature vectors
to a 128-dimensional embedding space for Prototypical Network classification.

Architecture:
    Input (63/20/83-D)
    → Linear(hidden_dim=256) → BatchNorm1d → ReLU → Dropout(0.3)
    → Linear(hidden_dim=256) → BatchNorm1d → ReLU → Dropout(0.3)
    → Linear(embedding_dim=128)

Parameter counts (Table 1):
    raw (63-D input):       ~116,096 params
    angle (20-D input):     ~105,088 params
    raw_angle (83-D input): ~121,216 params

References:
    Section 3.3 — Model Architecture (MLP encoder)
"""

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """
    Two-layer MLP encoder with BatchNorm and Dropout.

    Maps a hand-keypoint feature vector to a 128-dimensional embedding.

    Args:
        input_dim: Input feature dimension (63 for raw, 20 for angle, 83 for raw_angle).
        embedding_dim: Output embedding dimension (default: 128).
        hidden_dim: Hidden layer dimension (default: 256).
        dropout: Dropout probability (default: 0.3).
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # Two hidden layers: Linear → BatchNorm1d → ReLU → Dropout
        self.encoder = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            # Layer 2
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            # Final projection to embedding space
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Embedding tensor of shape (batch_size, embedding_dim).
        """
        return self.encoder(x)

    def get_last_layer(self) -> nn.Linear:
        """
        Get the final linear projection layer.

        Used for target-supervised adaptation (fine-tuning only the last layer).
        """
        return self.encoder[-1]

    def freeze_except_last(self):
        """
        Freeze all parameters except the last linear layer.

        Used in target-supervised cross-lingual adaptation.
        """
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze last linear layer
        last_layer = self.get_last_layer()
        for param in last_layer.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
