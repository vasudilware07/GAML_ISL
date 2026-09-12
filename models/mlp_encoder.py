"""
MLP-based landmark encoder.
"""

from typing import List

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """Multi-layer perceptron that encodes flattened hand landmarks.

    Input shape:
        ``(B, 21, 3)`` for raw landmarks → flattened to ``(B, 63)``
        ``(B, 210)`` for pairwise distances → kept as ``(B, 210)``

    Args:
        input_dim: Flattened input dimensionality.
        hidden_dims: List of hidden layer widths.
        embedding_dim: Output embedding size.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int = 21 * 3,
        hidden_dims: List[int] = [256, 256],
        embedding_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, embedding_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, 21, 3)`` or ``(B, D)``.

        Returns:
            Embedding tensor of shape ``(B, embedding_dim)``.
        """
        b = x.size(0)
        x = x.reshape(b, -1)
        return self.net(x)


def build_mlp_encoder(cfg: dict, representation: str = "raw") -> MLPEncoder:
    """Factory function to create an MLP encoder from config.

    Args:
        cfg: Full configuration dictionary.
        representation: ``'raw'``, ``'pairwise'``, or ``'graph'``.

    Returns:
        Configured ``MLPEncoder`` instance.
    """
    seq_len = cfg["dataset"].get("sequence_length", None)
    if representation == "pairwise":
        input_dim = 210
    elif representation == "angle":
        input_dim = 20
    elif representation == "raw_angle":
        input_dim = 83
    else:
        input_dim = 21 * 3  # 63

    model_cfg = cfg["model"]["mlp"]
    return MLPEncoder(
        input_dim=input_dim,
        hidden_dims=model_cfg["hidden_dims"],
        embedding_dim=cfg["model"]["embedding_dim"],
        dropout=model_cfg["dropout"],
    )
