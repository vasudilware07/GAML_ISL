"""
Model factory and utilities.

Routes encoder and model construction by name, providing a unified interface
for creating encoder + Prototypical Network combinations.

References:
    Section 3.3 — Model Architecture
    Table 1 — Encoder parameter counts
"""

from typing import Optional, Tuple

import torch.nn as nn

from models.mlp_encoder import MLPEncoder
from models.temporal_transformer import TransformerEncoder
from models.prototypical import PrototypicalNetwork


def create_encoder(
    encoder_name: str,
    input_dim: int,
    embedding_dim: int = 128,
    hidden_dim: int = 256,
    dropout: float = 0.3,
    # Transformer-specific
    n_heads: int = 4,
    n_layers: int = 2,
    transformer_dropout: float = 0.1,
    representation: str = "raw",
) -> nn.Module:
    """
    Create an encoder by name.

    Args:
        encoder_name: 'mlp' or 'transformer'.
        input_dim: Input feature dimension (63/20/83).
        embedding_dim: Output embedding dimension (128).
        hidden_dim: Hidden layer dimension (256).
        dropout: Dropout rate for MLP (0.3).
        n_heads: Number of attention heads for Transformer (4).
        n_layers: Number of Transformer encoder layers (2).
        transformer_dropout: Dropout rate for Transformer (0.1).
        representation: Representation name ('raw', 'angle', 'raw_angle').

    Returns:
        Encoder module mapping input_dim → embedding_dim.
    """
    if encoder_name == "mlp":
        return MLPEncoder(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    elif encoder_name == "transformer":
        return TransformerEncoder(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            d_model=embedding_dim,  # d_model = embedding_dim = 128
            n_heads=n_heads,
            n_layers=n_layers,
            ffn_dim=hidden_dim,
            dropout=transformer_dropout,
            representation=representation,
        )
    else:
        raise ValueError(
            f"Unknown encoder '{encoder_name}'. Choose from: 'mlp', 'transformer'"
        )


def create_model(
    encoder_name: str,
    input_dim: int,
    embedding_dim: int = 128,
    hidden_dim: int = 256,
    dropout: float = 0.3,
    n_heads: int = 4,
    n_layers: int = 2,
    transformer_dropout: float = 0.1,
    representation: str = "raw",
    distance: str = "euclidean",
) -> PrototypicalNetwork:
    """
    Create a complete Prototypical Network model with the specified encoder.

    Args:
        encoder_name: 'mlp' or 'transformer'.
        input_dim: Input feature dimension.
        embedding_dim: Output embedding dimension.
        hidden_dim: Hidden layer dimension.
        dropout: Dropout rate.
        n_heads: Number of attention heads (Transformer).
        n_layers: Number of layers (Transformer).
        transformer_dropout: Dropout (Transformer).
        representation: Representation name.
        distance: Distance metric for ProtoNet ('euclidean').

    Returns:
        PrototypicalNetwork instance.
    """
    encoder = create_encoder(
        encoder_name=encoder_name,
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        n_heads=n_heads,
        n_layers=n_layers,
        transformer_dropout=transformer_dropout,
        representation=representation,
    )

    return PrototypicalNetwork(
        encoder=encoder,
        distance=distance,
    )


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "create_encoder",
    "create_model",
    "count_parameters",
    "MLPEncoder",
    "TransformerEncoder",
    "PrototypicalNetwork",
]
