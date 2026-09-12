"""
Spatial Transformer Encoder for hand keypoint embedding.

For the raw representation, 21 keypoints are treated as a length-21 token
sequence with 3-D token embeddings, projected to d_model=128. For angle and
raw_angle inputs (where no natural per-keypoint tokenisation exists), the
full vector is treated as a single token, making self-attention degenerate
to a feedforward path.

Architecture (raw representation):
    21 keypoints × 3-D → Linear projection to d_model=128
    + Sinusoidal positional encodings
    → 2× TransformerEncoderLayer (4 heads, FFN=256, dropout=0.1)
    → Mean-pool across sequence → Linear → LayerNorm → 128-D

References:
    Section 3.3 — Model Architecture (Spatial Transformer encoder)
    Table 1 — Encoder parameter counts
"""

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.

    Adds position-dependent signals to token embeddings to encode
    sequential order. Used for the 21-keypoint token sequence.

    Args:
        d_model: Model dimension (128).
        max_len: Maximum sequence length (default: 100).
        dropout: Dropout probability (default: 0.1).
    """

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute sinusoidal positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor of same shape with positional encodings added.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    Spatial Transformer encoder for hand keypoints.

    For the 'raw' representation, treats 21 keypoints as a length-21 sequence
    of 3-D tokens, projects each to d_model, applies self-attention, and pools.
    For 'angle' and 'raw_angle', the full vector is a single token, so
    self-attention degenerates to a feedforward path.

    Args:
        input_dim: Input feature dimension (63 for raw, 20 for angle, 83 for raw_angle).
        embedding_dim: Output embedding dimension (default: 128).
        d_model: Internal model dimension (default: 128).
        n_heads: Number of attention heads (default: 4).
        n_layers: Number of Transformer encoder layers (default: 2).
        ffn_dim: Feed-forward network dimension (default: 256).
        dropout: Dropout probability (default: 0.1).
        representation: Representation type ('raw', 'angle', 'raw_angle').
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        representation: str = "raw",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.representation = representation

        if representation == "raw":
            # Tokenise: 21 keypoints, each 3-D → project to d_model
            self.token_dim = 3
            self.seq_len = 21
            self.input_projection = nn.Linear(self.token_dim, d_model)
            self.pos_encoding = SinusoidalPositionalEncoding(
                d_model=d_model, max_len=self.seq_len + 1, dropout=dropout
            )
        else:
            # angle (20-D) or raw_angle (83-D): single token
            self.token_dim = input_dim
            self.seq_len = 1
            self.input_projection = nn.Linear(self.token_dim, d_model)
            self.pos_encoding = None  # No positional encoding for single token

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Output projection with LayerNorm
        self.output_projection = nn.Linear(d_model, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Embedding tensor of shape (batch_size, embedding_dim).
        """
        batch_size = x.size(0)

        if self.representation == "raw":
            # Reshape to (batch_size, 21, 3) — tokenize as 21 keypoints
            x = x.view(batch_size, self.seq_len, self.token_dim)
        else:
            # Single token: (batch_size, 1, input_dim)
            x = x.unsqueeze(1)

        # Project tokens to d_model
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)

        # Add positional encoding (only for raw)
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)

        # Apply Transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)

        # Mean-pool across the sequence dimension
        x = x.mean(dim=1)  # (batch_size, d_model)

        # Output projection with LayerNorm
        x = self.output_projection(x)
        x = self.layer_norm(x)

        return x

    def get_last_layer(self) -> nn.Linear:
        """Get the final linear projection layer for target-supervised adaptation."""
        return self.output_projection

    def freeze_except_last(self):
        """Freeze all parameters except the output projection layer."""
        for param in self.parameters():
            param.requires_grad = False

        for param in self.output_projection.parameters():
            param.requires_grad = True
        for param in self.layer_norm.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
