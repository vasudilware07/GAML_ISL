"""
Spatial Graph Convolution encoder for hand landmarks (static images).
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Hand skeleton adjacency (MediaPipe 21 landmarks) ────────────────────────
# Landmark indices:
#   0: wrist
#   1-4: thumb (CMC→tip)
#   5-8: index finger
#   9-12: middle finger
#   13-16: ring finger
#   17-20: pinky finger

HAND_EDGES: List[Tuple[int, int]] = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm connections
    (5, 9), (9, 13), (13, 17),
]


def build_adjacency_matrix(num_nodes: int = 21, edges: List[Tuple[int, int]] = HAND_EDGES) -> torch.Tensor:
    """Build a symmetric normalised adjacency matrix (with self-loops).

    Args:
        num_nodes: Number of graph nodes.
        edges: List of ``(i, j)`` edge tuples.

    Returns:
        ``(num_nodes, num_nodes)`` normalised adjacency matrix.
    """
    A = torch.zeros(num_nodes, num_nodes)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    # Self-loops
    A = A + torch.eye(num_nodes)
    # Symmetric normalisation: D^{-1/2} A D^{-1/2}
    D = A.sum(dim=1)
    D_inv_sqrt = torch.diag(D.pow(-0.5))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm


class GraphConvLayer(nn.Module):
    """Single graph convolution layer: X' = σ(A_norm · X · W).

    Args:
        in_features: Input feature dimension per node.
        out_features: Output feature dimension per node.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: ``(B, V, C_in)``
            adj: ``(V, V)`` normalised adjacency.

        Returns:
            ``(B, V, C_out)``
        """
        # x: (B, V, C_in)  adj: (V, V)
        support = torch.matmul(x, self.weight)   # (B, V, C_out)
        out = torch.matmul(adj, support)          # (B, V, C_out)
        return out + self.bias


class GCNEncoder(nn.Module):
    """Spatial GCN encoder for static hand landmarks.

    Applies graph convolution on the hand skeleton, then pools across nodes.

    Input: ``(B, 21, 3)``
    Output: ``(B, embedding_dim)``

    Args:
        in_channels: Per-node input dim (3 for xyz).
        hidden_dim: GCN hidden width.
        embedding_dim: Final embedding size.
        num_layers: Number of GCN layers.
        dropout: Dropout rate.
        num_nodes: Number of graph nodes.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 128,
        embedding_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        num_nodes: int = 21,
    ) -> None:
        super().__init__()
        self.register_buffer("adj", build_adjacency_matrix(num_nodes))

        layers: List[nn.Module] = []
        prev_dim = in_channels
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else hidden_dim
            layers.append(GraphConvLayer(prev_dim, out_dim))
            prev_dim = out_dim
        self.gcn_layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(num_nodes) for _ in range(num_layers)])

        # Node pooling followed by projection
        self.fc = nn.Linear(hidden_dim * num_nodes, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(B, 21, 3)``

        Returns:
            ``(B, embedding_dim)``
        """
        B, V, C = x.shape

        for gcn, bn in zip(self.gcn_layers, self.bn_layers):
            x = gcn(x, self.adj)          # (B, V, H)
            x = bn(x)
            x = F.relu(x, inplace=True)
            x = self.dropout(x)

        # x: (B, V, H) → flatten nodes
        x = x.reshape(B, -1)             # (B, V*H)
        x = self.fc(x)                   # (B, embedding_dim)
        x = self.norm(x)
        return x


def build_gcn_encoder(cfg: dict, representation: str = "graph") -> GCNEncoder:
    """Factory to build a GCN encoder from config.

    Args:
        cfg: Full config dict.
        representation: Should be ``'graph'`` or ``'raw'``.

    Returns:
        Configured ``GCNEncoder``.
    """
    g_cfg = cfg["model"]["gcn"]
    return GCNEncoder(
        in_channels=3,
        hidden_dim=g_cfg["hidden_dim"],
        embedding_dim=cfg["model"]["embedding_dim"],
        num_layers=g_cfg["num_layers"],
        dropout=g_cfg["dropout"],
    )
