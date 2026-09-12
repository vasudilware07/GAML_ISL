"""Models package for sign metric learning."""

from models.mlp_encoder import MLPEncoder, build_mlp_encoder
from models.temporal_transformer import TemporalTransformerEncoder, build_transformer_encoder
from models.gcn_encoder import GCNEncoder, build_gcn_encoder
from models.prototypical import PrototypicalNetwork
from models.siamese import SiameseNetwork, MatchingNetwork


def build_encoder(cfg: dict, representation: str = "raw"):
    """Build an encoder based on config.

    Args:
        cfg: Full config dict.
        representation: ``'raw'``, ``'pairwise'``, or ``'graph'``.

    Returns:
        Encoder module.
    """
    encoder_name = cfg["model"]["encoder"]
    if encoder_name == "mlp":
        return build_mlp_encoder(cfg, representation)
    elif encoder_name == "transformer":
        return build_transformer_encoder(cfg, representation)
    elif encoder_name == "gcn":
        return build_gcn_encoder(cfg, representation)
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}")


def build_few_shot_model(cfg: dict, encoder):
    """Build a few-shot model wrapping the given encoder.

    Args:
        cfg: Full config dict.
        encoder: Backbone encoder module.

    Returns:
        Few-shot model (PrototypicalNetwork, SiameseNetwork, or MatchingNetwork).
    """
    method = cfg["few_shot"]["method"]
    distance = cfg.get("distance", "euclidean")
    if method == "prototypical":
        return PrototypicalNetwork(encoder, distance=distance)
    elif method == "siamese":
        return SiameseNetwork(encoder, embedding_dim=cfg["model"]["embedding_dim"])
    elif method == "matching":
        return MatchingNetwork(encoder)
    else:
        raise ValueError(f"Unknown few-shot method: {method}")


def create_model(
    encoder_name: str,
    input_dim: int,
    embedding_dim: int = 128,
    hidden_dim: int = 256,
    dropout: float = 0.3,
    representation: str = "raw",
):
    """Factory that creates an encoder model by name.

    Args:
        encoder_name: ``'mlp'``, ``'transformer'``, or ``'gcn'``.
        input_dim: Input feature dimensionality.
        embedding_dim: Output embedding size.
        hidden_dim: Hidden layer width (MLP) or d_model (Transformer).
        dropout: Dropout probability.
        representation: Feature representation name.

    Returns:
        Encoder module (nn.Module).
    """
    cfg = {
        "dataset": {"sequence_length": None},
        "model": {
            "encoder": encoder_name,
            "input_dim": input_dim,
            "embedding_dim": embedding_dim,
            "mlp": {
                "hidden_dims": [hidden_dim, hidden_dim],
                "dropout": dropout,
            },
            "transformer": {
                "num_heads": 4,
                "num_layers": 2,
                "dim_feedforward": hidden_dim,
                "dropout": dropout,
            },
            "gcn": {
                "hidden_dims": [hidden_dim, hidden_dim],
                "dropout": dropout,
            },
        },
    }
    if encoder_name == "mlp":
        return build_mlp_encoder(cfg, representation)
    elif encoder_name == "transformer":
        return build_transformer_encoder(cfg, representation)
    elif encoder_name == "gcn":
        return build_gcn_encoder(cfg, representation)
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}")



def count_parameters(model) -> int:
    """Return the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "MLPEncoder", "build_mlp_encoder",
    "TemporalTransformerEncoder", "build_transformer_encoder",
    "GCNEncoder", "build_gcn_encoder",
    "PrototypicalNetwork", "SiameseNetwork", "MatchingNetwork",
    "build_encoder", "build_few_shot_model",
    "create_model", "count_parameters",
]
