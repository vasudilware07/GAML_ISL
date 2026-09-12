"""
Supervised Contrastive Loss and ArcFace Loss.

Reference:
    Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
    Deng et al., "ArcFace: Additive Angular Margin Loss", CVPR 2019.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss.

    For each anchor, positives are other samples with the same label and
    negatives are samples with different labels. Supports batch mining
    naturally through label-based masking.

    Args:
        temperature: Scaling temperature τ.
        base_temperature: Denominator temperature for normalisation.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute supervised contrastive loss.

        Args:
            embeddings: ``(N, D)`` L2-normalised embeddings.
            labels: ``(N,)`` integer labels.

        Returns:
            Scalar loss.
        """
        device = embeddings.device
        N = embeddings.size(0)

        # L2 normalise
        embeddings = F.normalize(embeddings, dim=-1)

        # Similarity matrix
        sim = embeddings @ embeddings.t() / self.temperature  # (N, N)

        # Mask: same class (exclude self)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.t()).float().to(device)  # (N, N)
        self_mask = torch.eye(N, device=device)
        mask = mask - self_mask  # remove diagonal

        # For numerical stability
        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        # Exclude self from denominator
        exp_logits = torch.exp(logits) * (1 - self_mask)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Mean of log-likelihood over positive pairs
        num_positives = mask.sum(dim=1)
        mean_log_prob = (mask * log_prob).sum(dim=1) / (num_positives + 1e-8)

        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob
        # Only compute over anchors that have at least one positive
        valid = num_positives > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return loss[valid].mean()


class ArcFaceLoss(nn.Module):
    """ArcFace (Additive Angular Margin) Loss.

    Adds an angular margin penalty to the target logit in a normalised
    softmax classifier.

    Args:
        embedding_dim: Input embedding dimensionality.
        num_classes: Number of classes.
        scale: Feature re-scaling factor *s*.
        margin: Angular margin *m* in radians.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_classes: int = 100,
        scale: float = 30.0,
        margin: float = 0.50,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ArcFace loss.

        Args:
            embeddings: ``(N, D)`` embeddings (not necessarily normalised).
            labels: ``(N,)`` integer class labels.

        Returns:
            Scalar cross-entropy loss with angular margin.
        """
        # Normalise embeddings and weight
        x = F.normalize(embeddings, dim=-1)
        W = F.normalize(self.weight, dim=-1)

        cosine = x @ W.t()  # (N, C)
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(min=1e-8))

        # cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Numerical safety
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot
        one_hot = F.one_hot(labels, num_classes=self.weight.size(0)).float()

        logits = (one_hot * phi + (1.0 - one_hot) * cosine) * self.scale
        return F.cross_entropy(logits, labels)


def build_loss(cfg: dict) -> nn.Module:
    """Build a loss function from config.

    Args:
        cfg: Full config dict.

    Returns:
        Loss module.
    """
    name = cfg["loss"]["name"]
    if name == "triplet":
        return TripletLossWrapper(
            margin=cfg["loss"]["triplet"]["margin"],
            mining=cfg["loss"]["triplet"]["mining"],
        )
    elif name == "supcon":
        return SupConLoss(temperature=cfg["loss"]["supcon"]["temperature"])
    elif name == "arcface":
        return ArcFaceLoss(
            embedding_dim=cfg["model"]["embedding_dim"],
            num_classes=cfg["dataset"]["num_classes"],
            scale=cfg["loss"]["arcface"]["scale"],
            margin=cfg["loss"]["arcface"]["margin"],
        )
    else:
        raise ValueError(f"Unknown loss: {name}")


class TripletLossWrapper(nn.Module):
    """Thin wrapper to import TripletLoss consistently."""

    def __init__(self, margin: float = 0.5, mining: str = "semi_hard") -> None:
        super().__init__()
        from losses.triplet import TripletLoss
        self.loss_fn = TripletLoss(margin=margin, mining=mining)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(embeddings, labels)
