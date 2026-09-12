"""Losses package."""

from losses.triplet import TripletLoss
from losses.supcon import SupConLoss, ArcFaceLoss, build_loss

__all__ = ["TripletLoss", "SupConLoss", "ArcFaceLoss", "build_loss"]
