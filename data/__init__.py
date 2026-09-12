"""
Data pipeline for Geometry-Aware Metric Learning.

Modules:
    - representation: Raw, Angle, and RawAngle feature transforms
    - dataset: SplitLandmarkDataset for loading keypoint .npy files
    - episodes: EpisodicSampler for N-way K-shot episode generation
    - extract_keypoints: MediaPipe hand keypoint extraction from RGB images
"""

from data.representation import (
    RawRepresentation,
    AngleRepresentation,
    RawAngleRepresentation,
    get_representation,
)
from data.dataset import SplitLandmarkDataset
from data.episodes import EpisodicSampler, split_support_query

__all__ = [
    "RawRepresentation",
    "AngleRepresentation",
    "RawAngleRepresentation",
    "get_representation",
    "SplitLandmarkDataset",
    "EpisodicSampler",
    "split_support_query",
]
