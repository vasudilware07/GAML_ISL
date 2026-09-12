"""
SplitLandmarkDataset — PyTorch Dataset for loading hand keypoint landmarks.

Loads per-sample .npy files (shape 21×3) with deterministic train/test splits
stored as JSON files. Applies representation transforms (raw/angle/raw_angle)
on the fly.

References:
    Section 3.5 — Implementation Details (Data pipeline)
"""

import json
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from data.representation import get_representation


class SplitLandmarkDataset(Dataset):
    """
    Dataset that loads hand keypoint landmarks with deterministic train/test splits.

    Each sample is a .npy file of shape (21, 3) storing 3D keypoint coordinates
    for one hand. A representation transform converts these into one of three
    formats: raw (63-D), angle (20-D), or raw_angle (83-D).

    Split files are JSON dictionaries mapping class names to lists of sample
    file paths, stored in the splits/ directory.

    Args:
        data_dir: Root directory containing per-class subdirectories of .npy files.
        split_file: Path to a JSON file defining the split (train or test).
        representation: Name of the representation ('raw', 'angle', 'raw_angle').
        normalize: Whether to apply normalization to raw coordinates.
        transform: Optional additional transform to apply after representation.

    Raises:
        AssertionError: If train and test splits overlap.
    """

    def __init__(
        self,
        data_dir: str,
        split_file: str,
        representation: str = "angle",
        normalize: bool = True,
        transform=None,
    ):
        self.data_dir = Path(data_dir)
        self.representation = get_representation(representation, normalize=normalize)
        self.transform = transform

        # Load split file
        with open(split_file, "r") as f:
            split_data = json.load(f)

        # Build sample list: (file_path, class_index)
        self.samples: List[Tuple[str, int]] = []
        self.class_names: List[str] = sorted(split_data.keys())
        self.class_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.class_names)
        }

        for class_name, file_list in split_data.items():
            class_idx = self.class_to_idx[class_name]
            for file_path in file_list:
                full_path = os.path.join(data_dir, file_path)
                if os.path.exists(full_path):
                    self.samples.append((full_path, class_idx))

        # Build class-to-samples index for efficient episodic sampling
        self.class_samples: Dict[int, List[int]] = {}
        for idx, (_, class_idx) in enumerate(self.samples):
            if class_idx not in self.class_samples:
                self.class_samples[class_idx] = []
            self.class_samples[class_idx].append(idx)

        self.num_classes = len(self.class_names)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        file_path, class_idx = self.samples[idx]

        # Load keypoints: shape (21, 3)
        keypoints = np.load(file_path).astype(np.float32)

        # Apply representation transform
        features = self.representation(keypoints)

        # Convert to tensor
        features = torch.from_numpy(features).float()

        if self.transform is not None:
            features = self.transform(features)

        return features, class_idx

    def get_class_samples(self, class_idx: int) -> List[int]:
        """Get all sample indices belonging to a class."""
        return self.class_samples.get(class_idx, [])

    def get_eligible_classes(self, min_samples: int) -> List[int]:
        """
        Get class indices with at least `min_samples` samples.

        Used to determine which classes are eligible for episodic sampling
        (need at least K + Q samples in test split).
        """
        return [
            cls_idx
            for cls_idx, sample_indices in self.class_samples.items()
            if len(sample_indices) >= min_samples
        ]

    @staticmethod
    def validate_no_overlap(train_file: str, test_file: str):
        """
        Validate that train and test splits have zero overlap.

        Args:
            train_file: Path to train split JSON.
            test_file: Path to test split JSON.

        Raises:
            AssertionError: If any sample appears in both splits.
        """
        with open(train_file, "r") as f:
            train_data = json.load(f)
        with open(test_file, "r") as f:
            test_data = json.load(f)

        train_files = set()
        for file_list in train_data.values():
            train_files.update(file_list)

        test_files = set()
        for file_list in test_data.values():
            test_files.update(file_list)

        overlap = train_files & test_files
        assert len(overlap) == 0, (
            f"Train/test overlap detected! {len(overlap)} samples appear in both splits."
        )

    def __repr__(self):
        return (
            f"SplitLandmarkDataset("
            f"data_dir={self.data_dir}, "
            f"num_samples={len(self)}, "
            f"num_classes={self.num_classes}, "
            f"representation={self.representation})"
        )
