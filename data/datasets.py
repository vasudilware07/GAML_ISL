"""
PyTorch dataset classes for preprocessed hand-landmark data (static images).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from data.preprocess import compute_pairwise_distances, compute_joint_angles, compute_raw_angle

# ── Default splits directory (repo-root / splits/) ──────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = _REPO_ROOT / "splits"


def load_split_json(dataset_name: str, split: str) -> Dict[str, List[str]]:
    """Load a split JSON produced by ``tools/make_splits.py``.

    Args:
        dataset_name: e.g. ``'asl_alphabet'``.
        split: ``'train'`` or ``'test'``.

    Returns:
        Dict mapping class name → list of relative paths.

    Raises:
        RuntimeError: if the JSON file is missing or invalid.
    """
    path = SPLITS_DIR / f"{dataset_name}_{split}.json"
    if not path.exists():
        raise RuntimeError(
            f"Missing split file: {path}\n"
            f"Run:  python tools/make_splits.py --dataset {dataset_name} --seed 42 --ratio 0.7"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Integrity: no duplicates within the split
    all_paths = [p for paths in data.values() for p in paths]
    if len(all_paths) != len(set(all_paths)):
        raise RuntimeError(f"Duplicate paths detected inside {path}")
    return data


def validate_no_leak(dataset_name: str) -> None:
    """Assert train/test splits for *dataset_name* have zero overlap."""
    train_path = SPLITS_DIR / f"{dataset_name}_train.json"
    test_path = SPLITS_DIR / f"{dataset_name}_test.json"
    if not train_path.exists() or not test_path.exists():
        return  # nothing to validate
    with open(train_path) as f:
        train_data = json.load(f)
    with open(test_path) as f:
        test_data = json.load(f)
    train_set = {p for paths in train_data.values() for p in paths}
    test_set = {p for paths in test_data.values() for p in paths}
    overlap = train_set & test_set
    if overlap:
        raise RuntimeError(
            f"[{dataset_name}] Data leak! {len(overlap)} paths in BOTH train and test. "
            f"First: {list(overlap)[:3]}"
        )


class LandmarkDataset(Dataset):
    """Dataset that loads ``.npy`` hand-landmark files (static images).

    Folder layout::

        root/
            class_0/
                sample1.npy   # shape (21, 3)
                sample2.npy
            class_1/
                ...

    Args:
        root: Root directory of preprocessed ``.npy`` files.
        representation: One of ``'raw'``, ``'pairwise'``, ``'graph'``.
        transform: Optional callable applied to the tensor.
    """

    def __init__(
        self,
        root: str,
        representation: str = "raw",
        transform=None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.representation = representation
        self.transform = transform

        self.samples: List[Tuple[str, int]] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.classes: List[str] = []

        self._scan()

    def _scan(self) -> None:
        """Discover classes and samples from the directory tree."""
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        class_dirs = sorted([d for d in self.root.iterdir() if d.is_dir()])
        for idx, d in enumerate(class_dirs):
            cls_name = d.name
            self.classes.append(cls_name)
            self.class_to_idx[cls_name] = idx
            self.idx_to_class[idx] = cls_name
            for npy in sorted(d.glob("*.npy")):
                self.samples.append((str(npy), idx))

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        lm = np.load(path).astype(np.float32)  # (21, 3)

        # Handle legacy (T, 21, 3) files — take first frame
        if lm.ndim == 3:
            lm = lm[0]

        if self.representation == "pairwise":
            lm = compute_pairwise_distances(lm)  # (210,)
        elif self.representation == "angle":
            lm = compute_joint_angles(lm)          # (20,)
        elif self.representation == "raw_angle":
            lm = compute_raw_angle(lm)              # (83,)
        # raw / graph: kept as (21, 3)

        tensor = torch.from_numpy(lm)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, label


class SyntheticLandmarkDataset(Dataset):
    """Synthetic dataset for testing and demo purposes.

    Generates random landmark data for *num_classes* classes, where
    each class is centred at a distinct random prototype (single static image).

    Args:
        num_classes: Number of classes.
        samples_per_class: Samples generated per class.
        representation: ``'raw'`` | ``'pairwise'`` | ``'graph'``.
        noise_std: Standard deviation of Gaussian noise around prototypes.
    """

    def __init__(
        self,
        num_classes: int = 100,
        samples_per_class: int = 20,
        representation: str = "raw",
        noise_std: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.representation = representation
        self.noise_std = noise_std

        # Create class prototypes — single frame (21, 3)
        rng = np.random.RandomState(0)
        self.prototypes = [
            rng.randn(21, 3).astype(np.float32)
            for _ in range(num_classes)
        ]

        # Build sample list
        self.samples: List[Tuple[np.ndarray, int]] = []
        self.classes = [str(i) for i in range(num_classes)]
        self.class_to_idx = {str(i): i for i in range(num_classes)}
        self.idx_to_class = {i: str(i) for i in range(num_classes)}

        for cls_idx in range(num_classes):
            proto = self.prototypes[cls_idx]
            for _ in range(samples_per_class):
                sample = proto + rng.randn(*proto.shape).astype(np.float32) * noise_std
                self.samples.append((sample, cls_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        lm, label = self.samples[index]
        lm = lm.copy()

        if self.representation == "pairwise":
            lm = compute_pairwise_distances(lm)  # (210,)
        elif self.representation == "angle":
            lm = compute_joint_angles(lm)          # (20,)
        elif self.representation == "raw_angle":
            lm = compute_raw_angle(lm)              # (83,)

        tensor = torch.from_numpy(lm)
        return tensor, label

class SplitLandmarkDataset(LandmarkDataset):
    """LandmarkDataset that reads samples from a JSON split file.

    The JSON file is produced by ``tools/make_splits.py`` and contains::

        {"class_name": ["cls/file1.npy", "cls/file2.npy", ...], ...}

    Paths inside the JSON are **relative to the dataset root**.

    Args:
        dataset_name: Logical dataset name (e.g. ``'asl_alphabet'``).
        split: ``'train'`` or ``'test'``.
        data_root: Root of the flat preprocessed folder
                   (e.g. ``data/processed/asl_alphabet``).
        representation: One of ``'raw'``, ``'pairwise'``, ``'angle'``,
                        ``'raw_angle'``, ``'graph'``.
        transform: Optional callable applied to the tensor.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        data_root: str,
        representation: str = "raw",
        transform=None,
    ) -> None:
        self._dataset_name = dataset_name
        self._split = split
        # Validate split names early
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got '{split}'")
        # Do NOT call super().__init__() yet — we need to set these first
        # so _scan() can use them
        self._data_root = Path(data_root)
        # Now call parent init which will call _scan()
        super().__init__(root=data_root, representation=representation, transform=transform)
        # After scanning, validate no data leak
        validate_no_leak(dataset_name)

    def _scan(self) -> None:
        """Override: load samples from the JSON split file."""
        split_data = load_split_json(self._dataset_name, self._split)
        self.classes = sorted(split_data.keys())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes)}
        self.samples = []
        missing = []
        for cls_name in self.classes:
            idx = self.class_to_idx[cls_name]
            for rel_path in sorted(split_data[cls_name]):
                full = self._data_root / rel_path
                if not full.exists():
                    missing.append(str(full))
                    continue
                self.samples.append((str(full), idx))
        if missing:
            raise RuntimeError(
                f"[{self._dataset_name}/{self._split}] {len(missing)} files listed "
                f"in split JSON are missing on disk. First 5:\n"
                + "\n".join(missing[:5])
            )