"""
MediaPipe hand keypoint extraction from RGB images.

Processes raw RGB images through MediaPipe Hands v0.10 to extract
21 three-dimensional hand landmarks per image. Saves each as a .npy
file of shape (21, 3). Images where no hand is detected are discarded.

Also generates deterministic stratified 70/30 train/test splits.

Usage:
    python data/extract_keypoints.py \
        --input_dir path/to/dataset \
        --output_dir data/landmarks/asl \
        --splits_dir splits \
        --dataset asl \
        --seed 42

References:
    Section 3.1 — Keypoints and Raw-Coordinate Preprocessing
    Section 3.5 — Implementation Details (Data pipeline)
    MediaPipe Hands [Zhang et al., 2020]
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def extract_hand_keypoints(
    image_path: str,
    hands_detector,
) -> Optional[np.ndarray]:
    """
    Extract 21 hand keypoints from an RGB image using MediaPipe Hands.

    Args:
        image_path: Path to the RGB image file.
        hands_detector: MediaPipe Hands detector instance.

    Returns:
        Array of shape (21, 3) with (x, y, z) coordinates, or None if
        no hand is detected.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    # Convert BGR to RGB (MediaPipe expects RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process image
    results = hands_detector.process(image_rgb)

    if results.multi_hand_landmarks is None:
        return None

    # Take the first detected hand
    hand_landmarks = results.multi_hand_landmarks[0]

    # Extract 21 keypoints as (x, y, z)
    keypoints = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )
    assert keypoints.shape == (21, 3)

    return keypoints


def process_dataset(
    input_dir: str,
    output_dir: str,
    max_hands: int = 1,
    static_image_mode: bool = True,
    min_detection_confidence: float = 0.5,
) -> Dict[str, List[str]]:
    """
    Process an entire dataset of RGB images into .npy keypoint files.

    Expected input structure:
        input_dir/
            class_1/
                image_001.jpg
                image_002.jpg
                ...
            class_2/
                ...

    Output structure:
        output_dir/
            class_1/
                image_001.npy
                image_002.npy
                ...

    Args:
        input_dir: Root directory with per-class image subdirectories.
        output_dir: Root directory for output .npy files.
        max_hands: Maximum number of hands to detect per image.
        static_image_mode: Whether to treat each image independently.
        min_detection_confidence: Minimum detection confidence threshold.

    Returns:
        Dictionary mapping class names to lists of successfully processed
        file paths (relative to output_dir).
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_hands,
        min_detection_confidence=min_detection_confidence,
    )

    class_files: Dict[str, List[str]] = {}
    total_processed = 0
    total_skipped = 0

    # Get all class directories
    class_dirs = sorted([
        d for d in input_path.iterdir()
        if d.is_dir()
    ])

    print(f"Found {len(class_dirs)} classes in {input_dir}")

    for class_dir in tqdm(class_dirs, desc="Processing classes"):
        class_name = class_dir.name
        class_output = output_path / class_name
        class_output.mkdir(parents=True, exist_ok=True)

        class_files[class_name] = []

        # Process all images in this class
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = sorted([
            f for f in class_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ])

        for img_file in image_files:
            keypoints = extract_hand_keypoints(str(img_file), hands)

            if keypoints is not None:
                # Save as .npy
                npy_filename = img_file.stem + ".npy"
                npy_path = class_output / npy_filename
                np.save(str(npy_path), keypoints)

                # Store relative path
                rel_path = os.path.join(class_name, npy_filename)
                class_files[class_name].append(rel_path)
                total_processed += 1
            else:
                total_skipped += 1

    hands.close()

    print(f"\nExtraction complete:")
    print(f"  Processed: {total_processed}")
    print(f"  Skipped (no hand detected): {total_skipped}")
    print(f"  Classes: {len(class_files)}")

    return class_files


def generate_splits(
    class_files: Dict[str, List[str]],
    output_dir: str,
    dataset_name: str,
    train_ratio: float = 0.7,
    seed: int = 42,
) -> Tuple[str, str]:
    """
    Generate deterministic stratified train/test splits.

    Args:
        class_files: Dictionary mapping class names to file path lists.
        output_dir: Directory to save split JSON files.
        dataset_name: Name of the dataset (for file naming).
        train_ratio: Fraction of data for training.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_split_path, test_split_path).
    """
    os.makedirs(output_dir, exist_ok=True)

    train_split: Dict[str, List[str]] = {}
    test_split: Dict[str, List[str]] = {}

    for class_name, files in sorted(class_files.items()):
        if len(files) < 2:
            # Too few samples to split; skip class
            print(f"  Warning: class '{class_name}' has only {len(files)} samples, skipping.")
            continue

        train_files, test_files = train_test_split(
            files,
            train_size=train_ratio,
            random_state=seed,
            shuffle=True,
        )
        train_split[class_name] = sorted(train_files)
        test_split[class_name] = sorted(test_files)

    # Save splits
    train_path = os.path.join(output_dir, f"{dataset_name}_train.json")
    test_path = os.path.join(output_dir, f"{dataset_name}_test.json")

    with open(train_path, "w") as f:
        json.dump(train_split, f, indent=2)
    with open(test_path, "w") as f:
        json.dump(test_split, f, indent=2)

    # Print statistics
    total_train = sum(len(v) for v in train_split.values())
    total_test = sum(len(v) for v in test_split.values())

    print(f"\nSplit statistics ({dataset_name}):")
    print(f"  Train: {total_train} samples across {len(train_split)} classes")
    print(f"  Test:  {total_test} samples across {len(test_split)} classes")
    print(f"  Saved: {train_path}, {test_path}")

    # Validate no overlap
    train_set = set()
    for files in train_split.values():
        train_set.update(files)
    test_set = set()
    for files in test_split.values():
        test_set.update(files)
    assert len(train_set & test_set) == 0, "Train/test overlap detected!"

    return train_path, test_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe hand keypoints from RGB images."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root directory with per-class image subdirectories.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for .npy keypoint files.",
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default="splits",
        help="Directory for train/test split JSON files.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["asl", "libras", "arabic", "thai"],
        help="Dataset name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.5,
        help="Minimum MediaPipe detection confidence.",
    )

    args = parser.parse_args()

    # Step 1: Extract keypoints
    print(f"Extracting keypoints from {args.input_dir}...")
    class_files = process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_detection_confidence=args.min_confidence,
    )

    # Step 2: Generate splits
    print(f"\nGenerating {args.train_ratio:.0%}/{1 - args.train_ratio:.0%} splits...")
    generate_splits(
        class_files=class_files,
        output_dir=args.splits_dir,
        dataset_name=args.dataset,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
