"""
Test SO(3) invariance of the angle representation.

Verifies the theoretical prediction from Eq. (7): inter-joint angles are
invariant to rotation (SO(3)), translation, and isotropic scaling.

This test:
    1. Generates random hand keypoints
    2. Applies random similarity transforms (R, s, t)
    3. Verifies angle features remain unchanged (within numerical tolerance)

References:
    Section 3.2 — Geometry-Aware Angle Representation (Eq. 7)
    Section 4.6 — Normalisation Ablation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial.transform import Rotation

from data.representation import AngleRepresentation, RawRepresentation


def random_hand_keypoints(seed: int = 42) -> np.ndarray:
    """
    Generate realistic random hand keypoints of shape (21, 3).

    Uses a simplified kinematic model to generate anatomically plausible
    hand configurations.
    """
    rng = np.random.RandomState(seed)

    # Wrist at origin
    keypoints = np.zeros((21, 3), dtype=np.float32)

    # Finger chains with realistic bone lengths
    finger_chains = [
        [0, 1, 2, 3, 4],      # Thumb
        [0, 5, 6, 7, 8],      # Index
        [0, 9, 10, 11, 12],   # Middle
        [0, 13, 14, 15, 16],  # Ring
        [0, 17, 18, 19, 20],  # Pinky
    ]

    # Base directions for each finger (spread out)
    base_directions = [
        np.array([0.5, 0.8, 0.1]),   # Thumb
        np.array([0.2, 1.0, 0.0]),    # Index
        np.array([0.0, 1.0, 0.0]),    # Middle
        np.array([-0.15, 1.0, 0.0]),  # Ring
        np.array([-0.3, 0.95, 0.0]),  # Pinky
    ]

    bone_lengths = [0.03, 0.025, 0.02, 0.018]  # Approximate bone lengths

    for finger_idx, chain in enumerate(finger_chains):
        direction = base_directions[finger_idx].copy()
        direction /= np.linalg.norm(direction)

        for bone_idx in range(1, len(chain)):
            parent_pos = keypoints[chain[bone_idx - 1]]

            # Add small random perturbation to direction (simulates flexion)
            perturb = rng.randn(3) * 0.2
            current_dir = direction + perturb
            current_dir /= np.linalg.norm(current_dir)

            bone_len = bone_lengths[bone_idx - 1]
            keypoints[chain[bone_idx]] = parent_pos + current_dir * bone_len

    return keypoints


def random_rotation_matrix(seed: int = 0) -> np.ndarray:
    """Generate a random SO(3) rotation matrix."""
    return Rotation.random(random_state=seed).as_matrix()


def test_translation_invariance():
    """Test that angles are invariant to translation."""
    print("Test: Translation Invariance")
    angle_repr = AngleRepresentation()
    kp = random_hand_keypoints(seed=42)

    # Original angles
    angles_original = angle_repr(kp)

    # Apply random translations
    for i in range(10):
        rng = np.random.RandomState(i + 100)
        translation = rng.randn(3) * 100  # Large translation
        kp_translated = kp + translation

        angles_translated = angle_repr(kp_translated)
        diff = np.abs(angles_original - angles_translated)
        max_diff = diff.max()

        assert max_diff < 1e-5, (
            f"Translation invariance violated! Max diff: {max_diff:.2e}"
        )

    print(f"  PASSED — max diff < 1e-5 across 10 random translations")


def test_rotation_invariance():
    """Test that angles are invariant to SO(3) rotation."""
    print("Test: SO(3) Rotation Invariance")
    angle_repr = AngleRepresentation()
    kp = random_hand_keypoints(seed=42)

    angles_original = angle_repr(kp)

    for i in range(10):
        R = random_rotation_matrix(seed=i)
        kp_rotated = (R @ kp.T).T  # Apply rotation to all keypoints

        angles_rotated = angle_repr(kp_rotated.astype(np.float32))
        diff = np.abs(angles_original - angles_rotated)
        max_diff = diff.max()

        assert max_diff < 1e-5, (
            f"Rotation invariance violated! Max diff: {max_diff:.2e}"
        )

    print(f"  PASSED — max diff < 1e-5 across 10 random SO(3) rotations")


def test_isotropic_scaling_invariance():
    """Test that angles are invariant to isotropic scaling."""
    print("Test: Isotropic Scaling Invariance")
    angle_repr = AngleRepresentation()
    kp = random_hand_keypoints(seed=42)

    angles_original = angle_repr(kp)

    for scale in [0.01, 0.1, 0.5, 2.0, 10.0, 100.0, 1000.0]:
        kp_scaled = kp * scale

        angles_scaled = angle_repr(kp_scaled)
        diff = np.abs(angles_original - angles_scaled)
        max_diff = diff.max()

        assert max_diff < 1e-5, (
            f"Scaling invariance violated at scale={scale}! Max diff: {max_diff:.2e}"
        )

    print(f"  PASSED — max diff < 1e-5 across 7 scale factors")


def test_combined_similarity_transform():
    """Test invariance under full similarity transform: T(p) = sRp + t."""
    print("Test: Combined Similarity Transform (sRp + t)")
    angle_repr = AngleRepresentation()
    kp = random_hand_keypoints(seed=42)

    angles_original = angle_repr(kp)

    for i in range(20):
        rng = np.random.RandomState(i + 200)

        # Random rotation
        R = random_rotation_matrix(seed=i)

        # Random isotropic scale
        s = rng.uniform(0.01, 100.0)

        # Random translation
        t = rng.randn(3) * 50

        # Apply similarity transform: T(p) = sRp + t
        kp_transformed = (s * (R @ kp.T)).T + t

        angles_transformed = angle_repr(kp_transformed.astype(np.float32))
        diff = np.abs(angles_original - angles_transformed)
        max_diff = diff.max()

        assert max_diff < 1e-4, (
            f"Similarity invariance violated (trial {i})! "
            f"s={s:.4f}, max diff: {max_diff:.2e}"
        )

    print(f"  PASSED — max diff < 1e-4 across 20 random similarity transforms")


def test_raw_representation_changes():
    """Verify that raw coordinates DO change under transforms (control test)."""
    print("Test: Raw representation changes under transforms (control)")
    raw_repr = RawRepresentation(normalize=False)
    kp = random_hand_keypoints(seed=42)

    raw_original = raw_repr(kp)

    # Apply rotation
    R = random_rotation_matrix(seed=0)
    kp_rotated = (R @ kp.T).T
    raw_rotated = raw_repr(kp_rotated.astype(np.float32))

    diff = np.abs(raw_original - raw_rotated).max()
    assert diff > 0.001, (
        f"Raw features should change under rotation! Max diff: {diff:.2e}"
    )

    print(f"  PASSED — raw features change under rotation (max diff: {diff:.4f})")


def test_angle_dimensions():
    """Verify angle representation outputs exactly 20 dimensions."""
    print("Test: Angle Representation Dimensions")
    angle_repr = AngleRepresentation()
    kp = random_hand_keypoints(seed=42)

    angles = angle_repr(kp)
    assert angles.shape == (20,), f"Expected (20,), got {angles.shape}"
    assert angles.dtype == np.float32, f"Expected float32, got {angles.dtype}"

    print(f"  PASSED — output shape: {angles.shape}, dtype: {angles.dtype}")


def test_raw_dimensions():
    """Verify raw representation outputs exactly 63 dimensions."""
    print("Test: Raw Representation Dimensions")

    raw_repr = RawRepresentation(normalize=True)
    kp = random_hand_keypoints(seed=42)

    raw = raw_repr(kp)
    assert raw.shape == (63,), f"Expected (63,), got {raw.shape}"

    print(f"  PASSED — output shape: {raw.shape}")


def test_angles_in_valid_range():
    """Verify all angles are in [0, pi]."""
    print("Test: Angles in Valid Range [0, pi]")
    angle_repr = AngleRepresentation()

    for seed in range(10):
        kp = random_hand_keypoints(seed=seed)
        angles = angle_repr(kp)

        assert np.all(angles >= 0), f"Negative angle found: {angles.min()}"
        assert np.all(angles <= np.pi + 1e-6), f"Angle > pi found: {angles.max()}"

    print(f"  PASSED -- all angles in [0, pi] across 10 random hands")


if __name__ == "__main__":
    print("=" * 60)
    print("SO(3) Invariance Tests for Angle Representation")
    print("=" * 60)
    print()

    test_angle_dimensions()
    test_raw_dimensions()
    test_angles_in_valid_range()
    test_translation_invariance()
    test_rotation_invariance()
    test_isotropic_scaling_invariance()
    test_combined_similarity_transform()
    test_raw_representation_changes()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED [OK]")
    print("=" * 60)
