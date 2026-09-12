"""
Geometry-aware hand keypoint representations.

Implements three representations derived from 21 MediaPipe hand keypoints:
    1. RawRepresentation  — wrist-centred, scale-normalised, flattened to 63-D
    2. AngleRepresentation — 20 SO(3)-invariant inter-joint angles (20-D)
    3. RawAngleRepresentation — concatenation of both (83-D)

The angle representation is provably invariant to rotation, translation,
and isotropic scaling (Eq. 7 in the paper), making it intrinsically
portable across datasets captured under different camera configurations.

References:
    Section 3.1 — Keypoints and Raw-Coordinate Preprocessing
    Section 3.2 — Geometry-Aware Angle Representation
"""

import numpy as np
import torch
from typing import Union

# =============================================================================
# MediaPipe Hand Topology — Kinematic Chains
# =============================================================================
# Keypoint 0 = wrist (root)
# Each finger is a 4-joint chain rooted at the wrist:
#   Thumb:  0 → 1 → 2 → 3 → 4
#   Index:  0 → 5 → 6 → 7 → 8
#   Middle: 0 → 9 → 10 → 11 → 12
#   Ring:   0 → 13 → 14 → 15 → 16
#   Pinky:  0 → 17 → 18 → 19 → 20

# Parent map: parent[k] gives the parent of keypoint k in the tree
PARENT_MAP = {
    # Thumb
    1: 0, 2: 1, 3: 2, 4: 3,
    # Index
    5: 0, 6: 5, 7: 6, 8: 7,
    # Middle
    9: 0, 10: 9, 11: 10, 12: 11,
    # Ring
    13: 0, 14: 13, 15: 14, 16: 15,
    # Pinky
    17: 0, 18: 17, 19: 18, 20: 19,
}

# 20 anatomical triplets (parent, pivot/joint, child)
# Each non-wrist keypoint k defines a triplet where:
#   - a_k = parent of the pivot's parent (grandparent) OR parent of pivot
#   - j_k = the pivot joint
#   - c_k = the child
#
# Following the paper: for each finger chain [r, a, b, c, d] with r=wrist,
# we get 4 triplets per finger = 20 total:
#   (r, a, b), (a, b, c), (b, c, d), and for the tip: (c, d, *) — but tips
#   have no child.
#
# The paper states exactly 20 triplets for 20 non-wrist keypoints. The most
# consistent interpretation: each non-wrist keypoint k serves as the CHILD
# in a triplet (grandparent, parent, k), measuring the angle at parent(k)
# between the bone from grandparent→parent and parent→k.
#
# For k where parent(k) = 0 (wrist), the grandparent doesn't exist in the
# skeleton. We use a reference: the wrist itself connects to all 5 finger
# bases, so we measure the angle at the wrist between the wrist→finger_base
# direction and finger_base→next_joint direction, which is equivalent to
# triplet (0, finger_base, next_joint) — but that's only for internal joints.
#
# FINAL IMPLEMENTATION: 20 triplets, one per non-wrist keypoint k.
# For each k, triplet = (parent(parent(k)), parent(k), k) when grandparent exists.
# For k ∈ {1,5,9,13,17} (finger bases, parent=wrist=0), we use triplet (0, k, child(k)).

# Build child map from parent map
CHILD_MAP = {}
for child, parent in PARENT_MAP.items():
    if parent not in CHILD_MAP:
        CHILD_MAP[parent] = []
    CHILD_MAP[parent].append(child)

# Sort children for deterministic ordering
for parent in CHILD_MAP:
    CHILD_MAP[parent].sort()


def _build_angle_triplets():
    """
    Build 20 anatomical triplets for inter-joint angle computation.

    Returns a list of 20 tuples (a, j, c) where:
        a = parent/ancestor keypoint index
        j = pivot/joint keypoint index (where the angle is measured)
        c = child keypoint index

    The angle θ at joint j is computed between vectors (a→j) and (j→c),
    equivalently between vectors (j→a) and (j→c) via:
        θ = arccos( (a-j)·(c-j) / (||a-j|| · ||c-j||) )
    """
    triplets = []

    # Finger chains (each includes the wrist as root)
    finger_chains = [
        [0, 1, 2, 3, 4],      # Thumb
        [0, 5, 6, 7, 8],      # Index
        [0, 9, 10, 11, 12],   # Middle
        [0, 13, 14, 15, 16],  # Ring
        [0, 17, 18, 19, 20],  # Pinky
    ]

    for chain in finger_chains:
        # For a chain [0, a, b, c, d], we get 4 triplets:
        #   Angle at a: (0, a, b)   — 1st joint from wrist
        #   Angle at b: (a, b, c)   — 2nd joint
        #   Angle at c: (b, c, d)   — 3rd joint
        #   Angle at d: (c, d, ?)   — fingertip (no child)
        #
        # Since fingertips have no child, we get 3 triplets per finger = 15.
        # To reach 20 (one per non-wrist keypoint), we also measure the
        # angle at each fingertip using the last two bones reflected:
        #   For tip d: measure angle at d between (c→d) direction and the
        #   bone (b→c) direction "through" d. Equivalently: (c, d, d + (d-c))
        #   But this is artificial.
        #
        # Alternative: 4 consecutive-triplet angles per chain (overlapping)
        # Chain has 5 nodes → 3 consecutive triplets from internal nodes
        # Plus the angle between the chain direction and the "spread" direction
        #
        # The paper says "Each non-wrist keypoint has a unique parent...
        # producing 20 anatomical triplets (a_k, j_k, c_k)"
        # where a_k = parent, j_k = pivot, c_k = child.
        # This means each non-wrist keypoint k is the PIVOT, and we need
        # both parent(k) and child(k). For fingertips, child(k) doesn't exist.
        #
        # RESOLUTION: We interpret "20 anatomical triplets" as measuring
        # the angle at each of the 20 non-wrist keypoints where possible.
        # For joints with both parent and child (15 joints): standard triplet.
        # For fingertips (5 joints, no child): we use the angle between
        # the incoming bone and the line from the tip to the palm center,
        # approximated as the wrist.
        # This gives exactly 20 triplets.
        #
        # Actually simplest: for fingertips, the "child" direction is
        # extrapolated from the last bone. The angle would be π (straight).
        # Instead, let's measure the angle at the tip between parent→tip
        # and tip→wrist, giving a "curl" angle.
        for i in range(1, len(chain)):
            pivot = chain[i]
            parent = chain[i - 1]
            if i < len(chain) - 1:
                # Internal joint: standard triplet
                child = chain[i + 1]
            else:
                # Fingertip: use wrist as reference to measure finger curl
                # This gives a meaningful geometric feature (how much the
                # finger points toward/away from the wrist)
                child = 0
            triplets.append((parent, pivot, child))

    return triplets


# Pre-compute the 20 anatomical triplets
ANGLE_TRIPLETS = _build_angle_triplets()
assert len(ANGLE_TRIPLETS) == 20, f"Expected 20 triplets, got {len(ANGLE_TRIPLETS)}"


# =============================================================================
# Representation Classes
# =============================================================================

class RawRepresentation:
    """
    Raw coordinate representation (63-D).

    Steps (Section 3.1):
        1. Translation invariance: subtract wrist position (p0) from all keypoints
        2. Scale invariance: divide by maximum pairwise distance
        3. Flatten to 63-D vector (21 keypoints × 3 coordinates)

    Args:
        normalize: If True, apply wrist-centering and scale normalization.
                   If False, just flatten raw coordinates (for ablation study).
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.dim = 63  # 21 × 3

    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Transform keypoints to raw representation.

        Args:
            keypoints: Array of shape (21, 3) — 21 keypoints with (x, y, z) coords.

        Returns:
            Raw feature vector of shape (63,).
        """
        assert keypoints.shape == (21, 3), f"Expected (21, 3), got {keypoints.shape}"
        kp = keypoints.copy().astype(np.float32)

        if self.normalize:
            # Step 1: Wrist-centring (Eq. 1)
            wrist = kp[0].copy()
            kp = kp - wrist  # ˆpi = pi − p0

            # Step 2: Scale normalisation (Eq. 2)
            # Divide by maximum pairwise distance
            max_dist = 0.0
            for i in range(21):
                for j in range(i + 1, 21):
                    dist = np.linalg.norm(kp[i] - kp[j])
                    if dist > max_dist:
                        max_dist = dist

            if max_dist > 1e-8:
                kp = kp / max_dist

        # Step 3: Flatten to 63-D (Eq. 3)
        return kp.flatten()

    def __repr__(self):
        return f"RawRepresentation(normalize={self.normalize}, dim={self.dim})"


class AngleRepresentation:
    """
    Geometry-aware angle representation (20-D).

    Computes 20 inter-joint angles from anatomical triplets defined by the
    MediaPipe hand skeleton topology. Each angle θ_k is computed from
    displacement vectors at a pivot joint (Eq. 4-5):

        u_k = p_{a_k} - p_{j_k}    (pivot → parent)
        v_k = p_{c_k} - p_{j_k}    (pivot → child)
        θ_k = arccos(u_k · v_k / (||u_k|| · ||v_k||))

    Invariance Property (Eq. 7):
        These angles are provably invariant to SO(3) rotation, translation,
        and isotropic scaling. Translation cancels in displacement vectors;
        rotation and scaling cancel in the normalized dot product.

    IMPORTANT: Angles are computed from ORIGINAL (unnormalized) keypoints,
    NOT from wrist-centred/scaled coordinates.
    """

    def __init__(self):
        self.dim = 20
        self.triplets = ANGLE_TRIPLETS

    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Transform keypoints to angle representation.

        Args:
            keypoints: Array of shape (21, 3) — original (unnormalized) keypoints.

        Returns:
            Angle feature vector of shape (20,).
        """
        assert keypoints.shape == (21, 3), f"Expected (21, 3), got {keypoints.shape}"
        kp = keypoints.astype(np.float64)  # Use float64 for numerical stability

        angles = np.zeros(20, dtype=np.float32)

        for idx, (a, j, c) in enumerate(self.triplets):
            # Displacement vectors from pivot to parent and child (Eq. 4)
            u = kp[a] - kp[j]  # pivot → parent
            v = kp[c] - kp[j]  # pivot → child

            # Norms
            norm_u = np.linalg.norm(u)
            norm_v = np.linalg.norm(v)

            if norm_u < 1e-8 or norm_v < 1e-8:
                # Degenerate case: coincident points
                angles[idx] = 0.0
                continue

            # Normalised dot product (Eq. 5)
            cos_theta = np.dot(u, v) / (norm_u * norm_v)

            # Clamp to [-1, 1] for numerical stability
            cos_theta = np.clip(cos_theta, -1.0, 1.0)

            # Inter-joint angle
            angles[idx] = np.arccos(cos_theta).astype(np.float32)

        return angles

    def __repr__(self):
        return f"AngleRepresentation(dim={self.dim})"


class RawAngleRepresentation:
    """
    Combined raw + angle representation (83-D).

    Concatenation of wrist-centred/scale-normalised raw coordinates (63-D)
    and SO(3)-invariant inter-joint angles (20-D):

        x_raw_angle = [x_raw; x_angle] ∈ R^83    (Eq. 6)

    This combines positional and angular cues, which is beneficial when
    domain shift is moderate and data is abundant (e.g., ASL, LIBRAS↔Arabic).

    Args:
        normalize: If True, apply normalization to the raw component.
    """

    def __init__(self, normalize: bool = True):
        self.raw_repr = RawRepresentation(normalize=normalize)
        self.angle_repr = AngleRepresentation()
        self.dim = 83  # 63 + 20

    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Transform keypoints to raw_angle representation.

        Args:
            keypoints: Array of shape (21, 3) — original keypoints.

        Returns:
            Combined feature vector of shape (83,).
        """
        raw = self.raw_repr(keypoints)
        angle = self.angle_repr(keypoints)
        return np.concatenate([raw, angle])  # Eq. 6

    def __repr__(self):
        return f"RawAngleRepresentation(normalize={self.raw_repr.normalize}, dim={self.dim})"


# =============================================================================
# Factory
# =============================================================================

def get_representation(name: str, normalize: bool = True):
    """
    Get a representation transform by name.

    Args:
        name: One of 'raw', 'angle', 'raw_angle'.
        normalize: Whether to apply normalization to raw coordinates.

    Returns:
        Representation instance with .dim attribute and __call__ method.
    """
    representations = {
        "raw": lambda: RawRepresentation(normalize=normalize),
        "angle": lambda: AngleRepresentation(),
        "raw_angle": lambda: RawAngleRepresentation(normalize=normalize),
    }
    if name not in representations:
        raise ValueError(
            f"Unknown representation '{name}'. Choose from: {list(representations.keys())}"
        )
    return representations[name]()


def get_input_dim(name: str) -> int:
    """Get the input dimension for a representation name."""
    dims = {"raw": 63, "angle": 20, "raw_angle": 83}
    if name not in dims:
        raise ValueError(f"Unknown representation '{name}'.")
    return dims[name]
