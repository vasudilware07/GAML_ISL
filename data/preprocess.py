"""
Preprocessing pipeline: extract MediaPipe hand landmarks from **static images**,
normalise (translation + scale), and save as NumPy arrays.
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import cv2
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
    _MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task")
except ImportError:
    _HAS_MEDIAPIPE = False
    _MODEL_PATH = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x


# ── Single-image landmark extraction ────────────────────────────────────────

def _create_hand_landmarker(
    model_path: str = None,
    num_hands: int = 1,
    min_detection_confidence: float = 0.5,
):
    """Create a MediaPipe HandLandmarker (new tasks API)."""
    if model_path is None:
        model_path = _MODEL_PATH
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_detection_confidence,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def extract_landmarks_from_image(
    image_path: str,
    hands_model=None,
) -> Optional[np.ndarray]:
    """Extract 21 3-D hand landmarks from a single image.

    Args:
        image_path: Path to the image file.
        hands_model: Reusable MediaPipe HandLandmarker instance.

    Returns:
        Array of shape ``(21, 3)`` or ``None`` if no hand detected.
    """
    if not _HAS_MEDIAPIPE:
        raise ImportError(
            "mediapipe and opencv-python are required for image preprocessing. "
            "Install with: pip install mediapipe opencv-python"
        )
    own_model = hands_model is None
    if own_model:
        hands_model = _create_hand_landmarker()

    try:
        mp_image = mp.Image.create_from_file(image_path)
    except Exception:
        if own_model:
            hands_model.close()
        return None

    result = hands_model.detect(mp_image)

    if own_model:
        hands_model.close()

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        lm = np.array([[p.x, p.y, p.z] for p in hand], dtype=np.float32)
        return lm  # (21, 3)

    return None


# ── Normalisation ────────────────────────────────────────────────────────────

def normalize_landmarks(
    landmarks: np.ndarray,
    translate: bool = True,
    scale: bool = True,
) -> np.ndarray:
    """Normalise hand landmarks to be translation- and scale-invariant.

    Rigid-transform invariance property:
        If x' = Rx + t  then  ||x'_i - x'_j|| == ||x_i - x_j||
    Wrist-centring removes translation *t*; dividing by hand span removes
    the effect of distance to camera.

    Args:
        landmarks: Array of shape ``(21, 3)`` (single image).
        translate: Centre landmarks on the wrist (landmark 0).
        scale: Divide by max pairwise distance (hand span).

    Returns:
        Normalised landmarks array of the same shape.
    """
    lm = landmarks.copy()

    if np.allclose(lm, 0):
        return lm

    if translate:
        wrist = lm[0:1]  # (1, 3)
        lm = lm - wrist

    if scale:
        diffs = lm[:, None, :] - lm[None, :, :]  # (21, 21, 3)
        dists = np.linalg.norm(diffs, axis=-1)      # (21, 21)
        max_dist = dists.max()
        if max_dist > 1e-6:
            lm = lm / max_dist

    return lm


# ── Pairwise distance representation ────────────────────────────────────────

def compute_pairwise_distances(landmarks: np.ndarray) -> np.ndarray:
    """Convert landmarks to pairwise distance representation (210-D).

    Args:
        landmarks: Array of shape ``(21, 3)``.

    Returns:
        ``(210,)`` pairwise distance vector.
    """
    idx_i, idx_j = np.triu_indices(21, k=1)  # 21*20/2 = 210 pairs
    diffs = landmarks[idx_i] - landmarks[idx_j]  # (210, 3)
    return np.linalg.norm(diffs, axis=-1).astype(np.float32)


# ── Joint-angle representation ──────────────────────────────────────────────

# Each triplet (A, B, C) defines the angle at joint B formed by vectors BA→BC.
# We use 20 anatomically meaningful joint angles covering all fingers + wrist.
ANGLE_TRIPLETS: list = [
    # Thumb:  CMC, MCP, IP
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    # Index:  MCP, PIP, DIP
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    # Middle: MCP, PIP, DIP
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    # Ring:   MCP, PIP, DIP
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    # Pinky:  MCP, PIP, DIP
    (0, 17, 18), (17, 18, 19), (18, 19, 20),
    # Inter-finger spread angles (at palm)
    (5, 0, 9),     # index–middle spread
    (9, 0, 13),    # middle–ring spread
    (13, 0, 17),   # ring–pinky spread
    (1, 0, 5),     # thumb–index spread
    (17, 0, 1),    # pinky–thumb wrap-around
]


def compute_joint_angles(landmarks: np.ndarray) -> np.ndarray:
    """Compute joint angles (in radians) from landmark triplets.

    For each triplet ``(A, B, C)`` the angle at ``B`` is::

        angle = arccos( (BA · BC) / (|BA| · |BC|) )

    Args:
        landmarks: Array of shape ``(21, 3)``.

    Returns:
        ``(20,)`` vector of joint angles in [0, π].
    """
    angles = np.zeros(len(ANGLE_TRIPLETS), dtype=np.float32)
    for i, (a, b, c) in enumerate(ANGLE_TRIPLETS):
        ba = landmarks[a] - landmarks[b]
        bc = landmarks[c] - landmarks[b]
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angles[i] = np.arccos(cos_angle)
    return angles


def compute_raw_angle(landmarks: np.ndarray) -> np.ndarray:
    """Concatenate flattened raw landmarks (63-D) with joint angles (20-D).

    Both blocks are z-score normalised independently before concatenation
    to ensure consistent scaling.

    Args:
        landmarks: Array of shape ``(21, 3)``.

    Returns:
        ``(83,)`` concatenated feature vector.
    """
    raw = landmarks.reshape(-1).astype(np.float32)      # (63,)
    ang = compute_joint_angles(landmarks)                # (20,)

    # Per-block z-score normalisation
    def _znorm(x: np.ndarray) -> np.ndarray:
        std = x.std()
        if std < 1e-8:
            return x - x.mean()
        return (x - x.mean()) / std

    raw = _znorm(raw)
    ang = _znorm(ang)
    return np.concatenate([raw, ang], axis=0)  # (83,)


# ── Dataset preprocessing ───────────────────────────────────────────────────

def preprocess_dataset(
    image_dir: str,
    output_dir: str,
    normalize_translation: bool = True,
    normalize_scale: bool = True,
) -> None:
    """Preprocess a directory of static sign-language images.

    Expected structure::

        image_dir/
            class_A/
                img1.jpg
                img2.png
            class_B/
                ...

    Produces::

        output_dir/
            class_A/
                img1.npy   # shape (21, 3)
                ...

    Args:
        image_dir: Root directory containing class sub-folders with images.
        output_dir: Root directory for saving ``.npy`` landmark files.
        normalize_translation: Wrist-centre landmarks.
        normalize_scale: Scale-normalise landmarks.
    """
    if not _HAS_MEDIAPIPE:
        raise ImportError(
            "mediapipe and opencv-python are required. "
            "Install with: pip install mediapipe opencv-python"
        )

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    hands_model = _create_hand_landmarker()

    classes = sorted([d for d in image_dir.iterdir() if d.is_dir()])
    print(f"Found {len(classes)} classes in {image_dir}")

    total, success = 0, 0
    for cls_dir in tqdm(classes, desc="Classes"):
        cls_name = cls_dir.name
        out_cls = output_dir / cls_name
        out_cls.mkdir(parents=True, exist_ok=True)

        images = [f for f in cls_dir.iterdir() if f.suffix.lower() in image_exts]
        for img_path in tqdm(images, desc=f"  {cls_name}", leave=False):
            total += 1
            out_path = out_cls / (img_path.stem + ".npy")
            if out_path.exists():
                success += 1
                continue

            lm = extract_landmarks_from_image(str(img_path), hands_model=hands_model)
            if lm is None:
                continue

            lm = normalize_landmarks(lm, translate=normalize_translation, scale=normalize_scale)
            np.save(str(out_path), lm)
            success += 1

    hands_model.close()
    print(f"Preprocessing complete. {success}/{total} images → {output_dir}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess sign language images to landmarks")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Root dir with class sub-folders of images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for .npy landmark files")
    parser.add_argument("--no_translate", action="store_true")
    parser.add_argument("--no_scale", action="store_true")
    args = parser.parse_args()

    preprocess_dataset(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        normalize_translation=not args.no_translate,
        normalize_scale=not args.no_scale,
    )
