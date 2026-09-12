#!/usr/bin/env python3
"""
Auto-discover, download, extract, and filter sign-language datasets to
keep ONLY samples where exactly ONE hand is detected (MediaPipe Hands).

Supported dataset names
-----------------------
  sign-language-mnist   – Kaggle image dataset (28×28 grayscale CSV → PNG)
  asl-fingerspelling    – Kaggle image/video dataset
  thai-fingerspelling   – Kaggle image dataset
  bdslw60               – Kaggle video dataset (BdSLW60)
  lsa64                 – GitHub-released video dataset (Argentinian SL)

Example commands
----------------
  # Discover + download + extract + filter  (Kaggle credentials required)
 kaggle --version
ls -la ~/.kaggle/kaggle.json      --kaggle-slug datamunge/sign-language-mnist

  # Direct URL override  (any .zip / .tar.gz)
  python tools/auto_find_download_and_filter_onehand.py \\
      --dataset lsa64 --download --extract \\
      --url https://github.com/midusi/LSA64/releases/download/v1.0/data.zip

  # Filter only (dataset already in  data/raw/<name>/)
  python tools/auto_find_download_and_filter_onehand.py \\
      --dataset bdslw60 --seed 42

  # Dry-run: shows what *would* be kept / dropped without copying
  python tools/auto_find_download_and_filter_onehand.py \\
      --dataset thai-fingerspelling --dry-run

Script version: 1.0.0
"""

from __future__ import annotations

__version__ = "1.0.0"

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports – degrade gracefully
# ---------------------------------------------------------------------------
try:
    import requests

    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    import mediapipe as mp

    _HAS_MEDIAPIPE = True
except ImportError:
    _HAS_MEDIAPIPE = False

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore[override]
        return iterable

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("onehand")

# ---------------------------------------------------------------------------
# Project paths  (relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DOWNLOADS = REPO_ROOT / "data" / "downloads"
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_FILTERED = REPO_ROOT / "data" / "filtered_onehand"

_MODEL_PATH = str(REPO_ROOT / "models" / "hand_landmarker.task")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg"}

# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set deterministic seeds for random, numpy, and torch (if present)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False      # type: ignore[attr-defined]
    except ImportError:
        pass

# ============================================================================
#  1.  AUTO-DISCOVERY
# ============================================================================

# Curated list of *known* Kaggle slugs / URLs per dataset.
# We do NOT hallucinate URLs.  If a dataset is not in this table AND the user
# does not provide an override, we stop and tell them.
KNOWN_SOURCES: Dict[str, Dict[str, Any]] = {
    # ── 🥇 Source pretrain ─────────────────────────────────────────────
    "asl-alphabet": {
        "source_type": "kaggle",
        "identifier": "grassknoted/asl-alphabet",
        "notes": (
            "ASL Alphabet – clean, large (~1 GB), one-hand RGB images, 29 classes.  "
            "Use as source domain for pre-training."
        ),
    },
    # ── 🥈 Low-quality domain shift ───────────────────────────────────
    "sign-language-mnist": {
        "source_type": "kaggle",
        "identifier": "datamunge/sign-language-mnist",
        "notes": (
            "Sign Language MNIST – small 28×28 grayscale hand images (A-Z minus J,Z).  "
            "Low-resolution domain shift relative to ASL Alphabet."
        ),
    },
    # ── 🥉 Target adaptation ──────────────────────────────────────────
    "thai-fingerspelling": {
        "source_type": "kaggle",
        "identifier": "nickihartmann/thai-letter-sign-language",
        "notes": (
            "Thai Fingerspelling (image version) – different culture, hand morphology, "
            "and background variance.  Use as target adaptation domain."
        ),
    },
    # ── 🟡 Cross-language reinforcement ───────────────────────────────
    "libras-alphabet": {
        "source_type": "kaggle",
        "identifier": "williansoliveira/libras",
        "notes": (
            "Brazilian LIBRAS alphabet – static one-hand alphabet images (~304 MB).  "
            "Strengthens the cross-language claim."
        ),
    },
    # ── 🟡 Diversity ──────────────────────────────────────────────────
    "arabic-sign-alphabet": {
        "source_type": "kaggle",
        "identifier": "muhammadalbrham/rgb-arabic-alphabets-sign-language-dataset",
        "notes": (
            "RGB Arabic Alphabets Sign Language Dataset (~5 GB, 50 votes).  "
            "Image-based Arabic/Arab sign alphabet for additional diversity.  "
            "Alternative: 'ammarsayedtaha/arabic-sign-language-dataset-2022'."
        ),
    },
}


@dataclass
class DiscoveryResult:
    """Result returned by `discover_source`."""
    source_type: str          # kaggle | url | github | manual_required
    identifier: str           # slug, URL, or empty
    notes: str


def discover_source(
    dataset_name: str,
    *,
    kaggle_slug: Optional[str] = None,
    url: Optional[str] = None,
    github_release: Optional[str] = None,
) -> DiscoveryResult:
    """Return download source for *dataset_name*.

    Priority:
      1. Explicit CLI overrides  (``--kaggle-slug``, ``--url``, ``--github-release``)
      2. Curated lookup table ``KNOWN_SOURCES``
      3. Fail with ``manual_required``

    Args:
        dataset_name: Canonical short name (e.g. ``"bdslw60"``).
        kaggle_slug: Optional user-supplied Kaggle ``<owner>/<dataset>`` slug.
        url: Optional direct URL to a downloadable archive.
        github_release: Optional GitHub release asset URL.

    Returns:
        A :class:`DiscoveryResult` with the best source found.
    """
    # --- explicit overrides first -------------------------------------------
    if kaggle_slug:
        return DiscoveryResult("kaggle", kaggle_slug, "User-provided Kaggle slug.")
    if url:
        return DiscoveryResult("url", url, "User-provided direct URL.")
    if github_release:
        return DiscoveryResult("github", github_release, "User-provided GitHub release URL.")

    # --- curated table ------------------------------------------------------
    key = dataset_name.lower().strip()
    if key in KNOWN_SOURCES:
        src = KNOWN_SOURCES[key]
        return DiscoveryResult(src["source_type"], src["identifier"], src["notes"])

    # --- not found ----------------------------------------------------------
    return DiscoveryResult(
        "manual_required",
        "",
        (
            f"Could not auto-discover a download source for '{dataset_name}'.  "
            "Please supply --kaggle-slug, --url, or --github-release."
        ),
    )


# ============================================================================
#  2.  DOWNLOAD + EXTRACT
# ============================================================================

def _kaggle_download(slug: str, dest: Path) -> Path:
    """Download a Kaggle dataset via the ``kaggle`` CLI.

    Returns the path to the downloaded archive directory.
    """
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "The `kaggle` CLI is not installed or not on PATH.\n"
            "Install with:\n"
            "    pip install kaggle\n"
            "Then place your API token at  ~/.kaggle/kaggle.json"
        )
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"]
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle download failed (exit {result.returncode}).\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    log.info("Kaggle download complete → %s", dest)
    return dest


def _url_download(url: str, dest_dir: Path) -> Path:
    """Download a file from *url* into *dest_dir*; return the local path."""
    if not _HAS_REQUESTS:
        raise RuntimeError("The `requests` library is required for URL downloads.  pip install requests")
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Derive filename from URL
    fname = url.rstrip("/").split("/")[-1].split("?")[0]
    if not fname:
        fname = "download"
    local_path = dest_dir / fname
    if local_path.exists():
        log.info("Archive already exists: %s — skipping download", local_path)
        return local_path

    log.info("Downloading %s → %s", url, local_path)
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(local_path, "wb") as f:
        downloaded = 0
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {pct:5.1f}%  ({downloaded}/{total})", end="", flush=True)
    print()
    log.info("Download complete: %s (%d bytes)", local_path, local_path.stat().st_size)
    return local_path


def _extract(archive: Path, dest: Path) -> Path:
    """Extract .zip / .tar / .tar.gz / .tgz into *dest*.  Return *dest*."""
    dest.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    if name.endswith(".zip"):
        log.info("Extracting ZIP: %s → %s", archive, dest)
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(dest)
    elif name.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar")):
        log.info("Extracting TAR: %s → %s", archive, dest)
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(dest)
    else:
        raise ValueError(f"Unsupported archive format: {archive}")
    log.info("Extraction complete → %s", dest)
    return dest


def download_dataset(
    source: DiscoveryResult,
    dataset_name: str,
    do_download: bool = True,
    do_extract: bool = True,
) -> Path:
    """Download and extract a dataset.

    Returns the path to the extracted directory (``data/raw/<name>/``).
    """
    raw_dir = DATA_RAW / dataset_name

    if source.source_type == "manual_required":
        log.error(source.notes)
        raise SystemExit(1)

    if not do_download:
        if raw_dir.exists():
            log.info("Skipping download; using existing data at %s", raw_dir)
            return raw_dir
        log.warning("No existing data at %s and --download not set.", raw_dir)
        return raw_dir

    dl_dir = DATA_DOWNLOADS / dataset_name

    if source.source_type == "kaggle":
        # kaggle CLI extracts directly
        _kaggle_download(source.identifier, raw_dir)
    elif source.source_type in ("url", "github"):
        archive = _url_download(source.identifier, dl_dir)
        if do_extract:
            _extract(archive, raw_dir)
    else:
        raise ValueError(f"Unknown source_type: {source.source_type}")

    return raw_dir


# ============================================================================
#  3.  SCAN & CLASSIFY  (images vs. videos, discover class structure)
# ============================================================================

def _find_class_root(raw_dir: Path) -> Path:
    """Heuristically find the directory that contains class sub-folders.

    Sometimes archives nest: ``raw_dir / some_extra_folder / class_A / …``.
    We walk until we find a directory whose children are mostly dirs.
    """
    # If raw_dir itself already has class sub-dirs with media, use it
    if _has_media_subdirs(raw_dir):
        return raw_dir

    # Try one level deeper
    for child in sorted(raw_dir.iterdir()):
        if child.is_dir() and _has_media_subdirs(child):
            return child

    # Two levels
    for child in sorted(raw_dir.iterdir()):
        if child.is_dir():
            for grandchild in sorted(child.iterdir()):
                if grandchild.is_dir() and _has_media_subdirs(grandchild):
                    return grandchild

    # Fallback: raw_dir itself (may be flat)
    return raw_dir


def _has_media_subdirs(d: Path) -> bool:
    """Return True if *d* contains ≥2 sub-dirs that themselves contain media files."""
    subdirs_with_media = 0
    for child in d.iterdir():
        if not child.is_dir():
            continue
        for f in child.iterdir():
            if f.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS:
                subdirs_with_media += 1
                break
    return subdirs_with_media >= 2


@dataclass
class DatasetScan:
    """Metadata produced by scanning the raw directory."""
    class_root: Path
    classes: List[str]
    is_video: bool
    total_samples: int
    samples: List[Tuple[str, Path]]  # (class_name, file_path) — sorted


def scan_dataset(raw_dir: Path) -> DatasetScan:
    """Scan extracted data; detect class structure and media type.

    Args:
        raw_dir: Root of the extracted dataset.

    Returns:
        :class:`DatasetScan` with sorted sample list.
    """
    class_root = _find_class_root(raw_dir)
    log.info("Detected class root: %s", class_root)

    img_count = 0
    vid_count = 0
    samples: List[Tuple[str, Path]] = []

    classes = sorted(
        [d.name for d in class_root.iterdir() if d.is_dir()],
    )
    if not classes:
        raise FileNotFoundError(
            f"No class sub-folders found under {class_root}.  "
            "Expected structure: <class_root>/<class_name>/<media_files>"
        )

    for cls in classes:
        cls_dir = class_root / cls
        for fpath in sorted(cls_dir.rglob("*")):
            ext = fpath.suffix.lower()
            if ext in IMAGE_EXTS:
                img_count += 1
                samples.append((cls, fpath))
            elif ext in VIDEO_EXTS:
                vid_count += 1
                samples.append((cls, fpath))

    is_video = vid_count > img_count
    total = img_count + vid_count
    log.info(
        "Scan: %d classes, %d samples (%d images, %d videos) → treating as %s dataset",
        len(classes), total, img_count, vid_count, "VIDEO" if is_video else "IMAGE",
    )
    return DatasetScan(
        class_root=class_root,
        classes=classes,
        is_video=is_video,
        total_samples=total,
        samples=samples,
    )


# ============================================================================
#  4.  ONE-HAND FILTER
# ============================================================================

@dataclass
class SampleResult:
    """Per-sample filtering decision."""
    path_in: str
    path_out: str
    cls: str
    is_video: bool
    frames_checked: int
    num_frames_onehand: int
    num_frames_twohand: int
    num_frames_nohand: int
    detected_hands: int             # for images: 0/1/2; for videos: -1
    avg_det_conf: float
    decision: str                   # KEEP | DROP
    decision_reason: str
    seed: int
    timestamp: str
    mediapipe_config: str
    script_version: str


def _uniform_frame_indices(total_frames: int, n: int) -> List[int]:
    """Return *n* deterministic uniformly-spaced frame indices in [0, total)."""
    if total_frames <= 0 or n <= 0:
        return []
    if n == 1:
        return [0]
    if n >= total_frames:
        return list(range(total_frames))
    return [int(round(i * (total_frames - 1) / (n - 1))) for i in range(n)]


def _detect_hands_in_frame(
    frame_bgr: "np.ndarray",
    hands_model,
) -> Tuple[int, float]:
    """Run MediaPipe on a single BGR frame.

    Returns:
        ``(num_hands, max_detection_confidence)``
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = hands_model.detect(mp_image)
    if not result.hand_landmarks:
        return 0, 0.0
    n = len(result.hand_landmarks)
    max_conf = 0.0
    if result.handedness:
        for h in result.handedness:
            for c in h:
                max_conf = max(max_conf, c.score)
    return n, max_conf


def filter_image(
    path: Path,
    hands_model,
    min_det_conf: float,
) -> Tuple[int, float]:
    """Detect hands in a single image.

    Returns:
        ``(num_hands_detected, detection_confidence)``
    """
    try:
        mp_image = mp.Image.create_from_file(str(path))
    except Exception:
        return -1, 0.0
    result = hands_model.detect(mp_image)
    if not result.hand_landmarks:
        return 0, 0.0
    n_hands = len(result.hand_landmarks)
    max_conf = 0.0
    if result.handedness:
        for h in result.handedness:
            for c in h:
                max_conf = max(max_conf, c.score)
    return n_hands, max_conf


def filter_video(
    path: Path,
    hands_model,
    frames_per_video: int,
) -> Tuple[int, int, int, float]:
    """Check uniformly-sampled frames of a video for hand count.

    Returns:
        ``(onehand_frames, twohand_frames, nohand_frames, avg_conf)``
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0, 0, 0, 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = _uniform_frame_indices(total, frames_per_video)

    onehand = twohand = nohand = 0
    conf_sum = 0.0
    checked = 0

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        n_hands, conf = _detect_hands_in_frame(frame, hands_model)
        if n_hands == 1:
            onehand += 1
        elif n_hands >= 2:
            twohand += 1
        else:
            nohand += 1
        conf_sum += conf
        checked += 1

    cap.release()
    avg_conf = conf_sum / max(checked, 1)
    return onehand, twohand, nohand, avg_conf


def run_filter(
    scan: DatasetScan,
    dataset_name: str,
    *,
    seed: int,
    min_det_conf: float,
    min_track_conf: float,
    frames_per_video: int,
    min_onehand_ratio: float,
    strict_twohand: bool,
    dry_run: bool,
) -> List[SampleResult]:
    """Filter all samples, keeping only those with exactly one hand.

    Args:
        scan: Dataset scan metadata.
        dataset_name: Name used for output path.
        seed: Random seed (for report; filtering is deterministic).
        min_det_conf: Minimum detection confidence.
        min_track_conf: Minimum tracking confidence (videos).
        frames_per_video: Frames to sample per video.
        min_onehand_ratio: Ratio threshold for videos.
        strict_twohand: If True, drop videos with *any* two-hand frame.
        dry_run: If True, do not copy files.

    Returns:
        List of per-sample :class:`SampleResult`.
    """
    if not _HAS_MEDIAPIPE:
        raise RuntimeError(
            "mediapipe is required for one-hand filtering.\n"
            "Install with:  pip install mediapipe"
        )
    if not _HAS_CV2:
        raise RuntimeError(
            "opencv-python is required for one-hand filtering.\n"
            "Install with:  pip install opencv-python"
        )

    out_root = DATA_FILTERED / dataset_name
    if not dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    mp_config_str = (
        f"min_detection_confidence={min_det_conf}, "
        f"min_tracking_confidence={min_track_conf}, "
        f"max_num_hands=2, "
        f"static_image_mode={'True' if not scan.is_video else 'per-frame'}"
    )
    now_str = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Build MediaPipe model (new tasks API)
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=_MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=min_det_conf,
        min_hand_presence_confidence=min_det_conf,
    )
    hands = mp.tasks.vision.HandLandmarker.create_from_options(options)

    results: List[SampleResult] = []
    keep_count = 0
    drop_count = 0

    for cls_name, fpath in tqdm(scan.samples, desc="Filtering", unit="sample"):
        rel_out = out_root / cls_name / fpath.name
        sr = SampleResult(
            path_in=str(fpath),
            path_out=str(rel_out),
            cls=cls_name,
            is_video=scan.is_video,
            frames_checked=0,
            num_frames_onehand=0,
            num_frames_twohand=0,
            num_frames_nohand=0,
            detected_hands=-1,
            avg_det_conf=0.0,
            decision="DROP",
            decision_reason="",
            seed=seed,
            timestamp=now_str,
            mediapipe_config=mp_config_str,
            script_version=__version__,
        )

        try:
            if not scan.is_video:
                # --- IMAGE ---
                n_hands, conf = filter_image(fpath, hands, min_det_conf)
                sr.frames_checked = 1
                sr.detected_hands = n_hands
                sr.avg_det_conf = conf

                if n_hands == -1:
                    sr.decision = "DROP"
                    sr.decision_reason = "Could not read image (corrupted?)"
                elif n_hands == 1 and conf >= min_det_conf:
                    sr.decision = "KEEP"
                    sr.decision_reason = f"Exactly 1 hand, conf={conf:.3f}"
                    sr.num_frames_onehand = 1
                elif n_hands == 0:
                    sr.decision = "DROP"
                    sr.decision_reason = "No hand detected"
                    sr.num_frames_nohand = 1
                elif n_hands >= 2:
                    sr.decision = "DROP"
                    sr.decision_reason = f"{n_hands} hands detected"
                    sr.num_frames_twohand = 1
                else:
                    sr.decision = "DROP"
                    sr.decision_reason = f"1 hand but conf={conf:.3f} < {min_det_conf}"
                    sr.num_frames_onehand = 1
            else:
                # --- VIDEO ---
                onehand, twohand, nohand, avg_conf = filter_video(
                    fpath, hands, frames_per_video,
                )
                checked = onehand + twohand + nohand
                sr.frames_checked = checked
                sr.num_frames_onehand = onehand
                sr.num_frames_twohand = twohand
                sr.num_frames_nohand = nohand
                sr.avg_det_conf = avg_conf
                sr.detected_hands = -1  # N/A for videos

                if checked == 0:
                    sr.decision = "DROP"
                    sr.decision_reason = "Could not read any frames"
                else:
                    ratio = onehand / checked
                    if strict_twohand and twohand > 0:
                        sr.decision = "DROP"
                        sr.decision_reason = (
                            f"Two-hand frames detected ({twohand}/{checked}) "
                            f"and --strict-twohand is on"
                        )
                    elif ratio >= min_onehand_ratio:
                        sr.decision = "KEEP"
                        sr.decision_reason = (
                            f"One-hand ratio {ratio:.2f} >= {min_onehand_ratio} "
                            f"({onehand}/{checked})"
                        )
                    else:
                        sr.decision = "DROP"
                        sr.decision_reason = (
                            f"One-hand ratio {ratio:.2f} < {min_onehand_ratio} "
                            f"({onehand}/{checked})"
                        )
        except Exception as exc:
            sr.decision = "DROP"
            sr.decision_reason = f"Exception: {exc}"

        if sr.decision == "KEEP":
            keep_count += 1
            if not dry_run:
                rel_out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(fpath, rel_out)
        else:
            drop_count += 1

        results.append(sr)

    hands.close()
    log.info(
        "Filter complete: %d KEEP / %d DROP out of %d total",
        keep_count, drop_count, len(results),
    )
    return results


# ============================================================================
#  5.  MANIFEST + REPORT
# ============================================================================

_CSV_FIELDS = [
    "path_in", "path_out", "cls", "is_video", "frames_checked",
    "num_frames_onehand", "num_frames_twohand", "num_frames_nohand",
    "detected_hands", "avg_det_conf",
    "decision", "decision_reason",
    "seed", "timestamp", "mediapipe_config", "script_version",
]


def write_manifest_csv(results: List[SampleResult], path: Path) -> None:
    """Write manifest.csv sorted by (class, path_in)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_results = sorted(results, key=lambda r: (r.cls, r.path_in))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for sr in sorted_results:
            writer.writerow(asdict(sr))
    log.info("Wrote %s (%d rows)", path, len(sorted_results))


def write_manifest_json(results: List[SampleResult], path: Path) -> None:
    """Write manifest.json sorted by (class, path_in)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_results = sorted(results, key=lambda r: (r.cls, r.path_in))
    obj = {
        "version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_samples": len(sorted_results),
        "kept": sum(1 for r in sorted_results if r.decision == "KEEP"),
        "dropped": sum(1 for r in sorted_results if r.decision == "DROP"),
        "samples": [asdict(sr) for sr in sorted_results],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    log.info("Wrote %s", path)


def write_filter_report(
    results: List[SampleResult],
    path: Path,
    source: DiscoveryResult,
    args: argparse.Namespace,
) -> None:
    """Write a human-readable reproducibility report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = [r for r in results if r.decision == "KEEP"]
    dropped = [r for r in results if r.decision == "DROP"]

    # Per-class stats
    class_stats: Dict[str, Dict[str, int]] = {}
    for r in results:
        cs = class_stats.setdefault(r.cls, {"KEEP": 0, "DROP": 0})
        cs[r.decision] += 1

    lines: List[str] = [
        "=" * 72,
        "  ONE-HAND FILTER REPORT",
        "=" * 72,
        "",
        f"  Script version : {__version__}",
        f"  Generated      : {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"  Platform        : {platform.platform()}",
        f"  Python          : {sys.version}",
        "",
        "  Source discovery:",
        f"    type       : {source.source_type}",
        f"    identifier : {source.identifier}",
        f"    notes      : {source.notes}",
        "",
        "  CLI arguments:",
    ]
    for k, v in sorted(vars(args).items()):
        lines.append(f"    --{k.replace('_','-'):24s} = {v}")
    lines += [
        "",
        "-" * 72,
        f"  Total samples scanned : {len(results)}",
        f"  KEEP                  : {len(kept)}",
        f"  DROP                  : {len(dropped)}",
        f"  Keep rate             : {len(kept)/max(len(results),1)*100:.1f}%",
        "",
        "  Per-class breakdown:",
        f"    {'Class':<24s}  {'KEEP':>6s}  {'DROP':>6s}  {'Total':>6s}",
        "    " + "-" * 50,
    ]
    for cls in sorted(class_stats):
        cs = class_stats[cls]
        total = cs["KEEP"] + cs["DROP"]
        lines.append(f"    {cls:<24s}  {cs['KEEP']:>6d}  {cs['DROP']:>6d}  {total:>6d}")

    # Drop reason summary
    reason_counts: Dict[str, int] = {}
    for r in dropped:
        reason_counts[r.decision_reason] = reason_counts.get(r.decision_reason, 0) + 1
    lines += [
        "",
        "  Drop reasons (top 20):",
    ]
    for reason, cnt in sorted(reason_counts.items(), key=lambda x: -x[1])[:20]:
        lines.append(f"    [{cnt:>5d}]  {reason}")

    lines += ["", "=" * 72, ""]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info("Wrote %s", path)


# ============================================================================
#  6.  CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    p = argparse.ArgumentParser(
        description="Auto-discover, download, and one-hand-filter sign-language datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset", required=True, help="Dataset name (e.g. bdslw60, sign-language-mnist)")
    p.add_argument("--download", action="store_true", help="Download the dataset if not present")
    p.add_argument("--extract", action="store_true", help="Extract downloaded archive")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--min-det-conf", type=float, default=0.5, help="Min detection confidence (default: 0.5)")
    p.add_argument("--min-track-conf", type=float, default=0.5, help="Min tracking confidence (default: 0.5)")
    p.add_argument("--frames-per-video", type=int, default=32, help="Frames to sample per video (default: 32)")
    p.add_argument("--min-onehand-ratio", type=float, default=0.8, help="Min one-hand ratio for videos (default: 0.8)")
    p.add_argument("--strict-twohand", action="store_true", default=True,
                    help="Drop videos with ANY two-hand frame (default: True)")
    p.add_argument("--no-strict-twohand", dest="strict_twohand", action="store_false",
                    help="Allow some two-hand frames")
    p.add_argument("--workers", type=int, default=4, help="Worker count (currently unused; single-threaded for determinism)")
    p.add_argument("--dry-run", action="store_true", help="Show decisions without copying files")
    p.add_argument("--kaggle-slug", type=str, default=None, help="Override: Kaggle <owner>/<dataset> slug")
    p.add_argument("--url", type=str, default=None, help="Override: direct download URL")
    p.add_argument("--github-release", type=str, default=None, help="Override: GitHub release asset URL")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    set_seed(args.seed)
    dataset_name: str = args.dataset.lower().strip()

    log.info("=" * 60)
    log.info("  One-Hand Dataset Tool  v%s", __version__)
    log.info("=" * 60)
    log.info("Dataset : %s", dataset_name)
    log.info("Seed    : %d", args.seed)

    # ── 1. Discovery -------------------------------------------------------
    source = discover_source(
        dataset_name,
        kaggle_slug=args.kaggle_slug,
        url=args.url,
        github_release=args.github_release,
    )
    log.info("Source  : [%s]  %s", source.source_type, source.identifier)
    log.info("Notes   : %s", source.notes)

    if source.source_type == "manual_required":
        log.error(
            "\n  Auto-discovery failed for '%s'.\n"
            "  Provide one of:\n"
            "    --kaggle-slug <owner/dataset>\n"
            "    --url <direct_archive_url>\n"
            "    --github-release <release_asset_url>\n",
            dataset_name,
        )
        raise SystemExit(1)

    # ── 2. Download + Extract -----------------------------------------------
    raw_dir = DATA_RAW / dataset_name
    if args.download:
        raw_dir = download_dataset(
            source, dataset_name,
            do_download=args.download,
            do_extract=args.extract,
        )
    else:
        log.info("Skipping download (use --download to fetch).")

    if not raw_dir.exists():
        log.error(
            "Raw data directory does not exist: %s\n"
            "Run with --download --extract first.",
            raw_dir,
        )
        raise SystemExit(1)

    # ── 3. Scan -------------------------------------------------------------
    scan = scan_dataset(raw_dir)
    if scan.total_samples == 0:
        log.error("No media files found under %s", raw_dir)
        raise SystemExit(1)

    # ── 4. Filter -----------------------------------------------------------
    results = run_filter(
        scan,
        dataset_name,
        seed=args.seed,
        min_det_conf=args.min_det_conf,
        min_track_conf=args.min_track_conf,
        frames_per_video=args.frames_per_video,
        min_onehand_ratio=args.min_onehand_ratio,
        strict_twohand=args.strict_twohand,
        dry_run=args.dry_run,
    )

    # ── 5. Write manifests --------------------------------------------------
    out_root = DATA_FILTERED / dataset_name
    out_root.mkdir(parents=True, exist_ok=True)

    write_manifest_csv(results, out_root / "manifest.csv")
    write_manifest_json(results, out_root / "manifest.json")
    write_filter_report(results, out_root / "filter_report.txt", source, args)

    kept = sum(1 for r in results if r.decision == "KEEP")
    log.info("")
    log.info("Done!  %d / %d samples kept.", kept, len(results))
    if args.dry_run:
        log.info("(dry-run mode — no files were copied)")
    log.info("Output: %s", out_root)


if __name__ == "__main__":
    main()
