#!/usr/bin/env python3
"""
Write a reproducibility manifest (JSON) capturing the full environment
and configuration used for a run.

Usage:
    # Standalone
    python tools/write_manifest.py --output results/manifest.json

    # Programmatic
    from tools.write_manifest import write_manifest
    write_manifest("results/manifest.json", extra={"experiment": "matrix"})
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent


def _git_info() -> Dict[str, str]:
    """Return current git hash and dirty status."""
    info: Dict[str, str] = {}
    try:
        info["git_hash"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["git_dirty"] = bool(status)
    except (subprocess.CalledProcessError, FileNotFoundError):
        info["git_hash"] = "unknown"
        info["git_dirty"] = "unknown"
    return info


def _package_versions() -> Dict[str, str]:
    """Return versions of key packages."""
    versions: Dict[str, str] = {}
    for pkg in ["torch", "numpy", "scipy", "sklearn", "matplotlib",
                "mediapipe", "cv2", "yaml", "tqdm"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"
    return versions


def _config_hash(cfg: Optional[dict] = None) -> str:
    """SHA-256 of the serialised config for change detection."""
    if cfg is None:
        cfg_path = REPO_ROOT / "configs" / "base.yaml"
        if cfg_path.exists():
            return hashlib.sha256(cfg_path.read_bytes()).hexdigest()[:16]
        return "no-config"
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def build_manifest(
    config: Optional[dict] = None,
    cli_args: Optional[list] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a manifest dictionary.

    Args:
        config: Config dict used for the run (optional).
        cli_args: sys.argv or equivalent (optional).
        extra: Additional metadata to include.

    Returns:
        Manifest dictionary.
    """
    manifest: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "packages": _package_versions(),
        **_git_info(),
        "config_hash": _config_hash(config),
        "seed": (config or {}).get("seed", "unknown"),
    }
    if cli_args is not None:
        manifest["cli_args"] = cli_args
    if config is not None:
        manifest["config"] = config
    if extra is not None:
        manifest.update(extra)
    return manifest


def write_manifest(
    output_path: str,
    config: Optional[dict] = None,
    cli_args: Optional[list] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Build and write manifest to a JSON file.

    Args:
        output_path: Path to write the manifest.
        config: Config dict.
        cli_args: CLI arguments.
        extra: Extra metadata.
    """
    manifest = build_manifest(config, cli_args, extra)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"Manifest written to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Write reproducibility manifest")
    parser.add_argument("--output", type=str,
                        default="results/manifest.json")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config to include")
    args = parser.parse_args()

    config = None
    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)

    write_manifest(args.output, config=config, cli_args=sys.argv)


if __name__ == "__main__":
    main()
