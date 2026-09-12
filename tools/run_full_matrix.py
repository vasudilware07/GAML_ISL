#!/usr/bin/env python3
"""
Full evaluation matrix: Encoder × Representation × Shot across target datasets.

Runs episodic evaluation ONLY (no training) for every combination of
encoder, representation, and k-shot setting on each target dataset,
then outputs:
    • results/matrix.csv          – flat CSV with one row per experiment
    • results/matrix.md           – Markdown tables grouped by dataset
    • results/matrix_config.yaml  – full config snapshot for reproducibility

Example
-------
  python tools/run_full_matrix.py \\
      --source_dataset asl_alphabet \\
      --datasets asl_alphabet \\
      --encoders mlp transformer gcn \\
      --representations raw angle raw_angle \\
      --shots 1 3 5 \\
      --episodes 1000 \\
      --output results/matrix.csv

Script version: 1.0.0
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Project imports ──────────────────────────────────────────────────────────
# Add repo root to path so imports work when called as `python tools/...`
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.datasets import LandmarkDataset, SplitLandmarkDataset
from data.episodes import EpisodicSampler, split_support_query, collate_episode
from models import build_encoder, build_few_shot_model
from utils.metrics import accuracy, few_shot_accuracy_with_ci
from utils.seed import set_seed

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("matrix")

# ── Representation → compatible encoder mapping ─────────────────────────────
#  GCN requires (B, 21, 3) spatial input; angle/raw_angle/pairwise are flat.
#  We run GCN with raw and graph only.  For flat representations we skip GCN.
_REPR_DIM = {
    "raw": (21, 3),       # spatial
    "angle": (20,),       # flat
    "raw_angle": (83,),   # flat
    "pairwise": (210,),   # flat
    "graph": (21, 3),     # spatial (alias for raw in GCN context)
}

_REPR_COMPAT: Dict[str, List[str]] = {
    "mlp":         ["raw", "angle", "raw_angle", "pairwise", "graph"],
    "transformer": ["raw", "angle", "raw_angle", "pairwise", "graph"],
    "gcn":         ["raw", "graph"],  # GCN needs spatial (21,3) input
}


# ═════════════════════════════════════════════════════════════════════════════
#  Config builder
# ═════════════════════════════════════════════════════════════════════════════

def make_base_config(args: argparse.Namespace) -> dict:
    """Build a base config dict from CLI args (mirrors configs/base.yaml)."""
    return {
        "seed": args.seed,
        "deterministic": True,
        "distance": getattr(args, "metric", "euclidean"),
        "dataset": {
            "name": "matrix_eval",
            "root": "",                     # filled per experiment
            "num_classes": 28,              # default; overridden per dataset
            "num_landmarks": 21,
            "landmark_dim": 3,
            "normalize_translation": True,
            "normalize_scale": True,
        },
        "representation": "raw",            # overridden per experiment
        "model": {
            "encoder": "transformer",       # overridden per experiment
            "embedding_dim": 128,
            "mlp":         {"hidden_dims": [256, 256], "dropout": 0.3},
            "transformer": {"num_heads": 4, "num_layers": 2,
                            "dim_feedforward": 256, "dropout": 0.1},
            "gcn":         {"hidden_dim": 128, "num_layers": 3, "dropout": 0.2},
        },
        "few_shot": {
            "method": "prototypical",
            "n_way": args.n_way,
            "k_shot": 5,                    # overridden per experiment
            "q_query": args.q_query,
            "episodes_eval": args.episodes,
        },
        "training": {"num_workers": 0},
        "device": args.device or "auto",
    }


def _resolve_device(cfg: dict) -> torch.device:
    dev = cfg.get("device", "auto")
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


# ═════════════════════════════════════════════════════════════════════════════
#  Dataset discovery
# ═════════════════════════════════════════════════════════════════════════════

def find_dataset_root(name: str) -> Path:
    """Locate the processed dataset directory.

    Search order:
        1. $DATA_ROOT/<name>_split/              (env var override)
        2. $DATA_ROOT/<name>/
        3. data/processed/<name>_split/          (train/test split)
        4. data/processed/<name>/                (flat)
        5. data/filtered_onehand/<name>/

    Raises FileNotFoundError if none found.
    """
    data_root = Path(os.environ["DATA_ROOT"]) if "DATA_ROOT" in os.environ else None
    candidates = []
    if data_root is not None:
        candidates += [
            data_root / f"{name}_split",
            data_root / name,
        ]
    candidates += [
        REPO_ROOT / "data" / "processed" / f"{name}_split",
        REPO_ROOT / "data" / "processed" / name,
        REPO_ROOT / "data" / "filtered_onehand" / name,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Dataset '{name}' not found.  Searched:\n"
        + "\n".join(f"  • {c}" for c in candidates)
    )


def load_dataset(
    root: Path,
    split: str,
    representation: str,
    dataset_name: str = "",
    use_json_splits: bool = False,
) -> LandmarkDataset:
    """Load a LandmarkDataset, failing loudly if missing.

    Args:
        root: Dataset root (may contain train/test subdirs).
        split: 'train' or 'test'.
        representation: Representation string.
        dataset_name: Logical dataset name (used for JSON split lookup).
        use_json_splits: Use JSON-based splits from splits/ directory.

    Returns:
        LandmarkDataset (or SplitLandmarkDataset).

    Raises:
        FileNotFoundError: if the split directory is empty or missing.
    """
    # ── JSON-based splits (new protocol) ──────────────────────────────
    if use_json_splits:
        # Resolve the flat preprocessed directory
        flat_root = root
        if flat_root.name.endswith("_split"):
            flat_root = flat_root.parent / flat_root.name[: -len("_split")]
        if flat_root.exists():
            ds = SplitLandmarkDataset(
                dataset_name=dataset_name,
                split=split,
                data_root=str(flat_root),
                representation=representation,
            )
            log.info(
                "  Loaded %s/%s (JSON split): %d samples, %d classes",
                dataset_name, split, len(ds), ds.num_classes,
            )
            return ds
        raise FileNotFoundError(
            f"Flat preprocessed dir not found: {flat_root}. "
            f"Run preprocessing first."
        )

    # ── Directory-based splits (legacy) ───────────────────────────────
    split_dir = root / split
    if split_dir.exists() and any(split_dir.iterdir()):
        ds = LandmarkDataset(root=str(split_dir), representation=representation)
    elif root.exists() and any(d for d in root.iterdir() if d.is_dir()):
        # No train/test split — use the root directly
        log.warning("No '%s' split found in %s, using root directly.", split, root)
        ds = LandmarkDataset(root=str(root), representation=representation)
    else:
        raise FileNotFoundError(
            f"No data for split='{split}' in {root}.  "
            "Run preprocessing first. Do NOT fall back to synthetic data."
        )
    log.info("  Loaded %s/%s: %d samples, %d classes", root.name, split, len(ds), ds.num_classes)
    return ds


# ═════════════════════════════════════════════════════════════════════════════
#  Evaluation
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: LandmarkDataset,
    device: torch.device,
    n_way: int,
    k_shot: int,
    q_query: int,
    episodes: int,
    seed: int,
    auto_adjust_q: bool = False,
    dataset_name: str = "unknown",
    split_name: str = "test",
) -> Dict[str, float]:
    """Run episodic evaluation and return accuracy + CI.

    Seeds the RNG deterministically per call using (seed + k_shot)
    so that different shot settings sample different but reproducible episodes.
    """
    # Deterministic seeding for this evaluation run
    set_seed(seed + k_shot * 1000, deterministic=True)

    labels = [dataset[i][1] for i in range(len(dataset))]
    try:
        sampler = EpisodicSampler(
            labels, n_way=n_way, k_shot=k_shot, q_query=q_query,
            episodes=episodes, seed=seed,
            auto_adjust_q=auto_adjust_q,
            dataset_name=dataset_name, split_name=split_name,
        )
    except ValueError as e:
        log.warning("  Cannot evaluate k=%d: %s", k_shot, e)
        return {"accuracy": float("nan"), "ci": float("nan"), "loss": float("nan"),
                "actual_q_query": q_query}

    loader = DataLoader(dataset, batch_sampler=sampler,
                        collate_fn=collate_episode, num_workers=0)

    # Use the (possibly adjusted) q_query from the sampler
    actual_q = sampler.q_query

    model.eval()
    accs: List[float] = []
    total_loss = 0.0

    for batch in tqdm(loader, desc=f"  eval k={k_shot}", leave=False, unit="ep"):
        data, lbls = batch
        data, lbls = data.to(device), lbls.to(device)
        support_x, support_y, query_x, query_y = split_support_query(
            (data, lbls), n_way, k_shot, actual_q,
        )
        support_x = support_x.to(device)
        support_y = support_y.to(device)
        query_x = query_x.to(device)
        query_y = query_y.to(device)

        log_probs = model(support_x, support_y, query_x, n_way)
        loss = F.nll_loss(log_probs, query_y)
        acc = accuracy(log_probs, query_y)

        accs.append(acc)
        total_loss += loss.item()

    mean_acc, ci = few_shot_accuracy_with_ci(accs)
    return {
        "accuracy": mean_acc,
        "ci": ci,
        "loss": total_loss / max(len(accs), 1),
        "actual_q_query": sampler.q_query,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Checkpoint loading
# ═════════════════════════════════════════════════════════════════════════════

def try_load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Optional[str],
    encoder_name: str,
    representation: str,
) -> bool:
    """Attempt to load weights from a checkpoint (strict=False).

    Returns True if weights were loaded, False otherwise.
    """
    if checkpoint_path is None:
        return False
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        log.warning("  Checkpoint not found: %s — using random init", ckpt_path)
        return False

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        log.warning("  Loaded checkpoint with %d missing keys (encoder=%s, repr=%s)",
                     len(missing), encoder_name, representation)
    if unexpected:
        log.warning("  Loaded checkpoint with %d unexpected keys", len(unexpected))
    if not missing and not unexpected:
        log.info("  Loaded checkpoint: %s (exact match)", ckpt_path.name)
    return True


# ═════════════════════════════════════════════════════════════════════════════
#  Matrix runner
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_matrix(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Run the full evaluation matrix and return rows."""
    base_cfg = make_base_config(args)
    device = _resolve_device(base_cfg)
    log.info("Device: %s", device)

    # Pre-resolve all dataset roots to fail fast
    dataset_roots: Dict[str, Path] = {}
    for ds_name in args.datasets:
        dataset_roots[ds_name] = find_dataset_root(ds_name)
        log.info("Dataset %-25s → %s", ds_name, dataset_roots[ds_name])

    combos = list(itertools.product(args.encoders, args.representations))
    total_evals = len(combos) * len(args.datasets) * len(args.shots)
    log.info(
        "Matrix: %d encoders × %d representations × %d datasets × %d shot settings = %d evaluations",
        len(args.encoders), len(args.representations),
        len(args.datasets), len(args.shots), total_evals,
    )

    results: List[Dict[str, Any]] = []
    eval_idx = 0

    for encoder_name, representation in combos:
        # Check compatibility (GCN can only handle spatial 3-D input)
        if representation not in _REPR_COMPAT.get(encoder_name, []):
            log.info(
                "SKIP %s + %s (incompatible — GCN needs spatial input)",
                encoder_name, representation,
            )
            for ds_name in args.datasets:
                for k_shot in args.shots:
                    results.append({
                        "dataset": ds_name,
                        "encoder": encoder_name,
                        "representation": representation,
                        "k_shot": k_shot,
                        "n_way": args.n_way,
                        "q_query": args.q_query,
                        "episodes": args.episodes,
                        "seed": args.seed,
                        "accuracy_mean": float("nan"),
                        "ci95": float("nan"),
                        "notes": "SKIPPED: incompatible encoder-representation pair",
                    })
            continue

        # For GCN the internal representation key is 'graph'
        internal_repr = "graph" if encoder_name == "gcn" else representation

        # Build model once per (encoder, representation) pair
        cfg = copy.deepcopy(base_cfg)
        cfg["representation"] = internal_repr
        cfg["model"]["encoder"] = encoder_name

        encoder = build_encoder(cfg, internal_repr)
        model = build_few_shot_model(cfg, encoder).to(device)
        total_params = sum(p.numel() for p in model.parameters())

        loaded = try_load_checkpoint(model, args.checkpoint, encoder_name, representation)

        log.info(
            "\n═══ %s + %s  (%s params, ckpt=%s) ═══",
            encoder_name.upper(), representation, f"{total_params:,}",
            "loaded" if loaded else "random-init",
        )

        for ds_name in args.datasets:
            root = dataset_roots[ds_name]
            log.info("  Dataset: %s", ds_name)

            # Load dataset with appropriate representation
            use_json = getattr(args, 'json_splits', False)
            ds = load_dataset(
                root, args.eval_split, representation,
                dataset_name=ds_name, use_json_splits=use_json,
            )
            auto_q = getattr(args, 'auto_adjust_q', False)

            for k_shot in args.shots:
                eval_idx += 1
                log.info(
                    "  [%d/%d] %s | %s | %s | %d-shot ...",
                    eval_idx, total_evals, ds_name, encoder_name,
                    representation, k_shot,
                )
                t0 = time.time()

                metrics = evaluate(
                    model, ds, device,
                    n_way=args.n_way,
                    k_shot=k_shot,
                    q_query=args.q_query,
                    episodes=args.episodes,
                    seed=args.seed,
                    auto_adjust_q=auto_q,
                    dataset_name=ds_name,
                    split_name=args.eval_split,
                )

                elapsed = time.time() - t0
                acc = metrics["accuracy"]
                ci = metrics["ci"]
                actual_q = metrics.get("actual_q_query", args.q_query)

                notes_parts = []
                if not loaded:
                    notes_parts.append("no-pretrain")
                if np.isnan(acc):
                    notes_parts.append("insufficient-samples")
                if use_json:
                    notes_parts.append("json-splits")
                if actual_q != args.q_query:
                    notes_parts.append(f"auto_q={actual_q}")
                notes = "; ".join(notes_parts)

                log.info(
                    "    → acc=%.4f ± %.4f  (%.1fs)%s",
                    acc, ci, elapsed,
                    f"  [q adjusted to {actual_q}]" if actual_q != args.q_query else "",
                )

                results.append({
                    "dataset": ds_name,
                    "encoder": encoder_name,
                    "representation": representation,
                    "k_shot": k_shot,
                    "n_way": args.n_way,
                    "q_query": actual_q,
                    "episodes": args.episodes,
                    "seed": args.seed,
                    "accuracy_mean": round(acc, 6),
                    "ci95": round(ci, 6),
                    "notes": notes,
                })

    return results


# ═════════════════════════════════════════════════════════════════════════════
#  Output writers
# ═════════════════════════════════════════════════════════════════════════════

CSV_FIELDS = [
    "dataset", "encoder", "representation", "k_shot",
    "n_way", "q_query", "episodes", "seed",
    "accuracy_mean", "ci95", "notes",
]


def write_csv(results: List[Dict[str, Any]], path: Path) -> None:
    """Write flat CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
    log.info("Wrote %s (%d rows)", path, len(results))


def write_markdown(
    results: List[Dict[str, Any]],
    path: Path,
    shots: List[int],
) -> None:
    """Write pretty Markdown tables grouped by dataset.

    Layout per dataset::

        | Encoder | Representation | 1-shot | 3-shot | 5-shot |
        |---------|----------------|--------|--------|--------|
        | MLP     | raw            | 0.52±0.03 | ...  | ...    |
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Group by dataset
    datasets_seen: List[str] = []
    by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        ds = r["dataset"]
        if ds not in by_dataset:
            by_dataset[ds] = []
            datasets_seen.append(ds)
        by_dataset[ds].append(r)

    lines: List[str] = [
        "# Evaluation Matrix Results",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}Z",
        "",
    ]

    for ds_name in datasets_seen:
        rows = by_dataset[ds_name]
        lines.append(f"## {ds_name}")
        lines.append("")

        # Build header
        shot_cols = [f"{s}-shot" for s in shots]
        header = "| Encoder | Representation | " + " | ".join(shot_cols) + " |"
        sep = "|---------|----------------|" + "|".join(["--------"] * len(shots)) + "|"
        lines.append(header)
        lines.append(sep)

        # Collect unique (encoder, repr) combos in order
        seen_combos: List[Tuple[str, str]] = []
        for r in rows:
            combo = (r["encoder"], r["representation"])
            if combo not in seen_combos:
                seen_combos.append(combo)

        for enc, rep in seen_combos:
            cells: List[str] = []
            for s in shots:
                match = [r for r in rows
                         if r["encoder"] == enc
                         and r["representation"] == rep
                         and r["k_shot"] == s]
                if match:
                    r = match[0]
                    acc = r["accuracy_mean"]
                    ci = r["ci95"]
                    if np.isnan(acc):
                        cells.append("—")
                    else:
                        cells.append(f"{acc:.4f}±{ci:.4f}")
                else:
                    cells.append("—")
            line = f"| {enc:<7s} | {rep:<14s} | " + " | ".join(cells) + " |"
            lines.append(line)

        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info("Wrote %s", path)


def write_config_snapshot(args: argparse.Namespace, path: Path) -> None:
    """Dump the full CLI config as YAML for reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cli_args": vars(args),
        "base_config": make_base_config(args),
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(obj, f, default_flow_style=False, sort_keys=False)
    log.info("Wrote %s", path)


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Full encoder × representation × shot evaluation matrix.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--source_dataset", type=str, default="asl_alphabet",
        help="Name of the source (pre-trained) dataset (for logging; default: asl_alphabet)",
    )
    p.add_argument(
        "--datasets", type=str, nargs="+", default=["asl_alphabet"],
        help="Target dataset names to evaluate on (default: asl_alphabet)",
    )
    p.add_argument(
        "--encoders", type=str, nargs="+", default=["mlp", "transformer", "gcn"],
        help="Encoder architectures (default: mlp transformer gcn)",
    )
    p.add_argument(
        "--representations", type=str, nargs="+", default=["raw", "angle", "raw_angle"],
        help="Input representations (default: raw angle raw_angle)",
    )
    p.add_argument(
        "--shots", type=int, nargs="+", default=[1, 3, 5],
        help="K-shot values (default: 1 3 5)",
    )
    p.add_argument("--n_way", type=int, default=5, help="N-way (default: 5)")
    p.add_argument("--q_query", type=int, default=15, help="Queries per class (default: 15)")
    p.add_argument("--metric", type=str, default="euclidean",
                   choices=["euclidean", "cosine"],
                   help="Distance metric for ProtoNet (default: euclidean)")
    p.add_argument("--eval_split", type=str, default="test", help="Which split to evaluate on: test or train (default: test)")
    p.add_argument("--episodes", type=int, default=1000, help="Eval episodes (default: 1000)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--device", type=str, default=None, help="Device (default: auto)")
    p.add_argument(
        "--json_splits", action="store_true",
        help="Use JSON-based splits from splits/ directory (new protocol)",
    )
    p.add_argument(
        "--auto_adjust_q", action="store_true",
        help="Auto-lower q_query when classes have too few samples",
    )
    p.add_argument(
        "--output", type=str, default="results/matrix.csv",
        help="Output CSV path (default: results/matrix.csv)",
    )
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to pretrained checkpoint (optional). "
             "Loaded with strict=False; mismatched keys are warned.",
    )
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    log.info("═" * 65)
    log.info("  Full Evaluation Matrix")
    log.info("═" * 65)
    log.info("  Encoders       : %s", args.encoders)
    log.info("  Representations: %s", args.representations)
    log.info("  Shots          : %s", args.shots)
    log.info("  Datasets       : %s", args.datasets)
    log.info("  Episodes       : %d", args.episodes)
    log.info("  Seed           : %d", args.seed)
    log.info("═" * 65)

    set_seed(args.seed, deterministic=True)

    out_csv = Path(args.output)
    out_md = out_csv.with_suffix(".md")
    out_yaml = out_csv.with_name(out_csv.stem + "_config.yaml")

    # Save reproducibility config BEFORE running
    write_config_snapshot(args, out_yaml)

    # Run
    results = run_matrix(args)

    # Write outputs
    write_csv(results, out_csv)
    write_markdown(results, out_md, args.shots)

    # Write reproducibility manifest
    try:
        from tools.write_manifest import write_manifest
        manifest_path = out_csv.with_name(out_csv.stem + "_manifest.json")
        write_manifest(str(manifest_path), cli_args=sys.argv,
                       extra={"experiment": "full_matrix", "n_results": len(results)})
    except Exception as exc:
        log.warning("Could not write manifest: %s", exc)

    # Final summary to stdout
    log.info("\n" + "═" * 65)
    log.info("  DONE — %d evaluations completed", len(results))
    log.info("  CSV : %s", out_csv)
    log.info("  MD  : %s", out_md)
    log.info("  YAML: %s", out_yaml)
    log.info("═" * 65)


if __name__ == "__main__":
    main()
