#!/usr/bin/env python3
"""
Smoke test: verify that all imports, model construction, and a short
synthetic forward pass succeed without errors.

Use after cloning or modifying the repo to catch import or shape issues
before committing.

Example
-------
    python tools/smoke_test.py          # full check
    python tools/smoke_test.py --quick  # imports only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"


def _check(label: str, fn):
    """Run *fn*, print pass/fail, return success bool."""
    try:
        fn()
        print(f"  {PASS} {label}")
        return True
    except Exception as e:
        print(f"  {FAIL} {label}: {e}")
        return False


def test_imports():
    """Verify all project modules import cleanly."""

    def _import_data():
        import data  # noqa: F401
        from data.datasets import LandmarkDataset, SyntheticLandmarkDataset  # noqa: F401
        from data.episodes import EpisodicSampler, split_support_query, collate_episode  # noqa: F401
        from data.preprocess import (  # noqa: F401
            compute_joint_angles, compute_pairwise_distances,
            compute_raw_angle, normalize_landmarks,
        )

    def _import_models():
        import models  # noqa: F401
        from models import build_encoder, build_few_shot_model  # noqa: F401
        from models.mlp_encoder import MLPEncoder  # noqa: F401
        from models.temporal_transformer import TemporalTransformerEncoder  # noqa: F401
        from models.gcn_encoder import GCNEncoder  # noqa: F401
        from models.prototypical import PrototypicalNetwork  # noqa: F401
        from models.siamese import SiameseNetwork, MatchingNetwork  # noqa: F401

    def _import_losses():
        import losses  # noqa: F401
        from losses.triplet import TripletLoss  # noqa: F401
        from losses.supcon import SupConLoss, ArcFaceLoss, build_loss  # noqa: F401

    def _import_utils():
        import utils  # noqa: F401
        from utils.seed import set_seed  # noqa: F401
        from utils.logger import get_logger  # noqa: F401
        from utils.metrics import accuracy, few_shot_accuracy_with_ci  # noqa: F401

    def _import_scripts():
        import train  # noqa: F401
        import evaluate  # noqa: F401
        import ablation  # noqa: F401
        import adapt  # noqa: F401

    results = []
    results.append(_check("data/*", _import_data))
    results.append(_check("models/*", _import_models))
    results.append(_check("losses/*", _import_losses))
    results.append(_check("utils/*", _import_utils))
    results.append(_check("top-level scripts", _import_scripts))
    return all(results)


def test_model_forward():
    """Build each encoder + few-shot model and run a synthetic forward pass."""
    import torch
    import yaml
    from models import build_encoder, build_few_shot_model
    from utils.seed import set_seed

    set_seed(0)

    cfg_path = REPO_ROOT / "configs" / "base.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    results = []

    for encoder_name in ["mlp", "transformer", "gcn"]:
        for representation in ["raw", "angle", "raw_angle"]:
            # GCN only works with raw (spatial)
            if encoder_name == "gcn" and representation != "raw":
                continue
            internal_repr = "graph" if encoder_name == "gcn" else representation

            def _run(enc=encoder_name, rep=internal_repr, orig_rep=representation):
                cfg_copy = cfg.copy()
                cfg_copy["model"] = {**cfg["model"], "encoder": enc}
                cfg_copy["representation"] = rep

                encoder = build_encoder(cfg_copy, rep)
                model = build_few_shot_model(cfg_copy, encoder)
                model.eval()

                n_way, k_shot, q_query = 3, 2, 4
                if orig_rep == "raw" or enc == "gcn":
                    sx = torch.randn(n_way * k_shot, 21, 3)
                    qx = torch.randn(n_way * q_query, 21, 3)
                elif orig_rep == "angle":
                    sx = torch.randn(n_way * k_shot, 20)
                    qx = torch.randn(n_way * q_query, 20)
                elif orig_rep == "raw_angle":
                    sx = torch.randn(n_way * k_shot, 83)
                    qx = torch.randn(n_way * q_query, 83)

                sy = torch.arange(n_way).repeat_interleave(k_shot)
                with torch.no_grad():
                    logits = model(sx, sy, qx, n_way)
                assert logits.shape == (n_way * q_query, n_way), \
                    f"Bad shape: {logits.shape}"

            results.append(_check(f"{encoder_name}/{representation}", _run))

    return all(results)


def test_synthetic_episode():
    """Run one episode end-to-end with synthetic data."""
    import torch
    import torch.nn.functional as F
    from data.datasets import SyntheticLandmarkDataset
    from data.episodes import EpisodicSampler, split_support_query, collate_episode
    from models import build_encoder, build_few_shot_model
    from utils.seed import set_seed

    def _run():
        set_seed(0)
        ds = SyntheticLandmarkDataset(num_classes=10, samples_per_class=25,
                                      representation="raw")
        labels = [ds[i][1] for i in range(len(ds))]
        sampler = EpisodicSampler(labels, n_way=5, k_shot=3, q_query=5, episodes=2)
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_episode)

        cfg = {
            "model": {"encoder": "mlp", "embedding_dim": 64,
                      "mlp": {"hidden_dims": [64], "dropout": 0.0},
                      "transformer": {"num_heads": 4, "num_layers": 2,
                                       "dim_feedforward": 128, "dropout": 0.1},
                      "gcn": {"hidden_dim": 64, "num_layers": 2, "dropout": 0.1}},
            "few_shot": {"method": "prototypical"},
            "dataset": {"name": "synthetic", "root": "/tmp/nonexistent",
                        "num_classes": 10, "num_landmarks": 21, "landmark_dim": 3,
                        "normalize_translation": True, "normalize_scale": True},
            "representation": "raw",
            "loss": {"name": "supcon", "supcon": {"temperature": 0.07},
                     "triplet": {"margin": 0.5, "mining": "semi_hard"},
                     "arcface": {"scale": 30.0, "margin": 0.5}},
            "training": {"lr": 1e-4, "weight_decay": 1e-4, "epochs": 1,
                         "batch_size": 8, "num_workers": 0, "grad_clip": 1.0,
                         "patience": 5},
            "seed": 0,
        }
        encoder = build_encoder(cfg, "raw")
        model = build_few_shot_model(cfg, encoder)
        model.eval()

        for batch in loader:
            data, lbls = batch
            sx, sy, qx, qy = split_support_query((data, lbls), 5, 3, 5)
            with torch.no_grad():
                log_probs = model(sx, sy, qx, 5)
            loss = F.nll_loss(log_probs, qy)
            assert loss.item() >= 0
            break

    return _check("synthetic episode", _run)


def test_compileall():
    """Bytecode-compile every .py file to catch syntax errors."""
    import compileall
    import re

    def _run():
        ok = compileall.compile_dir(
            str(REPO_ROOT), maxlevels=5, quiet=2,
            rx=re.compile(r"(__pycache__|\.git|\.venv|venv|data/raw|data/processed)"),
        )
        if not ok:
            raise RuntimeError("compileall found syntax errors")

    return _check("compileall (syntax check)", _run)


def test_dataset_presence():
    """Check that preprocessed datasets and JSON splits exist."""
    DATASETS = [
        "asl_alphabet", "libras_alphabet",
        "arabic_sign_alphabet", "thai_fingerspelling",
    ]
    results = []

    for ds in DATASETS:
        def _check_ds(name=ds):
            proc_dir = REPO_ROOT / "data" / "processed" / name
            if not proc_dir.exists():
                raise FileNotFoundError(f"Missing: {proc_dir}")
            # Count .npy files
            npy_count = sum(1 for _ in proc_dir.rglob("*.npy"))
            if npy_count == 0:
                raise FileNotFoundError(f"No .npy files in {proc_dir}")
            # Check JSON splits
            train_json = REPO_ROOT / "splits" / f"{name}_train.json"
            test_json = REPO_ROOT / "splits" / f"{name}_test.json"
            if not train_json.exists():
                raise FileNotFoundError(f"Missing: {train_json}")
            if not test_json.exists():
                raise FileNotFoundError(f"Missing: {test_json}")

        results.append(_check(f"dataset: {ds}", _check_ds))

    return all(results)


def test_result_csvs():
    """Check that key result CSVs exist and have expected row counts."""
    results = []
    EXPECTED = {
        "matrix_final.csv": 72,
        "cross_domain.csv": 4,
        "baseline_linear.csv": 8,
        "robustness_seeds.csv": 4,
    }

    for fname, expected_rows in EXPECTED.items():
        def _check_csv(fn=fname, n=expected_rows):
            import csv as csv_mod
            path = REPO_ROOT / "results" / fn
            if not path.exists():
                raise FileNotFoundError(f"Missing: {path}")
            with open(path, newline="") as f:
                row_count = sum(1 for _ in csv_mod.reader(f)) - 1  # minus header
            if row_count != n:
                raise ValueError(f"{fn}: expected {n} rows, got {row_count}")

        results.append(_check(f"results/{fname} ({expected_rows} rows)", _check_csv))

    return all(results)


def main():
    parser = argparse.ArgumentParser(description="Repository smoke test")
    parser.add_argument("--quick", action="store_true",
                        help="Only run import checks (skip forward pass)")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 50)
    print("  Smoke Test — sign_metric_learning")
    print("=" * 50)

    all_ok = True

    print("\n[1/6] Imports")
    all_ok &= test_imports()

    print("\n[2/6] Compile check")
    all_ok &= test_compileall()

    print("\n[3/6] Dataset presence")
    all_ok &= test_dataset_presence()

    print("\n[4/6] Result CSVs")
    all_ok &= test_result_csvs()

    if not args.quick:
        print("\n[5/6] Model forward passes")
        all_ok &= test_model_forward()

        print("\n[6/6] Synthetic episode")
        all_ok &= test_synthetic_episode()
    else:
        print("\n[5/6] Skipped (--quick)")
        print("\n[6/6] Skipped (--quick)")

    elapsed = time.time() - t0
    print("\n" + "=" * 50)
    if all_ok:
        print(f"  {PASS} ALL PASSED  ({elapsed:.1f}s)")
    else:
        print(f"  {FAIL} SOME TESTS FAILED  ({elapsed:.1f}s)")
    print("=" * 50)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
