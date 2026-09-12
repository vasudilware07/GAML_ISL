#!/usr/bin/env python3
"""Train and evaluate within-domain few-shot models.

Each row trains one model on a dataset's JSON train split, evaluates on the
matching JSON test split, and writes an aggregate CSV.  This is different from
``tools/run_full_matrix.py``, which evaluates random-init encoders only.
"""

from __future__ import annotations

import argparse
import csv
import copy
import statistics
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.datasets import SplitLandmarkDataset
from data.episodes import EpisodicSampler, collate_episode
from losses.supcon import build_loss
from models import build_encoder, build_few_shot_model
from train import evaluate_episodes, get_device, load_config, train_one_epoch
from utils.seed import set_seed


DEFAULT_DATASETS = [
    "asl_alphabet",
    "arabic_sign_alphabet",
    "libras_alphabet",
    "thai_fingerspelling",
]

DEFAULT_ENCODERS = ["mlp", "transformer"]
DEFAULT_REPRESENTATIONS = ["raw", "angle", "raw_angle"]
DEFAULT_SHOTS = [1, 3, 5]

FIELDNAMES = [
    "dataset",
    "encoder",
    "representation",
    "n_way",
    "k_shot",
    "seed",
    "q_query_train",
    "q_query_test",
    "episodes_train",
    "episodes_eval",
    "epochs",
    "best_epoch",
    "accuracy_mean",
    "ci95",
    "eval_loss",
    "train_samples",
    "test_samples",
    "classes",
    "checkpoint",
]


def make_config(
    base: dict,
    dataset: str,
    encoder: str,
    representation: str,
    k_shot: int,
    args: argparse.Namespace,
) -> dict:
    cfg = copy.deepcopy(base)
    cfg["seed"] = args.seed
    cfg["device"] = args.device
    cfg["representation"] = representation
    cfg["dataset"]["name"] = dataset
    cfg["dataset"]["root"] = f"data/processed/{dataset}"
    cfg["model"]["encoder"] = encoder
    cfg["few_shot"]["n_way"] = args.n_way
    cfg["few_shot"]["k_shot"] = k_shot
    cfg["few_shot"]["q_query"] = args.q_query
    cfg["few_shot"]["episodes_train"] = args.episodes_train
    cfg["few_shot"]["episodes_eval"] = args.episodes_eval
    cfg["training"]["epochs"] = args.epochs
    cfg["training"]["lr"] = args.lr
    cfg["training"]["num_workers"] = 0
    return cfg


def labels_for(dataset: SplitLandmarkDataset) -> list[int]:
    return [dataset[i][1] for i in range(len(dataset))]


def run_experiment(
    dataset_name: str,
    encoder_name: str,
    representation: str,
    k_shot: int,
    base_cfg: dict,
    args: argparse.Namespace,
) -> dict:
    cfg = make_config(base_cfg, dataset_name, encoder_name, representation, k_shot, args)
    set_seed(args.seed, deterministic=True)
    device = get_device(cfg)

    root = REPO_ROOT / "data" / "processed" / dataset_name
    train_ds = SplitLandmarkDataset(dataset_name, "train", str(root), representation)
    test_ds = SplitLandmarkDataset(dataset_name, "test", str(root), representation)

    train_sampler = EpisodicSampler(
        labels_for(train_ds),
        n_way=args.n_way,
        k_shot=k_shot,
        q_query=args.q_query,
        episodes=args.episodes_train,
        seed=args.seed,
        auto_adjust_q=args.auto_adjust_q,
        dataset_name=dataset_name,
        split_name="train",
    )
    test_sampler = EpisodicSampler(
        labels_for(test_ds),
        n_way=args.n_way,
        k_shot=k_shot,
        q_query=args.q_query,
        episodes=args.episodes_eval,
        seed=args.seed + 10_000,
        auto_adjust_q=args.auto_adjust_q,
        dataset_name=dataset_name,
        split_name="test",
    )

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=collate_episode,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_sampler=test_sampler,
        collate_fn=collate_episode,
        num_workers=0,
    )

    encoder = build_encoder(cfg, representation)
    model = build_few_shot_model(cfg, encoder).to(device)
    loss_fn = build_loss(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    best = {
        "accuracy": -1.0,
        "ci": 0.0,
        "loss": 0.0,
        "epoch": 0,
    }
    ckpt_dir = REPO_ROOT / "results" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / (
        f"within_domain_trained_{dataset_name}_{encoder_name}_{representation}_{k_shot}shot.pt"
    )

    print(
        f"\n=== {dataset_name} | {encoder_name} | {representation} | "
        f"{k_shot}-shot | device={device} ===",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            args.n_way,
            k_shot,
            train_sampler.q_query,
            cfg["training"].get("grad_clip", 1.0),
        )
        eval_metrics = evaluate_episodes(
            model,
            test_loader,
            device,
            args.n_way,
            k_shot,
            test_sampler.q_query,
        )
        print(
            f"{dataset_name} {encoder_name}/{representation} {k_shot}-shot "
            f"epoch {epoch}/{args.epochs}: "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"eval_acc={eval_metrics['accuracy']:.4f} +/- {eval_metrics['ci']:.4f}",
            flush=True,
        )
        if eval_metrics["accuracy"] > best["accuracy"]:
            best = {
                "accuracy": eval_metrics["accuracy"],
                "ci": eval_metrics["ci"],
                "loss": eval_metrics["loss"],
                "epoch": epoch,
            }
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_accuracy": best["accuracy"],
                    "config": cfg,
                },
                ckpt_path,
            )

    return {
        "dataset": dataset_name,
        "encoder": encoder_name,
        "representation": representation,
        "n_way": args.n_way,
        "k_shot": k_shot,
        "seed": args.seed,
        "q_query_train": train_sampler.q_query,
        "q_query_test": test_sampler.q_query,
        "episodes_train": args.episodes_train,
        "episodes_eval": args.episodes_eval,
        "epochs": args.epochs,
        "best_epoch": best["epoch"],
        "accuracy_mean": f"{best['accuracy']:.6f}",
        "ci95": f"{best['ci']:.6f}",
        "eval_loss": f"{best['loss']:.6f}",
        "train_samples": len(train_ds),
        "test_samples": len(test_ds),
        "classes": train_ds.num_classes,
        "checkpoint": str(ckpt_path.relative_to(REPO_ROOT)),
    }


def append_row(out_path: Path, row: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    exists = out_path.exists() and out_path.stat().st_size > 0
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_markdown(csv_path: Path) -> Path:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows.extend(csv.DictReader(f))

    md_path = csv_path.with_suffix(".md")
    datasets = []
    for row in rows:
        if row["dataset"] not in datasets:
            datasets.append(row["dataset"])

    lines = ["# Trained Within-Domain Results", ""]
    for dataset in datasets:
        lines += [f"## {dataset}", ""]
        lines += ["| Encoder | Representation | 1-shot | 3-shot | 5-shot |"]
        lines += ["|---------|----------------|--------|--------|--------|"]
        combos = []
        for row in rows:
            key = (row["encoder"], row["representation"])
            if row["dataset"] == dataset and key not in combos:
                combos.append(key)
        for encoder_name, representation in combos:
            cells = []
            for shot in DEFAULT_SHOTS:
                match = next(
                    (
                        row
                        for row in rows
                        if row["dataset"] == dataset
                        and row["encoder"] == encoder_name
                        and row["representation"] == representation
                        and int(row["k_shot"]) == shot
                    ),
                    None,
                )
                if match:
                    acc = float(match["accuracy_mean"]) * 100
                    ci = float(match["ci95"]) * 100
                    cells.append(f"{acc:.2f} +/- {ci:.2f}")
                else:
                    cells.append("")
            lines.append(
                f"| {encoder_name} | {representation} | {cells[0]} | {cells[1]} | {cells[2]} |"
            )
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def write_seed_summary(csv_path: Path) -> tuple[Path, Path]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows.extend(csv.DictReader(f))

    grouped = {}
    for row in rows:
        key = (
            row["dataset"],
            row["encoder"],
            row["representation"],
            int(row["k_shot"]),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (dataset, encoder_name, representation, k_shot), group in grouped.items():
        accs = [float(row["accuracy_mean"]) for row in group]
        cis = [float(row["ci95"]) for row in group]
        seeds = sorted(int(row["seed"]) for row in group)
        summary_rows.append(
            {
                "dataset": dataset,
                "encoder": encoder_name,
                "representation": representation,
                "k_shot": k_shot,
                "seeds": " ".join(str(seed) for seed in seeds),
                "n_seeds": len(seeds),
                "accuracy_mean": f"{statistics.mean(accs):.6f}",
                "accuracy_std": f"{statistics.stdev(accs):.6f}" if len(accs) > 1 else "0.000000",
                "mean_episode_ci95": f"{statistics.mean(cis):.6f}",
            }
        )

    order = {name: i for i, name in enumerate(DEFAULT_DATASETS)}
    rep_order = {name: i for i, name in enumerate(DEFAULT_REPRESENTATIONS)}
    summary_rows.sort(
        key=lambda row: (
            order.get(row["dataset"], 999),
            row["encoder"],
            rep_order.get(row["representation"], 999),
            int(row["k_shot"]),
        )
    )

    summary_csv = csv_path.with_name(f"{csv_path.stem}_summary.csv")
    summary_fields = [
        "dataset",
        "encoder",
        "representation",
        "k_shot",
        "seeds",
        "n_seeds",
        "accuracy_mean",
        "accuracy_std",
        "mean_episode_ci95",
    ]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_md = csv_path.with_name(f"{csv_path.stem}_summary.md")
    datasets = []
    for row in summary_rows:
        if row["dataset"] not in datasets:
            datasets.append(row["dataset"])

    lines = ["# Trained Within-Domain Three-Seed Summary", ""]
    lines.append("Cells show mean accuracy +/- seed standard deviation, in percent.")
    lines.append("")
    for dataset in datasets:
        lines += [f"## {dataset}", ""]
        lines += ["| Encoder | Representation | 1-shot | 3-shot | 5-shot |"]
        lines += ["|---------|----------------|--------|--------|--------|"]
        combos = []
        for row in summary_rows:
            key = (row["encoder"], row["representation"])
            if row["dataset"] == dataset and key not in combos:
                combos.append(key)
        for encoder_name, representation in combos:
            cells = []
            for shot in DEFAULT_SHOTS:
                match = next(
                    (
                        row
                        for row in summary_rows
                        if row["dataset"] == dataset
                        and row["encoder"] == encoder_name
                        and row["representation"] == representation
                        and int(row["k_shot"]) == shot
                    ),
                    None,
                )
                if match:
                    acc = float(match["accuracy_mean"]) * 100
                    std = float(match["accuracy_std"]) * 100
                    cells.append(f"{acc:.2f} +/- {std:.2f}")
                else:
                    cells.append("")
            lines.append(
                f"| {encoder_name} | {representation} | {cells[0]} | {cells[1]} | {cells[2]} |"
            )
        lines.append("")

    summary_md.write_text("\n".join(lines), encoding="utf-8")
    return summary_csv, summary_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trained within-domain experiments.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--output", default="results/within_domain_trained_full_gpu.csv")
    parser.add_argument("--encoders", nargs="+", default=DEFAULT_ENCODERS, choices=["mlp", "transformer", "gcn"])
    parser.add_argument(
        "--representations",
        nargs="+",
        default=DEFAULT_REPRESENTATIONS,
        choices=["raw", "angle", "raw_angle", "graph"],
    )
    parser.add_argument("--shots", nargs="+", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--encoder", default="mlp", choices=["mlp", "transformer", "gcn"])
    parser.add_argument("--representation", default="angle", choices=["raw", "angle", "raw_angle", "graph"])
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=1)
    parser.add_argument("--q_query", type=int, default=5)
    parser.add_argument("--episodes_train", type=int, default=100)
    parser.add_argument("--episodes_eval", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--auto_adjust_q", action="store_true")
    parser.add_argument("--single", action="store_true", help="Use --encoder/--representation/--k_shot only.")
    parser.add_argument("--resume", action="store_true", help="Skip rows already present in --output.")
    args = parser.parse_args()

    if args.single:
        args.encoders = [args.encoder]
        args.representations = [args.representation]
        args.shots = [args.k_shot]

    if "gcn" in args.encoders:
        bad = [rep for rep in args.representations if rep not in {"raw", "graph"}]
        if bad:
            parser.error("gcn supports only raw or graph representation")

    base_cfg = load_config(args.config)
    out_path = REPO_ROOT / args.output
    completed = set()
    if args.resume and out_path.exists():
        with open(out_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                completed.add(
                    (
                        int(row.get("seed", args.seed)),
                        row["dataset"],
                        row["encoder"],
                        row["representation"],
                        int(row["k_shot"]),
                    )
                )
    elif out_path.exists():
        out_path.unlink()

    seeds = args.seeds if args.seeds else [args.seed]
    total = (
        len(seeds)
        * len(args.datasets)
        * len(args.encoders)
        * len(args.representations)
        * len(args.shots)
    )
    count = 0
    original_seed = args.seed
    for seed in seeds:
        args.seed = seed
        for encoder_name in args.encoders:
            for representation in args.representations:
                if encoder_name == "gcn" and representation not in {"raw", "graph"}:
                    continue
                for dataset_name in args.datasets:
                    for shot in args.shots:
                        count += 1
                        key = (seed, dataset_name, encoder_name, representation, shot)
                        if key in completed:
                            print(f"[{count}/{total}] skip existing {key}", flush=True)
                            continue
                        print(f"[{count}/{total}] running {key}", flush=True)
                        row = run_experiment(
                            dataset_name,
                            encoder_name,
                            representation,
                            shot,
                            base_cfg,
                            args,
                        )
                        append_row(out_path, row)
                        print(f"Saved row to {out_path}", flush=True)
    args.seed = original_seed

    md_path = write_markdown(out_path)
    summary_csv, summary_md = write_seed_summary(out_path)
    print(f"Saved {out_path}")
    print(f"Saved {md_path}")
    print(f"Saved {summary_csv}")
    print(f"Saved {summary_md}")


if __name__ == "__main__":
    main()
