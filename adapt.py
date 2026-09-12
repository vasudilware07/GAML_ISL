"""
Few-shot adaptation script for cross-lingual transfer.

Loads a pretrained encoder, freezes or fine-tunes it, and adapts to a
target language with 1-shot or 5-shot support.

Usage:
    python adapt.py --dataset BdSL --shot 1
    python adapt.py --dataset BdSL --shot 5 --mode full_finetune
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import LandmarkDataset, SyntheticLandmarkDataset
from data.episodes import EpisodicSampler, split_support_query, collate_episode
from losses.supcon import build_loss
from models import build_encoder, build_few_shot_model
from utils.logger import get_logger
from utils.metrics import accuracy, few_shot_accuracy_with_ci, plot_tsne
from utils.seed import set_seed
from train import get_dataset, get_device, load_config


def freeze_backbone(model, mode: str = "freeze") -> None:
    """Apply adaptation strategy to model parameters.

    Args:
        model: Few-shot model with ``.encoder`` attribute.
        mode: One of ``'freeze'``, ``'finetune_last'``, ``'full_finetune'``.
    """
    if mode == "freeze":
        for param in model.encoder.parameters():
            param.requires_grad = False
    elif mode == "finetune_last":
        # Freeze all except last layer
        params = list(model.encoder.parameters())
        for param in params[:-2]:  # freeze all but last weight + bias
            param.requires_grad = False
    elif mode == "full_finetune":
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown adaptation mode: {mode}")


def adapt(
    model,
    dataset,
    loss_fn,
    device: torch.device,
    cfg: dict,
    n_way: int,
    k_shot: int,
    q_query: int,
    adapt_epochs: int = 50,
    adapt_lr: float = 1e-4,
) -> Dict[str, float]:
    """Adapt model to target dataset using few-shot episodes.

    Args:
        model: Pretrained few-shot model.
        dataset: Target dataset.
        loss_fn: Metric learning loss.
        device: Device.
        cfg: Config dict.
        n_way: Ways.
        k_shot: Shots.
        q_query: Queries.
        adapt_epochs: Number of adaptation epochs.
        adapt_lr: Adaptation learning rate.

    Returns:
        Dict with final accuracy and CI.
    """
    # Only optimise trainable parameters
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=adapt_lr, weight_decay=1e-4)

    labels = [dataset[i][1] for i in range(len(dataset))]
    sampler = EpisodicSampler(
        labels, n_way=n_way, k_shot=k_shot, q_query=q_query,
        episodes=min(100, cfg["few_shot"].get("episodes_train", 100)),
    )
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_episode, num_workers=0)

    best_acc = 0.0
    best_state = None

    # Reduce adaptation epochs for efficiency
    adapt_epochs = min(adapt_epochs, 20)

    for epoch in range(1, adapt_epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        n_ep = 0

        for batch in loader:
            data, lbl = batch
            data, lbl = data.to(device), lbl.to(device)
            sx, sy, qx, qy = split_support_query((data, lbl), n_way, k_shot, q_query)
            sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)

            log_probs = model(sx, sy, qx, n_way)
            cls_loss = F.nll_loss(log_probs, qy)

            all_x = torch.cat([sx, qx])
            all_y = torch.cat([sy, qy])
            emb = model.get_embeddings(all_x)
            m_loss = loss_fn(emb, all_y)

            loss = cls_loss + 0.5 * m_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_acc += accuracy(log_probs, qy)
            n_ep += 1

        epoch_acc = total_acc / max(n_ep, 1)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    eval_sampler = EpisodicSampler(
        labels, n_way=n_way, k_shot=k_shot, q_query=q_query,
        episodes=min(500, cfg["few_shot"].get("episodes_eval", 500)),
    )
    eval_loader = DataLoader(dataset, batch_sampler=eval_sampler, collate_fn=collate_episode, num_workers=0)

    accs = []
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Eval adapt", leave=False):
            data, lbl = batch
            data, lbl = data.to(device), lbl.to(device)
            sx, sy, qx, qy = split_support_query((data, lbl), n_way, k_shot, q_query)
            sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)
            log_probs = model(sx, sy, qx, n_way)
            accs.append(accuracy(log_probs, qy))

    mean_acc, ci = few_shot_accuracy_with_ci(accs)
    return {"accuracy": mean_acc, "ci": ci}


def main():
    parser = argparse.ArgumentParser(description="Few-shot adaptation")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--dataset", type=str, default="Thai")
    parser.add_argument("--shot", type=int, default=1, choices=[1, 5])
    parser.add_argument("--mode", type=str, default="finetune_last",
                        choices=["freeze", "finetune_last", "full_finetune"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg)

    os.makedirs(cfg.get("output_dir", "results"), exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    logger = get_logger("adapt", "results/adapt.log")

    # Build model
    representation = cfg.get("representation", "raw")
    encoder = build_encoder(cfg, representation)
    model = build_few_shot_model(cfg, encoder).to(device)

    # Load pretrained checkpoint
    ckpt_path = args.checkpoint or os.path.join(
        cfg.get("checkpoint_dir", "results/checkpoints"),
        f"best_{cfg['dataset']['name'].lower()}.pt",
    )
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded pretrained checkpoint: {ckpt_path}")
    else:
        logger.warning(f"No checkpoint at {ckpt_path}, adapting from scratch")

    # Apply adaptation strategy
    freeze_backbone(model, args.mode)
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Adaptation mode: {args.mode} ({trainable_count:,}/{total_count:,} trainable)")

    # Target dataset
    target_cfg = cfg.copy()
    target_cfg["dataset"] = {
        **cfg["dataset"],
        "name": args.dataset,
        "root": f"data/processed/{args.dataset.lower()}",
    }
    target_ds = get_dataset(target_cfg, split="train")

    # Loss
    loss_fn = build_loss(cfg).to(device)

    n_way = cfg["few_shot"]["n_way"]
    q_query = cfg["few_shot"]["q_query"]

    logger.info(f"Adapting to {args.dataset} with {args.shot}-shot, mode={args.mode}")

    results = adapt(
        model, target_ds, loss_fn, device, cfg,
        n_way=n_way, k_shot=args.shot, q_query=q_query,
        adapt_epochs=cfg["adaptation"].get("epochs", 50),
        adapt_lr=cfg["adaptation"].get("lr", 1e-4),
    )
    logger.info(f"Adapted accuracy: {results['accuracy']:.4f} ± {results['ci']:.4f}")

    # t-SNE plot
    from evaluate import collect_embeddings
    embeddings, labels = collect_embeddings(model, target_ds, device)
    plot_tsne(
        embeddings, labels,
        save_path=f"results/plots/tsne_adapt_{args.dataset}_{args.shot}shot.png",
        title=f"t-SNE: {args.dataset} {args.shot}-shot ({args.mode})",
    )

    # Save results
    csv_path = os.path.join(cfg.get("output_dir", "results"), "few_shot.csv")
    header = ["dataset", "shot", "mode", "accuracy", "ci"]
    row = {
        "dataset": args.dataset,
        "shot": args.shot,
        "mode": args.mode,
        "accuracy": f"{results['accuracy']:.4f}",
        "ci": f"{results['ci']:.4f}",
    }
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    logger.info(f"Results appended to {csv_path}")

    # Save adapted checkpoint
    adapt_ckpt = os.path.join(
        cfg.get("checkpoint_dir", "results/checkpoints"),
        f"adapted_{args.dataset.lower()}_{args.shot}shot_{args.mode}.pt",
    )
    torch.save({
        "model_state_dict": model.state_dict(),
        "shot": args.shot,
        "mode": args.mode,
        "accuracy": results["accuracy"],
        "config": cfg,
    }, adapt_ckpt)
    logger.info(f"Adapted model saved to {adapt_ckpt}")


if __name__ == "__main__":
    main()
