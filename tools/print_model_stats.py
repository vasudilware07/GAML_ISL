#!/usr/bin/env python3
"""
Print model architecture statistics (parameter counts) for all
encoder × representation combinations.

Generates a Markdown table and optionally a LaTeX table fragment.

Usage:
    python tools/print_model_stats.py
    python tools/print_model_stats.py --latex paper/tables/tab_architecture.tex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models import build_encoder


REPRESENTATIONS = ["raw", "angle", "raw_angle"]
ENCODERS = ["mlp", "transformer"]
REPR_DIM = {"raw": 63, "angle": 20, "raw_angle": 83}


def build_cfg(encoder: str) -> dict:
    """Minimal config for encoder construction."""
    return {
        "model": {
            "encoder": encoder,
            "embedding_dim": 128,
            "mlp":         {"hidden_dims": [256, 256], "dropout": 0.3},
            "transformer": {"num_heads": 4, "num_layers": 2,
                            "dim_feedforward": 256, "dropout": 0.1},
            "gcn":         {"hidden_dim": 128, "num_layers": 3, "dropout": 0.2},
        },
        "dataset": {"num_landmarks": 21, "landmark_dim": 3},
    }


def count_params(module) -> Tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def collect_stats() -> List[Dict]:
    """Collect parameter counts for every encoder × repr combo."""
    rows = []
    for enc in ENCODERS:
        cfg = build_cfg(enc)
        for rep in REPRESENTATIONS:
            encoder = build_encoder(cfg, rep)
            total, trainable = count_params(encoder)
            rows.append({
                "encoder": enc,
                "repr": rep,
                "input_dim": REPR_DIM[rep],
                "total_params": total,
                "trainable_params": trainable,
            })
    # GCN (raw only)
    cfg_gcn = build_cfg("gcn")
    encoder_gcn = build_encoder(cfg_gcn, "raw")
    total, trainable = count_params(encoder_gcn)
    rows.append({
        "encoder": "gcn",
        "repr": "raw",
        "input_dim": "21×3",
        "total_params": total,
        "trainable_params": trainable,
    })
    return rows


def print_markdown(rows: List[Dict]) -> str:
    """Print Markdown table to stdout and return it."""
    lines = []
    lines.append("| Encoder | Repr | Input Dim | Total Params | Trainable |")
    lines.append("|---------|------|-----------|-------------|-----------|")
    for r in rows:
        lines.append(
            f"| {r['encoder']:12s} | {str(r['repr']):10s} | {str(r['input_dim']):>9s} "
            f"| {r['total_params']:>11,d} | {r['trainable_params']:>9,d} |"
        )
    table = "\n".join(lines)
    print(table)
    return table


def write_latex(rows: List[Dict], outpath: str) -> None:
    """Write LaTeX table fragment."""
    lines = [
        r"\begin{table}[t]",
        r"  \caption{Encoder parameter counts by representation. "
        r"All encoders output 128-D embeddings.}",
        r"  \label{tab:architecture}",
        r"  \centering\small",
        r"  \begin{tabular}{@{}llrr@{}}",
        r"    \toprule",
        r"    Encoder & Repr & Input Dim & Parameters \\",
        r"    \midrule",
    ]
    for r in rows:
        enc_disp = r["encoder"].upper() if r["encoder"] == "gcn" else r["encoder"].capitalize()
        lines.append(
            f"    {enc_disp} & \\texttt{{{r['repr']}}} "
            f"& {r['input_dim']} & {r['total_params']:,d} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nLaTeX table written to {outpath}")


def main():
    parser = argparse.ArgumentParser(description="Print model architecture stats")
    parser.add_argument("--latex", type=str, default=None,
                        help="Path to write LaTeX table fragment")
    args = parser.parse_args()

    rows = collect_stats()
    print_markdown(rows)

    if args.latex:
        write_latex(rows, args.latex)


if __name__ == "__main__":
    main()
