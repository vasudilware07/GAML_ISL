#!/usr/bin/env python3
"""
Export result CSVs to LaTeX table fragments for the paper.

Reads the canonical CSVs in results/ and writes .tex files that can be
\\input{} directly in the paper.  Each table is a standalone tabular
environment (no \\begin{table} wrapper — add that in the paper).

Usage:
    python tools/export_tables.py                  # all tables
    python tools/export_tables.py --table within   # one table
    python tools/export_tables.py --outdir paper/tables

Generated files:
    <outdir>/tab_within.tex          – 24-row within-domain matrix
    <outdir>/tab_cross.tex           – cross-domain transfer
    <outdir>/tab_ablation.tex        – normalisation ablation
    <outdir>/tab_linear.tex          – linear classifier baseline
    <outdir>/tab_robust.tex          – multi-seed robustness
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent


# ═════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pct(val: str, digits: int = 1) -> str:
    """Convert a string accuracy (0-1 float or already %) to display %."""
    try:
        v = float(val)
    except (ValueError, TypeError):
        return "---"
    if v <= 1.0:
        v *= 100
    return f"{v:.{digits}f}"


def _ci(val: str, digits: int = 1) -> str:
    try:
        v = float(val)
    except (ValueError, TypeError):
        return ""
    if v <= 1.0:
        v *= 100
    return f"{v:.{digits}f}"


def _bold(val: str) -> str:
    return f"\\textbf{{{val}}}"


# ═════════════════════════════════════════════════════════════════════════════
#  Within-domain matrix (Table 2 in paper)
# ═════════════════════════════════════════════════════════════════════════════

def export_within(outdir: Path) -> None:
    csv_path = REPO_ROOT / "results" / "matrix_final.csv"
    if not csv_path.exists():
        print(f"SKIP within: {csv_path} not found")
        return

    rows = _read_csv(csv_path)

    DS_ORDER = ["asl_alphabet", "libras_alphabet", "arabic_sign_alphabet", "thai_fingerspelling"]
    DS_LABEL = {"asl_alphabet": "ASL", "libras_alphabet": "LIBRAS",
                "arabic_sign_alphabet": "Arabic", "thai_fingerspelling": "Thai"}
    REPRS = ["raw", "angle", "raw_angle"]
    REPR_TEX = {"raw": "\\texttt{raw}", "angle": "\\texttt{angle}",
                "raw_angle": "\\texttt{raw\\_angle}"}
    ENCS = ["mlp", "transformer"]
    SHOTS = [1, 3, 5]

    # Index: (ds, enc, repr, shot) → row
    idx = {}
    for r in rows:
        key = (r["dataset"], r["encoder"], r["representation"], int(r["k_shot"]))
        idx[key] = r

    # Find best 5-shot per dataset
    best_5 = {}
    for ds in DS_ORDER:
        best_acc = -1
        for enc in ENCS:
            for rep in REPRS:
                r = idx.get((ds, enc, rep, 5))
                if r:
                    acc = float(r["accuracy_mean"])
                    if acc > best_acc:
                        best_acc = acc
                        best_5[ds] = (enc, rep)

    lines = []
    lines.append("\\begin{tabular}{@{}ll ccc ccc@{}}")
    lines.append("\\toprule")
    lines.append("& & \\multicolumn{3}{c}{MLP Encoder} & \\multicolumn{3}{c}{Transformer Encoder} \\\\")
    lines.append("\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}")
    lines.append("Dataset & Repr.& 1-shot & 3-shot & 5-shot & 1-shot & 3-shot & 5-shot \\\\")
    lines.append("\\midrule")

    for di, ds in enumerate(DS_ORDER):
        if di > 0:
            lines.append("\\midrule")
        for ri, rep in enumerate(REPRS):
            prefix = f"\\multirow{{3}}{{*}}{{{DS_LABEL[ds]}}}" if ri == 0 else ""
            cells = []
            for enc in ENCS:
                for shot in SHOTS:
                    r = idx.get((ds, enc, rep, shot))
                    if r:
                        acc = _pct(r["accuracy_mean"])
                        ci = _ci(r["ci95"])
                        cell = f"{acc}\\ci{{{ci}}}"
                        # Bold if this is the best (enc, rep) for this ds at 5-shot
                        if shot == 5 and best_5.get(ds) == (enc, rep):
                            cell = f"\\textbf{{{acc}}}\\ci{{{ci}}}"
                    else:
                        cell = "---"
                    cells.append(cell)
            line = f"  {prefix}"
            line += f"\n    & {REPR_TEX[rep]} & " + " & ".join(cells) + " \\\\"
            lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    out = outdir / "tab_within.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out} ({len(rows)} source rows)")


# ═════════════════════════════════════════════════════════════════════════════
#  Cross-domain (Table 3)
# ═════════════════════════════════════════════════════════════════════════════

def export_cross(outdir: Path) -> None:
    csv_path = REPO_ROOT / "results" / "cross_domain.csv"
    if not csv_path.exists():
        print(f"SKIP cross: {csv_path} not found")
        return

    rows = _read_csv(csv_path)
    TARGET_LABEL = {
        "asl_alphabet": "ASL", "libras_alphabet": "LIBRAS",
        "arabic_sign_alphabet": "Arabic", "thai_fingerspelling": "Thai",
    }

    lines = []
    lines.append("\\begin{tabular}{@{}lcc@{}}")
    lines.append("\\toprule")
    lines.append("Source $\\to$ Target & Accuracy (\\%) & 95\\% CI \\\\")
    lines.append("\\midrule")

    for r in rows:
        tgt = r.get("target_dataset", "")
        tgt_label = TARGET_LABEL.get(tgt, tgt)
        acc = _pct(r["accuracy_mean"])
        ci = _ci(r["ci95"])
        lines.append(f"ASL $\\to$ {tgt_label} & \\textbf{{{acc}}} & $\\pm${ci} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    out = outdir / "tab_cross.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out} ({len(rows)} rows)")


# ═════════════════════════════════════════════════════════════════════════════
#  Ablation (Table 4)
# ═════════════════════════════════════════════════════════════════════════════

def export_ablation(outdir: Path) -> None:
    csv_path = REPO_ROOT / "results" / "nonorm_ablation.csv"
    if not csv_path.exists():
        print(f"SKIP ablation: {csv_path} not found")
        return

    rows = _read_csv(csv_path)

    # Build lookup: (dataset, representation) → accuracy_mean
    idx = {}
    for r in rows:
        key = (r["dataset"], r["representation"])
        idx[key] = _pct(r["accuracy_mean"])

    # Paired table: nonorm vs normalised from matrix_final
    matrix_path = REPO_ROOT / "results" / "matrix_final.csv"
    norm_idx = {}
    if matrix_path.exists():
        for r in _read_csv(matrix_path):
            if r["encoder"] == "mlp" and int(r["k_shot"]) == 5:
                key = (r["dataset"], r["representation"])
                norm_idx[key] = _pct(r["accuracy_mean"])

    DS_PAIRS = [
        ("libras_alphabet_nonorm", "libras_alphabet", "LIBRAS"),
        ("arabic_sign_alphabet_nonorm", "arabic_sign_alphabet", "Arabic"),
    ]

    lines = []
    lines.append("\\begin{tabular}{@{}llccc@{}}")
    lines.append("\\toprule")
    lines.append("Dataset & Repr & No Norm & Normalised & $\\Delta$ \\\\")
    lines.append("\\midrule")

    for nonorm_ds, norm_ds, label in DS_PAIRS:
        for rep in ["raw", "angle"]:
            nn_val = idx.get((nonorm_ds, rep), "---")
            n_val = norm_idx.get((norm_ds, rep), "---")
            try:
                delta = float(n_val) - float(nn_val)
                delta_str = f"${'+'if delta>=0 else ''}{delta:.1f}$"
            except ValueError:
                delta_str = "---"
            bold_nn = _bold(nn_val) if rep == "angle" else nn_val
            bold_n = _bold(n_val) if rep == "angle" else n_val
            lines.append(
                f"{label} & \\texttt{{{rep}}} & {bold_nn} & {bold_n} & {delta_str} \\\\"
            )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    out = outdir / "tab_ablation.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


# ═════════════════════════════════════════════════════════════════════════════
#  Linear baseline (Table 6)
# ═════════════════════════════════════════════════════════════════════════════

def export_linear(outdir: Path) -> None:
    csv_path = REPO_ROOT / "results" / "baseline_linear.csv"
    if not csv_path.exists():
        print(f"SKIP linear: {csv_path} not found")
        return

    rows = _read_csv(csv_path)

    # Index: (dataset, repr) → test_acc
    idx = {}
    for r in rows:
        key = (r["dataset"], r["representation"])
        idx[key] = r["test_acc"]

    # Also get ProtoNet 5-shot angle from matrix
    matrix_path = REPO_ROOT / "results" / "matrix_final.csv"
    proto_idx = {}
    if matrix_path.exists():
        for r in _read_csv(matrix_path):
            if r["encoder"] == "mlp" and r["representation"] == "angle" and int(r["k_shot"]) == 5:
                proto_idx[r["dataset"]] = _pct(r["accuracy_mean"])

    DS_ORDER = ["asl_alphabet", "libras_alphabet", "arabic_sign_alphabet", "thai_fingerspelling"]
    DS_LABEL = {"asl_alphabet": "ASL", "libras_alphabet": "LIBRAS",
                "arabic_sign_alphabet": "Arabic", "thai_fingerspelling": "Thai"}

    lines = []
    lines.append("\\begin{tabular}{@{}lcc@{}}")
    lines.append("\\toprule")
    lines.append("Dataset & Linear (full data) & ProtoNet (5-shot) \\\\")
    lines.append("\\midrule")

    for ds in DS_ORDER:
        linear_val = _pct(idx.get((ds, "angle"), ""), 1)
        proto_val = proto_idx.get(ds, "---")
        lines.append(f"{DS_LABEL[ds]} & {linear_val} & {proto_val} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    out = outdir / "tab_linear.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


# ═════════════════════════════════════════════════════════════════════════════
#  Robustness (Table 7)
# ═════════════════════════════════════════════════════════════════════════════

def export_robust(outdir: Path) -> None:
    csv_path = REPO_ROOT / "results" / "robustness_seeds.csv"
    if not csv_path.exists():
        print(f"SKIP robust: {csv_path} not found")
        return

    rows = _read_csv(csv_path)

    DS_LABEL = {"asl_alphabet": "ASL", "libras_alphabet": "LIBRAS",
                "arabic_sign_alphabet": "Arabic", "thai_fingerspelling": "Thai"}
    SEEDS = [42, 1337, 2024]

    lines = []
    lines.append("\\begin{tabular}{@{}lcccc@{}}")
    lines.append("\\toprule")
    lines.append("Dataset & Seed 42 & Seed 1337 & Seed 2024 & Mean $\\pm$ Std \\\\")
    lines.append("\\midrule")

    for r in rows:
        ds = r["dataset"]
        label = DS_LABEL.get(ds, ds)
        seed_vals = [_pct(r.get(f"acc_seed_{s}", ""), 1) for s in SEEDS]
        mean_val = _pct(r.get("mean", ""), 1)
        std_val = _pct(r.get("std", ""), 1)
        lines.append(
            f"{label} & {' & '.join(seed_vals)} & {mean_val} $\\pm$ {std_val} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    out = outdir / "tab_robust.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


# ═════════════════════════════════════════════════════════════════════════════
#  Extended baselines (episode-linear + input-space)
# ═════════════════════════════════════════════════════════════════════════════

def export_baselines_extended(outdir: Path) -> None:
    """Unified baseline comparison table: ProtoNet vs episode-linear vs input-space."""
    ep_csv = REPO_ROOT / "results" / "baseline_episode_linear.csv"
    inp_csv = REPO_ROOT / "results" / "baseline_input_space.csv"
    matrix_csv = REPO_ROOT / "results" / "matrix_final.csv"

    if not ep_csv.exists() and not inp_csv.exists():
        print("SKIP baselines_ext: no episode_linear or input_space CSVs found")
        return

    DS_ORDER = ["asl_alphabet", "libras_alphabet", "arabic_sign_alphabet", "thai_fingerspelling"]
    DS_LABEL = {"asl_alphabet": "ASL", "libras_alphabet": "LIBRAS",
                "arabic_sign_alphabet": "Arabic", "thai_fingerspelling": "Thai"}

    # ProtoNet 5-shot from matrix (MLP encoder)
    proto = {}
    if matrix_csv.exists():
        for r in _read_csv(matrix_csv):
            if r["encoder"] == "mlp" and int(r["k_shot"]) == 5:
                key = (r["dataset"], r["representation"])
                proto[key] = (r["accuracy_mean"], r["ci95"])

    # Episode-linear
    ep_lin = {}
    if ep_csv.exists():
        for r in _read_csv(ep_csv):
            key = (r["dataset"], r["representation"])
            ep_lin[key] = (r["accuracy"], r["ci"])

    # Input-space
    inp_sp = {}
    if inp_csv.exists():
        for r in _read_csv(inp_csv):
            key = (r["dataset"], r["representation"])
            inp_sp[key] = (r["accuracy"], r["ci"])

    lines = []
    lines.append("\\begin{tabular}{@{}llccc@{}}")
    lines.append("\\toprule")
    lines.append("Dataset & Repr & Input-Space & Episode-Linear & ProtoNet (MLP) \\\\")
    lines.append("\\midrule")

    for ds in DS_ORDER:
        for rep in ["raw", "angle"]:
            label = DS_LABEL.get(ds, ds)
            cells = []
            for lookup in [inp_sp, ep_lin, proto]:
                val = lookup.get((ds, rep))
                if val:
                    acc = _pct(val[0])
                    ci = _ci(val[1])
                    cells.append(f"{acc}\\ci{{{ci}}}")
                else:
                    cells.append("---")
            lines.append(f"{label} & \\texttt{{{rep}}} & {' & '.join(cells)} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    out = outdir / "tab_baselines_ext.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

TABLE_MAP = {
    "within": export_within,
    "cross": export_cross,
    "ablation": export_ablation,
    "linear": export_linear,
    "robust": export_robust,
    "baselines_ext": export_baselines_extended,
}


def main():
    parser = argparse.ArgumentParser(
        description="Export result CSVs to LaTeX table fragments.")
    parser.add_argument("--table", type=str, default=None,
                        choices=list(TABLE_MAP.keys()),
                        help="Export a single table (default: all)")
    parser.add_argument("--outdir", type=str, default="paper/tables",
                        help="Output directory for .tex files (default: paper/tables)")
    args = parser.parse_args()

    outdir = REPO_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    if args.table:
        TABLE_MAP[args.table](outdir)
    else:
        for name, fn in TABLE_MAP.items():
            fn(outdir)

    print(f"\nDone. Tables in {outdir}/")


if __name__ == "__main__":
    main()
