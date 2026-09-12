"""
Export experimental results as LaTeX tables.

Formats JSON result files into publication-quality LaTeX tables
matching the paper's format (Tables 3, 4, 5, 6, 7, 8, 9).

Usage:
    python tools/export_tables.py --results_dir results --output tables.tex
"""

import argparse
import json
import os
from pathlib import Path


def format_accuracy(acc: float, ci: float = None) -> str:
    """Format accuracy with optional confidence interval."""
    if ci is not None:
        return f"{acc:.1f}$\\pm${ci:.1f}"
    return f"{acc:.1f}"


def export_table3_within_domain(results_dir: str) -> str:
    """
    Export Table 3: Within-domain few-shot accuracy.

    Format: Dataset × Representation × Encoder × K-shot
    """
    datasets = ["asl", "libras", "arabic", "thai"]
    reprs = ["raw", "angle", "raw_angle"]
    encoders = ["mlp", "transformer"]
    k_shots = [1, 3, 5]

    header = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{Within-domain few-shot accuracy (\\%). "
        "5-way K-shot, Q=15, ProtoNet (Euclidean), 600 episodes, seed 42.}\n"
        "\\label{tab:within-domain}\n"
        "\\begin{tabular}{ll|ccc|ccc}\n"
        "\\toprule\n"
        "& & \\multicolumn{3}{c|}{MLP Encoder} & "
        "\\multicolumn{3}{c}{Transformer Encoder} \\\\\n"
        "Dataset & Repr. & 1-shot & 3-shot & 5-shot & 1-shot & 3-shot & 5-shot \\\\\n"
        "\\midrule\n"
    )

    body = ""
    for ds in datasets:
        ds_name = {
            "asl": "ASL", "libras": "LIBRAS",
            "arabic": "Arabic", "thai": "Thai"
        }[ds]

        for repr_name in reprs:
            row = f"{ds_name} & {repr_name}"

            for encoder in encoders:
                for k in k_shots:
                    # Try to load result
                    result_file = os.path.join(
                        results_dir,
                        f"{ds}_{encoder}_{repr_name}_{k}shot.json"
                    )
                    if os.path.exists(result_file):
                        with open(result_file) as f:
                            data = json.load(f)
                        row += f" & {format_accuracy(data['accuracy'], data.get('ci_95'))}"
                    else:
                        row += " & --"

            body += row + " \\\\\n"

        if ds != datasets[-1]:
            body += "\\midrule\n"

    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )

    return header + body + footer


def export_table6_multi_source(results_dir: str) -> str:
    """
    Export Table 6: Best accuracy by source language.

    Format: Target × Source matrix with best representation noted.
    """
    datasets = ["asl", "libras", "arabic", "thai"]
    reprs = ["raw", "angle", "raw_angle"]
    repr_abbrev = {"raw": "r", "angle": "a", "raw_angle": "ra"}

    header = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Best accuracy (\\%) by source language (MLP, 5-way 5-shot).}\n"
        "\\label{tab:multi-source}\n"
        "\\begin{tabular}{l|cccc}\n"
        "\\toprule\n"
        "Target & ASL & LIBRAS & Arabic & Thai \\\\\n"
        "\\midrule\n"
    )

    body = ""
    cross_dir = os.path.join(results_dir, "cross_lingual")

    for target in datasets:
        target_name = target.upper() if target != "arabic" else "Arabic"
        if target == "thai":
            target_name = "Thai"
        row = f"{target_name}"

        for source in datasets:
            if source == target:
                # Within-domain baseline
                best_acc = 0
                best_repr = ""
                for repr_name in reprs:
                    result_file = os.path.join(
                        results_dir,
                        f"{target}_mlp_{repr_name}_5shot.json"
                    )
                    if os.path.exists(result_file):
                        with open(result_file) as f:
                            data = json.load(f)
                        if data["accuracy"] > best_acc:
                            best_acc = data["accuracy"]
                            best_repr = repr_name

                ci_str = ""
                row += f" & {best_acc:.1f}$^\\dagger$({repr_abbrev.get(best_repr, '')})"
            else:
                # Cross-lingual transfer
                best_acc = 0
                best_repr = ""
                for repr_name in reprs:
                    result_file = os.path.join(
                        cross_dir,
                        f"{source}_to_{target}_mlp_{repr_name}_frozen.json"
                    )
                    if os.path.exists(result_file):
                        with open(result_file) as f:
                            data = json.load(f)
                        if data["accuracy"] > best_acc:
                            best_acc = data["accuracy"]
                            best_repr = repr_name

                if best_acc > 0:
                    row += f" & {best_acc:.1f}({repr_abbrev.get(best_repr, '')})"
                else:
                    row += " & --"

        body += row + " \\\\\n"

    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    return header + body + footer


def main():
    parser = argparse.ArgumentParser(
        description="Export results as LaTeX tables."
    )
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Directory containing result JSON files.",
    )
    parser.add_argument(
        "--output", type=str, default="tables.tex",
        help="Output LaTeX file.",
    )

    args = parser.parse_args()

    tables = []

    # Table 3: Within-domain
    print("Generating Table 3 (within-domain)...")
    tables.append(export_table3_within_domain(args.results_dir))

    # Table 6: Multi-source transfer
    print("Generating Table 6 (multi-source transfer)...")
    tables.append(export_table6_multi_source(args.results_dir))

    # Write output
    with open(args.output, "w") as f:
        f.write("% Auto-generated LaTeX tables\n")
        f.write("% Geometry-Aware Metric Learning for Cross-Lingual Few-Shot SLR\n\n")
        for table in tables:
            f.write(table)
            f.write("\n\n")

    print(f"LaTeX tables saved: {args.output}")


if __name__ == "__main__":
    main()
