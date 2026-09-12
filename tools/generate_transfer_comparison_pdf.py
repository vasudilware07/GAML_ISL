#!/usr/bin/env python3
"""Generate a PDF with LaTeX-style academic tables comparing cross-domain transfer."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def draw_academic_table(ax, title, col_headers, row_data, footnote=None):
    """Draw a clean LaTeX-style academic table (horizontal rules, no vertical lines)."""
    ax.axis("off")

    n_rows = len(row_data)
    n_cols = len(col_headers)

    # Table geometry
    x_left = 0.05
    x_right = 0.95
    y_top = 0.88
    row_h = 0.065
    y_header = y_top
    y_first_data = y_top - row_h

    # Column x-positions (evenly spaced)
    col_xs = [x_left + (x_right - x_left) * (i + 0.5) / n_cols for i in range(n_cols)]

    # Title
    ax.text(0.5, 0.96, title, ha="center", va="top", fontsize=11,
            fontweight="bold", transform=ax.transAxes)

    # Top rule (thick)
    ax.plot([x_left, x_right], [y_top + 0.015, y_top + 0.015],
            color="black", linewidth=1.8, transform=ax.transAxes, clip_on=False)

    # Header text
    for j, h in enumerate(col_headers):
        ax.text(col_xs[j], y_header, h, ha="center", va="center",
                fontsize=9, fontweight="bold", transform=ax.transAxes)

    # Header rule
    ax.plot([x_left, x_right], [y_header - row_h * 0.5, y_header - row_h * 0.5],
            color="black", linewidth=0.8, transform=ax.transAxes, clip_on=False)

    # Data rows
    prev_group = None
    for i, row in enumerate(row_data):
        y = y_first_data - i * row_h

        # Light separator between groups (based on first column value changing)
        if prev_group is not None and row[0] != prev_group and row[0] != "":
            y_sep = y + row_h * 0.55
            ax.plot([x_left, x_right], [y_sep, y_sep],
                    color="#999999", linewidth=0.4, transform=ax.transAxes,
                    clip_on=False, linestyle="-")
        if row[0] != "":
            prev_group = row[0]

        for j, cell in enumerate(row):
            # Bold the best values (marked with *)
            is_bold = False
            display = cell
            if isinstance(cell, str) and cell.startswith("**") and cell.endswith("**"):
                is_bold = True
                display = cell[2:-2]

            ax.text(col_xs[j], y, display, ha="center", va="center",
                    fontsize=8.5,
                    fontweight="bold" if is_bold else "normal",
                    transform=ax.transAxes)

    # Bottom rule (thick)
    y_bottom = y_first_data - (n_rows - 0.4) * row_h
    ax.plot([x_left, x_right], [y_bottom, y_bottom],
            color="black", linewidth=1.8, transform=ax.transAxes, clip_on=False)

    # Footnote
    if footnote:
        ax.text(x_left, y_bottom - 0.03, footnote, ha="left", va="top",
                fontsize=7.5, fontstyle="italic", color="#444444",
                transform=ax.transAxes)


def fmt(acc, ci):
    return f"{acc*100:.1f}\u00b1{ci*100:.1f}"


def fmt_bold(acc, ci):
    return f"**{acc*100:.1f}\u00b1{ci*100:.1f}**"


def main():
    out_dir = Path("results/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 14))
    fig.suptitle(
        "Cross-Lingual Transfer Comparison\n"
        "MLP Encoder \u00b7 5-way 5-shot \u00b7 Q = 15 \u00b7 600 episodes \u00b7 seed 42",
        fontsize=13, fontweight="bold", y=0.99, va="top"
    )

    # -- Table 1: ASL -> Targets (frozen + adapted) ----------------------
    ax1 = fig.add_axes([0.02, 0.68, 0.96, 0.28])
    draw_academic_table(ax1,
        "Table 1.  ASL \u2192 Targets",
        ["Target", "Repr", "Frozen", "Target-sup."],
        [
            ["LIBRAS",  "raw",       fmt(0.865, 0.0076),       fmt(0.9421, 0.0053)],
            ["",        "angle",     fmt_bold(0.9501, 0.0044), fmt_bold(0.9595, 0.0040)],
            ["",        "raw_angle", fmt(0.8776, 0.0072),      fmt(0.9517, 0.0047)],
            ["Arabic",  "raw",       fmt(0.7416, 0.0082),      fmt(0.8940, 0.0061)],
            ["",        "angle",     fmt_bold(0.9129, 0.0054), fmt(0.9267, 0.0049)],
            ["",        "raw_angle", fmt(0.7659, 0.0085),      fmt_bold(0.9286, 0.0048)],
            ["Thai",    "raw",       fmt(0.5252, 0.0085),      fmt_bold(0.5853, 0.0085)],
            ["",        "angle",     fmt_bold(0.5321, 0.0083), fmt(0.5740, 0.0080)],
            ["",        "raw_angle", fmt(0.5058, 0.0081),      fmt(0.5733, 0.0081)],
        ],
        "Bold = best per target per mode."
    )

    # -- Table 2: LIBRAS -> Arabic ----------------------------------------
    ax2 = fig.add_axes([0.02, 0.46, 0.96, 0.18])
    draw_academic_table(ax2,
        "Table 2.  LIBRAS \u2192 Arabic",
        ["Target", "Repr", "Frozen", "Target-sup."],
        [
            ["Arabic",  "raw",       fmt(0.8978, 0.0061),      fmt(0.9180, 0.0056)],
            ["",        "angle",     fmt(0.9123, 0.0055),      fmt(0.9220, 0.0053)],
            ["",        "raw_angle", fmt_bold(0.9167, 0.0055), fmt_bold(0.9384, 0.0047)],
        ],
        "Bold = best per mode."
    )

    # -- Table 3: Arabic -> LIBRAS ----------------------------------------
    ax3 = fig.add_axes([0.02, 0.27, 0.96, 0.18])
    draw_academic_table(ax3,
        "Table 3.  Arabic \u2192 LIBRAS",
        ["Target", "Repr", "Frozen", "Target-sup."],
        [
            ["LIBRAS",  "raw",       fmt(0.9589, 0.0046),      fmt(0.9654, 0.0044)],
            ["",        "angle",     fmt(0.9487, 0.0049),      fmt(0.9587, 0.0043)],
            ["",        "raw_angle", fmt_bold(0.9713, 0.0037), fmt_bold(0.9738, 0.0037)],
        ],
        "Bold = best per mode."
    )

    # -- Table 4: Side-by-side best frozen comparison ---------------------
    ax4 = fig.add_axes([0.02, 0.02, 0.96, 0.22])
    draw_academic_table(ax4,
        "Table 4.  Best Frozen Accuracy by Source (\u0394 vs ASL baseline)",
        ["Source \u2192 Target", "Repr", "Accuracy", "\u0394 vs ASL"],
        [
            ["ASL \u2192 LIBRAS",    "angle",     fmt(0.9501, 0.0044), "\u2014"],
            ["Arabic \u2192 LIBRAS", "raw_angle", fmt_bold(0.9713, 0.0037), "**+2.1 pp**"],
            ["", "", "", ""],
            ["ASL \u2192 Arabic",    "angle",     fmt(0.9129, 0.0054), "\u2014"],
            ["LIBRAS \u2192 Arabic", "raw_angle", fmt_bold(0.9167, 0.0055), "**+0.4 pp**"],
        ],
        "Both non-ASL sources match or exceed the ASL baseline. raw_angle is best for non-ASL sources."
    )

    out_path = out_dir / "cross_lingual_transfer_comparison.pdf"
    fig.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
