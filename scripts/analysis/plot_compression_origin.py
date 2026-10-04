#!/usr/bin/env python3
"""Render the paper's compression-origin figure (diff_classification_macro).

Reads diff_classification_macro.csv from classify_compression.py: equal-weight
means over repositories, in percent of the baseline tokens. One stacked bar per
model; proof simplification and structural diff are drawn together as "Proof
rewrite", automation and syntax optimization form the warm cap, and added
declarations fall below zero. Renderer and style are the paper's.

Usage:
    python scripts/analysis/plot_compression_origin.py <dir>/diff_classification_macro.csv

PNG and SVG always use Matplotlib's own text; --tex sets the PDF in LaTeX
(needs a TeX installation).
"""
from __future__ import annotations

import argparse
import csv
import math
import shutil
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# Draw bottom to top; automation and syntax form the warm-colored cap.
BARS = (("added", "Added declarations", "#AD7843"),
        ("dead_code", "Dead code", "#173F55"),
        ("deleted", "Deleted declarations", "#327F91"),
        ("proof_rewrite", "Proof rewrite", "#69B6A1"),
        ("automation", "Automation rewrite", "#E77768"),
        ("syntax_optimization", "Syntax optimization", "#BB3048"))


def pct(value: float, tex: bool) -> str:
    return f"{value:.1f}\\%" if tex else f"{value:.1f}%"


def bars(ax, data, *, tex: bool, ylabel: str, parts=BARS, total: str = "saved",
         total_label=None, percent_labels: bool = False) -> None:
    models = list(data)
    height = max(sum(max(0.0, data[m][k]) for k, _, _ in parts) for m in models)
    tops = []
    for x, model in enumerate(models):
        entry, up, down = data[model], 0.0, 0.0
        for key, _, colour in parts:
            value = entry[key]
            base = up if value >= 0 else down
            ax.bar(x, value, bottom=base, width=0.64, color=colour, edgecolor="none", linewidth=0, antialiased=False, zorder=3)
            if abs(value) >= 0.07 * height:
                ax.text(x, base + value / 2, f"{value:.1f}%" if percent_labels else f"{value:.1f}",
                        ha="center", va="center", fontsize=6.8,
                        color="#143B3A" if key in ("proof_rewrite", "automation") else "white", zorder=5)
            if value >= 0:
                up += value
            else:
                down += value
        label = total_label(entry[total]) if total_label else pct(entry[total], tex)
        ax.text(x, up + 0.012 * height, label, ha="center", va="bottom", fontsize=7.5,
                color="#111111", zorder=5)
        tops.append(up)
    ax.axhline(0, color="#333333", linewidth=0.6, zorder=2)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([data[m]["label"] for m in models])
    ax.set_xlim(-0.5, len(models) - 0.5)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.spines["bottom"].set_visible(False)
    ax.set_ylim(top=max(tops) * 1.14)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    ax.set_axisbelow(True)


def classification_style(*, use_tex: bool) -> None:
    """The paper's figure style (experiments/.../plot_benchmark.py)."""
    if use_tex and not shutil.which("latex"):
        raise RuntimeError("latex is required; pass --no-tex for a preview")
    plt.rcParams.update({
        "text.usetex": use_tex,
        # Match the Times face loaded by the ICLR paper style.  Keeping the
        # same preamble here makes standalone PDFs retain the paper typography.
        "text.latex.preamble": r"\usepackage{times}\usepackage{amsmath}",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman"],
        "mathtext.fontset": "cm",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.65,
        "axes.titleweight": "regular",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.bottom": False,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "legend.fontsize": 7.2,
        "legend.frameon": False,
        "hatch.linewidth": 0.6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_classification(rows: list[dict], use_tex: bool, *, compact: bool = True,
                        legend_layout: str | None = None) -> plt.Figure:
    # CSVs retain the detailed categories; combine only for presentation.
    # The legacy proof_rewrites total already includes automation, so do not
    # use it here: automation now has its own stack segment.
    fields = ("saved", "dead_code", "deleted", "added", "syntax_optimization",
              "automation", "proof_simplification", "structural_diff")
    data = {row["model"]: {**row, **{key: number(row, key) for key in fields},
            "proof_rewrite": number(row, "proof_simplification") + number(row, "structural_diff")}
            for row in rows}
    models = list(data)
    legend_layout = legend_layout or ("upper_right" if compact and len(models) >= 5 else "two_rows")
    if legend_layout not in ("upper_right", "two_rows"):
        raise ValueError(f"Unknown classification legend layout: {legend_layout}")
    height = 2.75 if compact and legend_layout == "upper_right" else 3.15
    fig, ax = plt.subplots(figsize=(2.6, height) if compact else (5.5, 3.4))
    ylabel = r"Score (\%)" if use_tex else "Score (%)"
    bars(ax, data, tex=use_tex, ylabel=ylabel, total_label=lambda value: f"{value:.1f}")
    tick_labels = [data[m]["label"].replace(" 3.8 Flash", " 3.8\nFlash")
                   .replace(" 1.3 Spark", " 1.3\nSpark") for m in models]
    label_width = 7 if compact and len(models) >= 5 else 13
    tick_labels = ["\n".join(textwrap.fill(line, width=label_width, break_long_words=False,
                    break_on_hyphens=False) for line in label.split("\n")) for label in tick_labels]
    ax.set_xticks(range(len(models)), tick_labels)
    if compact:
        ax.tick_params(axis="both", labelsize=6.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}%"))
    extra_lines = max(0, max(label.count("\n") + 1 for label in tick_labels) - 2)
    # List categories from the top of the stack to the bottom. Matplotlib fills
    # multicolumn legends down each column, so transpose for row-wise reading.
    legend_parts = list(reversed(BARS))
    if legend_layout == "two_rows":
        legend_parts = [legend_parts[i] for i in (0, 3, 1, 4, 2, 5)]
    handles = [Patch(facecolor=color, edgecolor="none", label=label)
               for _, label, color in legend_parts]
    if legend_layout == "upper_right":
        ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.995),
                  ncol=1, frameon=False, fontsize=5.8 if compact else 7.4,
                  handlelength=1.1, handletextpad=0.4, borderaxespad=0.15,
                  labelspacing=0.45)
    elif compact:
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.14),
                  bbox_transform=fig.transFigure, ncol=3, frameon=False, fontsize=5.6,
                  handlelength=1.1, columnspacing=0.75, handletextpad=0.4)
    else:
        ax.legend(handles=handles, loc="upper center",
                  bbox_to_anchor=(0.5, -0.14 - 0.07 * extra_lines),
                  ncol=3, frameon=False, fontsize=7.4, handlelength=1.1,
                  columnspacing=1.2, handletextpad=0.4)
    if compact:
        plot_side_inches = 2.6 * (0.985 - 0.20)
        fig.subplots_adjust(left=0.20, right=0.985,
                            bottom=0.92 - plot_side_inches / height, top=0.92)
    else:
        fig.subplots_adjust(left=0.115, right=0.985, bottom=0.31 + 0.04 * extra_lines, top=0.91)
    return fig


def number(row: dict, key: str, *, allow_missing: bool = False) -> float:
    if allow_missing and row[key] == "":
        return math.nan
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite {key}: {row}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path, help="diff_classification_macro.csv")
    parser.add_argument("--output", type=Path, help="path without suffix (default: beside the CSV)")
    parser.add_argument("--formats", nargs="+", choices=("pdf", "png", "svg"), default=["pdf", "png"])
    parser.add_argument("--tex", action="store_true", help="typeset the PDF with LaTeX")
    args = parser.parse_args()
    with args.data.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit(f"{args.data}: no models")
    output = args.output or args.data.with_suffix("")
    for fmt in args.formats:
        use_tex = args.tex and fmt == "pdf"
        # Start each figure from clean defaults, as the paper renderer does.
        plt.rcdefaults()
        classification_style(use_tex=use_tex)
        fig = plot_classification(rows, use_tex, compact=True, legend_layout="upper_right")
        target = output.with_suffix("." + fmt)
        fig.savefig(target, dpi=300, metadata={"CreationDate": None, "ModDate": None} if fmt == "pdf" else None)
        plt.close(fig)
        print(target)


if __name__ == "__main__":
    main()
