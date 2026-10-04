#!/usr/bin/env python3
"""Render every data-driven figure and table of the paper from its CSVs.

Reads ``results/paper/data/*.csv`` (the paper's final export, checked against
``results/paper/manifest.json``) and writes the paper's tree under the output
directory:

    figures/experiments/*.pdf|png   figures/leanlean/*.pdf|png
    figures/examples/*.tex          tables/*.tex

The figures reproduce the paper's renderers and layouts. The three figures the
paper sets in LaTeX (frontier and the two by-scale panels) use TeX for their PDF
when ``latex`` is installed; ``--no-tex`` and every PNG use Matplotlib's text.
The example graphs are hand-laid-out TikZ and are copied as committed.

Usage:
    python scripts/paper/render.py [--data-dir results/paper/data] [--output-dir output/paper]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import (AutoMinorLocator, FixedLocator, FuncFormatter,  # noqa: E402
                               LogLocator, MaxNLocator, MultipleLocator)
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = Path(__file__).with_name("templates")
NO_DATES = {"CreationDate": None, "ModDate": None}

BAND_ORDER = ("≤ 10k", "10k–50k", "50k–250k", "> 250k")
BAND_LABELS = dict(zip(BAND_ORDER, ("Compact", "Standard", "Large", "Massive")))
BAND_COLORS = dict(zip(BAND_ORDER, ("#0072B2", "#009E73", "#E69F00", "#CC79A7")))
HISTOGRAM_MODELS = {
    "Opus 5": ("opus-5", "#D97757"),
    "Sol 5.6": ("gpt-5.6-sol", "#10A37F"),
    "Gemini 3.8 Flash": ("gemini-3.8-flash", "#F4B400"),
    "Muse Spark 1.3": ("muse-spark-1.3", "#0866FF"),
    "Luna 5.6": ("gpt-5.6-luna", "#000000"),
}
HISTOGRAMS = {
    "compression-reduction-histograms-compression-reduction": "compression",
    "compression-task-cost-histogram": "cost",
    "compression-task-duration-histogram": "duration",
}
# Draw bottom to top; colors are shared with the subject-family pie.
BARS = (("added", "Added declarations", "#58A98B"),
        ("dead_code", "Dead code", "#173F55"),
        ("deleted", "Deleted declarations", "#2B6F9C"),
        ("proof_rewrite", "Proof rewrite", "#4E8D9D"),
        ("automation", "Automation rewrite", "#D99533"),
        ("syntax_optimization", "Syntax optimization", "#C75D63"))
SERIF = ["Times New Roman", "Nimbus Roman"]


def number(row: dict, key: str, *, allow_missing: bool = False) -> float:
    if allow_missing and row[key] == "":
        return math.nan
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite {key}: {row}")
    return value


def percent_tick(value: float, _position: float) -> str:
    return f"{value:g}" + (r"\%" if plt.rcParams["text.usetex"] else "%")


def tex_preamble(use_tex: bool) -> dict:
    if use_tex and not shutil.which("latex"):
        raise RuntimeError("latex is required for --tex figures")
    return {"text.usetex": use_tex, "text.latex.preamble": r"\usepackage{times}\usepackage{amsmath}"}


# --- Benchmark figures --------------------------------------------------------

def benchmark_style(use_tex: bool) -> None:
    plt.rcParams.update({
        **tex_preamble(use_tex),
        "font.family": "serif", "font.serif": SERIF, "mathtext.fontset": "cm",
        "font.size": 9.4, "axes.titlesize": 9.5, "axes.titleweight": "regular",
        "axes.labelsize": 9.4, "axes.edgecolor": "#333333", "axes.linewidth": 0.65,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": 8.2, "ytick.labelsize": 8.2, "xtick.color": "black", "ytick.color": "black",
        "xtick.major.width": 0.55, "ytick.major.width": 0.55,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "legend.fontsize": 7.9, "legend.frameon": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.edgecolor": "none",
        "savefig.bbox": None, "savefig.pad_inches": 0.0,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def draw_by_scale(axis, rows: list[dict], metric: str) -> None:
    column = {"score": "mean_reduction_pct", "cost": "mean_cost_usd",
              "duration": "mean_duration_hours"}[metric]
    models = list(dict.fromkeys(row["model"] for row in rows))
    values_by_model = [
        [number(next(r for r in rows if r["model"] == model and r["token_band"] == band),
                column, allow_missing=True) for band in BAND_ORDER]
        for model in models]
    firsts = [next(r for r in rows if r["model"] == model) for model in models]
    width = min(0.16, 0.78 / len(firsts))
    centers = list(range(len(BAND_ORDER)))
    legend = {"gpt-5.6-sol": "GPT-5.6 Sol", "gpt-5.6-luna": "GPT-5.6 Luna"}
    for index, (row, values) in enumerate(zip(firsts, values_by_model, strict=True)):
        offset = (index - (len(firsts) - 1) / 2) * width
        axis.bar([c + offset for c in centers], values, width=width * 0.92, color=row["color"],
                 label=legend.get(row["model"], row["label"]))
    axis.set_xticks(centers, [BAND_LABELS[band] for band in BAND_ORDER])
    axis.tick_params(axis="x", labelsize=8.6)
    if metric == "duration":
        axis.set_ylabel("Hours per repo")
    elif metric == "cost":
        axis.set_ylabel("Cost per repo ({})".format(r"\$" if plt.rcParams["text.usetex"] else "$"))
    else:
        axis.set_ylabel(r"Score (\%)" if plt.rcParams["text.usetex"] else "Score (%)")
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.yaxis.set_major_locator(MultipleLocator(20))
    axis.set_ylim(top=max(v for values in values_by_model for v in values if math.isfinite(v))
                  * (1.3 if metric == "score" else 1.5))
    axis.axhline(0, color="#333333", linewidth=0.55)
    axis.grid(False)
    axis.spines["bottom"].set_visible(False)


def performance_by_scale(tables, use_tex):
    benchmark_style(use_tex)
    fig, axis = plt.subplots(figsize=(5.50 * 0.52, 1.45))
    draw_by_scale(axis, tables["benchmark-model-performance-by-scale"], "score")
    handles, labels = axis.get_legend_handles_labels()
    if len(handles) == 5:
        # Pad the first column so two models sit beside three in the second.
        handles = handles[:2] + [Patch(facecolor="none", edgecolor="none")] + handles[2:]
        labels = labels[:2] + [""] + labels[2:]
    axis.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.0, 0.99), ncol=2,
                columnspacing=0.5, fontsize=7.6, handlelength=0.9, labelspacing=0.25,
                handletextpad=0.45, borderaxespad=0)
    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.20, top=0.965)
    return fig


def cost_duration_by_scale(tables, use_tex):
    benchmark_style(use_tex)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.6))
    for axis, stem, metric in zip(axes, ("benchmark-model-cost-by-scale", "benchmark-model-duration-by-scale"),
                                  ("cost", "duration")):
        draw_by_scale(axis, tables[stem], metric)
        axis.set_ylim(top=axis.get_ylim()[1] / 1.5 * 1.08)
        axis.tick_params(axis="x", labelsize=7.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
               fontsize=7.6, handlelength=1.0, columnspacing=1.1, handletextpad=0.4, borderaxespad=0)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.15, top=0.86, wspace=0.26)
    return fig


def subject_families(tables, use_tex):
    """Label positions follow the reference artwork, independently of CSV row order."""
    benchmark_style(use_tex)
    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "xtick.color": "#333333", "ytick.color": "#333333",
                         "legend.fontsize": 7.2})
    # Clockwise from the upper left, on the reference's 644 x 360 canvas (y up).
    layout = {
        "Algebra & number theory": ("Algebra and number theory", "top", 319, 343),
        "Geometry & topology": ("Geometry and\ntopology", "right", 477, 285),
        "Logic & foundations": ("Logic and\nfoundations", "right", 477, 215),
        "Applied & computational math": ("Applied and\ncomputational\nmathematics", "right", 477, 138),
        "Combinatorics & discrete math": ("Combinatorics\nand discrete\nmathematics", "right", 477, 57),
        "Computer science": ("Computer science", "bottom", 277, 18),
        "Physics": ("Physics", "left", 184, 97),
        "Probability & statistics": ("Probability\nand statistics", "left", 184, 175),
        "Analysis & dynamics": ("Analysis and\ndynamics", "left", 184, 255),
    }
    by_family = {row["family"]: row for row in tables["benchmark-primary-subject-families"]}
    if by_family.keys() - layout.keys():
        raise ValueError(f"Subject families without a layout: {sorted(by_family.keys() - layout.keys())}")
    entries = [by_family[family] for family in layout if family in by_family]
    total = sum(int(row["repositories"]) for row in entries)
    fig = plt.figure(figsize=(2.695, 2.695 * 360 / 644))
    axis = fig.add_axes((0, 0, 1, 1))
    cx, cy, radius = 331, 179, 126
    wedges, _, _ = axis.pie(
        [int(row["repositories"]) for row in entries], colors=[row["color"] for row in entries],
        startangle=132, counterclock=False, center=(cx, cy), radius=radius,
        wedgeprops={"edgecolor": "none", "linewidth": 0},
        autopct=lambda pct: str(int(round(pct * total / 100))), pctdistance=0.67,
        textprops={"color": "white", "fontsize": 7.2})
    for wedge, row in zip(wedges, entries, strict=True):
        label, side, text_x, text_y = layout[row["family"]]
        angle = math.radians((wedge.theta1 + wedge.theta2) / 2)
        end_x = text_x + {"left": 8, "right": -7}.get(side, 0)
        end_y = text_y + {"top": -18, "bottom": 18}.get(side, 0)
        xs, ys = [cx + radius * math.cos(angle)], [cy + radius * math.sin(angle)]
        if row["family"] == "Applied & computational math":
            elbow = radius * 1.28 / 1.18
            xs.append(cx + elbow * math.cos(angle))
            ys.append(cy + elbow * math.sin(angle))
        else:
            # Extend along the slice radius, perpendicular to the circle.
            leader = (end_y - cy) / math.sin(angle) if side in ("top", "bottom") else radius + 28
            end_x, end_y = cx + leader * math.cos(angle), cy + leader * math.sin(angle)
            text_x = end_x + {"left": -8, "right": 7}.get(side, 0)
            text_y = end_y + {"top": 18, "bottom": -18}.get(side, 0)
        xs.append(end_x)
        ys.append(end_y)
        axis.plot(xs, ys, color=row["color"], linewidth=0.5, solid_capstyle="round", clip_on=False)
        axis.text(text_x, text_y, label, ha={"left": "right", "right": "left"}.get(side, "center"),
                  va="center", fontsize=7.2, linespacing=1.05, color="black")
    axis.set_xlim(0, 644)
    axis.set_ylim(0, 360)
    axis.set_axis_off()
    return fig


def small_style(size: float, tick: float) -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": SERIF, "mathtext.fontset": "cm",
        "font.size": size, "axes.labelsize": size, "axes.edgecolor": "#333333", "axes.linewidth": 0.65,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": tick, "ytick.labelsize": tick, "xtick.color": "#333333", "ytick.color": "#333333",
        "xtick.major.width": 0.55, "ytick.major.width": 0.55,
        "figure.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def repository_sizes(tables, _use_tex):
    """Repository sizes after preprocessing, stacked by subject family."""
    small_style(8.5, 7.5)
    plt.rcParams["xtick.minor.width"] = 0.4
    rows = tables["benchmark-preprocessing-removal"]
    sizes = np.array([float(row["stripped_tokens"]) for row in rows])
    family_of = {row["repository"]: row["family"] for row in tables["repository-subjects"]}
    families = np.array([family_of[row["repository"]] for row in rows])
    pie = tables["benchmark-primary-subject-families"]
    colors = {row["family"]: row["color"] for row in pie}
    for row in pie:
        if int(row["repositories"]) != int(np.sum(families == row["family"])):
            raise ValueError(f"{row['family']}: the pie and repository-subjects.csv disagree")
    # Stacked bottom to top in the pie's clockwise order. The scale bands
    # (10k, 50k, 250k) are bin edges, each split into five log-width bins.
    order = [f for f in ("Algebra & number theory", "Geometry & topology", "Logic & foundations",
                         "Applied & computational math", "Combinatorics & discrete math",
                         "Computer science", "Physics", "Probability & statistics",
                         "Analysis & dynamics") if f in set(families)]
    fig, axis = plt.subplots(figsize=(2.695, 2.695 * 360 / 644))
    steps = np.arange(np.floor(np.log(sizes.min() / 1e4) / np.log(5) * 5),
                      np.ceil(np.log(sizes.max() / 1e4) / np.log(5) * 5) + 1)
    bins = 1e4 * 5 ** (steps / 5)
    axis.hist([sizes[families == f] for f in order], bins=bins, stacked=True,
              color=[colors[f] for f in order], edgecolor="none", linewidth=0, zorder=3)
    axis.set_xscale("log")
    axis.set_xlim(bins[0], bins[-1])
    axis.xaxis.set_major_locator(FixedLocator([5e3, 1e4, 2e4, 5e4, 1e5, 2.5e5, 5e5, 1e6]))
    axis.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}k"))
    axis.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(1, 10)))
    axis.tick_params(axis="x", which="minor", length=2)
    axis.tick_params(axis="x", which="major", labelsize=6.5, pad=1.5, length=2.5)
    axis.tick_params(axis="y", pad=1.5, length=2.5)
    axis.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    axis.set_xlabel("Lean tokens", labelpad=1)
    axis.set_ylabel("Count", labelpad=1)
    fig.subplots_adjust(left=0.095, right=0.97, bottom=0.215, top=0.97)
    return fig


def preprocessing_removal(tables, _use_tex):
    """Lean tokens removed by preprocessing, stacked by post-preprocessing size band."""
    small_style(8.0, 7.0)
    plt.rcParams.update({"xtick.major.size": 2.5, "ytick.major.size": 2.5})
    rows = tables["benchmark-preprocessing-removal"]
    fig, axis = plt.subplots(figsize=(1.68, 1.30))
    axis.hist([[float(r["reduction_pct"]) for r in rows if r["token_band"] == band] for band in BAND_ORDER],
              bins=range(0, 101, 5), stacked=True, color=list(BAND_COLORS.values()),
              edgecolor="none", linewidth=0, zorder=3, label=list(BAND_LABELS.values()))
    axis.legend(loc="upper right", frameon=False, fontsize=6.5, handlelength=0.8, handleheight=0.8,
                handletextpad=0.4, labelspacing=0.2, borderaxespad=0.1)
    axis.set_xlim(0, 100)
    axis.set_xticks(range(0, 101, 25))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
    axis.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    axis.tick_params(pad=1.5)
    axis.set_xlabel("Lean tokens removed", labelpad=1.5)
    axis.set_ylabel("Count", labelpad=1.5)
    fig.subplots_adjust(left=0.17, right=0.91, bottom=0.25, top=0.97)
    return fig


# --- Experiment figures ---------------------------------------------------------

def histogram_style() -> None:
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42, "axes.titlesize": 9,
                         "axes.labelsize": 9.9, "xtick.labelsize": 7.7, "ytick.labelsize": 7.7})


def introduction(tables, _use_tex):
    histogram_style()
    results = tables["introduction-results"]
    fig = plt.figure(figsize=(2.2, 2.0), facecolor="white")
    ax = fig.add_axes((.04, .31, .92, .64))
    scores = [number(r, "mean_reduction_pct") for r in results]
    ax.bar(range(len(results)), scores, width=.80, color=[r["color"] for r in results], alpha=.88,
           edgecolor="white", linewidth=.45)
    ax.set_xlim(-.5, len(results) - .5)
    ax.set_ylim(min(0, min(scores) - 2), max(54, max(scores) * 1.15))
    ax.set_yticks(())
    ax.set_xticks(())
    labels = {"Gemini 3.8 Flash": "Gemini\n3.8 Flash", "Muse Spark 1.3": "Muse\nSpark 1.3*",
              "Leanstral 1.5": "Leanstral\n1.5"}
    for x, result in enumerate(results):
        fig.text(.04 + .92 * (x + .5) / len(results), .285, labels.get(result["label"], result["label"]),
                 rotation=55, ha="center", va="top", fontsize=7.7, linespacing=1.0)
    ax.grid(False)
    ax.spines[:].set_visible(False)
    for x, score in enumerate(scores):
        ax.text(x, score + 1, f"{score:.1f}%", ha="center", va="bottom", fontsize=7.7, color="black")
    return fig


def plot_histogram(rows: list[dict], metric: str, axes) -> None:
    """Compression and duration fold out-of-range values into the edge bins; cost uses log bins."""
    field, bins, limits, ticks, xlabel, min_count = {
        "compression": ("reduction_pct", list(range(0, 101, 5)), (.01, 99.99), [0, 50, 100],
                        "Compression (%)", 25),
        "duration": ("hours", list(range(13)), (0, 11.999), [0, 3, 6, 9, 12], "Duration (hours)", 40),
        "cost": ("cost_usd", None, None, None, "Cost (USD)", 40),
    }[metric]
    if metric == "cost":
        costs = [number(row, field) for row in rows]
        if min(costs) <= 0:
            raise ValueError("Logarithmic cost histograms require strictly positive costs.")
        low = math.floor(math.log10(min(costs)))
        high = max(low + 1, math.ceil(math.log10(max(costs))))
        bins = np.logspace(low, high, 5 * (high - low) + 1)
        ticks = [10.0 ** exponent for exponent in range(low, high + 1)]
    for ax, (label, (key, color)) in zip(axes, HISTOGRAM_MODELS.items()):
        values = [number(row, field) for row in rows if row["model"] == key]
        plotted = values if metric == "cost" else np.clip(values, *limits)
        ax.hist(plotted, bins=bins, color=color, alpha=.88, edgecolor="white", linewidth=.45)
        ax.set_box_aspect(1)
        ax.set_anchor("N")
        ax.set_title(label, loc="left", pad=3, fontsize=9)
        if metric == "cost":
            ax.set_xscale("log")
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xticks(ticks)
        if metric == "cost":
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.tick_params(axis="x", labelsize=6.2)
        counts, _ = np.histogram(plotted, bins=bins)
        ax.set_ylim(0, max(min_count, int(counts.max()) * 1.1))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        ax.grid(False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#777777")
        ax.set_xlabel(xlabel, fontsize=8)
    axes[0].set_ylabel("Runs")


def histograms(tables, _use_tex):
    """Compression, cost and duration of passing submissions as three rows of square panels."""
    histogram_style()
    width = 5.5
    left, right, column_gap = .40, .06, .16
    top, bottom, row_gap = .52, .38, .85
    side = (width - left - right - 4 * column_gap) / 5
    height = top + bottom + 3 * side + 2 * row_gap
    fig, axes = plt.subplots(3, len(HISTOGRAM_MODELS), figsize=(width, height), sharex="row", sharey="row")
    fig.subplots_adjust(left=left / width, right=1 - right / width, bottom=bottom / height,
                        top=1 - top / height, wspace=column_gap / side, hspace=row_gap / side)
    for row_axes, (stem, metric), title in zip(axes, HISTOGRAMS.items(), ("Compression", "Cost", "Duration")):
        plot_histogram(tables[stem], metric, row_axes)
        for ax in row_axes:
            if metric == "cost":
                ax.tick_params(axis="x", labelsize=7)
            # Leave room for the preceding panel's final tick label.
            ax.get_xticklabels()[0].set_ha("left")
        position = row_axes[0].get_position()
        fig.text(position.x0, position.y1 + 16 / (72 * height), title, ha="left", va="bottom", fontsize=12)
    return fig


def classification_style() -> None:
    plt.rcParams.update({
        **tex_preamble(False),
        "font.family": "serif", "font.serif": SERIF, "mathtext.fontset": "cm",
        "font.size": 9.4, "axes.titlesize": 9.5, "axes.labelsize": 9.4,
        "axes.edgecolor": "#333333", "axes.linewidth": 0.65, "axes.titleweight": "regular",
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.bottom": False,
        "xtick.labelsize": 8.2, "ytick.labelsize": 8.2, "xtick.color": "black", "ytick.color": "black",
        "legend.fontsize": 7.9, "legend.frameon": False, "hatch.linewidth": 0.6,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def classification_panel(ax, rows: list[dict]) -> None:
    """One stacked bar per model; proof simplification and structural diff form Proof rewrite."""
    fields = ("saved", "dead_code", "deleted", "added", "syntax_optimization",
              "automation", "proof_simplification", "structural_diff")
    data = {row["model"]: {**row, **{key: number(row, key) for key in fields},
            "proof_rewrite": number(row, "proof_simplification") + number(row, "structural_diff")}
            for row in rows}
    models = list(data)
    height = max(sum(max(0.0, data[m][k]) for k, _, _ in BARS) for m in models)
    tops = []
    for x, model in enumerate(models):
        entry, up, down = data[model], 0.0, 0.0
        for key, _, colour in BARS:
            value = entry[key]
            base = up if value >= 0 else down
            ax.bar(x, value, bottom=base, width=0.64, color=colour, edgecolor="none", linewidth=0,
                   antialiased=False, zorder=3)
            if abs(value) >= 0.07 * height:
                ax.text(x, base + value / 2, f"{value:.1f}", ha="center", va="center", fontsize=6.5,
                        color="white", zorder=5)
            if value >= 0:
                up += value
            else:
                down += value
        ax.text(x, up + 0.012 * height, f"{entry['saved']:.1f}", ha="center", va="bottom", fontsize=6.5,
                color="black", zorder=5)
        tops.append(up)
    ax.axhline(0, color="#333333", linewidth=0.6, zorder=2)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([data[m]["label"] for m in models])
    ax.set_xlim(-0.5, len(models) - 0.5)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.spines["bottom"].set_visible(False)
    ax.set_ylim(top=max(tops) * 1.14)
    ax.set_ylabel("Score (%)")
    ax.grid(False)
    ax.set_axisbelow(True)
    labels = [data[m]["label"].replace("5.6 Luna", "Luna 5.6").replace(" 3.8 Flash", " 3.8")
              .replace(" 1.3 Spark", " 1.3") for m in models]
    width = 7 if len(models) >= 5 else 13
    labels = ["\n".join(textwrap.fill(line, width=width, break_long_words=False, break_on_hyphens=False)
                        for line in label.split("\n")) for label in labels]
    ax.set_xticks(range(len(models)), labels)
    ax.tick_params(axis="both", labelsize=7.2)
    ax.tick_params(axis="x", labelsize=7.6)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))


def classification_comparison(tables, _use_tex):
    """Compression origin by model (left) and by Opus prompt (right), with one legend."""
    classification_style()
    prompts = [{**row, "label": "Reduce\nsize"} if row["model"] == "reduce-size-without-anticheat" else row
               for row in tables["opus-prompt-ablation"] if row["model"] != "reduce-size-with-anticheat"]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.35), sharey=True, gridspec_kw={"width_ratios": (47, 35)})
    for ax, rows in zip(axes, (tables["diff_classification_macro"], prompts), strict=True):
        classification_panel(ax, rows)
    axes[1].set_ylabel("")
    limits = [edge for ax in axes for patch in ax.patches
              for edge in (patch.get_y(), patch.get_y() + patch.get_height())]
    axes[0].set_ylim(10 * math.floor(min(0, min(limits)) / 10), 60)
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    # Matplotlib fills columns first; this order reads in stack order across rows.
    parts = list(reversed(BARS))
    parts = [parts[i] for i in (0, 3, 1, 4, 2, 5)]
    fig.legend(handles=[Patch(facecolor=color, edgecolor="none", label=label) for _, label, color in parts],
               loc="upper center", bbox_to_anchor=(0.545, 0.975), ncol=3, frameon=False, fontsize=8.5,
               handlelength=1.0, columnspacing=0.7, handletextpad=0.35, labelspacing=0.2,
               borderaxespad=0, borderpad=0)
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.265 / 2.35, top=0.79,
                        wspace=2 * 0.08 / (0.47 + 0.35))
    return fig


def agent_actions(tables, _use_tex):
    """Mean tool calls per trajectory in the ten action classes."""
    categories = ["Build", "Verify", "Measure", "Read", "Search", "Edit", "Git", "Lake", "Sleep", "Other"]
    colors = ["#4E8D9D", "#86A85F", "#D99533", "#2B6F9C", "#9B77A8",
              "#C75D63", "#6C83B5", "#58A98B", "#D17C4B", "#B7B7BB"]
    models = {"opus-5": "Opus", "gpt-5.6-sol": "Sol", "gemini-3.8-flash": "Gemini",
              "muse-spark-1.3": "Muse", "gpt-5.6-luna": "Luna"}
    by_model = {row["model"]: row for row in tables["agent-actions"]}
    summary = [by_model[model] for model in models]
    for row in summary:
        if not np.isclose(sum(float(row[c]) for c in categories), float(row["total"]), rtol=0, atol=1e-8):
            raise ValueError(f"Action classes do not sum to the total: {row['model']}")
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "Tinos", "Nimbus Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42})
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8})
    fig, ax = plt.subplots(figsize=(182 / 72, 142 / 72))
    bottom = np.zeros(len(summary))
    for category, color in zip(categories, colors):
        values = np.array([float(r[category]) for r in summary])
        ax.bar(range(len(summary)), values, bottom=bottom, color=color, label=category, width=.65)
        bottom += values
    ax.set_xticks(range(len(summary)), list(models.values()))
    ax.set_ylabel("Tool calls per trajectory", labelpad=3)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
    ax.tick_params(pad=2, length=3)
    ax.set_ylim(0, max(bottom) * 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#777777")
    handles, labels = ax.get_legend_handles_labels()
    order = [j for col in range(5) for j in range(col, len(handles), 5)]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center",
               bbox_to_anchor=(.5, .01), ncol=5, frameon=False, handlelength=.9, handletextpad=.35,
               columnspacing=.5, labelspacing=.25, borderpad=0, borderaxespad=0)
    fig.subplots_adjust(left=33.52 / 182, right=180.04 / 182, bottom=38.064 / 142, top=141.024 / 142)
    return fig


# The paper's saved editor layout for the frontier figure (frontier-layout.json).
FRONTIER = {
    "opus-5": ("Opus 5", "#DB8A70", (-23.04, -1.64), "center"),
    "gpt-5.6-sol": ("Sol 5.6", "#3DAD8F", (-23.04, -1.16), "center"),
    "gemini-3.8-flash": ("Gemini 3.8", "#F3BD3A", (30.96, -0.92), "center"),
    "muse-spark-1.3": ("Muse 1.3", "#387EFB", (27.04, -1.52), "center"),
    "gpt-5.6-luna": ("Luna 5.6", "#222222", (6.12, 10.6), "center"),
    "leanstral-1.5": ("Leanstral 1.5", "#F97316", (9.04, 6.16), "left"),
}


def frontier(tables, use_tex):
    """Score against average cost per repository, on a shifted-log cost axis."""
    plt.rcParams.update({
        **tex_preamble(use_tex), "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
        "axes.edgecolor": "#333333", "axes.linewidth": 0.65,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.pad": 2, "ytick.major.pad": 2, "xtick.color": "#333333", "ytick.color": "#333333",
        "xtick.major.width": 0.55, "ytick.major.width": 0.55,
        "figure.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    if not use_tex:
        plt.rcParams["font.serif"] = SERIF
    width, height = 2.35, 1.5625
    fig, ax = plt.subplots(figsize=(width, height))
    for row in tables["leaderboard"]:
        label, color, offset, ha = FRONTIER[row["model"]]
        cost, score = float(row["cost_per_task_usd"]), float(row["compression_pct"])
        # Leanstral sits on the x-axis.
        ax.scatter(cost, score, s=78.0, color=color, edgecolor="white", linewidth=0.5, zorder=3, clip_on=False)
        ax.annotate(label, (cost, score), xytext=offset, textcoords="offset points", ha=ha, va="center",
                    fontsize=10.0, color="#000000")
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_xlabel(r"Cost (\$)" if use_tex else "Cost ($)", labelpad=1, fontsize=10.0, color="#000000")
    ax.set_ylabel(r"Score (\%)" if use_tex else "Score (%)", labelpad=1, fontsize=10.0, color="#000000")
    ax.set_xscale("function", functions=(np.log1p, np.expm1))
    ax.set_xlim(1.0, 250.0)
    ax.set_ylim(-1.0, 56.0)
    ax.xaxis.set_major_locator(FixedLocator([0, 5, 10, 50, 100]))
    ax.xaxis.set_minor_locator(FixedLocator([1, 2, 3, 4, 6, 7, 8, 9, 20, 30, 40, 60, 70, 80, 90, 200]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.65)
    for which, scale in (("major", 1), ("minor", 0.6)):
        ax.tick_params(which=which, bottom=True, left=True, top=False, right=False,
                       labelbottom=which == "major", labelleft=which == "major",
                       length=2.5 * scale, width=0.55, direction="out", color="#333333",
                       labelcolor="#000000", labelsize=7)
    ax.set_axisbelow(True)
    ax.grid(False, which="both")
    for axis in ("x", "y"):
        ax.grid(True, axis=axis, which="major", color="#cccccc", linewidth=0.6, alpha=0.5)
    fig.subplots_adjust(left=0.30 / width, right=1 - 0.04 / width, bottom=0.29 / height, top=1 - 0.02 / height)
    return fig


BUDGET_MODELS = {"gemini": ("Gemini 3.8 Flash", "#F3BD3A"), "luna": ("Luna 5.6", "#222222"),
                 "muse": ("Muse Spark 1.3", "#387EFB"), "sol": ("Sol 5.6", "#3DAD8F"),
                 "opus": ("Opus 5", "#DB8A70")}


def budget(tables, _use_tex):
    """Score of the latest passing checkpoint within each time and cost budget (budget-layout.json)."""
    rows = tables["budget_log_pair"]
    curves = {}
    for axis in ("elapsed_minutes", "cost_usd"):
        for model in BUDGET_MODELS:
            series = sorted((r for r in rows if r["axis"] == axis and r["model"] == model),
                            key=lambda r: float(r["budget"]))
            if not series:
                raise ValueError(f"Missing budget series: {axis}/{model}")
            curves[axis, model] = {field: np.array([float(r[field]) if r[field] else np.nan for r in series])
                                   for field in ("budget", "score_pct", "lower_pct", "upper_pct")}
    # The paper saved this figure outside these settings, so its axis titles are
    # drawn in Matplotlib's default serif; the caller saves after the context.
    with plt.rc_context({"text.usetex": False, "font.family": "serif", "font.serif": SERIF,
                         "mathtext.fontset": "cm", "font.size": 7.9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.facecolor": "white"}):
        return budget_axes(curves)


def budget_axes(curves):
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.15), sharey=True, dpi=100)
    lines = {}
    for ax, name, column, divisor in zip(axes, ("time", "cost"), ("elapsed_minutes", "cost_usd"), (60, 1)):
        for model, (_, color) in BUDGET_MODELS.items():
            series = curves[column, model]
            x = series["budget"] / divisor
            if name == "cost" and model == "gemini":
                band = ax.fill_between(x, series["lower_pct"], series["upper_pct"], step="post",
                                       linewidth=0, zorder=2)
                band.set_facecolor(color)
                band.set_alpha(0.2)
            lines[name, model], = ax.step(x, series["score_pct"], where="post", zorder=3,
                                          color=color, linewidth=1.35)
    fig.subplots_adjust(left=0.095, right=0.975, bottom=0.22, top=0.8, wspace=0.15)
    for ax, name, shift, ticks, label in zip(
            axes, ("time", "cost"), (1 / 6, 1), ([0, .25, .5, 1, 2, 4, 8, 12], [0, 1, 3, 10, 30, 100, 800]),
            ("$t$ (h)", "Cost ($)")):
        # Shifted log keeps zero: ten minutes for time, one dollar for cost.
        ax.set_xscale("function", functions=(lambda x, shift=shift: np.log1p(np.asarray(x) / shift),
                                             lambda x, shift=shift: shift * np.expm1(np.asarray(x))))
        ax.set_xlim(0, ticks[-1])
        ax.set_ylim(-5, 56.0)
        ax.xaxis.set_major_locator(FixedLocator(sorted(set(ticks))))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        major = np.array(ax.get_xticks())
        ax.xaxis.set_minor_locator(FixedLocator(
            [shift * np.expm1(v) for lo, hi in zip(major[:-1], major[1:])
             for v in np.linspace(np.log1p(lo / shift), np.log1p(hi / shift), 5)[1:-1]]))
        ax.yaxis.set_major_locator(FixedLocator([0.0, 15.0, 30.0, 30.0, 45.0]))
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.set_xlabel(label, labelpad=3, fontsize=8.1, color="#000000")
        for spine in ax.spines.values():
            spine.set_color("#1F2529")
            spine.set_linewidth(0.55)
        for which, factor in (("major", 1), ("minor", .6)):
            ax.tick_params(which=which, bottom=which == "major", left=which == "major", top=False, right=False,
                           labelbottom=which == "major", labelleft=which == "major" and name == "time",
                           length=2.5 * factor, width=0.5, color="#000000", labelcolor="#000000",
                           labelsize=7.3, direction="out", pad=2)
        for tick in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            tick.set_fontfamily("Times New Roman")
        ax.set_axisbelow(True)
        ax.grid(False, which="both")
    axes[0].set_ylabel("Score (%)", labelpad=3, fontsize=8.1, color="#000000")
    order = ("opus", "sol", "gemini", "muse", "luna")
    fig.legend([lines["time", m] for m in order], [BUDGET_MODELS[m][0] for m in order], ncol=5,
               loc="lower center", bbox_to_anchor=(0.5, 0.81), borderaxespad=0, frameon=False,
               prop={"family": "Times New Roman", "size": 7.9}, labelcolor="#000000",
               handlelength=1.3, handletextpad=.3, columnspacing=.55)
    return fig


# name: (paper path, renderer, inputs, savefig options); TeX applies to the PDF only.
FIGURES = {
    "introduction-results": ("experiments/introduction-results", introduction, ("introduction-results",), {}),
    "frontier": ("experiments/frontier", frontier, ("leaderboard",), {}),
    "budget-time-dollar-scores": ("experiments/budget-time-dollar-scores", budget, ("budget_log_pair",), {}),
    "benchmark-model-performance-by-scale": ("leanlean/benchmark-model-performance-by-scale", performance_by_scale,
                                             ("benchmark-model-performance-by-scale",), {}),
    "compression-classification-comparison": ("experiments/compression-classification-comparison",
                                              classification_comparison,
                                              ("diff_classification_macro", "opus-prompt-ablation"), {}),
    "agent-actions": ("experiments/agent-actions", agent_actions, ("agent-actions",), {}),
    "compression-task-histograms": ("experiments/compression-task-histograms", histograms, tuple(HISTOGRAMS),
                                    {"bbox_inches": "tight"}),
    "benchmark-model-cost-duration-by-scale": ("leanlean/benchmark-model-cost-duration-by-scale",
                                               cost_duration_by_scale,
                                               ("benchmark-model-cost-by-scale", "benchmark-model-duration-by-scale"),
                                               {}),
    "benchmark-primary-subject-families": ("leanlean/benchmark-primary-subject-families", subject_families,
                                           ("benchmark-primary-subject-families",), {}),
    "repository-sizes": ("leanlean/repository-sizes", repository_sizes,
                         ("benchmark-preprocessing-removal", "repository-subjects",
                          "benchmark-primary-subject-families"), {}),
    "preprocessing-removal": ("leanlean/preprocessing-removal", preprocessing_removal,
                              ("benchmark-preprocessing-removal",), {}),
}
TEX_FIGURES = {"frontier", "benchmark-model-performance-by-scale", "benchmark-model-cost-duration-by-scale"}


# --- Tables ---------------------------------------------------------------------

def escape(value) -> str:
    mapping = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
               "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}
    return "".join(mapping.get(c, c) for c in str(value))


def num(value, digits=2, signed=False) -> str:
    return "---" if value in ("", None) else format(float(value), f"{'+' if signed else ''}.{digits}f")


def count(value) -> str:
    return "---" if value in ("", None) else f"{int(float(value)):,}".replace(",", r"{,}")


def pct(value) -> str:
    if value in ("", None):
        return "---"
    rendered = num(value) + r"\%"
    return "$" + rendered + "$" if float(value) < 0 else rendered


def thousands(value) -> str:
    if value in ("", None):
        return "---"
    value = float(value)
    return f"{value / 1000:.1f}k" if value < 100_000 else f"{value / 1000:.0f}k"


LABELS = {"gpt-5.6-sol": "GPT-5.6 Sol", "gpt-5.6-luna": "GPT-5.6 Luna", "gpt-6-astra-mini-v3": "GPT-6 Astra",
          "gpt-6-astra-xhigh": "GPT-6 Astra", "gemini-3.8-flash": "Gemini 3.8 Flash",
          "muse-spark-1.3": "Muse Spark 1.3", "opus-5": "Opus 5"}
EFFORTS = {"opus-5": "high", "gpt-5.6-sol": "xhigh", "gpt-5.6-luna": "xhigh", "gemini-3.8-flash": "high",
           "muse-spark-1.3": "max", "leanstral-1.5": "high"}


def table(stem: str, rows: list[dict]) -> str:
    """Fill the paper's template for one table; templates hold captions and layout."""
    body = []

    def add(cells, indent=""):
        body.append(indent + " & ".join(cells) + r" \\")

    if stem == "preprocessing-scale":
        sizes = {"Compact": r"$\leq 10$k", "Standard": r"$10$k--$50$k", "Large": r"$50$k--$250$k",
                 "Massive": r"$>250$k", "All": "---"}
        for row in rows:
            if row["scale"] == "All":
                body.append(r"\midrule")
            core = ["---" if not row[k] else f"{float(row[k]):g}" for k in ("core_theorems_median", "core_theorems_max")]
            add([escape(row["scale"]), sizes[row["scale"]], row["repositories"], *core,
                 *[thousands(row[k]) for k in ("loc_median", "loc_max", "lean_tokens_median", "lean_tokens_max")]])
    elif stem == "dependency-layers":
        labels = {"Kernel (.olean)": r"Kernel (\texttt{.olean})", ".ilean": r"\texttt{.ilean}"}
        for row in rows:
            if row["layer"] == "Found by all three":
                body.append(r"\midrule")
            body.append(labels.get(row["layer"], escape(row["layer"])))
            add(["", count(row["edges"]), pct(row["coverage_pct"]), count(row["unique_edges"])], " ")
    elif stem == "heartbeat-leaderboard":
        # The export averages over measured repositories; failed builds score zero
        # over all expected repositories instead. Leanstral has too few measurements.
        scores = [(row, float(row["average_reduction_pct"]) * int(row["scored"]) / int(row["expected"]),
                   100 * int(row["improved_repositories"]) / int(row["expected"]))
                  for row in rows if row["model"] != "leanstral-1.5"]
        for row, reduction, improved in sorted(scores, key=lambda score: -score[1]):
            add([escape(LABELS.get(row["model"], row["label"])), num(reduction, signed=True), num(improved, 1)])
    elif stem in ("leaderboard", "benchmark-extra-leaderboard"):
        main = stem == "leaderboard"
        best = {key: optimum(float(row[key]) for row in rows) for key, optimum in
                (("compression_pct", max), ("cost_per_task_usd", min), ("duration_hours", min))} if main else {}

        def metric(row, key):
            value = num(row[key])
            return r"\textbf{" + value + "}" if key in best and float(row[key]) == best[key] else value

        for row in rows:
            label = escape(LABELS.get(row["model"], row["label"]))
            if main and row["model"] == "muse-spark-1.3":
                label += r"\textsuperscript{\ref{fn:muse-configuration}}"
            if main and row["model"] in EFFORTS:
                label += f' ({EFFORTS[row["model"]]})'
            add([label, metric(row, "compression_pct"), metric(row, "cost_per_task_usd"),
                 metric(row, "duration_hours")], "    ")
    elif stem == "union-ablation":
        by_method = {row["method"]: row for row in rows}
        rows = [by_method[method] for method in ("gpt-5.6-sol", "opus-5", "start-opus")]
        best = max(float(row["compression_pct"]) for row in rows)
        labels = {"start-opus": "Merge", "opus-5": "Opus 5", "gpt-5.6-sol": "GPT-5.6 Sol"}
        for row in rows:
            value = num(row["compression_pct"])
            add([labels[row["method"]], r"\textbf{" + value + "}" if float(row["compression_pct"]) == best else value],
                "    ")
    else:
        raise ValueError(f"Unknown paper table: {stem}")
    template = (TEMPLATES / f"{stem}.tex.in").read_text()
    return template.replace("@@ROWS@@", "\n".join(body))


def results_macros(tables) -> str:
    """The numbers the paper's prose repeats (tables/results.tex)."""
    aliases = {"opus-5": "Opus", "gpt-5.6-sol": "Sol", "gpt-5.6-luna": "Luna", "gemini-3.8-flash": "Gemini",
               "muse-spark-1.3": "Muse", "leanstral-1.5": "Leanstral"}
    commands = {}
    for row in tables["leaderboard"]:
        for field, suffix in (("compression_pct", "Compression"), ("cost_per_task_usd", "Cost")):
            commands[aliases[row["model"]] + suffix] = f"{float(row[field]):.2f}" if row[field] else "---"
    for row in tables["benchmark-model-performance-by-scale"]:
        if row["model"] == "opus-5" and row["token_band"] == "> 250k":
            commands["OpusMassiveCompression"] = f'{float(row["mean_reduction_pct"]):.2f}'
    union = {row["method"]: float(row["compression_pct"]) for row in tables["union-ablation"]}
    for key, alias in (("start-opus", "Opus"), ("start-sol", "Sol")):
        commands["Union" + alias + "Gain"] = f'{union[key] - union["opus-5"]:+.2f}'
    lines = ["% Generated by scripts/paper/render.py; edit the exported data, not this file."]
    return "\n".join(lines + [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in commands.items()]) + "\n"


TABLES = {stem: (stem,) for stem in ("leaderboard", "heartbeat-leaderboard", "benchmark-extra-leaderboard",
                                     "union-ablation", "dependency-layers", "preprocessing-scale")}
TABLES["results"] = ("leaderboard", "benchmark-model-performance-by-scale", "union-ablation")


# --- Command line ---------------------------------------------------------------

def read_tables(data_dir: Path, stems) -> dict[str, list[dict]]:
    manifest_path = data_dir.parent / "manifest.json"
    pinned = json.loads(manifest_path.read_text())["tables"] if manifest_path.is_file() else {}
    tables = {}
    for stem in stems:
        path = data_dir / f"{stem}.csv"
        data = path.read_bytes()
        if path.name in pinned and hashlib.sha256(data).hexdigest() != pinned[path.name]["sha256"]:
            raise ValueError(f"CSV differs from {manifest_path}: {path}")
        tables[stem] = list(csv.DictReader(io.StringIO(data.decode("utf-8"), newline="")))
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "results/paper/data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/paper")
    parser.add_argument("--only", nargs="+", choices=[*FIGURES, *TABLES], help="render a subset")
    parser.add_argument("--no-tex", action="store_true", help="never typeset with LaTeX")
    args = parser.parse_args()
    use_tex = not args.no_tex and shutil.which("latex") is not None
    selected = args.only or [*FIGURES, *TABLES]
    inputs = {name: FIGURES[name][2] if name in FIGURES else TABLES[name] for name in selected}
    tables = read_tables(args.data_dir, sorted({stem for stems in inputs.values() for stem in stems}))
    for name in selected:
        if name in TABLES:
            continue
        path, renderer, _, options = FIGURES[name]
        for fmt in ("pdf", "png"):
            # Each figure starts from clean defaults, as the paper's separate renderers did.
            plt.rcdefaults()
            fig = renderer(tables, use_tex and fmt == "pdf" and name in TEX_FIGURES)
            target = args.output_dir / "figures" / f"{path}.{fmt}"
            target.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target, dpi=300, metadata=NO_DATES if fmt == "pdf" else None, **options)
            plt.close(fig)
            print(target)
    output = args.output_dir / "tables"
    output.mkdir(parents=True, exist_ok=True)
    for name in selected:
        if name in TABLES:
            text = results_macros(tables) if name == "results" else table(name, tables[name])
            (output / f"{name}.tex").write_text(text, encoding="utf-8")
            print(output / f"{name}.tex")
    if not args.only and (args.data_dir.parent / "examples").is_dir():
        examples = args.output_dir / "figures/examples"
        shutil.copytree(args.data_dir.parent / "examples", examples, dirs_exist_ok=True)
        print(examples)


if __name__ == "__main__":
    main()
