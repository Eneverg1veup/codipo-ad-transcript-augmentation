#!/usr/bin/env python
"""Build cross-cohort operating results with augmentation-scale sensitivity."""

from __future__ import annotations

import argparse
from pathlib import Path
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


BOOTSTRAP_CSV: Path
COHORT_CSV: Path
FIG_DIR: Path
SOURCE_DIR: Path
K_SCALING_CSV: Path
OUT_STEM = "Figure4_cross_cohort_performance_k_scaling_external_20260720"

METHODS = [
    ("BERT", "BERT", "No augmentation reference"),
    ("EDA", "EDA", "Representative non-LLM anchors"),
    ("CDA", "CDA", "Representative non-LLM anchors"),
    ("ICL Direct", "ICL Direct", "LLM generation strategies"),
    ("ICL Rewrite", "ICL Rewrite", "LLM generation strategies"),
    ("ICL Imitation", "ICL Imitation", "LLM generation strategies"),
    ("w/o DPO, vanilla augmentation", "Vanilla", "LLM generation strategies"),
    ("w/o DPO, XYZ hard filtering", "Hard Filter", "Proxy-guided preference and selection"),
    ("CoDiPO", "CoDiPO", "Proxy-guided preference and selection"),
]

METRICS = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("sensitivity", "Sens."),
    ("f1", "F1"),
    ("specificity", "Specificity"),
    ("auroc", "AUROC"),
    ("auprc", "AUPRC"),
]

COHORTS = [
    ("ADReSS validation", "ADReSS validation", "^", -0.22),
    ("Lu", "Lu", "s", -0.07),
    ("Pitt", "Pitt", "D", 0.08),
]

METHOD_COLORS = {
    "BERT": "#555B61",
    "EDA": "#E1B2BC",
    "CDA": "#9D8DB8",
    "ICL Direct": "#D58470",
    "ICL Rewrite": "#A9CADB",
    "ICL Imitation": "#AFCEAD",
    "Vanilla": "#A6AFB7",
    "Hard Filter": "#3D9896",
    "CoDiPO": "#28688E",
}

K_SCALING_DISPLAY = {
    "xyz": "CoDiPO",
    "direct": "ICL Direct",
    "imitation": "ICL Imitation",
    "rewrite": "ICL Rewrite",
}

K_SCALING_COLORS = {
    "xyz": METHOD_COLORS["CoDiPO"],
    "direct": "#7D878D",
    "imitation": "#D58E79",
    "rewrite": "#C9A23F",
}

INK = "#202124"
GRID = "#E3E7EA"
FAMILY_BANDS = ["#F5F6F7", "#FBFBFB", "#F5F6F7", "#EEF5F6"]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8.6,
            "axes.titlesize": 9.3,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.7,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_source() -> pd.DataFrame:
    bootstrap = pd.read_csv(BOOTSTRAP_CSV)
    cohort = pd.read_csv(COHORT_CSV)
    method_meta = pd.DataFrame(METHODS, columns=["method_display", "display_method", "method_family"])

    bootstrap = bootstrap[
        bootstrap["method_display"].isin(method_meta["method_display"])
        & bootstrap["metric"].isin([metric for metric, _ in METRICS])
    ].copy()
    bootstrap = bootstrap.merge(method_meta, on="method_display", how="left", validate="many_to_one")
    bootstrap["cohort"] = "External-cohort average"
    bootstrap["point_percent"] = bootstrap["point_estimate"].astype(float) * 100
    bootstrap["ci_low_percent"] = bootstrap["ci_low"].astype(float) * 100
    bootstrap["ci_high_percent"] = bootstrap["ci_high"].astype(float) * 100
    bootstrap["point_role"] = "external_cohort_bootstrap_ci"

    cohort = cohort[
        cohort["method_display"].isin(method_meta["method_display"])
        & cohort["metric"].isin([metric for metric, _ in METRICS])
        & cohort["cohort"].isin([cohort_name for cohort_name, _, _, _ in COHORTS])
    ].copy()
    cohort = cohort.merge(method_meta, on="method_display", how="left", validate="many_to_one")
    cohort["point_percent"] = cohort["mean"].astype(float) * 100
    cohort["ci_low_percent"] = math.nan
    cohort["ci_high_percent"] = math.nan
    cohort["point_role"] = "cohort_mean"

    columns = [
        "method_display",
        "display_method",
        "method_family",
        "cohort",
        "metric",
        "point_percent",
        "ci_low_percent",
        "ci_high_percent",
        "point_role",
    ]
    source = pd.concat([bootstrap[columns], cohort[columns]], ignore_index=True)
    expected = len(METHODS) * len(METRICS) * (1 + len(COHORTS))
    if len(source) != expected:
        raise ValueError(f"Expected {expected} source rows, found {len(source)}")

    method_order = {raw: i for i, (raw, _, _) in enumerate(METHODS)}
    metric_order = {metric: i for i, (metric, _) in enumerate(METRICS)}
    cohort_order = {"External-cohort average": 0, "ADReSS validation": 1, "Lu": 2, "Pitt": 3}
    source["_method_order"] = source["method_display"].map(method_order)
    source["_metric_order"] = source["metric"].map(metric_order)
    source["_cohort_order"] = source["cohort"].map(cohort_order)
    source = source.sort_values(["_metric_order", "_method_order", "_cohort_order"]).drop(
        columns=["_method_order", "_metric_order", "_cohort_order"]
    )

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    source.to_csv(SOURCE_DIR / "figure4_cross_cohort_performance_source.csv", index=False, encoding="utf-8-sig")
    return source


def load_k_scaling_source() -> pd.DataFrame:
    source = pd.read_csv(K_SCALING_CSV)
    required = {"run_num", "method", "macro_score_mean", "macro_score_std", "n_runs"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"K-scaling source missing columns: {sorted(missing)}")
    source = source[source["method"].isin(K_SCALING_DISPLAY)].copy()
    source["mean_percent"] = source["macro_score_mean"].astype(float) * 100.0
    source["sd_percent"] = source["macro_score_std"].astype(float) * 100.0
    if not source["n_runs"].eq(25).all():
        raise ValueError("Expected 25 run-level evaluations for every method-K setting")
    expected = len(K_SCALING_DISPLAY) * 10
    if len(source) != expected:
        raise ValueError(f"Expected {expected} K-scaling rows, found {len(source)}")
    source.to_csv(
        SOURCE_DIR / "figure4_panel_h_k_scaling_source.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            {
                "figure": OUT_STEM,
                "external_average_source": BOOTSTRAP_CSV.name,
                "cohort_source": COHORT_CSV.name,
                "k_scaling_source": K_SCALING_CSV.name,
                "metrics": "; ".join(label for _, label in METRICS),
                "external_uncertainty": "n=10000 percentile bootstrap 95% CI",
                "cohort_marker": "locked cohort mean without interval",
                "k_scaling_uncertainty": "sample SD across run-level evaluations",
            }
        ]
    ).to_csv(SOURCE_DIR / "figure4_cross_cohort_performance_manifest.csv", index=False, encoding="utf-8-sig")
    return source


def add_family_bands(ax: plt.Axes) -> None:
    spans = [(-0.45, 0.45), (0.55, 2.45), (2.55, 6.45), (6.55, 8.45)]
    for (low, high), color in zip(spans, FAMILY_BANDS):
        ax.axhspan(low, high, color=color, zorder=-4)
    for boundary in [0.5, 2.5, 6.5]:
        ax.axhline(boundary, color="#C9CED2", linewidth=0.65, zorder=-1)


def metric_xlim(data: pd.DataFrame, metric: str) -> tuple[float, float, list[float]]:
    sub = data[data["metric"].eq(metric)]
    values = sub["point_percent"].astype(float).tolist()
    values += pd.to_numeric(sub["ci_low_percent"], errors="coerce").dropna().tolist()
    values += pd.to_numeric(sub["ci_high_percent"], errors="coerce").dropna().tolist()
    low = math.floor((min(values) - 1.2) / 2) * 2
    high = math.ceil((max(values) + 1.2) / 2) * 2
    if high - low < 10:
        middle = (high + low) / 2
        low = math.floor((middle - 5) / 2) * 2
        high = math.ceil((middle + 5) / 2) * 2
    span = high - low
    step = 10 if span >= 30 else (5 if span >= 16 else 4)
    first = math.ceil(low / step) * step
    ticks = list(range(first, int(high) + 1, step))
    return float(low), float(high), [float(tick) for tick in ticks]


def draw_metric(ax: plt.Axes, source: pd.DataFrame, metric: str, title: str, show_y: bool) -> None:
    add_family_bands(ax)
    for y, (raw_method, display_method, _) in enumerate(METHODS):
        color = METHOD_COLORS[display_method]
        rows = source[source["method_display"].eq(raw_method) & source["metric"].eq(metric)]
        external = rows[rows["cohort"].eq("External-cohort average")]
        if len(external) != 1:
            raise ValueError(f"Missing external-cohort row: {raw_method} / {metric}")
        row = external.iloc[0]
        point = float(row["point_percent"])
        low = float(row["ci_low_percent"])
        high = float(row["ci_high_percent"])
        ax.errorbar(
            point,
            y + 0.23,
            xerr=[[point - low], [high - point]],
            fmt="o",
            markersize=5.3 if display_method == "CoDiPO" else 4.6,
            markerfacecolor=color,
            markeredgecolor=INK,
            markeredgewidth=0.5,
            ecolor=color,
            elinewidth=1.05,
            capsize=2.0,
            zorder=4,
        )
        for cohort_name, _, marker, offset in COHORTS:
            cohort_row = rows[rows["cohort"].eq(cohort_name)]
            if len(cohort_row) != 1:
                raise ValueError(f"Missing cohort row: {raw_method} / {cohort_name} / {metric}")
            ax.scatter(
                float(cohort_row.iloc[0]["point_percent"]),
                y + offset,
                s=16,
                marker=marker,
                facecolor="white",
                edgecolor=color,
                linewidth=0.75,
                alpha=0.68,
                zorder=3,
            )

    low, high, ticks = metric_xlim(source, metric)
    ax.set_xlim(low, high)
    ax.set_xticks(ticks)
    ax.set_ylim(len(METHODS) - 0.45, -0.45)
    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    ax.set_xlabel("Score (%)")
    ax.grid(axis="x", color=GRID, linewidth=0.65)
    ax.tick_params(axis="y", length=0)
    if show_y:
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels([display for _, display, _ in METHODS])
        for tick, (_, display, _) in zip(ax.get_yticklabels(), METHODS):
            if display == "CoDiPO":
                tick.set_fontweight("bold")
                tick.set_color(METHOD_COLORS["CoDiPO"])
    else:
        ax.tick_params(axis="y", labelleft=False)


def draw_k_scaling_panel(ax: plt.Axes, source: pd.DataFrame) -> None:
    order = ["direct", "imitation", "rewrite", "xyz"]
    for method in order:
        part = source[source["method"].eq(method)].sort_values("run_num")
        x = part["run_num"].to_numpy(dtype=float)
        y = part["mean_percent"].to_numpy(dtype=float)
        sd = part["sd_percent"].to_numpy(dtype=float)
        color = K_SCALING_COLORS[method]
        is_codipo = method == "xyz"
        ax.plot(
            x,
            y,
            color=color,
            linewidth=1.8 if is_codipo else 1.05,
            marker="o",
            markersize=4.1 if is_codipo else 3.4,
            markeredgecolor=INK,
            markeredgewidth=0.38,
            alpha=1.0 if is_codipo else 0.84,
            label=K_SCALING_DISPLAY[method],
            zorder=3 if is_codipo else 2,
        )
        ax.fill_between(
            x,
            y - sd,
            y + sd,
            color=color,
            alpha=0.14 if is_codipo else 0.08,
            linewidth=0,
            zorder=1,
        )

    ax.set_xlim(0.8, 10.2)
    ax.set_xticks(np.arange(1, 11))
    lower = float((source["mean_percent"] - source["sd_percent"]).min())
    upper = float((source["mean_percent"] + source["sd_percent"]).max())
    ax.set_ylim(math.floor(lower - 0.8), math.ceil(upper + 0.8))
    ax.set_xlabel("Augmentations per source transcript (K)")
    ax.set_ylabel("External-cohort average F1 (%)")
    ax.set_title("Augmentation-scale sensitivity", loc="left", fontweight="bold", pad=5)
    ax.grid(color=GRID, linewidth=0.65)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=4,
        frameon=False,
        handlelength=1.7,
        handletextpad=0.35,
        columnspacing=1.2,
    )


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.16) -> None:
    ax.text(
        x,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=10.2,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=INK,
    )


def save_bundle(fig: plt.Figure) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "pdf": {},
        "svg": {},
        "png": {"dpi": 600},
        "tiff": {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}},
    }
    for extension, kwargs in outputs.items():
        fig.savefig(FIG_DIR / f"{OUT_STEM}.{extension}", bbox_inches="tight", facecolor="white", **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Figure 4 from the fixed-checkpoint statistical lock and the "
            "precomputed K-scaling summary."
        )
    )
    parser.add_argument("--external-bootstrap-csv", required=True, type=Path)
    parser.add_argument("--cohort-metrics-csv", required=True, type=Path)
    parser.add_argument("--k-scaling-csv", required=True, type=Path)
    parser.add_argument("--figure-dir", required=True, type=Path)
    parser.add_argument("--source-data-dir", required=True, type=Path)
    parser.add_argument("--output-stem", default=OUT_STEM)
    return parser.parse_args()


def configure(args: argparse.Namespace) -> None:
    global BOOTSTRAP_CSV, COHORT_CSV, K_SCALING_CSV, FIG_DIR, SOURCE_DIR, OUT_STEM
    BOOTSTRAP_CSV = args.external_bootstrap_csv
    COHORT_CSV = args.cohort_metrics_csv
    K_SCALING_CSV = args.k_scaling_csv
    FIG_DIR = args.figure_dir
    SOURCE_DIR = args.source_data_dir
    OUT_STEM = args.output_stem


def main() -> None:
    configure(parse_args())
    setup_style()
    source = load_source()
    k_scaling = load_k_scaling_source()
    fig = plt.figure(figsize=(183 / 25.4, 205 / 25.4), facecolor="white")
    grid = fig.add_gridspec(
        3,
        4,
        left=0.14,
        right=0.985,
        top=0.975,
        bottom=0.10,
        height_ratios=[1.0, 1.0, 1.08],
        hspace=0.52,
        wspace=0.30,
    )
    metric_positions = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 0),
        (1, 1),
        (1, 2),
    ]
    metric_axes = []
    for i, ((metric, title), (row, col)) in enumerate(zip(METRICS, metric_positions)):
        ax = fig.add_subplot(grid[row, col])
        draw_metric(ax, source, metric, title, show_y=col == 0)
        add_panel_label(ax, chr(ord("a") + i), x=-0.23 if col == 0 else -0.18)
        metric_axes.append(ax)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="-", color=INK, markerfacecolor=INK, markersize=4.4, label="External-cohort average (95% CI)"),
        Line2D([0], [0], marker="^", linestyle="none", markerfacecolor="white", markeredgecolor=INK, markersize=4.2, label="ADReSS validation"),
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor="white", markeredgecolor=INK, markersize=4.2, label="Lu"),
        Line2D([0], [0], marker="D", linestyle="none", markerfacecolor="white", markeredgecolor=INK, markersize=4.0, label="Pitt"),
    ]
    legend_ax = fig.add_subplot(grid[1, 3])
    legend_ax.axis("off")
    legend_ax.legend(
        handles=legend_handles,
        loc="center",
        ncol=1,
        frameon=False,
        handletextpad=0.5,
        labelspacing=0.85,
    )
    k_ax = fig.add_subplot(grid[2, :])
    draw_k_scaling_panel(k_ax, k_scaling)
    add_panel_label(k_ax, "h", x=-0.055)
    save_bundle(fig)
    plt.close(fig)
    print(f"Wrote {OUT_STEM} to {FIG_DIR}")


if __name__ == "__main__":
    main()
