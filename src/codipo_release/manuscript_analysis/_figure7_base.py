#!/usr/bin/env python
"""Build three reframed manuscript mechanism figures.

The outputs are review candidates and do not overwrite active manuscript figures.
All plotting, preview, and export work is performed with Python/matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from . import _figure6_proxy_helpers as proxy


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


ROOT = Path(__file__).resolve().parent
FINAL_DIR = ROOT.parents[1]
ARTICLE_DIR = FINAL_DIR / "08_manuscript" / "Article Title"
ARTICLE_SOURCE = ARTICLE_DIR / "source_data"
FIG_DIR = ROOT / "figures"
SOURCE_DIR = ROOT / "source_data"

QUALITY_PATH = (
    FINAL_DIR
    / "analysis_outputs_file_level_quality_plus_baselines_0624"
    / "tables"
    / "01_file_level_summary.csv"
)
BOUNDARY_PATH = (
    FINAL_DIR
    / "analysis_outputs_file_level_quality_plus_baselines_0624"
    / "quality_landscape_figures_v4"
    / "data"
    / "quality_space_region_boundaries.csv"
)
UTILITY_PATH = ARTICLE_SOURCE / "figure5_proxy_utility_source.csv"
BASELINE_BUCKET_PATH = (
    ARTICLE_SOURCE / "fig_proxy_failure_anatomy_2x6_baseline_bucket_source.csv"
)
ABLATION_BUCKET_PATH = (
    ARTICLE_SOURCE / "fig_proxy_failure_anatomy_2x6_ablation_bucket_source.csv"
)
BRIDGE_PATH = (
    FINAL_DIR
    / "distribution_analysis"
    / "directional_representation_bridge_v4"
    / "tables"
    / "02_directional_representation_merged_seed_aligned.csv"
)
CORRELATION_PATH = (
    FINAL_DIR
    / "distribution_analysis"
    / "maintext_composites_20260628"
    / "source_data"
    / "figure5_predictor_f1_spearman_bootstrap_summary.csv"
)

INK = "#202124"
MUTED = "#687078"
GRID = "#E5E9EC"
BLUE = "#28688E"
SALMON = "#D58E79"
TEAL = "#3D9896"
GOLD = "#C9A23F"

METHOD_COLORS = {
    "CoDiPO": BLUE,
    "EDA": "#D8A7B1",
    "ICL Direct": "#D9907A",
    "ICL Imitation": "#B7D3B0",
    "ICL Rewrite": "#A7C7D9",
    "Hard filter": TEAL,
    "w/o X": "#C67B6C",
    "w/o Y/Z": "#6F9B76",
    "Cosine sim. as X": "#737373",
    "Cosine-only": GOLD,
    "Vanilla": "#8D96A0",
}

RAW_TO_DISPLAY = {
    "CoDiPO": "CoDiPO",
    "EDA": "EDA",
    "ICL Direct": "ICL Direct",
    "ICL Imitate": "ICL Imitation",
    "ICL Imitation": "ICL Imitation",
    "ICL Rewrite": "ICL Rewrite",
    "XYZ Hard Filtering w/o DPO": "Hard filter",
    "Hard filter": "Hard filter",
    "w/o X": "w/o X",
    "w/o YZ": "w/o Y/Z",
    "w/o Y/Z": "w/o Y/Z",
    "Cosine as X": "Cosine sim. as X",
    "Cos-X": "Cosine sim. as X",
    "Cosine-only Preference": "Cosine-only",
    "Cosine-only": "Cosine-only",
    "Vanilla Generation w/o DPO": "Vanilla",
    "Vanilla": "Vanilla",
}

BASELINE_CORE = ["CoDiPO", "EDA", "ICL Direct", "ICL Imitation", "ICL Rewrite"]
BASELINE_WITH_HARD = BASELINE_CORE + ["Hard filter"]
ABLATION_ORDER = [
    "CoDiPO",
    "w/o X",
    "w/o Y/Z",
    "Cosine sim. as X",
    "Cosine-only",
    "Vanilla",
    "Hard filter",
]

BUCKETS = [
    ("yz_pass_x_pass_percent_mean", "Y/Z pass / X pass", BLUE),
    ("yz_pass_x_fail_percent_mean", "Y/Z pass / X fail", "#F3E7B7"),
    ("yz_fail_x_pass_percent_mean", "Y/Z fail / X pass", "#F0AAA3"),
    ("yz_fail_x_fail_percent_mean", "Y/Z fail / X fail", "#DDD5DB"),
]




def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 7.0,
            "axes.titlesize": 7.5,
            "axes.labelsize": 6.9,
            "axes.linewidth": 0.65,
            "axes.edgecolor": INK,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "legend.fontsize": 5.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )


def clean_axis(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis in {"x", "y", "both"}:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.52)
    ax.set_axisbelow(True)


def panel_label(
    ax: plt.Axes,
    label: str,
    x: float = -0.15,
    y: float = 1.05,
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color=INK,
    )


def canonicalize_method(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["method"] = frame["method"].replace(RAW_TO_DISPLAY)
    return frame


def save_bundle(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for extension, kwargs in {
        "svg": {},
        "pdf": {},
        "png": {"dpi": 450},
        "tiff": {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}},
    }.items():
        path = FIG_DIR / f"{stem}.{extension}"
        try:
            fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        except TypeError:
            kwargs.pop("pil_kwargs", None)
            fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)


def method_handles(methods: list[str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor=INK,
            markeredgewidth=0.45,
            markersize=4.6,
            label=method,
        )
        for method in methods
    ]


def draw_bucket_stack(
    ax: plt.Axes,
    frame: pd.DataFrame,
    order: list[str],
    title: str,
) -> None:
    data = canonicalize_method(frame).set_index("method").reindex(order)
    x = np.arange(len(order))
    bottom = np.zeros(len(order))
    for column, label, color in BUCKETS:
        values = data[column].to_numpy(dtype=float)
        ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.68,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=label,
            zorder=3,
        )
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Generated candidates (%)")
    ax.set_title(title, loc="left", fontweight="bold", pad=4.0)
    clean_axis(ax, "y")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.39),
        ncol=2,
        columnspacing=0.9,
        handlelength=1.2,
        handletextpad=0.3,
        borderaxespad=0,
    )


def proxy_method_mean(quality: pd.DataFrame, method: str) -> tuple[float, float]:
    row = quality[quality["method"].eq(method)]
    return (
        float(row["residual_cos_mean"].mean()),
        float(row["yz_violation_mean"].mean()),
    )


def build_proxy_balance_figure() -> None:
    quality = pd.read_csv(QUALITY_PATH)
    boundaries = pd.read_csv(BOUNDARY_PATH)
    utility = pd.read_csv(UTILITY_PATH)
    baseline_bucket = pd.read_csv(BASELINE_BUCKET_PATH)
    ablation_bucket = pd.read_csv(ABLATION_BUCKET_PATH)

    quality.to_csv(SOURCE_DIR / "figure_a_proxy_quality_seed_source.csv", index=False)
    utility.to_csv(SOURCE_DIR / "figure_a_proxy_utility_source.csv", index=False)
    canonicalize_method(baseline_bucket).to_csv(
        SOURCE_DIR / "figure_a_baseline_bucket_source.csv",
        index=False,
    )
    canonicalize_method(ablation_bucket).to_csv(
        SOURCE_DIR / "figure_a_ablation_bucket_source.csv",
        index=False,
    )

    fig = plt.figure(figsize=(183 / 25.4, 126 / 25.4), facecolor="white")
    outer = fig.add_gridspec(
        2,
        1,
        left=0.065,
        right=0.992,
        top=0.955,
        bottom=0.15,
        height_ratios=[1.02, 0.88],
        hspace=0.72,
    )
    top = outer[0].subgridspec(
        1,
        5,
        width_ratios=[1.0, 1.0, 1.0, 1.48, 1.48],
        wspace=0.54,
    )
    bottom = outer[1].subgridspec(1, 2, wspace=0.30)
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    ax_c = fig.add_subplot(top[0, 2])
    ax_d = fig.add_subplot(top[0, 3:])
    ax_e = fig.add_subplot(bottom[0, 0])
    ax_f = fig.add_subplot(bottom[0, 1])

    proxy.draw_quality_panel(
        ax_a,
        quality,
        proxy.PREFERENCE_METHODS,
        boundaries,
        "Preference controls",
    )
    proxy.draw_quality_panel(
        ax_b,
        quality,
        proxy.BASELINE_METHODS,
        boundaries,
        "Augmentation baselines",
    )
    proxy.draw_quality_panel(
        ax_c,
        quality,
        proxy.REFERENCE_METHODS,
        boundaries,
        "Soft vs hard constraint",
    )
    proxy.draw_utility_panel(ax_d, utility)
    ax_d.set_title(
        "Utility peaks near a proxy balance",
        loc="left",
        fontweight="bold",
        pad=5.0,
    )

    vanilla_xy = proxy_method_mean(quality, "Vanilla Generation w/o DPO")
    codipo_xy = proxy_method_mean(quality, "CoDiPO")
    hard_xy = proxy_method_mean(quality, "XYZ Hard Filtering w/o DPO")
    ax_c.annotate(
        "",
        xy=codipo_xy,
        xytext=vanilla_xy,
        arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.0},
        zorder=6,
    )
    ax_c.annotate(
        "soft preference\nshift",
        xy=((codipo_xy[0] + vanilla_xy[0]) / 2, (codipo_xy[1] + vanilla_xy[1]) / 2),
        xytext=(-3, 4),
        textcoords="offset points",
        fontsize=5.4,
        color=BLUE,
        ha="right",
    )
    ax_c.annotate(
        "",
        xy=hard_xy,
        xytext=vanilla_xy,
        arrowprops={"arrowstyle": "->", "color": TEAL, "lw": 1.0},
        zorder=6,
    )
    ax_c.annotate(
        "hard selection",
        xy=((hard_xy[0] + vanilla_xy[0]) / 2, (hard_xy[1] + vanilla_xy[1]) / 2),
        xytext=(2, -7),
        textcoords="offset points",
        fontsize=5.4,
        color=TEAL,
        ha="left",
    )

    draw_bucket_stack(
        ax_e,
        baseline_bucket,
        BASELINE_CORE,
        "Baseline distributions occupy distinct proxy regions",
    )
    draw_bucket_stack(
        ax_f,
        ablation_bucket,
        ABLATION_ORDER,
        "Preference learning and hard selection reach different distributions",
    )

    panel_x = {"a": -0.15, "b": -0.15, "c": -0.24, "d": -0.12, "e": -0.20, "f": -0.15}
    for label, ax in zip("abcdef", [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f]):
        panel_label(ax, label, x=panel_x[label])

    top_methods = [
        "CoDiPO",
        "w/o X",
        "w/o Y/Z",
        "Cosine sim. as X",
        "Cosine-only",
        "Vanilla",
        "Hard filter",
        "EDA",
        "ICL Direct",
        "ICL Rewrite",
        "ICL Imitation",
    ]
    fig.legend(
        handles=method_handles(top_methods),
        loc="center",
        bbox_to_anchor=(0.5, 0.525),
        ncol=6,
        columnspacing=0.75,
        handletextpad=0.28,
    )
    save_bundle(fig, "figA_task_specific_proxy_balance")
    plt.close(fig)


def method_summary(bridge: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "ad_overcompletion_score_mean",
        "hc_evidence_loss_score_mean",
        "overall_contamination",
    ]
    means = bridge.groupby("method", observed=True)[metrics].mean().add_suffix("_mean")
    sds = bridge.groupby("method", observed=True)[metrics].std().add_suffix("_sd")
    out = means.join(sds).reindex(BASELINE_WITH_HARD).reset_index()
    out["overall_contamination_mean"] *= 100.0
    out["overall_contamination_sd"] *= 100.0
    return out


def draw_failure_map(
    ax: plt.Axes,
    bridge: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    for method in BASELINE_WITH_HARD:
        part = bridge[bridge["method"].eq(method)]
        color = METHOD_COLORS[method]
        ax.scatter(
            part["ad_overcompletion_score_mean"],
            part["hc_evidence_loss_score_mean"],
            s=14,
            color=color,
            alpha=0.28,
            linewidth=0,
            zorder=1,
        )
        row = summary[summary["method"].eq(method)].iloc[0]
        ax.errorbar(
            row["ad_overcompletion_score_mean_mean"],
            row["hc_evidence_loss_score_mean_mean"],
            xerr=row["ad_overcompletion_score_mean_sd"],
            yerr=row["hc_evidence_loss_score_mean_sd"],
            fmt="none",
            ecolor=color,
            elinewidth=0.8,
            capsize=2.0,
            alpha=0.88,
            zorder=2,
        )
        ax.scatter(
            row["ad_overcompletion_score_mean_mean"],
            row["hc_evidence_loss_score_mean_mean"],
            s=42,
            color=color,
            edgecolor=INK,
            linewidth=0.55,
            zorder=3,
        )

    offsets = {
        "CoDiPO": (-0.08, 0.17, "right"),
        "EDA": (0.10, 0.10, "left"),
        "ICL Direct": (-0.10, 0.16, "right"),
        "ICL Imitation": (0.10, 0.13, "left"),
        "ICL Rewrite": (0.10, -0.14, "left"),
        "Hard filter": (0.10, -0.10, "left"),
    }
    for method in BASELINE_WITH_HARD:
        row = summary[summary["method"].eq(method)].iloc[0]
        dx, dy, align = offsets[method]
        ax.text(
            row["ad_overcompletion_score_mean_mean"] + dx,
            row["hc_evidence_loss_score_mean_mean"] + dy,
            method,
            ha=align,
            va="center",
            fontsize=5.9,
            color=METHOD_COLORS[method],
            fontweight="bold" if method == "CoDiPO" else "normal",
        )
    ax.annotate(
        "Generative over-completion",
        xy=(4.45, 0.24),
        xytext=(2.75, 0.24),
        arrowprops={"arrowstyle": "->", "color": SALMON, "lw": 0.9},
        fontsize=6.0,
        color=SALMON,
        ha="center",
        va="center",
    )
    ax.annotate(
        "Lexical evidence loss",
        xy=(0.33, 2.45),
        xytext=(0.33, 1.62),
        arrowprops={"arrowstyle": "->", "color": "#B47786", "lw": 0.9},
        fontsize=6.0,
        color="#B47786",
        ha="center",
        va="center",
        rotation=90,
    )
    ax.set_xlim(0.22, 4.65)
    ax.set_ylim(0.12, 2.62)
    ax.set_xlabel("AD-source over-completion severity")
    ax.set_ylabel("HC-source evidence-loss severity")
    ax.set_title(
        "CoDiPO avoids the dominant failure extremes of baseline families",
        loc="left",
        fontweight="bold",
        pad=5.0,
    )
    clean_axis(ax, "both")


def draw_summary_bar(
    ax: plt.Axes,
    summary: pd.DataFrame,
    mean_col: str,
    sd_col: str,
    title: str,
    xlabel: str,
    show_labels: bool,
) -> None:
    y = np.arange(len(summary))
    colors = [METHOD_COLORS[method] for method in summary["method"]]
    ax.barh(
        y,
        summary[mean_col],
        xerr=summary[sd_col],
        color=colors,
        edgecolor="white",
        linewidth=0.35,
        capsize=1.8,
        error_kw={"elinewidth": 0.75, "ecolor": INK},
        zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(summary["method"] if show_labels else [])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontweight="bold", pad=4.0)
    clean_axis(ax, "x")


def build_directional_failure_figure(bridge: pd.DataFrame) -> None:
    summary = method_summary(bridge)
    bridge.to_csv(SOURCE_DIR / "figure_b_directional_seed_source.csv", index=False)
    summary.to_csv(SOURCE_DIR / "figure_b_directional_method_summary.csv", index=False)

    fig = plt.figure(figsize=(183 / 25.4, 114 / 25.4), facecolor="white")
    outer = fig.add_gridspec(
        2,
        1,
        left=0.075,
        right=0.992,
        top=0.95,
        bottom=0.11,
        height_ratios=[1.18, 0.92],
        hspace=0.46,
    )
    ax_a = fig.add_subplot(outer[0, 0])
    bottom = outer[1].subgridspec(1, 3, wspace=0.42)
    ax_b, ax_c, ax_d = [fig.add_subplot(bottom[0, i]) for i in range(3)]

    draw_failure_map(ax_a, bridge, summary)
    draw_summary_bar(
        ax_b,
        summary,
        "ad_overcompletion_score_mean_mean",
        "ad_overcompletion_score_mean_sd",
        "AD-source over-completion",
        "Continuous upper-band severity",
        True,
    )
    draw_summary_bar(
        ax_c,
        summary,
        "hc_evidence_loss_score_mean_mean",
        "hc_evidence_loss_score_mean_sd",
        "HC-source evidence loss",
        "Continuous lower-band severity",
        False,
    )
    draw_summary_bar(
        ax_d,
        summary,
        "overall_contamination_mean",
        "overall_contamination_sd",
        "Opposite-class contamination",
        "Opposite-class kNN rate (%)",
        False,
    )

    for label, ax in zip("abcd", [ax_a, ax_b, ax_c, ax_d]):
        panel_label(ax, label, x=-0.11 if label == "a" else -0.22)
    save_bundle(fig, "figB_opposing_directional_failure_axes")
    plt.close(fig)


def regression_line(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    if len(x) < 2 or np.unique(x).size < 2:
        return
    slope, intercept = np.polyfit(x, y, 1)
    xx = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    ax.plot(xx, intercept + slope * xx, color="#5D646A", linewidth=1.0, zorder=2)


def draw_bridge_scatter(
    ax: plt.Axes,
    bridge: pd.DataFrame,
    xcol: str,
    ycol: str,
    xlabel: str,
    ylabel: str,
    title: str,
    xscale: float = 1.0,
    yscale: float = 1.0,
    hero: bool = False,
) -> dict[str, float | int | str]:
    subset = bridge[["method", "seed", xcol, ycol]].dropna().copy()
    subset["x_plot"] = subset[xcol] * xscale
    subset["y_plot"] = subset[ycol] * yscale
    for method in BASELINE_WITH_HARD:
        part = subset[subset["method"].eq(method)]
        ax.scatter(
            part["x_plot"],
            part["y_plot"],
            s=15 if not hero else 17,
            color=METHOD_COLORS[method],
            alpha=0.35,
            linewidth=0,
            zorder=1,
        )
        mean_x = float(part["x_plot"].mean())
        mean_y = float(part["y_plot"].mean())
        ax.scatter(
            mean_x,
            mean_y,
            s=32 if not hero else 38,
            color=METHOD_COLORS[method],
            edgecolor=INK,
            linewidth=0.5,
            zorder=3,
        )
    x = subset["x_plot"].to_numpy(dtype=float)
    y = subset["y_plot"].to_numpy(dtype=float)
    regression_line(ax, x, y)
    rho = float(spearmanr(x, y).statistic)
    ax.text(
        0.035,
        0.955,
        rf"$r_s$ = {rho:.2f}; n = {len(subset)} method-seed observations",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.9,
        color=INK,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold", pad=4.0)
    clean_axis(ax, "both")
    return {
        "x": xcol,
        "y": ycol,
        "spearman_rho": rho,
        "n": len(subset),
        "methods": "; ".join(BASELINE_WITH_HARD),
    }


def draw_correlation_forest(ax: plt.Axes, correlations: pd.DataFrame) -> None:
    predictor_order = [
        "AD over-completion",
        "HC evidence loss",
        "Opposite-class contamination",
        "Source\u2013augmentation distance",
        "Train\u2013evaluation MMD",
    ]
    display_labels = [
        "AD over-completion",
        "HC evidence loss",
        "kNN contamination",
        "Source distance",
        "Train-eval MMD",
    ]
    scopes = [
        ("Baseline family", BLUE, "o", -0.08),
        ("All aligned methods", SALMON, "s", 0.08),
    ]
    y = np.arange(len(predictor_order))[::-1]
    ax.axvline(0, color="#A7ADB1", linestyle="--", linewidth=0.75, zorder=0)
    for scope, color, marker, offset in scopes:
        part = (
            correlations[correlations["scope"].eq(scope)]
            .set_index("predictor")
            .reindex(predictor_order)
        )
        estimates = part["spearman_rho"].to_numpy(dtype=float)
        lows = part["ci_low"].to_numpy(dtype=float)
        highs = part["ci_high"].to_numpy(dtype=float)
        ax.errorbar(
            estimates,
            y + offset,
            xerr=np.vstack([estimates - lows, highs - estimates]),
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=0.8,
            capsize=2.0,
            markersize=4.2,
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=scope,
            zorder=3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(display_labels)
    ax.set_xlim(-1.0, 0.75)
    ax.set_ylim(-0.35, 4.75)
    ax.set_xlabel(r"Spearman $r_s$ with OverallMacro F1")
    ax.set_title(
        "Contamination is the strongest stable utility correlate",
        loc="left",
        fontweight="bold",
        pad=4.0,
    )
    clean_axis(ax, "x")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        borderaxespad=0,
        columnspacing=0.8,
        handletextpad=0.3,
    )


def build_representation_bridge_figure(
    bridge: pd.DataFrame,
    correlations: pd.DataFrame,
) -> None:
    bridge.to_csv(SOURCE_DIR / "figure_c_bridge_seed_source.csv", index=False)
    correlations.to_csv(
        SOURCE_DIR / "figure_c_predictor_correlation_source.csv",
        index=False,
    )

    fig = plt.figure(figsize=(183 / 25.4, 124 / 25.4), facecolor="white")
    outer = fig.add_gridspec(
        2,
        1,
        left=0.075,
        right=0.992,
        top=0.95,
        bottom=0.15,
        height_ratios=[1.0, 1.12],
        hspace=0.50,
    )
    top = outer[0].subgridspec(1, 2, wspace=0.34)
    bottom = outer[1].subgridspec(
        1,
        2,
        width_ratios=[1.42, 1.0],
        wspace=0.46,
    )
    ax_a, ax_b = [fig.add_subplot(top[0, i]) for i in range(2)]
    ax_c = fig.add_subplot(bottom[0, 0])
    ax_d = fig.add_subplot(bottom[0, 1])

    stats = []
    stats.append(
        draw_bridge_scatter(
            ax_a,
            bridge,
            "ad_overcompletion_score_mean",
            "ad_to_hc_contamination",
            "AD-source upper-band severity",
            "AD-to-HC contamination (%)",
            "AD over-completion maps into HC-like neighborhoods",
            yscale=100.0,
        )
    )
    stats.append(
        draw_bridge_scatter(
            ax_b,
            bridge,
            "hc_evidence_loss_score_mean",
            "hc_to_ad_contamination",
            "HC-source lower-band severity",
            "HC-to-AD contamination (%)",
            "HC evidence loss maps into AD-like neighborhoods",
            yscale=100.0,
        )
    )
    stats.append(
        draw_bridge_scatter(
            ax_c,
            bridge,
            "overall_contamination",
            "OverallMacro_F1",
            "Opposite-class kNN contamination (%)",
            "OverallMacro F1 (%)",
            "Class contamination links directional distortion to lost utility",
            xscale=100.0,
            hero=True,
        )
    )
    draw_correlation_forest(ax_d, correlations)

    panel_x = {"a": -0.16, "b": -0.16, "c": -0.14, "d": -0.20}
    for label, ax in zip("abcd", [ax_a, ax_b, ax_c, ax_d]):
        panel_label(ax, label, x=panel_x[label])

    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=METHOD_COLORS[method],
                markeredgecolor="none",
                markersize=4.5,
                label=method,
            )
            for method in BASELINE_WITH_HARD
        ],
        loc="lower center",
        bbox_to_anchor=(0.39, 0.025),
        ncol=6,
        columnspacing=0.8,
        handletextpad=0.28,
    )
    pd.DataFrame(stats).to_csv(
        SOURCE_DIR / "figure_c_panel_statistics.csv",
        index=False,
    )
    save_bundle(fig, "figC_directional_contamination_utility_bridge")
    plt.close(fig)


def main() -> None:
    set_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    bridge = pd.read_csv(BRIDGE_PATH)
    bridge = bridge[bridge["method"].isin(BASELINE_WITH_HARD)].copy()
    correlations = pd.read_csv(CORRELATION_PATH)

    build_proxy_balance_figure()
    build_directional_failure_figure(bridge)
    build_representation_bridge_figure(bridge, correlations)
    print(f"Wrote review figures to {FIG_DIR}")
    print(f"Wrote source data to {SOURCE_DIR}")


