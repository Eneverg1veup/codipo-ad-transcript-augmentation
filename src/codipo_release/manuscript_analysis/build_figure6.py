#!/usr/bin/env python
"""Build the refined combined Figure 5 candidate.

Top panels reuse the original compact proxy-quality grammar. Lower panels add
baseline and ablation source-relative proxy-use audits. The refined version keeps
paired comparisons unconnected, uses explicit marker semantics, and applies
broken x-axes only where the pass-rate distribution has a real empty gap.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from . import _figure6_proxy_helpers as proxy

BRIDGE_PATH: Path
AUG_LONG_PATH: Path
FIG_DIR: Path
SOURCE_DIR: Path
EXTERNAL_BOOTSTRAP_CSV: Path
QUALITY_SUMMARY_CSV: Path
REGION_BOUNDARIES_CSV: Path

INK = proxy.INK
GRID = proxy.GRID
BLUE = proxy.BLUE


def load_external_utility_source(quality: pd.DataFrame) -> pd.DataFrame:
    file_range = (
        quality.groupby("method", observed=True)["joint_pass_rate"]
        .agg(joint_pass_mean="mean", joint_pass_sd="std", joint_pass_min="min", joint_pass_max="max")
        .reset_index()
    )
    external = pd.read_csv(EXTERNAL_BOOTSTRAP_CSV)
    external = external[external["metric"].eq("f1")][
        ["method_display", "point_estimate", "ci_low", "ci_high", "n_bootstrap"]
    ].rename(
        columns={
            "method_display": "locked_method_display",
            "point_estimate": "overall_f1_mean_percent",
            "ci_low": "overall_f1_ci_low",
            "ci_high": "overall_f1_ci_high",
        }
    )
    for column in ["overall_f1_mean_percent", "overall_f1_ci_low", "overall_f1_ci_high"]:
        external[column] = external[column].astype(float) * 100.0
    file_range["locked_method_display"] = file_range["method"].map(proxy.UTILITY_TO_LOCKED_METHOD)
    utility = file_range.merge(external, on="locked_method_display", how="left", validate="many_to_one")
    if utility[["overall_f1_mean_percent", "overall_f1_ci_low", "overall_f1_ci_high"]].isna().any().any():
        raise RuntimeError("Missing external F1 for proxy-utility methods")
    utility["source_definition"] = "joint proxy feasibility plus locked Lu-Pitt external-cohort F1"
    return utility

BASELINE_ORDER = ["CoDiPO", "EDA", "ICL Rewrite", "ICL Imitation"]
ABLATION_ORDER = [
    "CoDiPO",
    "Vanilla",
    "Hard Filter",
    "w/o X",
    "w/o YZ",
    "Cosine sim. as X",
    "Cosine-only",
]

DISPLAY_TO_RAW = {
    "CoDiPO": "CoDiPO",
    "EDA": "EDA",
    "ICL Rewrite": "ICL Rewrite",
    "ICL Imitation": "ICL Imitation",
    "Vanilla": "Vanilla",
    "Hard Filter": "Hard filter",
    "w/o X": "w/o X",
    "w/o YZ": "w/o Y/Z",
    "Cosine sim. as X": "Cos-X",
    "Cosine-only": "Cosine-only",
}
RAW_TO_DISPLAY = {raw: display for display, raw in DISPLAY_TO_RAW.items()}
RAW_TO_DISPLAY.update({"ICL Imitate": "ICL Imitation", "Cosine as X": "Cosine sim. as X"})

METHOD_COLORS = {
    "CoDiPO": proxy.METHOD_COLORS["CoDiPO"],
    "EDA": proxy.METHOD_COLORS["EDA"],
    "ICL Rewrite": proxy.METHOD_COLORS["ICL Rewrite"],
    "ICL Imitation": proxy.METHOD_COLORS["ICL Imitate"],
    "Vanilla": "#A6AFB7",
    "Hard Filter": proxy.METHOD_COLORS["XYZ Hard Filtering w/o DPO"],
    "w/o X": proxy.METHOD_COLORS["w/o X"],
    "w/o YZ": proxy.METHOD_COLORS["w/o YZ"],
    "Cosine sim. as X": "#7A5AA6",
    "Cosine-only": proxy.METHOD_COLORS["Cosine-only Preference"],
}

METHOD_LEGEND_ORDER = [
    "CoDiPO",
    "EDA",
    "ICL Rewrite",
    "ICL Imitation",
    "Vanilla",
    "Hard Filter",
    "w/o X",
    "w/o YZ",
    "Cosine sim. as X",
    "Cosine-only",
]

SINGLE_PROXY_METRICS = [
    ("proxy_yz_pass_rate", "Y/Z pass"),
    ("proxy_x_pass_rate", "X pass"),
    ("proxy_joint_pass_rate", "Joint pass"),
]

# Only ablation panels use broken axes. X pass is intentionally left continuous.
ABLATION_SINGLE_PROXY_BREAKS = {
    "proxy_yz_pass_rate": {
        "break_cfg": (30.0, 60.0, 34.0),
        "left_ticks": [0, 10, 20, 30],
        "right_ticks": [60, 65, 70],
        "xlim": (0, 45),
    },
    "proxy_joint_pass_rate": {
        "break_cfg": (12.0, 24.0, 15.0),
        "left_ticks": [0, 5, 10, 12],
        "right_ticks": [24, 27, 30],
        "xlim": (0, 22),
    },
}


def configure_proxy_module() -> None:
    proxy.FIG_DIR = FIG_DIR
    proxy.SOURCE_DIR = SOURCE_DIR
    proxy.UTILITY_METHODS = [m for m in proxy.UTILITY_METHODS if m != "ICL Direct"]
    proxy.METHOD_COLORS["Cosine as X"] = METHOD_COLORS["Cosine sim. as X"]
    proxy.METHOD_COLORS["Vanilla Generation w/o DPO"] = METHOD_COLORS["Vanilla"]
    proxy.METHOD_LABELS["ICL Imitate"] = "ICL imitation"
    proxy.METHOD_LABELS["ICL Rewrite"] = "ICL rewrite"




def setup_refined_style() -> None:
    """Increase print readability while keeping the dense multi-panel layout."""
    proxy.setup_style()
    plt.rcParams.update(
        {
            "font.size": 7.9,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.6,
            "xtick.labelsize": 6.75,
            "ytick.labelsize": 6.75,
            "legend.fontsize": 6.65,
            "axes.linewidth": 0.70,
            "xtick.major.width": 0.60,
            "ytick.major.width": 0.60,
        }
    )


def save_bundle(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for extension, kwargs in {
        "pdf": {},
        "svg": {},
        "png": {"dpi": 600},
        "tiff": {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}},
    }.items():
        path = FIG_DIR / f"{stem}.{extension}"
        try:
            fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        except TypeError:
            kwargs.pop("pil_kwargs", None)
            fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)


def clean_axis(ax: plt.Axes, axis: str = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=axis, color=GRID, linewidth=0.45)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str, x: float = -0.14, y: float = 1.05) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.2,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )


def broken_x(value: float, left_max: float, right_min: float, right_start: float) -> float:
    if value <= left_max:
        return value
    return right_start + (value - right_min)


def clipped_error_bounds(
    x: float,
    err: float,
    left_max: float,
    right_min: float,
    right_start: float,
) -> tuple[float, float, float]:
    """Map a point and clip its error bar so it never crosses the hidden gap."""
    if pd.isna(err):
        err = 0.0
    mapped = broken_x(x, left_max, right_min, right_start)
    if x <= left_max:
        lo = x - err
        hi = min(x + err, left_max)
    elif x >= right_min:
        lo = max(x - err, right_min)
        hi = x + err
    else:
        # A mean in the hidden gap is a bad axis choice. Draw a zero-width
        # interval at the original value rather than silently fabricating one.
        lo = x
        hi = x
    return (
        broken_x(lo, left_max, right_min, right_start),
        mapped,
        broken_x(hi, left_max, right_min, right_start),
    )


def draw_broken_marks(ax: plt.Axes, left_max: float, right_start: float) -> None:
    trans = ax.get_xaxis_transform()
    gap = right_start - left_max
    for xpos in [left_max + gap * 0.38, left_max + gap * 0.62]:
        ax.plot(
            [xpos - 0.20, xpos + 0.20],
            [-0.025, 0.030],
            transform=trans,
            color=INK,
            linewidth=0.50,
            clip_on=False,
        )


def set_broken_xaxis(
    ax: plt.Axes,
    left_ticks: list[float],
    right_ticks: list[float],
    left_max: float,
    right_min: float,
    right_start: float,
    xlim: tuple[float, float],
) -> None:
    ticks = left_ticks + [broken_x(t, left_max, right_min, right_start) for t in right_ticks]
    labels = [f"{t:g}" for t in left_ticks + right_ticks]
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    draw_broken_marks(ax, left_max, right_start)


def draw_xerr_point(
    ax: plt.Axes,
    x: float,
    y: float,
    err: float,
    color: str,
    marker: str = "o",
    *,
    break_cfg: tuple[float, float, float] | None = None,
    size: float = 4.1,
) -> None:
    if pd.isna(err):
        err = 0.0
    if break_cfg is None:
        lo, mapped, hi = x - err, x, x + err
    else:
        lo, mapped, hi = clipped_error_bounds(x, err, *break_cfg)
    ax.hlines(y, lo, hi, color=color, linewidth=0.62, zorder=2)
    ax.vlines([lo, hi], y - 0.10, y + 0.10, color=color, linewidth=0.60, zorder=2)
    ax.scatter(
        mapped,
        y,
        marker=marker,
        s=size**2,
        color=color,
        edgecolor=INK,
        linewidth=0.35,
        zorder=3,
    )


def set_continuous_pass_axis(ax: plt.Axes, indexed: pd.DataFrame, metric: str) -> None:
    """Use an honest continuous x-axis for a pass-rate metric."""
    means = indexed[f"{metric}_mean"].astype(float)
    sds = indexed[f"{metric}_sd"].fillna(0).astype(float)
    data_min = float((means - sds).min())
    data_max = float((means + sds).max())
    if not np.isfinite(data_min) or not np.isfinite(data_max):
        ax.set_xlim(0, 70)
        return

    if metric == "proxy_x_pass_rate":
        lower = max(0, 5 * np.floor((data_min - 3) / 5))
        upper = min(100, 5 * np.ceil((data_max + 3) / 5))
        if upper - lower < 20:
            mid = 0.5 * (upper + lower)
            lower = max(0, 5 * np.floor((mid - 10) / 5))
            upper = min(100, 5 * np.ceil((mid + 10) / 5))
    else:
        lower = 0
        upper = min(100, max(18, 5 * np.ceil((data_max + 3) / 5)))

    ax.set_xlim(lower, upper)
    step = 10 if upper - lower > 25 else 5
    ax.set_xticks(np.arange(lower, upper + 0.1, step))


def load_proxy_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    bridge = pd.read_csv(BRIDGE_PATH)
    aug = pd.read_csv(AUG_LONG_PATH)
    bridge["method_display"] = bridge["method"].replace(RAW_TO_DISPLAY)
    aug["method_display"] = aug["method"].replace(RAW_TO_DISPLAY)
    return bridge, aug


def summarize_bridge(bridge: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    rows = []
    for display in order:
        raw = DISPLAY_TO_RAW[display]
        part = bridge[bridge["method"].eq(raw)]
        row = {"method": display, "method_raw": raw, "n_seeds": len(part)}
        for metric in [
            "proxy_yz_pass_rate",
            "proxy_x_pass_rate",
            "proxy_joint_pass_rate",
            "conditional_x_pass_within_yz_pass",
            "conditional_x_pass_within_yz_fail",
            "ExternalCohort_F1",
        ]:
            row[f"{metric}_mean"] = part[metric].mean()
            row[f"{metric}_sd"] = part[metric].std()
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_joint_source(aug: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    raw_methods = [DISPLAY_TO_RAW[m] for m in order]
    seed_source = (
        aug[aug["method"].isin(raw_methods)]
        .groupby(["method", "method_display", "seed", "source_label"], observed=True)["joint_pass"]
        .mean()
        .reset_index()
    )
    summary = (
        seed_source.groupby(["method", "method_display", "source_label"], observed=True)["joint_pass"]
        .agg(mean="mean", sd="std", n_seeds="size")
        .reset_index()
    )
    return summary


def draw_single_proxy_panel(
    fig: plt.Figure,
    slot,
    summary: pd.DataFrame,
    order: list[str],
    title: str,
    label: str,
    use_break: bool,
) -> None:
    grid = slot.subgridspec(1, 3, wspace=0.18)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    indexed = summary.set_index("method").reindex(order)
    y = np.arange(len(order))

    for ax, (metric, subtitle) in zip(axes, SINGLE_PROXY_METRICS):
        spec = ABLATION_SINGLE_PROXY_BREAKS.get(metric) if use_break else None
        break_cfg = None if spec is None else spec["break_cfg"]

        for yi, method in zip(y, order):
            row = indexed.loc[method]
            draw_xerr_point(
                ax,
                row[f"{metric}_mean"],
                yi,
                row[f"{metric}_sd"],
                METHOD_COLORS[method],
                break_cfg=break_cfg,
            )

        if spec is not None:
            set_broken_xaxis(
                ax,
                spec["left_ticks"],
                spec["right_ticks"],
                *break_cfg,
                xlim=spec["xlim"],
            )
        else:
            set_continuous_pass_axis(ax, indexed, metric)

        ax.set_title(subtitle, fontsize=7.0, fontweight="bold", pad=2)
        ax.set_xlabel("Pass rate (%)")
        ax.set_yticks(y)
        if ax is axes[0]:
            ax.set_yticklabels(order)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        ax.invert_yaxis()
        clean_axis(ax, "x")

    bbox = slot.get_position(fig)
    fig.text(
        bbox.x0,
        bbox.y1 + 0.014,
        title,
        ha="left",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        bbox.x0 - 0.025,
        bbox.y1 + 0.014,
        label,
        ha="left",
        va="bottom",
        fontsize=10.2,
        fontweight="bold",
        color=INK,
    )


def draw_conditional_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    order: list[str],
    title: str,
    break_cfg: tuple[float, float, float] | None,
    *,
    connect_pairs: bool = False,
) -> None:
    indexed = summary.set_index("method").reindex(order)
    y = np.arange(len(order))
    offset = 0.16
    for yi, method in zip(y, order):
        row = indexed.loc[method]
        values = [
            row["conditional_x_pass_within_yz_pass_mean"],
            row["conditional_x_pass_within_yz_fail_mean"],
        ]
        if connect_pairs:
            mapped = [broken_x(v, *break_cfg) if break_cfg else v for v in values]
            ax.plot(
                mapped,
                [yi - offset, yi + offset],
                color="#B8C0C5",
                alpha=0.45,
                linewidth=0.65,
                zorder=1,
            )
        draw_xerr_point(
            ax,
            values[0],
            yi - offset,
            row["conditional_x_pass_within_yz_pass_sd"],
            METHOD_COLORS[method],
            marker="o",
            break_cfg=break_cfg,
        )
        draw_xerr_point(
            ax,
            values[1],
            yi + offset,
            row["conditional_x_pass_within_yz_fail_sd"],
            METHOD_COLORS[method],
            marker="s",
            break_cfg=break_cfg,
            size=3.8,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    if break_cfg:
        set_broken_xaxis(ax, [30, 40, 50], [90, 95], *break_cfg, xlim=(25, 68))
    else:
        xmax = max(
            20,
            float(
                indexed[
                    [
                        "conditional_x_pass_within_yz_pass_mean",
                        "conditional_x_pass_within_yz_fail_mean",
                    ]
                ].max().max()
                + indexed[
                    [
                        "conditional_x_pass_within_yz_pass_sd",
                        "conditional_x_pass_within_yz_fail_sd",
                    ]
                ].fillna(0).max().max()
                + 4
            ),
        )
        ax.set_xlim(0, min(70, xmax))
    ax.set_xlabel("Conditional X-pass rate (%)")
    ax.set_title(title, loc="left", fontweight="bold", pad=4)
    clean_axis(ax, "x")


def draw_joint_source_panel(
    ax: plt.Axes,
    joint_summary: pd.DataFrame,
    order: list[str],
    title: str,
    break_cfg: tuple[float, float, float] | None,
    *,
    connect_pairs: bool = False,
) -> None:
    indexed = {
        (row["method_display"], int(row["source_label"])): row
        for _, row in joint_summary.iterrows()
    }
    y = np.arange(len(order))
    offset = 0.16
    for yi, method in zip(y, order):
        hc = indexed[(method, 0)]
        ad = indexed[(method, 1)]
        values = [hc["mean"] * 100, ad["mean"] * 100]
        if connect_pairs:
            mapped = [broken_x(v, *break_cfg) if break_cfg else v for v in values]
            ax.plot(
                mapped,
                [yi - offset, yi + offset],
                color="#B8C0C5",
                alpha=0.45,
                linewidth=0.65,
                zorder=1,
            )
        draw_xerr_point(
            ax,
            values[0],
            yi - offset,
            hc["sd"] * 100,
            METHOD_COLORS[method],
            marker="o",
            break_cfg=break_cfg,
        )
        draw_xerr_point(
            ax,
            values[1],
            yi + offset,
            ad["sd"] * 100,
            METHOD_COLORS[method],
            marker="^",
            break_cfg=break_cfg,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    if break_cfg:
        set_broken_xaxis(ax, [5, 10, 15], [25, 30, 35], *break_cfg, xlim=(0, 35))
    else:
        ax.set_xlim(0, 15)
    ax.set_xlabel("Joint pass rate (%)")
    ax.set_title(title, loc="left", fontweight="bold", pad=4)
    clean_axis(ax, "x")


def legend_handles(methods: list[str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=METHOD_COLORS[m],
            markeredgecolor=INK,
            markeredgewidth=0.35,
            markersize=4.8,
            label=m,
        )
        for m in methods
    ]


def marker_semantics_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=INK,
            markeredgewidth=0.60,
            markersize=4.6,
            label="Y/Z-pass or HC source",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=INK,
            markeredgewidth=0.60,
            markersize=4.25,
            label="Y/Z-fail",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=INK,
            markeredgewidth=0.60,
            markersize=4.6,
            label="AD source",
        ),
    ]


def add_shared_legends(fig: plt.Figure) -> None:
    method_legend = fig.legend(
        handles=legend_handles(METHOD_LEGEND_ORDER),
        loc="upper center",
        ncol=10,
        bbox_to_anchor=(0.5, 0.990),
        frameon=False,
        columnspacing=0.42,
        handletextpad=0.22,
        fontsize=6.65,
    )
    fig.add_artist(method_legend)
    fig.legend(
        handles=marker_semantics_handles(),
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.963),
        frameon=False,
        columnspacing=0.66,
        handletextpad=0.26,
        fontsize=6.45,
    )


def build_figure() -> None:
    configure_proxy_module()
    setup_refined_style()
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    quality = pd.read_csv(QUALITY_SUMMARY_CSV)
    boundaries = pd.read_csv(REGION_BOUNDARIES_CSV)
    utility = load_external_utility_source(quality)
    utility = utility[utility["method"].ne("ICL Direct")].copy()
    bridge, aug = load_proxy_audit()
    baseline_summary = summarize_bridge(bridge, BASELINE_ORDER)
    ablation_summary = summarize_bridge(bridge, ABLATION_ORDER)
    baseline_joint = summarize_joint_source(aug, BASELINE_ORDER)
    ablation_joint = summarize_joint_source(aug, ABLATION_ORDER)

    quality.to_csv(SOURCE_DIR / "figure5_top_proxy_quality_space_source.csv", index=False)
    utility.to_csv(SOURCE_DIR / "figure5_top_proxy_utility_source.csv", index=False)
    baseline_summary.to_csv(SOURCE_DIR / "figure5_baseline_proxy_audit_summary.csv", index=False)
    ablation_summary.to_csv(SOURCE_DIR / "figure5_ablation_proxy_audit_summary.csv", index=False)
    baseline_joint.to_csv(SOURCE_DIR / "figure5_baseline_joint_source_summary.csv", index=False)
    ablation_joint.to_csv(SOURCE_DIR / "figure5_ablation_joint_source_summary.csv", index=False)

    fig = plt.figure(figsize=(11.2, 8.35), constrained_layout=False)
    outer = fig.add_gridspec(
        3,
        1,
        left=0.055,
        right=0.992,
        top=0.905,
        bottom=0.055,
        height_ratios=[1.0, 1.0, 1.0],
        hspace=0.50,
    )

    top = outer[0, 0].subgridspec(1, 5, width_ratios=[1, 1, 1, 1.32, 1.32], wspace=0.42)
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    ax_c = fig.add_subplot(top[0, 2])
    ax_d = fig.add_subplot(top[0, 3:])
    proxy.draw_quality_panel(ax_a, quality, proxy.PREFERENCE_METHODS, boundaries, "Preference controls")
    proxy.draw_quality_panel(ax_b, quality, ["CoDiPO", "EDA", "ICL Rewrite", "ICL Imitate"], boundaries, "Augmentation baselines")
    proxy.draw_quality_panel(ax_c, quality, proxy.REFERENCE_METHODS, boundaries, "Preference-guided vs\ndeterministic proxy use")
    proxy.draw_utility_panel(ax_d, utility)
    for annotation in list(ax_d.texts):
        annotation.remove()
    ax_d.set_title(
        "Joint proxy feasibility and held-out cross-cohort F1",
        loc="left",
        fontweight="bold",
        pad=5,
    )
    ax_d.set_ylabel("External-cohort average F1 (%)")
    for label, ax in zip("abcd", [ax_a, ax_b, ax_c, ax_d]):
        proxy.panel_label(ax, label, x=-0.19 if label == "c" else -0.12, y=1.12 if label == "c" else 1.06)

    baseline_row = outer[1, 0].subgridspec(1, 3, width_ratios=[1.85, 1.30, 1.30], wspace=0.36)
    draw_single_proxy_panel(fig, baseline_row[0, 0], baseline_summary, BASELINE_ORDER, "Baseline single-proxy pass rates", "e", False)
    ax_f = fig.add_subplot(baseline_row[0, 1])
    draw_conditional_panel(ax_f, baseline_summary, BASELINE_ORDER, "Baseline conditional X-pass", None)
    panel_label(ax_f, "f", x=-0.20)
    ax_g = fig.add_subplot(baseline_row[0, 2])
    draw_joint_source_panel(ax_g, baseline_joint, BASELINE_ORDER, "Baseline joint pass by source", None)
    panel_label(ax_g, "g", x=-0.20)

    ablation_row = outer[2, 0].subgridspec(1, 3, width_ratios=[1.85, 1.30, 1.30], wspace=0.36)
    draw_single_proxy_panel(fig, ablation_row[0, 0], ablation_summary, ABLATION_ORDER, "Ablation single-proxy pass rates", "h", True)
    ax_i = fig.add_subplot(ablation_row[0, 1])
    draw_conditional_panel(ax_i, ablation_summary, ABLATION_ORDER, "Ablation conditional X-pass", (55.0, 88.0, 60.0))
    panel_label(ax_i, "i", x=-0.20)
    ax_j = fig.add_subplot(ablation_row[0, 2])
    draw_joint_source_panel(ax_j, ablation_joint, ABLATION_ORDER, "Ablation joint pass by source", (17.0, 21.0, 20.0))
    panel_label(ax_j, "j", x=-0.20)

    add_shared_legends(fig)
    save_bundle(fig, "Figure6_proxy_quality_use_combined_external_20260720")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Figure 6 from locked aggregate and candidate-audit sources."
    )
    parser.add_argument("--quality-summary-csv", required=True, type=Path)
    parser.add_argument("--region-boundaries-csv", required=True, type=Path)
    parser.add_argument("--external-bootstrap-csv", required=True, type=Path)
    parser.add_argument("--utility-bridge-csv", required=True, type=Path)
    parser.add_argument("--augmentation-audit-csv", required=True, type=Path)
    parser.add_argument("--figure-dir", required=True, type=Path)
    parser.add_argument("--source-data-dir", required=True, type=Path)
    return parser.parse_args()


def configure(args: argparse.Namespace) -> None:
    global QUALITY_SUMMARY_CSV, REGION_BOUNDARIES_CSV, EXTERNAL_BOOTSTRAP_CSV
    global BRIDGE_PATH, AUG_LONG_PATH, FIG_DIR, SOURCE_DIR
    QUALITY_SUMMARY_CSV = args.quality_summary_csv
    REGION_BOUNDARIES_CSV = args.region_boundaries_csv
    EXTERNAL_BOOTSTRAP_CSV = args.external_bootstrap_csv
    BRIDGE_PATH = args.utility_bridge_csv
    AUG_LONG_PATH = args.augmentation_audit_csv
    FIG_DIR = args.figure_dir
    SOURCE_DIR = args.source_data_dir


def main() -> None:
    configure(parse_args())
    build_figure()
    print(f"Wrote combined Figure 5 to {FIG_DIR}")
    print(f"Wrote source data to {SOURCE_DIR}")


if __name__ == "__main__":
    main()
