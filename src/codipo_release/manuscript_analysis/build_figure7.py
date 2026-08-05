#!/usr/bin/env python
"""Build a joint baseline/ablation Figure 7 candidate.

The left block follows the current baseline FigB logic. ICL Direct is excluded
from source-relative Y--Z and paired-source residual-similarity panels, and
retained in the source-independent PCA/kNN mixing panels. The right block
mirrors the same analyses for ablation controls.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from . import _figure7_ablation as abl
from . import _figure7_baseline as v3

FIG_DIR: Path
SOURCE_DIR: Path
EXTERNAL_BRIDGE: Path
AUGMENTATION_AUDIT_CSV: Path
ABLATION_UTILITY_CSV: Path
BASELINE_PCA_CSV: Path
BASELINE_PCA_REFERENCE_CSV: Path
ABLATION_PCA_CSV: Path
ABLATION_PCA_REFERENCE_CSV: Path
base = v3.base
atlas = v3.v2.atlas

BASELINE_FAILURE_METHODS = ["CoDiPO", "EDA", "ICL Imitation", "ICL Rewrite", "Hard filter"]
BASELINE_REP_METHODS = list(base.BASELINE_WITH_HARD)
BASELINE_RESIDUAL_METHODS = [method for method in BASELINE_REP_METHODS if method != "ICL Direct"]
ABLATION_RAW_METHODS = list(abl.ABLATION_METHODS)
BASELINE_PCA_METHODS_MAIN = list(BASELINE_REP_METHODS)
ABLATION_PCA_METHODS_MAIN = list(ABLATION_RAW_METHODS)
ABLATION_DISPLAY = dict(abl.DISPLAY_LABELS)


def color_get(mapping: dict[str, str], *keys: str, default: str) -> str:
    """Return the first available color while tolerating raw/display name drift."""
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


BASELINE_COLORS = dict(base.METHOD_COLORS)
BASELINE_COLORS["ICL Imitation"] = color_get(
    BASELINE_COLORS, "ICL Imitation", "ICL Imitate", default="#B7D3B0"
)
BASELINE_COLORS["ICL Imitate"] = BASELINE_COLORS["ICL Imitation"]
BASELINE_COLORS["Hard filter"] = color_get(
    BASELINE_COLORS, "Hard filter", "XYZ Hard Filtering w/o DPO", default="#3D9896"
)
BASELINE_COLORS["w/o Y/Z"] = color_get(BASELINE_COLORS, "w/o Y/Z", "w/o YZ", default="#6F9B76")
BASELINE_COLORS["Cosine-only"] = color_get(
    BASELINE_COLORS, "Cosine-only", "Cosine-only Preference", default="#C9A23F"
)
BASELINE_COLORS["Vanilla"] = color_get(
    BASELINE_COLORS, "Vanilla", "Vanilla Generation w/o DPO", default="#A6AFB7"
)
BASELINE_COLORS["Cosine sim. as X"] = color_get(
    BASELINE_COLORS, "Cosine sim. as X", "Cosine as X", "Cos-X", default="#7A5AA6"
)

ABLATION_COLORS = {
    "CoDiPO": color_get(BASELINE_COLORS, "CoDiPO", default="#28688E"),
    "w/o X": color_get(BASELINE_COLORS, "w/o X", default="#C67B6C"),
    "w/o Y/Z": color_get(BASELINE_COLORS, "w/o Y/Z", "w/o YZ", default="#6F9B76"),
    "Cos-X": color_get(BASELINE_COLORS, "Cosine sim. as X", "Cosine as X", "Cos-X", default="#7A5AA6"),
    "Cosine-only": color_get(BASELINE_COLORS, "Cosine-only", "Cosine-only Preference", default="#C9A23F"),
    "Vanilla": color_get(BASELINE_COLORS, "Vanilla", "Vanilla Generation w/o DPO", default="#A6AFB7"),
    "Hard filter": color_get(BASELINE_COLORS, "Hard filter", "XYZ Hard Filtering w/o DPO", default="#3D9896"),
}

METHOD_ALIASES = {
    "ICL Imitation": "ICL Imitation",
    "ICL Imitate": "ICL Imitation",
    "Cosine sim. as X": "Cos-X",
}

DISPLAY_OVERRIDES = {
    "ICL Imitate": "ICL Imitation",
    "w/o X": "w/o X",
    "w/o Y/Z": "w/o YZ",
    "Cos-X": "Cosine sim. as X",
    "Hard filter": "Hard Filter",
}

PCA_TITLE_OVERRIDES = {
    "ICL Imitate": "ICL imitation",
    "ICL Imitation": "ICL imitation",
    "ICL Rewrite": "ICL rewrite",
    "w/o X": "w/o X",
    "w/o Y/Z": "w/o YZ",
    "Cos-X": "Cosine sim. as X",
    "Cosine sim. as X": "Cosine sim. as X",
    "Cosine-only": "Cosine-only",
}

# Baseline d keeps a broken x-axis because one baseline is far to the right.
KNN_BASELINE_BREAK_CFG = (44.0, 52.0, 46.0)
KNN_BASELINE_LEFT_TICKS = [35, 40, 44]
KNN_BASELINE_RIGHT_TICKS = [52, 55, 60]
KNN_BASELINE_BROKEN_XLIM = (35.0, 54.5)

# Ablation h should emphasize the clustered ablation differences, not inherit
# the wider baseline range.
KNN_ABLATION_XLIM = (37.0, 43.0)
KNN_ABLATION_TICKS = [37, 39, 41, 43]
KNN_ABLATION_ERROR_ALPHA = 0.28
KNN_ABLATION_ERROR_LINEWIDTH = 0.55
KNN_ABLATION_ERROR_CAP_LINEWIDTH = 0.48

# Ablation failure-axis e: separate Hard filter from the non-hard-filter cluster
# while preserving it as an anchor.
ABLATION_FAILURE_BREAK_CFG = (0.78, 1.12, 0.92)
ABLATION_FAILURE_LEFT_TICKS = [0.5, 0.75]
ABLATION_FAILURE_RIGHT_TICKS = [1.2, 1.5, 1.8, 2.1, 2.4]
ABLATION_FAILURE_BROKEN_XLIM = (0.35, 2.42)

ABLATION_RESIDUAL_XLIM = (0.10, 0.22)
ABLATION_RESIDUAL_TICKS = [0.10, 0.15, 0.20]

COMBINED_LEGEND_METHODS = [
    ("CoDiPO", "baseline"),
    ("EDA", "baseline"),
    ("ICL Direct", "baseline"),
    ("ICL Imitation", "baseline"),
    ("ICL Rewrite", "baseline"),
    ("Hard filter", "baseline"),
    ("w/o X", "ablation"),
    ("w/o Y/Z", "ablation"),
    ("Cos-X", "ablation"),
    ("Cosine-only", "ablation"),
    ("Vanilla", "ablation"),
]


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


def canonical_method(method: str) -> str:
    return METHOD_ALIASES.get(method, method)


def display_name(method: str) -> str:
    return DISPLAY_OVERRIDES.get(method, ABLATION_DISPLAY.get(method, method))


def pca_title_name(method: str) -> str:
    return PCA_TITLE_OVERRIDES.get(method, display_name(method))


def pca_method_color(method: str, family: str) -> str:
    """Match the original standalone PCA atlas title-color grammar."""
    if family == "ablation":
        display = ABLATION_DISPLAY.get(method, display_name(method))
        return color_get(
            base.METHOD_COLORS,
            display,
            method,
            canonical_method(method),
            default=method_color(method, family),
        )
    return method_color(method, family)


def method_color(method: str, family: str) -> str:
    key = canonical_method(method)
    if family == "ablation":
        if key in ABLATION_COLORS:
            return ABLATION_COLORS[key]
        if method in ABLATION_COLORS:
            return ABLATION_COLORS[method]
    if key in BASELINE_COLORS:
        return BASELINE_COLORS[key]
    if method in BASELINE_COLORS:
        return BASELINE_COLORS[method]
    raise KeyError(f"No color configured for method={method!r}, family={family!r}")


def assert_methods_present(frame: pd.DataFrame, methods: list[str], context: str) -> None:
    observed = set(frame["method"].dropna().astype(str).unique())
    missing = [method for method in methods if method not in observed]
    if missing:
        raise ValueError(f"{context}: missing methods in input data: {missing}")


def finite_sd(value: float) -> float:
    if pd.isna(value):
        return 0.0
    return float(value)


def setup_refined_style() -> None:
    """Start from the source figure style, then increase readability."""
    base.set_style()
    plt.rcParams.update(
        {
            "font.size": 8.9,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.35,
            "xtick.labelsize": 7.45,
            "ytick.labelsize": 7.45,
            "legend.fontsize": 5.9,
            "axes.linewidth": 0.70,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
        }
    )


def summarize_bridge(bridge: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    metrics = [
        "ad_overcompletion_score_mean",
        "hc_evidence_loss_score_mean",
        "overall_contamination",
        "ExternalCohort_F1",
    ]
    means = bridge.groupby("method", observed=True)[metrics].mean().add_suffix("_mean")
    sds = bridge.groupby("method", observed=True)[metrics].std().add_suffix("_sd")
    out = means.join(sds).reindex(methods).reset_index()
    out["overall_contamination_mean"] *= 100.0
    out["overall_contamination_sd"] *= 100.0
    out["method_display"] = out["method"].map(display_name)
    return out


def panel_label(ax: plt.Axes, label: str, x: float = -0.14, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.0,
        fontweight="bold",
        color=base.INK,
        clip_on=False,
    )


def broken_x(value: float, left_max: float, right_min: float, right_start: float) -> float:
    if pd.isna(value):
        return np.nan
    if value <= left_max:
        return float(value)
    return float(right_start + (value - right_min))


def clipped_xerr_bounds(
    x: float,
    err: float,
    left_max: float,
    right_min: float,
    right_start: float,
) -> tuple[float, float, float]:
    err = finite_sd(err)
    x = float(x)
    mapped = broken_x(x, left_max, right_min, right_start)
    if x <= left_max:
        lo = x - err
        hi = min(x + err, left_max)
    elif x >= right_min:
        lo = max(x - err, right_min)
        hi = x + err
    else:
        # Values inside the hidden gap should not occur. Collapse them rather
        # than drawing a misleading line across the discontinuity.
        lo = x
        hi = x
    return (
        broken_x(lo, left_max, right_min, right_start),
        mapped,
        broken_x(hi, left_max, right_min, right_start),
    )


def draw_broken_x_marks(ax: plt.Axes, left_max: float, right_start: float) -> None:
    trans = ax.get_xaxis_transform()
    gap = right_start - left_max
    x0, x1 = ax.get_xlim()
    half_width = 0.0105 * max(x1 - x0, 1e-6)
    for xpos in [left_max + gap * 0.38, left_max + gap * 0.62]:
        ax.plot(
            [xpos - half_width, xpos + half_width],
            [-0.025, 0.030],
            transform=trans,
            color=base.INK,
            linewidth=0.55,
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
    draw_broken_x_marks(ax, left_max, right_start)


def draw_xy_errorbar_broken_x(
    ax: plt.Axes,
    x: float,
    y: float,
    xerr: float,
    yerr: float,
    color: str,
    *,
    marker: str = "o",
    markersize: float = 4.9,
    break_cfg: tuple[float, float, float] | None = None,
    x_cap_height: float = 0.13,
    y_cap_half_width: float = 0.16,
    elinewidth: float = 0.75,
    cap_linewidth: float = 0.70,
    error_alpha: float = 1.0,
    draw_marker: bool = True,
) -> None:
    xerr = finite_sd(xerr)
    yerr = finite_sd(yerr)
    if break_cfg is None:
        mapped_x = x
        lo = x - xerr
        hi = x + xerr
    else:
        lo, mapped_x, hi = clipped_xerr_bounds(x, xerr, *break_cfg)
    ax.hlines(y, lo, hi, color=color, linewidth=elinewidth, alpha=error_alpha, zorder=2)
    ax.vlines([lo, hi], y - x_cap_height, y + x_cap_height, color=color, linewidth=cap_linewidth, alpha=error_alpha, zorder=2)
    ax.vlines(mapped_x, y - yerr, y + yerr, color=color, linewidth=elinewidth, alpha=error_alpha, zorder=2)
    ax.hlines(
        [y - yerr, y + yerr],
        mapped_x - y_cap_half_width,
        mapped_x + y_cap_half_width,
        color=color,
        linewidth=cap_linewidth,
        alpha=error_alpha,
        zorder=2,
    )
    if draw_marker:
        ax.scatter(
            mapped_x,
            y,
            marker=marker,
            s=markersize**2,
            color=color,
            edgecolor=base.INK,
            linewidth=0.42,
            zorder=3,
        )


def soft_draw_reference_points(ax: plt.Axes, pca_reference: pd.DataFrame) -> None:
    before_collections = len(ax.collections)
    before_lines = len(ax.lines)
    atlas.draw_reference_points(ax, pca_reference)
    for collection in ax.collections[before_collections:]:
        collection.set_alpha(0.18)
        try:
            collection.set_sizes(np.full_like(collection.get_sizes(), 3.0))
        except Exception:
            pass
    for line in ax.lines[before_lines:]:
        line.set_alpha(0.28)
        line.set_markersize(3.0)


def auto_limits(values: list[float], pad_frac: float = 0.07, floor: float | None = None) -> tuple[float, float]:
    arr = np.asarray([v for v in values if not pd.isna(v)], dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot compute limits from an empty value list.")
    lo = float(arr.min())
    hi = float(arr.max())
    span = max(hi - lo, 1e-6)
    pad = span * pad_frac
    lo -= pad
    hi += pad
    if floor is not None:
        lo = max(floor, lo)
    return lo, hi


def failure_axis_limits(
    raw_frames: list[pd.DataFrame], summaries: list[pd.DataFrame]
) -> tuple[tuple[float, float], tuple[float, float]]:
    xs: list[float] = []
    ys: list[float] = []
    for frame in raw_frames:
        xs.extend(frame["ad_overcompletion_score_mean"].dropna().astype(float).tolist())
        ys.extend(frame["hc_evidence_loss_score_mean"].dropna().astype(float).tolist())
    for summary in summaries:
        for _, row in summary.iterrows():
            x = row["ad_overcompletion_score_mean_mean"]
            y = row["hc_evidence_loss_score_mean_mean"]
            xsd = finite_sd(row["ad_overcompletion_score_mean_sd"])
            ysd = finite_sd(row["hc_evidence_loss_score_mean_sd"])
            xs.extend([x - xsd, x + xsd])
            ys.extend([y - ysd, y + ysd])
    return auto_limits(xs, floor=0.0), auto_limits(ys, floor=0.0)


def residual_xlim_from_summaries(specs: list[tuple[pd.DataFrame, str, str]]) -> tuple[float, float]:
    values: list[float] = []
    for frame, mean_col, sd_col in specs:
        for _, row in frame.iterrows():
            mean = row[mean_col]
            sd = finite_sd(row[sd_col])
            values.extend([mean - sd, mean + sd])
    lo, hi = auto_limits(values, pad_frac=0.10)
    return min(-0.20, lo), max(0.85, hi)


def draw_failure_axes(
    ax: plt.Axes,
    bridge: pd.DataFrame,
    summary: pd.DataFrame,
    methods: list[str],
    family: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    break_cfg: tuple[float, float, float] | None = None,
    left_ticks: list[float] | None = None,
    right_ticks: list[float] | None = None,
    xticks: list[float] | None = None,
) -> None:
    for method in methods:
        part = bridge[bridge["method"].eq(method)]
        color = method_color(method, family)
        raw_x = part["ad_overcompletion_score_mean"].astype(float)
        x_values = raw_x.map(lambda value: broken_x(value, *break_cfg) if break_cfg else value)
        ax.scatter(
            x_values,
            part["hc_evidence_loss_score_mean"],
            s=13,
            color=color,
            alpha=0.28,
            linewidth=0,
            zorder=1,
        )
        row = summary[summary["method"].eq(method)].iloc[0]
        draw_xy_errorbar_broken_x(
            ax,
            float(row["ad_overcompletion_score_mean_mean"]),
            float(row["hc_evidence_loss_score_mean_mean"]),
            finite_sd(row["ad_overcompletion_score_mean_sd"]),
            finite_sd(row["hc_evidence_loss_score_mean_sd"]),
            color,
            markersize=6.1,
            break_cfg=break_cfg,
            x_cap_height=0.055,
            y_cap_half_width=0.045,
            elinewidth=0.80,
            cap_linewidth=0.70,
        )

    if break_cfg is not None:
        if left_ticks is None or right_ticks is None:
            raise ValueError("left_ticks and right_ticks are required when break_cfg is set.")
        set_broken_xaxis(ax, left_ticks, right_ticks, *break_cfg, xlim=xlim)
    else:
        ax.set_xlim(*xlim)
        if xticks is not None:
            ax.set_xticks(xticks)
    ax.set_ylim(*ylim)
    ax.set_xlabel("AD-source upper-band deviation", labelpad=1.2)
    ax.set_ylabel("HC-source lower-band deviation", labelpad=1.8)
    ax.set_title("Diagnosis-directional Y–Z deviation axes", loc="left", fontweight="bold", pad=4.0)
    base.clean_axis(ax, "both")


def draw_residual_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    methods: list[str],
    family: str,
    method_col: str,
    mean_col: str,
    sd_col: str,
    xlim: tuple[float, float],
    xticks: list[float] | None = None,
    *,
    mean_focus: bool = False,
) -> None:
    indexed = summary.set_index(method_col).reindex(methods)
    y = np.arange(len(indexed))
    for yi, (method, row) in enumerate(indexed.iterrows()):
        color = method_color(str(method), family)
        mean = float(row[mean_col])
        sd = finite_sd(row[sd_col])
        if mean_focus:
            # Mean-only display: the x-axis is reserved for comparing method
            # means. SD is reported as a compact side annotation so large
            # dispersion does not force a wide axis or visually swamp the means.
            ax.scatter(
                mean,
                yi,
                s=6.8**2,
                color=color,
                edgecolor=base.INK,
                linewidth=0.62,
                zorder=4,
            )
            ax.text(
                1.025,
                yi,
                f"±{sd:.2f}",
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=5.9,
                color=base.MUTED if hasattr(base, "MUTED") else "#687078",
                clip_on=False,
            )
        else:
            ax.errorbar(
                mean,
                yi,
                xerr=sd,
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=0.75,
                capsize=1.8,
                markersize=4.9,
                markeredgecolor=base.INK,
                markeredgewidth=0.42,
                zorder=3,
            )
    ax.set_yticks(y)
    ax.set_yticklabels([display_name(str(method)) for method in indexed.index])
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if mean_focus:
        ax.text(
            1.025,
            1.025,
            "SD",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=5.9,
            fontweight="bold",
            color=base.INK,
            clip_on=False,
        )
        ax.set_xlabel("Residual similarity (mean)")
    else:
        ax.set_xlabel("Residual similarity (mean ± SD)")
    ax.set_title("Candidate-level residual similarity", loc="left", fontweight="bold", pad=4.0)
    base.clean_axis(ax, "x")

def pca_limits(
    baseline_aug: pd.DataFrame,
    ablation_aug: pd.DataFrame,
    baseline_reference: pd.DataFrame,
    ablation_reference: pd.DataFrame,
) -> tuple[tuple[float, float], tuple[float, float]]:
    combined_aug = pd.concat(
        [
            baseline_aug[["dim1", "dim2"]],
            ablation_aug[["dim1", "dim2"]],
        ],
        ignore_index=True,
    )
    combined_reference = pd.concat(
        [
            baseline_reference[["dim1", "dim2"]],
            ablation_reference[["dim1", "dim2"]],
        ],
        ignore_index=True,
    )
    return atlas.projection_limits(combined_aug, combined_reference)


def draw_pca_atlas(
    fig: plt.Figure,
    slot,
    pca_aug: pd.DataFrame,
    pca_reference: pd.DataFrame,
    methods: list[str],
    family: str,
    panel_letter_text: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    grid = slot.subgridspec(
        4,
        len(methods),
        height_ratios=[0.16, 1.0, 1.0, 0.12],
        wspace=0.055,
        hspace=0.18,
    )
    title_ax = fig.add_subplot(grid[0, :])
    title_ax.set_axis_off()
    title_ax.text(
        0.0,
        0.08,
        "PCA atlas of class-specific kNN mixing",
        ha="left",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color=base.INK,
        transform=title_ax.transAxes,
    )
    title_ax.text(
        -0.050,
        0.08,
        panel_letter_text,
        ha="left",
        va="bottom",
        fontsize=11.0,
        fontweight="bold",
        color=base.INK,
        transform=title_ax.transAxes,
        clip_on=False,
    )
    scatter = None
    for row_idx, source_label in enumerate([0, 1]):
        for col_idx, method in enumerate(methods):
            ax = fig.add_subplot(grid[row_idx + 1, col_idx])
            # Use the original atlas reference-point styling here. The PCA
            # panels encode mixing via color, so softening the reference
            # points makes the color grammar look different from the standalone
            # atlas.
            atlas.draw_reference_points(ax, pca_reference)
            panel = pca_aug[
                pca_aug["method"].eq(method)
                & pca_aug["source_label"].astype(int).eq(source_label)
            ]
            if panel.empty:
                raise ValueError(
                    f"No PCA points for family={family}, method={method}, source_label={source_label}"
                )
            scatter = ax.scatter(
                panel["dim1"],
                panel["dim2"],
                c=panel["knn_opposite_rate"],
                cmap=atlas.CONTAMINATION_CMAP,
                norm=atlas.CONTAMINATION_NORM,
                s=7.0,
                edgecolors="none",
                alpha=0.70,
                zorder=3,
            )
            ax.text(
                0.96,
                0.95,
                f"{100 * panel['knn_opposite_rate'].mean():.1f}%",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=6.75,
                fontweight="bold",
                color=base.INK,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.76,
                    "pad": 0.55,
                },
                zorder=5,
            )
            atlas.style_projection_axis(
                ax,
                xlim,
                ylim,
                show_x=row_idx == 1 and col_idx == 0,
                show_y=col_idx == 0,
                x_label="PC1",
                y_label="PC2",
            )
            if col_idx != 0:
                ax.set_xticklabels([])
                ax.tick_params(axis="x", length=0)
            if col_idx == 0:
                ax.text(
                    0.035,
                    0.95,
                    "HC source" if source_label == 0 else "AD source",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=6.85,
                    fontweight="bold",
                    color=base.INK,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.74, "pad": 0.45},
                )
            if row_idx == 0:
                ax.set_title(
                    pca_title_name(method),
                    fontsize=7.0,
                    fontweight="bold",
                    color=pca_method_color(method, family),
                    pad=1.6,
                )

    if scatter is None:
        raise RuntimeError(f"No PCA points were drawn for {family}.")

    start = max(1, len(methods) // 2 - 1)
    end = min(len(methods), start + 3)
    color_ax = fig.add_subplot(grid[3, start:end])
    colorbar = fig.colorbar(scatter, cax=color_ax, orientation="horizontal")
    colorbar.set_ticks(np.linspace(0, 1, 6))
    colorbar.set_ticklabels(["0", "20", "40", "60", "80", "100"])
    colorbar.ax.tick_params(labelsize=6.05, length=1.8, pad=0.9)
    colorbar.outline.set_linewidth(0.45)
    colorbar.set_label("Opposite-class kNN mixing (%)", fontsize=6.45, labelpad=1.6)


def draw_knn_f1_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    methods: list[str],
    family: str,
    x_col: str,
    x_sd_col: str,
    y_col: str,
    y_sd_col: str,
    x_scale: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    break_cfg: tuple[float, float, float] | None = None,
    left_ticks: list[float] | None = None,
    right_ticks: list[float] | None = None,
    xticks: list[float] | None = None,
    mean_focus: bool = False,
    mean_focus_error_alpha: float | None = None,
    mean_focus_elinewidth: float = 0.55,
    mean_focus_cap_linewidth: float = 0.48,
) -> None:
    indexed = summary.set_index("method").reindex(methods)
    for method, row in indexed.iterrows():
        color = method_color(str(method), family)
        mean_x = float(row[x_col]) * x_scale
        mean_y = float(row[y_col])
        if mean_focus:
            # The axis range is fixed by the means, but h still shows faint SD
            # whiskers so uncertainty is visible without allowing it to set the
            # scale of the panel. Whiskers extending outside xlim/ylim are
            # clipped by the axes.
            if mean_focus_error_alpha is not None and mean_focus_error_alpha > 0:
                draw_xy_errorbar_broken_x(
                    ax,
                    mean_x,
                    mean_y,
                    finite_sd(row[x_sd_col]) * x_scale,
                    finite_sd(row[y_sd_col]),
                    color,
                    break_cfg=break_cfg,
                    elinewidth=mean_focus_elinewidth,
                    cap_linewidth=mean_focus_cap_linewidth,
                    error_alpha=mean_focus_error_alpha,
                    draw_marker=False,
                )
            ax.scatter(
                mean_x,
                mean_y,
                s=6.5**2,
                color=color,
                edgecolor=base.INK,
                linewidth=0.55,
                zorder=4,
            )
        else:
            draw_xy_errorbar_broken_x(
                ax,
                mean_x,
                mean_y,
                finite_sd(row[x_sd_col]) * x_scale,
                finite_sd(row[y_sd_col]),
                color,
                break_cfg=break_cfg,
            )
    if break_cfg is not None:
        if left_ticks is None or right_ticks is None:
            raise ValueError("left_ticks and right_ticks are required when break_cfg is set.")
        set_broken_xaxis(ax, left_ticks, right_ticks, *break_cfg, xlim=xlim)
    else:
        ax.set_xlim(*xlim)
        if xticks is not None:
            ax.set_xticks(xticks)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Opposite-class kNN mixing (%)")
    ax.set_ylabel("External-cohort average F1 (%)")
    ax.set_title(
        "kNN mixing and held-out cross-cohort F1",
        loc="left",
        fontweight="bold",
        pad=4.0,
    )
    base.clean_axis(ax, "both")


def method_legend_handles(methods: list[str], family: str) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=method_color(method, family),
            markeredgecolor=base.INK,
            markeredgewidth=0.4,
            markersize=4.9,
            label=display_name(method),
        )
        for method in methods
    ]



def combined_method_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=method_color(method, family),
            markeredgecolor=base.INK,
            markeredgewidth=0.48,
            markersize=5.4,
            label=display_name(method),
        )
        for method, family in COMBINED_LEGEND_METHODS
    ]


def combined_reference_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=atlas.HC_REFERENCE,
            markeredgewidth=0.70,
            markersize=5.1,
            label="HC training reference",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="none",
            color=atlas.AD_REFERENCE,
            markeredgewidth=0.78,
            markersize=5.1,
            label="AD training reference",
        ),
    ]


def panel_label_figcoords(
    fig: plt.Figure,
    ax: plt.Axes,
    label: str,
    *,
    dx: float = -0.022,
    dy: float = 0.009,
) -> None:
    bbox = ax.get_position(fig)
    fig.text(
        bbox.x0 + dx,
        bbox.y1 + dy,
        label,
        ha="left",
        va="bottom",
        fontsize=11.0,
        fontweight="bold",
        color=base.INK,
    )


def build_figure() -> None:
    setup_refined_style()
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    bridge = pd.read_csv(EXTERNAL_BRIDGE)
    baseline_bridge = bridge[bridge["method"].isin(BASELINE_REP_METHODS)].copy()
    baseline_failure_bridge = bridge[bridge["method"].isin(BASELINE_FAILURE_METHODS)].copy()
    ablation_bridge = bridge[bridge["method"].isin(ABLATION_RAW_METHODS)].copy()

    assert_methods_present(baseline_bridge, BASELINE_REP_METHODS, "baseline representation panels")
    assert_methods_present(baseline_failure_bridge, BASELINE_FAILURE_METHODS, "baseline failure panel")
    assert_methods_present(ablation_bridge, ABLATION_RAW_METHODS, "ablation panels")

    baseline_failure_summary = summarize_bridge(baseline_failure_bridge, BASELINE_FAILURE_METHODS)
    ablation_failure_summary = summarize_bridge(ablation_bridge, ABLATION_RAW_METHODS)
    baseline_utility_summary = (
        baseline_bridge.groupby("method", observed=True)
        .agg(
            contamination_mean=("overall_contamination", "mean"),
            contamination_sd=("overall_contamination", "std"),
            f1_mean=("ExternalCohort_F1", "mean"),
            f1_sd=("ExternalCohort_F1", "std"),
        )
        .reindex(BASELINE_REP_METHODS)
        .reset_index()
    )
    baseline_utility_summary["contamination_mean"] *= 100.0
    baseline_utility_summary["contamination_sd"] *= 100.0
    baseline_utility_summary["method_display"] = baseline_utility_summary["method"]
    ablation_utility_summary = pd.read_csv(ABLATION_UTILITY_CSV)
    external_ablation_f1 = (
        ablation_bridge.groupby("method", observed=True)["ExternalCohort_F1"]
        .agg(f1_mean="mean", f1_sd="std")
        .reset_index()
        .rename(columns={"method": "method_raw"})
    )
    ablation_utility_summary = (
        ablation_utility_summary.drop(columns=["f1_mean", "f1_sd"])
        .merge(external_ablation_f1, on="method_raw", how="left", validate="one_to_one")
    )
    if ablation_utility_summary[["f1_mean", "f1_sd"]].isna().any().any():
        raise RuntimeError("Missing external F1 for ablation utility summary")
    ablation_knn_summary = ablation_utility_summary.copy()
    ablation_knn_summary["method"] = ablation_knn_summary["method_raw"]

    aug_long = pd.read_csv(AUGMENTATION_AUDIT_CSV)
    baseline_x_candidates, baseline_x_summary = v3.residual_similarity_summary(aug_long)
    baseline_x_candidates = baseline_x_candidates[
        baseline_x_candidates["method"].isin(BASELINE_RESIDUAL_METHODS)
    ].copy()
    baseline_x_summary = baseline_x_summary[
        baseline_x_summary["method"].isin(BASELINE_RESIDUAL_METHODS)
    ].copy()
    ablation_x_summary = ablation_utility_summary[
        [
            "method_raw",
            "method",
            "residual_similarity_mean",
            "residual_similarity_sd",
        ]
    ].copy()

    baseline_pca_aug = pd.read_csv(BASELINE_PCA_CSV)
    baseline_pca_reference = pd.read_csv(BASELINE_PCA_REFERENCE_CSV)
    ablation_pca_aug = pd.read_csv(ABLATION_PCA_CSV)
    ablation_pca_reference = pd.read_csv(ABLATION_PCA_REFERENCE_CSV)
    xlim, ylim = pca_limits(
        baseline_pca_aug,
        ablation_pca_aug,
        baseline_pca_reference,
        ablation_pca_reference,
    )

    failure_xlim, failure_ylim = failure_axis_limits(
        [baseline_failure_bridge, ablation_bridge],
        [baseline_failure_summary, ablation_failure_summary],
    )
    residual_xlim = residual_xlim_from_summaries(
        [
            (baseline_x_summary, "mean", "sd"),
            (ablation_x_summary, "residual_similarity_mean", "residual_similarity_sd"),
        ]
    )

    f1_min = min(
        float((baseline_utility_summary["f1_mean"] - baseline_utility_summary["f1_sd"]).min()),
        float((ablation_utility_summary["f1_mean"] - ablation_utility_summary["f1_sd"]).min()),
    )
    f1_max = max(
        float((baseline_utility_summary["f1_mean"] + baseline_utility_summary["f1_sd"]).max()),
        float((ablation_utility_summary["f1_mean"] + ablation_utility_summary["f1_sd"]).max()),
    )
    f1_ylim = (np.floor(f1_min - 0.8), np.ceil(f1_max + 0.6))
    ablation_f1_ylim = (80.0, f1_ylim[1])

    # Keep both method families visible at the same scale. The source canvas is
    # wide, but the manuscript includes it at text width, matching the reviewed
    # composite layout rather than stacking the two families vertically.
    fig = plt.figure(figsize=(373.8 / 25.4, 214.4 / 25.4), facecolor="white")
    outer = fig.add_gridspec(
        1,
        2,
        left=0.058,
        right=0.990,
        top=0.840,
        bottom=0.075,
        wspace=0.095,
    )
    left = outer[0, 0].subgridspec(
        3,
        1,
        height_ratios=[0.66, 1.28, 0.66],
        hspace=0.48,
    )
    right = outer[0, 1].subgridspec(
        3,
        1,
        height_ratios=[0.66, 1.28, 0.66],
        hspace=0.48,
    )

    left_top = left[0, 0].subgridspec(1, 2, width_ratios=[1.55, 1.10], wspace=0.30)
    right_top = right[0, 0].subgridspec(1, 2, width_ratios=[1.55, 1.10], wspace=0.30)
    ax_a = fig.add_subplot(left_top[0, 0])
    ax_b = fig.add_subplot(left_top[0, 1])
    ax_e = fig.add_subplot(right_top[0, 0])
    ax_f = fig.add_subplot(right_top[0, 1])

    draw_failure_axes(
        ax_a,
        baseline_failure_bridge,
        baseline_failure_summary,
        BASELINE_FAILURE_METHODS,
        "baseline",
        failure_xlim,
        failure_ylim,
    )
    draw_residual_panel(
        ax_b,
        baseline_x_summary,
        BASELINE_RESIDUAL_METHODS,
        "baseline",
        "method",
        "mean",
        "sd",
        residual_xlim,
    )
    draw_pca_atlas(
        fig,
        left[1, 0],
        baseline_pca_aug,
        baseline_pca_reference,
        BASELINE_PCA_METHODS_MAIN,
        "baseline",
        "c",
        xlim,
        ylim,
    )
    ax_d = fig.add_subplot(left[2, 0])
    draw_knn_f1_panel(
        ax_d,
        baseline_utility_summary,
        BASELINE_REP_METHODS,
        "baseline",
        "contamination_mean",
        "contamination_sd",
        "f1_mean",
        "f1_sd",
        1.0,
        KNN_BASELINE_BROKEN_XLIM,
        f1_ylim,
        break_cfg=KNN_BASELINE_BREAK_CFG,
        left_ticks=KNN_BASELINE_LEFT_TICKS,
        right_ticks=KNN_BASELINE_RIGHT_TICKS,
    )

    draw_failure_axes(
        ax_e,
        ablation_bridge,
        ablation_failure_summary,
        ABLATION_RAW_METHODS,
        "ablation",
        ABLATION_FAILURE_BROKEN_XLIM,
        failure_ylim,
        break_cfg=ABLATION_FAILURE_BREAK_CFG,
        left_ticks=ABLATION_FAILURE_LEFT_TICKS,
        right_ticks=ABLATION_FAILURE_RIGHT_TICKS,
    )
    draw_residual_panel(
        ax_f,
        ablation_x_summary,
        ABLATION_RAW_METHODS,
        "ablation",
        "method_raw",
        "residual_similarity_mean",
        "residual_similarity_sd",
        (0.10, 0.22),
        xticks=[0.10, 0.15, 0.20],
        mean_focus=True,
    )
    draw_pca_atlas(
        fig,
        right[1, 0],
        ablation_pca_aug,
        ablation_pca_reference,
        ABLATION_PCA_METHODS_MAIN,
        "ablation",
        "g",
        xlim,
        ylim,
    )
    ax_h = fig.add_subplot(right[2, 0])
    draw_knn_f1_panel(
        ax_h,
        ablation_knn_summary,
        ABLATION_RAW_METHODS,
        "ablation",
        "overall_contamination_mean",
        "overall_contamination_sd",
        "f1_mean",
        "f1_sd",
        100.0,
        KNN_ABLATION_XLIM,
        ablation_f1_ylim,
        xticks=KNN_ABLATION_TICKS,
        mean_focus_error_alpha=KNN_ABLATION_ERROR_ALPHA,
        mean_focus_elinewidth=KNN_ABLATION_ERROR_LINEWIDTH,
        mean_focus_cap_linewidth=KNN_ABLATION_ERROR_CAP_LINEWIDTH,
    )

    for label, ax in zip("abef", [ax_a, ax_b, ax_e, ax_f]):
        panel_label(
            ax,
            label,
            x=-0.20 if label in {"a", "e"} else -0.18,
            y=1.06,
        )
    panel_label_figcoords(fig, ax_d, "d", dx=0.000, dy=0.022)
    panel_label_figcoords(fig, ax_h, "h", dx=0.000, dy=0.022)

    left_bbox = outer[0, 0].get_position(fig)
    right_bbox = outer[0, 1].get_position(fig)
    fig.text(
        left_bbox.x0,
        left_bbox.y1 + 0.016,
        "Baselines",
        ha="left",
        va="bottom",
        fontsize=10.8,
        fontweight="bold",
        color=base.INK,
    )
    fig.text(
        right_bbox.x0,
        right_bbox.y1 + 0.016,
        "Ablations",
        ha="left",
        va="bottom",
        fontsize=10.8,
        fontweight="bold",
        color=base.INK,
    )

    fig.legend(
        handles=combined_method_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=6,
        fontsize=6.85,
        columnspacing=0.95,
        handletextpad=0.30,
        borderaxespad=0.0,
        frameon=False,
    )
    fig.legend(
        handles=combined_reference_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.927),
        ncol=2,
        fontsize=6.70,
        columnspacing=1.05,
        handletextpad=0.34,
        borderaxespad=0.0,
        frameon=False,
    )

    baseline_failure_summary.to_csv(SOURCE_DIR / "panel_a_baseline_failure_summary_no_icl_direct.csv", index=False)
    baseline_x_candidates.to_csv(SOURCE_DIR / "panel_b_baseline_residual_similarity_candidate_source.csv", index=False)
    baseline_x_summary.to_csv(SOURCE_DIR / "panel_b_baseline_residual_similarity_summary.csv", index=False)
    baseline_pca_aug.to_csv(SOURCE_DIR / "panel_c_baseline_pca_contamination_source.csv", index=False)
    baseline_pca_aug[baseline_pca_aug["method"].isin(BASELINE_PCA_METHODS_MAIN)].to_csv(
        SOURCE_DIR / "panel_c_baseline_pca_displayed_methods_source.csv", index=False
    )
    baseline_pca_reference.to_csv(SOURCE_DIR / "panel_c_baseline_pca_training_reference_source.csv", index=False)
    baseline_utility_summary.to_csv(SOURCE_DIR / "panel_d_baseline_knn_f1_summary.csv", index=False)
    ablation_failure_summary.to_csv(SOURCE_DIR / "panel_e_ablation_failure_summary.csv", index=False)
    ablation_x_summary.to_csv(SOURCE_DIR / "panel_f_ablation_residual_similarity_summary.csv", index=False)
    ablation_pca_aug.to_csv(SOURCE_DIR / "panel_g_ablation_pca_contamination_source.csv", index=False)
    ablation_pca_aug[ablation_pca_aug["method"].isin(ABLATION_PCA_METHODS_MAIN)].to_csv(
        SOURCE_DIR / "panel_g_ablation_pca_displayed_methods_source.csv", index=False
    )
    ablation_pca_reference.to_csv(SOURCE_DIR / "panel_g_ablation_pca_training_reference_source.csv", index=False)
    ablation_utility_summary.to_csv(SOURCE_DIR / "panel_h_ablation_knn_f1_summary.csv", index=False)

    save_bundle(fig, "Figure7_baseline_ablation_joint_external_20260720")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Figure 7 from locked utility, residual, and PCA sources."
    )
    parser.add_argument("--utility-bridge-csv", required=True, type=Path)
    parser.add_argument("--augmentation-audit-csv", required=True, type=Path)
    parser.add_argument("--ablation-utility-csv", required=True, type=Path)
    parser.add_argument("--baseline-pca-csv", required=True, type=Path)
    parser.add_argument("--baseline-pca-reference-csv", required=True, type=Path)
    parser.add_argument("--ablation-pca-csv", required=True, type=Path)
    parser.add_argument("--ablation-pca-reference-csv", required=True, type=Path)
    parser.add_argument("--figure-dir", required=True, type=Path)
    parser.add_argument("--source-data-dir", required=True, type=Path)
    return parser.parse_args()


def configure(args: argparse.Namespace) -> None:
    global EXTERNAL_BRIDGE, AUGMENTATION_AUDIT_CSV, ABLATION_UTILITY_CSV
    global BASELINE_PCA_CSV, BASELINE_PCA_REFERENCE_CSV
    global ABLATION_PCA_CSV, ABLATION_PCA_REFERENCE_CSV, FIG_DIR, SOURCE_DIR
    EXTERNAL_BRIDGE = args.utility_bridge_csv
    AUGMENTATION_AUDIT_CSV = args.augmentation_audit_csv
    ABLATION_UTILITY_CSV = args.ablation_utility_csv
    BASELINE_PCA_CSV = args.baseline_pca_csv
    BASELINE_PCA_REFERENCE_CSV = args.baseline_pca_reference_csv
    ABLATION_PCA_CSV = args.ablation_pca_csv
    ABLATION_PCA_REFERENCE_CSV = args.ablation_pca_reference_csv
    FIG_DIR = args.figure_dir
    SOURCE_DIR = args.source_data_dir


def main() -> None:
    configure(parse_args())
    build_figure()
    print(f"Wrote joint Figure 7 candidate to {FIG_DIR}")
    print(f"Wrote source data to {SOURCE_DIR}")


if __name__ == "__main__":
    main()
