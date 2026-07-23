from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


ARTICLE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ARTICLE_DIR.parents[1]
FIG_DIR = ARTICLE_DIR / "figures"
SOURCE_DIR = ARTICLE_DIR / "source_data"
BUCKET_ROOT = PROJECT_ROOT / "analysis_outputs_file_level_quality_plus_baselines_0624"
TABLE_SRC = BUCKET_ROOT / "tables"
LANDSCAPE_SRC = BUCKET_ROOT / "quality_landscape_figures_v4" / "data"
STAT_SRC = PROJECT_ROOT / "11_manuscript_lock_20260620" / "01_statistics"

INK = "#202124"
MUTED = "#667079"
GRID = "#E4E8EB"
BLUE = "#1F5B83"
GOLD = "#C9A23F"

METHOD_COLORS = {
    "CoDiPO": BLUE,
    "w/o X": "#C67B6C",
    "w/o YZ": "#6F9B76",
    "Cosine as X": "#737373",
    "Cosine-only Preference": GOLD,
    "Vanilla Generation w/o DPO": "#8D96A0",
    "XYZ Hard Filtering w/o DPO": "#2F8C8C",
    "EDA": "#D8A7B1",
    "ICL Rewrite": "#A7C7D9",
    "ICL Imitate": "#B7D3B0",
    "ICL Direct": "#D9907A",
}

METHOD_LABELS = {
    "CoDiPO": "CoDiPO",
    "w/o X": "w/o X",
    "w/o YZ": "w/o Y/Z",
    "Cosine as X": "Cosine sim. as X",
    "Cosine-only Preference": "Cosine-only",
    "Vanilla Generation w/o DPO": "Vanilla",
    "XYZ Hard Filtering w/o DPO": "Hard filter",
    "EDA": "EDA",
    "ICL Rewrite": "ICL rewrite",
    "ICL Imitate": "ICL imitation",
    "ICL Direct": "ICL direct",
}

PREFERENCE_METHODS = [
    "CoDiPO",
    "w/o X",
    "w/o YZ",
    "Cosine as X",
    "Cosine-only Preference",
    "Vanilla Generation w/o DPO",
]
BASELINE_METHODS = ["CoDiPO", "EDA", "ICL Rewrite", "ICL Imitate", "ICL Direct"]
REFERENCE_METHODS = ["CoDiPO", "Vanilla Generation w/o DPO", "XYZ Hard Filtering w/o DPO"]
UTILITY_METHODS = [
    "EDA",
    "ICL Direct",
    "ICL Imitate",
    "ICL Rewrite",
    "Vanilla Generation w/o DPO",
    "w/o X",
    "w/o YZ",
    "Cosine as X",
    "Cosine-only Preference",
    "CoDiPO",
    "XYZ Hard Filtering w/o DPO",
]

UTILITY_TO_LOCKED_METHOD = {
    "CoDiPO": "CoDiPO",
    "w/o X": "CoDiPO w/o X",
    "w/o YZ": "CoDiPO w/o YZ",
    "Cosine as X": "w/o residual decomposition",
    "Cosine-only Preference": "Cosine-only preference",
    "Vanilla Generation w/o DPO": "w/o DPO, vanilla augmentation",
    "XYZ Hard Filtering w/o DPO": "w/o DPO, XYZ hard filtering",
    "EDA": "EDA",
    "ICL Rewrite": "ICL Rewrite",
    "ICL Imitate": "ICL Imitation",
    "ICL Direct": "ICL Direct",
}


def load_utility_source(quality: pd.DataFrame) -> pd.DataFrame:
    file_range = (
        quality.groupby("method", observed=True)["joint_pass_rate"]
        .agg(joint_pass_mean="mean", joint_pass_sd="std", joint_pass_min="min", joint_pass_max="max")
        .reset_index()
    )
    overall_f1 = pd.read_csv(STAT_SRC / "overallmacro_bootstrap_method_metrics.csv")
    overall_f1 = overall_f1[overall_f1["metric"].eq("f1")].copy()
    overall_f1 = overall_f1[
        ["method_display", "point_estimate", "ci_low", "ci_high", "n_bootstrap"]
    ].rename(
        columns={
            "method_display": "locked_method_display",
            "point_estimate": "overall_f1_mean_percent",
            "ci_low": "overall_f1_ci_low",
            "ci_high": "overall_f1_ci_high",
        }
    )
    for col in ["overall_f1_mean_percent", "overall_f1_ci_low", "overall_f1_ci_high"]:
        overall_f1[col] = overall_f1[col] * 100.0
    file_range["locked_method_display"] = file_range["method"].map(UTILITY_TO_LOCKED_METHOD)
    utility = file_range.merge(overall_f1, on="locked_method_display", how="left")
    utility["source_definition"] = (
        "0624 file-level proxy summary plus locked OverallMacro F1 from "
        "11_manuscript_lock_20260620/01_statistics/overallmacro_bootstrap_method_metrics.csv"
    )
    return utility


def register_arial() -> None:
    for candidate in font_manager.findSystemFonts():
        path = Path(candidate)
        if path.stem.lower() == "arial":
            font_manager.fontManager.addfont(str(path))
            break


def setup_style() -> None:
    register_arial()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.3,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "savefig.dpi": 600,
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.06) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="top", fontsize=9.5, fontweight="bold", color=INK, clip_on=False)


def add_quality_regions(ax: plt.Axes, x_boundary: float, y_boundary: float) -> None:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    regions = [
        (xmin, ymin, x_boundary - xmin, y_boundary - ymin, "#EAF3F7"),
        (x_boundary, ymin, xmax - x_boundary, y_boundary - ymin, "#F6F1E6"),
        (xmin, y_boundary, x_boundary - xmin, ymax - y_boundary, "#F7EDEC"),
        (x_boundary, y_boundary, xmax - x_boundary, ymax - y_boundary, "#F8F5F2"),
    ]
    for x, y, width, height, color in regions:
        ax.add_patch(Rectangle((x, y), width, height, facecolor=color, edgecolor="none", zorder=-5))
    ax.axvline(x_boundary, color="#9BA1A6", linestyle="--", linewidth=0.8, zorder=-1)
    ax.axhline(y_boundary, color="#9BA1A6", linestyle="--", linewidth=0.8, zorder=-1)


def draw_quality_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    methods: list[str],
    boundaries: pd.DataFrame,
    title: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    labels: dict[str, tuple[float, float, str]] | None = None,
) -> None:
    sub = data[data["method"].isin(methods)].copy()
    if xlim is None:
        xpad = max(0.006, (sub["residual_cos_mean"].max() - sub["residual_cos_mean"].min()) * 0.16)
        xlim = (sub["residual_cos_mean"].min() - xpad, sub["residual_cos_mean"].max() + xpad)
    if ylim is None:
        ypad = max(0.12, (sub["yz_violation_mean"].max() - sub["yz_violation_mean"].min()) * 0.16)
        ylim = (max(0, sub["yz_violation_mean"].min() - ypad), sub["yz_violation_mean"].max() + ypad)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    x_boundary = float(boundaries.loc[boundaries["boundary"].eq("residual_similarity_median"), "value"].iloc[0])
    y_boundary = float(boundaries.loc[boundaries["boundary"].eq("yz_violation_median"), "value"].iloc[0])
    add_quality_regions(ax, x_boundary, y_boundary)

    means = []
    for method in methods:
        frame = sub[sub["method"].eq(method)]
        if frame.empty:
            continue
        color = METHOD_COLORS[method]
        ax.scatter(frame["residual_cos_mean"], frame["yz_violation_mean"], s=24, color=color, alpha=0.42, edgecolor="white", linewidth=0.45)
        mean_x = frame["residual_cos_mean"].mean()
        mean_y = frame["yz_violation_mean"].mean()
        means.append({"method": method, "x": mean_x, "y": mean_y})
        ax.scatter(mean_x, mean_y, s=76, marker="D", color=color, edgecolor=INK, linewidth=0.65, zorder=5)

    means_df = pd.DataFrame(means).set_index("method")
    labels = labels or {}
    for method, (dx, dy, ha) in labels.items():
        if method not in means_df.index:
            continue
        row = means_df.loc[method]
        ax.text(float(row["x"]) + dx, float(row["y"]) + dy, METHOD_LABELS[method], fontsize=7.0, color=INK, ha=ha, va="center")

    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    ax.set_xlabel("Residual similarity to source")
    ax.set_ylabel("YZ violation score")
    ax.grid(axis="y", color=GRID, linewidth=0.45, zorder=-3)


def draw_utility_panel(ax: plt.Axes, utility: pd.DataFrame) -> None:
    data = utility[utility["method"].isin(UTILITY_METHODS)].copy()
    for _, row in data.iterrows():
        method = row["method"]
        color = METHOD_COLORS[method]
        x = row["joint_pass_mean"] * 100
        x_low = row["joint_pass_min"] * 100
        x_high = row["joint_pass_max"] * 100
        y = row["overall_f1_mean_percent"]
        y_low = row["overall_f1_ci_low"]
        y_high = row["overall_f1_ci_high"]
        alpha = 1.0 if method in {"CoDiPO", "XYZ Hard Filtering w/o DPO"} else 0.68
        size = 56 if method in {"CoDiPO", "XYZ Hard Filtering w/o DPO"} else 34
        ax.errorbar(
            x,
            y,
            xerr=np.array([[x - x_low], [x_high - x]]),
            yerr=np.array([[y - y_low], [y_high - y]]),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=0.85,
            capsize=2.2,
            markersize=np.sqrt(size),
            alpha=alpha,
            markeredgecolor="white",
            markeredgewidth=0.55,
            zorder=4 if method in {"CoDiPO", "XYZ Hard Filtering w/o DPO"} else 3,
        )

    codipo = data[data["method"].eq("CoDiPO")].iloc[0]
    hard = data[data["method"].eq("XYZ Hard Filtering w/o DPO")].iloc[0]
    ax.annotate(
        "CoDiPO",
        xy=(codipo["joint_pass_mean"] * 100, codipo["overall_f1_mean_percent"]),
        xytext=(codipo["joint_pass_mean"] * 100 + 0.82, codipo["overall_f1_mean_percent"] + 0.72),
        fontsize=7.0,
        color=INK,
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=0.35),
    )
    ax.annotate(
        "Hard filter",
        xy=(hard["joint_pass_mean"] * 100, hard["overall_f1_mean_percent"]),
        xytext=(min(29.0, hard["joint_pass_mean"] * 100 - 0.5), hard["overall_f1_mean_percent"] - 1.2),
        arrowprops=dict(arrowstyle="->", color="#7A8085", lw=0.75),
        fontsize=7.0,
        color=INK,
        ha="right",
        va="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=0.30),
    )
    ax.annotate(
        "preference\noptimization",
        xy=(codipo["joint_pass_mean"] * 100, codipo["overall_f1_mean_percent"]),
        xytext=(codipo["joint_pass_mean"] * 100 - 2.0, codipo["overall_f1_mean_percent"] + 1.15),
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.85),
        fontsize=7.0,
        color=BLUE,
        ha="center",
        va="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=0.25),
    )
    ax.set_title("Joint proxy feasibility and cohort-averaged F1", loc="left", fontweight="bold", pad=5)
    ax.set_xlabel("Joint proxy feasibility (%)")
    ax.set_ylabel("Cohort-averaged F1 (%)")
    ax.set_xlim(-1.2, 30)
    ymin = max(72.0, float(data["overall_f1_ci_low"].min()) - 1.0)
    ymax = min(91.5, float(data["overall_f1_ci_high"].max()) + 0.9)
    ax.set_ylim(ymin, ymax)
    ax.grid(color=GRID, linewidth=0.5, zorder=0)


def save_figure(fig: mpl.figure.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    try:
        fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    except TypeError:
        fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def draw_figure(
    quality: pd.DataFrame,
    boundaries: pd.DataFrame,
    utility: pd.DataFrame,
    stem: str,
    compact: bool,
) -> None:
    if compact:
        fig = plt.figure(figsize=(10.9, 3.12), constrained_layout=False)
        grid = fig.add_gridspec(
            1,
            5,
            width_ratios=[1.0, 1.0, 1.0, 1.32, 1.32],
            left=0.055,
            right=0.992,
            top=0.84,
            bottom=0.245,
            wspace=0.42,
        )
        ax_a = fig.add_subplot(grid[0, 0])
        ax_b = fig.add_subplot(grid[0, 1])
        ax_c = fig.add_subplot(grid[0, 2])
        ax_d = fig.add_subplot(grid[0, 3:])
    else:
        fig = plt.figure(figsize=(7.25, 6.35), constrained_layout=False)
        grid = fig.add_gridspec(
            2,
            3,
            height_ratios=[1.05, 0.92],
            left=0.075,
            right=0.99,
            top=0.955,
            bottom=0.185,
            wspace=0.34,
            hspace=0.48,
        )
        ax_a = fig.add_subplot(grid[0, 0])
        ax_b = fig.add_subplot(grid[0, 1])
        ax_c = fig.add_subplot(grid[0, 2])
        ax_d = fig.add_subplot(grid[1, :])

    draw_quality_panel(
        ax_a,
        quality,
        PREFERENCE_METHODS,
        boundaries,
        "Ablation methods",
    )
    draw_quality_panel(
        ax_b,
        quality,
        BASELINE_METHODS,
        boundaries,
        "Augmentation baselines",
    )
    draw_quality_panel(
        ax_c,
        quality,
        REFERENCE_METHODS,
        boundaries,
        "Proxy-selected reference",
    )
    draw_utility_panel(ax_d, utility)

    for label, ax in zip("abcd", [ax_a, ax_b, ax_c, ax_d]):
        if label == "c":
            panel_label(ax, label, x=-0.19, y=1.12)
        else:
            panel_label(ax, label)

    handles = [
        Line2D([0], [0], marker="D", linestyle="none", markerfacecolor=METHOD_COLORS[m], markeredgecolor=INK, markersize=5.6, label=METHOD_LABELS[m])
        for m in [
            "CoDiPO",
            "w/o X",
            "w/o YZ",
            "Cosine as X",
            "Cosine-only Preference",
            "Vanilla Generation w/o DPO",
            "XYZ Hard Filtering w/o DPO",
            "EDA",
            "ICL Direct",
            "ICL Rewrite",
            "ICL Imitate",
        ]
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6 if not compact else 11,
        bbox_to_anchor=(0.5, 0.012 if not compact else 0.035),
        frameon=False,
        columnspacing=0.75 if not compact else 0.44,
        handletextpad=0.28 if compact else 0.32,
        fontsize=6.5 if not compact else 6.1,
    )
    save_figure(fig, FIG_DIR / stem)


def main() -> None:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    quality = pd.read_csv(TABLE_SRC / "01_file_level_summary.csv")
    boundaries = pd.read_csv(LANDSCAPE_SRC / "quality_space_region_boundaries.csv")
    utility = load_utility_source(quality)
    quality.to_csv(SOURCE_DIR / "figure5_proxy_quality_space_source.csv", index=False, encoding="utf-8-sig")
    utility.to_csv(SOURCE_DIR / "figure5_proxy_utility_source.csv", index=False, encoding="utf-8-sig")

    draw_figure(quality, boundaries, utility, "fig_proxy_quality_utility_mechanism", compact=False)
    draw_figure(quality, boundaries, utility, "fig_proxy_quality_utility_mechanism_compact", compact=True)
    print("Wrote fig_proxy_quality_utility_mechanism.* and compact variants.")

