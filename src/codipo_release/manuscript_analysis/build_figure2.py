#!/usr/bin/env python
"""Build the proxy-only Y/Z distribution figure for the active manuscript.

The output intentionally keeps the historical stem ``fig3_yz_distributions`` so
the article TeX does not need a figure filename migration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch
import fitz
import numpy as np
import pandas as pd
from PIL import Image


YZ_PARTICIPANT_CSV: Path
YZ_SUMMARY_CSV: Path | None
FIG_DIR: Path

INK = "#293033"
MUTED = "#6F777B"
GRID = "#E6EBEE"
HC = "#4F7FAE"
AD = "#CB4D54"
JOINT = "#2F5364"

DATASET_ORDER = ["Train", "Test", "Pitt", "Lu"]
DATASET_LABELS = {
    "Train": "ADReSS\ntrain",
    "Test": "ADReSS\nvalidation",
    "Pitt": "Pitt",
    "Lu": "Lu",
}
DATASET_FULL_LABELS = {
    "Train": "ADReSS train",
    "Test": "ADReSS validation",
    "Pitt": "Pitt",
    "Lu": "Lu",
}
METRIC_LABELS = {
    "Y": "Y",
    "Z": "Z",
    "YZ_joint_index": "Joint Y-Z",
}
METRIC_MARKERS = {
    "Y": "o",
    "Z": "s",
    "YZ_joint_index": "D",
}
METRIC_COLORS = {
    "Y": "#2F6F9E",
    "Z": "#B7655C",
    "YZ_joint_index": JOINT,
}


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
            "font.size": 6.8,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 600,
        }
    )


def soften_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIG_DIR / f"{stem}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)

    # Rasterize the verified vector export. Direct Agg export of this dense
    # multi-axes layout can displace a few terminal tick labels on Windows.
    doc = fitz.open(pdf_path)
    pix = doc[0].get_pixmap(dpi=600, alpha=False)
    png_path = FIG_DIR / f"{stem}.png"
    pix.save(png_path)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    image.save(FIG_DIR / f"{stem}.tiff", compression="tiff_lzw", dpi=(600, 600))
    doc.close()


def load_yz_data() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    yz_data = pd.read_csv(YZ_PARTICIPANT_CSV)
    yz_summary = (
        pd.read_csv(YZ_SUMMARY_CSV)
        if YZ_SUMMARY_CSV is not None
        else None
    )
    return yz_data, yz_summary


def add_joint_index(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["Y_percentile_within_cohort"] = np.nan
    out["Z_percentile_within_cohort"] = np.nan
    for dataset, idx in out.groupby("dataset").groups.items():
        n = len(idx)
        out.loc[idx, "Y_percentile_within_cohort"] = out.loc[idx, "Y"].rank(method="average") / (n + 1)
        out.loc[idx, "Z_percentile_within_cohort"] = out.loc[idx, "Z"].rank(method="average") / (n + 1)
    out["YZ_joint_index"] = np.sqrt(out["Y_percentile_within_cohort"] * out["Z_percentile_within_cohort"])
    return out


def bootstrap_mean_diff(hc_vals: np.ndarray, ad_vals: np.ndarray, rng: np.random.Generator, n_boot: int = 10000) -> tuple[float, float, float]:
    if len(hc_vals) == 0 or len(ad_vals) == 0:
        return np.nan, np.nan, np.nan
    obs = float(np.nanmean(hc_vals) - np.nanmean(ad_vals))
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        h = rng.choice(hc_vals, size=len(hc_vals), replace=True)
        a = rng.choice(ad_vals, size=len(ad_vals), replace=True)
        boot[i] = np.nanmean(h) - np.nanmean(a)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return obs, float(lo), float(hi)


def bootstrap_standardized_diff(
    hc_vals: np.ndarray,
    ad_vals: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 10000,
) -> tuple[float, float, float]:
    def _std_diff(h: np.ndarray, a: np.ndarray) -> float:
        pooled = np.sqrt(((len(h) - 1) * np.nanvar(h, ddof=1) + (len(a) - 1) * np.nanvar(a, ddof=1)) / (len(h) + len(a) - 2))
        if not np.isfinite(pooled) or pooled == 0:
            return np.nan
        return float((np.nanmean(h) - np.nanmean(a)) / pooled)

    obs = _std_diff(hc_vals, ad_vals)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        h = rng.choice(hc_vals, size=len(hc_vals), replace=True)
        a = rng.choice(ad_vals, size=len(ad_vals), replace=True)
        boot[i] = _std_diff(h, a)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return obs, float(lo), float(hi)


def summarize_effects(data: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260625)
    rows = []
    for dataset in DATASET_ORDER:
        sub = data[data["dataset"].eq(dataset)]
        for metric in ["Y", "Z", "YZ_joint_index"]:
            hc_vals = sub.loc[sub["diagnosis_group"].eq("HC"), metric].dropna().to_numpy(dtype=float)
            ad_vals = sub.loc[sub["diagnosis_group"].eq("AD"), metric].dropna().to_numpy(dtype=float)
            mean_diff, mean_lo, mean_hi = bootstrap_mean_diff(hc_vals, ad_vals, rng)
            std_diff, std_lo, std_hi = bootstrap_standardized_diff(hc_vals, ad_vals, rng)
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_display": DATASET_FULL_LABELS[dataset],
                    "metric": metric,
                    "n_hc": len(hc_vals),
                    "n_ad": len(ad_vals),
                    "hc_mean": float(np.nanmean(hc_vals)) if len(hc_vals) else np.nan,
                    "ad_mean": float(np.nanmean(ad_vals)) if len(ad_vals) else np.nan,
                    "mean_difference_hc_minus_ad": mean_diff,
                    "mean_difference_ci_low": mean_lo,
                    "mean_difference_ci_high": mean_hi,
                    "standardized_difference_hc_minus_ad": std_diff,
                    "standardized_difference_ci_low": std_lo,
                    "standardized_difference_ci_high": std_hi,
                    "n_bootstrap": 10000,
                }
            )
    return pd.DataFrame(rows)


def cohort_counts(data: pd.DataFrame) -> dict[str, tuple[int, int]]:
    counts: dict[str, tuple[int, int]] = {}
    for dataset in DATASET_ORDER:
        sub = data[data["dataset"].eq(dataset)]
        counts[dataset] = (int((sub["diagnosis_group"] == "HC").sum()), int((sub["diagnosis_group"] == "AD").sum()))
    return counts


def draw_proxy_violin(ax: plt.Axes, data: pd.DataFrame, metric: str, title: str, ylabel: str) -> None:
    positions = np.arange(len(DATASET_ORDER), dtype=float)
    offsets = {"HC": -0.18, "AD": 0.18}
    colors = {"HC": HC, "AD": AD}
    counts = cohort_counts(data)

    for i, dataset in enumerate(DATASET_ORDER):
        for group in ["HC", "AD"]:
            vals = data.loc[(data["dataset"].eq(dataset)) & (data["diagnosis_group"].eq(group)), metric].dropna().to_numpy()
            if len(vals) == 0:
                continue
            pos = positions[i] + offsets[group]
            parts = ax.violinplot([vals], positions=[pos], widths=0.30, showmeans=False, showmedians=False, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(colors[group])
                body.set_edgecolor(colors[group])
                body.set_alpha(0.22)
                body.set_linewidth(0.9)
            q1, median, q3 = np.percentile(vals, [25, 50, 75])
            mean = float(np.mean(vals))
            ax.plot([pos, pos], [q1, q3], color=colors[group], linewidth=3.0, solid_capstyle="butt", zorder=4)
            ax.plot([pos - 0.10, pos + 0.10], [median, median], color="white", linewidth=1.3, zorder=5)
            ax.scatter(pos, mean, marker="D", s=24, color=colors[group], edgecolor="white", linewidth=0.75, zorder=6)

    labels = [f"{DATASET_LABELS[d]}\n{counts[d][0]}/{counts[d][1]}" for d in DATASET_ORDER]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_title(title, loc="left", fontweight="bold", pad=4)
    ax.set_ylabel(ylabel)
    if metric in {"Y", "YZ_joint_index"}:
        ax.set_ylim(-0.02, 1.04)
    elif metric == "Z":
        ax.set_ylim(0.002, 0.0345)
    soften_axes(ax)


def add_cov_ellipse(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str) -> None:
    if len(x) < 5:
        return
    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * np.sqrt(np.maximum(vals, 0))
    ell = Ellipse(
        (float(np.mean(x)), float(np.mean(y))),
        width=width,
        height=height,
        angle=angle,
        facecolor="none",
        edgecolor=color,
        linewidth=1.0,
        alpha=0.75,
        zorder=3,
    )
    ax.add_patch(ell)


def draw_joint_scatter(ax: plt.Axes, data: pd.DataFrame, dataset: str, show_ylabel: bool) -> None:
    sub = data[data["dataset"].eq(dataset)]
    for group, color, marker in [("HC", HC, "o"), ("AD", AD, "^")]:
        g = sub[sub["diagnosis_group"].eq(group)]
        ax.scatter(
            g["Y"],
            g["Z"],
            s=13,
            marker=marker,
            color=color,
            alpha=0.44 if dataset == "Pitt" else 0.62,
            edgecolor="white",
            linewidth=0.25,
            zorder=2,
        )
        if len(g):
            ax.scatter(
                [g["Y"].mean()],
                [g["Z"].mean()],
                s=42,
                marker=marker,
                color=color,
                edgecolor=INK,
                linewidth=0.55,
                zorder=4,
            )
            add_cov_ellipse(ax, g["Y"].to_numpy(dtype=float), g["Z"].to_numpy(dtype=float), color)
    ax.set_xlim(-0.02, 1.04)
    ax.set_ylim(0.002, 0.0345)
    ax.set_title(DATASET_FULL_LABELS[dataset], fontweight="bold", pad=3)
    ax.set_xlabel("Y")
    ax.set_ylabel("Z" if show_ylabel else "")
    if not show_ylabel:
        ax.set_yticklabels([])
    soften_axes(ax, grid_axis="")


def draw_effect_panel(ax: plt.Axes, effects: pd.DataFrame) -> None:
    offsets = {"Y": 0.22, "Z": 0.0, "YZ_joint_index": -0.22}
    base_y = np.arange(len(DATASET_ORDER))[::-1]
    for metric in ["Y", "Z", "YZ_joint_index"]:
        sub = effects[effects["metric"].eq(metric)].set_index("dataset").loc[DATASET_ORDER].reset_index()
        y = base_y + offsets[metric]
        x = sub["standardized_difference_hc_minus_ad"].to_numpy(dtype=float)
        lo = sub["standardized_difference_ci_low"].to_numpy(dtype=float)
        hi = sub["standardized_difference_ci_high"].to_numpy(dtype=float)
        ax.errorbar(
            x,
            y,
            xerr=[x - lo, hi - x],
            fmt=METRIC_MARKERS[metric],
            color=METRIC_COLORS[metric],
            ecolor=METRIC_COLORS[metric],
            elinewidth=0.95,
            capsize=2.4,
            markersize=4.4,
            markeredgecolor="white",
            markeredgewidth=0.4,
            label=METRIC_LABELS[metric],
            zorder=3,
        )
    ax.axvline(0, color="#8F979C", linewidth=0.8)
    ax.set_yticks(base_y)
    ax.set_yticklabels([DATASET_FULL_LABELS[d] for d in DATASET_ORDER])
    ax.set_xlabel("HC - AD standardized mean difference")
    ax.set_title("Diagnosis effects in proxy space", loc="left", fontweight="bold", pad=4)
    ax.set_xlim(-0.35, 1.55)
    handles = [
        Line2D(
            [0],
            [0],
            marker=METRIC_MARKERS[metric],
            linestyle="none",
            markerfacecolor=METRIC_COLORS[metric],
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=5.0,
            label=METRIC_LABELS[metric],
        )
        for metric in ["Y", "Z", "YZ_joint_index"]
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        frameon=False,
        ncol=3,
        borderaxespad=0,
        columnspacing=1.0,
        handletextpad=0.35,
    )
    soften_axes(ax, grid_axis="x")


def build() -> None:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    yz_data, yz_summary = load_yz_data()
    yz_data = add_joint_index(yz_data)
    effects = summarize_effects(yz_data)

    yz_data.to_csv(FIG_DIR / "Figure2_participant_source_data.csv", index=False)
    if yz_summary is not None:
        yz_summary.to_csv(FIG_DIR / "Figure2_statistical_summary.csv", index=False)
    effects.to_csv(FIG_DIR / "Figure2_joint_index_effect_summary.csv", index=False)

    fig = plt.figure(figsize=(7.28, 6.55), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        2,
        left=0.075,
        right=0.99,
        top=0.965,
        bottom=0.095,
        hspace=0.58,
        wspace=0.34,
        height_ratios=[1.05, 1.08, 1.0],
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    draw_proxy_violin(ax_a, yz_data, "Y", "Y proxy distributions", "Y proxy")
    draw_proxy_violin(ax_b, yz_data, "Z", "Z proxy distributions", "Z proxy")
    panel_label(ax_a, "a")
    panel_label(ax_b, "b")

    joint_gs = gs[1, :].subgridspec(1, 4, wspace=0.18)
    joint_axes = [fig.add_subplot(joint_gs[0, i]) for i in range(4)]
    for i, (ax, dataset) in enumerate(zip(joint_axes, DATASET_ORDER)):
        draw_joint_scatter(ax, yz_data, dataset, show_ylabel=i == 0)
    # Give the row heading its own line above the cohort titles.  Anchoring both
    # labels in figure coordinates prevents the shared heading from colliding
    # with the first cohort title after journal-width scaling.
    joint_row_heading_y = max(ax.get_position().y1 for ax in joint_axes) + 0.042
    fig.text(0.030, joint_row_heading_y, "c", ha="left", va="bottom", fontsize=8.8, fontweight="bold", color=INK)
    fig.text(0.075, joint_row_heading_y, "Joint Y-Z observation space", ha="left", va="bottom", fontsize=8.0, fontweight="bold", color=INK)

    ax_d = fig.add_subplot(gs[2, 0])
    ax_e = fig.add_subplot(gs[2, 1])
    draw_proxy_violin(ax_d, yz_data, "YZ_joint_index", "Joint Y-Z coverage index", "Percentile-rank\ngeometric mean")
    draw_effect_panel(ax_e, effects)
    panel_label(ax_d, "d")
    panel_label(ax_e, "e")

    handles = [
        Patch(facecolor=HC, edgecolor=HC, alpha=0.35, label="HC"),
        Patch(facecolor=AD, edgecolor=AD, alpha=0.35, label="AD"),
        Line2D([0], [0], marker="D", linestyle="none", markerfacecolor=INK, markeredgecolor="white", markersize=4.6, label="Mean"),
    ]
    ax_d.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=3,
        frameon=False,
        borderaxespad=0,
        columnspacing=1.0,
        handletextpad=0.35,
    )

    save_figure(fig, "Figure2_yz_distributions_validation_20260720")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Figure 2 from the locked participant-level Y/Z audit data."
    )
    parser.add_argument("--yz-participant-csv", required=True, type=Path)
    parser.add_argument("--yz-summary-csv", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    global YZ_PARTICIPANT_CSV, YZ_SUMMARY_CSV, FIG_DIR
    args = parse_args()
    YZ_PARTICIPANT_CSV = args.yz_participant_csv
    YZ_SUMMARY_CSV = args.yz_summary_csv
    FIG_DIR = args.output_dir
    build()


if __name__ == "__main__":
    main()
