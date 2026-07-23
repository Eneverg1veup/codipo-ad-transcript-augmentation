#!/usr/bin/env python
"""Build the revised Figure B with class-aware PCA contamination evidence."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from . import _figure7_atlas as atlas
from . import _figure7_base as base

HERE = Path(__file__).resolve().parent
PCA_AUG_PATH = HERE / "source_data" / "pca_contamination_atlas_source.csv"
PCA_REFERENCE_PATH = HERE / "source_data" / "pca_training_reference_source.csv"


def draw_compact_pca_atlas(
    fig: plt.Figure,
    slot,
    pca_aug: pd.DataFrame,
    pca_reference: pd.DataFrame,
    panel_letter: str = "b",
    panel_title: str = (
        "Diagnosis-directional failures resolve into "
        "class-specific representation regions"
    ),
) -> list[plt.Axes]:
    grid = slot.subgridspec(
        3,
        6,
        height_ratios=[1.0, 1.0, 0.10],
        wspace=0.055,
        hspace=0.12,
    )
    xlim, ylim = atlas.projection_limits(pca_aug, pca_reference)
    axes: list[plt.Axes] = []
    scatter = None

    for row, source_label in enumerate([0, 1]):
        for col, method in enumerate(base.BASELINE_WITH_HARD):
            ax = fig.add_subplot(grid[row, col])
            axes.append(ax)
            atlas.draw_reference_points(ax, pca_reference)
            panel = pca_aug[
                pca_aug["method"].eq(method)
                & pca_aug["source_label"].astype(int).eq(source_label)
            ]
            scatter = ax.scatter(
                panel["dim1"],
                panel["dim2"],
                c=panel["knn_opposite_rate"],
                cmap=atlas.CONTAMINATION_CMAP,
                norm=atlas.CONTAMINATION_NORM,
                s=4.2,
                edgecolors="none",
                alpha=0.67,
                zorder=3,
            )
            ax.text(
                0.96,
                0.95,
                f"{100 * panel['knn_opposite_rate'].mean():.1f}%",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=4.8,
                fontweight="bold",
                color=base.INK,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                    "pad": 0.5,
                },
            )
            atlas.style_projection_axis(
                ax,
                xlim,
                ylim,
                show_x=row == 1 and col == 0,
                show_y=col == 0,
                x_label="PC1 (16.5%)",
                y_label="PC2 (7.4%)",
            )
            if col != 0:
                ax.set_xticklabels([])
                ax.tick_params(axis="x", length=0)
            if col == 0:
                ax.text(
                    0.035,
                    0.95,
                    "HC source" if source_label == 0 else "AD source",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=4.8,
                    fontweight="bold",
                    color=base.INK,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.75,
                        "pad": 0.45,
                    },
                )
            if row == 0:
                ax.set_title(method, fontsize=5.9, fontweight="bold", pad=2.0)

    color_ax = fig.add_subplot(grid[2, 2:4])
    if scatter is None:
        raise RuntimeError("No PCA contamination points were drawn.")
    colorbar = fig.colorbar(scatter, cax=color_ax, orientation="horizontal")
    colorbar.set_ticks(np.linspace(0, 1, 6))
    colorbar.set_ticklabels(["0", "20", "40", "60", "80", "100"])
    colorbar.ax.tick_params(labelsize=4.8, length=1.7, pad=0.8)
    colorbar.outline.set_linewidth(0.4)
    colorbar.set_label(
        "Opposite-class kNN contamination (%)",
        fontsize=5.3,
        labelpad=1.5,
    )

    source_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=atlas.HC_REFERENCE,
            markeredgewidth=0.55,
            markersize=3.5,
            label="HC train",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="none",
            color=atlas.AD_REFERENCE,
            markeredgewidth=0.65,
            markersize=3.5,
            label="AD train",
        ),
    ]
    legend_ax = fig.add_subplot(grid[2, 4:6])
    legend_ax.axis("off")
    legend_ax.legend(
        handles=source_handles,
        loc="center",
        ncol=2,
        columnspacing=0.7,
        handletextpad=0.25,
        fontsize=5.2,
    )

    bbox = slot.get_position(fig)
    fig.text(
        bbox.x0,
        bbox.y1 + 0.014,
        panel_title,
        ha="left",
        va="bottom",
        fontsize=7.2,
        fontweight="bold",
        color=base.INK,
    )
    fig.text(
        bbox.x0 - 0.045,
        bbox.y1 + 0.014,
        panel_letter,
        ha="left",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color=base.INK,
    )
    return axes


def draw_contamination_utility_panel(
    ax: plt.Axes,
    bridge: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        bridge.groupby("method", observed=True)
        .agg(
            contamination_mean=("overall_contamination", "mean"),
            contamination_sd=("overall_contamination", "std"),
            f1_mean=("OverallMacro_F1", "mean"),
            f1_sd=("OverallMacro_F1", "std"),
        )
        .reindex(base.BASELINE_WITH_HARD)
        .reset_index()
    )
    summary["contamination_mean"] *= 100
    summary["contamination_sd"] *= 100

    for _, row in summary.iterrows():
        method = row["method"]
        ax.errorbar(
            row["contamination_mean"],
            row["f1_mean"],
            xerr=row["contamination_sd"],
            yerr=row["f1_sd"],
            fmt="o",
            color=base.METHOD_COLORS[method],
            ecolor=base.METHOD_COLORS[method],
            elinewidth=0.75,
            capsize=1.8,
            markersize=4.8,
            markeredgecolor=base.INK,
            markeredgewidth=0.45,
            zorder=3,
        )

    codipo = summary[summary["method"].eq("CoDiPO")].iloc[0]
    hard = summary[summary["method"].eq("Hard filter")].iloc[0]
    ax.plot(
        [codipo["contamination_mean"], hard["contamination_mean"]],
        [codipo["f1_mean"], hard["f1_mean"]],
        color="#7D858A",
        linewidth=0.7,
        linestyle=":",
        zorder=1,
    )

    offsets = {
        "CoDiPO": (0.6, 0.25, "left"),
        "EDA": (0.5, -0.35, "left"),
        "ICL Direct": (0.5, -0.35, "left"),
        "ICL Imitation": (-1.0, 0.55, "right"),
        "ICL Rewrite": (0.8, -0.55, "left"),
        "Hard filter": (0.5, -0.40, "left"),
    }
    for _, row in summary.iterrows():
        dx, dy, alignment = offsets[row["method"]]
        ax.text(
            row["contamination_mean"] + dx,
            row["f1_mean"] + dy,
            row["method"],
            ha=alignment,
            va="center",
            fontsize=5.1,
            color=base.METHOD_COLORS[row["method"]],
            fontweight="bold" if row["method"] == "CoDiPO" else "normal",
        )

    ax.set_xlabel("Opposite-class kNN rate (%)")
    ax.set_ylabel("OverallMacro F1 (%)")
    ax.set_title(
        "Similar contamination, different utility",
        loc="left",
        fontweight="bold",
        pad=4.0,
    )
    base.clean_axis(ax, "both")
    return summary


def main() -> None:
    base.set_style()
    bridge = pd.read_csv(base.BRIDGE_PATH)
    bridge = bridge[bridge["method"].isin(base.BASELINE_WITH_HARD)].copy()
    summary = base.method_summary(bridge)
    pca_aug = pd.read_csv(PCA_AUG_PATH)
    pca_reference = pd.read_csv(PCA_REFERENCE_PATH)

    fig = plt.figure(figsize=(183 / 25.4, 183 / 25.4), facecolor="white")
    outer = fig.add_gridspec(
        3,
        1,
        left=0.075,
        right=0.992,
        top=0.965,
        bottom=0.065,
        height_ratios=[0.95, 1.20, 0.82],
        hspace=0.53,
    )

    ax_a = fig.add_subplot(outer[0, 0])
    base.draw_failure_map(ax_a, bridge, summary)
    ax_a.set_title(
        "Methods occupy opposing diagnosis-specific failure axes",
        loc="left",
        fontweight="bold",
        pad=5.0,
    )
    base.panel_label(ax_a, "a", x=-0.065, y=1.02)

    draw_compact_pca_atlas(fig, outer[1, 0], pca_aug, pca_reference)

    bottom = outer[2, 0].subgridspec(1, 3, wspace=0.43)
    ax_c, ax_d, ax_e = [fig.add_subplot(bottom[0, i]) for i in range(3)]
    base.draw_summary_bar(
        ax_c,
        summary,
        "ad_overcompletion_score_mean_mean",
        "ad_overcompletion_score_mean_sd",
        "AD-source over-completion",
        "Continuous upper-band severity",
        True,
    )
    base.draw_summary_bar(
        ax_d,
        summary,
        "hc_evidence_loss_score_mean_mean",
        "hc_evidence_loss_score_mean_sd",
        "HC-source evidence loss",
        "Continuous lower-band severity",
        False,
    )
    utility_summary = draw_contamination_utility_panel(ax_e, bridge)

    for label, ax in zip("cde", [ax_c, ax_d, ax_e]):
        base.panel_label(ax, label, x=-0.22 if label != "e" else -0.18)

    bridge.to_csv(
        base.SOURCE_DIR / "figure_b_v2_directional_seed_source.csv",
        index=False,
    )
    summary.to_csv(
        base.SOURCE_DIR / "figure_b_v2_directional_method_summary.csv",
        index=False,
    )
    utility_summary.to_csv(
        base.SOURCE_DIR / "figure_b_v2_contamination_utility_summary.csv",
        index=False,
    )
    base.save_bundle(fig, "figB_v2_directional_failure_with_pca")
    plt.close(fig)
    print(f"Wrote revised Figure B to {base.FIG_DIR}")


