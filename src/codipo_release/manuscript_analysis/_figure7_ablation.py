#!/usr/bin/env python
"""Build an ablation-only PCA contamination atlas.

Contamination is computed in the original 768-D embedding space. PCA is used
only as a shared visualization layer.
"""

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

ABLATION_METHODS = [
    "CoDiPO",
    "w/o X",
    "w/o Y/Z",
    "Cos-X",
    "Cosine-only",
    "Vanilla",
    "Hard filter",
]

DISPLAY_LABELS = {
    "CoDiPO": "CoDiPO",
    "w/o X": "w/o X",
    "w/o Y/Z": "w/o Y/Z",
    "Cos-X": "Cosine sim. as X",
    "Cosine-only": "Cosine-only",
    "Vanilla": "Vanilla",
    "Hard filter": "Hard filter",
}




def method_color(method: str) -> str:
    return base.METHOD_COLORS[DISPLAY_LABELS[method]]


def build_ablation_pca_atlas(
    pca_aug: pd.DataFrame,
    pca_source: pd.DataFrame,
    pca_variance: np.ndarray,
) -> None:
    shown = pca_aug[pca_aug["method"].isin(ABLATION_METHODS)].copy()
    xlim, ylim = atlas.projection_limits(shown, pca_source)

    fig = plt.figure(figsize=(183 / 25.4, 104 / 25.4), facecolor="white")
    grid = fig.add_gridspec(
        2,
        len(ABLATION_METHODS),
        left=0.082,
        right=0.992,
        top=0.830,
        bottom=0.215,
        wspace=0.050,
        hspace=0.105,
    )

    scatter = None
    for row, source_label in enumerate([0, 1]):
        for col, method in enumerate(ABLATION_METHODS):
            ax = fig.add_subplot(grid[row, col])
            atlas.draw_reference_points(ax, pca_source)
            panel = shown[
                shown["method"].eq(method)
                & shown["source_label"].astype(int).eq(source_label)
            ]
            scatter = ax.scatter(
                panel["dim1"],
                panel["dim2"],
                c=panel["knn_opposite_rate"],
                cmap=atlas.CONTAMINATION_CMAP,
                norm=atlas.CONTAMINATION_NORM,
                s=5.6,
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
                fontsize=4.9,
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
                show_x=row == 1 and col == 0,
                show_y=col == 0,
                x_label=f"PC1 ({100 * pca_variance[0]:.1f}%)",
                y_label=f"PC2 ({100 * pca_variance[1]:.1f}%)",
            )
            if row == 0:
                ax.set_title(
                    DISPLAY_LABELS[method],
                    fontsize=5.8,
                    fontweight="bold",
                    color=method_color(method),
                    pad=2.3,
                )

    fig.text(
        0.017,
        0.615,
        "HC source\nHC-to-AD",
        ha="left",
        va="center",
        rotation=90,
        fontsize=6.0,
        fontweight="bold",
        color=base.INK,
    )
    fig.text(
        0.017,
        0.362,
        "AD source\nAD-to-HC",
        ha="left",
        va="center",
        rotation=90,
        fontsize=6.0,
        fontweight="bold",
        color=base.INK,
    )
    fig.suptitle(
        "Ablation controls reveal the representation component of CoDiPO's proxy balance",
        x=0.082,
        y=0.953,
        ha="left",
        fontsize=8.3,
        fontweight="bold",
        color=base.INK,
    )

    reference_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=atlas.HC_REFERENCE,
            markeredgewidth=0.6,
            markersize=4.0,
            label="HC training reference",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="none",
            color=atlas.AD_REFERENCE,
            markeredgewidth=0.7,
            markersize=4.0,
            label="AD training reference",
        ),
    ]
    fig.legend(
        handles=reference_handles,
        loc="upper right",
        bbox_to_anchor=(0.990, 0.925),
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.32,
        fontsize=5.7,
    )

    if scatter is None:
        raise RuntimeError("No ablation PCA points were drawn.")
    color_ax = fig.add_axes([0.325, 0.090, 0.360, 0.023])
    colorbar = fig.colorbar(scatter, cax=color_ax, orientation="horizontal")
    colorbar.set_ticks(np.linspace(0, 1, 6))
    colorbar.set_ticklabels(["0", "20", "40", "60", "80", "100"])
    colorbar.ax.tick_params(labelsize=5.6, length=2.0, pad=1.1)
    colorbar.outline.set_linewidth(0.45)
    colorbar.set_label(
        "Opposite-class kNN contamination in original representation space (%)",
        fontsize=6.1,
        labelpad=2.2,
    )

    atlas.save_bundle(fig, "fig_representation_contamination_pca_atlas_ablations")
    plt.close(fig)


def summarize_ablation_contamination(pca_aug: pd.DataFrame) -> pd.DataFrame:
    shown = pca_aug[pca_aug["method"].isin(ABLATION_METHODS)].copy()
    rows: list[dict[str, float | int | str]] = []
    for method in ABLATION_METHODS:
        for source_label in [0, 1]:
            group = shown[
                shown["method"].eq(method)
                & shown["source_label"].astype(int).eq(source_label)
            ]
            rows.append(
                {
                    "method_raw": method,
                    "method": DISPLAY_LABELS[method],
                    "source_label": source_label,
                    "source_diagnosis": "HC" if source_label == 0 else "AD",
                    "contamination_direction": "HC-to-AD"
                    if source_label == 0
                    else "AD-to-HC",
                    "n": len(group),
                    "mean": group["knn_opposite_rate"].mean(),
                    "median": group["knn_opposite_rate"].median(),
                    "sd": group["knn_opposite_rate"].std(),
                    "rate_ge_50_percent": 100
                    * (group["knn_opposite_rate"] >= 0.5).mean(),
                    "rate_ge_80_percent": 100
                    * (group["knn_opposite_rate"] >= 0.8).mean(),
                }
            )
    return pd.DataFrame(rows)


def summarize_proxy_utility_bridge(pca_aug: pd.DataFrame) -> pd.DataFrame:
    bridge = pd.read_csv(base.BRIDGE_PATH)
    bridge = bridge[bridge["method"].isin(ABLATION_METHODS)].copy()
    bridge["method"] = pd.Categorical(
        bridge["method"],
        categories=ABLATION_METHODS,
        ordered=True,
    )
    summary = (
        bridge.groupby("method", observed=True)
        .agg(
            f1_mean=("OverallMacro_F1", "mean"),
            f1_sd=("OverallMacro_F1", "std"),
            ad_overcompletion_mean=("ad_overcompletion_score_mean", "mean"),
            ad_overcompletion_sd=("ad_overcompletion_score_mean", "std"),
            hc_evidence_loss_mean=("hc_evidence_loss_score_mean", "mean"),
            hc_evidence_loss_sd=("hc_evidence_loss_score_mean", "std"),
            overall_contamination_mean=("overall_contamination", "mean"),
            overall_contamination_sd=("overall_contamination", "std"),
            ad_to_hc_contamination_mean=("ad_to_hc_contamination", "mean"),
            ad_to_hc_contamination_sd=("ad_to_hc_contamination", "std"),
            hc_to_ad_contamination_mean=("hc_to_ad_contamination", "mean"),
            hc_to_ad_contamination_sd=("hc_to_ad_contamination", "std"),
            proxy_x_pass_rate_mean=("proxy_x_pass_rate", "mean"),
            proxy_yz_pass_rate_mean=("proxy_yz_pass_rate", "mean"),
            proxy_joint_pass_rate_mean=("proxy_joint_pass_rate", "mean"),
        )
        .reset_index()
    )
    summary["method_raw"] = summary["method"].astype(str)
    summary["method"] = summary["method_raw"].map(DISPLAY_LABELS)

    candidate_x = (
        pca_aug[pca_aug["method"].isin(ABLATION_METHODS)]
        .groupby("method", observed=True)["residual_cos"]
        .agg(residual_similarity_mean="mean", residual_similarity_sd="std")
        .reset_index()
        .rename(columns={"method": "method_raw"})
    )
    summary = summary.merge(candidate_x, on="method_raw", how="left")
    columns = ["method_raw", "method"] + [
        col for col in summary.columns if col not in {"method_raw", "method"}
    ]
    return summary[columns]


def export_source_data(pca_aug: pd.DataFrame, pca_source: pd.DataFrame) -> None:
    atlas.SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    shown = pca_aug[pca_aug["method"].isin(ABLATION_METHODS)].copy()
    shown["method"] = pd.Categorical(
        shown["method"],
        categories=ABLATION_METHODS,
        ordered=True,
    )
    shown = shown.sort_values(
        ["method", "source_label", "seed", "source_id", "aug_round"]
    )
    shown["method_display"] = shown["method"].map(DISPLAY_LABELS)
    aug_columns = [
        "aug_index",
        "method",
        "method_display",
        "seed",
        "source_id",
        "source_row_index",
        "source_label",
        "aug_round",
        "residual_cos",
        "knn_opposite_rate",
        "dim1",
        "dim2",
    ]
    source_columns = [
        "source_row_index",
        "source_id",
        "source_label",
        "dim1",
        "dim2",
    ]
    shown[aug_columns].to_csv(
        atlas.SOURCE_DIR / "pca_contamination_atlas_ablations_source.csv",
        index=False,
    )
    pca_source[source_columns].to_csv(
        atlas.SOURCE_DIR / "pca_contamination_atlas_ablations_training_reference_source.csv",
        index=False,
    )
    summarize_ablation_contamination(pca_aug).to_csv(
        atlas.SOURCE_DIR / "method_class_contamination_summary_ablations.csv",
        index=False,
    )
    summarize_proxy_utility_bridge(pca_aug).to_csv(
        atlas.SOURCE_DIR / "ablation_proxy_representation_utility_summary.csv",
        index=False,
    )


def main() -> None:
    atlas.set_style()
    cache, source, aug, pca_coords = atlas.load_inputs()
    aug = atlas.compute_sample_contamination(cache, source, aug)
    selected = atlas.reproduce_pca_selection(aug)
    pca_aug, pca_source = atlas.attach_existing_pca(selected, source, pca_coords)
    pca_variance = atlas.compute_existing_pca_variance(cache, selected)

    build_ablation_pca_atlas(pca_aug, pca_source, pca_variance)
    export_source_data(pca_aug, pca_source)
    print(f"Wrote ablation PCA atlas to {atlas.FIG_DIR}")
    print(f"Wrote ablation source data to {atlas.SOURCE_DIR}")


