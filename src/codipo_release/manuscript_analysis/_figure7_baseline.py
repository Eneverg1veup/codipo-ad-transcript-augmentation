#!/usr/bin/env python
"""Build Figure B v3 linking X/YZ proxy phenotypes and representation space."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from . import _figure7_v2 as v2

HERE = Path(__file__).resolve().parent
PCA_AUG_PATH = HERE / "source_data" / "pca_contamination_atlas_source.csv"
PCA_REFERENCE_PATH = HERE / "source_data" / "pca_training_reference_source.csv"
AUG_LONG_PATH = (
    HERE.parents[1]
    / "embedding_audit_v3_with_proxy"
    / "tables"
    / "aug_long_loaded_with_optional_proxy.csv"
)
base = v2.base


def residual_similarity_summary(aug_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = aug_long[
        aug_long["method"].isin(base.BASELINE_WITH_HARD)
    ][
        [
            "method",
            "seed",
            "source_id",
            "source_row_index",
            "source_label",
            "aug_round",
            "residual_cos",
        ]
    ].copy()
    pair_keys = ["method", "seed", "source_id", "aug_round"]
    if candidates[pair_keys].isna().any().any():
        raise ValueError("Residual Similarity pairing keys contain missing values.")
    if candidates.duplicated(pair_keys).any():
        raise ValueError(
            "Residual Similarity rows are not unique by method, seed, source, "
            "and augmentation round."
        )
    if not candidates["source_id"].astype(str).equals(
        candidates["source_row_index"].astype(str)
    ):
        raise ValueError(
            "Residual Similarity source_id does not match its source row."
        )
    if candidates["residual_cos"].isna().any():
        raise ValueError("Residual Similarity is missing for one or more source pairs.")
    summary = (
        candidates.groupby("method", observed=True)["residual_cos"]
        .agg(n_candidates="size", mean="mean", sd="std")
        .reindex(base.BASELINE_WITH_HARD)
        .reset_index()
    )
    return candidates, summary


def draw_x_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
) -> None:
    y_positions = np.arange(len(summary))
    for y, method in zip(y_positions, summary["method"]):
        color = base.METHOD_COLORS[method]
        row = summary[summary["method"].eq(method)].iloc[0]
        ax.errorbar(
            row["mean"],
            y,
            xerr=row["sd"],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=0.8,
            capsize=2.0,
            markersize=5.0,
            markeredgecolor=base.INK,
            markeredgewidth=0.45,
            zorder=3,
        )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(summary["method"])
    ax.invert_yaxis()
    ax.set_xlim(-0.20, 0.85)
    ax.set_xlabel("Residual Similarity (mean 卤 SD)")
    ax.set_title(
        "Candidate-level Residual Similarity",
        loc="left",
        fontweight="bold",
        pad=4.0,
    )
    base.clean_axis(ax, "x")


def utility_trend_summary(bridge: pd.DataFrame) -> pd.DataFrame:
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
    return summary


def draw_utility_departure_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
) -> None:
    for _, row in summary.iterrows():
        method = row["method"]
        color = base.METHOD_COLORS[method]
        ax.errorbar(
            row["contamination_mean"],
            row["f1_mean"],
            xerr=row["contamination_sd"],
            yerr=row["f1_sd"],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=0.75,
            capsize=1.8,
            markersize=5.2,
            markeredgecolor=base.INK,
            markeredgewidth=0.45,
            zorder=3,
        )

    ax.set_xlabel("Opposite-class kNN rate (%)")
    ax.set_ylabel("OverallMacro F1 (%)")
    ax.set_title(
        "Representation contamination and downstream utility",
        loc="left",
        fontweight="bold",
        pad=4.0,
    )
    base.clean_axis(ax, "both")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=base.METHOD_COLORS[method],
            markeredgecolor=base.INK,
            markeredgewidth=0.4,
            markersize=4.3,
            label=method,
        )
        for method in base.BASELINE_WITH_HARD
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.29),
        ncol=6,
        fontsize=5.4,
        columnspacing=0.9,
        handletextpad=0.3,
        borderaxespad=0,
    )


def main() -> None:
    base.set_style()
    bridge = pd.read_csv(base.BRIDGE_PATH)
    bridge = bridge[bridge["method"].isin(base.BASELINE_WITH_HARD)].copy()
    directional_summary = base.method_summary(bridge)
    aug_long = pd.read_csv(AUG_LONG_PATH)
    x_candidates, x_summary = residual_similarity_summary(aug_long)
    pca_aug = pd.read_csv(PCA_AUG_PATH)
    pca_reference = pd.read_csv(PCA_REFERENCE_PATH)
    utility_summary = utility_trend_summary(bridge)

    fig = plt.figure(figsize=(183 / 25.4, 165 / 25.4), facecolor="white")
    outer = fig.add_gridspec(
        3,
        1,
        left=0.075,
        right=0.992,
        top=0.965,
        bottom=0.070,
        height_ratios=[0.92, 1.18, 0.78],
        hspace=0.54,
    )

    top = outer[0, 0].subgridspec(
        1,
        2,
        width_ratios=[2.05, 1.0],
        wspace=0.34,
    )
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    base.draw_failure_map(ax_a, bridge, directional_summary)
    ax_a.set_title(
        "Methods occupy opposing diagnosis-specific Y/Z failure axes",
        loc="left",
        fontweight="bold",
        pad=5.0,
    )
    draw_x_panel(ax_b, x_summary)
    base.panel_label(ax_a, "a", x=-0.09, y=1.02)
    base.panel_label(ax_b, "b", x=-0.23, y=1.02)

    v2.draw_compact_pca_atlas(
        fig,
        outer[1, 0],
        pca_aug,
        pca_reference,
        panel_letter="c",
        panel_title=(
            "Proxy phenotypes correspond to class-specific "
            "representation contamination regions"
        ),
    )

    ax_d = fig.add_subplot(outer[2, 0])
    draw_utility_departure_panel(ax_d, utility_summary)
    base.panel_label(ax_d, "d", x=-0.065, y=1.02)

    directional_summary.to_csv(
        base.SOURCE_DIR / "figure_b_v3_directional_method_summary.csv",
        index=False,
    )
    x_candidates.to_csv(
        base.SOURCE_DIR / "figure_b_v3_residual_similarity_candidate_source.csv",
        index=False,
    )
    x_summary.to_csv(
        base.SOURCE_DIR / "figure_b_v3_residual_similarity_method_summary.csv",
        index=False,
    )
    utility_summary.to_csv(
        base.SOURCE_DIR / "figure_b_v3_contamination_utility_summary.csv",
        index=False,
    )
    base.save_bundle(fig, "figB_v3_xyz_representation_joint")
    plt.close(fig)
    print(f"Wrote Figure B v3 to {base.FIG_DIR}")


