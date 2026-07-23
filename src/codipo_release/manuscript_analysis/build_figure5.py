#!/usr/bin/env python
"""Build ablation mechanism audit tables and a four-panel diagnostic figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.stats import spearmanr


FIG_DIR: Path
SOURCE_DIR: Path
BRIDGE_PATH: Path
AUG_LONG_PATH: Path
METHODS = [
    "CoDiPO",
    "w/o X",
    "w/o Y/Z",
    "Cos-X",
    "Cosine-only",
    "Vanilla",
    "Hard filter",
]

DISPLAY = {
    "CoDiPO": "CoDiPO",
    "w/o X": "w/o X",
    "w/o Y/Z": "w/o YZ",
    "Cos-X": "Cosine sim. as X",
    "Cosine-only": "Cosine-only",
    "Vanilla": "Vanilla",
    "Hard filter": "Hard Filter",
}

INK = "#202124"
MUTED = "#687078"
GRID = "#E5E9EC"
BLUE = "#28688E"
SALMON = "#D58E79"
TEAL = "#3D9896"
GOLD = "#C9A23F"

METHOD_COLORS = {
    "CoDiPO": BLUE,
    "w/o X": "#C67B6C",
    "w/o Y/Z": "#6F9B76",
    "Cos-X": "#737373",
    "Cosine-only": GOLD,
    "Vanilla": "#8D96A0",
    "Hard filter": TEAL,
}

BALANCE_METRICS = [
    {
        "column": "residual_similarity_mean",
        "label": "Residual\nSimilarity",
        "domain": "X",
    },
    {
        "column": "proxy_joint_pass_rate",
        "label": "Joint\npass",
        "domain": "Proxy selection",
    },
    {
        "column": "conditional_x_pass_within_yz_fail",
        "label": "X pass\nif Y/Z fail",
        "domain": "Proxy selection",
    },
    {
        "column": "yz_x_phi",
        "label": "Y/Z-X\nphi",
        "domain": "Proxy selection",
    },
    {
        "column": "ad_overcompletion_score_mean",
        "label": "AD over-\ncompletion",
        "domain": "Directional Y/Z",
    },
    {
        "column": "hc_evidence_loss_score_mean",
        "label": "HC evidence\nloss",
        "domain": "Directional Y/Z",
    },
    {
        "column": "ad_to_hc_contamination",
        "label": "AD-to-HC\nkNN",
        "domain": "Representation",
    },
    {
        "column": "hc_to_ad_contamination",
        "label": "HC-to-AD\nkNN",
        "domain": "Representation",
    },
]

CONTRAST_COLUMNS = [
    "ExternalCohort_F1",
    "residual_similarity_mean",
    "proxy_joint_pass_rate",
    "conditional_x_pass_within_yz_fail",
    "yz_x_phi",
    "ad_overcompletion_score_mean",
    "hc_evidence_loss_score_mean",
    "ad_to_hc_contamination",
    "hc_to_ad_contamination",
]


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 6.8,
            "axes.titlesize": 7.3,
            "axes.labelsize": 6.8,
            "axes.linewidth": 0.6,
            "axes.edgecolor": INK,
            "xtick.labelsize": 5.9,
            "ytick.labelsize": 5.9,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "legend.fontsize": 5.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )


def save_bundle(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "svg": {},
        "pdf": {},
        "png": {"dpi": 450},
        "tiff": {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}},
    }
    for extension, kwargs in outputs.items():
        path = FIG_DIR / f"{stem}.{extension}"
        try:
            fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        except TypeError:
            kwargs.pop("pil_kwargs", None)
            fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)


def clean_axis(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis in {"x", "y", "both"}:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.50)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=INK,
    )


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [p for p in [BRIDGE_PATH, AUG_LONG_PATH] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    bridge = pd.read_csv(BRIDGE_PATH)
    bridge = bridge[bridge["method"].isin(METHODS)].copy()
    bridge["method"] = pd.Categorical(
        bridge["method"], categories=METHODS, ordered=True
    )

    aug = pd.read_csv(AUG_LONG_PATH)
    candidates = aug[aug["method"].isin(METHODS)][
        [
            "method",
            "seed",
            "source_id",
            "source_row_index",
            "source_label",
            "aug_round",
            "bucket_name",
            "yz_pass",
            "x_pass",
            "joint_pass",
            "residual_cos",
        ]
    ].copy()
    candidates["method"] = pd.Categorical(
        candidates["method"], categories=METHODS, ordered=True
    )
    validate_candidate_pairs(candidates)

    seed_residual = (
        candidates.groupby(["method", "seed"], observed=True)["residual_cos"]
        .agg(
            residual_similarity_mean="mean",
            residual_similarity_sd="std",
            residual_similarity_n="size",
        )
        .reset_index()
    )
    bridge = bridge.merge(seed_residual, on=["method", "seed"], how="left")
    if bridge["residual_similarity_mean"].isna().any():
        raise ValueError("Seed-level residual similarity did not merge onto bridge.")

    bridge["method_display"] = bridge["method"].astype(str).map(DISPLAY)
    candidates["method_display"] = candidates["method"].astype(str).map(DISPLAY)
    return bridge, candidates, seed_residual


def validate_candidate_pairs(candidates: pd.DataFrame) -> None:
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
            "Each augmentation candidate must be paired only with its own source row."
        )
    if candidates["residual_cos"].isna().any():
        raise ValueError("Residual Similarity is missing for one or more candidates.")
    counts = candidates.groupby("method", observed=True).size().reindex(METHODS)
    if not counts.eq(1080).all():
        raise ValueError(f"Unexpected ablation candidate counts: {counts.to_dict()}")


def method_summary(bridge: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ExternalCohort_F1",
        "proxy_x_pass_rate",
        "proxy_yz_pass_rate",
        "proxy_joint_pass_rate",
        "conditional_x_pass_within_yz_pass",
        "conditional_x_pass_within_yz_fail",
        "conditional_x_pass_diff",
        "yz_x_phi",
        "ad_overcompletion_score_mean",
        "hc_evidence_loss_score_mean",
        "ad_to_hc_contamination",
        "hc_to_ad_contamination",
        "overall_contamination",
        "residual_similarity_mean",
    ]
    rows: list[dict[str, float | str]] = []
    for method in METHODS:
        group = bridge[bridge["method"].astype(str).eq(method)]
        row: dict[str, float | str] = {
            "method_raw": method,
            "method": DISPLAY[method],
            "n_seeds": len(group),
        }
        for column in columns:
            row[f"{column}_mean"] = group[column].mean()
            row[f"{column}_sd"] = group[column].std()
        rows.append(row)
    return pd.DataFrame(rows)


def compute_balance_distance(
    bridge: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_columns = [item["column"] for item in BALANCE_METRICS]
    means = (
        bridge.groupby("method", observed=True)[metric_columns + ["ExternalCohort_F1"]]
        .mean()
        .reindex(METHODS)
    )
    means.index = means.index.astype(str)

    metric_means = means[metric_columns]
    metric_sd = metric_means.std(axis=0, ddof=0).replace(0, np.nan)
    z = (metric_means - metric_means.mean(axis=0)) / metric_sd
    codipo = z.loc["CoDiPO"]
    deviation = z.subtract(codipo, axis=1)

    component_rows: list[dict[str, float | str]] = []
    for method in METHODS:
        for item in BALANCE_METRICS:
            component_rows.append(
                {
                    "method_raw": method,
                    "method": DISPLAY[method],
                    "metric": item["column"],
                    "metric_label": item["label"].replace("\n", " "),
                    "domain": item["domain"],
                    "method_value": means.loc[method, item["column"]],
                    "codipo_value": means.loc["CoDiPO", item["column"]],
                    "standardized_deviation_from_codipo": deviation.loc[
                        method, item["column"]
                    ],
                }
            )
    components = pd.DataFrame(component_rows)

    domain_rows: list[dict[str, float | str]] = []
    for method in METHODS:
        row: dict[str, float | str] = {
            "method_raw": method,
            "method": DISPLAY[method],
            "f1_mean": means.loc[method, "ExternalCohort_F1"],
        }
        row["overall_balance_distance"] = float(
            np.sqrt(np.nanmean(np.square(deviation.loc[method, metric_columns])))
        )
        for domain in sorted({item["domain"] for item in BALANCE_METRICS}):
            cols = [
                item["column"]
                for item in BALANCE_METRICS
                if item["domain"] == domain
            ]
            key = domain.lower().replace(" ", "_").replace("/", "_")
            row[f"{key}_distance"] = float(
                np.sqrt(np.nanmean(np.square(deviation.loc[method, cols])))
            )
        domain_rows.append(row)
    distance = pd.DataFrame(domain_rows)
    domain_columns = [c for c in distance.columns if c.endswith("_distance")]
    domain_only = [c for c in domain_columns if c != "overall_balance_distance"]
    distance["dominant_departure_domain"] = (
        distance.set_index("method_raw")[domain_only]
        .idxmax(axis=1)
        .str.replace("_distance", "", regex=False)
        .str.replace("_", " ")
        .reindex(METHODS)
        .to_numpy()
    )
    distance.loc[
        distance["method_raw"].eq("CoDiPO"), "dominant_departure_domain"
    ] = "reference"

    correlation_rows = []
    rho = spearmanr(
        distance["overall_balance_distance"],
        distance["f1_mean"],
    )
    correlation_rows.append(
        {
            "scope": "method_level_with_codipo",
            "n": len(distance),
            "spearman_rho": rho.statistic,
            "p_value": rho.pvalue,
        }
    )
    without_codipo = distance[~distance["method_raw"].eq("CoDiPO")]
    rho_wo = spearmanr(
        without_codipo["overall_balance_distance"],
        without_codipo["f1_mean"],
    )
    correlation_rows.append(
        {
            "scope": "method_level_without_codipo",
            "n": len(without_codipo),
            "spearman_rho": rho_wo.statistic,
            "p_value": rho_wo.pvalue,
        }
    )
    correlations = pd.DataFrame(correlation_rows)
    return distance, components, correlations


def matched_contrasts(summary: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    indexed = summary.set_index("method_raw")
    component_index = components.set_index(["method_raw", "metric"])
    rows: list[dict[str, float | str]] = []
    for method in METHODS:
        if method == "CoDiPO":
            continue
        method_components = components[components["method_raw"].eq(method)].copy()
        dominant = method_components.loc[
            method_components["standardized_deviation_from_codipo"].abs().idxmax()
        ]
        row: dict[str, float | str] = {
            "method_raw": method,
            "method": DISPLAY[method],
            "dominant_standardized_departure_metric": dominant["metric_label"],
            "dominant_departure_domain": dominant["domain"],
            "dominant_standardized_departure": dominant[
                "standardized_deviation_from_codipo"
            ],
        }
        for column in CONTRAST_COLUMNS:
            col = f"{column}_mean"
            method_value = indexed.loc[method, col]
            codipo_value = indexed.loc["CoDiPO", col]
            delta = method_value - codipo_value
            row[f"{column}_codipo"] = codipo_value
            row[f"{column}_method"] = method_value
            row[f"delta_{column}"] = delta
            if (method, column) in component_index.index:
                row[f"standardized_delta_{column}"] = component_index.loc[
                    (method, column), "standardized_deviation_from_codipo"
                ]
        rows.append(row)
    return pd.DataFrame(rows)


def residual_similarity_tables(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = candidates[
        [
            "method",
            "method_display",
            "seed",
            "source_id",
            "source_row_index",
            "source_label",
            "aug_round",
            "bucket_name",
            "yz_pass",
            "x_pass",
            "joint_pass",
            "residual_cos",
        ]
    ].copy()
    source["method"] = source["method"].astype(str)
    source = source.sort_values(
        ["method", "source_label", "seed", "source_id", "aug_round"]
    )

    quantiles = (
        source.groupby(["method", "method_display"], observed=True)["residual_cos"]
        .quantile([0.05, 0.25, 0.50, 0.75, 0.95])
        .unstack()
        .reset_index()
        .rename(
            columns={
                0.05: "q05",
                0.25: "q25",
                0.50: "median",
                0.75: "q75",
                0.95: "q95",
            }
        )
    )
    summary = (
        source.groupby(["method", "method_display"], observed=True)["residual_cos"]
        .agg(n="size", mean="mean", sd="std")
        .reset_index()
    )
    quantiles = summary.merge(quantiles, on=["method", "method_display"], how="left")
    quantiles["method"] = pd.Categorical(
        quantiles["method"], categories=METHODS, ordered=True
    )
    quantiles = quantiles.sort_values("method")
    return source, quantiles


def conditional_coupling_summary(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "method_raw",
        "method",
        "conditional_x_pass_within_yz_pass_mean",
        "conditional_x_pass_within_yz_pass_sd",
        "conditional_x_pass_within_yz_fail_mean",
        "conditional_x_pass_within_yz_fail_sd",
        "conditional_x_pass_diff_mean",
        "conditional_x_pass_diff_sd",
        "yz_x_phi_mean",
        "yz_x_phi_sd",
        "proxy_x_pass_rate_mean",
        "proxy_yz_pass_rate_mean",
        "proxy_joint_pass_rate_mean",
    ]
    return summary[columns].copy()


def export_tables(
    bridge: pd.DataFrame,
    candidates: pd.DataFrame,
    summary: pd.DataFrame,
    distance: pd.DataFrame,
    components: pd.DataFrame,
    correlations: pd.DataFrame,
    contrasts: pd.DataFrame,
    residual_source: pd.DataFrame,
    residual_quantiles: pd.DataFrame,
    coupling: pd.DataFrame,
) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    bridge.to_csv(SOURCE_DIR / "ablation_seed_level_source.csv", index=False)
    summary.to_csv(SOURCE_DIR / "ablation_method_summary.csv", index=False)
    distance.to_csv(SOURCE_DIR / "ablation_balance_distance_source.csv", index=False)
    components.to_csv(
        SOURCE_DIR / "ablation_balance_distance_components.csv", index=False
    )
    correlations.to_csv(
        SOURCE_DIR / "ablation_balance_distance_correlations.csv", index=False
    )
    contrasts.to_csv(
        SOURCE_DIR / "ablation_matched_contrasts_vs_codipo.csv", index=False
    )
    residual_source.to_csv(
        SOURCE_DIR / "ablation_residual_similarity_candidate_source.csv",
        index=False,
    )
    residual_quantiles.to_csv(
        SOURCE_DIR / "ablation_residual_similarity_quantiles.csv",
        index=False,
    )
    coupling.to_csv(
        SOURCE_DIR / "ablation_conditional_coupling_summary.csv",
        index=False,
    )
    candidates.groupby(["method_display", "bucket_name"], observed=True).agg(
        n=("residual_cos", "size"),
        residual_similarity_mean=("residual_cos", "mean"),
        joint_pass_rate=("joint_pass", "mean"),
    ).reset_index().to_csv(
        SOURCE_DIR / "ablation_candidate_bucket_residual_summary.csv",
        index=False,
    )


def draw_balance_distance_panel(
    ax: plt.Axes,
    distance: pd.DataFrame,
    summary: pd.DataFrame,
    correlations: pd.DataFrame,
) -> None:
    f1_sd = summary.set_index("method_raw")["ExternalCohort_F1_sd"]
    for _, row in distance.iterrows():
        method = row["method_raw"]
        ax.errorbar(
            row["overall_balance_distance"],
            row["f1_mean"],
            yerr=f1_sd.loc[method],
            fmt="o",
            color=METHOD_COLORS[method],
            ecolor=METHOD_COLORS[method],
            elinewidth=0.75,
            capsize=1.8,
            markersize=5.0,
            markeredgecolor=INK,
            markeredgewidth=0.42,
            zorder=3,
        )

    ax.set_xlabel("Standardized multi-metric distance from CoDiPO")
    ax.set_ylabel("External-cohort average F1 (%)")
    ax.set_title("Multi-metric distance and external F1", loc="left", fontweight="bold")
    clean_axis(ax, "both")


def draw_coupling_panel(ax: plt.Axes, coupling: pd.DataFrame) -> None:
    x = np.arange(len(METHODS))
    indexed = coupling.set_index("method_raw").reindex(METHODS)
    y_pass = indexed["conditional_x_pass_within_yz_pass_mean"].to_numpy()
    y_fail = indexed["conditional_x_pass_within_yz_fail_mean"].to_numpy()
    y_pass_sd = indexed["conditional_x_pass_within_yz_pass_sd"].to_numpy()
    y_fail_sd = indexed["conditional_x_pass_within_yz_fail_sd"].to_numpy()

    for i, method in enumerate(METHODS):
        ax.plot(
            [x[i], x[i]],
            [y_pass[i], y_fail[i]],
            color="#B8C0C5",
            linewidth=0.75,
            zorder=1,
        )
        ax.errorbar(
            x[i] - 0.08,
            y_pass[i],
            yerr=y_pass_sd[i],
            fmt="o",
            color=METHOD_COLORS[method],
            markersize=4.5,
            capsize=1.5,
            elinewidth=0.65,
            markeredgecolor=INK,
            markeredgewidth=0.35,
            zorder=3,
        )
        ax.errorbar(
            x[i] + 0.08,
            y_fail[i],
            yerr=y_fail_sd[i],
            fmt="s",
            color=METHOD_COLORS[method],
            markersize=4.2,
            capsize=1.5,
            elinewidth=0.65,
            markeredgecolor=INK,
            markeredgewidth=0.35,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[m] for m in METHODS], rotation=28, ha="right")
    ax.set_ylabel("Conditional X-pass rate (%)")
    ax.set_title("Conditional X-pass by Y/Z feasibility", loc="left", fontweight="bold")
    clean_axis(ax, "y")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=INK,
            markersize=4.3,
            label="P(X-pass | Y/Z-pass)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            color=INK,
            markersize=4.0,
            label="P(X-pass | Y/Z-fail)",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.34),
        ncol=2,
        columnspacing=0.9,
        handletextpad=0.35,
    )


def draw_contrast_heatmap(ax: plt.Axes, components: pd.DataFrame) -> None:
    matrix = components.pivot(
        index="method_raw",
        columns="metric",
        values="standardized_deviation_from_codipo",
    ).reindex(METHODS[1:])
    metric_order = [item["column"] for item in BALANCE_METRICS]
    matrix = matrix[metric_order]
    vmax = float(np.nanmax(np.abs(matrix.to_numpy())))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_yticks(np.arange(len(METHODS) - 1))
    ax.set_yticklabels([DISPLAY[m] for m in METHODS[1:]])
    ax.set_xticks(np.arange(len(metric_order)))
    ax.set_xticklabels(
        [item["label"] for item in BALANCE_METRICS],
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.45)
        spine.set_edgecolor("#B7C0C5")
    ax.set_title("Matched ablation deviations from CoDiPO", loc="left", fontweight="bold")
    colorbar = plt.colorbar(im, ax=ax, fraction=0.030, pad=0.018)
    colorbar.set_label("Deviation from CoDiPO (SD units)", fontsize=5.8)
    colorbar.ax.tick_params(labelsize=5.4, length=2)


def draw_residual_panel(
    ax: plt.Axes,
    residual_source: pd.DataFrame,
    residual_quantiles: pd.DataFrame,
) -> None:
    data = [
        residual_source[residual_source["method"].eq(method)]["residual_cos"].to_numpy()
        for method in METHODS
    ]
    positions = np.arange(len(METHODS))
    box = ax.boxplot(
        data,
        vert=False,
        positions=positions,
        widths=0.58,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": INK, "linewidth": 0.85},
        whiskerprops={"color": "#7D878D", "linewidth": 0.65},
        capprops={"color": "#7D878D", "linewidth": 0.65},
    )
    for patch, method in zip(box["boxes"], METHODS):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.34)
        patch.set_edgecolor(METHOD_COLORS[method])
        patch.set_linewidth(0.75)

    indexed = residual_quantiles.set_index("method").reindex(METHODS)
    ax.scatter(
        indexed["mean"],
        positions,
        marker="D",
        s=13,
        color=[METHOD_COLORS[m] for m in METHODS],
        edgecolor=INK,
        linewidth=0.35,
        zorder=4,
        label="mean",
    )
    ax.set_yticks(positions)
    ax.set_yticklabels([DISPLAY[m] for m in METHODS])
    ax.invert_yaxis()
    ax.set_xlabel("Residual Similarity to paired source transcript")
    ax.set_title("Candidate-level Residual Similarity", loc="left", fontweight="bold")
    clean_axis(ax, "x")


def build_figure(
    distance: pd.DataFrame,
    summary: pd.DataFrame,
    correlations: pd.DataFrame,
    coupling: pd.DataFrame,
    components: pd.DataFrame,
    residual_source: pd.DataFrame,
    residual_quantiles: pd.DataFrame,
) -> None:
    set_style()
    fig = plt.figure(figsize=(183 / 25.4, 158 / 25.4), facecolor="white")
    outer = fig.add_gridspec(
        3,
        2,
        left=0.075,
        right=0.988,
        top=0.905,
        bottom=0.075,
        height_ratios=[0.95, 1.05, 0.90],
        hspace=0.68,
        wspace=0.38,
    )

    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    ax_c = fig.add_subplot(outer[1, :])
    ax_d = fig.add_subplot(outer[2, :])

    draw_balance_distance_panel(ax_a, distance, summary, correlations)
    draw_coupling_panel(ax_b, coupling)
    draw_contrast_heatmap(ax_c, components)
    draw_residual_panel(ax_d, residual_source, residual_quantiles)

    panel_label(ax_a, "a", x=-0.18)
    panel_label(ax_b, "b", x=-0.14)
    panel_label(ax_c, "c", x=-0.055)
    panel_label(ax_d, "d", x=-0.075)

    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor=INK,
            markeredgewidth=0.4,
            markersize=4.4,
            label=DISPLAY[method],
        )
        for method in METHODS
    ]
    fig.legend(
        handles=method_handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.982),
        ncol=len(METHODS),
        frameon=False,
        columnspacing=0.7,
        handletextpad=0.3,
        borderaxespad=0,
    )

    save_bundle(fig, "Figure5_ablation_proxy_representation_audit_external_20260720")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Figure 5 and its aggregate source tables from the locked "
            "seed-level utility bridge and generated-data audit table."
        )
    )
    parser.add_argument("--utility-bridge-csv", required=True, type=Path)
    parser.add_argument("--augmentation-audit-csv", required=True, type=Path)
    parser.add_argument("--figure-dir", required=True, type=Path)
    parser.add_argument("--source-data-dir", required=True, type=Path)
    return parser.parse_args()


def configure(args: argparse.Namespace) -> None:
    global BRIDGE_PATH, AUG_LONG_PATH, FIG_DIR, SOURCE_DIR
    BRIDGE_PATH = args.utility_bridge_csv
    AUG_LONG_PATH = args.augmentation_audit_csv
    FIG_DIR = args.figure_dir
    SOURCE_DIR = args.source_data_dir


def main() -> None:
    configure(parse_args())
    bridge, candidates, _ = load_inputs()
    summary = method_summary(bridge)
    distance, components, correlations = compute_balance_distance(bridge)
    contrasts = matched_contrasts(summary, components)
    residual_source, residual_quantiles = residual_similarity_tables(candidates)
    coupling = conditional_coupling_summary(summary)

    export_tables(
        bridge=bridge,
        candidates=candidates,
        summary=summary,
        distance=distance,
        components=components,
        correlations=correlations,
        contrasts=contrasts,
        residual_source=residual_source,
        residual_quantiles=residual_quantiles,
        coupling=coupling,
    )
    build_figure(
        distance=distance,
        summary=summary,
        correlations=correlations,
        coupling=coupling,
        components=components,
        residual_source=residual_source,
        residual_quantiles=residual_quantiles,
    )
    print(f"Wrote ablation mechanism audit figures to {FIG_DIR}")
    print(f"Wrote ablation mechanism audit source data to {SOURCE_DIR}")


if __name__ == "__main__":
    main()
