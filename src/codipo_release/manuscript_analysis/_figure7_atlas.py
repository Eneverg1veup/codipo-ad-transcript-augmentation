#!/usr/bin/env python
"""Build class-aware PCA and t-SNE contamination atlases.

Opposite-class kNN contamination is always computed in the original 768-D
embedding space. PCA and t-SNE are visualization layers only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize


HERE = Path(__file__).resolve().parent
FINAL_DIR = HERE.parents[2]
AUDIT_DIR = FINAL_DIR / "distribution_analysis" / "embedding_audit_v3_with_proxy"
FIG_DIR = HERE / "figures"
SOURCE_DIR = HERE / "source_data"

CACHE_PATH = AUDIT_DIR / "bert_embedding_cache.npz"
SOURCE_PATH = AUDIT_DIR / "tables" / "source_train_loaded.csv"
AUG_PATH = AUDIT_DIR / "tables" / "aug_long_loaded_with_optional_proxy.csv"
PCA_COORD_PATH = AUDIT_DIR / "figures" / "embedding_pca_coordinates.csv"

METHODS = [
    "CoDiPO",
    "EDA",
    "ICL Direct",
    "ICL Imitation",
    "ICL Rewrite",
    "Hard filter",
]
DISPLAY_METHOD = {
    "CoDiPO": "CoDiPO",
    "EDA": "EDA",
    "ICL Direct": "ICL Direct",
    "ICL Imitation": "ICL Imitation",
    "ICL Rewrite": "ICL Rewrite",
    "Hard filter": "Hard filter",
}
LABEL_NAME = {0: "HC-source augmentation", 1: "AD-source augmentation"}
CONTAMINATION_NAME = {0: "HC-to-AD", 1: "AD-to-HC"}

INK = "#202124"
MUTED = "#687078"
LIGHT = "#DDE2E5"
HC_REFERENCE = "#8D969C"
AD_REFERENCE = "#30383D"

CONTAMINATION_CMAP = LinearSegmentedColormap.from_list(
    "contamination",
    ["#DCEBF0", "#72B6B2", "#E0B05A", "#D87969", "#A84458"],
    N=11,
)
CONTAMINATION_NORM = BoundaryNorm(
    np.linspace(-0.05, 1.05, 12),
    CONTAMINATION_CMAP.N,
    clip=True,
)


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 6.7,
            "axes.titlesize": 7.1,
            "axes.labelsize": 6.7,
            "axes.linewidth": 0.55,
            "axes.edgecolor": INK,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
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


def load_inputs() -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = [CACHE_PATH, SOURCE_PATH, AUG_PATH, PCA_COORD_PATH]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    cache_file = np.load(CACHE_PATH)
    cache = {key: cache_file[key] for key in cache_file.files}
    source = pd.read_csv(SOURCE_PATH)
    aug = pd.read_csv(AUG_PATH)
    pca_coords = pd.read_csv(PCA_COORD_PATH)

    if len(source) != len(cache["source_emb"]):
        raise ValueError("Source table and source embedding cache are misaligned.")
    if len(aug) != len(cache["aug_emb"]):
        raise ValueError("Augmentation table and augmentation cache are misaligned.")
    return cache, source, aug, pca_coords


def compute_sample_contamination(
    cache: dict[str, np.ndarray],
    source: pd.DataFrame,
    aug: pd.DataFrame,
    k_neighbors: int = 10,
) -> pd.DataFrame:
    nn = NearestNeighbors(
        n_neighbors=min(k_neighbors, len(source)),
        metric="cosine",
    )
    nn.fit(cache["source_emb"])
    neighbor_index = nn.kneighbors(cache["aug_emb"], return_distance=False)
    source_labels = source["source_label"].astype(int).to_numpy()
    aug_labels = aug["source_label"].astype(int).to_numpy()
    neighbor_labels = source_labels[neighbor_index]

    out = aug.copy()
    out["aug_index"] = np.arange(len(out))
    out["knn_opposite_rate"] = (
        neighbor_labels != aug_labels[:, np.newaxis]
    ).mean(axis=1)
    return out


def reproduce_pca_selection(aug: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(2024)
    selected_indices: list[int] = []
    for _, group in aug.groupby("method"):
        indices = group.index.to_numpy()
        if len(indices) > 800:
            indices = rng.choice(indices, size=800, replace=False)
        selected_indices.extend(indices.tolist())
    return aug.loc[selected_indices].copy().reset_index(drop=True)


def attach_existing_pca(
    selected: pd.DataFrame,
    source: pd.DataFrame,
    coords: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aug_coords = coords[coords["kind"].eq("augmented")].reset_index(drop=True)
    source_coords = coords[coords["kind"].eq("source_train")].reset_index(drop=True)

    if len(selected) != len(aug_coords):
        raise ValueError("Reproduced PCA sample count does not match saved coordinates.")
    if not np.array_equal(
        selected["method"].astype(str).to_numpy(),
        aug_coords["method"].astype(str).to_numpy(),
    ):
        raise ValueError("Reproduced PCA method order does not match saved coordinates.")
    if len(source) != len(source_coords):
        raise ValueError("PCA source coordinates do not match source table.")

    aug_pca = selected.copy()
    aug_pca["dim1"] = aug_coords["dim1"].to_numpy(dtype=float)
    aug_pca["dim2"] = aug_coords["dim2"].to_numpy(dtype=float)

    source_pca = source.copy()
    source_pca["dim1"] = source_coords["dim1"].to_numpy(dtype=float)
    source_pca["dim2"] = source_coords["dim2"].to_numpy(dtype=float)
    return aug_pca, source_pca


def compute_existing_pca_variance(
    cache: dict[str, np.ndarray],
    selected: pd.DataFrame,
) -> np.ndarray:
    selected_index = selected["aug_index"].astype(int).to_numpy()
    embeddings = np.vstack(
        [
            cache["source_emb"],
            cache["aug_emb"][selected_index],
            cache["eval_emb"],
        ]
    )
    scaled = StandardScaler(with_mean=True, with_std=True).fit_transform(embeddings)
    return PCA(n_components=2, random_state=2024).fit(
        scaled
    ).explained_variance_ratio_


def compute_joint_tsne(
    cache: dict[str, np.ndarray],
    selected: pd.DataFrame,
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shown = selected[selected["method"].isin(METHODS)].copy().reset_index(drop=True)
    aug_index = shown["aug_index"].astype(int).to_numpy()

    embeddings = np.vstack([cache["source_emb"], cache["aug_emb"][aug_index]])
    embeddings = normalize(embeddings, norm="l2")
    reduced = PCA(n_components=50, random_state=2024).fit_transform(embeddings)
    coordinates = TSNE(
        n_components=2,
        perplexity=40,
        early_exaggeration=12,
        learning_rate="auto",
        n_iter=1500,
        init="pca",
        random_state=2024,
        method="barnes_hut",
        angle=0.5,
        verbose=0,
    ).fit_transform(reduced)

    source_tsne = source.copy()
    source_tsne["dim1"] = coordinates[: len(source), 0]
    source_tsne["dim2"] = coordinates[: len(source), 1]

    aug_tsne = shown.copy()
    aug_tsne["dim1"] = coordinates[len(source) :, 0]
    aug_tsne["dim2"] = coordinates[len(source) :, 1]
    return aug_tsne, source_tsne


def projection_limits(
    aug: pd.DataFrame,
    source: pd.DataFrame,
) -> tuple[tuple[float, float], tuple[float, float]]:
    x = np.concatenate([aug["dim1"].to_numpy(), source["dim1"].to_numpy()])
    y = np.concatenate([aug["dim2"].to_numpy(), source["dim2"].to_numpy()])
    x_pad = max(np.ptp(x) * 0.035, 0.1)
    y_pad = max(np.ptp(y) * 0.035, 0.1)
    return (float(x.min() - x_pad), float(x.max() + x_pad)), (
        float(y.min() - y_pad),
        float(y.max() + y_pad),
    )


def style_projection_axis(
    ax: plt.Axes,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    show_x: bool,
    show_y: bool,
    x_label: str,
    y_label: str,
) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_facecolor("#FBFCFC")
    for spine in ax.spines.values():
        spine.set_color("#AEB6BA")
        spine.set_linewidth(0.42)
    ax.tick_params(length=2.0, pad=1.5, color="#7E878C")
    if not show_x:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    else:
        ax.set_xlabel(x_label)
    if not show_y:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    else:
        ax.set_ylabel(y_label)


def draw_reference_points(ax: plt.Axes, source: pd.DataFrame) -> None:
    hc = source[source["source_label"].astype(int).eq(0)]
    ad = source[source["source_label"].astype(int).eq(1)]
    ax.scatter(
        hc["dim1"],
        hc["dim2"],
        s=8.0,
        marker="o",
        facecolors="none",
        edgecolors=HC_REFERENCE,
        linewidths=0.38,
        alpha=0.34,
        zorder=1,
    )
    ax.scatter(
        ad["dim1"],
        ad["dim2"],
        s=8.0,
        marker="x",
        c=AD_REFERENCE,
        linewidths=0.42,
        alpha=0.27,
        zorder=1,
    )


def build_atlas(
    aug: pd.DataFrame,
    source: pd.DataFrame,
    projection: str,
    stem: str,
    x_label: str,
    y_label: str,
) -> None:
    shown = aug[aug["method"].isin(METHODS)].copy()
    xlim, ylim = projection_limits(shown, source)

    fig = plt.figure(figsize=(183 / 25.4, 101 / 25.4), facecolor="white")
    grid = fig.add_gridspec(
        2,
        6,
        left=0.105,
        right=0.987,
        top=0.835,
        bottom=0.205,
        wspace=0.055,
        hspace=0.075,
    )

    axes = np.empty((2, 6), dtype=object)
    scatter = None
    for row, label in enumerate([0, 1]):
        for col, method in enumerate(METHODS):
            ax = fig.add_subplot(grid[row, col])
            axes[row, col] = ax
            draw_reference_points(ax, source)
            panel = shown[
                shown["method"].eq(method)
                & shown["source_label"].astype(int).eq(label)
            ]
            scatter = ax.scatter(
                panel["dim1"],
                panel["dim2"],
                c=panel["knn_opposite_rate"],
                cmap=CONTAMINATION_CMAP,
                norm=CONTAMINATION_NORM,
                s=6.5,
                marker="o",
                edgecolors="none",
                alpha=0.70,
                zorder=3,
            )
            mean_rate = panel["knn_opposite_rate"].mean() * 100
            ax.text(
                0.965,
                0.965,
                f"mean {mean_rate:.1f}%",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=5.4,
                fontweight="bold",
                color=INK,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                    "pad": 0.7,
                },
                zorder=5,
            )
            style_projection_axis(
                ax,
                xlim,
                ylim,
                show_x=row == 1,
                show_y=col == 0,
                x_label=x_label,
                y_label=y_label,
            )
            if row == 0:
                ax.set_title(DISPLAY_METHOD[method], fontweight="bold", pad=3.0)

    fig.text(
        0.015,
        0.655,
        "HC source\n(HC-to-AD)",
        ha="left",
        va="center",
        rotation=90,
        fontsize=6.4,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.015,
        0.375,
        "AD source\n(AD-to-HC)",
        ha="left",
        va="center",
        rotation=90,
        fontsize=6.4,
        fontweight="bold",
        color=INK,
    )
    fig.suptitle(
        "Opposite-class contamination occupies diagnosis-specific representation regions",
        x=0.105,
        y=0.955,
        ha="left",
        fontsize=8.5,
        fontweight="bold",
        color=INK,
    )

    reference_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=HC_REFERENCE,
            markeredgewidth=0.6,
            markersize=4.0,
            label="HC training reference",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="none",
            color=AD_REFERENCE,
            markeredgewidth=0.7,
            markersize=4.0,
            label="AD training reference",
        ),
    ]
    fig.legend(
        handles=reference_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.925),
        ncol=2,
        columnspacing=0.9,
        handletextpad=0.35,
    )

    if scatter is None:
        raise RuntimeError("Atlas did not draw any augmentation points.")
    color_ax = fig.add_axes([0.34, 0.083, 0.34, 0.022])
    colorbar = fig.colorbar(scatter, cax=color_ax, orientation="horizontal")
    colorbar.set_ticks(np.linspace(0, 1, 6))
    colorbar.set_ticklabels(["0", "20", "40", "60", "80", "100"])
    colorbar.ax.tick_params(labelsize=5.7, length=2.0, pad=1.2)
    colorbar.outline.set_linewidth(0.45)
    colorbar.set_label(
        "Opposite-class kNN contamination in original representation space (%)",
        fontsize=6.2,
        labelpad=2.2,
    )

    save_bundle(fig, stem)
    plt.close(fig)


def summarize_contamination(aug: pd.DataFrame) -> pd.DataFrame:
    shown = aug[aug["method"].isin(METHODS)].copy()
    rows: list[dict[str, float | int | str]] = []
    for method in METHODS:
        for label in [0, 1]:
            group = shown[
                shown["method"].eq(method)
                & shown["source_label"].astype(int).eq(label)
            ]
            rows.append(
                {
                    "method": method,
                    "source_label": label,
                    "source_diagnosis": "HC" if label == 0 else "AD",
                    "contamination_direction": CONTAMINATION_NAME[label],
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


def projection_visibility_statistics(
    aug: pd.DataFrame,
    projection: str,
) -> pd.DataFrame:
    shown = aug[aug["method"].isin(METHODS)].copy()
    rows: list[dict[str, float | int | str]] = []
    for label in [0, 1]:
        group = shown[shown["source_label"].astype(int).eq(label)].copy()
        x = group[["dim1", "dim2"]].to_numpy(dtype=float)
        y = group["knn_opposite_rate"].to_numpy(dtype=float)
        seeds = group["seed"].to_numpy()
        n_splits = len(np.unique(seeds))
        cv = GroupKFold(n_splits=n_splits)

        local_prediction = cross_val_predict(
            KNeighborsRegressor(n_neighbors=40, weights="distance"),
            x,
            y,
            groups=seeds,
            cv=cv,
        )
        high = (y >= 0.5).astype(int)
        high_prediction = cross_val_predict(
            LogisticRegression(max_iter=2000),
            x,
            high,
            groups=seeds,
            cv=cv,
            method="predict_proba",
        )[:, 1]
        rows.append(
            {
                "projection": projection,
                "source_label": label,
                "source_diagnosis": "HC" if label == 0 else "AD",
                "n": len(group),
                "spearman_dim1": spearmanr(group["dim1"], y).statistic,
                "spearman_dim2": spearmanr(group["dim2"], y).statistic,
                "local_prediction_spearman": spearmanr(y, local_prediction).statistic,
                "local_prediction_r2": r2_score(y, local_prediction),
                "local_prediction_mae": mean_absolute_error(y, local_prediction),
                "high_contamination_auc": roc_auc_score(high, high_prediction),
            }
        )
    return pd.DataFrame(rows)


def export_source_data(
    pca_aug: pd.DataFrame,
    pca_source: pd.DataFrame,
    tsne_aug: pd.DataFrame,
    tsne_source: pd.DataFrame,
) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    aug_columns = [
        "aug_index",
        "method",
        "seed",
        "source_id",
        "source_label",
        "aug_round",
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
    pca_aug[pca_aug["method"].isin(METHODS)][aug_columns].to_csv(
        SOURCE_DIR / "pca_contamination_atlas_source.csv",
        index=False,
    )
    pca_source[source_columns].to_csv(
        SOURCE_DIR / "pca_training_reference_source.csv",
        index=False,
    )
    tsne_aug[aug_columns].to_csv(
        SOURCE_DIR / "tsne_contamination_atlas_source.csv",
        index=False,
    )
    tsne_source[source_columns].to_csv(
        SOURCE_DIR / "tsne_training_reference_source.csv",
        index=False,
    )

    summary = summarize_contamination(pca_aug)
    summary.to_csv(
        SOURCE_DIR / "method_class_contamination_summary.csv",
        index=False,
    )
    visibility = pd.concat(
        [
            projection_visibility_statistics(pca_aug, "PCA"),
            projection_visibility_statistics(tsne_aug, "t-SNE"),
        ],
        ignore_index=True,
    )
    visibility.to_csv(
        SOURCE_DIR / "projection_visibility_statistics.csv",
        index=False,
    )


def main() -> None:
    set_style()
    cache, source, aug, pca_coords = load_inputs()
    aug = compute_sample_contamination(cache, source, aug)
    selected = reproduce_pca_selection(aug)
    pca_aug, pca_source = attach_existing_pca(selected, source, pca_coords)
    pca_variance = compute_existing_pca_variance(cache, selected)
    tsne_aug, tsne_source = compute_joint_tsne(cache, selected, source)

    build_atlas(
        pca_aug,
        pca_source,
        projection="PCA",
        stem="fig_representation_contamination_pca_atlas",
        x_label=f"PC1 ({100 * pca_variance[0]:.1f}%)",
        y_label=f"PC2 ({100 * pca_variance[1]:.1f}%)",
    )
    build_atlas(
        tsne_aug,
        tsne_source,
        projection="t-SNE",
        stem="fig_representation_contamination_tsne_atlas",
        x_label="t-SNE 1",
        y_label="t-SNE 2",
    )
    export_source_data(pca_aug, pca_source, tsne_aug, tsne_source)

    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote source data to {SOURCE_DIR}")



