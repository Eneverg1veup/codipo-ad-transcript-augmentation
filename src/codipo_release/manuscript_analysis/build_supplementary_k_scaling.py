import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


MAIN_SOURCE: Path
COHORT_SOURCE: Path
OUT_DIR: Path

METHOD_ORDER = ["xyz", "direct", "imitation", "rewrite"]
METHOD_LABELS = {
    "xyz": "CoDiPO",
    "direct": "ICL Direct",
    "imitation": "ICL Imitation",
    "rewrite": "ICL Rewrite",
}
METHOD_COLORS = {
    "xyz": "#245A73",
    "direct": "#D57C6C",
    "imitation": "#6E9C79",
    "rewrite": "#C49A3A",
}
DATASET_ORDER = ["test", "pitt", "lu"]
DATASET_LABELS = {"test": "ADReSS validation", "pitt": "Pitt", "lu": "Lu"}


def main() -> None:
    global MAIN_SOURCE, COHORT_SOURCE, OUT_DIR
    parser = argparse.ArgumentParser(
        description="Build the Supplementary K-scaling sensitivity figure."
    )
    parser.add_argument("--overall-k-scaling-csv", required=True, type=Path)
    parser.add_argument("--cohort-k-scaling-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    MAIN_SOURCE = args.overall_k_scaling_csv
    COHORT_SOURCE = args.cohort_k_scaling_csv
    OUT_DIR = args.output_dir
    overall = pd.read_csv(MAIN_SOURCE)
    cohort = pd.read_csv(COHORT_SOURCE)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.0),
        gridspec_kw={"width_ratios": [1.2, 1.0], "wspace": 0.42},
    )

    for method in METHOD_ORDER:
        sub = overall.loc[overall["method"] == method].sort_values("run_num")
        x = sub["run_num"].to_numpy(float)
        mean = sub["mean_percent"].to_numpy(float)
        sd = sub["sd_percent"].to_numpy(float)
        color = METHOD_COLORS[method]
        ax_a.plot(
            x,
            mean,
            marker="o",
            markersize=3.2,
            linewidth=1.5,
            color=color,
            label=METHOD_LABELS[method],
        )
        ax_a.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.10, linewidth=0)

    ax_a.set_title("External-cohort F1 across augmentation budgets", loc="left", fontweight="bold")
    ax_a.set_xlabel("Generated samples per source, K")
    ax_a.set_ylabel("External-cohort average F1 (%)")
    ax_a.set_xticks(range(1, 11))
    ax_a.grid(axis="y", color="#E5EAED", linewidth=0.6)
    ax_a.legend(frameon=False, ncol=2, loc="lower right")
    ax_a.text(-0.12, 1.05, "a", transform=ax_a.transAxes, fontweight="bold", fontsize=9)

    rows = []
    for (k, dataset), group in cohort.groupby(["run_num", "dataset"], observed=True):
        codipo = group.loc[group["method"] == "xyz", "score_mean"]
        icl = group.loc[group["method"].isin(["direct", "imitation", "rewrite"]), "score_mean"]
        if not codipo.empty and not icl.empty:
            rows.append(
                {
                    "K": int(k),
                    "dataset": dataset,
                    "delta": 100.0 * (float(codipo.iloc[0]) - float(icl.max())),
                }
            )
    delta = pd.DataFrame(rows)
    matrix = (
        delta.pivot(index="dataset", columns="K", values="delta")
        .reindex(DATASET_ORDER)
        .reindex(columns=range(1, 11))
    )
    vmax = float(np.nanmax(np.abs(matrix.to_numpy())))
    cmap = LinearSegmentedColormap.from_list("delta", ["#C96B5C", "#F7F7F7", "#2F5D7E"])
    ax_b.imshow(matrix.to_numpy(), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax_b.set_title("CoDiPO F1 difference from best ICL", loc="left", fontweight="bold")
    ax_b.set_xlabel("Generated samples per source, K")
    ax_b.set_xticks(np.arange(10), labels=range(1, 11))
    ax_b.set_yticks(np.arange(3), labels=[DATASET_LABELS[x] for x in DATASET_ORDER])
    ax_b.tick_params(length=0)
    for spine in ax_b.spines.values():
        spine.set_visible(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            if pd.notna(value):
                color = "white" if abs(value) > 0.48 * vmax else "#303538"
                ax_b.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=6.3, color=color)
    ax_b.text(-0.15, 1.05, "b", transform=ax_b.transAxes, fontweight="bold", fontsize=9)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "Figure_S1_k_scaling_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "Figure_S1_k_scaling_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
