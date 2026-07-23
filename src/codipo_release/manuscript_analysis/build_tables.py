#!/usr/bin/env python
"""Build main Tables 2 and 3 from the external-cohort statistical lock."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


STAT_DIR: Path
OUT_DIR: Path
N_BOOTSTRAP = 10000

DATASETS = ["Test", "Lu", "Pitt", "ExternalCohortAverage"]
DATASET_HEADERS = {
    "Test": "ADReSS validation",
    "Lu": "Lu",
    "Pitt": "Pitt",
    "ExternalCohortAverage": "External-cohort average",
}
METHOD_LABELS = {
    "BERT": "BERT",
    "EDA": "EDA",
    "CDA": "CDA",
    "ICL Direct": "ICL Direct",
    "ICL Rewrite": "ICL Rewrite",
    "ICL Imitation": "ICL Imitation",
    "w/o DPO, vanilla augmentation": "Vanilla",
    "w/o DPO, XYZ hard filtering": "Hard Filter",
    "CoDiPO": "CoDiPO",
    "CoDiPO w/o X": r"Without \(X\)",
    "CoDiPO w/o YZ": r"Without \(Y\) and \(Z\)",
    "w/o residual decomposition": r"Cosine sim. as \(X\)",
    "Cosine-only preference": "Cosine-only",
}
SOURCE_METHOD_LABELS = {
    method: label.replace(r"\(", "").replace(r"\)", "")
    for method, label in METHOD_LABELS.items()
}

TABLE2_METHODS = [
    "BERT",
    "EDA",
    "CDA",
    "ICL Direct",
    "ICL Rewrite",
    "ICL Imitation",
    "w/o DPO, vanilla augmentation",
    "w/o DPO, XYZ hard filtering",
    "CoDiPO",
]
TABLE3_METHODS = [
    "CoDiPO",
    "CoDiPO w/o X",
    "CoDiPO w/o YZ",
    "w/o residual decomposition",
    "Cosine-only preference",
    "w/o DPO, vanilla augmentation",
    "w/o DPO, XYZ hard filtering",
]


def fmt_pm(mean: float, sd: float, bold: bool = False) -> str:
    value = f"{100 * mean:.1f} \\(\\pm\\) {100 * sd:.1f}"
    return rf"\textbf{{{value}}}" if bold else value


def fmt_ci(low: float, high: float, bold: bool = False) -> str:
    value = f"{100 * low:.1f}--{100 * high:.1f}"
    return rf"\textbf{{{value}}}" if bold else value


def fmt_delta(value: float, low: float | None = None, high: float | None = None) -> str:
    if low is None or high is None:
        return f"{100 * value:.1f}"
    return f"{100 * low:.1f}--{100 * high:.1f}"


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed = pd.read_csv(STAT_DIR / "cohort_generation_level_method_metrics_external.csv")
    boot = pd.read_csv(STAT_DIR / f"cohort_bootstrap_method_metrics_n{N_BOOTSTRAP}.csv")
    contrasts = pd.read_csv(
        STAT_DIR / f"cohort_bootstrap_codipo_contrasts_n{N_BOOTSTRAP}.csv"
    )
    return seed, boot, contrasts


def make_lookup(frame: pd.DataFrame, value: str) -> dict[tuple[str, str, str], float]:
    return {
        (str(row.method_display), str(row.dataset), str(row.metric)): float(getattr(row, value))
        for row in frame.itertuples(index=False)
    }


def best_cells(seed: pd.DataFrame, methods: list[str], metrics: list[str]) -> set[tuple[str, str, str]]:
    selected = seed[
        seed["method_display"].isin(methods)
        & seed["dataset"].isin(DATASETS)
        & seed["metric"].isin(metrics)
    ]
    out: set[tuple[str, str, str]] = set()
    for (dataset, metric), group in selected.groupby(["dataset", "metric"]):
        maximum = group["mean"].max()
        for method in group.loc[np.isclose(group["mean"], maximum), "method_display"]:
            out.add((str(method), str(dataset), str(metric)))
    return out


def build_table2(seed: pd.DataFrame, boot: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    mean = make_lookup(seed, "mean")
    sd = make_lookup(seed, "sd")
    low = make_lookup(boot, "ci_low")
    high = make_lookup(boot, "ci_high")
    best = best_cells(seed, TABLE2_METHODS, ["accuracy", "f1"])

    source_rows: list[dict[str, object]] = []
    latex_rows: list[str] = []
    for method_index, method in enumerate(TABLE2_METHODS):
        mean_cells: list[str] = []
        ci_cells: list[str] = []
        for dataset in DATASETS:
            for metric in ["accuracy", "f1"]:
                key = (method, dataset, metric)
                is_best = key in best
                mean_cells.append(fmt_pm(mean[key], sd[key], is_best))
                ci_cells.append(fmt_ci(low[key], high[key], is_best))
                source_rows.append(
                    {
                        "table": "Table 2",
                        "method_display": method,
                        "method_label": METHOD_LABELS[method],
                        "dataset": dataset,
                        "cohort": DATASET_HEADERS[dataset],
                        "metric": metric,
                        "mean": mean[key],
                        "sd": sd[key],
                        "ci_low": low[key],
                        "ci_high": high[key],
                        "is_best": is_best,
                    }
                )
        latex_rows.append(
            rf"\multirow{{2}}{{*}}{{{METHOD_LABELS[method]}}} & Mean \(\pm\) SD & "
            + " & ".join(mean_cells)
            + r" \\"
        )
        latex_rows.append(" & 95\\% CI & " + " & ".join(ci_cells) + r" \\"
        )
        if method_index < len(TABLE2_METHODS) - 1:
            latex_rows.append(r"\addlinespace[0.25em]")

    latex = rf"""\begin{{table*}}[t]
\centering
\caption{{\textbf{{Primary fixed-threshold performance across augmentation families.}} Values are percentages. The official ADReSS test partition was used as the source-domain validation cohort and is reported separately. External-cohort averages are unweighted means of the Lu and Pitt cohort-level metrics. Each method is reported in two rows: mean \(\pm\) SD across the locked top-level seed groups and the percentile 95\% confidence interval from {N_BOOTSTRAP:,} diagnosis-stratified evaluation-unit bootstrap resamples. For transcript-generating methods, classifier seeds were averaged within generation seed before calculating the SD; BERT and CDA use their locked classifier or corruption seed groups.}}
\label{{tab:family_grouped_main}}
\begingroup
\footnotesize
\renewcommand{{\arraystretch}}{{1.08}}
\setlength{{\tabcolsep}}{{1.65pt}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}llcccccccc@{{}}}}
\toprule
\multirow{{2}}{{*}}{{Method}} & & \multicolumn{{2}}{{c}}{{ADReSS validation}} & \multicolumn{{2}}{{c}}{{Lu}} & \multicolumn{{2}}{{c}}{{Pitt}} & \multicolumn{{2}}{{c}}{{External-cohort average}} \\
\cmidrule(lr){{3-4}}\cmidrule(lr){{5-6}}\cmidrule(lr){{7-8}}\cmidrule(lr){{9-10}}
 & & Acc. & F1 & Acc. & F1 & Acc. & F1 & Acc. & F1 \\
\midrule
{chr(10).join(latex_rows)}
\bottomrule
\end{{tabular}}%
}}
\endgroup
\end{{table*}}
"""
    return pd.DataFrame(source_rows), latex


def build_table3(
    seed: pd.DataFrame,
    boot: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    mean = make_lookup(seed, "mean")
    sd = make_lookup(seed, "sd")
    low = make_lookup(boot, "ci_low")
    high = make_lookup(boot, "ci_high")
    contrast = contrasts[
        contrasts["dataset"].eq("ExternalCohortAverage")
        & contrasts["metric"].eq("f1")
    ].set_index("comparator")

    source_rows: list[dict[str, object]] = []
    contrast_rows: list[dict[str, object]] = []
    latex_rows: list[str] = []
    for method_index, method in enumerate(TABLE3_METHODS):
        mean_cells = [fmt_pm(mean[(method, dataset, "f1")], sd[(method, dataset, "f1")], method == "CoDiPO") for dataset in ["Test", "Lu", "Pitt"]]
        mean_cells.extend(
            [
                fmt_pm(mean[(method, "ExternalCohortAverage", "accuracy")], sd[(method, "ExternalCohortAverage", "accuracy")], method == "CoDiPO"),
                fmt_pm(mean[(method, "ExternalCohortAverage", "f1")], sd[(method, "ExternalCohortAverage", "f1")], method == "CoDiPO"),
            ]
        )
        ci_cells = [fmt_ci(low[(method, dataset, "f1")], high[(method, dataset, "f1")], method == "CoDiPO") for dataset in ["Test", "Lu", "Pitt"]]
        ci_cells.extend(
            [
                fmt_ci(low[(method, "ExternalCohortAverage", "accuracy")], high[(method, "ExternalCohortAverage", "accuracy")], method == "CoDiPO"),
                fmt_ci(low[(method, "ExternalCohortAverage", "f1")], high[(method, "ExternalCohortAverage", "f1")], method == "CoDiPO"),
            ]
        )

        if method == "CoDiPO":
            delta_mean = r"\textbf{0.0 (ref.)}"
            delta_ci = "--"
        else:
            row = contrast.loc[method]
            delta_mean = fmt_delta(float(row["observed_delta"]))
            delta_ci = fmt_delta(
                float(row["observed_delta"]),
                float(row["ci_low"]),
                float(row["ci_high"]),
            )
            contrast_rows.append(
                row.to_dict()
                | {
                    "comparator": method,
                    "comparator_label": SOURCE_METHOD_LABELS[method],
                }
            )

        latex_rows.append(
            rf"\multirow{{2}}{{*}}{{{METHOD_LABELS[method]}}} & Mean \(\pm\) SD & "
            + " & ".join(mean_cells)
            + f" & {delta_mean} "
            + r"\\"
        )
        latex_rows.append(
            " & 95\\% CI & " + " & ".join(ci_cells) + f" & {delta_ci} " + r"\\"
        )
        if method_index < len(TABLE3_METHODS) - 1:
            latex_rows.append(r"\addlinespace[0.25em]")

        for dataset in ["Test", "Lu", "Pitt"]:
            source_rows.append(
                {
                    "table": "Table 3",
                    "method_display": method,
                    "method_label": METHOD_LABELS[method],
                    "dataset": dataset,
                    "cohort": DATASET_HEADERS[dataset],
                    "metric": "f1",
                    "mean": mean[(method, dataset, "f1")],
                    "sd": sd[(method, dataset, "f1")],
                    "ci_low": low[(method, dataset, "f1")],
                    "ci_high": high[(method, dataset, "f1")],
                }
            )
        for metric in ["accuracy", "f1"]:
            source_rows.append(
                {
                    "table": "Table 3",
                    "method_display": method,
                    "method_label": METHOD_LABELS[method],
                    "dataset": "ExternalCohortAverage",
                    "cohort": DATASET_HEADERS["ExternalCohortAverage"],
                    "metric": metric,
                    "mean": mean[(method, "ExternalCohortAverage", metric)],
                    "sd": sd[(method, "ExternalCohortAverage", metric)],
                    "ci_low": low[(method, "ExternalCohortAverage", metric)],
                    "ci_high": high[(method, "ExternalCohortAverage", metric)],
                }
            )

    latex = rf"""\begin{{table*}}[t]
\centering
\caption{{\textbf{{Prompt/proxy controls and preference-signal ablation summary.}} Values are percentages except \(\Delta\)F1, which is reported in percentage points (pp). ADReSS validation denotes the official ADReSS test partition used for source-domain validation. External-cohort averages are unweighted means of Lu and Pitt. Each method is reported in two rows: mean \(\pm\) SD across the locked top-level seed groups and the percentile 95\% confidence interval from {N_BOOTSTRAP:,} diagnosis-stratified evaluation-unit bootstrap resamples. \(\Delta\)F1 denotes CoDiPO minus the corresponding comparator on the external-cohort average; its interval is obtained from the paired bootstrap, and positive values favor CoDiPO. Vanilla is the prompt-only control; Hard Filter is deterministic proxy-ranked top-2 selection without preference optimization.}}
\label{{tab:ablation_performance}}
\begingroup
\footnotesize
\renewcommand{{\arraystretch}}{{1.08}}
\setlength{{\tabcolsep}}{{2.0pt}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}llcccccc@{{}}}}
\toprule
\multirow{{2}}{{*}}{{Method}} & & \multicolumn{{3}}{{c}}{{Cohort F1}} & \multicolumn{{2}}{{c}}{{External-cohort average}} & \multirow{{2}}{{*}}{{\shortstack{{\(\Delta\)F1 vs\\CoDiPO (pp)}}}} \\
\cmidrule(lr){{3-5}}\cmidrule(lr){{6-7}}
 & & ADReSS validation & Lu & Pitt & Acc. & F1 & \\
\midrule
{chr(10).join(latex_rows)}
\bottomrule
\end{{tabular}}%
}}
\endgroup
\end{{table*}}
"""
    return pd.DataFrame(source_rows), pd.DataFrame(contrast_rows), latex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate manuscript Tables 2 and 3 from a statistical lock."
    )
    parser.add_argument("--statistics-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    return parser.parse_args()


def configure(args: argparse.Namespace) -> None:
    global STAT_DIR, OUT_DIR, N_BOOTSTRAP
    STAT_DIR = args.statistics_dir
    OUT_DIR = args.output_dir
    N_BOOTSTRAP = args.n_bootstrap
    if N_BOOTSTRAP < 1:
        raise ValueError("n_bootstrap must be positive.")


def main() -> None:
    configure(parse_args())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed, boot, contrasts = load_sources()
    table2_source, table2_tex = build_table2(seed, boot)
    table3_source, table3_contrasts, table3_tex = build_table3(seed, boot, contrasts)

    table2_source.to_csv(OUT_DIR / "table2_external_source_data.csv", index=False, encoding="utf-8-sig")
    table3_source.to_csv(OUT_DIR / "table3_external_source_data.csv", index=False, encoding="utf-8-sig")
    table3_contrasts.to_csv(OUT_DIR / "table3_external_f1_contrasts.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "table2_external_generated.tex").write_text(table2_tex, encoding="utf-8")
    (OUT_DIR / "table3_external_generated.tex").write_text(table3_tex, encoding="utf-8")
    print(f"Wrote external Tables 2 and 3 to {OUT_DIR}")


if __name__ == "__main__":
    main()
