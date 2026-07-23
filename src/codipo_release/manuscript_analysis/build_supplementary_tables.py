"""Build submission-facing SI tables from the locked analysis sources.

The generated LaTeX contains scientific labels only. Local paths, notebook
names and artifact-version labels are intentionally kept out of the PDF.
"""

import argparse
from pathlib import Path

import pandas as pd


FULL_METRICS: Path
K_SCALING: Path
SUBGROUP_DIR: Path
YZ_DIR: Path
FIGURE_LOCK: Path
OUTPUT_FILE: Path


METHOD_ORDER = [
    "BERT",
    "EDA",
    "CDA",
    "ICL Direct",
    "ICL Rewrite",
    "ICL Imitation",
    "Vanilla",
    "Hard Filter",
    "CoDiPO",
    "w/o X",
    "w/o YZ",
    "Cosine sim. as X",
    "Cosine-only",
]
METHOD_NORMALIZATION = {
    "CoDiPO w/o X": "w/o X",
    "CoDiPO w/o YZ": "w/o YZ",
    "w/o residual decomposition": "Cosine sim. as X",
    "Cosine-only preference": "Cosine-only",
    "w/o DPO, vanilla augmentation": "Vanilla",
    "w/o DPO, XYZ hard filtering": "Hard Filter",
}
DATASET_ORDER = ["ADReSS validation", "Lu", "Pitt", "External-cohort average"]
METRIC_ORDER = [
    "accuracy",
    "precision",
    "sensitivity",
    "f1",
    "specificity",
    "auroc",
    "auprc",
]
METRIC_LABEL = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "sensitivity": "Sensitivity",
    "f1": "F1",
    "specificity": "Specificity",
    "auroc": "AUROC",
    "auprc": "AUPRC",
}


def esc(value: object) -> str:
    text = str(value)
    text = text.replace(">=75 years", r"$\geq75$ years")
    text = text.replace("<65 years", r"$<65$ years")
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def pm(mean: float, sd: float, scale: float = 100.0) -> str:
    if pd.isna(mean):
        return "--"
    return f"{mean * scale:.2f} $\\pm$ {sd * scale:.2f}"


def ci(value: float, low: float, high: float, scale: float = 100.0) -> str:
    if pd.isna(value):
        return "--"
    return f"{value * scale:.1f} [{low * scale:.1f}, {high * scale:.1f}]"


def full_operating_metrics() -> str:
    frame = pd.read_csv(FULL_METRICS)
    frame["method_display"] = frame["method_display"].replace(METHOD_NORMALIZATION)
    frame = frame[frame["metric"].isin(METRIC_ORDER)].copy()
    frame["method_display"] = pd.Categorical(
        frame["method_display"], METHOD_ORDER, ordered=True
    )
    frame["cohort"] = pd.Categorical(frame["cohort"], DATASET_ORDER, ordered=True)
    wide_mean = frame.pivot_table(
        index=["method_display", "cohort"], columns="metric", values="mean", observed=False
    )
    wide_sd = frame.pivot_table(
        index=["method_display", "cohort"], columns="metric", values="sd", observed=False
    )
    rows = []
    method_label = {
        "Cosine sim. as X": r"Cosine sim. as $X$",
        "w/o YZ": r"w/o $YZ$",
        "w/o X": r"w/o $X$",
    }
    cohort_label = {cohort: cohort for cohort in DATASET_ORDER}
    for method in METHOD_ORDER:
        for cohort in DATASET_ORDER:
            key = (method, cohort)
            if key not in wide_mean.index:
                continue
            cells = [pm(wide_mean.loc[key, m], wide_sd.loc[key, m]) for m in METRIC_ORDER]
            rows.append(
                f"{method_label.get(method, esc(method))} & {cohort_label.get(cohort, esc(cohort))} & "
                + " & ".join(cells) + r" \\"
            )
    header = " & ".join(METRIC_LABEL[m] for m in METRIC_ORDER)
    return rf"""
\begin{{landscape}}
\begingroup\scriptsize
\setlength{{\tabcolsep}}{{2pt}}
\begin{{longtable}}{{P{{2.8cm}}P{{2.0cm}}rrrrrrr}}
\caption{{\textbf{{Complete cohort-level operating metrics.}} Values are percentages (mean $\pm$ SD) under the locked seed aggregation. ADReSS validation is reported separately as source-domain validation; the external-cohort average is the unweighted mean of Lu and Pitt.}}\label{{tab:supp_full_metrics}}\\
\toprule
Method & Cohort & {header} \\
\midrule
\endfirsthead
\toprule
Method & Cohort & {header} \\
\midrule
\endhead
{chr(10).join(rows)}
\bottomrule
\end{{longtable}}
\endgroup
\end{{landscape}}
"""


def probability_reliability() -> str:
    frame = pd.read_csv(FULL_METRICS)
    frame["method_display"] = frame["method_display"].replace(METHOD_NORMALIZATION)
    frame = frame[frame["metric"].isin(["brier", "ece"])].copy()
    means = frame.pivot_table(
        index=["method_display", "cohort"], columns="metric", values="mean"
    )
    sds = frame.pivot_table(index=["method_display", "cohort"], columns="metric", values="sd")
    rows = []
    cohort_label = {cohort: cohort for cohort in DATASET_ORDER}
    for method in METHOD_ORDER:
        for cohort in DATASET_ORDER:
            key = (method, cohort)
            if key not in means.index:
                continue
            rows.append(
                f"{esc(method)} & {cohort_label.get(cohort, esc(cohort))} & {pm(means.loc[key, 'brier'], sds.loc[key, 'brier'], 1)} & "
                f"{pm(means.loc[key, 'ece'], sds.loc[key, 'ece'], 1)} " + r"\\"
            )
    return rf"""
\begingroup\scriptsize
\begin{{longtable}}{{P{{0.23\textwidth}}P{{0.20\textwidth}}cc}}
\caption{{\textbf{{Cohort-specific and external-cohort probability reliability.}} Brier score and expected calibration error (ECE) are mean $\pm$ SD. ADReSS validation is reported separately; the external-cohort average is the unweighted mean of Lu and Pitt. ECE used ten equal-frequency bins. Lower values are better.}}\label{{tab:supp_reliability}}\\
\toprule
Method & Cohort & Brier score & ECE \\
\midrule
\endfirsthead
\toprule
Method & Cohort & Brier score & ECE \\
\midrule
\endhead
{chr(10).join(rows)}
\bottomrule
\end{{longtable}}
\endgroup
"""


def k_scaling() -> str:
    frame = pd.read_csv(K_SCALING)
    name = {"xyz": "CoDiPO", "direct": "ICL Direct", "rewrite": "ICL Rewrite", "imitation": "ICL Imitation"}
    frame["method"] = frame["method"].map(name)
    mean = frame.pivot(index="run_num", columns="method", values="macro_score_mean")
    sd = frame.pivot(index="run_num", columns="method", values="macro_score_std")
    methods = ["CoDiPO", "ICL Direct", "ICL Rewrite", "ICL Imitation"]
    rows = []
    for k in sorted(mean.index):
        rows.append(str(k) + " & " + " & ".join(pm(mean.loc[k, m], sd.loc[k, m]) for m in methods) + r" \\")
    return rf"""
\begin{{table*}}[htbp]
\centering
\caption{{\textbf{{F1 sensitivity across matched augmentation budgets.}} Values are external-cohort average F1 percentages (mean $\pm$ SD), defined as the unweighted Lu--Pitt mean across 25 generation-seed--classifier-seed evaluations. This descriptive analysis was not used to select the primary $K=2$.}}
\label{{tab:supp_k_scaling}}
\scriptsize
\begin{{tabular}}{{rcccc}}
\toprule
$K$ & CoDiPO & ICL Direct & ICL Rewrite & ICL Imitation \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""


def subgroup_tables() -> str:
    definitions = pd.read_csv(SUBGROUP_DIR / "subgroup_definitions_used.csv")
    keep_types = ["overall", "sex", "age_group", "text_length_group"]
    definitions = definitions[
        definitions["dataset"].isin(["Test", "Pitt", "Lu"])
        & definitions["subgroup_type"].isin(keep_types)
        & (definitions["analysis_priority"] == "primary")
    ].copy()
    type_label = {
        "overall": "Overall",
        "sex": "Sex",
        "age_group": "Age",
        "text_length_group": "Transcript length",
    }
    dataset_label = {"Test": "ADReSS validation", "Pitt": "Pitt", "Lu": "Lu"}
    definitions["type_label"] = definitions["subgroup_type"].map(type_label)
    rows_support = [
        f"{dataset_label[r.dataset]} & {r.type_label} & {esc(r.subgroup)} & {int(r.n_participants)} & {int(r.n_hc)} & {int(r.n_ad)} " + r"\\"
        for r in definitions.itertuples()
    ]

    metrics = pd.read_csv(SUBGROUP_DIR / "observed_subgroup_metrics_summary.csv")
    metrics = metrics[
        metrics["dataset"].isin(["Test", "Pitt", "Lu"])
        & metrics["subgroup_type"].isin(keep_types)
        & (metrics["analysis_priority"] == "primary")
        & (metrics["method_display"] == "CoDiPO")
    ].copy()
    contrasts = pd.read_csv(SUBGROUP_DIR / "paired_bootstrap_subgroup_contrasts.csv")
    contrasts = contrasts[
        contrasts["dataset"].isin(["Test", "Pitt", "Lu"])
        & contrasts["subgroup_type"].isin(keep_types)
        & (contrasts["analysis_priority"] == "primary")
        & contrasts["metric"].isin(["balanced_accuracy", "f1"])
    ].copy()
    key_cols = ["dataset", "subgroup_type", "subgroup"]
    contrast_wide = contrasts.pivot_table(
        index=key_cols,
        columns=["comparator", "metric"],
        values=["observed_delta", "ci_low", "ci_high"],
        aggfunc="first",
    )
    rows_metric = []
    for row in metrics.itertuples():
        key = (row.dataset, row.subgroup_type, row.subgroup)
        def delta(comp: str, metric: str) -> str:
            try:
                return ci(
                    contrast_wide.loc[key, ("observed_delta", comp, metric)],
                    contrast_wide.loc[key, ("ci_low", comp, metric)],
                    contrast_wide.loc[key, ("ci_high", comp, metric)],
                )
            except KeyError:
                return "--"
        rows_metric.append(
            f"{dataset_label[row.dataset]} & {esc(row.subgroup)} & "
            f"{ci(row.balanced_accuracy, row.balanced_accuracy_ci_low, row.balanced_accuracy_ci_high)} & "
            f"{ci(row.f1, row.f1_ci_low, row.f1_ci_high)} & "
            f"{delta('Baseline mean', 'f1')} & {delta('Ablation mean', 'f1')} " + r"\\"
        )
    return rf"""
\begingroup\scriptsize
\begin{{longtable}}{{P{{0.14\textwidth}}P{{0.16\textwidth}}P{{0.19\textwidth}}rrr}}
\caption{{\textbf{{Participant support for the primary subgroup sensitivity analyses.}} Age strata were $<65$, 65--74 and $\geq75$ years. Transcript length used pooled tertiles across the retained training and evaluation cohorts: short $\leq76$ words, intermediate 77--113 words and long $>113$ words.}}\label{{tab:supp_subgroup_support}}\\
\toprule
Cohort & Dimension & Stratum & Participants & HC & AD \\
\midrule
\endfirsthead
\toprule
Cohort & Dimension & Stratum & Participants & HC & AD \\
\midrule
\endhead
{chr(10).join(rows_support)}
\bottomrule
\end{{longtable}}
\endgroup

\begin{{landscape}}
\begingroup\scriptsize
\begin{{longtable}}{{P{{0.12\textwidth}}P{{0.16\textwidth}}P{{0.18\textwidth}}P{{0.18\textwidth}}P{{0.17\textwidth}}P{{0.17\textwidth}}}}
\caption{{\textbf{{Participant-aggregated CoDiPO subgroup estimates and family-level F1 contrasts.}} Entries are percentage point estimates with 95\% participant-bootstrap intervals from $n=1{{,}}000$ resamples. Contrasts are CoDiPO minus the indicated family mean.}}\label{{tab:supp_subgroup_metrics}}\\
\toprule
Cohort & Stratum & Balanced accuracy & F1 & $\Delta$F1 vs baseline family & $\Delta$F1 vs control family \\
\midrule
\endfirsthead
\toprule
Cohort & Stratum & Balanced accuracy & F1 & $\Delta$F1 vs baseline family & $\Delta$F1 vs control family \\
\midrule
\endhead
{chr(10).join(rows_metric)}
\bottomrule
\end{{longtable}}
\endgroup
\end{{landscape}}
"""


def yz_cells() -> str:
    frame = pd.read_csv(YZ_DIR / "Figure_YZ2_performance_source_data.csv")
    dataset_label = {"Test": "ADReSS validation", "Pitt": "Pitt", "Lu": "Lu"}
    rows = []
    for row in frame.itertuples():
        status = "Expl." if int(row.n_participants) < 10 else "Eval."
        rows.append(
            f"{dataset_label[row.dataset]} & {esc(row.subgroup)} & {int(row.n_participants)} & {int(row.n_hc)} & {int(row.n_ad)} & "
            f"{ci(row.balanced_accuracy, row.balanced_accuracy_ci_low, row.balanced_accuracy_ci_high)} & "
            f"{ci(row.f1, row.f1_ci_low, row.f1_ci_high)} & {row.sensitivity * 100:.1f} & {row.specificity * 100:.1f} & {status} " + r"\\"
        )
    return rf"""
\clearpage
\begin{{landscape}}
\begingroup
\scriptsize
\setlength{{\tabcolsep}}{{2pt}}
\begin{{longtable}}{{llrrrccrrl}}
\caption{{\textbf{{CoDiPO performance within cohort-specific joint $Y$--$Z$ tertile cells.}} Cells with fewer than five participants were not estimated and are absent; cells with five to nine participants are marked exploratory. Intervals used $n=1{{,}}000$ participant-bootstrap resamples.}}\label{{tab:supp_yz_cells}}\\
\toprule
Cohort & Cell & $n$ & HC & AD & Balanced accuracy & F1 & Sens. & Spec. & Status \\
\midrule
\endfirsthead
\toprule
Cohort & Cell & $n$ & HC & AD & Balanced accuracy & F1 & Sens. & Spec. & Status \\
\midrule
\endhead
{chr(10).join(rows)}
\bottomrule
\end{{longtable}}
\endgroup
\end{{landscape}}
\clearpage
"""


def generated_audits() -> str:
    base = FIGURE_LOCK / "figure5_proxy_quality_use_combined_20260706" / "figure5_baseline_proxy_audit_summary.csv"
    control = FIGURE_LOCK / "figure5_proxy_quality_use_combined_20260706" / "figure5_ablation_proxy_audit_summary.csv"
    source = pd.concat([pd.read_csv(base), pd.read_csv(control)], ignore_index=True)
    source = source.drop_duplicates("method", keep="first")
    knn_base = pd.read_csv(
        FIGURE_LOCK / "figure7_baseline_ablation_joint_20260706" / "panel_d_baseline_knn_f1_summary.csv"
    )[["method_display", "contamination_mean"]]
    knn_control = pd.read_csv(
        FIGURE_LOCK / "figure7_baseline_ablation_joint_20260706" / "panel_h_ablation_knn_f1_summary.csv"
    )[["method", "overall_contamination_mean"]].rename(
        columns={"method": "method_display", "overall_contamination_mean": "contamination_mean"}
    )
    knn_control["contamination_mean"] *= 100.0
    knn = pd.concat([knn_base, knn_control], ignore_index=True).drop_duplicates("method_display")
    method_map = {"Hard filter": "Hard Filter", "Cosine sim. as X": "Cosine sim. as X", "w/o Y/Z": "w/o YZ", "w/o X": "w/o X"}
    source["display"] = source["method"].replace(method_map)
    knn["display"] = knn["method_display"].replace(method_map)
    merged = source.merge(knn[["display", "contamination_mean"]], on="display", how="outer")
    rows = []
    for row in merged.sort_values("display").itertuples():
        def f(name: str) -> str:
            value = getattr(row, name, float("nan"))
            return "NA" if pd.isna(value) else f"{value:.1f}"
        rows.append(
            f"{esc(row.display)} & {f('proxy_yz_pass_rate_mean')} & {f('proxy_x_pass_rate_mean')} & "
            f"{f('proxy_joint_pass_rate_mean')} & {f('conditional_x_pass_within_yz_pass_mean')} & "
            f"{f('conditional_x_pass_within_yz_fail_mean')} & {f('contamination_mean')} " + r"\\"
        )
    return rf"""
\begin{{table*}}[htbp]
\centering
\caption{{\textbf{{Compact generated-data audit.}} Values are percentages averaged across five top-level seeds. Source-relative quantities are not applicable (NA) to ICL Direct because it was not conditioned on a source transcript. Opposite-class kNN mixing is source independent; lower values indicate less cross-class neighborhood mixing.}}
\label{{tab:supp_generated_audit}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{P{{0.20\textwidth}}rrrrrr}}
\toprule
Method & $Y$--$Z$ pass & $X$ pass & Joint pass & $X$ pass $\mid$ coverage pass & $X$ pass $\mid$ coverage fail & Opposite-class kNN mixing \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
}}
\end{{table*}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the submission-facing Supplementary tables."
    )
    parser.add_argument("--full-metrics-csv", required=True, type=Path)
    parser.add_argument("--k-scaling-csv", required=True, type=Path)
    parser.add_argument("--subgroup-dir", required=True, type=Path)
    parser.add_argument("--yz-source-dir", required=True, type=Path)
    parser.add_argument("--main-figure-source-dir", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    return parser.parse_args()


def configure(args: argparse.Namespace) -> None:
    global FULL_METRICS, K_SCALING, SUBGROUP_DIR, YZ_DIR, FIGURE_LOCK, OUTPUT_FILE
    FULL_METRICS = args.full_metrics_csv
    K_SCALING = args.k_scaling_csv
    SUBGROUP_DIR = args.subgroup_dir
    YZ_DIR = args.yz_source_dir
    FIGURE_LOCK = args.main_figure_source_dir
    OUTPUT_FILE = args.output_file


def main() -> None:
    configure(parse_args())
    parts = [
        full_operating_metrics(),
        probability_reliability(),
        k_scaling(),
        subgroup_tables(),
        yz_cells(),
        generated_audits(),
    ]
    output = OUTPUT_FILE
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
