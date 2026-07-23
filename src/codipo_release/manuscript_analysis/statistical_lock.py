#!/usr/bin/env python
"""Build the Lu–Pitt external-cohort statistical lock from retained predictions.

The official ADReSS test partition is retained as a separately reported
source-domain validation cohort.  The primary aggregate is calculated within
seed or bootstrap replicate as the unweighted mean of Lu and Pitt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PREDICTION_FILE: Path
OUT_DIR: Path
N_BOOTSTRAP = 10000
RANDOM_SEED = 20260621
CHUNK_SIZE = 100
ECE_BINS = 10

EVALUATION_DATASETS = ["Test", "Lu", "Pitt"]
EXTERNAL_DATASETS = ["Lu", "Pitt"]
EXTERNAL_KEY = "ExternalCohortAverage"
DATASET_LABELS = {
    "Test": "ADReSS validation",
    "Lu": "Lu",
    "Pitt": "Pitt",
    EXTERNAL_KEY: "External-cohort average",
}

PERFORMANCE_METHODS = [
    "BERT",
    "CDA",
    "EDA",
    "ICL Direct",
    "ICL Imitation",
    "ICL Rewrite",
    "CoDiPO",
]
ABLATION_METHODS = [
    "CoDiPO w/o X",
    "CoDiPO w/o YZ",
    "w/o residual decomposition",
    "Cosine-only preference",
    "w/o DPO, vanilla augmentation",
    "w/o DPO, XYZ hard filtering",
]
ALL_METHODS = PERFORMANCE_METHODS + ABLATION_METHODS
TARGET_METHOD = "CoDiPO"

METHOD_LABELS = {
    "BERT": "BERT",
    "CDA": "CDA",
    "EDA": "EDA",
    "ICL Direct": "ICL Direct",
    "ICL Imitation": "ICL Imitation",
    "ICL Rewrite": "ICL Rewrite",
    "CoDiPO": "CoDiPO",
    "CoDiPO w/o X": "Without X",
    "CoDiPO w/o YZ": "Without Y and Z",
    "w/o residual decomposition": "Cosine sim. as X",
    "Cosine-only preference": "Cosine-only",
    "w/o DPO, vanilla augmentation": "Vanilla",
    "w/o DPO, XYZ hard filtering": "Hard Filter",
}

METRIC_ORDER = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1",
    "auroc",
    "auprc",
    "brier",
    "ece",
]
BOOTSTRAP_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1",
    "auroc",
    "auprc",
]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "precision": "Precision",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "f1": "F1",
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "brier": "Brier score",
    "ece": "ECE",
}


@dataclass(frozen=True)
class MethodDatasetPrediction:
    y_true: np.ndarray
    y_pred_by_weight: np.ndarray
    y_prob_by_weight: np.ndarray
    weight_ids: list[str]
    weight_group: dict[str, str]
    group_basis: str


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full_like(num, np.nan, dtype=float)
    np.divide(num, den, out=out, where=den != 0)
    return out


def equal_frequency_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = ECE_BINS) -> np.ndarray:
    """Positive-class ECE for each weight using equal-frequency bins."""
    y_true = np.asarray(y_true, dtype=float)
    probs = np.asarray(probs, dtype=float)
    n_weights, n_obs = probs.shape
    n_bins_eff = min(n_bins, n_obs)
    out = np.zeros(n_weights, dtype=float)
    for w in range(n_weights):
        order = np.argsort(probs[w], kind="mergesort")
        y_sorted = y_true[order]
        p_sorted = probs[w, order]
        for idx in np.array_split(np.arange(n_obs), n_bins_eff):
            if len(idx) == 0:
                continue
            out[w] += (len(idx) / n_obs) * abs(float(y_sorted[idx].mean() - p_sorted[idx].mean()))
    return out


def per_weight_metrics(pred: MethodDatasetPrediction) -> pd.DataFrame:
    y_true = np.asarray(pred.y_true, dtype=int)
    y_pred = np.asarray(pred.y_pred_by_weight, dtype=int)
    y_prob = np.asarray(pred.y_prob_by_weight, dtype=float)
    y_matrix = y_true[None, :]

    tp = np.sum((y_matrix == 1) & (y_pred == 1), axis=1).astype(float)
    tn = np.sum((y_matrix == 0) & (y_pred == 0), axis=1).astype(float)
    fp = np.sum((y_matrix == 0) & (y_pred == 1), axis=1).astype(float)
    fn = np.sum((y_matrix == 1) & (y_pred == 0), axis=1).astype(float)

    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    precision = safe_divide(tp, tp + fp)
    f1 = safe_divide(2.0 * tp, 2.0 * tp + fp + fn)
    accuracy = safe_divide(tp + tn, tp + tn + fp + fn)
    balanced_accuracy = (sensitivity + specificity) / 2.0
    brier = np.mean((y_prob - y_matrix) ** 2, axis=1)
    ece = equal_frequency_ece(y_true, y_prob)

    pos_mask = y_true == 1
    n_pos = int(pos_mask.sum())
    n_neg = int(len(y_true) - n_pos)
    order_asc = np.argsort(y_prob, axis=1)
    ranks = np.empty_like(order_asc, dtype=float)
    np.put_along_axis(
        ranks,
        order_asc,
        np.broadcast_to(np.arange(1, len(y_true) + 1, dtype=float), order_asc.shape),
        axis=1,
    )
    auroc = (np.sum(ranks[:, pos_mask], axis=1) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    order_desc = np.argsort(-y_prob, axis=1)
    y_sorted = y_true[order_desc]
    cum_tp = np.cumsum(y_sorted, axis=1)
    precision_at_rank = cum_tp / np.arange(1, len(y_true) + 1, dtype=float)
    auprc = np.sum(precision_at_rank * y_sorted, axis=1) / n_pos

    return pd.DataFrame(
        {
            "weight_id": pred.weight_ids,
            "weight_group": [pred.weight_group[w] for w in pred.weight_ids],
            "group_basis": pred.group_basis,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "precision": precision,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1": f1,
            "auroc": auroc,
            "auprc": auprc,
            "brier": brier,
            "ece": ece,
        }
    )


def load_prediction_arrays() -> tuple[
    dict[str, np.ndarray],
    dict[tuple[str, str], MethodDatasetPrediction],
    pd.DataFrame,
]:
    columns = [
        "dataset",
        "participant_id",
        "method_display",
        "family",
        "method_group",
        "weight_id",
        "aug_seed",
        "train_seed",
        "y_true",
        "y_pred_binary",
        "y_prob_ad",
    ]
    frame = pd.read_csv(PREDICTION_FILE, usecols=columns)
    frame = frame[
        frame["dataset"].isin(EVALUATION_DATASETS)
        & frame["method_display"].isin(ALL_METHODS)
    ].copy()

    family = (
        frame[["method_display", "family", "method_group"]]
        .drop_duplicates()
        .sort_values(["method_display", "family", "method_group"])
        .drop_duplicates("method_display", keep="first")
    )

    arrays: dict[tuple[str, str], MethodDatasetPrediction] = {}
    cohort_labels: dict[str, np.ndarray] = {}

    for dataset in EVALUATION_DATASETS:
        ds = frame[frame["dataset"].eq(dataset)]
        base = (
            ds[["participant_id", "y_true"]]
            .drop_duplicates()
            .sort_values("participant_id")
            .reset_index(drop=True)
        )
        participant_index = pd.Index(base["participant_id"])
        cohort_labels[dataset] = base["y_true"].astype(int).to_numpy()

        for method in ALL_METHODS:
            method_ds = ds[ds["method_display"].eq(method)]
            weight_meta = (
                method_ds[["weight_id", "aug_seed", "train_seed"]]
                .drop_duplicates()
                .sort_values("weight_id")
            )
            if weight_meta.empty:
                raise RuntimeError(f"No predictions for {method} in {dataset}")

            has_aug_seed = weight_meta["aug_seed"].notna().any()
            group_basis = "generation_seed" if has_aug_seed else "classifier_seed"
            pred_rows: list[np.ndarray] = []
            prob_rows: list[np.ndarray] = []
            weight_ids: list[str] = []
            group_map: dict[str, str] = {}
            y_reference: np.ndarray | None = None

            for _, meta in weight_meta.iterrows():
                weight_id = str(meta["weight_id"])
                one = (
                    method_ds[method_ds["weight_id"].eq(meta["weight_id"])][
                        ["participant_id", "y_true", "y_pred_binary", "y_prob_ad"]
                    ]
                    .drop_duplicates("participant_id")
                    .set_index("participant_id")
                    .reindex(participant_index)
                )
                if one.isna().any().any():
                    raise RuntimeError(f"Incomplete prediction grid for {method}, {dataset}, {weight_id}")
                y_current = one["y_true"].astype(int).to_numpy()
                if y_reference is None:
                    y_reference = y_current
                elif not np.array_equal(y_reference, y_current):
                    raise RuntimeError(f"Outcome mismatch across weights for {method}, {dataset}")

                pred_rows.append(one["y_pred_binary"].astype(int).to_numpy())
                prob_rows.append(one["y_prob_ad"].astype(float).to_numpy())
                weight_ids.append(weight_id)
                if has_aug_seed:
                    group_map[weight_id] = f"aug_seed_{int(float(meta['aug_seed']))}"
                else:
                    seed_value = meta["train_seed"]
                    group_map[weight_id] = (
                        f"train_seed_{int(float(seed_value))}"
                        if pd.notna(seed_value)
                        else f"weight_{weight_id}"
                    )

            assert y_reference is not None
            arrays[(method, dataset)] = MethodDatasetPrediction(
                y_true=y_reference,
                y_pred_by_weight=np.vstack(pred_rows),
                y_prob_by_weight=np.vstack(prob_rows),
                weight_ids=weight_ids,
                weight_group=group_map,
                group_basis=group_basis,
            )

    return cohort_labels, arrays, family


def build_seed_level_outputs(
    arrays: dict[tuple[str, str], MethodDatasetPrediction]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_parts: list[pd.DataFrame] = []
    for method in ALL_METHODS:
        for dataset in EVALUATION_DATASETS:
            per_weight = per_weight_metrics(arrays[(method, dataset)])
            group = (
                per_weight.groupby(["weight_group", "group_basis"], as_index=False)[METRIC_ORDER]
                .mean()
            )
            group.insert(0, "dataset", dataset)
            group.insert(0, "cohort", DATASET_LABELS[dataset])
            group.insert(0, "method_label", METHOD_LABELS[method])
            group.insert(0, "method_display", method)
            detail_parts.append(group)

    cohort_detail = pd.concat(detail_parts, ignore_index=True)
    external_parts: list[pd.DataFrame] = []
    for method in ALL_METHODS:
        method_rows = cohort_detail[cohort_detail["method_display"].eq(method)]
        lu = method_rows[method_rows["dataset"].eq("Lu")]
        pitt = method_rows[method_rows["dataset"].eq("Pitt")]
        keys = ["method_display", "method_label", "weight_group", "group_basis"]
        if set(lu["weight_group"]) != set(pitt["weight_group"]):
            raise RuntimeError(f"Lu/Pitt seed groups differ for {method}")
        merged = lu[keys + METRIC_ORDER].merge(
            pitt[keys + METRIC_ORDER],
            on=keys,
            suffixes=("_lu", "_pitt"),
            validate="one_to_one",
        )
        external = merged[keys].copy()
        external.insert(2, "cohort", DATASET_LABELS[EXTERNAL_KEY])
        external.insert(3, "dataset", EXTERNAL_KEY)
        for metric in METRIC_ORDER:
            external[metric] = (merged[f"{metric}_lu"] + merged[f"{metric}_pitt"]) / 2.0
        external_parts.append(external)

    external_detail = pd.concat(external_parts, ignore_index=True)
    full_detail = pd.concat([cohort_detail, external_detail], ignore_index=True)

    summary_rows: list[dict[str, object]] = []
    for (method, method_label, dataset, cohort, basis), group in full_detail.groupby(
        ["method_display", "method_label", "dataset", "cohort", "group_basis"],
        sort=False,
    ):
        for metric in METRIC_ORDER:
            values = group[metric].astype(float).to_numpy()
            summary_rows.append(
                {
                    "method_display": method,
                    "method_label": method_label,
                    "dataset": dataset,
                    "cohort": cohort,
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "mean": float(np.nanmean(values)),
                    "sd": float(np.nanstd(values, ddof=1)) if np.isfinite(values).sum() > 1 else np.nan,
                    "n_groups": int(np.isfinite(values).sum()),
                    "aggregation_basis": basis,
                    "table_unit": (
                        "generation seed; classifier seeds averaged within generation"
                        if basis == "generation_seed"
                        else "classifier/corruption seed"
                    ),
                }
            )
    return full_detail, pd.DataFrame(summary_rows)


def build_bootstrap_samples(cohort_labels: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)
    samples = {
        dataset: np.empty((N_BOOTSTRAP, len(labels)), dtype=np.int32)
        for dataset, labels in cohort_labels.items()
    }
    for replicate in range(N_BOOTSTRAP):
        for dataset, labels in cohort_labels.items():
            neg = np.flatnonzero(labels == 0)
            pos = np.flatnonzero(labels == 1)
            sampled = np.concatenate(
                [
                    rng.choice(neg, size=len(neg), replace=True),
                    rng.choice(pos, size=len(pos), replace=True),
                ]
            )
            rng.shuffle(sampled)
            samples[dataset][replicate] = sampled
    return samples


def bootstrap_metrics(pred: MethodDatasetPrediction, samples: np.ndarray) -> dict[str, np.ndarray]:
    output = {metric: np.empty(N_BOOTSTRAP, dtype=float) for metric in BOOTSTRAP_METRICS}
    n_weights = pred.y_pred_by_weight.shape[0]

    for start in range(0, N_BOOTSTRAP, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, N_BOOTSTRAP)
        idx = samples[start:stop]
        y_true = pred.y_true[idx]
        y_pred = pred.y_pred_by_weight[:, idx]
        y_prob = pred.y_prob_by_weight[:, idx]
        y_matrix = np.broadcast_to(y_true[None, :, :], y_pred.shape)

        tp = np.sum((y_matrix == 1) & (y_pred == 1), axis=2).astype(float)
        tn = np.sum((y_matrix == 0) & (y_pred == 0), axis=2).astype(float)
        fp = np.sum((y_matrix == 0) & (y_pred == 1), axis=2).astype(float)
        fn = np.sum((y_matrix == 1) & (y_pred == 0), axis=2).astype(float)

        sensitivity = safe_divide(tp, tp + fn)
        specificity = safe_divide(tn, tn + fp)
        precision = safe_divide(tp, tp + fp)
        f1 = safe_divide(2.0 * tp, 2.0 * tp + fp + fn)
        accuracy = safe_divide(tp + tn, tp + tn + fp + fn)
        balanced_accuracy = (sensitivity + specificity) / 2.0

        chunk_metrics = {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "precision": precision,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1": f1,
        }

        n_obs = y_true.shape[1]
        pos_counts = np.sum(y_true == 1, axis=1).astype(float)
        neg_counts = n_obs - pos_counts
        order_asc = np.argsort(y_prob, axis=2)
        ranks = np.empty_like(order_asc, dtype=float)
        rank_values = np.broadcast_to(
            np.arange(1, n_obs + 1, dtype=float),
            (n_weights, stop - start, n_obs),
        )
        np.put_along_axis(ranks, order_asc, rank_values, axis=2)
        rank_sum_pos = np.sum(ranks * (y_matrix == 1), axis=2)
        auroc = (
            rank_sum_pos - pos_counts[None, :] * (pos_counts[None, :] + 1.0) / 2.0
        ) / (pos_counts[None, :] * neg_counts[None, :])

        order_desc = np.argsort(-y_prob, axis=2)
        y_sorted = np.take_along_axis(y_matrix, order_desc, axis=2)
        cumulative_tp = np.cumsum(y_sorted, axis=2)
        precision_at_rank = cumulative_tp / np.arange(1, n_obs + 1, dtype=float)
        auprc = np.sum(precision_at_rank * y_sorted, axis=2) / pos_counts[None, :]
        chunk_metrics["auroc"] = auroc
        chunk_metrics["auprc"] = auprc

        for metric, values in chunk_metrics.items():
            output[metric][start:stop] = np.nanmean(values, axis=0)

    return output


def summarize_bootstrap(
    arrays: dict[tuple[str, str], MethodDatasetPrediction],
    samples: dict[str, np.ndarray],
    seed_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    boot: dict[tuple[str, str, str], np.ndarray] = {}
    rows: list[dict[str, object]] = []

    observed_lookup = seed_summary.set_index(["method_display", "dataset", "metric"])["mean"]
    for method_index, method in enumerate(ALL_METHODS, start=1):
        print(f"[{method_index}/{len(ALL_METHODS)}] bootstrap {method}", flush=True)
        for dataset in EVALUATION_DATASETS:
            result = bootstrap_metrics(arrays[(method, dataset)], samples[dataset])
            for metric, values in result.items():
                boot[(method, dataset, metric)] = values

        for metric in BOOTSTRAP_METRICS:
            boot[(method, EXTERNAL_KEY, metric)] = np.mean(
                np.vstack([boot[(method, dataset, metric)] for dataset in EXTERNAL_DATASETS]),
                axis=0,
            )

        for dataset in EVALUATION_DATASETS + [EXTERNAL_KEY]:
            for metric in BOOTSTRAP_METRICS:
                values = boot[(method, dataset, metric)]
                rows.append(
                    {
                        "method_display": method,
                        "method_label": METHOD_LABELS[method],
                        "dataset": dataset,
                        "cohort": DATASET_LABELS[dataset],
                        "metric": metric,
                        "metric_label": METRIC_LABELS[metric],
                        "point_estimate": float(observed_lookup.loc[(method, dataset, metric)]),
                        "bootstrap_mean": float(np.nanmean(values)),
                        "bootstrap_sd": float(np.nanstd(values, ddof=1)),
                        "ci_low": float(np.nanquantile(values, 0.025)),
                        "ci_high": float(np.nanquantile(values, 0.975)),
                        "n_bootstrap": N_BOOTSTRAP,
                        "bootstrap_seed": RANDOM_SEED,
                        "bootstrap_unit": "diagnosis-stratified evaluation-unit resampling within cohort",
                        "aggregate_definition": (
                            "unweighted mean of Lu and Pitt within replicate"
                            if dataset == EXTERNAL_KEY
                            else "cohort-specific"
                        ),
                    }
                )

    contrast_rows: list[dict[str, object]] = []
    for comparator in ALL_METHODS:
        if comparator == TARGET_METHOD:
            continue
        for dataset in EVALUATION_DATASETS + [EXTERNAL_KEY]:
            for metric in BOOTSTRAP_METRICS:
                delta = boot[(TARGET_METHOD, dataset, metric)] - boot[(comparator, dataset, metric)]
                observed_delta = float(
                    observed_lookup.loc[(TARGET_METHOD, dataset, metric)]
                    - observed_lookup.loc[(comparator, dataset, metric)]
                )
                n_valid = int(np.isfinite(delta).sum())
                p_low = (np.sum(delta <= 0) + 1) / (n_valid + 1)
                p_high = (np.sum(delta >= 0) + 1) / (n_valid + 1)
                contrast_rows.append(
                    {
                        "target_method": TARGET_METHOD,
                        "comparator": comparator,
                        "comparator_label": METHOD_LABELS[comparator],
                        "dataset": dataset,
                        "cohort": DATASET_LABELS[dataset],
                        "metric": metric,
                        "observed_delta": observed_delta,
                        "bootstrap_mean_delta": float(np.nanmean(delta)),
                        "bootstrap_sd_delta": float(np.nanstd(delta, ddof=1)),
                        "ci_low": float(np.nanquantile(delta, 0.025)),
                        "ci_high": float(np.nanquantile(delta, 0.975)),
                        "p_bootstrap_two_sided": float(min(1.0, 2.0 * min(p_low, p_high))),
                        "n_bootstrap": N_BOOTSTRAP,
                        "bootstrap_seed": RANDOM_SEED,
                        "delta_direction": "positive favors CoDiPO",
                    }
                )

    npz_payload = {
        f"{method_index:02d}_{dataset}_{metric}": values
        for method_index, method in enumerate(ALL_METHODS)
        for dataset in EVALUATION_DATASETS + [EXTERNAL_KEY]
        for metric in BOOTSTRAP_METRICS
        if (values := boot.get((method, dataset, metric))) is not None
    }
    return pd.DataFrame(rows), pd.DataFrame(contrast_rows), npz_payload


def write_qa(
    seed_detail: pd.DataFrame,
    seed_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    qa_rows: list[dict[str, object]] = []
    for method in ALL_METHODS:
        for metric in METRIC_ORDER:
            summary = seed_summary[
                seed_summary["method_display"].eq(method)
                & seed_summary["metric"].eq(metric)
            ].set_index("dataset")["mean"]
            expected = float((summary.loc["Lu"] + summary.loc["Pitt"]) / 2.0)
            observed = float(summary.loc[EXTERNAL_KEY])
            qa_rows.append(
                {
                    "check": "seed_summary_external_equals_lu_pitt_mean",
                    "method_display": method,
                    "metric": metric,
                    "expected": expected,
                    "observed": observed,
                    "abs_diff": abs(expected - observed),
                    "pass": abs(expected - observed) < 1e-12,
                }
            )

    qa = pd.DataFrame(qa_rows)
    qa.to_csv(OUT_DIR / "qa_external_point_estimates.csv", index=False, encoding="utf-8-sig")
    if not qa["pass"].all():
        raise RuntimeError("External point-estimate QA failed")

    key_counts = (
        seed_detail.groupby(["method_display", "dataset"])["weight_group"]
        .nunique()
        .reset_index(name="n_groups")
    )
    key_counts.to_csv(OUT_DIR / "qa_seed_group_counts.csv", index=False, encoding="utf-8-sig")

    bootstrap_counts = (
        bootstrap.groupby(["method_display", "dataset"])["metric"]
        .nunique()
        .reset_index(name="n_bootstrapped_metrics")
    )
    bootstrap_counts.to_csv(OUT_DIR / "qa_bootstrap_grid.csv", index=False, encoding="utf-8-sig")
    if not bootstrap_counts["n_bootstrapped_metrics"].eq(len(BOOTSTRAP_METRICS)).all():
        raise RuntimeError("Bootstrap grid is incomplete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute cohort-specific metrics, the unweighted Lu-Pitt external-cohort "
            "average, and diagnosis-stratified bootstrap intervals from a fixed "
            "prediction lock."
        )
    )
    parser.add_argument(
        "--prediction-lock",
        required=True,
        type=Path,
        help="CSV containing predictions from already locked checkpoints.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260621)
    parser.add_argument("--chunk-size", type=int, default=100)
    return parser.parse_args()


def configure(args: argparse.Namespace) -> None:
    global PREDICTION_FILE, OUT_DIR, N_BOOTSTRAP, RANDOM_SEED, CHUNK_SIZE
    PREDICTION_FILE = args.prediction_lock
    OUT_DIR = args.output_dir
    N_BOOTSTRAP = args.n_bootstrap
    RANDOM_SEED = args.bootstrap_seed
    CHUNK_SIZE = args.chunk_size
    if N_BOOTSTRAP < 1:
        raise ValueError("n_bootstrap must be positive.")
    if CHUNK_SIZE < 1:
        raise ValueError("chunk_size must be positive.")


def main() -> None:
    configure(parse_args())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(PREDICTION_FILE)

    prediction_hash = sha256_file(PREDICTION_FILE)
    print(f"Prediction lock: {PREDICTION_FILE}")
    print(f"SHA-256: {prediction_hash}")
    print(f"Bootstrap: n={N_BOOTSTRAP}, seed={RANDOM_SEED}, chunk={CHUNK_SIZE}")

    cohort_labels, arrays, family = load_prediction_arrays()
    seed_detail, seed_summary = build_seed_level_outputs(arrays)
    samples = build_bootstrap_samples(cohort_labels)
    bootstrap, contrasts, replicate_payload = summarize_bootstrap(arrays, samples, seed_summary)

    seed_detail.to_csv(
        OUT_DIR / "seed_group_level_metrics_external.csv",
        index=False,
        encoding="utf-8-sig",
    )
    seed_summary.to_csv(
        OUT_DIR / "cohort_generation_level_method_metrics_external.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap.to_csv(
        OUT_DIR / f"cohort_bootstrap_method_metrics_n{N_BOOTSTRAP}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    external_bootstrap = bootstrap[bootstrap["dataset"].eq(EXTERNAL_KEY)].copy()
    external_bootstrap.to_csv(
        OUT_DIR / "external_cohort_bootstrap_method_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    contrasts.to_csv(
        OUT_DIR / f"cohort_bootstrap_codipo_contrasts_n{N_BOOTSTRAP}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    external_contrasts = contrasts[contrasts["dataset"].eq(EXTERNAL_KEY)].copy()
    external_contrasts.to_csv(
        OUT_DIR / "external_cohort_bootstrap_primary_contrasts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    external_contrasts[
        external_contrasts["metric"].isin(
            ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1"]
        )
    ].to_csv(
        OUT_DIR / "external_cohort_bootstrap_primary_contrasts_primary_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    family.to_csv(OUT_DIR / "method_family_lock.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(OUT_DIR / "bootstrap_replicates_all_cohorts.npz", **replicate_payload)

    write_qa(seed_detail, seed_summary, bootstrap)

    metadata = {
        "prediction_file": PREDICTION_FILE.name,
        "prediction_sha256": prediction_hash,
        "prediction_rows": int(sum(1 for _ in PREDICTION_FILE.open("rb")) - 1),
        "evaluation_datasets": EVALUATION_DATASETS,
        "source_domain_validation": "Test (official ADReSS test partition)",
        "external_datasets": EXTERNAL_DATASETS,
        "external_definition": "unweighted Lu-Pitt cohort-level mean within seed/replicate",
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_seed": RANDOM_SEED,
        "ece_definition": "10 equal-frequency bins",
    }
    (OUT_DIR / "STATISTICAL_LOCK_METADATA.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (OUT_DIR / "STATISTICAL_ANALYSIS_LOCK_EXTERNAL_COHORT.md").write_text(
        f"""# External-cohort statistical analysis lock

- Prediction lock SHA-256: `{prediction_hash}`
- Source-domain validation: official ADReSS test partition, reported separately.
- Primary aggregate: unweighted mean of Lu and Pitt within each top-level seed group or bootstrap replicate.
- Bootstrap: {N_BOOTSTRAP:,} diagnosis-stratified evaluation-unit resamples within cohort; seed {RANDOM_SEED}; shared replicates across methods.
- Pitt primary estimand: transcript-observation performance.
- Seed aggregation: classifier seeds are averaged within generation seed before cross-generation mean and sample SD; non-generative methods use their locked classifier/corruption seed groups.
- ECE: 10 equal-frequency bins, computed within cohort and seed before Lu-Pitt averaging.
- The aggregate is not a pooled-population estimate and does not include ADReSS validation.
""",
        encoding="utf-8",
    )
    print(f"Wrote external statistical lock to {OUT_DIR}")


if __name__ == "__main__":
    main()
