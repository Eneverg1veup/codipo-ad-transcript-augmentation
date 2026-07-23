"""Build the locked aligned source-local DPO pairs from scored candidates."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


MINIMAL_COLUMNS = ["anchor_text", "chosen_text", "rejected_text", "label"]
FULL_COLUMNS = [
    *MINIMAL_COLUMNS,
    "train_id",
    "pair_mode",
    "pair_rank",
    "chosen_yz_pass",
    "rejected_yz_pass",
    "chosen_bucket",
    "rejected_bucket",
    "chosen_prompt_type",
    "rejected_prompt_type",
    "chosen_residual_cos",
    "rejected_residual_cos",
    "chosen_yz_utilization",
    "rejected_yz_utilization",
    "chosen_yz_violation",
    "rejected_yz_violation",
    "pair_margin",
]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def deduplicate_candidates(
    candidates: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        key = normalize_text(candidate.get("candidate_text", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(dict(candidate))
    return output


def chosen_selection_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        0 if bool(item["yz_pass"]) else 1,
        safe_float(item["residual_cos"]),
        -safe_float(item["yz_utilization"]),
        -safe_float(item["origin_cos"]),
    )


def rejected_selection_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    if not bool(item["yz_pass"]):
        return (
            0,
            -safe_float(item["yz_violation_score"]),
            safe_float(item["residual_cos"]),
            safe_float(item["origin_cos"]),
        )
    return (
        1,
        -safe_float(item["residual_cos"]),
        safe_float(item["yz_utilization"]),
        safe_float(item["origin_cos"]),
    )


def assign_candidate_roles(
    candidates: Iterable[Mapping[str, Any]],
    *,
    chosen_count: int = 6,
    rejected_count: int = 30,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge routes and independently rank final chosen/rejected roles."""

    pool = deduplicate_candidates(candidates)
    chosen_sorted = sorted(pool, key=chosen_selection_key)
    rejected_sorted = sorted(pool, key=rejected_selection_key)
    chosen: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    used: set[str] = set()
    for item in chosen_sorted:
        key = normalize_text(item["candidate_text"])
        if key in used:
            continue
        chosen.append(item)
        used.add(key)
        if len(chosen) >= chosen_count:
            break
    for item in rejected_sorted:
        key = normalize_text(item["candidate_text"])
        if key in used:
            continue
        rejected.append(item)
        used.add(key)
        if len(rejected) >= rejected_count:
            break
    return chosen, rejected


def chosen_pair_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        0 if bool(item.get("yz_pass", False)) else 1,
        safe_float(item.get("residual_cos"), 999.0),
        -safe_float(item.get("yz_utilization")),
    )


def rejected_pair_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    if not bool(item.get("yz_pass", False)):
        return (
            0,
            -safe_float(item.get("yz_violation_score")),
            safe_float(item.get("residual_cos")),
        )
    return (
        1,
        -safe_float(item.get("residual_cos")),
        safe_float(item.get("yz_utilization")),
    )


def rejected_is_strictly_worse(
    chosen: Mapping[str, Any],
    rejected: Mapping[str, Any],
) -> bool:
    chosen_yz = bool(chosen.get("yz_pass", False))
    rejected_yz = bool(rejected.get("yz_pass", False))
    if chosen_yz and not rejected_yz:
        return True
    if not chosen_yz and rejected_yz:
        return False
    if chosen_yz and rejected_yz:
        chosen_residual = safe_float(chosen.get("residual_cos"), 999.0)
        rejected_residual = safe_float(rejected.get("residual_cos"), 999.0)
        if rejected_residual != chosen_residual:
            return rejected_residual > chosen_residual
        return safe_float(rejected.get("yz_utilization")) < safe_float(
            chosen.get("yz_utilization")
        )
    chosen_violation = safe_float(chosen.get("yz_violation_score"))
    rejected_violation = safe_float(rejected.get("yz_violation_score"))
    return rejected_violation > chosen_violation


def compute_pair_margin(
    chosen: Mapping[str, Any],
    rejected: Mapping[str, Any],
) -> float:
    chosen_yz = 1.0 if bool(chosen.get("yz_pass", False)) else 0.0
    rejected_yz = 1.0 if bool(rejected.get("yz_pass", False)) else 0.0
    return float(
        10.0 * (chosen_yz - rejected_yz)
        + 2.0
        * (
            safe_float(rejected.get("residual_cos"))
            - safe_float(chosen.get("residual_cos"))
        )
        + 2.0
        * (
            safe_float(rejected.get("yz_violation_score"))
            - safe_float(chosen.get("yz_violation_score"))
        )
        + (
            safe_float(chosen.get("yz_utilization"))
            - safe_float(rejected.get("yz_utilization"))
        )
    )


def build_aligned_pairs_for_source(
    *,
    train_id: Any,
    anchor_text: str,
    label: int,
    candidates: Iterable[Mapping[str, Any]],
    chosen_count: int = 6,
    rejected_count: int = 30,
    max_chosen_per_source: int = 10,
    max_rejected_per_source: int = 40,
    minimum_pair_margin: float = 1.0,
) -> list[dict[str, Any]]:
    chosen, rejected = assign_candidate_roles(
        candidates,
        chosen_count=chosen_count,
        rejected_count=rejected_count,
    )
    chosen = sorted(
        deduplicate_candidates(chosen), key=chosen_pair_key
    )[:max_chosen_per_source]
    rejected = sorted(
        deduplicate_candidates(rejected), key=rejected_pair_key
    )[:max_rejected_per_source]
    pairs: list[dict[str, Any]] = []
    for pair_rank, (preferred, dispreferred) in enumerate(
        zip(chosen, rejected)
    ):
        if not rejected_is_strictly_worse(preferred, dispreferred):
            continue
        margin = compute_pair_margin(preferred, dispreferred)
        if margin < minimum_pair_margin:
            continue
        pairs.append(
            {
                "anchor_text": str(anchor_text),
                "chosen_text": preferred["candidate_text"],
                "rejected_text": dispreferred["candidate_text"],
                "label": int(label),
                "train_id": train_id,
                "pair_mode": "aligned",
                "pair_rank": pair_rank,
                "chosen_yz_pass": bool(preferred.get("yz_pass", False)),
                "rejected_yz_pass": bool(
                    dispreferred.get("yz_pass", False)
                ),
                "chosen_bucket": preferred.get("bucket_name"),
                "rejected_bucket": dispreferred.get("bucket_name"),
                "chosen_prompt_type": preferred.get(
                    "prompt_type", preferred.get("prompt_route")
                ),
                "rejected_prompt_type": dispreferred.get(
                    "prompt_type", dispreferred.get("prompt_route")
                ),
                "chosen_residual_cos": safe_float(
                    preferred.get("residual_cos")
                ),
                "rejected_residual_cos": safe_float(
                    dispreferred.get("residual_cos")
                ),
                "chosen_yz_utilization": safe_float(
                    preferred.get("yz_utilization")
                ),
                "rejected_yz_utilization": safe_float(
                    dispreferred.get("yz_utilization")
                ),
                "chosen_yz_violation": safe_float(
                    preferred.get("yz_violation_score")
                ),
                "rejected_yz_violation": safe_float(
                    dispreferred.get("yz_violation_score")
                ),
                "pair_margin": margin,
            }
        )
    return pairs


def build_all_pairs(
    sources: Sequence[Mapping[str, Any]],
    scored_candidates: Iterable[Mapping[str, Any]],
    *,
    id_column: str,
    text_column: str,
    label_column: str,
    chosen_count: int = 6,
    rejected_count: int = 30,
    minimum_pair_margin: float = 1.0,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in scored_candidates:
        grouped[str(candidate[id_column])].append(dict(candidate))
    output: list[dict[str, Any]] = []
    for order, source in enumerate(sources):
        source_id = source.get(id_column, order)
        candidates = grouped.get(str(source_id), [])
        output.extend(
            build_aligned_pairs_for_source(
                train_id=source_id,
                anchor_text=str(source[text_column]),
                label=int(source[label_column]),
                candidates=candidates,
                chosen_count=chosen_count,
                rejected_count=rejected_count,
                minimum_pair_margin=minimum_pair_margin,
            )
        )
    return output


def validate_pair_frame(
    frame: pd.DataFrame,
    *,
    expected_sources: int | None = None,
    expected_pairs: int | None = None,
    minimum_pairs_per_source: int | None = None,
    maximum_pairs_per_source: int | None = None,
) -> dict[str, Any]:
    missing = set(FULL_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing full-pair columns: {sorted(missing)}")
    if set(frame["pair_mode"].dropna().unique()) != {"aligned"}:
        raise ValueError("The locked builder accepts only aligned pairs.")
    recomputed = frame.apply(
        lambda row: compute_pair_margin(
            {
                "yz_pass": row["chosen_yz_pass"],
                "residual_cos": row["chosen_residual_cos"],
                "yz_utilization": row["chosen_yz_utilization"],
                "yz_violation_score": row["chosen_yz_violation"],
            },
            {
                "yz_pass": row["rejected_yz_pass"],
                "residual_cos": row["rejected_residual_cos"],
                "yz_utilization": row["rejected_yz_utilization"],
                "yz_violation_score": row["rejected_yz_violation"],
            },
        ),
        axis=1,
    )
    maximum_margin_error = float(
        (recomputed - frame["pair_margin"].astype(float)).abs().max()
    )
    if maximum_margin_error > 1e-10:
        raise ValueError(
            f"Pair-margin mismatch; maximum absolute error={maximum_margin_error}."
        )
    counts = frame.groupby("train_id", dropna=False).size()
    if expected_sources is not None and len(counts) != expected_sources:
        raise ValueError(f"Expected {expected_sources} sources, found {len(counts)}.")
    if expected_pairs is not None and len(frame) != expected_pairs:
        raise ValueError(f"Expected {expected_pairs} pairs, found {len(frame)}.")
    if minimum_pairs_per_source is not None and counts.min() < minimum_pairs_per_source:
        raise ValueError("At least one source has too few pairs.")
    if maximum_pairs_per_source is not None and counts.max() > maximum_pairs_per_source:
        raise ValueError("At least one source has too many pairs.")
    if (frame["pair_margin"].astype(float) < 1.0).any():
        raise ValueError("At least one pair has margin below 1.0.")
    return {
        "sources": int(len(counts)),
        "pairs": int(len(frame)),
        "minimum_pairs_per_source": int(counts.min()),
        "maximum_pairs_per_source": int(counts.max()),
        "maximum_margin_error": maximum_margin_error,
    }


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table type: {path.suffix}")


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    elif path.suffix.lower() == ".xlsx":
        frame.to_excel(path, index=False)
    else:
        raise ValueError("Output must be CSV or XLSX.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--scored-candidates", required=True, type=Path)
    parser.add_argument("--output-full", required=True, type=Path)
    parser.add_argument("--output-training", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--id-column", default="source_id")
    parser.add_argument("--text-column", default="text1")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--chosen-count", type=int, default=6)
    parser.add_argument("--rejected-count", type=int, default=30)
    parser.add_argument("--minimum-pair-margin", type=float, default=1.0)
    parser.add_argument("--expect-locked-shape", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_frame = _read_table(args.sources)
    candidate_records = [
        json.loads(line)
        for line in args.scored_candidates.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pairs = build_all_pairs(
        source_frame.to_dict(orient="records"),
        candidate_records,
        id_column=args.id_column,
        text_column=args.text_column,
        label_column=args.label_column,
        chosen_count=args.chosen_count,
        rejected_count=args.rejected_count,
        minimum_pair_margin=args.minimum_pair_margin,
    )
    full_frame = pd.DataFrame(pairs, columns=FULL_COLUMNS)
    expected = (
        {
            "expected_sources": 108,
            "expected_pairs": 646,
            "minimum_pairs_per_source": 5,
            "maximum_pairs_per_source": 6,
        }
        if args.expect_locked_shape
        else {}
    )
    summary = validate_pair_frame(full_frame, **expected)
    _write_table(full_frame, args.output_full)
    _write_table(full_frame[MINIMAL_COLUMNS], args.output_training)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
