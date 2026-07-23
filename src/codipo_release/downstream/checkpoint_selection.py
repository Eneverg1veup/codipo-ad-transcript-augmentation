"""Select one checkpoint from one explicitly declared validation dataset.

Lu and Pitt rows are rejected even when they would not change the selected
checkpoint. This makes evaluation-data isolation a property of the input
contract rather than a convention hidden in the training loop.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


ALLOWED_VALIDATION_ROLES = {"cv_validation", "adress_validation"}
FORBIDDEN_EXTERNAL_ROLES = {
    "lu",
    "lu_external",
    "pitt",
    "pitt_external",
    "external",
    "external_evaluation",
}
REQUIRED_COLUMNS = {
    "checkpoint",
    "epoch",
    "dataset_role",
    "metric_name",
    "metric_value",
}


@dataclass(frozen=True)
class MetricRow:
    checkpoint: str
    epoch: int
    dataset_role: str
    metric_name: str
    metric_value: float


@dataclass(frozen=True)
class SelectionRecord:
    checkpoint: str
    epoch: int
    validation_role: str
    primary_metric: str
    primary_value: float
    secondary_metric: str | None
    secondary_value: float | None
    tie_breaker: str
    metrics_csv_sha256: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_role(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def read_metric_rows(path: Path) -> list[MetricRow]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing metric columns: {sorted(missing)}")
        rows = [
            MetricRow(
                checkpoint=row["checkpoint"].strip(),
                epoch=int(row["epoch"]),
                dataset_role=normalize_role(row["dataset_role"]),
                metric_name=row["metric_name"].strip().lower(),
                metric_value=float(row["metric_value"]),
            )
            for row in reader
        ]
    if not rows:
        raise ValueError("The metrics CSV is empty.")
    return rows


def reject_external_rows(rows: list[MetricRow]) -> None:
    external = sorted(
        {row.dataset_role for row in rows if row.dataset_role in FORBIDDEN_EXTERNAL_ROLES}
    )
    if external:
        raise ValueError(
            "Checkpoint-selection input contains external-evaluation rows: "
            + ", ".join(external)
        )


def select_checkpoint(
    rows: list[MetricRow],
    *,
    validation_role: str,
    primary_metric: str,
    secondary_metric: str | None = None,
) -> tuple[MetricRow, float | None]:
    validation_role = normalize_role(validation_role)
    primary_metric = primary_metric.strip().lower()
    secondary_metric = (
        None if secondary_metric is None else secondary_metric.strip().lower()
    )
    if validation_role not in ALLOWED_VALIDATION_ROLES:
        raise ValueError(
            f"Unsupported validation role {validation_role!r}; expected one of "
            f"{sorted(ALLOWED_VALIDATION_ROLES)}"
        )
    reject_external_rows(rows)

    roles = {row.dataset_role for row in rows}
    if roles != {validation_role}:
        raise ValueError(
            "Checkpoint-selection CSV must contain exactly one declared validation "
            f"role ({validation_role}); found {sorted(roles)}"
        )

    by_checkpoint: dict[tuple[str, int], dict[str, float]] = {}
    for row in rows:
        by_checkpoint.setdefault((row.checkpoint, row.epoch), {})[
            row.metric_name
        ] = row.metric_value

    candidates: list[tuple[tuple[float, float, int, str], MetricRow, float | None]] = []
    for (checkpoint, epoch), metrics in by_checkpoint.items():
        if primary_metric not in metrics:
            continue
        secondary_value = (
            None if secondary_metric is None else metrics.get(secondary_metric)
        )
        if secondary_metric is not None and secondary_value is None:
            continue
        primary_value = metrics[primary_metric]
        rank = (
            primary_value,
            float("-inf") if secondary_value is None else secondary_value,
            -epoch,
            checkpoint,
        )
        candidates.append(
            (
                rank,
                MetricRow(
                    checkpoint=checkpoint,
                    epoch=epoch,
                    dataset_role=validation_role,
                    metric_name=primary_metric,
                    metric_value=primary_value,
                ),
                secondary_value,
            )
        )
    if not candidates:
        raise ValueError("No checkpoint has all requested validation metrics.")
    _, selected, secondary_value = max(candidates, key=lambda item: item[0])
    return selected, secondary_value


def build_selection_record(
    metrics_csv: Path,
    *,
    validation_role: str,
    primary_metric: str,
    secondary_metric: str | None,
) -> SelectionRecord:
    rows = read_metric_rows(metrics_csv)
    selected, secondary_value = select_checkpoint(
        rows,
        validation_role=validation_role,
        primary_metric=primary_metric,
        secondary_metric=secondary_metric,
    )
    return SelectionRecord(
        checkpoint=selected.checkpoint,
        epoch=selected.epoch,
        validation_role=selected.dataset_role,
        primary_metric=selected.metric_name,
        primary_value=selected.metric_value,
        secondary_metric=secondary_metric,
        secondary_value=secondary_value,
        tie_breaker=(
            "secondary metric, then earliest epoch"
            if secondary_metric
            else "earliest epoch"
        ),
        metrics_csv_sha256=sha256(metrics_csv),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument(
        "--validation-role",
        choices=sorted(ALLOWED_VALIDATION_ROLES),
        required=True,
    )
    parser.add_argument("--primary-metric", required=True)
    parser.add_argument("--secondary-metric")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    record = build_selection_record(
        args.metrics_csv,
        validation_role=args.validation_role,
        primary_metric=args.primary_metric,
        secondary_metric=args.secondary_metric,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as stream:
        json.dump(asdict(record), stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"Selected checkpoint: {record.checkpoint} (epoch {record.epoch})")


if __name__ == "__main__":
    main()
