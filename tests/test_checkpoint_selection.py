from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from codipo_release.downstream.checkpoint_selection import (
    MetricRow,
    build_selection_record,
    select_checkpoint,
)


class CheckpointSelectionTests(unittest.TestCase):
    def test_uses_one_declared_validation_role(self) -> None:
        rows = [
            MetricRow("epoch1.pt", 1, "cv_validation", "accuracy", 0.80),
            MetricRow("epoch1.pt", 1, "cv_validation", "f1", 0.78),
            MetricRow("epoch2.pt", 2, "cv_validation", "accuracy", 0.80),
            MetricRow("epoch2.pt", 2, "cv_validation", "f1", 0.82),
        ]
        selected, secondary = select_checkpoint(
            rows,
            validation_role="cv_validation",
            primary_metric="accuracy",
            secondary_metric="f1",
        )
        self.assertEqual(selected.checkpoint, "epoch2.pt")
        self.assertEqual(secondary, 0.82)

    def test_rejects_lu_even_when_validation_rows_exist(self) -> None:
        rows = [
            MetricRow("epoch1.pt", 1, "adress_validation", "f1", 0.80),
            MetricRow("epoch1.pt", 1, "lu_external", "f1", 0.90),
        ]
        with self.assertRaisesRegex(ValueError, "external-evaluation"):
            select_checkpoint(
                rows,
                validation_role="adress_validation",
                primary_metric="f1",
            )

    def test_rejects_pitt_as_validation_role(self) -> None:
        rows = [MetricRow("epoch1.pt", 1, "pitt_external", "f1", 0.80)]
        with self.assertRaisesRegex(ValueError, "Unsupported validation role"):
            select_checkpoint(
                rows,
                validation_role="pitt_external",
                primary_metric="f1",
            )

    def test_record_contains_input_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metrics.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "checkpoint",
                        "epoch",
                        "dataset_role",
                        "metric_name",
                        "metric_value",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "checkpoint": "epoch1.pt",
                        "epoch": 1,
                        "dataset_role": "adress_validation",
                        "metric_name": "accuracy",
                        "metric_value": 0.8,
                    }
                )
            record = build_selection_record(
                path,
                validation_role="adress_validation",
                primary_metric="accuracy",
                secondary_metric=None,
            )
            self.assertEqual(len(record.metrics_csv_sha256), 64)


if __name__ == "__main__":
    unittest.main()
