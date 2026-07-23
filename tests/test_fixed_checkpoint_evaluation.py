from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from codipo_release.downstream.evaluate_fixed_checkpoints import (
    EvaluationSettings,
    validate_checkpoint_manifest,
    validate_dataset_manifest,
)


class FixedCheckpointEvaluationTests(unittest.TestCase):
    def test_locked_defaults(self):
        settings = EvaluationSettings(base_model="bert")
        self.assertEqual(settings.max_length, 320)
        self.assertEqual(settings.batch_size, 32)
        self.assertTrue(settings.strict_load)

    def test_checkpoint_manifest_preserves_all_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.pt"
            second = root / "second.pt"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            rows = [
                {"checkpoint_id": "a", "checkpoint_path": str(first)},
                {"checkpoint_id": "b", "checkpoint_path": str(second)},
            ]
            validate_checkpoint_manifest(rows)
            self.assertEqual([row["checkpoint_id"] for row in rows], ["a", "b"])

    def test_checkpoint_hash_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            path.write_bytes(b"model")
            rows = [
                {
                    "checkpoint_id": "a",
                    "checkpoint_path": str(path),
                    "checkpoint_sha256": "0" * 64,
                }
            ]
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_checkpoint_manifest(rows)

    def test_dataset_names_are_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            path.write_text("text1,label,id\ntext,0,1\n", encoding="utf-8")
            rows = [
                {
                    "dataset": "evaluation",
                    "dataset_role": "evaluation",
                    "data_path": str(path),
                },
                {
                    "dataset": "evaluation",
                    "dataset_role": "evaluation",
                    "data_path": str(path),
                },
            ]
            with self.assertRaisesRegex(ValueError, "must be unique"):
                validate_dataset_manifest(rows)


if __name__ == "__main__":
    unittest.main()
