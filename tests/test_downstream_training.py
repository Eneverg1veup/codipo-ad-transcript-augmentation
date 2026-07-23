from __future__ import annotations

import unittest

from codipo_release.downstream.train_classifier import (
    TrainingSettings,
    validate_run_manifest,
)


def valid_manifest():
    return [
        {
            "generation_seed": str(generation_seed),
            "classifier_seed": str(classifier_seed),
            "source_training_data": "source.csv",
            "augmentation_data": f"aug_{generation_seed}.csv",
            "validation_data": "validation.csv",
            "validation_role": "adress_validation",
            "checkpoint_output": f"weights/{generation_seed}_{classifier_seed}.pt",
        }
        for generation_seed in (10, 20, 30, 40, 50)
        for classifier_seed in (1, 2, 3, 4, 5)
    ]


class DownstreamTrainingTests(unittest.TestCase):
    def test_locked_defaults(self):
        settings = TrainingSettings(base_model="bert")
        self.assertEqual(settings.max_length, 320)
        self.assertEqual(settings.batch_size, 32)
        self.assertEqual(settings.epochs, 20)
        self.assertEqual(settings.patience, 8)
        self.assertTrue(settings.use_cls)

    def test_manifest_requires_complete_five_by_five_grid(self):
        validate_run_manifest(valid_manifest())
        with self.assertRaisesRegex(ValueError, "25 predefined runs"):
            validate_run_manifest(valid_manifest()[:-1])

    def test_manifest_rejects_unsupported_validation_role(self):
        rows = valid_manifest()
        rows[0]["validation_role"] = "reported_evaluation"
        with self.assertRaisesRegex(ValueError, "Unsupported validation roles"):
            validate_run_manifest(rows)

    def test_manifest_requires_unique_outputs(self):
        rows = valid_manifest()
        rows[1]["checkpoint_output"] = rows[0]["checkpoint_output"]
        with self.assertRaisesRegex(ValueError, "unique checkpoint"):
            validate_run_manifest(rows)


if __name__ == "__main__":
    unittest.main()
