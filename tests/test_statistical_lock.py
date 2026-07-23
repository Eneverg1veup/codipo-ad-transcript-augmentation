import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np

from codipo_release.manuscript_analysis import statistical_lock as lock


class StatisticalLockTests(unittest.TestCase):
    def test_safe_divide_marks_zero_denominator_as_nan(self) -> None:
        result = lock.safe_divide(
            np.array([2.0, 1.0, 0.0]),
            np.array([4.0, 0.0, 0.0]),
        )
        self.assertEqual(result[0], 0.5)
        self.assertTrue(np.isnan(result[1]))
        self.assertTrue(np.isnan(result[2]))

    def test_equal_frequency_ece_is_zero_for_empirically_calibrated_bins(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        probabilities = np.array([[0.5, 0.5, 0.5, 0.5]])
        result = lock.equal_frequency_ece(y_true, probabilities, n_bins=1)
        np.testing.assert_allclose(result, np.array([0.0]))

    def test_per_weight_metrics_uses_positive_ad_class(self) -> None:
        prediction = lock.MethodDatasetPrediction(
            y_true=np.array([0, 0, 1, 1]),
            y_pred_by_weight=np.array([[0, 1, 1, 1]]),
            y_prob_by_weight=np.array([[0.1, 0.6, 0.7, 0.9]]),
            weight_ids=["seed_42"],
            weight_group={"seed_42": "train_seed_42"},
            group_basis="classifier_seed",
        )
        row = lock.per_weight_metrics(prediction).iloc[0]
        self.assertAlmostEqual(row["accuracy"], 0.75)
        self.assertAlmostEqual(row["sensitivity"], 1.0)
        self.assertAlmostEqual(row["specificity"], 0.5)
        self.assertAlmostEqual(row["precision"], 2.0 / 3.0)
        self.assertAlmostEqual(row["f1"], 0.8)

    def test_bootstrap_sampling_preserves_class_counts(self) -> None:
        previous_n = lock.N_BOOTSTRAP
        previous_seed = lock.RANDOM_SEED
        try:
            lock.N_BOOTSTRAP = 8
            lock.RANDOM_SEED = 17
            labels = {"Lu": np.array([0, 0, 0, 1, 1])}
            samples = lock.build_bootstrap_samples(labels)["Lu"]
            sampled_labels = labels["Lu"][samples]
            np.testing.assert_array_equal(
                np.sum(sampled_labels == 0, axis=1),
                np.full(8, 3),
            )
            np.testing.assert_array_equal(
                np.sum(sampled_labels == 1, axis=1),
                np.full(8, 2),
            )
        finally:
            lock.N_BOOTSTRAP = previous_n
            lock.RANDOM_SEED = previous_seed

    def test_external_average_excludes_adress_validation(self) -> None:
        arrays = {}
        for dataset, y_pred in {
            "Test": np.array([[1, 0, 1, 0]]),
            "Lu": np.array([[0, 0, 1, 1]]),
            "Pitt": np.array([[0, 1, 0, 1]]),
        }.items():
            arrays[("BERT", dataset)] = lock.MethodDatasetPrediction(
                y_true=np.array([0, 0, 1, 1]),
                y_pred_by_weight=y_pred,
                y_prob_by_weight=y_pred.astype(float) * 0.8 + 0.1,
                weight_ids=["seed_42"],
                weight_group={"seed_42": "train_seed_42"},
                group_basis="classifier_seed",
            )

        original_methods = lock.ALL_METHODS
        try:
            lock.ALL_METHODS = ["BERT"]
            _, summary = lock.build_seed_level_outputs(arrays)
        finally:
            lock.ALL_METHODS = original_methods

        f1 = summary[
            summary["method_display"].eq("BERT") & summary["metric"].eq("f1")
        ].set_index("dataset")["mean"]
        self.assertAlmostEqual(
            f1[lock.EXTERNAL_KEY],
            (f1["Lu"] + f1["Pitt"]) / 2.0,
        )
        self.assertNotAlmostEqual(f1[lock.EXTERNAL_KEY], f1["Test"])

    def test_configure_rejects_nonpositive_bootstrap_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = argparse.Namespace(
                prediction_lock=Path(temp_dir) / "predictions.csv",
                output_dir=Path(temp_dir) / "output",
                n_bootstrap=0,
                bootstrap_seed=1,
                chunk_size=10,
            )
            with self.assertRaisesRegex(ValueError, "n_bootstrap"):
                lock.configure(args)


if __name__ == "__main__":
    unittest.main()
