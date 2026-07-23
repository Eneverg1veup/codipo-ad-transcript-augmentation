from __future__ import annotations

import unittest

import pandas as pd

from codipo_release.baselines.eda import (
    EDASettings,
    FINAL_EDA_SEEDS,
    build_training_table,
    eda,
    get_only_chars,
)
from codipo_release.downstream.train_classifier import validate_run_manifest


def fake_synonyms(word: str) -> list[str]:
    mapping = {"boy": ["child"], "cookie": ["biscuit"]}
    return mapping.get(word, [])


def valid_manifest() -> list[dict[str, str]]:
    return [
        {
            "generation_seed": str(generation_seed),
            "classifier_seed": str(classifier_seed),
            "source_training_data": "source.csv",
            "augmentation_data": f"eda_{generation_seed}.csv",
            "validation_data": "validation.csv",
            "validation_role": "adress_validation",
            "checkpoint_output": f"weights/{generation_seed}_{classifier_seed}.pt",
            "training_mode": "full_training_table",
            "shuffle_seed_source": "classifier_seed",
        }
        for generation_seed in FINAL_EDA_SEEDS
        for classifier_seed in FINAL_EDA_SEEDS
    ]


class EDATests(unittest.TestCase):
    def test_final_settings(self):
        self.assertEqual(EDASettings().num_aug, 2)
        self.assertEqual(len(FINAL_EDA_SEEDS), 5)

    def test_official_cleaning(self):
        self.assertEqual(
            get_only_chars("Boy's cookie-theft; UH!"), "boys cookie theft uh"
        )

    def test_eda_returns_two_generated_variants(self):
        output = eda(
            "the boy takes a cookie",
            settings=EDASettings(),
            seed=2024,
            synonym_provider=fake_synonyms,
        )
        self.assertEqual(len(output), 2)

    def test_training_table_has_two_generated_rows_per_source(self):
        source = pd.DataFrame(
            {"text1": ["the boy takes a cookie", "mother washes"], "label": [1, 0]}
        )
        output = build_training_table(
            source, seed=1077, synonym_provider=fake_synonyms
        )
        self.assertEqual(len(output), 4)
        self.assertEqual(output.groupby("source_id").size().tolist(), [2, 2])
        self.assertEqual(int(output["is_original"].sum()), 0)

    def test_full_training_table_mode_is_admitted(self):
        validate_run_manifest(valid_manifest())

    def test_unknown_training_mode_is_rejected(self):
        rows = valid_manifest()
        rows[0]["training_mode"] = "concatenate_everything"
        with self.assertRaisesRegex(ValueError, "Unsupported training modes"):
            validate_run_manifest(rows)


if __name__ == "__main__":
    unittest.main()
