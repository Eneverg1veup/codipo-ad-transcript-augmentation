from __future__ import annotations

import random
import unittest
from importlib.util import find_spec

import pandas as pd

from codipo_release.baselines.cda import (
    CDASettings,
    FINAL_CDA_SEEDS,
    build_training_records,
    cda_margin_loss,
    random_delete_words,
    symmetric_bernoulli_kl,
)


class CDATests(unittest.TestCase):
    def test_corrected_protocol_defaults(self):
        settings = CDASettings(base_model="bert")
        self.assertEqual(settings.max_length, 320)
        self.assertEqual(settings.n_aug, 3)
        self.assertEqual(settings.n_neg, 3)
        self.assertEqual(settings.patience, 5)
        self.assertEqual(settings.learning_rate, 1e-4)
        self.assertEqual(FINAL_CDA_SEEDS, (42, 52, 62, 72, 82))

    def test_training_records_have_three_repeats_and_three_negatives(self):
        frame = pd.DataFrame({"text1": ["one two three"], "label": [1]})
        settings = CDASettings(base_model="bert")
        records = build_training_records(frame, settings=settings, seed=42)
        self.assertEqual(len(records), 3)
        self.assertTrue(
            all(len(record["negative_texts"]) == 3 for record in records)
        )

    def test_random_deletion_can_preserve_one_token_when_requested(self):
        output = random_delete_words(
            "one two",
            delete_probability=1.0,
            rng=random.Random(1),
            allow_empty=False,
        )
        self.assertIn(output, {"one", "two"})

    @unittest.skipUnless(find_spec("torch"), "PyTorch is not installed")
    def test_symmetric_kl_is_zero_for_equal_logits(self):
        import torch

        logits = torch.tensor([-1.0, 0.0, 1.0])
        value = symmetric_bernoulli_kl(logits, logits)
        self.assertAlmostEqual(float(value), 0.0, places=7)

    @unittest.skipUnless(find_spec("torch"), "PyTorch is not installed")
    def test_margin_direction_matches_ad_positive_definition(self):
        import torch

        original = torch.tensor([0.0])
        more_ad_like_negative = torch.tensor([2.0, 2.0, 2.0])
        less_ad_like_negative = torch.tensor([-2.0, -2.0, -2.0])
        low_loss = cda_margin_loss(
            original, more_ad_like_negative, margin=0.1, n_neg=3
        )
        high_loss = cda_margin_loss(
            original, less_ad_like_negative, margin=0.1, n_neg=3
        )
        self.assertLess(float(low_loss), float(high_loss))


if __name__ == "__main__":
    unittest.main()
