from __future__ import annotations

import unittest

from codipo_release.dpo.alignment_prompts import build_alignment_task
from codipo_release.dpo.train_dpo import clean_completion


class AlignmentPromptTests(unittest.TestCase):
    def test_ad_and_hc_use_distinct_label_contexts(self) -> None:
        ad = build_alignment_task(1, "the boy is reaching")
        hc = build_alignment_task(0, "the boy is reaching")
        self.assertIn("Alzheimer's Disease", ad)
        self.assertIn("Healthy Control", hc)
        self.assertNotEqual(ad, hc)

    def test_pair_context_requests_chosen_mode(self) -> None:
        task = build_alignment_task(1, "sample source")
        self.assertIn("MODE=CHOSEN", task)
        self.assertIn("[sample source]", task)

    def test_invalid_label_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_alignment_task(2, "sample source")

    def test_legacy_chat_prefix_is_cleaned(self) -> None:
        self.assertEqual(clean_completion("ASSISTANT: hello"), "hello")


if __name__ == "__main__":
    unittest.main()
