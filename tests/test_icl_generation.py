from __future__ import annotations

import unittest

from codipo_release.augmentation_inference.generate_icl_augmentations import (
    FINAL_ICL_METHODS,
    FINAL_ICL_SEEDS,
    ICLGenerationSettings,
    strip_assistant_prefix,
)
from codipo_release.augmentation_inference.icl_prompts import build_icl_prompt


class ICLGenerationTests(unittest.TestCase):
    def test_final_generation_defaults(self):
        settings = ICLGenerationSettings()
        self.assertEqual(settings.temperature, 1.2)
        self.assertEqual(settings.max_new_tokens, 320)
        self.assertEqual(settings.samples_per_source, 2)
        self.assertEqual(settings.methods, FINAL_ICL_METHODS)
        self.assertEqual(FINAL_ICL_SEEDS, (5942, 3012, 4951, 2921, 9032))

    def test_rewrite_is_source_conditioned(self):
        _, prompt = build_icl_prompt("rewrite", 1, "source words")
        self.assertIn("Original text:\nsource words", prompt)
        self.assertIn("Alzheimer's Disease", prompt)

    def test_imitation_is_source_conditioned_but_not_line_rewrite(self):
        _, prompt = build_icl_prompt("imitation", 0, "reference words")
        self.assertIn("Reference text:\nreference words", prompt)
        self.assertIn("not a line-by-line rewrite", prompt)

    def test_direct_does_not_include_source(self):
        _, prompt = build_icl_prompt("direct", 1, "secret source words")
        self.assertNotIn("secret source words", prompt)
        self.assertNotIn("Reference text", prompt)

    def test_invalid_method_and_label_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported ICL method"):
            build_icl_prompt("unknown", 1, "text")
        with self.assertRaisesRegex(ValueError, "Label must be"):
            build_icl_prompt("direct", 2, "text")

    def test_assistant_prefix_is_removed(self):
        self.assertEqual(
            strip_assistant_prefix("context ASSISTANT: generated words"),
            "generated words",
        )


if __name__ == "__main__":
    unittest.main()
