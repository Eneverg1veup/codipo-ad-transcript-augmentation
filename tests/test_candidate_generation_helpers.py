from __future__ import annotations

import unittest

from codipo_release.pair_construction.generate_candidates import (
    GenerationSettings,
    extract_generated_text,
    normalize_for_deduplication,
)


class CandidateGenerationHelperTests(unittest.TestCase):
    def test_executed_generation_defaults_are_locked(self) -> None:
        settings = GenerationSettings("model", "image", "sources", "output")
        self.assertEqual(settings.candidates_per_route, 40)
        self.assertEqual(settings.temperature, 1.2)
        self.assertEqual(settings.max_new_tokens, 320)

    def test_normalization_deduplicates_case_and_whitespace(self) -> None:
        left = normalize_for_deduplication("  The  boy\nfalls ")
        right = normalize_for_deduplication("the boy falls")
        self.assertEqual(left, right)

    def test_assistant_prefix_is_removed(self) -> None:
        self.assertEqual(extract_generated_text("ASSISTANT: sample"), "sample")


if __name__ == "__main__":
    unittest.main()
