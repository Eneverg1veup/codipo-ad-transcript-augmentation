from __future__ import annotations

import unittest

from codipo_release.pair_construction.candidate_prompts import (
    build_candidate_prompt,
    get_candidate_instructions,
)


class CandidatePromptTests(unittest.TestCase):
    def test_four_class_route_prompts_are_distinct(self) -> None:
        rendered = {
            (label, route): build_candidate_prompt(label, route, "source text")
            for label in (0, 1)
            for route in ("chosen", "rejected")
        }
        self.assertEqual(len(set(rendered.values())), 4)

    def test_route_is_proposal_provenance_not_final_pair_role(self) -> None:
        prompt = build_candidate_prompt(1, "rejected", "source text")
        self.assertIn("intentionally fail in ONE main way", prompt)
        self.assertIn("SOURCE TEXT:\nsource text", prompt)

    def test_invalid_route_and_label_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_candidate_prompt(1, "other", "source")
        with self.assertRaises(ValueError):
            get_candidate_instructions(3)


if __name__ == "__main__":
    unittest.main()
