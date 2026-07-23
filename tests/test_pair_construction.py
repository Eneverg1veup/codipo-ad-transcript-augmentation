import unittest

import pandas as pd

from codipo_release.pair_construction.build_pairs import (
    FULL_COLUMNS,
    assign_candidate_roles,
    build_aligned_pairs_for_source,
    compute_pair_margin,
    validate_pair_frame,
)


def candidate(text, yz_pass, residual, utilization, violation, origin=0.8):
    if yz_pass and residual < 0.14:
        bucket = "yz_pass_x_pass"
    elif yz_pass:
        bucket = "yz_pass_x_fail"
    elif residual < 0.14:
        bucket = "yz_fail_x_pass"
    else:
        bucket = "yz_fail_x_fail"
    return {
        "candidate_text": text,
        "yz_pass": yz_pass,
        "residual_cos": residual,
        "yz_utilization": utilization,
        "yz_violation_score": violation,
        "origin_cos": origin,
        "bucket_name": bucket,
        "prompt_type": "chosen_prompt",
    }


class PairConstructionTests(unittest.TestCase):
    def test_routes_do_not_assign_final_roles(self):
        pool = [
            {
                **candidate("better", True, 0.01, 0.8, 0.0),
                "prompt_type": "rejected_prompt",
            },
            {
                **candidate("worse", False, 0.4, 0.0, 2.0),
                "prompt_type": "chosen_prompt",
            },
        ]
        chosen, rejected = assign_candidate_roles(
            pool, chosen_count=1, rejected_count=1
        )
        self.assertEqual(chosen[0]["candidate_text"], "better")
        self.assertEqual(rejected[0]["candidate_text"], "worse")

    def test_locked_margin_formula(self):
        chosen = candidate("chosen", True, 0.05, 0.8, 0.0)
        rejected = candidate("rejected", False, 0.40, 0.1, 2.0)
        expected = 10 + 2 * 0.35 + 2 * 2.0 + 0.7
        self.assertAlmostEqual(compute_pair_margin(chosen, rejected), expected)

    def test_aligned_pairing(self):
        pool = [
            candidate(f"chosen-{index}", True, 0.01 + index * 0.01, 0.8, 0.0)
            for index in range(6)
        ] + [
            candidate(f"rejected-{index}", False, 0.3, 0.1, 2.0 + index)
            for index in range(6)
        ]
        pairs = build_aligned_pairs_for_source(
            train_id=0,
            anchor_text="source",
            label=1,
            candidates=pool,
            chosen_count=6,
            rejected_count=6,
        )
        self.assertEqual(len(pairs), 6)
        self.assertEqual([row["pair_rank"] for row in pairs], list(range(6)))
        self.assertTrue(all(row["pair_mode"] == "aligned" for row in pairs))

    def test_frame_validation(self):
        pool = [
            candidate(f"chosen-{index}", True, 0.01, 0.8, 0.0)
            for index in range(5)
        ] + [
            candidate(f"rejected-{index}", False, 0.3, 0.1, 2.0)
            for index in range(5)
        ]
        pairs = build_aligned_pairs_for_source(
            train_id=0,
            anchor_text="source",
            label=1,
            candidates=pool,
            chosen_count=5,
            rejected_count=5,
        )
        frame = pd.DataFrame(pairs, columns=FULL_COLUMNS)
        summary = validate_pair_frame(
            frame,
            expected_sources=1,
            expected_pairs=5,
            minimum_pairs_per_source=5,
            maximum_pairs_per_source=5,
        )
        self.assertEqual(summary["pairs"], 5)


if __name__ == "__main__":
    unittest.main()
