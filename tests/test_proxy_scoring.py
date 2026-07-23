import unittest

import numpy as np

from codipo_release.proxy_scoring.core import (
    ProxySettings,
    candidate_bound_check,
    compute_dynamic_bounds,
    compute_yz,
    estimate_residual_projection,
    fuse_clip_scores,
    residual_similarity,
)
from codipo_release.proxy_scoring.clip_coverage import split_text_smartly
from codipo_release.proxy_scoring.score_candidates import map_prompt_type


class ProxyScoringTests(unittest.TestCase):
    def test_yz_uses_threshold_excess(self):
        result = compute_yz([0.18, 0.20, 0.25], threshold=0.19)
        self.assertAlmostEqual(result["Y"], 2 / 3)
        self.assertAlmostEqual(result["Z"], 0.035)

    def test_clip_fusion_matches_manuscript(self):
        fused = fuse_clip_scores([0.1, 0.2], [0.3, 0.4], chunk_weight=0.8)
        np.testing.assert_allclose(fused, [0.26, 0.36])

    def test_residual_projection_removes_class_direction(self):
        ad = [[2.0, 0.0], [4.0, 0.0]]
        hc = [[-2.0, 0.0], [-4.0, 0.0]]
        center, direction = estimate_residual_projection(ad, hc)
        value = residual_similarity([1.0, 1.0], [-1.0, 2.0], center, direction)
        self.assertAlmostEqual(value, 1.0)

    def test_directional_bounds(self):
        stats = {"y_scale": 2.0, "z_scale": 4.0}
        ad = compute_dynamic_bounds(10, 20, 1, stats)
        self.assertAlmostEqual(ad["eps_y_plus"], 0.8)
        self.assertAlmostEqual(ad["eps_y_minus"], 2.0)
        hc = compute_dynamic_bounds(10, 20, 0, stats)
        self.assertAlmostEqual(hc["eps_y_plus"], 2.0)
        self.assertAlmostEqual(hc["eps_y_minus"], 0.8)
        self.assertEqual(ProxySettings().risk_band_multiplier, 0.4)

    def test_bound_violation(self):
        bounds = {
            "y_i": 0.5,
            "z_i": 0.1,
            "eps_y_plus": 0.1,
            "eps_y_minus": 0.2,
            "eps_z_plus": 0.05,
            "eps_z_minus": 0.05,
            "y_lower_i": 0.3,
            "y_upper_i": 0.6,
            "z_lower_i": 0.05,
            "z_upper_i": 0.15,
        }
        result = candidate_bound_check(0.7, 0.1, bounds)
        self.assertFalse(result["yz_pass"])
        self.assertAlmostEqual(result["y_violate_high"], 1.0)

    def test_chunking_respects_eight_word_limit(self):
        chunks = split_text_smartly(
            "one two three four five six seven eight nine ten.", max_words=8
        )
        self.assertGreaterEqual(len(chunks), 2)

    def test_proposal_route_mapping(self):
        self.assertEqual(
            map_prompt_type({"prompt_route": "chosen"}), "chosen_prompt"
        )
        self.assertEqual(
            map_prompt_type({"prompt_route": "rejected"}), "rejected_prompt"
        )


if __name__ == "__main__":
    unittest.main()
