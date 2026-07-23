import unittest

from codipo_release.manuscript_analysis import build_figure4
from codipo_release.manuscript_analysis import build_figure7


class ManuscriptFigureTests(unittest.TestCase):
    def test_figure4_reports_adress_separately_from_external_average(self) -> None:
        cohort_names = [name for name, _, _, _ in build_figure4.COHORTS]
        self.assertIn("ADReSS validation", cohort_names)
        self.assertNotIn("External-cohort average", cohort_names)

    def test_figure7_uses_locked_cosine_display_name(self) -> None:
        self.assertEqual(build_figure7.display_name("Cos-X"), "Cosine sim. as X")
        self.assertEqual(
            build_figure7.PCA_TITLE_OVERRIDES["Cosine sim. as X"],
            "Cosine sim. as X",
        )

    def test_figure7_helpers_are_package_local(self) -> None:
        self.assertTrue(build_figure7.v3.__name__.endswith("._figure7_baseline"))
        self.assertTrue(build_figure7.abl.__name__.endswith("._figure7_ablation"))


if __name__ == "__main__":
    unittest.main()
