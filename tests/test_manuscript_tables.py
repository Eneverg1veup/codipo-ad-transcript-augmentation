import argparse
import tempfile
import unittest
from pathlib import Path

from codipo_release.manuscript_analysis import build_tables


class ManuscriptTableTests(unittest.TestCase):
    def test_locked_display_names(self) -> None:
        self.assertEqual(
            build_tables.METHOD_LABELS["CoDiPO w/o X"],
            r"Without \(X\)",
        )
        self.assertEqual(
            build_tables.METHOD_LABELS["CoDiPO w/o YZ"],
            r"Without \(Y\) and \(Z\)",
        )
        self.assertEqual(
            build_tables.METHOD_LABELS["w/o residual decomposition"],
            r"Cosine sim. as \(X\)",
        )

    def test_configure_requires_positive_bootstrap_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = argparse.Namespace(
                statistics_dir=Path(temp_dir) / "statistics",
                output_dir=Path(temp_dir) / "tables",
                n_bootstrap=0,
            )
            with self.assertRaisesRegex(ValueError, "n_bootstrap"):
                build_tables.configure(args)


if __name__ == "__main__":
    unittest.main()
