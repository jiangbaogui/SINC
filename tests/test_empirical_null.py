import tempfile
import unittest
from pathlib import Path

import numpy as np

from onboard.core.empirical_null import (
    apply_empirical_null_thresholds,
    estimate_empirical_null_thresholds,
)


class EmpiricalNullTests(unittest.TestCase):
    def test_empirical_null_round_trip(self):
        rng = np.random.default_rng(42)
        sample_count = 6000
        samples = {
            "d2": rng.gamma(3.0, 1.0, sample_count),
            "cfar": rng.gamma(2.0, 0.5, sample_count),
            "sam": rng.beta(2.0, 12.0, sample_count),
            "dynamic": rng.exponential(0.01, sample_count),
        }
        thresholds = estimate_empirical_null_thresholds(
            samples, n_bands=7, alpha=0.01
        )
        with tempfile.TemporaryDirectory() as directory:
            path = thresholds.save(Path(directory) / "thresholds.json")
            self.assertTrue(path.exists())
        self.assertEqual(thresholds.alpha, 0.01)
        self.assertGreater(thresholds.fusion_threshold, 0)

    def test_joint_decision_matches_score_threshold(self):
        thresholds = {
            "schema_version": 4,
            "decision_rule": "global_empirical_null_d2_gate_with_cfar_or_sam",
            "alpha": 0.01,
            "d2_threshold": 2.0,
            "cfar_threshold": 3.0,
            "sam_threshold": 0.2,
            "empirical_sam_threshold": 0.2,
            "sample_count": 2000,
            "stable_sample_count": 1900,
            "null_sample_count": 1800,
            "n_bands": 7,
            "preliminary_null_limit": 10.0,
            "robust_scale": 1.0,
            "fusion_threshold": 1.0,
            "reference_quantile": 0.95,
        }
        d2 = np.array([[3.0, 3.0, 1.0]], dtype=np.float32)
        cfar = np.array([[4.0, 1.0, 4.0]], dtype=np.float32)
        sam = np.array([[0.1, 0.3, 0.3]], dtype=np.float32)
        valid = np.ones_like(d2, dtype=bool)
        maps, _ = apply_empirical_null_thresholds(
            d2, cfar, sam, valid, thresholds
        )
        np.testing.assert_array_equal(maps["mask"], np.array([[1, 1, 0]]))
        np.testing.assert_array_equal(
            maps["fusion"] > 1.0, maps["mask"].astype(bool)
        )


if __name__ == "__main__":
    unittest.main()
