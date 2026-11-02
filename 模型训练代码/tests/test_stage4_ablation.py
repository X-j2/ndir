# -*- coding: utf-8 -*-
"""Stage 4 ablation criterion tests."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.stage4_ablation import band_ratio_stability_decision  # noqa: E402


def conditions(gain_rho=1.0, bias_rho=1.0, gain_med=0.0, bias_med=0.0,
               noise_med=5.0, gamma_med=5.0):
    return {
        "gain0.5": {"spearman": gain_rho, "median_relative_change": gain_med},
        "gain2": {"spearman": gain_rho, "median_relative_change": gain_med},
        "bias+0.5": {"spearman": bias_rho, "median_relative_change": bias_med},
        "bias-0.5": {"spearman": bias_rho, "median_relative_change": bias_med},
        "noise1": {"spearman": 0.1, "median_relative_change": noise_med},
        "gamma1.25": {"spearman": 0.2,
                      "median_relative_change": gamma_med},
    }


class TestStage4AblationCriterion(unittest.TestCase):
    def test_noise_and_gamma_do_not_replace_gain_bias_criterion(self):
        d = band_ratio_stability_decision(conditions(
            noise_med=10.0, gamma_med=10.0))
        self.assertTrue(d["stable"])
        self.assertEqual(d["median_relative_change_gain_bias"], 0.0)

    def test_high_gain_bias_median_change_fails(self):
        d = band_ratio_stability_decision(conditions(gain_med=0.3))
        self.assertFalse(d["stable"])

    def test_low_gain_bias_rho_fails(self):
        d = band_ratio_stability_decision(conditions(bias_rho=0.5))
        self.assertFalse(d["stable"])


if __name__ == "__main__":
    unittest.main()
