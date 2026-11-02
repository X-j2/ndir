# -*- coding: utf-8 -*-
"""Stage 4 waveform perturbation invariants and time-axis preservation."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.si_features import extract_si_features  # noqa: E402
from training.stage4_perturb import (  # noqa: E402
    perturb_signal, perturb_signal_detail,
)


def wave(n=1500):
    t = np.arange(n) / 100.0
    return 20.0 + 15.0 * (np.sin(2 * np.pi * 0.25 * t) > 0) + 1.0 * np.sin(
        2 * np.pi * 0.5 * t)


class TestStage4Perturb(unittest.TestCase):
    def test_length_and_time_axis_preserved(self):
        x = wave()
        for kind, level in (("gain", 2.0), ("bias", 0.5), ("noise", 1.0),
                            ("drop", 0.05), ("burst_drop", 1.0),
                            ("lag", 2.0), ("drift", 0.5), ("gamma", 1.25),
                            ("clip", 1.0), ("negative", -1.0),
                            ("quantize", 10)):
            y = perturb_signal(x, kind, level, 0)
            self.assertEqual(len(y), len(x), kind)
            self.assertTrue(np.isnan(y).sum() >= np.isnan(x).sum(), kind)

    def test_drop_is_nan_not_deleted(self):
        x = wave()
        y = perturb_signal(x, "drop", 0.10, 1)
        self.assertEqual(len(y), len(x))
        self.assertGreater(int(np.isnan(y).sum()), 0)
        self.assertTrue(np.all(np.isfinite(x)))

    def test_positive_affine_features_are_invariant(self):
        x = wave()
        base = np.asarray(extract_si_features(x)["vector"])
        for kind, level in (("gain", 0.5), ("gain", 2.0),
                            ("bias", 0.5), ("bias", -0.2)):
            y = perturb_signal(x, kind, level, 0)
            v = np.asarray(extract_si_features(y)["vector"])
            self.assertLess(float(np.max(np.abs(v - base))), 1e-6,
                            f"{kind} {level}")

    def test_gamma_is_finite_on_negative_z(self):
        x = wave() - 15.0
        y = perturb_signal(x, "gamma", 1.25, 0)
        self.assertTrue(np.all(np.isfinite(y)))

    def test_burst_drop_short_window_complete_gap_for_all_seeds(self):
        x = wave(1500)
        for seed in range(5):
            for sec, n in ((0.2, 20), (1.0, 100), (5.0, 500)):
                y, info = perturb_signal_detail(
                    x, "burst_drop", sec, seed, mode="window")
                self.assertEqual(info["status"], "ok")
                self.assertEqual(len(info["gaps"]), 1)
                g = info["gaps"][0]
                self.assertEqual(g["length"], n)
                self.assertGreaterEqual(g["start"], 0)
                self.assertLessEqual(g["end"], 1500)
                self.assertEqual(info["actual_gap_samples"], n)
                self.assertTrue(np.all(np.isnan(y[g["start"]:g["end"]])))
                self.assertEqual(int(np.isnan(y).sum()), n)

    def test_burst_drop_insufficient_window_does_not_insert(self):
        x = wave(100)
        y, info = perturb_signal_detail(
            x, "burst_drop", 5.0, 0, mode="window")
        self.assertEqual(info["status"], "insufficient_window")
        self.assertEqual(info["actual_gap_samples"], 0)
        self.assertEqual(int(np.isnan(y).sum()), 0)

    def test_burst_drop_preserves_existing_nan_and_counts_new_only(self):
        x = wave(1500)
        x[100:150] = np.nan
        y, info = perturb_signal_detail(
            x, "burst_drop", 1.0, 1, mode="window")
        self.assertTrue(np.all(np.isnan(y[100:150])))
        self.assertEqual(int(np.isnan(y).sum()),
                         50 + info["new_missing_samples"])
        self.assertLessEqual(info["new_missing_samples"], 100)

    def test_burst_drop_case_mode_tail_is_skipped(self):
        x = wave(12300)  # 123 s; the 120 s minute cannot fit a 5 s burst
        y, info = perturb_signal_detail(
            x, "burst_drop", 5.0, 0, mode="case")
        self.assertGreaterEqual(info["tail_skipped"], 1)
        for g in info["gaps"]:
            self.assertLessEqual(g["end"], len(x))
        self.assertLess(int(np.isnan(y).sum()), len(x))


if __name__ == "__main__":
    unittest.main()
