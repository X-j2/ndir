# -*- coding: utf-8 -*-
"""Tests for the frozen 15-dimensional SI feature extractor."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from stage0_common import FS
from features.si_features import (
    FEATURE_NAMES, N_FEATURES, SPEC, extract_si_features, feature_vector,
)
from tests.test_quality_state import flat_wave, normal_wave


def sparse_pulse_wave(duration_sec, value=5.0, amplitude=40.0,
                      pulse_secs=(3.0, 8.0, 12.0), pulse_len_sec=0.2):
    """Mostly constant trace with a few brief pulses.

    q10=q90=value because >80% of samples are constant, but finite samples
    still vary. It is a quantile-degenerate artifact, not a flat window.
    """
    x = flat_wave(duration_sec, value=value)
    width = int(pulse_len_sec * FS)
    for sec in pulse_secs:
        i = int(sec * FS)
        if i + width <= len(x):
            x[i:i + width] = value + amplitude
    return x


class TestSI103Features(unittest.TestCase):
    def test_schema_is_fifteen_dimensions(self):
        self.assertEqual(N_FEATURES, 15)
        self.assertEqual(len(FEATURE_NAMES), 15)
        self.assertEqual(len(set(FEATURE_NAMES)), 15)
        self.assertEqual(SPEC["version"], "si_feature_spec_v2")

    def test_positive_affine_invariance(self):
        base = normal_wave(30, rr=15.0)
        ref = extract_si_features(base)["vector"]
        for gain, offset in [(0.5, 0.0), (2.0, 10.0), (0.1, -3.0),
                             (5.0, 0.25)]:
            got = extract_si_features(gain * base + offset)["vector"]
            self.assertEqual(len(got), 15)
            self.assertLess(max(abs(a - b) for a, b in zip(ref, got)), 1e-6,
                            (gain, offset))

    def test_flat_a_zero_has_finite_neutral_vector(self):
        r = extract_si_features(flat_wave(15))
        self.assertEqual(len(r["vector"]), 15)
        self.assertTrue(all(np.isfinite(r["vector"])))
        self.assertTrue(r["A_invalid"])
        self.assertEqual(r["features"]["peak_count"], 0.0)
        self.assertEqual(r["features"]["peak_rate"], 0.0)
        self.assertEqual(r["features"]["flat_run_frac"], 1.0)
        self.assertEqual(r["features"]["band_ratio"], 0.0)
        self.assertEqual(r["features"]["acf_peak"], 0.0)
        self.assertTrue(r["constant_flat"])
        self.assertFalse(r["quantile_artifact"])
        self.assertEqual(r["finite_span"], 0.0)

    def test_sparse_pulse_is_quantile_artifact_not_constant_flat(self):
        x = sparse_pulse_wave(15)
        r = extract_si_features(x)
        self.assertEqual(r["A"], 0.0)
        self.assertTrue(r["A_invalid"])
        self.assertGreater(r["finite_span"], 0.0)
        self.assertFalse(r["constant_flat"])
        self.assertTrue(r["quantile_artifact"])
        self.assertEqual(len(r["vector"]), 15)
        self.assertTrue(all(np.isfinite(r["vector"])))

    def test_sparse_pulse_artifact_is_affine_invariant(self):
        x = sparse_pulse_wave(15)
        for gain, offset in [(0.01, 3.0), (10.0, -2.0)]:
            r = extract_si_features(gain * x + offset)
            self.assertEqual(r["A"], 0.0)
            self.assertTrue(r["quantile_artifact"], (gain, offset))
            self.assertFalse(r["constant_flat"])

    def test_normal_wave_has_peaks_and_rhythm(self):
        r = extract_si_features(normal_wave(30, rr=15.0))
        f = r["features"]
        self.assertFalse(r["A_invalid"])
        self.assertGreaterEqual(f["peak_count"], 3)
        self.assertAlmostEqual(f["peak_rate"], 15.0, delta=4.0)
        self.assertGreater(f["rhythm_confidence"], 0.8)
        self.assertGreater(f["acf_peak"], 0.5)
        self.assertEqual(len(feature_vector(normal_wave(30))), 15)
        self.assertEqual(feature_vector(normal_wave(30)).dtype, np.float32)

    def test_skewness_is_standardized_third_moment(self):
        x = normal_wave(30, rr=15.0)
        r = extract_si_features(x)
        A, B = r["A"], r["B"]
        z = (x - B) / A
        expected = float(np.mean((z - np.mean(z)) ** 3) / (np.std(z) ** 3))
        self.assertAlmostEqual(r["features"]["skewness"], expected, places=10)
        # Scaling the waveform must not change the standardized moment.
        scaled = extract_si_features(0.01 * x + 7.0)
        self.assertAlmostEqual(r["features"]["skewness"],
                               scaled["features"]["skewness"], places=10)

    def test_negative_gain_is_flagged_not_flipped(self):
        r = extract_si_features(-normal_wave(30, rr=15.0))
        self.assertTrue(r["polarity_invalid"])
        self.assertFalse(extract_si_features(normal_wave(30, rr=15.0))[
            "polarity_invalid"])

    def test_short_gap_is_filled(self):
        x = normal_wave(15)
        x[300:500] = np.nan  # 2 s, within the 200-sample fill limit
        r = extract_si_features(x)
        self.assertEqual(len(r["vector"]), 15)
        self.assertTrue(all(np.isfinite(r["vector"])))

    def test_long_gap_is_reported(self):
        x = normal_wave(15)
        x[300:800] = np.nan  # 5 s
        r = extract_si_features(x)
        self.assertGreater(r["max_nan_run_sec"], 2.0)
        self.assertGreater(r["nan_frac"], 0.0)


if __name__ == "__main__":
    unittest.main()
