# -*- coding: utf-8 -*-
"""Boundary tests for causal quality/abstention states."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from stage0_common import FS
from quality_state import (
    A_FLOOR_DEFAULT, classify_quality_state, history_from_bins,
    summarize_history,
)


def normal_wave(duration_sec, rr=15.0, etco2=40.0, base=5.0):
    t = np.arange(int(duration_sec * FS)) / FS
    x = base + (etco2 - base) * (np.sin(2 * np.pi * rr / 60.0 * t) > 0.3).astype(float)
    return x


def flat_wave(duration_sec, value=5.0):
    return np.full(int(duration_sec * FS), value, dtype=float)


class TestQualityStateCausal(unittest.TestCase):
    def test_flat_with_verified_history(self):
        prior = normal_wave(60)
        h = summarize_history(prior)
        self.assertTrue(h.available)
        self.assertTrue(h.has_breathing)
        w = flat_wave(15)
        self.assertEqual(classify_quality_state(w, h), "FLAT_WITH_HISTORY")

    def test_flat_with_no_history_breathing(self):
        prior = flat_wave(60)
        h = summarize_history(prior)
        self.assertTrue(h.available)
        self.assertFalse(h.has_breathing)
        w = flat_wave(15)
        self.assertEqual(classify_quality_state(w, h), "FLAT_NO_HISTORY")

    def test_flat_with_insufficient_history(self):
        h = summarize_history(flat_wave(10))
        self.assertFalse(h.available)
        self.assertIsNone(h.has_breathing)
        w = flat_wave(15)
        self.assertEqual(classify_quality_state(w, h), "FLAT_HISTORY_UNKNOWN")

    def test_flat_with_ever_but_not_recent_history(self):
        h = summarize_history(flat_wave(60), ever_has_breathing=True)
        self.assertFalse(h.has_breathing)
        self.assertTrue(h.ever_has_breathing)
        self.assertEqual(classify_quality_state(flat_wave(15), h),
                         "FLAT_EVER_HISTORY")

    def test_future_bin_is_not_history(self):
        bins = [
            {"start_s": 0, "end_s": 15, "A": 30.0, "peaks": 10, "has_breathing": True},
            {"start_s": 15, "end_s": 30, "A": 30.0, "peaks": 10, "has_breathing": True},
            {"start_s": 30, "end_s": 45, "A": 30.0, "peaks": 10, "has_breathing": True},
        ]
        # Query at 20 s: only the first bin is entirely in the past.
        h = history_from_bins(bins, 20.0, recent_sec=300, min_history_sec=15)
        self.assertTrue(h.available)
        self.assertTrue(h.ever_has_breathing)
        # Query at 10 s: no bin has ended yet; the future bins must not count.
        h2 = history_from_bins(bins, 10.0, recent_sec=300, min_history_sec=15)
        self.assertFalse(h2.available)
        self.assertIsNone(h2.ever_has_breathing)

    def test_future_breathing_does_not_change_state(self):
        # The function only receives the current window and a past summary.
        # A future normal segment is not part of the call; this test documents
        # that a flat window after a flat past must remain FLAT_NO_HISTORY.
        prior_flat = flat_wave(60)
        future_normal = normal_wave(60)  # deliberately not passed
        h = summarize_history(prior_flat)
        self.assertFalse(h.has_breathing)
        state = classify_quality_state(flat_wave(15), h)
        self.assertEqual(state, "FLAT_NO_HISTORY")
        self.assertNotEqual(state, "FLAT_WITH_HISTORY")
        self.assertEqual(len(future_normal), 60 * FS)

    def test_gap(self):
        w = normal_wave(15)
        w[300:800] = np.nan  # 5 s gap
        h = summarize_history(normal_wave(60))
        self.assertEqual(classify_quality_state(w, h), "GAP")

    def test_saturated(self):
        w = normal_wave(15)
        h = summarize_history(normal_wave(60))
        self.assertEqual(classify_quality_state(w, h, saturated=True), "SATURATED")

    def test_polarity_invalid(self):
        w = normal_wave(15)
        h = summarize_history(normal_wave(60))
        self.assertEqual(classify_quality_state(w, h, polarity_invalid=True),
                         "POLARITY_INVALID")

    def test_noisy_nonflat(self):
        w = normal_wave(15)
        h = summarize_history(normal_wave(60))
        self.assertEqual(classify_quality_state(w, h, noise_ok=False), "NOISY")

    def test_flat_is_not_noisy(self):
        # A clean flat line has low HF noise; it must not be rejected as NOISY
        # before the history branch is considered.
        w = flat_wave(15)
        h = summarize_history(normal_wave(60))
        self.assertEqual(classify_quality_state(w, h, noise_ok=False),
                         "FLAT_WITH_HISTORY")

    def test_history_summary_uses_only_past(self):
        # Smoke test: prior length is the only history input; no case id or
        # future argument exists in the API.
        h = summarize_history(normal_wave(120))
        self.assertEqual(h.n_samples, 120 * FS)
        self.assertGreaterEqual(h.A, A_FLOOR_DEFAULT)
        self.assertGreaterEqual(h.peaks, 2)


if __name__ == "__main__":
    unittest.main()
