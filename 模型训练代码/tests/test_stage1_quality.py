# -*- coding: utf-8 -*-
"""Stage 1 flat / disconnect / saturation state tests."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from stage0_common import FS
from quality_state import history_from_bins, summarize_history
from features.quality_states import (
    HistoryIndex, UNKNOWN_STATES, classify_signal_state,
    detect_saturation, si_breathing_bins, state_is_unknown,
)
from features.si_features import extract_si_features
from tests.test_quality_state import flat_wave, normal_wave


def sparse_pulse_wave(duration_sec, value=5.0, amplitude=40.0,
                      pulse_secs=(3.0, 8.0, 12.0), pulse_len_sec=0.2):
    x = flat_wave(duration_sec, value=value)
    width = int(pulse_len_sec * FS)
    for sec in pulse_secs:
        i = int(sec * FS)
        if i + width <= len(x):
            x[i:i + width] = value + amplitude
    return x


def state_of(window, prior, rail_low=None, rail_high=None):
    diag = extract_si_features(window)
    hist = HistoryIndex(si_breathing_bins(prior)).summary(len(prior) / FS)
    return classify_signal_state(
        window, hist, feature_diag=diag,
        rail_low=rail_low, rail_high=rail_high)


class TestStage1Quality(unittest.TestCase):
    def test_flat_zero_with_history_is_candidate_not_unknown(self):
        s = state_of(flat_wave(15), normal_wave(60))
        self.assertEqual(s, "FLAT_WITH_HISTORY")
        self.assertFalse(state_is_unknown(s))
        self.assertNotIn(s, UNKNOWN_STATES)

    def test_flat_zero_without_history_is_unknown(self):
        s = state_of(flat_wave(15), flat_wave(60))
        self.assertEqual(s, "FLAT_NO_HISTORY")
        self.assertTrue(state_is_unknown(s))

    def test_flat_zero_insufficient_history_is_unknown(self):
        s = state_of(flat_wave(15), flat_wave(10))
        self.assertEqual(s, "FLAT_HISTORY_UNKNOWN")
        self.assertTrue(state_is_unknown(s))

    def test_ever_history_is_low_confidence_not_unknown(self):
        h = summarize_history(flat_wave(60), ever_has_breathing=True)
        s = classify_signal_state(flat_wave(15), h,
                                  feature_diag=extract_si_features(flat_wave(15)))
        self.assertEqual(s, "FLAT_EVER_HISTORY")
        self.assertFalse(state_is_unknown(s))

    def test_disconnect_gap_is_unknown(self):
        w = normal_wave(15)
        w[300:800] = np.nan  # 5 s interruption
        s = state_of(w, normal_wave(60))
        self.assertEqual(s, "GAP")
        self.assertTrue(state_is_unknown(s))

    def test_saturation_is_not_flat(self):
        w = normal_wave(15)
        rail = float(np.percentile(w, 95))
        w[w >= rail] = rail
        self.assertTrue(detect_saturation(w, rail_high=rail))
        s = state_of(w, normal_wave(60), rail_high=rail)
        self.assertEqual(s, "SATURATED")
        self.assertTrue(state_is_unknown(s))

    def test_polarity_invalid_is_unknown(self):
        s = state_of(-normal_wave(15), normal_wave(60))
        self.assertEqual(s, "POLARITY_INVALID")
        self.assertTrue(state_is_unknown(s))

    def test_high_noise_is_unknown(self):
        rng = np.random.default_rng(7)
        w = rng.normal(0, 10, 15 * FS)
        s = state_of(w, normal_wave(60))
        self.assertIn(s, {"NOISY", "POLARITY_INVALID"})
        self.assertTrue(state_is_unknown(s))

    def test_valid_normal_window_is_usable(self):
        s = state_of(normal_wave(15), normal_wave(60))
        self.assertEqual(s, "VALID")
        self.assertFalse(state_is_unknown(s))

    def test_history_index_matches_list_function(self):
        bins = [
            {"start_s": s, "end_s": s + 15, "A": 30.0, "peaks": 10,
             "has_breathing": s < 60}
            for s in range(0, 150, 15)
        ]
        idx = HistoryIndex(bins)
        for start in (10.0, 30.0, 75.0, 140.0, 200.0):
            a = idx.summary(start)
            b = history_from_bins(bins, start, min_history_sec=30)
            self.assertEqual(a.available, b.available)
            self.assertEqual(a.has_breathing, b.has_breathing)
            self.assertEqual(a.ever_has_breathing, b.ever_has_breathing)
            self.assertAlmostEqual(a.lookback_sec, b.lookback_sec)

    def test_si_history_uses_si_peaks(self):
        normal = si_breathing_bins(normal_wave(60))
        flat = si_breathing_bins(flat_wave(60))
        self.assertTrue(any(b["has_breathing"] for b in normal))
        self.assertTrue(all(b["quality_ok"] for b in normal[1:]))
        self.assertFalse(any(b["has_breathing"] for b in flat))

    def test_si_history_excludes_gap_bin(self):
        x = normal_wave(60)
        x[15 * FS:30 * FS] = np.nan
        bins = si_breathing_bins(x)
        self.assertFalse(bins[1]["quality_ok"])
        self.assertFalse(bins[1]["has_breathing"])

    def test_si_history_excludes_noisy_bin(self):
        x = normal_wave(60)
        t = np.arange(len(x)) / FS
        x[15 * FS:30 * FS] += 20.0 * np.sin(2 * np.pi * 10.0 * t[15 * FS:30 * FS])
        bins = si_breathing_bins(x)
        self.assertFalse(bins[1]["quality_ok"])
        self.assertFalse(bins[1]["has_breathing"])

    def test_si_history_excludes_saturated_bin(self):
        x = normal_wave(60)
        rail = float(np.percentile(x, 95))
        x[x >= rail] = rail
        bins = si_breathing_bins(x, rail_high=rail)
        self.assertTrue(all(not b["has_breathing"] for b in bins))

    def test_si_history_is_gain_invariant(self):
        a = si_breathing_bins(normal_wave(60))
        b = si_breathing_bins(10.0 * normal_wave(60) + 25.0)
        self.assertEqual([x["has_breathing"] for x in a],
                         [x["has_breathing"] for x in b])
        self.assertEqual([x["peaks"] for x in a], [x["peaks"] for x in b])
        for x, y in zip(a, b):
            if np.isfinite(x["A"]) and np.isfinite(y["A"]):
                self.assertAlmostEqual(y["A"] / x["A"], 10.0, places=4)

    def test_normal_state_is_scale_invariant(self):
        # A fixed A<1 floor used to flag a 0.01x normal wave as flat. The
        # relative-history rule must keep it VALID at both gains.
        prior = normal_wave(60)
        window = normal_wave(15)
        for gain in (0.01, 1.0, 10.0):
            hp = HistoryIndex(si_breathing_bins(gain * prior))
            diag = extract_si_features(gain * window)
            s = classify_signal_state(window * gain, hp.summary(60.0),
                                      feature_diag=diag)
            self.assertEqual(s, "VALID", gain)

    def test_relative_low_amplitude_with_history_is_flat(self):
        prior = normal_wave(60)
        low = 0.01 * normal_wave(15)
        hp = HistoryIndex(si_breathing_bins(prior))
        s = classify_signal_state(low, hp.summary(60.0),
                                  feature_diag=extract_si_features(low))
        self.assertEqual(s, "FLAT_WITH_HISTORY")

    def test_a_ref_ignores_zero_peak_drift_bin(self):
        # A slow drift bin can have a large A but no SI peaks. It must not
        # raise A_ref, otherwise a normal low-amplitude window could be
        # misclassified relative to the drift.
        x = normal_wave(60)
        x[15 * FS:30 * FS] = np.linspace(0.0, 200.0, 15 * FS)
        bins = si_breathing_bins(x)
        self.assertFalse(bins[1]["has_breathing"])
        self.assertFalse(np.isfinite(bins[1]["A"]))
        self.assertTrue(np.isfinite(bins[1]["A_raw"]))
        h = HistoryIndex(bins).summary(60.0)
        self.assertIsNotNone(h.A)
        self.assertLess(h.A, 100.0)

    def test_history_index_a_ref_uses_breathing_bins_only(self):
        bins = [
            {"start_s": 0, "end_s": 15, "A": 1000.0, "peaks": 0,
             "has_breathing": False},
            {"start_s": 15, "end_s": 30, "A": 30.0, "peaks": 10,
             "has_breathing": True},
        ]
        h = HistoryIndex(bins).summary(30.0)
        self.assertEqual(h.A, 30.0)

    def test_nonzero_low_amplitude_noise_is_noisy(self):
        rng = np.random.default_rng(11)
        prior = normal_wave(60)
        low_noisy = 0.01 * normal_wave(15) + rng.normal(0, 0.2, 15 * FS)
        diag = extract_si_features(low_noisy)
        self.assertFalse(diag["A_invalid"])
        hp = HistoryIndex(si_breathing_bins(prior))
        s = classify_signal_state(low_noisy, hp.summary(60.0),
                                  feature_diag=diag)
        self.assertEqual(s, "NOISY")

    def test_exact_a_zero_with_history_is_candidate(self):
        prior = normal_wave(60)
        w = flat_wave(15, value=0.0)
        diag = extract_si_features(w)
        self.assertTrue(diag["A_invalid"])
        self.assertTrue(diag["constant_flat"])
        hp = HistoryIndex(si_breathing_bins(prior))
        s = classify_signal_state(w, hp.summary(60.0), feature_diag=diag)
        self.assertEqual(s, "FLAT_WITH_HISTORY")

    def test_sparse_pulse_with_history_is_artifact_not_candidate(self):
        # q90-q10=0 because most samples are constant, but the window still
        # contains pulses. It must not be treated as a flat apnea candidate
        # merely because A=0 and history exists.
        prior = normal_wave(60)
        w = sparse_pulse_wave(15)
        diag = extract_si_features(w)
        self.assertEqual(diag["A"], 0.0)
        self.assertFalse(diag["constant_flat"])
        self.assertTrue(diag["quantile_artifact"])
        hp = HistoryIndex(si_breathing_bins(prior))
        s = classify_signal_state(w, hp.summary(60.0), feature_diag=diag)
        self.assertEqual(s, "QUANTILE_ARTIFACT")
        self.assertIn(s, UNKNOWN_STATES)
        self.assertTrue(state_is_unknown(s))

    def test_sparse_pulse_without_history_is_artifact_not_flat(self):
        w = sparse_pulse_wave(15)
        diag = extract_si_features(w)
        s = classify_signal_state(
            w, HistoryIndex(si_breathing_bins(flat_wave(60))).summary(60.0),
            feature_diag=diag)
        self.assertEqual(s, "QUANTILE_ARTIFACT")
        self.assertTrue(state_is_unknown(s))

    def test_si_history_excludes_quantile_artifact_bin(self):
        x = sparse_pulse_wave(
            60, pulse_secs=(3, 8, 12, 18, 23, 27, 33, 38, 42, 48, 53, 57))
        bins = si_breathing_bins(x)
        self.assertTrue(bins)
        self.assertTrue(all(b["quantile_artifact"] for b in bins))
        self.assertTrue(all(not b["quality_ok"] for b in bins))
        self.assertTrue(all(not b["has_breathing"] for b in bins))

    def test_modules_import_from_repository_root(self):
        import subprocess
        code = "import features.si_features, features.quality_states, features.rule_baseline"
        r = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
