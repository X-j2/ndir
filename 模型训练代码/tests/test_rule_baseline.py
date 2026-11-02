# -*- coding: utf-8 -*-
"""Tests for the Stage 1 B0 rule baseline and dev-only guard."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from features.rule_baseline import PREDICT_LABELS, UNKNOWN, predict_rule
from features.quality_states import classify_signal_state
from features.si_features import extract_si_features
from tools.quality_state import summarize_history
from tools.run_stage1_rule_baseline import load_dev_ids
from tests.test_quality_state import flat_wave, normal_wave
from tests.test_stage1_quality import sparse_pulse_wave


def feats(rate):
    return {"peak_rate": rate}


class TestRuleBaseline(unittest.TestCase):
    def test_unknown_quality_never_becomes_normal(self):
        for state in ("GAP", "SATURATED", "POLARITY_INVALID",
                      "QUANTILE_ARTIFACT", "NOISY",
                      "FLAT_NO_HISTORY", "FLAT_HISTORY_UNKNOWN"):
            r = predict_rule(feats(12.0), state)
            self.assertEqual(r["label"], UNKNOWN, state)

    def test_flat_with_history_is_u2_even_a_zero(self):
        r = predict_rule(feats(0.0), "FLAT_WITH_HISTORY")
        self.assertEqual(r["label"], "U2")
        self.assertTrue(r["flat_candidate"])
        self.assertEqual(r["confidence"], "high")

    def test_flat_ever_history_is_low_confidence_u2(self):
        r = predict_rule(feats(0.0), "FLAT_EVER_HISTORY")
        self.assertEqual(r["label"], "U2")
        self.assertEqual(r["confidence"], "low")

    def test_rate_thresholds(self):
        self.assertEqual(predict_rule(feats(24.0), "VALID")["label"], "U1")
        self.assertEqual(predict_rule(feats(18.0), "VALID")["label"], "U1")
        self.assertEqual(predict_rule(feats(17.9), "VALID")["label"], "U0")
        self.assertEqual(predict_rule(feats(7.0), "VALID")["label"], "U0")
        self.assertEqual(predict_rule(feats(6.9), "VALID")["label"], "U3")
        self.assertEqual(predict_rule(feats(3.0), "VALID")["label"], "U3")
        self.assertEqual(predict_rule(feats(2.9), "VALID")["label"], UNKNOWN)

    def test_a_zero_end_to_end_with_history_is_u2(self):
        w = flat_wave(15)
        diag = extract_si_features(w)
        state = classify_signal_state(
            w, summarize_history(normal_wave(60)), feature_diag=diag)
        r = predict_rule(diag["features"], state)
        self.assertEqual(r["label"], "U2")

    def test_a_zero_end_to_end_without_history_is_unknown(self):
        w = flat_wave(15)
        diag = extract_si_features(w)
        state = classify_signal_state(
            w, summarize_history(flat_wave(60)), feature_diag=diag)
        r = predict_rule(diag["features"], state)
        self.assertEqual(r["label"], UNKNOWN)
        self.assertNotEqual(r["label"], "U0")

    def test_sparse_pulse_with_history_is_unknown_not_u2(self):
        w = sparse_pulse_wave(15)
        diag = extract_si_features(w)
        state = classify_signal_state(
            w, summarize_history(normal_wave(60)), feature_diag=diag)
        r = predict_rule(diag["features"], state)
        self.assertEqual(state, "QUANTILE_ARTIFACT")
        self.assertEqual(r["label"], UNKNOWN)
        self.assertFalse(r["flat_candidate"])

    def test_prediction_space(self):
        for rate in (0, 3, 7, 18, 30):
            for state in ("VALID", "FLAT_WITH_HISTORY", "FLAT_NO_HISTORY"):
                self.assertIn(predict_rule(feats(rate), state)["label"],
                              PREDICT_LABELS)

    def test_dev_ids_are_200_and_disjoint_from_final(self):
        ids = load_dev_ids()
        self.assertEqual(len(ids), 200)
        self.assertEqual(len(set(ids)), 200)
        import json
        with open(ROOT / "data_splits" / "final_test_seed42.json",
                  "r", encoding="utf-8") as f:
            final = json.load(f)
        self.assertFalse(set(ids) & set(final["case_ids"]))


if __name__ == "__main__":
    unittest.main()
