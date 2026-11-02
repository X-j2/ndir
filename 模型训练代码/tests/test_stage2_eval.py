# -*- coding: utf-8 -*-
"""Stage 2 v2 causal eventization, scorable clipping and B0 tests."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from features.rule_baseline import predict_rule  # noqa: E402
from features.si_features import FEATURE_NAMES  # noqa: E402
from training.stage2_data import STATE_CODE  # noqa: E402
from training.stage2_eval import (  # noqa: E402
    aggregate_events, b0_predict, evaluate_events_for_cases,
    match_active_causal, metrics_from_window_confusion,
    raw_decision_episodes, scored_decision_alarms, select_threshold,
    window_confusion,
)


def make_data(case_id, starts_s, states, features=None):
    n = len(starts_s)
    if features is None:
        features = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    return {
        "case_id": np.full(n, case_id, dtype=np.int32),
        "start": (np.asarray(starts_s, dtype=float) * 100).astype(np.int32),
        "state": np.asarray([STATE_CODE[s] for s in states], dtype=np.int8),
        "features": np.asarray(features, dtype=np.float32),
        "ref_valid_frac": np.ones(n, dtype=np.float32),
    }


def make_meta(total, scorable, reliable=None, low=None, step=15.0):
    return {
        "step_sec": float(step),
        "window_sec": float(step),
        "case_meta": {
            "1": {
                "total_sec": float(total),
                "scorable_intervals": [list(x) for x in scorable],
                "scorable_sec": float(sum(e - s for s, e in scorable)),
                "excluded_sec": float(total - sum(e - s for s, e in scorable)),
                "reference_unknown_excluded_sec": 0.0,
                "low_support_excluded_sec": float(
                    total - sum(e - s for s, e in scorable)),
                "low_support_unknown_overlap_sec": 0.0,
                "reliable_u2_events": reliable or [],
                "low_support_u2_events": low or [],
            }
        },
    }


def eval_one(data, meta, rows, p, threshold=0.5):
    return evaluate_events_for_cases(
        data, meta, {1: rows}, [1], p_u2=np.asarray(p),
        threshold=threshold, policy="rf")


class TestCausalEventization(unittest.TestCase):
    def test_window_end_is_decision_time(self):
        data = make_data(1, [0], ["VALID"])
        meta = make_meta(60, [(0, 60)],
                         reliable=[{"start": 20, "end": 40}], step=30)
        per = eval_one(data, meta, np.array([0]), [0.9])
        ev = aggregate_events(per, [1])
        self.assertEqual(ev["tp"], 1)
        self.assertEqual(ev["ttd_s"]["n_early"], 0)
        self.assertEqual(ev["ttd_s"]["median"], 10.0)
        self.assertEqual(ev["ttd_s"]["median_nonnegative"], 10.0)

    def test_positive_holds_one_step_only_and_does_not_merge_negative_gap(self):
        starts = np.array([0.0, 10.0, 20.0, 30.0])
        pos = np.array([True, True, False, True])
        eps = raw_decision_episodes(starts + 10.0, pos, 10.0, 100.0)
        self.assertEqual(len(eps), 2)
        self.assertEqual(eps[0]["start"], 10.0)
        self.assertEqual(eps[0]["end"], 30.0)
        self.assertEqual(eps[1]["start"], 40.0)

    def test_low_support_only_alarm_is_unscorable_not_fp(self):
        data = make_data(1, [0], ["VALID"])
        # decision 10, hold [10,20), low-support [10,20) -> no scorable part.
        meta = make_meta(30, [(0, 10), (20, 30)],
                         reliable=[], low=[{"start": 10, "end": 20}], step=10)
        per = eval_one(data, meta, np.array([0]), [0.9])
        self.assertEqual(per[1]["fp"], 0)
        self.assertEqual(per[1]["unscorable_alarm_episodes"], 1)
        ev = aggregate_events(per, [1])
        self.assertEqual(ev["fp"], 0)
        self.assertEqual(ev["unscorable_alarm_episodes"], 1)

    def test_alarm_crossing_unscorable_is_split_and_keeps_decision_time(self):
        data = make_data(1, [0], ["VALID"])
        # decision 15, hold [15,45), scorable [0,20) and [30,50)
        meta = make_meta(50, [(0, 20), (30, 50)],
                         reliable=[], low=[{"start": 20, "end": 30}],
                         step=30)
        out = scored_decision_alarms(
            np.array([15.0]), np.array([True]), 30.0, 50.0,
            [(0.0, 20.0), (30.0, 50.0)], 1, "ACTIVE")
        self.assertEqual(len(out["alarms"]), 2)
        self.assertEqual(out["alarms"][0]["start"], 15.0)
        self.assertEqual(out["alarms"][0]["end"], 20.0)
        self.assertEqual(out["alarms"][1]["start"], 30.0)
        self.assertEqual(out["alarms"][1]["end"], 45.0)
        self.assertTrue(all(a["decision_time"] == 15.0 for a in out["alarms"]))

    def test_no_window_case_truth_is_followed_as_fn(self):
        data = make_data(1, [], [])
        meta = make_meta(100, [(0, 100)],
                         reliable=[{"start": 10, "end": 20},
                                   {"start": 50, "end": 60}],
                         low=[{"start": 80, "end": 90}])
        per = eval_one(data, meta, np.empty(0, dtype=int), [])
        self.assertEqual(per[1]["n_truth"], 2)
        self.assertEqual(per[1]["fn"], 2)
        self.assertEqual(per[1]["n_truth_no_windows"], 2)
        ev = aggregate_events(per, [1])
        self.assertEqual(ev["event_recall"], 0.0)
        self.assertEqual(ev["n_low_support_events"], 1)

    def test_match_uses_decision_time_for_ordering(self):
        truth = [{"case_id": 1, "class": "U2", "start": 20, "end": 40}]
        alarms = [{"case_id": 1, "class": "U2", "start": 30, "end": 50,
                   "decision_time": 30}]
        r = match_active_causal(truth, alarms)
        self.assertEqual(r["tp"], 1)
        self.assertEqual(r["ttd_s"], [10.0])

    def test_threshold_selection_uses_causal_time(self):
        data = make_data(1, [0], ["VALID"])
        meta = make_meta(60, [(0, 60)],
                         reliable=[{"start": 20, "end": 40}], step=30)
        sel = select_threshold(
            data, meta, {1: np.array([0])}, np.array([0.9]), [1],
            candidates=np.array([0.5, 0.99]))
        self.assertEqual(sel["threshold"], 0.5)
        self.assertEqual(sel["inner_metrics"]["ttd_s"]["median"], 10.0)


class TestB0AndWindowMetrics(unittest.TestCase):
    def test_b0_keeps_u3_as_non_u2_prediction(self):
        y = np.array([0, 1, 2, 0])
        pred = np.array([3, -1, 2, 0])
        m = metrics_from_window_confusion(window_confusion(y, pred))
        self.assertEqual(m["covered"], 3)  # U3 is covered but wrong
        self.assertEqual(m["u2_binary"]["fp"], 0)
        self.assertAlmostEqual(m["u2_binary"]["recall"], 1.0)

    def test_vectorized_b0_matches_rule(self):
        n = 5
        feats = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
        feats[:, FEATURE_NAMES.index("peak_rate")] = [24, 12, 4, 2, 0]
        data = make_data(
            1, [0, 10, 20, 30, 40],
            ["VALID", "VALID", "VALID", "VALID", "FLAT_WITH_HISTORY"],
            feats)
        got = b0_predict(data)
        want = []
        for i in range(n):
            d = {name: float(feats[i, j])
                 for j, name in enumerate(FEATURE_NAMES)}
            state = ["VALID", "VALID", "VALID", "VALID",
                     "FLAT_WITH_HISTORY"][i]
            r = predict_rule(d, state)["label"]
            want.append({"U0": 0, "U1": 1, "U2": 2, "U3": 3,
                         "UNKNOWN": -1}[r])
        np.testing.assert_array_equal(got, want)


if __name__ == "__main__":
    unittest.main()
