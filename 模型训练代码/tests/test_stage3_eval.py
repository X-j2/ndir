# -*- coding: utf-8 -*-
"""Stage 3 engine evaluation: causal delay, scorable clipping, no-window FN."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from training.stage2_data import STATE_CODE  # noqa: E402
from training.stage3_eval import evaluate_engine_case  # noqa: E402
from training.stage3_event_engine import EventParams  # noqa: E402


def make_case(cid=1, starts=(0.0, 15.0), states=("VALID", "VALID")):
    return {
        "case_id": np.full(len(starts), cid, dtype=np.int32),
        "start": (np.asarray(starts, dtype=float) * 100).astype(np.int32),
        "state": np.asarray([STATE_CODE[s] for s in states], dtype=np.int8),
    }


def make_meta(total=100, scorable=((0, 100),), reliable=(), low=()):
    st = sum(e - s for s, e in scorable)
    return {
        "window_sec": 30.0, "step_sec": 15.0,
        "case_meta": {"1": {
            "total_sec": total, "scorable_intervals": [list(x) for x in scorable],
            "scorable_sec": st, "excluded_sec": total - st,
            "reference_unknown_excluded_sec": total - st,
            "low_support_excluded_sec": 0.0,
            "low_support_unknown_overlap_sec": 0.0,
            "reliable_u2_events": list(reliable),
            "low_support_u2_events": list(low),
        }},
    }


class TestStage3Evaluation(unittest.TestCase):
    def test_active_delay_uses_first_active_output(self):
        data = make_case()
        meta = make_meta(reliable=[{"start": 20, "end": 40}])
        params = EventParams(suspect_p=.1, active_enter_p=.5,
                             active_exit_p=.1, suspect_on_count=1,
                             suspect_off_count=1, active_on_count=1,
                             active_off_count=1)
        p = np.array([[0, 0, .95], [0, 0, .95]], dtype=float)
        per = evaluate_engine_case(data, meta, 1, np.array([0, 1]), p,
                                   params, "U30", 30.0, 15.0)
        self.assertEqual(per["tp"], 1)
        self.assertEqual(per["ttds"], [10.0])
        self.assertEqual(per["unknown_decisions"], 0)

    def test_no_window_truth_is_fn(self):
        data = make_case(starts=(), states=())
        meta = make_meta(reliable=[{"start": 10, "end": 20},
                                   {"start": 50, "end": 60}],
                         low=[{"start": 70, "end": 80}])
        params = EventParams()
        p = np.zeros((0, 3), dtype=float)
        per = evaluate_engine_case(data, meta, 1, np.empty(0, dtype=int), p,
                                   params, "U15", 15.0, 7.5)
        self.assertEqual(per["n_truth"], 2)
        self.assertEqual(per["fn"], 2)
        self.assertEqual(per["n_low_support_truth"], 1)

    def test_engine_unknown_closes_active(self):
        data = make_case(states=("VALID", "GAP"))
        meta = make_meta(reliable=[{"start": 0, "end": 50}])
        params = EventParams(active_on_count=1, active_off_count=1)
        p = np.array([[0, 0, .95], [0, 0, .95]], dtype=float)
        per = evaluate_engine_case(data, meta, 1, np.array([0, 1]), p,
                                   params, "U30", 30.0, 15.0)
        # Active output stops at the UNKNOWN decision (45s), not after it.
        self.assertGreaterEqual(per["unknown_decisions"], 1)
        self.assertEqual(per["low_confidence_active_episodes"], 0)

    def test_missing_decision_is_an_interruption_in_evaluation(self):
        # Decisions at 30 s and 150 s: the latter is a missing-decision gap.
        data = make_case(starts=(0.0, 120.0), states=("VALID", "VALID"))
        meta = make_meta(
            total=200, scorable=((0, 200),),
            reliable=[{"start": 20, "end": 60},
                      {"start": 145, "end": 175}])
        params = EventParams(active_on_count=1, active_off_count=1,
                             suspect_off_count=1)
        p = np.array([[0, 0, .95], [0, 0, .95]], dtype=float)
        per = evaluate_engine_case(data, meta, 1, np.array([0, 1]), p,
                                   params, "U30", 30.0, 15.0)
        self.assertEqual(per["gap_interruptions"], 1)
        self.assertEqual(per["interruptions"], 1)
        self.assertEqual(per["engine_episodes"], 2)
        self.assertEqual(per["tp"], 2)


if __name__ == "__main__":
    unittest.main()
