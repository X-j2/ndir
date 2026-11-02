# -*- coding: utf-8 -*-
"""Boundary tests for Stage 0 event construction."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from stage0_common import (
    FS, build_events, build_u2_events, classify_window, count_actual_readings, iter_windows,
    merge_intervals_no_unknown, split_events_by_support,
)


def make_track(duration_sec, segments, fill=30.0):
    """segments: list of (start_s, end_s, value_or_nan)."""
    x = np.full(int(duration_sec * FS), fill, dtype=float)
    for s, e, v in segments:
        x[int(s * FS):int(e * FS)] = v
    return x


class TestEventMergeUnknown(unittest.TestCase):
    def test_observed_gap_merges(self):
        # Two 6 s low runs separated by 5 s observed normal -> merge to 17 s.
        x = make_track(17, [(0, 6, 5.0), (11, 17, 5.0)])
        ev = build_events(x, threshold=10.0, cap_sec=10, min_sec=10, merge_gap_sec=10)
        self.assertEqual(ev["events"], [(0, 17)])
        self.assertEqual(ev["merge_blocked_by_unknown"], 0)

    def test_short_low_runs_with_observed_gap_merge_before_drop(self):
        # Two 6 s runs + 5 s observed gap: separately they are <10 s and would
        # be dropped; merged first, they are kept.
        x = make_track(17, [(0, 6, 5.0), (11, 17, 5.0)])
        ev = build_events(x, threshold=10.0, cap_sec=10, min_sec=10, merge_gap_sec=10)
        self.assertEqual(len(ev["events"]), 1)
        self.assertEqual(ev["events"][0][1] - ev["events"][0][0], 17)

    def test_gap_with_unknown_does_not_merge(self):
        # cap=5: 7 s gap leaves 2 s unknown. Gap 7 s <= merge_gap 10 s, but
        # unknown blocks the merge; two 12 s runs remain separate.
        x = make_track(31, [(0, 12, 5.0), (12, 19, np.nan), (19, 31, 5.0)])
        ev = build_events(x, threshold=10.0, cap_sec=5, min_sec=10, merge_gap_sec=10)
        self.assertEqual(len(ev["events"]), 2, ev)
        self.assertGreaterEqual(ev["merge_blocked_by_unknown"], 1)

    def test_short_runs_with_unknown_gap_are_dropped(self):
        # Minimal counterexample requested: two low runs, gap <=10 s, but the
        # gap contains unknown. Without merge, both are <10 s and must be
        # dropped; they must NOT become one event.
        x = make_track(15, [(0, 4, 5.0), (4, 11, np.nan), (11, 15, 5.0)])
        ev = build_events(x, threshold=10.0, cap_sec=5, min_sec=10, merge_gap_sec=10)
        self.assertEqual(ev["events"], [], ev)
        self.assertGreaterEqual(ev["merge_blocked_by_unknown"], 1)

    def test_rr_sensitivity_gap_with_unknown_does_not_merge(self):
        # RR<3 uses cap=10, merge=20. A 15 s gap leaves 5 s unknown; the
        # 15 s gap is <=20 s but must not bridge the two events.
        x = make_track(39, [(0, 12, 2.0), (12, 27, np.nan), (27, 39, 2.0)])
        ev = build_events(x, threshold=3.0, cap_sec=10, min_sec=10, merge_gap_sec=20)
        self.assertEqual(len(ev["events"]), 2, ev)
        self.assertGreaterEqual(ev["merge_blocked_by_unknown"], 1)

    def test_merge_function_direct(self):
        merged, blocked = merge_intervals_no_unknown(
            [(0, 6), (11, 17)], np.array([False] * 6 + [True] * 5 + [False] * 6), 10)
        self.assertEqual(merged, [(0, 6), (11, 17)])
        self.assertEqual(blocked, 1)

    def test_merge_function_observed(self):
        merged, blocked = merge_intervals_no_unknown(
            [(0, 6), (11, 17)], np.zeros(17, dtype=bool), 10)
        self.assertEqual(merged, [(0, 17)])
        self.assertEqual(blocked, 0)


class TestEventSupport(unittest.TestCase):
    def test_public_support_counter_includes_pre_grid_readings(self):
        x = np.full(30 * FS, np.nan)
        x[int(0.5 * FS)] = 5.0
        self.assertEqual(count_actual_readings(x, 1, 11, 10.0, cap_sec=10)["low_readings"], 1)
        x[int(10.5 * FS)] = 5.0
        self.assertEqual(count_actual_readings(x, 1, 21, 10.0, cap_sec=10)["low_readings"], 2)

    def test_half_second_single_reading_before_event_start(self):
        # The only low reading is at 0.5 s. It ffill-feeds grid cells 1..10,
        # so the event starts at 1 s while the trigger reading is before it.
        x = np.full(30 * FS, np.nan)
        x[int(0.5 * FS)] = 5.0
        res = build_u2_events(x, cap_sec=10, min_sec=10, merge_gap_sec=10)
        self.assertEqual(res["events"], [(1, 11)], res)
        self.assertEqual(res["support"][0]["low_readings"], 1, res["support"])
        reliable, low = split_events_by_support(res)
        self.assertEqual(reliable, [])
        self.assertEqual(low, [(1, 11)])

    def test_half_and_ten_point_five_second_two_readings(self):
        # Sources at 0.5 s and 10.5 s both feed the event grid. The first is
        # before the integer-second event start (1 s); both are real readings.
        x = np.full(30 * FS, np.nan)
        x[int(0.5 * FS)] = 5.0
        x[int(10.5 * FS)] = 5.0
        res = build_u2_events(x, cap_sec=10, min_sec=10, merge_gap_sec=10)
        self.assertEqual(res["events"], [(1, 21)], res)
        self.assertEqual(res["support"][0]["low_readings"], 2, res["support"])
        reliable, low = split_events_by_support(res)
        self.assertEqual(reliable, [(1, 21)])
        self.assertEqual(low, [])

    def test_real_case_source_association_regression(self):
        # Extracted unchanged from VitalDB case 1, Primus/ETCO2 [355, 380) s.
        # Event local [5, 23) has three actual low readings. The first one is
        # at local 4.03 s, before the event's integer-second start; a
        # slice-only counter used to return 2.
        import json
        fixture = Path(__file__).resolve().parent / "fixtures" / "case1_etco2_355_380.json"
        with open(fixture, "r", encoding="utf-8") as f:
            spec = json.load(f)
        x = np.full(int(spec["n_samples"]), np.nan, dtype=float)
        for idx, value in spec["nonzero_samples"]:
            x[int(idx)] = float(value)
        res = build_u2_events(x, cap_sec=10, min_sec=10, merge_gap_sec=10)
        self.assertEqual(res["events"], [(5, 23)], res)
        self.assertEqual(res["support"][0]["low_readings"], 3, res["support"])
        reliable, low = split_events_by_support(res)
        self.assertEqual(reliable, [(5, 23)])
        self.assertEqual(low, [])

    def test_iter_windows_default_splits_low_support(self):
        # A continuous low reference is one actual reading run -> low-support.
        # Calling iter_windows() without explicit event lists must not promote
        # it to a reliable U2 positive.
        n = 100 * FS
        case = {
            "co2": np.full(n, 30.0),
            "rr": np.full(n, 12.0),
            "etco2": np.full(n, 5.0),
        }
        recs = list(iter_windows(999, 90.0, 90.0, case_data=case))
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["label"], "ignore_low_support", recs[0])
        self.assertEqual(recs[0]["overlap"], 0.0)
        self.assertGreaterEqual(recs[0]["overlap_low"], 0.50)

    def test_single_reading_is_low_support(self):
        # One low reading at t=0, followed by missing samples; ffill holds it
        # for 10 s. This is the low-support case, not a reliable event.
        x = np.full(30 * FS, np.nan)
        x[0] = 5.0
        res = build_u2_events(x, cap_sec=10, min_sec=10, merge_gap_sec=10)
        self.assertEqual(len(res["events"]), 1, res)
        self.assertEqual(res["support"][0]["low_readings"], 1)
        reliable, low = split_events_by_support(res)
        self.assertEqual(reliable, [])
        self.assertEqual(len(low), 1)

    def test_multiple_readings_are_reliable(self):
        # One low reading every 2 s for 20 s -> >=2 actual readings.
        segs = [(t, t + 0.01, 5.0) for t in range(0, 20, 2)]
        x = make_track(30, segs)
        res = build_u2_events(x, cap_sec=10, min_sec=10, merge_gap_sec=10)
        self.assertEqual(len(res["events"]), 1, res)
        self.assertGreaterEqual(res["support"][0]["low_readings"], 2)
        reliable, low = split_events_by_support(res)
        self.assertEqual(len(reliable), 1)
        self.assertEqual(low, [])

    def test_low_support_window_is_ignore_not_negative(self):
        rr = np.full(9000, np.nan)
        et = np.full(9000, np.nan)
        rr[::200] = 12.0
        et[::200] = 40.0
        # Overlap only with a low-support event.
        lab = classify_window(rr, et, overlap=0.0, overlap_low=1.0)
        self.assertEqual(lab["label"], "ignore_low_support")
        # Reliable event overlap takes precedence.
        lab2 = classify_window(rr, et, overlap=1.0, overlap_low=0.0)
        self.assertEqual(lab2["label"], "U2")
        # Low-support boundary 10-50% is also ignore, not a negative.
        lab3 = classify_window(rr, et, overlap=0.0, overlap_low=0.2)
        self.assertEqual(lab3["label"], "ignore_low_support")


if __name__ == "__main__":
    unittest.main()
