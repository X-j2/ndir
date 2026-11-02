# -*- coding: utf-8 -*-
"""Stage 4 fusion boundary tests: causality, one vote, validity, low confidence."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.stage4_fusion import (  # noqa: E402
    REFERENCE_EVENT, FusionParams, fuse_at_time, fuse_case,
)


def ser(scale, times, states, p, window, step):
    return {
        "scale": scale, "times": np.asarray(times, dtype=float),
        "states": tuple(states), "p": np.asarray(p, dtype=float),
        "window_sec": float(window), "step_sec": float(step),
    }


class TestStage4Fusion(unittest.TestCase):
    def test_future_decision_is_not_used(self):
        s = {
            "U15": ser("U15", [15.0], ["VALID"], [0.9], 15, 7.5),
            "U30": ser("U30", [30.0], ["VALID"], [0.9], 30, 15),
        }
        frames = fuse_case(s, 30.0, FusionParams(
            weight_profile="equal", tau_sec=60, min_active_sources=1,
            event=REFERENCE_EVENT))
        at15 = [f for f in frames if abs(f[0] - 15.0) < 1e-9][0]
        self.assertEqual(at15[1], "VALID")
        self.assertGreater(at15[2], 0.8)
        # U30's 30 s decision cannot be used at 15 s.
        at15b = [f for f in frames if abs(f[0] - 22.5) < 1e-9][0]
        self.assertEqual(at15b[1], "GAP")

    def test_scale_expires_without_next_decision(self):
        s = {
            "U15": ser("U15", [15.0], ["VALID"], [0.9], 15, 7.5),
        }
        frames = fuse_case(s, 30.0, FusionParams(
            weight_profile="equal", tau_sec=60, min_active_sources=1,
            event=REFERENCE_EVENT))
        self.assertEqual(frames[0][1], "VALID")
        self.assertEqual(frames[1][1], "GAP")

    def test_flat_ever_alone_is_suspect_only(self):
        s = {
            "U15": ser("U15", [15.0], ["FLAT_EVER_HISTORY"], [np.nan],
                       15, 7.5),
        }
        frames = fuse_case(s, 22.5, FusionParams(
            weight_profile="equal", tau_sec=60, min_active_sources=1,
            event=REFERENCE_EVENT))
        self.assertEqual(frames[0][1], "FLAT_EVER_HISTORY")
        self.assertIsNone(frames[0][2])

    def test_min_active_sources_caps_score_below_active_enter(self):
        s = {
            "U15": ser("U15", [30.0], ["VALID"], [0.95], 15, 7.5),
        }
        frames = fuse_case(s, 30.0, FusionParams(
            weight_profile="equal", tau_sec=60, min_active_sources=2,
            event=REFERENCE_EVENT))
        f = [x for x in frames if abs(x[0] - 30.0) < 1e-9][0]
        self.assertEqual(f[1], "VALID")
        self.assertLess(f[2], REFERENCE_EVENT.active_enter_p)

    def test_one_vote_per_scale_and_bounded_score(self):
        s = {
            "U15": ser("U15", [15.0], ["VALID"], [0.8], 15, 7.5),
            "U30": ser("U30", [15.0], ["VALID"], [0.8], 30, 15),
        }
        frames = fuse_case(s, 15.0, FusionParams(
            weight_profile="equal", tau_sec=60, min_active_sources=1,
            event=REFERENCE_EVENT))
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0][1], "VALID")
        self.assertAlmostEqual(frames[0][2], 0.8, places=9)

    def test_no_evidence_is_gap_not_negative(self):
        s = {"U15": ser("U15", [], [], [], 15, 7.5)}
        frames = fuse_case(s, 15.0, FusionParams(
            weight_profile="equal", tau_sec=60, min_active_sources=1,
            event=REFERENCE_EVENT))
        self.assertEqual(frames[0][1], "GAP")
        self.assertIsNone(frames[0][2])

    def test_scale_subset_renormalizes(self):
        s = {
            "U15": ser("U15", [30.0], ["VALID"], [0.8], 15, 7.5),
            "U30": ser("U30", [30.0], ["VALID"], [0.2], 30, 15),
        }
        all_frames = fuse_case(s, 30.0, FusionParams(
            weight_profile="equal", tau_sec=60, min_active_sources=1,
            event=REFERENCE_EVENT))
        only15 = fuse_case(s, 30.0, FusionParams(
            weight_profile="equal", tau_sec=60, min_active_sources=1,
            event=REFERENCE_EVENT), scales=("U15",))
        f_all = [x for x in all_frames if abs(x[0] - 30.0) < 1e-9][0]
        f_15 = [x for x in only15 if abs(x[0] - 30.0) < 1e-9][0]
        self.assertAlmostEqual(f_15[2], 0.8, places=9)
        self.assertLess(f_all[2], f_15[2])

    def test_streaming_tick_matches_whole_case(self):
        s = {
            "U15": ser("U15", [15.0, 30.0], ["VALID", "VALID"],
                       [0.7, 0.9], 15, 7.5),
            "U30": ser("U30", [30.0], ["VALID"], [0.3], 30, 15),
        }
        params = FusionParams(weight_profile="balanced", tau_sec=60,
                              min_active_sources=1, event=REFERENCE_EVENT)
        frames = fuse_case(s, 30.0, params)
        for t, q, p in frames:
            q2, p2 = fuse_at_time(s, t, params)
            self.assertEqual(q, q2)
            if p is None:
                self.assertIsNone(p2)
            else:
                self.assertAlmostEqual(p, p2, places=12)


if __name__ == "__main__":
    unittest.main()
