# -*- coding: utf-8 -*-
"""Boundary tests for the frozen one-to-one event matching."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from event_matching import match_active, match_suspect, temporal_iou


def ev(start, end, cls="U2", case=1, stream=None):
    d = {"case_id": case, "class": cls, "start": start, "end": end}
    if stream is not None:
        d["stream"] = stream
    return d


class TestEventMatching(unittest.TestCase):
    def test_iou_is_intersection_over_union(self):
        t = ev(100, 110)
        a = ev(95, 105)
        # intersection 5, union 10+10-5=15
        self.assertAlmostEqual(temporal_iou(t, a), 5.0 / 15.0)
        self.assertAlmostEqual(temporal_iou(ev(100, 110), ev(105, 115)), 5.0 / 15.0)
        self.assertEqual(temporal_iou(ev(100, 110), ev(95, 96)), 0.0)

    def test_active_requires_overlap(self):
        r = match_active([ev(100, 110)], [ev(95, 96)])
        self.assertEqual(r["tp"], 0)
        self.assertEqual(r["fp"], 1)
        self.assertEqual(r["fn"], 1)
        self.assertEqual(len(r["early_alarms"]), 1)

    def test_tiny_overlap_is_not_tp(self):
        # 0.1 s touch must not count as TP.
        r = match_active([ev(100, 110)], [ev(100, 100.1)])
        self.assertEqual(r["tp"], 0)
        self.assertEqual(r["fp"], 1)
        # 1.0 s overlap is valid (>=1 s and >=5% of the 1 s alarm).
        r2 = match_active([ev(100, 110)], [ev(100, 101)])
        self.assertEqual(r2["tp"], 1)
        self.assertEqual(r2["ttd_s"], [0.0])

    def test_active_late_grace_is_not_tp(self):
        r = match_active([ev(100, 110)], [ev(112, 113)])
        self.assertEqual(r["tp"], 0)
        self.assertEqual(r["fp"], 1)
        self.assertEqual(len(r["late_alarms"]), 1)

    def test_active_overlap_tp_and_ttd(self):
        r = match_active([ev(100, 110)], [ev(95, 105)])
        self.assertEqual(r["tp"], 1)
        self.assertEqual(r["fp"], 0)
        self.assertEqual(r["ttd_s"], [-5.0])
        self.assertAlmostEqual(r["iou"][0], 1.0 / 3.0)
        r2 = match_active([ev(100, 110)], [ev(105, 115)])
        self.assertEqual(r2["tp"], 1)
        self.assertEqual(r2["ttd_s"], [5.0])

    def test_duplicate_alarms_one_tp_one_fp(self):
        r = match_active([ev(100, 110)], [ev(105, 106), ev(107, 108)])
        self.assertEqual(r["tp"], 1)
        self.assertEqual(r["fp"], 1)
        self.assertEqual(r["fn"], 0)

    def test_suspect_is_separate_from_active(self):
        truth = [ev(100, 110)]
        suspects = [ev(95, 96, stream="SUSPECT")]
        # ACTIVE matcher filters SUSPECT out.
        ra = match_active(truth, suspects)
        self.assertEqual(ra["tp"], 0)
        self.assertEqual(ra["fp"], 0)
        # Separate pre-onset warning evaluation.
        rs = match_suspect(truth, suspects)
        self.assertEqual(rs["suspect_recall"], 1.0)
        self.assertEqual(rs["suspect_recall_pre"], 1.0)
        self.assertEqual(rs["suspect_recall_post"], 0.0)
        self.assertEqual(rs["leads_s"], [5.0])
        self.assertEqual(rs["median_lead_s"], 5.0)

    def test_suspect_at_onset_is_post_onset_zero_ttd(self):
        truth = [ev(100, 110)]
        suspects = [ev(100, 101, stream="SUSPECT")]
        rs = match_suspect(truth, suspects)
        self.assertEqual(rs["pre_onset_matches"], [])
        self.assertEqual(len(rs["post_onset_matches"]), 1)
        self.assertEqual(rs["ttds_s"], [0.0])
        self.assertEqual(rs["median_ttd_s"], 0.0)

    def test_suspect_after_onset_five_seconds(self):
        truth = [ev(100, 110)]
        suspects = [ev(105, 106, stream="SUSPECT")]
        rs = match_suspect(truth, suspects)
        self.assertEqual(rs["pre_onset_matches"], [])
        self.assertEqual(len(rs["post_onset_matches"]), 1)
        self.assertEqual(rs["ttds_s"], [5.0])
        self.assertEqual(rs["median_ttd_s"], 5.0)
        self.assertEqual(rs["suspect_recall_post"], 1.0)

    def test_suspect_after_offset_is_not_early_detection(self):
        truth = [ev(100, 110)]
        suspects = [ev(112, 113, stream="SUSPECT")]
        rs = match_suspect(truth, suspects)
        self.assertEqual(rs["matched_truth"], [])
        self.assertEqual(len(rs["unmatched_suspects"]), 1)

    def test_suspect_one_to_one(self):
        truth = [ev(100, 110)]
        suspects = [ev(95, 96, stream="SUSPECT"), ev(96, 97, stream="SUSPECT")]
        rs = match_suspect(truth, suspects)
        self.assertEqual(rs["suspect_recall"], 1.0)
        self.assertEqual(len(rs["unmatched_suspects"]), 1)

    def test_class_and_case_must_match(self):
        truth = [ev(100, 110, cls="U2", case=1)]
        alarms = [ev(105, 115, cls="U1", case=1), ev(105, 115, cls="U2", case=2)]
        r = match_active(truth, alarms)
        self.assertEqual(r["tp"], 0)
        self.assertEqual(r["fp"], 2)


if __name__ == "__main__":
    unittest.main()
