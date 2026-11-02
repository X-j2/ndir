# -*- coding: utf-8 -*-
"""Boundary tests for the Stage 3 causal event engine."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.stage3_event_engine import EventEngine, EventParams  # noqa: E402


def make(**kw):
    p = dict(suspect_p=0.20, active_enter_p=0.60, active_exit_p=0.20,
             suspect_on_count=1, suspect_off_count=1,
             active_on_count=2, active_off_count=2)
    p.update(kw)
    return EventParams(**p), EventEngine(EventParams(**p), 7.5, "U15")


class TestStage3EventEngine(unittest.TestCase):
    def test_first_prompt_and_first_active_are_separate(self):
        p, e = make()
        e.update(15.0, "VALID", 0.30)   # suspect
        e.update(22.5, "VALID", 0.90)   # active #1
        e.update(30.0, "VALID", 0.90)   # active #2 -> ACTIVE
        self.assertEqual(e.episodes[0]["first_prompt_time"], 15.0)
        self.assertEqual(e.episodes[0]["first_active_time"], 30.0)
        self.assertEqual(e.episodes[0]["estimated_onset"], 15.0)
        e.finish(37.5)
        self.assertEqual(e.active_intervals()[0]["start"], 30.0)
        self.assertEqual(e.active_intervals()[0]["decision_time"], 30.0)

    def test_flat_ever_never_creates_active(self):
        p, e = make()
        for t in (15.0, 22.5, 30.0, 37.5):
            e.update(t, "FLAT_EVER_HISTORY", None)
        self.assertEqual(e.state, "SUSPECT")
        self.assertEqual(e.active_intervals(), [])
        self.assertTrue(e.episodes[0]["low_confidence"])

    def test_flat_with_can_create_active(self):
        p, e = make(active_on_count=2)
        e.update(15.0, "FLAT_WITH_HISTORY", None)
        e.update(22.5, "FLAT_WITH_HISTORY", None)
        self.assertEqual(e.state, "ACTIVE")
        self.assertFalse(e.episodes[0]["low_confidence"])

    def test_unknown_closes_and_resets_without_carrying(self):
        p, e = make(active_on_count=2)
        e.update(15.0, "VALID", 0.90)
        e.update(22.5, "VALID", 0.90)
        self.assertEqual(e.state, "ACTIVE")
        trans = e.update(30.0, "GAP", None)
        self.assertEqual(e.state, "UNKNOWN")
        self.assertTrue(any(x["kind"] == "UNKNOWN" for x in trans))
        self.assertEqual(e.active_intervals()[0]["end"], 30.0)
        # One new high decision must not re-enter ACTIVE immediately.
        e.update(37.5, "VALID", 0.90)
        self.assertEqual(e.state, "SUSPECT")
        self.assertEqual(len(e.active_intervals()), 1)

    def test_active_hysteresis_exit(self):
        p, e = make(active_on_count=2, active_off_count=2)
        e.update(15.0, "VALID", 0.90)
        e.update(22.5, "VALID", 0.90)   # ACTIVE
        e.update(30.0, "VALID", 0.10)   # exit count 1
        self.assertEqual(e.state, "ACTIVE")
        e.update(37.5, "VALID", 0.10)   # exit count 2 -> SUSPECT/NORMAL
        self.assertFalse(e._active_open)
        self.assertEqual(e.active_intervals()[0]["end"], 37.5)

    def test_neutral_probability_does_not_exit_active(self):
        p, e = make(active_on_count=2, active_off_count=1)
        e.update(15.0, "VALID", 0.90)
        e.update(22.5, "VALID", 0.90)
        e.update(30.0, "VALID", 0.40)  # between exit and enter -> neutral
        self.assertEqual(e.state, "ACTIVE")
        e.update(37.5, "VALID", 0.10)  # exit
        self.assertFalse(e._active_open)

    def test_unknown_does_not_create_alarm(self):
        p, e = make()
        trans = e.update(15.0, "GAP", None)
        self.assertEqual(e.state, "UNKNOWN")
        self.assertFalse(any(x["kind"] == "ACTIVE_START" for x in trans))
        self.assertEqual(e.active_intervals(), [])

    def test_finish_closes_open_interval_at_theoretical_end(self):
        p, e = make(active_on_count=1)
        e.update(15.0, "VALID", 0.90)
        e.finish(22.5)
        self.assertFalse(e._active_open)
        self.assertEqual(e.active_intervals()[0]["end"], 22.5)

    def test_neutral_breaks_active_entry_consecutive_count(self):
        p, e = make(active_on_count=2)
        e.update(15.0, "VALID", 0.90)   # active run 1
        e.update(22.5, "VALID", 0.20)   # neutral -> active run 0
        self.assertEqual(e._active_run, 0)
        e.update(30.0, "VALID", 0.90)   # active run 1, must NOT open ACTIVE
        self.assertEqual(e.state, "SUSPECT")
        self.assertFalse(e._active_open)
        e.update(37.5, "VALID", 0.90)   # active run 2 -> ACTIVE
        self.assertEqual(e.state, "ACTIVE")
        self.assertEqual(e.episodes[0]["first_active_time"], 37.5)

    def test_neutral_breaks_active_exit_consecutive_count(self):
        p, e = make(active_on_count=1, active_off_count=2)
        e.update(15.0, "VALID", 0.90)   # ACTIVE
        e.update(22.5, "VALID", 0.10)   # active-exit run 1
        e.update(30.0, "VALID", 0.40)   # neutral -> exit run 0, still ACTIVE
        self.assertEqual(e.state, "ACTIVE")
        e.update(37.5, "VALID", 0.10)   # exit run 1, still ACTIVE
        self.assertEqual(e.state, "ACTIVE")
        e.update(45.0, "VALID", 0.10)   # exit run 2 -> close
        self.assertFalse(e._active_open)
        self.assertEqual(e.active_intervals()[0]["end"], 45.0)

    def test_missing_decision_interrupts_at_deadline_and_clears_evidence(self):
        p, e = make(active_on_count=2)
        e.update(15.0, "VALID", 0.90)   # old episode begins (SUSPECT)
        # Expected next decision is 22.5; 150.0 is a data interruption.
        trans = e.update(150.0, "VALID", 0.90)
        self.assertEqual(e.gap_interruptions, 1)
        self.assertEqual(e.interruptions, 1)
        unknown = [x for x in trans if x["kind"] == "UNKNOWN"]
        self.assertTrue(unknown)
        self.assertEqual(unknown[0]["time"], 22.5)
        self.assertEqual(unknown[0]["reason"], "MISSING_DECISION")
        # Old episode ended at the expected deadline, not at 150.
        self.assertEqual(e.episodes[0]["end_time"], 22.5)
        # The late decision starts a new episode with zero carried counters.
        self.assertEqual(e.episodes[1]["estimated_onset"], 150.0)
        self.assertEqual(e._active_run, 1)
        self.assertEqual(e.state, "SUSPECT")
        e.update(157.5, "VALID", 0.90)
        self.assertEqual(e.state, "ACTIVE")
        self.assertEqual(e.episodes[1]["first_active_time"], 157.5)

    def test_exact_step_interval_is_not_an_interruption(self):
        p, e = make()
        e.update(15.0, "VALID", 0.90)
        e.update(22.5, "VALID", 0.90)
        self.assertEqual(e.gap_interruptions, 0)
        self.assertEqual(e.interruptions, 0)

    def test_transition_snapshots_carry_key_times_and_ids(self):
        p, e = make(active_on_count=1, active_off_count=1,
                    suspect_off_count=1)
        e.update(15.0, "VALID", 0.90)
        e.update(22.5, "VALID", 0.10)
        starts = [x for x in e.transitions if x["kind"] == "ACTIVE_START"]
        ends = [x for x in e.transitions if x["kind"] == "ACTIVE_END"]
        suspect_end = [x for x in e.transitions
                       if x["kind"] == "SUSPECT_END"]
        self.assertEqual(len(starts), 1)
        self.assertEqual(len(ends), 1)
        self.assertIsNotNone(starts[0]["episode_id"])
        self.assertEqual(starts[0]["transition_seq"] + 1, ends[0]["transition_seq"])
        self.assertEqual(starts[0]["episode"]["first_prompt_time"], 15.0)
        self.assertEqual(starts[0]["episode"]["first_active_time"], 15.0)
        self.assertEqual(ends[0]["episode"]["active_intervals"][0]["end"],
                         22.5)
        self.assertEqual(suspect_end[0]["episode"]["end_time"], 22.5)
        self.assertEqual(suspect_end[0]["episode"]["end_reason"], "NORMAL")

    def test_public_interrupt_uses_expected_deadline(self):
        p, e = make(active_on_count=1)
        e.update(15.0, "VALID", 0.90)
        trans = e.interrupt(reason="DATA_INTERRUPTION")
        self.assertEqual(e.interruptions, 1)
        self.assertEqual(e.active_intervals()[0]["end"], 22.5)
        self.assertEqual(trans[-1]["time"], 22.5)
        self.assertEqual(trans[-1]["reason"], "DATA_INTERRUPTION")


if __name__ == "__main__":
    unittest.main()
