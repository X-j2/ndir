# -*- coding: utf-8 -*-
"""Stage 4 experimental host runtime: isolation, timing, long run, consistency."""
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from co2_monitor import CO2MonitorWindow  # noqa: E402
from db_manager import DatabaseManager  # noqa: E402
from host_core.paths import ensure_training_path  # noqa: E402
from host_core.replay import ReplayPack, ReplaySession  # noqa: E402
from host_core.runtime import _resample_25_to_100  # noqa: E402
from host_core.stage4_runtime import Stage4Runtime  # noqa: E402

ensure_training_path()
from features.si_features import extract_si_features  # noqa: E402
from features.quality_states import classify_signal_state  # noqa: E402
from tools.quality_state import HistorySummary  # noqa: E402
from stage0_common import load_case  # noqa: E402
from stage4_host_reference import build_reference, compare_host  # noqa: E402
from tools import verify_stage4_host_consistency as verify_script  # noqa: E402

ROOT = Path(__file__).resolve().parent
_FIXTURE = None


def four_scale_fixture():
    """Build the 150 s development host/reference pair once per test run."""
    global _FIXTURE
    if _FIXTURE is None:
        d = load_case(1)
        start = 330 * 100
        sig100 = np.asarray(d["co2"][start:start + 150 * 100], dtype=float)
        sig25 = sig100[::4]
        ref = build_reference(sig25, 25, 150.0)
        runtime = Stage4Runtime()
        if not runtime.available:
            raise RuntimeError(runtime.error)
        for i in range(0, len(sig25), 25):
            runtime.feed_samples(sig25[i:i + 25])
        runtime.finish(150.0)
        _FIXTURE = (runtime, ref)
    return _FIXTURE


class TestStage4Host(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.tmp = tempfile.mkdtemp(prefix="stage4_host_")
        cls.win = CO2MonitorWindow()
        cls.win.db.close()
        cls.win.db = DatabaseManager(os.path.join(cls.tmp, "test.db"))

    @classmethod
    def tearDownClass(cls):
        try:
            cls.win.db.close()
            cls.win.close()
        except Exception:
            pass

    def setUp(self):
        self.win.set_display_mode("relative")
        self.win._reset_stage3_session("test")

    def test_stage4_runtime_loads_and_has_fusion(self):
        r = Stage4Runtime()
        self.assertTrue(r.available, r.error)
        states = r.states()
        for s in ("U15", "U30", "U60", "U90", "FUSION"):
            self.assertIn(s, states)
        pack = ReplayPack.load(ROOT / "replay_data" / "dev_demo.json")
        r.feed_samples(pack.samples_mmhg[:750])
        self.assertEqual(r.states()["FUSION"]["decision_time"], 30.0)

    def test_stage4_replay_is_isolated_from_production(self):
        before_dev = len(self.win.db.query_device_data(limit=100000))
        before_alarm = len(self.win.db.query_alarms())
        before_event = len(self.win.db.query_event_records(limit=100000))
        notify = []
        old_show = self.win._tray.showMessage
        self.win._tray.showMessage = lambda *a, **k: notify.append(a)
        try:
            self.win._rp_engine.setCurrentText("Stage4实验（四尺度）")
            self.win._load_replay_pack_path(str(
                ROOT / "replay_data" / "dev_demo.json"))
            self.assertIsNotNone(self.win._replay_session)
            self.assertIsInstance(self.win._replay_runtime, Stage4Runtime)
            self.win._replay_play_pause()
            for _ in range(12):
                self.win._replay_tick()
            self.win._replay_restart()
            self.win._replay_slider_moved(100)
        finally:
            self.win._tray.showMessage = old_show
        self.assertEqual(len(self.win.db.query_device_data(limit=100000)),
                         before_dev)
        self.assertEqual(len(self.win.db.query_alarms()), before_alarm)
        self.assertEqual(len(self.win.db.query_event_records(limit=100000)),
                         before_event)
        self.assertEqual(notify, [])

    def test_stage4_real_runtime_speed_pause_seek(self):
        pack = ReplayPack.load(ROOT / "replay_data" / "dev_demo.json")
        a = ReplaySession(pack, Stage4Runtime())
        b = ReplaySession(pack, Stage4Runtime())
        self.assertTrue(a.runtime.available, a.runtime.error)
        a.play()
        b.play()
        for _ in range(40):
            a.advance(1.0, 1.0)       # 40 s signal
        for _ in range(40):
            b.advance(0.5, 2.0)       # 40 s signal in 20 s wall
        self.assertEqual(a.position, 1000)
        self.assertEqual(a.position, b.position)
        self.assertEqual(
            json.dumps(a.runtime.states(), sort_keys=True, default=str),
            json.dumps(b.runtime.states(), sort_keys=True, default=str))
        self.assertEqual(a.events, b.events)
        a.pause()
        pos = a.position
        a.advance(1.0, 1.0)
        self.assertEqual(a.position, pos)
        c = ReplaySession(pack, Stage4Runtime())
        c.seek(40.0)
        self.assertEqual(c.position, pos)
        self.assertEqual(
            json.dumps(c.runtime.states(), sort_keys=True, default=str),
            json.dumps(a.runtime.states(), sort_keys=True, default=str))

    def test_stage4_interrupt_and_reset_clear_evidence(self):
        r = Stage4Runtime()
        pack = ReplayPack.load(ROOT / "replay_data" / "dev_demo.json")
        r.feed_samples(pack.samples_mmhg[:750])
        trans = r.interrupt(reason="DATA_INTERRUPTION")
        self.assertIsInstance(trans, list)
        r.reset()
        st = r.states()
        self.assertEqual(st["FUSION"]["state"], "WAITING")
        self.assertEqual(st["U15"]["state"], "WAITING")

    def test_stage4_long_run_is_finite(self):
        # 600 s synthesized replay, no production DB involved.
        t = np.arange(15000, dtype=float) / 25.0
        samples = 20.0 + 10.0 * np.sin(2 * np.pi * 0.2 * t)
        pack = ReplayPack(
            version="stage3_replay_pack_v1", source="synthetic",
            stitched=False, sample_rate_hz=25.0,
            samples_mmhg=[float(x) for x in samples])
        sess = ReplaySession(pack, Stage4Runtime())
        self.assertTrue(sess.runtime.available, sess.runtime.error)
        sess.play()
        for _ in range(600):
            sess.advance(1.0, 1.0)
        self.assertEqual(sess.position, len(samples))
        st = sess.runtime.states()
        for s in ("U15", "U30", "U60", "U90", "FUSION"):
            self.assertIn(st[s]["state"],
                          ("WAITING", "UNKNOWN", "NORMAL", "SUSPECT",
                           "ACTIVE"))

    def test_stage4_fusion_host_matches_offline_frames(self):
        pack = ReplayPack.load(ROOT / "replay_data" / "dev_demo.json")
        r = Stage4Runtime()
        r.feed_samples(pack.samples_mmhg[:1000])
        r.finish(40.0)
        series = r._series()
        eng = r.modules["EventEngine"](
            r.fusion_params.event, 7.5, "FUSION")
        t = 15.0
        while t <= 40.0 + 1e-9:
            q, p = r.modules["fuse_at_time"](series, t, r.fusion_params)
            eng.update(t, q, p)
            t += 7.5
        eng.finish(40.0)
        self.assertEqual(eng.state, r._fusion_engine.state)
        self.assertEqual([(x["kind"], x["time"]) for x in eng.transitions],
                         [(x["kind"], x["time"])
                          for x in r._fusion_engine.transitions])

    def test_stage4_four_scale_independent_offline_reference(self):
        r, ref = four_scale_fixture()
        comp = compare_host(r, ref)
        for scale in ("U15", "U30", "U60", "U90"):
            rec = comp["scales"][scale]
            self.assertGreater(rec["n_host"], 0, scale)
            self.assertEqual(rec["n_host"], rec["n_reference"], scale)
            self.assertTrue(rec["frame_count_equal"], scale)
            self.assertEqual(rec["frame_count_diff"], 0, scale)
            self.assertEqual(rec["decision_time_mismatch"], 0, scale)
            self.assertEqual(rec["window_mismatch"], 0, scale)
            self.assertEqual(rec["history_mismatch"], 0, scale)
            self.assertEqual(rec["state_mismatch"], 0, scale)
            self.assertEqual(rec["feature_nonfinite_host"], 0, scale)
            self.assertEqual(rec["feature_nonfinite_reference"], 0, scale)
            self.assertEqual(rec["feature_tolerance_exceeded"], 0, scale)
            self.assertEqual(rec["p_none_mismatch"], 0, scale)
            self.assertEqual(rec["p_nonfinite_host"], 0, scale)
            self.assertEqual(rec["p_nonfinite_reference"], 0, scale)
            self.assertEqual(rec["p_tolerance_exceeded"], 0, scale)
            self.assertLessEqual(rec["max_feature_abs_delta"], 1e-9, scale)
            self.assertIsNotNone(rec["max_p_abs_delta"], scale)
            self.assertLessEqual(rec["max_p_abs_delta"], 1e-9, scale)
            self.assertTrue(rec["transition_equal"], scale)
            self.assertEqual(rec["transition_mismatch"], 0, scale)
            self.assertEqual(len(rec["transition_host"]),
                             len(rec["transition_reference"]), scale)
            for host_t, ref_t in zip(rec["transition_host"],
                                     rec["transition_reference"]):
                self.assertEqual(host_t["kind"], ref_t["kind"], scale)
                self.assertAlmostEqual(host_t["time"], ref_t["time"],
                                       places=9, msg=scale)
                self.assertEqual(host_t["episode_id"],
                                 ref_t["episode_id"], scale)
        fusion = comp["fusion"]
        self.assertEqual(fusion["n_host"], fusion["n_reference"])
        self.assertTrue(fusion["frame_count_equal"])
        self.assertEqual(fusion["frame_count_diff"], 0)
        self.assertEqual(fusion["frame_mismatch"], 0)
        self.assertEqual(fusion["time_mismatch"], 0)
        self.assertEqual(fusion["quality_mismatch"], 0)
        self.assertEqual(fusion["p_none_mismatch"], 0)
        self.assertEqual(fusion["p_nonfinite_host"], 0)
        self.assertEqual(fusion["p_nonfinite_reference"], 0)
        self.assertEqual(fusion["p_tolerance_exceeded"], 0)
        self.assertIsNotNone(fusion["max_p_abs_delta"])
        self.assertLessEqual(fusion["max_p_abs_delta"], 1e-9)
        self.assertTrue(fusion["transition_equal"])
        self.assertEqual(fusion["transition_mismatch"], 0)
        self.assertEqual(len(fusion["transition_host"]),
                         len(fusion["transition_reference"]))
        for host_t, ref_t in zip(fusion["transition_host"],
                                 fusion["transition_reference"]):
            self.assertEqual(host_t["kind"], ref_t["kind"])
            self.assertAlmostEqual(host_t["time"], ref_t["time"], places=9)
            self.assertEqual(host_t["episode_id"], ref_t["episode_id"])
        self.assertTrue(comp["ok"], comp["failure_reasons"])

    def test_compare_host_detects_deleted_fusion_frame(self):
        r, ref = four_scale_fixture()
        mutated = copy.deepcopy(ref)
        mutated["fusion_frames"].pop()
        comp = compare_host(r, mutated)
        self.assertFalse(comp["ok"])
        self.assertFalse(comp["fusion"]["frame_count_equal"])
        self.assertEqual(comp["fusion"]["frame_count_diff"], 1)
        self.assertIn("FUSION.frame_count", comp["failure_reasons"])

    def test_compare_host_detects_fusion_time_shift(self):
        r, ref = four_scale_fixture()
        mutated = copy.deepcopy(ref)
        mutated["fusion_frames"][0][0] = float(
            mutated["fusion_frames"][0][0]) + 0.5
        comp = compare_host(r, mutated)
        self.assertFalse(comp["ok"])
        self.assertEqual(comp["fusion"]["time_mismatch"], 1)
        self.assertGreater(comp["fusion"]["frame_mismatch"], 0)
        self.assertIn("FUSION.time_mismatch", comp["failure_reasons"])
        self.assertEqual(verify_script.consistency_exit_code(comp), 1)

    def test_compare_host_detects_fusion_quality_change(self):
        r, ref = four_scale_fixture()
        mutated = copy.deepcopy(ref)
        mutated["fusion_frames"][0][1] = "BAD_QUALITY"
        comp = compare_host(r, mutated)
        self.assertFalse(comp["ok"])
        self.assertEqual(comp["fusion"]["quality_mismatch"], 1)
        self.assertGreater(comp["fusion"]["frame_mismatch"], 0)
        self.assertIn("FUSION.quality_mismatch", comp["failure_reasons"])
        self.assertEqual(verify_script.consistency_exit_code(comp), 1)

    def test_compare_host_detects_probability_offset(self):
        r, ref = four_scale_fixture()
        mutated = copy.deepcopy(ref)
        scale_row = next(x for x in mutated["scales"]["U15"]["decisions"]
                         if x["p_u2"] is not None)
        scale_row["p_u2"] += 0.2
        fusion_row = next(x for x in mutated["fusion_frames"]
                          if x[2] is not None)
        fusion_row[2] += 0.2
        comp = compare_host(r, mutated)
        self.assertFalse(comp["ok"])
        self.assertGreater(comp["scales"]["U15"]["p_tolerance_exceeded"], 0)
        self.assertGreater(comp["fusion"]["p_tolerance_exceeded"], 0)
        self.assertIn("U15.p_tolerance_exceeded", comp["failure_reasons"])
        self.assertIn("FUSION.p_tolerance_exceeded", comp["failure_reasons"])

    def test_compare_host_detects_one_sided_none_probability(self):
        r, ref = four_scale_fixture()
        mutated = copy.deepcopy(ref)
        scale_row = next(x for x in mutated["scales"]["U15"]["decisions"]
                         if x["p_u2"] is not None)
        scale_row["p_u2"] = None
        fusion_row = next(x for x in mutated["fusion_frames"]
                          if x[2] is not None)
        fusion_row[2] = None
        comp = compare_host(r, mutated)
        self.assertFalse(comp["ok"])
        self.assertEqual(comp["scales"]["U15"]["p_none_mismatch"], 1)
        self.assertEqual(comp["fusion"]["p_none_mismatch"], 1)
        self.assertIn("U15.p_none_mismatch", comp["failure_reasons"])
        self.assertIn("FUSION.p_none_mismatch", comp["failure_reasons"])

    def test_compare_host_detects_nonfinite_values(self):
        r, ref = four_scale_fixture()
        mutated = copy.deepcopy(ref)
        fusion_row = next(x for x in mutated["fusion_frames"]
                          if x[2] is not None)
        fusion_row[2] = float("nan")
        mutated["scales"]["U15"]["decisions"][0]["features"][0] = float("inf")
        comp = compare_host(r, mutated)
        self.assertFalse(comp["ok"])
        self.assertEqual(comp["scales"]["U15"]["feature_nonfinite_reference"], 1)
        self.assertEqual(comp["fusion"]["p_nonfinite_reference"], 1)
        self.assertIn("U15.feature_nonfinite_reference",
                      comp["failure_reasons"])
        self.assertIn("FUSION.p_nonfinite_reference",
                      comp["failure_reasons"])

    def test_compare_host_detects_event_transition_field_changes(self):
        r, ref = four_scale_fixture()
        targets = ("U15", "U30", "U60", "U90", "FUSION")
        for target in targets:
            for field in ("kind", "time", "episode_id"):
                with self.subTest(target=target, field=field):
                    mutated = copy.deepcopy(ref)
                    rows = (mutated["fusion_transitions"] if target == "FUSION"
                            else mutated["scales"][target]["transitions"])
                    row = rows[0]
                    if field == "kind":
                        row["kind"] = f"{row['kind']}_MUTATED"
                    elif field == "time":
                        row["time"] = float(row["time"]) + 0.5
                    else:
                        row["episode_id"] = (
                            999 if row.get("episode_id") is None
                            else int(row["episode_id"]) + 1000)
                    comp = compare_host(r, mutated)
                    self.assertFalse(comp["ok"], (target, field))
                    rec = (comp["fusion"] if target == "FUSION"
                           else comp["scales"][target])
                    self.assertGreater(rec["transition_mismatch"], 0,
                                       (target, field))
                    self.assertIn(f"{target}.transition_mismatch",
                                  comp["failure_reasons"])

    def test_verify_script_exit_code_tracks_compare_failures(self):
        self.assertEqual(verify_script.consistency_exit_code(
            {"ok": True, "failure_reasons": []}), 0)
        self.assertEqual(verify_script.consistency_exit_code(
            {"ok": False, "failure_reasons": ["FUSION.frame_count"]}), 1)

    def test_stage4_host_matches_direct_offline_u15_window(self):
        pack = ReplayPack.load(ROOT / "replay_data" / "dev_demo.json")
        r = Stage4Runtime()
        r.feed_samples(pack.samples_mmhg[:375])
        w100 = _resample_25_to_100(np.asarray(pack.samples_mmhg[:375],
                                              dtype=float))
        feat = extract_si_features(w100)
        hist = HistorySummary(False, None, None, 0.0, None, None, 0)
        state = classify_signal_state(w100, hist, feature_diag=feat)
        self.assertEqual(r.scales["U15"].last_state, state)
        np.testing.assert_allclose(
            r.scales["U15"].last_features, feat["vector"], atol=1e-6)
        if state == "VALID":
            p_raw = r.modules["predict_full"](
                r.models["U15"], np.asarray([feat["vector"]],
                                             dtype=np.float32))
            p = r.modules["apply_calibrator"](
                r.calibrators["U15"], p_raw)[0, 2]
            self.assertAlmostEqual(r.scales["U15"].last_p, float(p),
                                   places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
