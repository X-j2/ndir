# -*- coding: utf-8 -*-
"""Stage 3 host regressions: relative mode, alarm cutover, isolated replay."""
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from co2_monitor import (  # noqa: E402
    CO2MonitorWindow, DISPLAY_MODE_RELATIVE, decode_frame, encode_frame,
)
from db_manager import DatabaseManager  # noqa: E402
from host_core.replay import ReplayPack, ReplaySession  # noqa: E402
from host_core.runtime import Stage3Runtime  # noqa: E402

ROOT = Path(__file__).resolve().parent


class TestStage3Host(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.tmp = tempfile.mkdtemp(prefix="stage3_host_")
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

    def test_relative_mode_has_no_absolute_units(self):
        from PySide6.QtWidgets import QLabel
        self.assertEqual(self.win.display_mode, DISPLAY_MODE_RELATIVE)
        self.assertIn("相对模式", self.win.mode_label.text())
        texts = [x.text() for x in self.win.etco2_value.findChildren(QLabel)]
        self.assertTrue(any("未标定" in t for t in texts), texts)

    def test_legacy_classes_never_trigger_legacy_alarm_chain(self):
        before_alarms = len(self.win.db.query_alarms())
        notify = []
        old_show = self.win._tray.showMessage
        self.win._tray.showMessage = lambda *a, **k: notify.append(a)
        try:
            for cls in (0, 1, 2, 3):
                for _ in range(3):
                    f = decode_frame(encode_frame(
                        [1500.0] * 5, ai_class=cls, rr_per_min=10.0))
                    self.win.latest_probs = {0: .1, 1: .1, 2: .7, 3: .1}
                    self.win._apply_frame(f)
        finally:
            self.win._tray.showMessage = old_show
        self.assertEqual(len(self.win.db.query_alarms()), before_alarms,
                         "相对模式旧固件分类不得写旧告警")
        self.assertEqual(notify, [], "相对模式旧固件分类不得触发托盘通知")

    def test_c_mode_request_is_rejected(self):
        self.win.set_display_mode("C")
        self.assertEqual(self.win.display_mode, DISPLAY_MODE_RELATIVE)

    def test_replay_does_not_touch_production_db_or_notifications(self):
        before_dev = len(self.win.db.query_device_data(limit=100000))
        before_alarm = len(self.win.db.query_alarms())
        before_event = len(self.win.db.query_event_records(limit=100000))
        notify = []
        old_show = self.win._tray.showMessage
        self.win._tray.showMessage = lambda *a, **k: notify.append(a)
        try:
            self.win._load_replay_pack_path(str(
                ROOT / "replay_data" / "dev_demo.json"))
            self.assertIsNotNone(self.win._replay_session)
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
        self.assertEqual(notify, [], "Replay 不得触发生产通知")

    def test_replay_pause_restart_seek_reset_state(self):
        self.win._load_replay_pack_path(str(
            ROOT / "replay_data" / "dev_demo.json"))
        sess = self.win._replay_session
        self.assertIsNotNone(sess)
        self.win._replay_play_pause()
        self.win._replay_tick()
        paused = sess.position
        self.win._replay_play_pause()
        self.win._replay_tick()
        self.assertEqual(sess.position, paused, "暂停后不应推进")
        self.win._replay_restart()
        self.assertEqual(sess.position, 0)
        self.assertEqual(sess.events, [])
        self.win._replay_slider_moved(100)
        self.assertGreater(sess.position, 0)
        self.assertLessEqual(sess.signal_time, 100 / 25.0 + 0.1)

    def test_replay_rejects_final_source(self):
        p = Path(self.tmp) / "bad.json"
        p.write_text(json.dumps({
            "version": "stage3_replay_pack_v1",
            "source": "final_test",
            "stitched": False,
            "sample_rate_hz": 25.0,
            "samples_mmhg": [1.0, 2.0],
        }), encoding="utf-8")
        with self.assertRaises(ValueError):
            ReplayPack.load(p)

    def test_replay_speed_does_not_change_signal_decisions(self):
        class FakeRuntime:
            def __init__(self):
                self.fed = []
            def reset(self):
                self.fed.clear()
            def feed_samples(self, values):
                self.fed.extend(values)
                return []
            def states(self):
                return {}

        pack = ReplayPack(
            version="stage3_replay_pack_v1", source="synthetic",
            stitched=True, sample_rate_hz=25.0,
            samples_mmhg=[float(i) for i in range(100)])
        a = ReplaySession(pack, FakeRuntime())
        b = ReplaySession(pack, FakeRuntime())
        a.play()
        b.play()
        a.advance(1.0, 1.0)
        b.advance(0.5, 2.0)
        self.assertEqual(a.position, b.position)
        self.assertEqual(a.runtime.fed, b.runtime.fed)
        self.assertTrue(a.pack.stitched)

    def test_event_records_use_transition_snapshot_key_times(self):
        base = {
            "episode_id": 999, "scale": "U15",
            "estimated_onset": 15.0, "first_prompt_time": 15.0,
            "first_active_time": None, "suspect_start": 15.0,
            "suspect_end": None, "active_intervals": [],
            "end_time": None, "end_reason": None,
            "low_confidence": False,
        }
        start_ep = dict(base)
        active_ep = {
            **base, "first_active_time": 15.0,
            "active_intervals": [{"start": 15.0, "end": None}],
        }
        active_end_ep = {
            **active_ep,
            "active_intervals": [{"start": 15.0, "end": 22.5}],
        }
        suspect_end_ep = {
            **active_end_ep, "suspect_end": 22.5,
            "end_time": 22.5, "end_reason": "NORMAL",
        }
        transitions = [
            {"transition_seq": 1, "kind": "SUSPECT_START", "time": 15.0,
             "scale": "U15", "episode_id": 999, "state": "SUSPECT",
             "episode": start_ep},
            {"transition_seq": 2, "kind": "ACTIVE_START", "time": 15.0,
             "scale": "U15", "episode_id": 999, "state": "ACTIVE",
             "episode": active_ep},
            {"transition_seq": 3, "kind": "ACTIVE_END", "time": 22.5,
             "scale": "U15", "episode_id": 999, "state": "SUSPECT",
             "episode": active_end_ep},
            {"transition_seq": 4, "kind": "SUSPECT_END", "time": 22.5,
             "scale": "U15", "episode_id": 999, "state": "NORMAL",
             "episode": suspect_end_ep},
        ]
        old_alert = self.win._trigger_event_alert
        self.win._trigger_event_alert = lambda *a, **k: None
        try:
            self.win._handle_stage3_transitions(transitions)
        finally:
            self.win._trigger_event_alert = old_alert
        rows = [r for r in self.win.db.query_event_records(limit=100)
                if r["episode_id"] == 999]
        by_state = {r["state"]: r for r in rows}
        self.assertEqual(by_state["ACTIVE_START"]["estimated_onset"], 15.0)
        self.assertEqual(by_state["ACTIVE_START"]["first_prompt_time"], 15.0)
        self.assertEqual(by_state["ACTIVE_START"]["first_active_time"], 15.0)
        self.assertIsNone(by_state["ACTIVE_START"]["end_time"])
        self.assertEqual(by_state["ACTIVE_END"]["estimated_onset"], 15.0)
        self.assertEqual(by_state["ACTIVE_END"]["first_active_time"], 15.0)
        self.assertEqual(by_state["ACTIVE_END"]["end_time"], 22.5)
        self.assertEqual(by_state["SUSPECT_END"]["end_time"], 22.5)
        self.assertEqual(by_state["ACTIVE_START"]["transition_seq"], 2)

    def test_event_table_migrates_old_schema_without_data_loss(self):
        path = os.path.join(self.tmp, "old_events.db")
        conn = sqlite3.connect(path)
        conn.execute(
            "CREATE TABLE event_records (id INTEGER PRIMARY KEY, "
            "timestamp TEXT, session_id TEXT, mode TEXT, scale TEXT, "
            "state TEXT, label TEXT, estimated_onset REAL, "
            "first_prompt_time REAL, first_active_time REAL, end_time REAL, "
            "source TEXT, confidence REAL, acknowledged INTEGER DEFAULT 0)")
        conn.execute(
            "INSERT INTO event_records (timestamp, scale, state) "
            "VALUES ('old', 'U15', 'SUSPECT_START')")
        conn.commit()
        conn.close()
        db = DatabaseManager(path)
        try:
            cols = {r[1] for r in db._conn.execute(
                "PRAGMA table_info(event_records)")}
            self.assertIn("episode_id", cols)
            self.assertIn("transition_seq", cols)
            self.assertEqual(len(db.query_event_records(limit=10)), 1)
        finally:
            db.close()

    def _write_replay_pack(self, name, data):
        p = Path(self.tmp) / name
        p.write_text(json.dumps(data), encoding="utf-8")
        return p

    def test_replay_rejects_dev_pack_with_final_case(self):
        p = self._write_replay_pack("leak.json", {
            "version": "stage3_replay_pack_v1", "source": "dev",
            "stitched": False, "sample_rate_hz": 25.0,
            "samples_mmhg": [1.0, 2.0], "case_ids": [3],
            "segments": [{"case_id": 3, "start_s": 0, "end_s": 1}],
        })
        with self.assertRaises(ValueError):
            ReplayPack.load(p)

    def test_replay_rejects_dev_segment_outside_whitelist(self):
        p = self._write_replay_pack("bad_segment.json", {
            "version": "stage3_replay_pack_v1", "source": "dev",
            "stitched": False, "sample_rate_hz": 25.0,
            "samples_mmhg": [1.0, 2.0], "case_ids": [1],
            "segments": [{"case_id": 3, "start_s": 0, "end_s": 1}],
        })
        with self.assertRaises(ValueError):
            ReplayPack.load(p)

    def test_replay_rejects_synthetic_real_case_ids(self):
        p = self._write_replay_pack("synthetic_case.json", {
            "version": "stage3_replay_pack_v1", "source": "synthetic",
            "stitched": False, "sample_rate_hz": 25.0,
            "samples_mmhg": [1.0, 2.0], "case_ids": [1],
        })
        with self.assertRaises(ValueError):
            ReplayPack.load(p)

    def test_replay_real_runtime_cumulative_scheduling(self):
        pack = ReplayPack.load(ROOT / "replay_data" / "dev_demo.json")
        a = ReplaySession(pack, Stage3Runtime())
        b = ReplaySession(pack, Stage3Runtime())
        self.assertTrue(a.runtime.available, a.runtime.error)
        a.play()
        b.play()
        for _ in range(30):
            a.advance(1.0, 1.0)       # 30 s at 1x
        for _ in range(30):
            b.advance(0.5, 2.0)       # 30 s signal in 15 s wall at 2x
        self.assertEqual(a.position, 750)
        self.assertEqual(a.position, b.position)
        self.assertEqual(
            json.dumps(a.runtime.states(), sort_keys=True, default=str),
            json.dumps(b.runtime.states(), sort_keys=True, default=str))
        self.assertEqual(a.events, b.events)
        u15_times = a.runtime.scales["U15"].decision_times
        self.assertEqual(u15_times[:3], [15.0, 22.5, 30.0])
        self.assertTrue(all(abs(d - 7.5) < 1e-9
                            for d in (u15_times[1] - u15_times[0],
                                      u15_times[2] - u15_times[1])))

    def test_replay_real_runtime_pause_and_seek(self):
        pack = ReplayPack.load(ROOT / "replay_data" / "dev_demo.json")
        a = ReplaySession(pack, Stage3Runtime())
        self.assertTrue(a.runtime.available, a.runtime.error)
        a.play()
        for _ in range(20):
            a.advance(1.0, 1.0)       # 20 s
        state_20 = json.dumps(a.runtime.states(), sort_keys=True,
                              default=str)
        events_20 = list(a.events)
        a.pause()
        pos = a.position
        a.advance(1.0, 1.0)
        self.assertEqual(a.position, pos)
        c = ReplaySession(pack, Stage3Runtime())
        c.seek(20.0)
        self.assertEqual(c.position, pos)
        self.assertEqual(
            json.dumps(c.runtime.states(), sort_keys=True, default=str),
            state_20)
        self.assertEqual(c.events, events_20)
        a.play()
        a.advance(1.0, 1.0)
        self.assertEqual(a.position, pos + 25)


if __name__ == "__main__":
    unittest.main(verbosity=2)
