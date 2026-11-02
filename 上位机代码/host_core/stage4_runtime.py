"""Experimental Stage 4 four-scale + consensus runtime.

This module is deliberately separate from the default Stage 3 runtime.  It is
used only by the isolated replay experiment; it never writes the production
database, never triggers notifications, and never controls the serial port.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from host_core.paths import ensure_training_path
from host_core.runtime import FS_RAW, HistoryTracker, ScaleRuntime

SCALES = (("U15", 15.0, 7.5), ("U30", 30.0, 15.0),
          ("U60", 60.0, 30.0), ("U90", 90.0, 45.0))


def _make_modules(root: Path) -> dict:
    from features.si_features import FEATURE_NAMES, extract_si_features
    from features.quality_states import (
        HistoryIndex, classify_signal_state, si_breathing_bins,
    )
    from training.stage2_calibration import apply_calibrator
    from training.stage3_event_engine import EventEngine, params_from_dict
    from training.stage4_fusion import fuse_at_time, fusion_params_from_dict

    def predict_full(clf, X):
        p = clf.predict_proba(X)
        out = np.zeros((len(X), 3), dtype=float)
        for j, c in enumerate(clf.classes_):
            out[:, int(c)] = p[:, j]
        return out

    return {
        "extract_si_features": extract_si_features,
        "classify_signal_state": classify_signal_state,
        "si_breathing_bins": si_breathing_bins,
        "HistoryIndex": HistoryIndex,
        "apply_calibrator": apply_calibrator,
        "EventEngine": EventEngine,
        "params_from_dict": params_from_dict,
        "predict_full": predict_full,
        "FEATURE_NAMES": FEATURE_NAMES,
        "fuse_at_time": fuse_at_time,
        "fusion_params_from_dict": fusion_params_from_dict,
    }


class Stage4Runtime:
    """Four independent scales plus a causal 7.5 s consensus engine."""

    def __init__(self, training_root: Path | None = None):
        self.root = training_root or ensure_training_path()
        self.available = False
        self.error = ""
        self.modules = {}
        self.models = {}
        self.calibrators = {}
        self.params = {}
        self.fusion_params = None
        self._loaded = False
        self._load()

    def _load(self) -> None:
        if self.root is None:
            self.error = "未找到模型训练仓库"
            return
        try:
            import joblib
            self.modules = _make_modules(self.root)
            for scale, _, _ in SCALES:
                model_path = self.root / "models" / "stage2" / f"{scale.lower()}_rf.joblib"
                cal_path = self.root / "models" / "stage2" / f"{scale.lower()}_calibrator.json"
                par_path = self.root / "models" / "stage4" / f"{scale.lower()}_event_params.json"
                if not par_path.exists() and scale in ("U15", "U30"):
                    par_path = self.root / "models" / "stage3" / f"{scale.lower()}_event_params.json"
                if not (model_path.exists() and cal_path.exists() and par_path.exists()):
                    raise FileNotFoundError(f"{scale} Stage 4 model/config missing")
                payload = joblib.load(model_path)
                model = payload["model"]
                try:
                    model.n_jobs = 1
                except Exception:
                    pass
                self.models[scale] = model
                with open(cal_path, encoding="utf-8") as f:
                    self.calibrators[scale] = json.load(f)
                with open(par_path, encoding="utf-8") as f:
                    pj = json.load(f)
                self.params[scale] = self.modules["params_from_dict"](
                    pj.get("params", pj))
            fusion_path = self.root / "models" / "stage4" / "fusion_event_params.json"
            if not fusion_path.exists():
                raise FileNotFoundError("fusion_event_params.json missing")
            with open(fusion_path, encoding="utf-8") as f:
                fj = json.load(f)
            self.fusion_params = self.modules["fusion_params_from_dict"](
                fj["params"])
            self._loaded = True
            self.reset()
            self.available = True
        except Exception as exc:
            self.error = str(exc)
            self.available = False

    def reset(self) -> None:
        if not self._loaded:
            return
        self.history = HistoryTracker(self.modules)
        self.scales = {
            scale: ScaleRuntime(
                scale, ws, step, self.models[scale],
                self.calibrators[scale], self.params[scale],
                self.history, self.modules, audit=True)
            for scale, ws, step in SCALES
        }
        self._frames = {s: [] for s, _, _ in SCALES}
        self._frame_count = {s: 0 for s, _, _ in SCALES}
        self._fusion_frames = []
        self._next_fusion = 15.0
        self._fusion_engine = self.modules["EventEngine"](
            self.fusion_params.event, 7.5, "FUSION")
        self._last_fused_quality = "WAITING"
        self._last_fused_p = None

    def _series(self) -> dict:
        out = {}
        for scale, _, step in SCALES:
            frames = self._frames[scale]
            out[scale] = {
                "scale": scale,
                "times": np.asarray([x[0] for x in frames], dtype=float),
                "states": tuple(x[1] for x in frames),
                "p": np.asarray([x[2] for x in frames], dtype=float),
                "window_sec": dict((s, w) for s, w, _ in SCALES)[scale],
                "step_sec": float(step),
            }
        return out

    def _advance_fusion(self, current_time: float) -> list[dict]:
        out = []
        series = None
        while self._next_fusion <= current_time + 1e-9:
            if series is None:
                series = self._series()
            q, p = self.modules["fuse_at_time"](
                series, self._next_fusion, self.fusion_params)
            self._last_fused_quality = q
            self._last_fused_p = p
            self._fusion_frames.append(
                (float(self._next_fusion), str(q),
                 None if p is None else float(p)))
            out.extend(self._fusion_engine.update(self._next_fusion, q, p))
            self._next_fusion += 7.5
        return out

    def feed_samples(self, values) -> list[dict]:
        if not self.available:
            return []
        vals = [float(x) for x in values]
        if not vals:
            return []
        self.history.add(vals)
        total = self.history.total
        out = []
        for scale, _, _ in SCALES:
            out.extend(self.scales[scale].process(total))
            frames = self.scales[scale].decision_frames
            start = self._frame_count[scale]
            if len(frames) > start:
                self._frames[scale].extend(frames[start:])
                self._frame_count[scale] = len(frames)
        out.extend(self._advance_fusion(total / FS_RAW))
        return out

    def finish(self, case_end: float | None = None) -> list[dict]:
        if not self.available:
            return []
        t = float(case_end) if case_end is not None else self.history.total / FS_RAW
        out = []
        for scale, _, _ in SCALES:
            out.extend(self.scales[scale].finish(t))
        out.extend(self._advance_fusion(t))
        out.extend(self._fusion_engine.finish(t))
        return out

    def interrupt(self, at_time: float | None = None,
                  reason: str = "DATA_INTERRUPTION") -> list[dict]:
        if not self.available:
            return []
        out = []
        for scale, _, _ in SCALES:
            out.extend(self.scales[scale].engine.interrupt(at_time, reason))
        out.extend(self._fusion_engine.interrupt(at_time, reason))
        return out

    def decision_audit(self, scale: str) -> list[dict]:
        return list(self.scales[scale].audit_records)

    def scale_transitions(self, scale: str) -> list[dict]:
        return list(self.scales[scale].engine.transitions)

    def fusion_frames(self) -> list[tuple[float, str, float | None]]:
        return list(self._fusion_frames)

    def fusion_transitions(self) -> list[dict]:
        return list(self._fusion_engine.transitions)

    def states(self) -> dict:
        if not self.available:
            return {}
        out = {
            s: {
                "state": self.scales[s].engine.state,
                "quality": self.scales[s].last_state,
                "p_u2": self.scales[s].last_p,
                "decision_time": self.scales[s].last_decision_time,
                "event": self.scales[s].engine.current_event(),
            }
            for s, _, _ in SCALES
        }
        out["FUSION"] = {
            "state": self._fusion_engine.state,
            "quality": self._last_fused_quality,
            "p_u2": self._last_fused_p,
            "decision_time": self._fusion_engine._last_decision_time,
            "event": self._fusion_engine.current_event(),
        }
        return out

    def event_text(self) -> str:
        if not self.available:
            return f"Stage4 实验引擎不可用: {self.error}"
        st = self.states()
        return "Stage4实验 · " + " | ".join(
            f"{s}:{st[s]['state']}" for s in ("U15", "U30", "U60",
                                              "U90", "FUSION"))
