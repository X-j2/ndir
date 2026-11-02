"""Causal U15/U30 runtime for the host relative mode.

The host feeds 25 Hz CO2 samples (frame values are already converted from ppm
to mmHg for waveform shape).  A 25 -> 100 Hz interpolation reuses the reviewed
Stage 1 feature definition.  Decisions are made only when a window ends; the
Stage 3 engine receives quality state and calibrated U2 probability.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from host_core.paths import ensure_training_path

FS_RAW = 25.0
FS_FEATURE = 100


def _resample_25_to_100(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    n_out = len(x) * (FS_FEATURE // int(FS_RAW))
    src = np.arange(len(x), dtype=float)
    dst = np.linspace(0.0, len(x) - 1.0, n_out)
    return np.interp(dst, src, x)


class HistoryTracker:
    """15 s causal quality-history bins; future bins are filtered by HistoryIndex."""

    def __init__(self, modules):
        self.modules = modules
        self.fs = int(FS_RAW)
        self.bin_samples = int(15 * self.fs)
        self.samples: list[float] = []
        self.base = 0
        self.total = 0
        self.next_bin_end = self.bin_samples
        self.bins: list[dict] = []

    def add(self, values) -> None:
        for v in values:
            self.samples.append(float(v))
            self.total += 1
        while self.next_bin_end <= self.total:
            start = self.next_bin_end - self.bin_samples
            end = self.next_bin_end
            i0 = start - self.base
            i1 = end - self.base
            if i0 >= 0 and i1 <= len(self.samples):
                seg = np.asarray(self.samples[i0:i1], dtype=float)
                seg100 = _resample_25_to_100(seg)
                bins = self.modules["si_breathing_bins"](seg100)
                if bins:
                    b = dict(bins[0])
                    b["start_s"] = start / self.fs
                    b["end_s"] = end / self.fs
                    self.bins.append(b)
            self.next_bin_end += self.bin_samples
        keep = int(360 * self.fs)
        if len(self.samples) > keep:
            drop = len(self.samples) - keep
            self.samples = self.samples[drop:]
            self.base += drop

    def window(self, count: int, end: int) -> np.ndarray | None:
        start = end - count
        i0 = start - self.base
        i1 = end - self.base
        if i0 < 0 or i1 > len(self.samples):
            return None
        return np.asarray(self.samples[i0:i1], dtype=float)

    def summary(self, start_s: float):
        return self.modules["HistoryIndex"](self.bins).summary(float(start_s))


class ScaleRuntime:
    def __init__(self, scale: str, window_sec: float, step_sec: float,
                 model, calibrator: dict, params, history: HistoryTracker,
                 modules, audit: bool = False):
        self.scale = scale
        self.window_sec = float(window_sec)
        self.step_sec = float(step_sec)
        self.model = model
        self.calibrator = calibrator
        self.history = history
        self.modules = modules
        self.audit = bool(audit)
        self.audit_records: list[dict] = []
        self.window_samples = int(round(window_sec * FS_RAW))
        # Decision times are exact multiples of the configured step.  At
        # 25 Hz, U15's 7.5 s step maps to alternating 188/187 sample windows;
        # using a fixed 188-sample step would silently become 7.52 s.
        self._next_decision_index = 0
        self.decision_times: list[float] = []
        self.engine = modules["EventEngine"](params, self.step_sec, scale)
        self.last_state = "WAITING"
        self.last_p = None
        self.last_decision_time = None
        self.last_features = None
        self.decision_frames: list[tuple[float, str, float | None]] = []

    @property
    def next_decision_time(self) -> float:
        return self.window_sec + self._next_decision_index * self.step_sec

    def _end_sample_for(self, decision_t: float) -> int:
        # floor, never ceil: feature windows must not include future samples.
        return int(math.floor(decision_t * FS_RAW + 1e-9))

    def process(self, total_samples: int) -> list[dict]:
        out = []
        current_s = total_samples / FS_RAW
        while current_s + 1e-9 >= self.next_decision_time:
            decision_t = self.next_decision_time
            end = self._end_sample_for(decision_t)
            w25 = self.history.window(self.window_samples, end)
            if w25 is None:
                self._next_decision_index += 1
                continue
            w100 = _resample_25_to_100(w25)
            feat = self.modules["extract_si_features"](w100)
            start_s = decision_t - self.window_sec
            hist = self.history.summary(start_s)
            state = self.modules["classify_signal_state"](
                w100, hist, feature_diag=feat)
            p_u2 = None
            if state == "VALID":
                p_raw = self.modules["predict_full"](
                    self.model, np.asarray([feat["vector"]], dtype=np.float32))
                p_raw = self.modules["apply_calibrator"](
                    self.calibrator, p_raw)
                p_u2 = float(p_raw[0, 2])
            self.last_state = state
            self.last_p = p_u2
            self.last_decision_time = decision_t
            self.last_features = np.asarray(feat["vector"], dtype=float)
            self.decision_times.append(decision_t)
            self.decision_frames.append((float(decision_t), str(state), p_u2))
            if self.audit:
                self.audit_records.append({
                    "scale": self.scale,
                    "decision_time": float(decision_t),
                    "window_start_sample": int(end - self.window_samples),
                    "window_end_sample": int(end),
                    "window_start_s": float((end - self.window_samples) / FS_RAW),
                    "window_end_s": float(end / FS_RAW),
                    "start_s": float(start_s),
                    "state": str(state),
                    "quality": str(self.last_state),
                    "history": {
                        "available": bool(hist.available),
                        "has_breathing": (None if hist.has_breathing is None
                                          else bool(hist.has_breathing)),
                        "ever_has_breathing": (
                            None if hist.ever_has_breathing is None
                            else bool(hist.ever_has_breathing)),
                        "lookback_sec": float(hist.lookback_sec),
                        "A": (None if hist.A is None else float(hist.A)),
                        "peaks": (None if hist.peaks is None
                                  else int(hist.peaks)),
                        "n_samples": int(hist.n_samples),
                    },
                    "features": [float(x) for x in feat["vector"]],
                    "p_u2": (None if p_u2 is None else float(p_u2)),
                })
            out.extend(self.engine.update(decision_t, state, p_u2))
            self._next_decision_index += 1
        return out

    def finish(self, case_end: float) -> list[dict]:
        return self.engine.finish(float(case_end))


class Stage3Runtime:
    """Two independent single-scale engines over one shared sample history."""

    def __init__(self, training_root: Path | None = None):
        self.root = training_root or ensure_training_path()
        self.available = False
        self.error = ""
        self.models = {}
        self.calibrators = {}
        self.params = {}
        self.modules = {}
        self._loaded = False
        self._load()

    def _load(self) -> None:
        if self.root is None:
            self.error = "未找到模型训练仓库"
            return
        try:
            from features.si_features import FEATURE_NAMES, extract_si_features
            from features.quality_states import (
                HistoryIndex, classify_signal_state, si_breathing_bins,
            )
            from training.stage2_calibration import apply_calibrator
            from training.stage3_event_engine import EventEngine, params_from_dict

            def predict_full(clf, X):
                p = clf.predict_proba(X)
                out = np.zeros((len(X), 3), dtype=float)
                for j, c in enumerate(clf.classes_):
                    out[:, int(c)] = p[:, j]
                return out

            self.modules = {
                "extract_si_features": extract_si_features,
                "classify_signal_state": classify_signal_state,
                "si_breathing_bins": si_breathing_bins,
                "HistoryIndex": HistoryIndex,
                "apply_calibrator": apply_calibrator,
                "EventEngine": EventEngine,
                "params_from_dict": params_from_dict,
                "predict_full": predict_full,
                "FEATURE_NAMES": FEATURE_NAMES,
            }
            import joblib
            root = self.root
            for scale, ws, step in (("U15", 15.0, 7.5), ("U30", 30.0, 15.0)):
                model_path = root / "models" / "stage2" / f"{scale.lower()}_rf.joblib"
                cal_path = root / "models" / "stage2" / f"{scale.lower()}_calibrator.json"
                par_path = root / "models" / "stage3" / f"{scale.lower()}_event_params.json"
                if not (model_path.exists() and cal_path.exists()):
                    raise FileNotFoundError(f"{scale} model/calibrator missing")
                payload = joblib.load(model_path)
                model = payload["model"]
                # The saved RF was trained with n_jobs=-1.  Parallel
                # predict_proba changes floating-point summation order
                # between runs; force single-thread inference so replay at
                # different tick speeds is bit-identical.
                try:
                    model.n_jobs = 1
                except Exception:
                    pass
                self.models[scale] = model
                with open(cal_path, encoding="utf-8") as f:
                    self.calibrators[scale] = json.load(f)
                if par_path.exists():
                    with open(par_path, encoding="utf-8") as f:
                        pj = json.load(f)
                    self.params[scale] = params_from_dict(pj["params"])
                else:
                    self.params[scale] = params_from_dict({})
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
                self.history, self.modules)
            for scale, ws, step in (("U15", 15.0, 7.5), ("U30", 30.0, 15.0))
        }

    def feed_samples(self, values) -> list[dict]:
        if not self.available:
            return []
        vals = [float(x) for x in values]
        if not vals:
            return []
        self.history.add(vals)
        total = self.history.total
        out = []
        for scale in ("U15", "U30"):
            out.extend(self.scales[scale].process(total))
        return out

    def finish(self, case_end: float | None = None) -> list[dict]:
        if not self.available:
            return []
        t = float(case_end) if case_end is not None else self.history.total / FS_RAW
        out = []
        for scale in ("U15", "U30"):
            out.extend(self.scales[scale].finish(t))
        return out

    def interrupt(self, at_time: float | None = None,
                  reason: str = "DATA_INTERRUPTION") -> list[dict]:
        """Apply the same expected-deadline interruption rule as offline eval."""
        if not self.available:
            return []
        out = []
        for scale in ("U15", "U30"):
            out.extend(self.scales[scale].engine.interrupt(at_time, reason))
        return out

    def states(self) -> dict:
        if not self.available:
            return {}
        return {
            s: {
                "state": self.scales[s].engine.state,
                "quality": self.scales[s].last_state,
                "p_u2": self.scales[s].last_p,
                "decision_time": self.scales[s].last_decision_time,
                "event": self.scales[s].engine.current_event(),
            }
            for s in ("U15", "U30")
        }

    def event_text(self) -> str:
        if not self.available:
            return f"事件引擎不可用: {self.error}"
        labels = []
        for s in ("U15", "U30"):
            st = self.scales[s].engine.state
            labels.append(f"{s}: {st}")
        return "事件引擎 · " + " | ".join(labels)
