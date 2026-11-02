# -*- coding: utf-8 -*-
"""Independent offline reference for the experimental Stage 4 host runtime.

The reference is built from the original waveform, the frozen feature/quality
functions, the saved Stage 4 models/calibrators/parameters and an independent
consensus implementation.  It deliberately does not use the host runtime's
history, scale series, or last-feature fields.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from features.si_features import FEATURE_NAMES, extract_si_features  # noqa: E402
from features.quality_states import (  # noqa: E402
    UNKNOWN_STATES, HistoryIndex, classify_signal_state, si_breathing_bins,
)
from training.stage2_calibration import apply_calibrator  # noqa: E402
from training.stage3_event_engine import EventEngine, params_from_dict  # noqa: E402
from training.stage4_fusion import (  # noqa: E402
    LOW_CONF_WEIGHT, WEIGHT_PROFILES, fusion_params_from_dict,
)

SCALES = (("U15", 15.0, 7.5), ("U30", 30.0, 15.0),
          ("U60", 60.0, 30.0), ("U90", 90.0, 45.0))
FS_FEATURE = 100
BIN_SEC = 15.0
COMPARE_TOL = 1e-9
_MISSING = "<missing>"


def _resample(x: np.ndarray, n_out: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    src = np.arange(len(x), dtype=float)
    dst = np.linspace(0.0, len(x) - 1.0, int(n_out))
    return np.interp(dst, src, x)


def _history(signal: np.ndarray, fs: int, start_s: float):
    bins = []
    n_bins = int(np.floor(start_s / BIN_SEC + 1e-9))
    for j in range(n_bins):
        a = int(round(j * BIN_SEC * fs))
        b = int(round((j + 1) * BIN_SEC * fs))
        seg = signal[a:b]
        if len(seg) < int(BIN_SEC * fs):
            continue
        seg100 = (_resample(seg, int(BIN_SEC * FS_FEATURE))
                  if fs != FS_FEATURE else seg)
        bs = si_breathing_bins(seg100)
        for x in bs:
            x = dict(x)
            x["start_s"] = float(j * BIN_SEC + x["start_s"])
            x["end_s"] = float(j * BIN_SEC + x["end_s"])
            bins.append(x)
    return HistoryIndex(bins)


def _load_assets(root: Path) -> tuple[dict, dict, dict, dict]:
    import joblib
    models, cals, params = {}, {}, {}
    for scale, _, _ in SCALES:
        payload = joblib.load(
            root / "models" / "stage2" / f"{scale.lower()}_rf.joblib")
        models[scale] = payload["model"]
        with open(root / "models" / "stage2" /
                  f"{scale.lower()}_calibrator.json", encoding="utf-8") as f:
            cals[scale] = json.load(f)
        with open(root / "models" / "stage4" /
                  f"{scale.lower()}_event_params.json", encoding="utf-8") as f:
            params[scale] = params_from_dict(json.load(f)["params"])
    with open(root / "models" / "stage4" /
              "fusion_event_params.json", encoding="utf-8") as f:
        fp = fusion_params_from_dict(json.load(f)["params"])
    return models, cals, params, fp


def _predict_full(clf, X):
    p = clf.predict_proba(X)
    out = np.zeros((len(X), 3), dtype=float)
    for j, c in enumerate(clf.classes_):
        out[:, int(c)] = p[:, j]
    return out


def _reference_fuse(series: dict, t: float, params) -> tuple[str, float | None]:
    """Independent consensus implementation from the frozen spec."""
    weights = WEIGHT_PROFILES[params.weight_profile]
    contrib = []
    for scale, ser in series.items():
        times = ser["times"]
        if len(times) == 0:
            continue
        idx = int(np.searchsorted(times, t, side="right") - 1)
        if idx < 0:
            continue
        age = float(t) - float(times[idx])
        step = float(ser["step_sec"])
        if age < 0 or age >= step - max(1e-6, step * 1e-6):
            continue
        state = ser["states"][idx]
        if state in UNKNOWN_STATES:
            continue
        if state == "VALID":
            v = float(ser["p"][idx])
            high, low = True, False
        elif state == "FLAT_WITH_HISTORY":
            v, high, low = 1.0, True, False
        elif state == "FLAT_EVER_HISTORY":
            v, high, low = 1.0, False, True
        else:
            continue
        w = float(weights.get(scale, 0.0))
        if w <= 0:
            continue
        if low:
            w *= LOW_CONF_WEIGHT
        decay = (1.0 if not np.isfinite(params.tau_sec) or params.tau_sec <= 0
                 else float(np.exp(-age / params.tau_sec)))
        contrib.append((w * decay, v, high))
    if not contrib:
        return "GAP", None
    high_count = sum(1 for _, _, h in contrib if h)
    if high_count == 0:
        return "FLAT_EVER_HISTORY", None
    denom = sum(w for w, _, _ in contrib)
    score = sum(w * v for w, v, _ in contrib) / denom
    if high_count < params.min_active_sources:
        score = min(score, params.event.active_enter_p - 1e-6)
    return "VALID", float(min(1.0, max(0.0, score)))


def build_reference(signal: np.ndarray, fs: int, case_end: float,
                    root: Path | None = None) -> dict:
    root = root or ROOT
    models, cals, params, fparams = _load_assets(root)
    signal = np.asarray(signal, dtype=float)
    out = {"fs": int(fs), "case_end": float(case_end), "scales": {},
           "fusion_frames": [], "fusion_transitions": []}
    series = {}
    for scale, ws, st in SCALES:
        decisions = []
        engine = EventEngine(params[scale], st, scale)
        t = ws
        while t <= case_end + 1e-9:
            end = int(np.floor(t * fs + 1e-9))
            count = int(round(ws * fs))
            start = end - count
            if start >= 0:
                w = signal[start:end]
                w100 = _resample(w, int(round(ws * FS_FEATURE))) \
                    if fs != FS_FEATURE else w
                start_s = t - ws
                hist = _history(signal, fs, start_s)
                feat = extract_si_features(w100)
                state = classify_signal_state(
                    w100, hist.summary(start_s), feature_diag=feat)
                p_u2 = None
                if state == "VALID":
                    p = apply_calibrator(
                        cals[scale],
                        _predict_full(models[scale], np.asarray(
                            [feat["vector"]], dtype=np.float32)))
                    p_u2 = float(p[0, 2])
                decisions.append({
                    "decision_time": float(t),
                    "window_start_sample": int(start),
                    "window_end_sample": int(end),
                    "start_s": float(start_s),
                    "history": {
                        "available": bool(hist.summary(start_s).available),
                        "has_breathing": (
                            None if hist.summary(start_s).has_breathing is None
                            else bool(hist.summary(start_s).has_breathing)),
                        "ever_has_breathing": (
                            None if hist.summary(start_s).ever_has_breathing
                            is None else bool(
                                hist.summary(start_s).ever_has_breathing)),
                        "lookback_sec": float(hist.summary(start_s).lookback_sec),
                        "A": (None if hist.summary(start_s).A is None
                              else float(hist.summary(start_s).A)),
                        "peaks": (None if hist.summary(start_s).peaks is None
                                  else int(hist.summary(start_s).peaks)),
                        "n_samples": int(hist.summary(start_s).n_samples),
                    },
                    "state": str(state),
                    "features": [float(v) for v in feat["vector"]],
                    "p_u2": p_u2,
                })
                engine.update(float(t), state, p_u2)
            t += st
        engine.finish(float(case_end))
        out["scales"][scale] = {
            "decisions": decisions,
            "transitions": [
                {"kind": x["kind"], "time": x["time"],
                 "episode_id": x.get("episode_id")}
                for x in engine.transitions],
        }
        series[scale] = {
            "scale": scale,
            "times": np.asarray([d["decision_time"] for d in decisions],
                                dtype=float),
            "states": tuple(d["state"] for d in decisions),
            "p": np.asarray([np.nan if d["p_u2"] is None else d["p_u2"]
                             for d in decisions], dtype=float),
            "window_sec": ws, "step_sec": st,
        }
    feng = EventEngine(fparams.event, 7.5, "FUSION")
    t = 15.0
    while t <= case_end + 1e-9:
        q, p = _reference_fuse(series, t, fparams)
        out["fusion_frames"].append([float(t), str(q),
                                     None if p is None else float(p)])
        feng.update(float(t), q, p)
        t += 7.5
    feng.finish(float(case_end))
    out["fusion_transitions"] = [
        {"kind": x["kind"], "time": x["time"],
         "episode_id": x.get("episode_id")} for x in feng.transitions]
    return out


def _finite(value) -> bool:
    try:
        return value is not None and bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _transition_rows(transitions) -> list[dict]:
    rows = []
    for item in transitions:
        if not isinstance(item, dict):
            rows.append({"kind": _MISSING, "time": _MISSING,
                         "episode_id": _MISSING})
            continue
        time = item.get("time", _MISSING)
        if isinstance(time, np.generic):
            time = time.item()
        rows.append({
            "kind": item.get("kind", _MISSING),
            "time": time,
            "episode_id": item.get("episode_id", _MISSING),
        })
    return rows


def _transition_equal(a: dict, b: dict) -> bool:
    if a["kind"] != b["kind"] or a["episode_id"] != b["episode_id"]:
        return False
    if a["time"] is _MISSING or b["time"] is _MISSING:
        return a["time"] == b["time"]
    try:
        return abs(float(a["time"]) - float(b["time"])) <= COMPARE_TOL
    except (TypeError, ValueError):
        return a["time"] == b["time"]


def _transition_diff(host_rows: list[dict],
                     reference_rows: list[dict]) -> dict:
    mismatch = abs(len(host_rows) - len(reference_rows))
    for a, b in zip(host_rows, reference_rows):
        if not _transition_equal(a, b):
            mismatch += 1
    return {
        "transition_host": host_rows,
        "transition_reference": reference_rows,
        "transition_mismatch": int(mismatch),
        "transition_equal": bool(
            len(host_rows) == len(reference_rows) and mismatch == 0),
    }


def _record_status(rec: dict, prefix: str) -> list[str]:
    reasons = []
    if not rec["frame_count_equal"]:
        reasons.append(f"{prefix}.frame_count")
    for key in ("decision_time_mismatch", "window_mismatch",
                "history_mismatch", "state_mismatch",
                "time_mismatch", "quality_mismatch", "frame_mismatch",
                "feature_dim_mismatch", "feature_nonfinite_host",
                "feature_nonfinite_reference", "feature_tolerance_exceeded",
                "p_none_mismatch", "p_nonfinite_host",
                "p_nonfinite_reference", "p_tolerance_exceeded",
                "transition_mismatch"):
        if rec.get(key, 0):
            reasons.append(f"{prefix}.{key}")
    rec["ok"] = not reasons
    rec["failure_reasons"] = reasons
    return reasons


def compare_host(host, reference: dict) -> dict:
    """Compare host decisions/fusion with the independent reference.

    Count differences, one-sided ``None`` probabilities, non-finite values and
    out-of-tolerance differences are explicit failures.  The reference was
    built from raw development data, not from host runtime history.
    """
    out = {"scales": {}, "fusion": {}}
    failures = []
    for scale, _, _ in SCALES:
        h = host.decision_audit(scale)
        r = reference["scales"][scale]["decisions"]
        rec = {
            "n_host": len(h), "n_reference": len(r),
            "frame_count_diff": len(h) - len(r),
            "frame_count_equal": len(h) == len(r),
            "decision_time_mismatch": 0, "window_mismatch": 0,
            "history_mismatch": 0, "state_mismatch": 0,
            "feature_dim_mismatch": 0,
            "feature_nonfinite_host": 0, "feature_nonfinite_reference": 0,
            "feature_tolerance": COMPARE_TOL,
            "feature_tolerance_exceeded": 0,
            "max_feature_abs_delta": None,
            "p_none_mismatch": 0,
            "p_nonfinite_host": 0, "p_nonfinite_reference": 0,
            "p_tolerance": COMPARE_TOL, "p_tolerance_exceeded": 0,
            "max_p_abs_delta": None,
        }
        fd, pd = [], []
        for a, b in zip(h, r):
            if abs(a["decision_time"] - b["decision_time"]) > COMPARE_TOL:
                rec["decision_time_mismatch"] += 1
            if (a["window_start_sample"] != b["window_start_sample"] or
                    a["window_end_sample"] != b["window_end_sample"]):
                rec["window_mismatch"] += 1
            if a["history"] != b["history"]:
                rec["history_mismatch"] += 1
            if a["state"] != b["state"]:
                rec["state_mismatch"] += 1
            fa = np.asarray(a["features"], dtype=float).ravel()
            fb = np.asarray(b["features"], dtype=float).ravel()
            if fa.shape != fb.shape:
                rec["feature_dim_mismatch"] += 1
            else:
                for x, y in zip(fa, fb):
                    xf, yf = _finite(x), _finite(y)
                    if not xf:
                        rec["feature_nonfinite_host"] += 1
                    if not yf:
                        rec["feature_nonfinite_reference"] += 1
                    if xf and yf:
                        delta = abs(float(x) - float(y))
                        fd.append(delta)
                        if delta > COMPARE_TOL:
                            rec["feature_tolerance_exceeded"] += 1
            pa, pb = a.get("p_u2"), b.get("p_u2")
            if (pa is None) != (pb is None):
                rec["p_none_mismatch"] += 1
            if pa is not None and not _finite(pa):
                rec["p_nonfinite_host"] += 1
            if pb is not None and not _finite(pb):
                rec["p_nonfinite_reference"] += 1
            if _finite(pa) and _finite(pb):
                delta = abs(float(pa) - float(pb))
                pd.append(delta)
                if delta > COMPARE_TOL:
                    rec["p_tolerance_exceeded"] += 1
        rec["max_feature_abs_delta"] = max(fd) if fd else None
        rec["max_p_abs_delta"] = max(pd) if pd else None
        rec.update(_transition_diff(
            _transition_rows(host.scale_transitions(scale)),
            _transition_rows(reference["scales"][scale]["transitions"])))
        failures.extend(_record_status(rec, scale))
        out["scales"][scale] = rec

    hf = host.fusion_frames()
    rf = reference["fusion_frames"]
    frec = {
        "n_host": len(hf), "n_reference": len(rf),
        "frame_count_diff": len(hf) - len(rf),
        "frame_count_equal": len(hf) == len(rf),
        "time_mismatch": 0, "quality_mismatch": 0, "frame_mismatch": 0,
        "p_none_mismatch": 0,
        "p_nonfinite_host": 0, "p_nonfinite_reference": 0,
        "p_tolerance": COMPARE_TOL, "p_tolerance_exceeded": 0,
        "max_p_abs_delta": None,
    }
    fd = []
    for a, b in zip(hf, rf):
        time_bad = abs(float(a[0]) - float(b[0])) > COMPARE_TOL
        quality_bad = a[1] != b[1]
        if time_bad:
            frec["time_mismatch"] += 1
        if quality_bad:
            frec["quality_mismatch"] += 1
        if time_bad or quality_bad:
            frec["frame_mismatch"] += 1
        pa, pb = a[2], b[2]
        if (pa is None) != (pb is None):
            frec["p_none_mismatch"] += 1
        if pa is not None and not _finite(pa):
            frec["p_nonfinite_host"] += 1
        if pb is not None and not _finite(pb):
            frec["p_nonfinite_reference"] += 1
        if _finite(pa) and _finite(pb):
            delta = abs(float(pa) - float(pb))
            fd.append(delta)
            if delta > COMPARE_TOL:
                frec["p_tolerance_exceeded"] += 1
    frec["max_p_abs_delta"] = max(fd) if fd else None
    frec.update(_transition_diff(
        _transition_rows(host.fusion_transitions()),
        _transition_rows(reference["fusion_transitions"])))
    failures.extend(_record_status(frec, "FUSION"))
    out["fusion"] = frec
    out["ok"] = not failures
    out["failure_reasons"] = failures
    return out
