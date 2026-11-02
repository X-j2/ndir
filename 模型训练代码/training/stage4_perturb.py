# -*- coding: utf-8 -*-
"""Stage 4 waveform-only perturbations.

Perturbations operate on the CO2 waveform/input chain only.  Reference RR and
EtCO2 are never changed.  Missing samples are represented as NaN on the
original absolute sample grid; the time axis is never compressed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from stage0_common import (
    FS, build_u2_events, iter_windows, split_events_by_support,
)
from features.si_features import extract_si_features
from features.quality_states import (
    HistoryIndex, classify_signal_state, si_breathing_bins,
)
from training.stage2_data import LABEL_CODE, STATE_CODE

EPS = 1e-12


def case_context(co2: np.ndarray) -> dict:
    x = np.asarray(co2, dtype=float)
    finite = x[np.isfinite(x)]
    if not len(finite):
        return {"B": 0.0, "A": 0.0, "q1": 0.0, "q99": 0.0}
    B = float(np.percentile(finite, 10))
    A = float(np.percentile(finite, 90) - B)
    q1, q99 = np.percentile(finite, [1, 99])
    return {"B": B, "A": A, "q1": float(q1), "q99": float(q99)}


def _burst_drop_detail(x: np.ndarray, level: float, seed: int,
                       mode: str) -> tuple[np.ndarray, dict]:
    """Insert complete NaN bursts and return an auditable gap record.

    ``mode='window'`` inserts at most one burst fully inside a short window;
    ``mode='case'`` schedules one burst per minute and skips an incomplete
    tail instead of compressing it onto the final sample.
    """
    y = np.asarray(x, dtype=float).copy()
    L = len(y)
    n = int(round(float(level) * FS))
    old_nan = ~np.isfinite(y)
    info = {
        "kind": "burst_drop", "mode": mode, "seed": int(seed),
        "level_sec": float(level), "requested_samples": int(n),
        "window_samples": int(L), "gaps": [], "status": "ok",
        "tail_skipped": 0, "existing_nan_samples": int(old_nan.sum()),
    }
    if n <= 0:
        info["status"] = "zero_length"
        return y, info
    if L < n:
        info["status"] = "insufficient_window"
        info.update({"actual_gap_samples": 0, "new_missing_samples": 0,
                     "new_missing_fraction": 0.0,
                     "missing_before_fraction": float(old_nan.mean()
                                                      if L else 0.0),
                     "missing_after_fraction": float(old_nan.mean()
                                                     if L else 0.0)})
        return y, info
    rng = np.random.default_rng(int(seed))
    gaps = []
    if mode == "window":
        start = int(rng.integers(0, L - n + 1))
        gaps.append((start, start + n))
    elif mode == "case":
        step = int(60 * FS)
        for base in range(0, L, step):
            # A full burst must fit after this minute's base and within L.
            latest = min(base + step - n, L - n)
            if latest < base:
                info["tail_skipped"] += 1
                continue
            start = base + int(rng.integers(0, latest - base + 1))
            if start + n > L:
                info["tail_skipped"] += 1
                continue
            gaps.append((start, start + n))
    else:
        raise ValueError(f"unknown burst mode: {mode}")
    new_mask = np.zeros(L, dtype=bool)
    for a, b in gaps:
        y[a:b] = np.nan
        new_mask[a:b] = ~old_nan[a:b]
    info["gaps"] = [{"start": int(a), "end": int(b),
                     "length": int(b - a)} for a, b in gaps]
    info["actual_gap_samples"] = int(sum(b - a for a, b in gaps))
    info["status"] = "ok" if gaps else info["status"]
    info["new_missing_samples"] = int(new_mask.sum())
    info["new_missing_fraction"] = float(new_mask.mean() if L else 0.0)
    info["missing_before_fraction"] = float(old_nan.mean() if L else 0.0)
    info["missing_after_fraction"] = float((~np.isfinite(y)).mean()
                                           if L else 0.0)
    return y, info


def perturb_signal_detail(co2: np.ndarray, kind: str, level: float,
                          seed: int = 0, mode: str = "window"
                          ) -> tuple[np.ndarray, dict]:
    """Apply one perturbation and optionally return burst-gap metadata."""
    if kind == "burst_drop":
        return _burst_drop_detail(co2, level, seed, mode)
    return perturb_signal(co2, kind, level, seed), {}


def perturb_signal(co2: np.ndarray, kind: str, level: float,
                   seed: int = 0, mode: str = "window") -> np.ndarray:
    if kind == "burst_drop":
        return _burst_drop_detail(co2, level, seed, mode)[0]
    x = np.asarray(co2, dtype=float).copy()
    ctx = case_context(x)
    A = max(float(ctx["A"]), EPS)
    rng = np.random.default_rng(int(seed))
    if kind == "none":
        return x
    if kind == "gain":
        return x * float(level)
    if kind == "bias":
        return x + float(level) * A
    if kind == "noise":
        return x + rng.normal(0.0, float(level), size=len(x))
    if kind == "drop":
        frac = float(level)
        if frac <= 0:
            return x
        x[rng.random(len(x)) < min(1.0, frac)] = np.nan
        return x
    if kind == "lag":
        tau = float(level)
        if tau <= 0:
            return x
        from scipy.signal import lfilter
        alpha = 1.0 - np.exp(-1.0 / (FS * tau))
        y = np.full_like(x, np.nan)
        finite = np.isfinite(x)
        if not finite.any():
            return y
        # Filter each finite run independently: an IIR filter must not carry
        # state or NaN across a missing-sample gap.
        d = np.diff(np.r_[False, finite, False].astype(np.int8))
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            seg = x[a:b]
            yy, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], seg,
                            zi=[(1.0 - alpha) * seg[0]])
            y[a:b] = yy
        return np.asarray(y, dtype=float)
    if kind == "drift":
        slope = float(level) * A / 60.0  # A per minute
        t = np.arange(len(x), dtype=float) / FS
        return x + slope * t
    if kind == "gamma":
        gamma = float(level)
        B, A0 = float(ctx["B"]), max(float(ctx["A"]), EPS)
        z = (x - B) / A0
        y = A0 * np.sign(z) * np.power(np.abs(z), gamma) + B
        y[~np.isfinite(x)] = np.nan
        return y
    if kind == "clip":
        return np.clip(x, ctx["q1"], ctx["q99"])
    if kind == "negative":
        return -x
    if kind == "quantize":
        bits = int(level)
        finite = x[np.isfinite(x)]
        if not len(finite):
            return x
        lo, hi = float(np.min(finite)), float(np.max(finite))
        if hi <= lo:
            return x
        step = (hi - lo) / (2 ** bits - 1)
        y = x.copy()
        finite_mask = np.isfinite(x)
        y[finite_mask] = np.round((x[finite_mask] - lo) / step) * step + lo
        return y
    raise ValueError(f"unknown perturbation: {kind}")


def build_perturbed_case_rows(case: dict, case_id: int,
                              window_sec: float, step_sec: float,
                              kind: str, level: float, seed: int = 0
                              ) -> tuple[dict, dict]:
    """Build Stage 2-shaped rows from a perturbed waveform.

    The original reference tracks and truth events are reused unchanged.
    """
    y, gap_audit = perturb_signal_detail(
        case["co2"], kind, level, seed, mode="case")
    d = {"co2": y, "rr": case["rr"], "etco2": case["etco2"]}
    reliable, low = split_events_by_support(build_u2_events(case["etco2"]))
    hist = HistoryIndex(si_breathing_bins(y))
    starts, labels, states, feats = [], [], [], []
    for rec in iter_windows(
            case_id, window_sec, step_sec, u2_events=reliable,
            low_support_events=low, include_co2=True, case_data=d):
        feat = extract_si_features(rec["co2"])
        state = classify_signal_state(
            rec["co2"], hist.summary(rec["start"] / FS),
            feature_diag=feat)
        starts.append(int(rec["start"]))
        labels.append(LABEL_CODE[rec["label"]])
        vec = np.asarray(feat["vector"], dtype=float)
        fmax = float(np.finfo(np.float32).max)
        if not np.isfinite(vec).all() or np.any(np.abs(vec) > fmax):
            # Extreme nonlinear perturbations can overflow a derived feature.
            # Treat it as an abstention, never feed infinity to sklearn.
            state = "NOISY"
            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
            vec = np.clip(vec, -fmax, fmax)
        states.append(STATE_CODE[state])
        feats.append([float(v) for v in vec])
    n = len(starts)
    data = {
        "case_id": np.full(n, int(case_id), dtype=np.int32),
        "start": np.asarray(starts, dtype=np.int32),
        "label": np.asarray(labels, dtype=np.int8),
        "state": np.asarray(states, dtype=np.int8),
        "features": np.asarray(feats, dtype=np.float32),
        "ref_valid_frac": np.ones(n, dtype=np.float32),
        "A_invalid": np.zeros(n, dtype=bool),
        "constant_flat": np.zeros(n, dtype=bool),
        "quantile_artifact": np.zeros(n, dtype=bool),
    }
    return data, {
        "case_id": int(case_id),
        "n_windows": n,
        "n_missing_samples": int(np.sum(~np.isfinite(y))),
        "n_samples": int(len(y)),
        "missing_fraction": float(np.mean(~np.isfinite(y))) if len(y) else 1.0,
        "gap_audit": gap_audit,
    }
