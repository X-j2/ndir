# -*- coding: utf-8 -*-
"""
Causal quality/abstention state for CF-REM Stage 0.

Only the current window and a summary of the PAST are accepted. There is no
case-level / future-data argument by design. This is the executable version of
docs/specs/quality_abstention_spec_v1.md.

History is split into:
  - recent breathing: a verified breathing bin within the last 300 s
  - ever breathing:   a verified breathing bin anywhere earlier in the record
Both are computed from bins whose END is <= window start; future bins are
never used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stage0_common import FS, peak_count, quality_metrics

A_FLOOR_DEFAULT = 1.0          # mmHg in VitalDB; device must use relative A
MIN_HISTORY_SEC = 30.0
HISTORY_LOOKBACK_SEC = 300.0
HISTORY_BIN_SEC = 15.0


@dataclass(frozen=True)
class HistorySummary:
    available: bool
    has_breathing: bool | None          # recent <=300 s
    ever_has_breathing: bool | None     # any time earlier in the record
    lookback_sec: float
    A: float | None
    peaks: int | None
    n_samples: int


def summarize_history(prior: np.ndarray,
                      a_floor: float = A_FLOOR_DEFAULT,
                      min_history_sec: float = MIN_HISTORY_SEC,
                      ever_has_breathing: bool | None = None) -> HistorySummary:
    """Summarize breathing evidence in a PAST-ONLY waveform.

    The caller must pass ``co2[start-lookback:start]``; never pass the whole
    case. If there is not enough history, ``available=False`` and
    ``has_breathing=None``. ``ever_has_breathing`` can be supplied by a
    causal bin scan; if omitted it defaults to the recent result.
    """
    prior = np.asarray(prior, dtype=float)
    n = len(prior)
    if n < int(min_history_sec * FS):
        return HistorySummary(False, None, None, n / FS, None, None, n)
    q = quality_metrics(prior)
    peaks = int(peak_count(prior))
    has = bool(np.isfinite(q["A"]) and q["A"] >= a_floor and peaks >= 2)
    ever = has if ever_has_breathing is None else bool(ever_has_breathing)
    return HistorySummary(True, has, ever, n / FS, float(q["A"]), peaks, n)


def breathing_bins(co2: np.ndarray,
                   bin_sec: float = HISTORY_BIN_SEC,
                   step_sec: float = HISTORY_BIN_SEC,
                   a_floor: float = A_FLOOR_DEFAULT) -> list[dict]:
    """Precompute 15 s breathing bins over a case.

    A bin is a candidate history only when its END is <= the query window
    start; the caller must enforce that via ``history_from_bins``.
    """
    x = np.asarray(co2, dtype=float)
    win = int(bin_sec * FS)
    step = int(step_sec * FS)
    bins = []
    for start in range(0, max(0, len(x) - win + 1), step):
        seg = x[start:start + win]
        q = quality_metrics(seg)
        peaks = int(peak_count(seg))
        bins.append({
            "start_s": start / FS,
            "end_s": (start + win) / FS,
            "A": float(q["A"]),
            "peaks": peaks,
            "has_breathing": bool(np.isfinite(q["A"]) and q["A"] >= a_floor and peaks >= 2),
        })
    return bins


def history_from_bins(bins: list[dict], start_s: float,
                      recent_sec: float = HISTORY_LOOKBACK_SEC,
                      min_history_sec: float = MIN_HISTORY_SEC) -> HistorySummary:
    """Causal history summary at ``start_s``.

    Only bins with ``end_s <= start_s`` are eligible. A future bin cannot
    influence the result even if it contains obvious breathing.
    """
    past = [b for b in bins if b["end_s"] <= start_s]
    if not past:
        return HistorySummary(False, None, None, 0.0, None, None, 0)
    past_sec = sum(b["end_s"] - b["start_s"] for b in past)
    if past_sec < min_history_sec:
        return HistorySummary(False, None, None, past_sec, None, None,
                              int(past_sec * FS))
    recent = [b for b in past if b["start_s"] >= start_s - recent_sec]
    has = bool(any(b["has_breathing"] for b in recent))
    ever = bool(any(b["has_breathing"] for b in past))
    A = max((b["A"] for b in past), default=None)
    peaks = max((b["peaks"] for b in past), default=None)
    return HistorySummary(True, has, ever, past_sec, A, peaks, int(past_sec * FS))


def classify_quality_state(window: np.ndarray,
                           history: HistorySummary,
                           a_floor: float = A_FLOOR_DEFAULT,
                           nan_frac_max: float = 0.10,
                           max_nan_run_sec: float = 2.0,
                           saturated: bool = False,
                           polarity_invalid: bool = False,
                           noise_ok: bool = True) -> str:
    """Return one of the frozen Stage 0 quality states.

    States: GAP, SATURATED, POLARITY_INVALID, NOISY, FLAT_WITH_HISTORY,
    FLAT_EVER_HISTORY, FLAT_NO_HISTORY, FLAT_HISTORY_UNKNOWN, VALID.
    """
    q = quality_metrics(window)
    if q["nan_frac"] > nan_frac_max or q["max_nan_run_sec"] > max_nan_run_sec:
        return "GAP"
    if saturated:
        return "SATURATED"
    if polarity_invalid:
        return "POLARITY_INVALID"
    flat_run = q["flat_run_sec"]
    if flat_run is None and q["A_invalid"]:
        flat_run = len(window) / FS
    flat = bool((q["A"] < a_floor) or (flat_run is not None and flat_run >= 10.0))
    if not flat:
        return "VALID" if noise_ok else "NOISY"
    if not history.available:
        return "FLAT_HISTORY_UNKNOWN"
    if history.has_breathing is True:
        return "FLAT_WITH_HISTORY"
    if history.ever_has_breathing is True:
        # No breathing in the last 300 s, but the record had breathing
        # earlier: a long-apnea candidate, lower confidence than recent.
        return "FLAT_EVER_HISTORY"
    return "FLAT_NO_HISTORY"
