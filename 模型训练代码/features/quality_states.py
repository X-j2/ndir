# -*- coding: utf-8 -*-
"""
Stage 1 quality/abstention states built on the frozen Stage 0 history summary.

Stage 0 files remain unchanged. This module adds:
  - fast causal history lookup (HistoryIndex);
  - disconnect (GAP), saturation and polarity checks;
  - an explicit UNKNOWN family so A=0 flat windows are not discarded.

Only the current window and past 15 s bins are used. Future bins never enter.
"""
from __future__ import annotations

import numpy as np

from stage0_common import FS, longest_true_run, quality_metrics
from features.si_features import SPEC, extract_si_features
from tools.quality_state import (
    HISTORY_LOOKBACK_SEC, MIN_HISTORY_SEC, HistorySummary,
)

NOISE_RATIO_HF_MAX = float(SPEC["quality"]["noise_ratio_hf_max"])
SATURATION_FRACTION_MAX = float(SPEC["quality"]["saturation_fraction_max"])
SATURATION_RUN_SEC = float(SPEC["quality"]["saturation_run_sec"])
LOW_AMPLITUDE_FRAC = float(SPEC["quality"]["low_amplitude_frac"])
HISTORY_PEAKS_MIN = int(SPEC["quality"]["history_requires_si_peaks_ge"])

UNKNOWN_STATES = frozenset({
    "GAP",                  # signal interruption / missing waveform
    "SATURATED",            # rail or clipping
    "POLARITY_INVALID",     # negative gain / inverted trace
    "QUANTILE_ARTIFACT",    # A=0 but finite samples still vary (sparse pulse)
    "NOISY",                # high-frequency noise
    "FLAT_NO_HISTORY",      # flat but no credible breathing history
    "FLAT_HISTORY_UNKNOWN", # not enough past to judge
})
FLAT_CANDIDATE_STATES = frozenset({"FLAT_WITH_HISTORY", "FLAT_EVER_HISTORY"})
LOW_CONFIDENCE_STATES = frozenset({"FLAT_EVER_HISTORY"})


def state_is_unknown(state: str) -> bool:
    return state in UNKNOWN_STATES


class HistoryIndex:
    """Fast causal history lookup over non-overlapping 15 s bins.

    Equivalent in semantics to ``history_from_bins`` but uses binary search
    so a full 200-case sweep does not rescan every bin for every window.
    """

    def __init__(self, bins: list[dict],
                 recent_sec: float = HISTORY_LOOKBACK_SEC,
                 min_history_sec: float = MIN_HISTORY_SEC):
        items = sorted(bins, key=lambda b: (b["start_s"], b["end_s"]))
        self.starts = np.asarray([b["start_s"] for b in items], dtype=float)
        self.ends = np.asarray([b["end_s"] for b in items], dtype=float)
        self.has = np.asarray([bool(b["has_breathing"]) for b in items], dtype=bool)
        self.A = np.asarray([
            float(b["A"]) if b.get("A") is not None else np.nan
            for b in items
        ], dtype=float)
        self.peaks = np.asarray([b["peaks"] for b in items], dtype=int)
        self.recent_sec = float(recent_sec)
        self.min_history_sec = float(min_history_sec)

    def summary(self, start_s: float) -> HistorySummary:
        n = int(np.searchsorted(self.ends, start_s, side="right"))
        if n == 0:
            return HistorySummary(False, None, None, 0.0, None, None, 0)
        past_sec = float(np.sum(self.ends[:n] - self.starts[:n]))
        if past_sec < self.min_history_sec:
            return HistorySummary(False, None, None, past_sec, None, None,
                                  int(past_sec * FS))
        recent_start = start_s - self.recent_sec
        r = int(np.searchsorted(self.starts[:n], recent_start, side="left"))
        has = bool(np.any(self.has[r:n]))
        ever = bool(np.any(self.has[:n]))
        # A_ref comes only from bins that are both quality-valid and
        # breathing-evidence bins (SI peaks >= 2). A slow drift bin with a
        # large A but zero peaks must never raise A_ref.
        breath_mask = self.has[:n] & np.isfinite(self.A[:n])
        past_A = self.A[:n][breath_mask]
        A = float(np.max(past_A)) if len(past_A) else None
        peaks = int(np.max(self.peaks[:n]))
        return HistorySummary(True, has, ever, past_sec, A, peaks,
                              int(past_sec * FS))


def gap_metrics(window: np.ndarray) -> dict:
    q = quality_metrics(window)
    return {
        "nan_frac": float(q["nan_frac"]),
        "max_nan_run_sec": float(q["max_nan_run_sec"]),
    }


def detect_disconnect(window: np.ndarray,
                      nan_frac_max: float = 0.10,
                      max_nan_run_sec: float = 2.0) -> bool:
    """Missing/NaN waveform is a signal interruption, not apnea evidence."""
    m = gap_metrics(window)
    return bool(m["nan_frac"] > nan_frac_max or
                m["max_nan_run_sec"] > max_nan_run_sec)


def saturation_metrics(window: np.ndarray,
                       rail_low: float | None = None,
                       rail_high: float | None = None) -> dict:
    x = np.asarray(window, dtype=float)
    finite = np.isfinite(x)
    n = int(finite.sum())
    if n == 0 or (rail_low is None and rail_high is None):
        return {"at_rail_frac": 0.0, "longest_rail_run_sec": 0.0,
                "saturated": False}
    at = np.zeros(len(x), dtype=bool)
    if rail_low is not None:
        at |= finite & np.isclose(x, float(rail_low), rtol=1e-6, atol=1e-9)
    if rail_high is not None:
        at |= finite & np.isclose(x, float(rail_high), rtol=1e-6, atol=1e-9)
    frac = float(at.sum() / n)
    run_sec = longest_true_run(at) / FS
    return {
        "at_rail_frac": frac,
        "longest_rail_run_sec": float(run_sec),
        "saturated": bool(frac > SATURATION_FRACTION_MAX or
                          run_sec > SATURATION_RUN_SEC),
    }


def detect_saturation(window: np.ndarray,
                      rail_low: float | None = None,
                      rail_high: float | None = None) -> bool:
    return bool(saturation_metrics(window, rail_low, rail_high)["saturated"])


def detect_polarity_invalid(window: np.ndarray) -> bool:
    from features.si_features import extract_si_features
    return bool(extract_si_features(window)["polarity_invalid"])


def noise_is_ok(window: np.ndarray,
                noise_ratio_hf_max: float = NOISE_RATIO_HF_MAX) -> bool:
    q = quality_metrics(window)
    return bool(np.isfinite(q["noise_ratio_hf"]) and
                q["noise_ratio_hf"] <= noise_ratio_hf_max)


def si_breathing_bins(co2: np.ndarray,
                      bin_sec: float = 15.0,
                      step_sec: float = 15.0,
                      rail_low: float | None = None,
                      rail_high: float | None = None,
                      noise_ratio_hf_max: float = NOISE_RATIO_HF_MAX) -> list[dict]:
    """Causal history bins using SI peaks and quality screening.

    A bin is credible breathing history only if it is not GAP/SATURATED/
    POLARITY_INVALID/QUANTILE_ARTIFACT/NOISY and the SI adaptive peak
    detector finds at least two peaks. No absolute amplitude threshold is
    used; the SI peak test is invariant to positive gain/offset.
    """
    x = np.asarray(co2, dtype=float)
    win = int(bin_sec * FS)
    step = int(step_sec * FS)
    bins = []
    for start in range(0, max(0, len(x) - win + 1), step):
        seg = x[start:start + win]
        diag = extract_si_features(seg)
        gap = bool(diag["nan_frac"] > 0.10 or
                   diag["max_nan_run_sec"] > 2.0)
        sat = detect_saturation(seg, rail_low, rail_high)
        polar = bool(diag["polarity_invalid"])
        artifact = bool(diag["quantile_artifact"])
        noisy = bool(np.isfinite(diag["noise_ratio_hf"]) and
                     diag["noise_ratio_hf"] > noise_ratio_hf_max)
        quality_ok = not (gap or sat or polar or artifact or noisy)
        peaks = int(diag["features"]["peak_count"])
        has_breathing = bool(quality_ok and peaks >= HISTORY_PEAKS_MIN)
        # Only breathing-evidence bins may define A_ref. A zero-peak drift
        # segment is quality-valid but must not pollute the amplitude
        # reference used by the relative low-amplitude test.
        A = (float(diag["A"]) if has_breathing and not diag["A_invalid"]
             else np.nan)
        bins.append({
            "start_s": start / FS,
            "end_s": (start + win) / FS,
            "A": A,
            "A_raw": (float(diag["A"]) if quality_ok and not diag["A_invalid"]
                      else np.nan),
            "peaks": peaks,
            "has_breathing": has_breathing,
            "quality_ok": bool(quality_ok),
            "gap": bool(gap),
            "saturated": bool(sat),
            "polarity_invalid": bool(polar),
            "quantile_artifact": bool(artifact),
            "noisy": bool(noisy),
            "noise_ratio_hf": float(diag["noise_ratio_hf"]),
        })
    return bins


def classify_signal_state(window: np.ndarray,
                          history: HistorySummary,
                          rail_low: float | None = None,
                          rail_high: float | None = None,
                          noise_ratio_hf_max: float = NOISE_RATIO_HF_MAX,
                          low_amplitude_frac: float = LOW_AMPLITUDE_FRAC,
                          polarity_invalid: bool | None = None,
                          noise_ratio_hf: float | None = None,
                          feature_diag: dict | None = None) -> str:
    """Full Stage 1 state decision.

    Order: disconnect/GAP -> saturation -> polarity -> quantile artifact ->
    noise (for non-A=0 windows) -> flat/history. A=0 is never itself a reason
    to discard a window, but it covers two distinct cases: a genuinely
    constant window (all finite samples equal), which with causal breathing
    history remains a U2 candidate; and a quantile-degenerate window that
    still contains sparse pulses or other variation, which is
    QUANTILE_ARTIFACT/UNKNOWN. A non-zero low-amplitude noisy window is
    NOISY/UNKNOWN, not a flat candidate; the noise check must not be bypassed
    by the low-amplitude branch.

    A caller that already computed ``extract_si_features`` may pass the
    result as ``feature_diag``; this avoids a second pass over the window.
    """
    diag = feature_diag if feature_diag is not None else extract_si_features(window)
    if diag["nan_frac"] > 0.10 or diag["max_nan_run_sec"] > 2.0:
        return "GAP"
    if detect_saturation(window, rail_low, rail_high):
        return "SATURATED"
    if polarity_invalid is None:
        polarity_invalid = bool(diag["polarity_invalid"])
    if polarity_invalid:
        return "POLARITY_INVALID"
    # q90-q10=0 is not the same as a constant trace. Sparse pulses can make
    # both quantiles equal the baseline while real variation remains. Such a
    # window is an artifact/UNKNOWN, never a flat-apnea candidate, even when
    # breathing history is present. Only an all-equal finite trace keeps the
    # A=0 candidate path below.
    if bool(diag.get("quantile_artifact", False)):
        return "QUANTILE_ARTIFACT"

    A = float(diag["A"])
    a_invalid = bool(diag["A_invalid"])
    flat_run = diag["flat_run_sec"]
    if noise_ratio_hf is None:
        noise_ratio_hf = diag["noise_ratio_hf"]
    if flat_run is None and a_invalid:
        flat_run = len(window) / FS
    # Noise is checked before the flat/low-amplitude branch. A nonzero
    # low-amplitude signal whose 2 Hz residual dominates is NOISY/UNKNOWN,
    # not a flat-apnea candidate. Exact A=0 is exempt because it is the
    # scale-free "no amplitude at all" path and has zero residual.
    noise_ok = bool(np.isfinite(noise_ratio_hf) and
                    noise_ratio_hf <= noise_ratio_hf_max)
    if not a_invalid and not noise_ok:
        return "NOISY"
    # Scale-free flat test: A=0, a long flat run in z, or low amplitude
    # *relative to causal history*. Never use a fixed absolute A floor:
    # that would make the state depend on unknown sensor gain.
    low_amp = bool(history.A is not None and np.isfinite(history.A) and
                   history.A > 0 and A < low_amplitude_frac * history.A)
    flat = bool(a_invalid or low_amp or
                (flat_run is not None and flat_run >= 10.0))
    if not flat:
        return "VALID"
    if not history.available:
        return "FLAT_HISTORY_UNKNOWN"
    if history.has_breathing is True:
        return "FLAT_WITH_HISTORY"
    if history.ever_has_breathing is True:
        return "FLAT_EVER_HISTORY"
    return "FLAT_NO_HISTORY"
