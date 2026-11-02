# -*- coding: utf-8 -*-
"""
Stage 0 shared definitions for CF-REM.

This module is the executable counterpart of:
  - docs/specs/labels_spec_v1.md
  - docs/specs/event_spec_v1.md
  - docs/specs/quality_abstention_spec_v1.md

It deliberately does NOT train models or write to the existing cache.
Read-only with respect to cache/vitaldb.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CACHE_DIR = REPO_ROOT / "cache" / "vitaldb"
FS = 100

# Frozen event definitions (v1.1 + Stage 0 review decisions).
U2_TRACK = "etco2"
U2_THRESHOLD = 10.0          # reference EtCO2 below this (mmHg in VitalDB)
U2_CAP_SEC = 10              # forward-fill cap
U2_MIN_SEC = 10              # minimum event duration
U2_MERGE_SEC = 10            # merge same-class gaps

RR3_TRACK = "rr"
RR3_THRESHOLD = 3.0
RR3_CAP_SEC = 10
RR3_MIN_SEC = 10
RR3_MERGE_SEC = 20

# Window-overlap rule.
U2_POS_OVERLAP = 0.50
U2_IGNORE_OVERLAP = 0.10

# Reference reliability for RR/EtCO2 window medians.
MIN_REFERENCE_POINTS = 2

LABELS = ("U0", "U1", "U2", "U3", "ambiguous", "ignore",
          "ignore_low_support", "unlabeled", "other")
LABEL_NAMES_ZH = {
    "U0": "节律正常（未评估气体交换）",
    "U1": "急促",
    "U2": "参考 EtCO2 低值/呼气 CO2 缺失样事件（可靠支撑）",
    "U3": "慢呼吸/观察态",
    "ambiguous": "RR<3 但无 EtCO2<10 事件重叠",
    "ignore": "可靠事件边界（重叠 10–50%）",
    "ignore_low_support": "低支撑事件（≤1 条实际 EtCO2 读数），不判正/负",
    "unlabeled": "RR 缺失/参考不足",
    "other": "参考值越界",
}
LOW_SUPPORT_READINGS = 2   # <2 actual low readings = low-support event


def case_id_from_path(path: Path) -> int:
    m = re.search(r"(\d+)_i", path.name)
    if not m:
        raise ValueError(f"cannot parse case id from {path}")
    return int(m.group(1))


def list_case_ids() -> list[int]:
    """All cached cases, sorted numerically (the frozen order)."""
    paths = sorted(CACHE_DIR.glob("*.npz"), key=case_id_from_path)
    return [case_id_from_path(p) for p in paths]


def case_path(case_id: int, interval: float = 0.01) -> Path:
    return CACHE_DIR / f"{case_id}_i{interval}.npz"


def load_case(case_id: int) -> dict:
    path = case_path(case_id)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def ffill_100(x: np.ndarray, cap_sec: float | None) -> np.ndarray:
    """Forward-fill a 100 Hz reference track.

    Non-positive and NaN values are missing. A value is considered stale
    after ``cap_sec`` seconds and becomes NaN again.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return x.copy()
    valid = ~np.isnan(x) & (x > 0)
    idx = np.where(valid, np.arange(n), -1)
    idx = np.maximum.accumulate(idx)
    out = np.full(n, np.nan, dtype=float)
    m = idx >= 0
    out[m] = x[idx[m]]
    if cap_sec is not None:
        age = np.arange(n) - idx
        out[(idx >= 0) & (age > cap_sec * FS)] = np.nan
    return out


def sample_1s(x: np.ndarray, cap_sec: float | None) -> np.ndarray:
    """Forward-fill at 100 Hz, then sample one value per second."""
    return ffill_100(x, cap_sec)[::FS]


def ffill_source_indices(x: np.ndarray) -> np.ndarray:
    """Return, for every 100 Hz sample, the index of its ffill source.

    ``-1`` means there is no preceding valid sample. The cap is deliberately
    not applied here: callers only use this for grid cells that survived
    ``ffill_100``'s cap, so the source is necessarily within the cap.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    valid = np.isfinite(x) & (x > 0)
    idx = np.where(valid, np.arange(n, dtype=np.int64), -1)
    return np.maximum.accumulate(idx)


def _run_ids(mask: np.ndarray) -> np.ndarray:
    """Label each True sample with its contiguous run id (-1 elsewhere)."""
    mask = np.asarray(mask, dtype=bool)
    ids = np.full(len(mask), -1, dtype=np.int64)
    for k, (a, b) in enumerate(runs(mask)):
        ids[a:b] = k
    return ids


def _support_from_context(start_s: float, end_s: float,
                          threshold: float,
                          series: np.ndarray,
                          source_idx: np.ndarray,
                          low_run_id: np.ndarray,
                          valid_run_id: np.ndarray) -> dict:
    """Count raw reading runs that actually feed this event's 1 s grid.

    Frozen boundary rule (rev3): a raw reading may lie before the integer-
    second event start. It counts iff it is the ffill source of at least one
    low grid cell inside the event. Raw low samples that never become a grid
    cell's source are not readings of this event.
    """
    g0, g1 = int(start_s), int(end_s)
    grid = np.arange(g0, g1, dtype=np.int64)
    if len(grid) == 0:
        return {"low_readings": 0, "valid_readings": 0}
    seg = series[g0:g1]
    valid_cells = np.isfinite(seg)
    low_cells = valid_cells & (seg < threshold)

    src_valid = source_idx[grid[valid_cells] * FS]
    src_valid = src_valid[src_valid >= 0]
    valid_ids = valid_run_id[src_valid]
    valid_ids = valid_ids[valid_ids >= 0]

    src_low = source_idx[grid[low_cells] * FS]
    src_low = src_low[src_low >= 0]
    low_ids = low_run_id[src_low]
    low_ids = low_ids[low_ids >= 0]

    return {
        "low_readings": int(len(np.unique(low_ids))),
        "valid_readings": int(len(np.unique(valid_ids))),
    }


def count_actual_readings(track: np.ndarray, start_s: float, end_s: float,
                          threshold: float,
                          cap_sec: float | None = U2_CAP_SEC) -> dict:
    """Count actual monitor readings associated with an event interval.

    A "reading" is a contiguous run of valid samples, not every 100 Hz
    sample. Crucially, the run may start before ``start_s``: if its value is
    forward-filled into a low grid cell inside the event, it is the trigger
    reading for that cell. See ``_support_from_context``.
    """
    x = np.asarray(track, dtype=float)
    series = sample_1s(x, cap_sec)
    valid = np.isfinite(x) & (x > 0)
    low = valid & (x < threshold)
    return _support_from_context(
        start_s, end_s, threshold, series, ffill_source_indices(x),
        _run_ids(low), _run_ids(valid))


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open [start, end) runs of True on a 1 s grid."""
    if not mask.any():
        return []
    d = np.diff(np.r_[False, mask.astype(bool), False].astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def merge_intervals(intervals: Iterable[tuple[int, int]], gap_sec: int) -> list[tuple[int, int]]:
    """Merge half-open intervals when the gap between them is <= gap_sec.

    This legacy helper does NOT know about missing/unknown samples. Use
    ``merge_intervals_no_unknown`` for event timelines.
    """
    items = sorted(intervals)
    if not items:
        return []
    out = [list(items[0])]
    for s, e in items[1:]:
        if s - out[-1][1] <= gap_sec:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(int(s), int(e)) for s, e in out]


def merge_intervals_no_unknown(intervals: Iterable[tuple[int, int]],
                               missing: np.ndarray,
                               gap_sec: int) -> tuple[list[tuple[int, int]], int]:
    """Merge intervals only if the gap is short AND fully observed.

    ``missing`` is the 1 s unknown mask. A gap is mergeable only when
    ``s - prev_end <= gap_sec`` and ``missing[prev_end:s].any() == False``.
    Returns ``(merged, blocked_by_unknown_count)``.
    """
    items = sorted(intervals)
    if not items:
        return [], 0
    missing = np.asarray(missing, dtype=bool)
    out = [list(items[0])]
    blocked = 0
    for s, e in items[1:]:
        gap_start = out[-1][1]
        gap_len = s - gap_start
        if gap_len <= gap_sec:
            if gap_len > 0 and missing[gap_start:s].any():
                blocked += 1
                out.append([s, e])
            else:
                out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(int(s), int(e)) for s, e in out], blocked


def build_events(
    track: np.ndarray,
    threshold: float,
    cap_sec: float | None,
    min_sec: int,
    merge_gap_sec: int,
) -> dict:
    """Build event intervals on a 1 s grid.

    Frozen order: build runs -> merge -> drop short segments.
    Returns events plus the 1 s series and missing mask for audit.
    """
    series = sample_1s(track, cap_sec)
    mask = series < threshold
    missing = np.isnan(series)
    raw = runs(mask)
    merged, blocked = merge_intervals_no_unknown(raw, missing, merge_gap_sec)
    events = [(s, e) for s, e in merged if e - s >= min_sec]
    track = np.asarray(track, dtype=float)
    source_idx = ffill_source_indices(track)
    valid_raw = np.isfinite(track) & (track > 0)
    low_raw = valid_raw & (track < threshold)
    low_run_id = _run_ids(low_raw)
    valid_run_id = _run_ids(valid_raw)
    support = [
        _support_from_context(s, e, threshold, series, source_idx,
                              low_run_id, valid_run_id)
        for s, e in events
    ]
    return {
        "series": series,
        "missing": missing,
        "raw_runs": raw,
        "merged_runs": merged,
        "merge_blocked_by_unknown": int(blocked),
        "events": events,
        "support": support,
    }


def build_u2_events(etco2: np.ndarray, cap_sec: float = U2_CAP_SEC,
                    min_sec: int = U2_MIN_SEC,
                    merge_gap_sec: int = U2_MERGE_SEC) -> dict:
    return build_events(etco2, U2_THRESHOLD, cap_sec, min_sec, merge_gap_sec)


def split_events_by_support(result: dict,
                            min_readings: int = LOW_SUPPORT_READINGS) -> tuple[list, list]:
    """Split U2 events into reliable (>=min_readings) and low-support (<min).

    Returns (reliable_events, low_support_events). Low-support events must not
    be silently counted as positive or negative; they map to ignore_low_support.
    """
    events = result.get("events", [])
    support = result.get("support", [])
    reliable, low = [], []
    for ev, sup in zip(events, support):
        if int(sup.get("low_readings", 0)) >= min_readings:
            reliable.append(ev)
        else:
            low.append(ev)
    return reliable, low


def build_rr3_events(rr: np.ndarray, cap_sec: float = RR3_CAP_SEC,
                     min_sec: int = RR3_MIN_SEC,
                     merge_gap_sec: int = RR3_MERGE_SEC) -> dict:
    return build_events(rr, RR3_THRESHOLD, cap_sec, min_sec, merge_gap_sec)


def overlap_fraction(start_sample: int, window_samples: int,
                     events: list[tuple[int, int]]) -> float:
    """Fraction of a window covered by any event interval.

    Window/event times are converted to seconds; events are on a 1 s grid.
    """
    if window_samples <= 0:
        return 0.0
    ws = start_sample / FS
    we = (start_sample + window_samples) / FS
    overlap = 0.0
    for s, e in events:
        overlap += max(0.0, min(we, e) - max(ws, s))
    return float(min(1.0, overlap / (we - ws)))


def valid_values(seg: np.ndarray) -> np.ndarray:
    seg = np.asarray(seg, dtype=float)
    return seg[~np.isnan(seg) & (seg > 0)]


def reference_stats(rr_seg: np.ndarray, et_seg: np.ndarray) -> dict:
    rr_valid = valid_values(rr_seg)
    et_valid = valid_values(et_seg)
    return {
        "rr_median": float(np.median(rr_valid)) if len(rr_valid) else None,
        "et_median": float(np.median(et_valid)) if len(et_valid) else None,
        "rr_count": int(len(rr_valid)),
        "et_count": int(len(et_valid)),
    }


def classify_window(rr_seg: np.ndarray, et_seg: np.ndarray,
                    overlap: float, overlap_low: float = 0.0,
                    min_ref: int = MIN_REFERENCE_POINTS) -> dict:
    """Frozen Stage 0 window label order.

    ``overlap`` is overlap with reliable U2 events (>=2 actual low readings);
    ``overlap_low`` is overlap with low-support U2 events (<2 readings).

    Precedence:
      1. reliable overlap >= 0.50 -> U2
      2. low-support overlap >= 0.50 -> ignore_low_support
      3. reliable overlap >= 0.10 -> ignore
      4. low-support overlap >= 0.10 -> ignore_low_support
      5. RR missing/insufficient -> unlabeled
      6. RR < 3 -> ambiguous
      7. RR >= 18 -> U1
      8. 7 <= RR < 18 -> U0
      9. 3 <= RR < 7 -> U3
      10. otherwise -> other
    """
    st = reference_stats(rr_seg, et_seg)
    if overlap >= U2_POS_OVERLAP:
        return {**st, "overlap": overlap, "overlap_low": overlap_low, "label": "U2"}
    if overlap_low >= U2_POS_OVERLAP:
        return {**st, "overlap": overlap, "overlap_low": overlap_low,
                "label": "ignore_low_support"}
    if overlap >= U2_IGNORE_OVERLAP:
        return {**st, "overlap": overlap, "overlap_low": overlap_low,
                "label": "ignore"}
    if overlap_low >= U2_IGNORE_OVERLAP:
        return {**st, "overlap": overlap, "overlap_low": overlap_low,
                "label": "ignore_low_support"}
    if st["rr_median"] is None or st["rr_count"] < min_ref:
        return {**st, "overlap": overlap, "overlap_low": overlap_low,
                "label": "unlabeled"}
    rr = st["rr_median"]
    if not np.isfinite(rr):
        label = "other"
    elif rr < 3:
        label = "ambiguous"
    elif rr >= 18:
        label = "U1"
    elif rr >= 7:
        label = "U0"
    elif rr >= 3:
        label = "U3"
    else:
        label = "other"
    return {**st, "overlap": overlap, "overlap_low": overlap_low, "label": label}


def iter_windows(case_id: int, window_sec: float, step_sec: float,
                 u2_events: list[tuple[int, int]] | None = None,
                 low_support_events: list[tuple[int, int]] | None = None,
                 include_co2: bool = False,
                 case_data: dict | None = None) -> Iterator[dict]:
    """Yield production-compatible windows with Stage 0 labels.

    Uses preprocessing.make_windows_with_reference exactly; the only extra
    work is event-overlap labeling. Set include_co2=False for aggregate stats
    to avoid retaining 15 s arrays. Pass ``case_data``/``u2_events`` to reuse
    a loaded case across scales.
    """
    from preprocessing.make_windows import make_windows_with_reference

    d = case_data if case_data is not None else load_case(case_id)
    # Do not let the default path treat low-support events as reliable. If
    # either side of the partition is omitted, rebuild and split; pass []
    # explicitly to request no low-support handling.
    if u2_events is None or low_support_events is None:
        reliable, low = split_events_by_support(build_u2_events(d["etco2"]))
        if u2_events is None:
            u2_events = reliable
        if low_support_events is None:
            low_support_events = low
    ws = int(round(window_sec * FS))
    step = int(round(step_sec * FS))
    records = make_windows_with_reference(
        d["co2"], d["rr"], d["etco2"],
        window_size=ws, step=step, sampling_rate=FS)
    for r in records:
        ov = overlap_fraction(r["start"], ws, u2_events)
        ov_low = overlap_fraction(r["start"], ws, low_support_events)
        lab = classify_window(r["rr"], r["etco2"], ov, ov_low)
        rec = {
            "case_id": case_id,
            "start": int(r["start"]),
            "window_sec": float(window_sec),
            "overlap": ov,
            "overlap_low": ov_low,
            **lab,
        }
        if include_co2:
            rec["co2"] = r["co2"]
        yield rec


def peak_count(window: np.ndarray) -> int:
    """Peak count using the existing training-side definition."""
    from features.extract_features import find_breath_peaks
    from preprocessing.signal_processing import preprocess_window

    x, _ = preprocess_window(window, sampling_rate=FS)
    if x is None or len(x) == 0 or np.all(np.isnan(x)):
        return 0
    x = np.asarray(x, dtype=float)
    if np.isnan(x).any():
        # windows from valid segments should not have NaN; be defensive.
        finite = np.isfinite(x)
        if not finite.any():
            return 0
        idx = np.arange(len(x))
        x = np.interp(idx, idx[finite], x[finite])
    return int(len(find_breath_peaks(x, sampling_rate=FS)))


def longest_true_run(mask: np.ndarray) -> int:
    """Longest run of True, in samples."""
    if not mask.any():
        return 0
    d = np.diff(np.r_[False, mask.astype(bool), False].astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return int(np.max(ends - starts))


def longest_nan_run(x: np.ndarray) -> int:
    return longest_true_run(np.isnan(x))


def quality_metrics(window: np.ndarray) -> dict:
    """Pre-normalization quality metrics used by Stage 0 analysis.

    This is intentionally descriptive; the frozen quality gate is in
    docs/specs/quality_abstention_spec_v1.md and will be tuned in Stage 1.
    """
    x = np.asarray(window, dtype=float)
    n = len(x)
    if n == 0:
        return {
            "n": 0, "nan_frac": 1.0, "zero_frac": 1.0,
            "max_nan_run_sec": 0.0, "A": 0.0, "B": 0.0,
            "noise": 0.0, "noise_ratio": float("inf"),
            "noise_alt": 0.0, "noise_ratio_alt": float("inf"),
            "noise_hf": 0.0, "noise_ratio_hf": float("inf"),
            "zero_diff_frac": 1.0,
            "A_invalid": True, "flat_run_sec": None,
            "min": float("nan"), "max": float("nan"),
            "p99": float("nan"), "max_value_frac": 0.0,
        }
    nan_frac = float(np.isnan(x).mean())
    max_nan_run = longest_nan_run(x) / FS
    finite = x[np.isfinite(x)]
    if len(finite) == 0:
        return {
            "n": n, "nan_frac": nan_frac, "zero_frac": 1.0,
            "max_nan_run_sec": max_nan_run, "A": 0.0, "B": 0.0,
            "noise": 0.0, "noise_ratio": float("inf"),
            "noise_alt": 0.0, "noise_ratio_alt": float("inf"),
            "noise_hf": 0.0, "noise_ratio_hf": float("inf"),
            "zero_diff_frac": 1.0,
            "A_invalid": True, "flat_run_sec": None,
            "min": float("nan"), "max": float("nan"),
            "p99": float("nan"), "max_value_frac": 0.0,
        }
    zero_frac = float((finite == 0).sum() / len(finite))
    B = float(np.percentile(finite, 10))
    q90 = float(np.percentile(finite, 90))
    A = float(q90 - B)
    if len(finite) >= 2:
        dabs = np.abs(np.diff(finite))
        zero_diff_frac = float((dabs == 0).mean())
        noise = float(1.4826 * np.median(dabs) / math.sqrt(2.0))
    else:
        zero_diff_frac = 1.0
        noise = 0.0
    noise_ratio = float(noise / max(A, 1e-12))
    # Alternative noise estimate: residual from a short median filter.
    # MAD(diff) collapses to 0 on capnogram plateaus/quantization; this
    # residual estimate is reported for Stage 1 comparison.
    if len(finite) >= 5:
        from scipy.ndimage import median_filter
        resid = finite - median_filter(finite, size=5, mode="nearest")
        noise_alt = float(1.4826 * np.median(np.abs(resid - np.median(resid))))
    else:
        noise_alt = 0.0
    noise_ratio_alt = float(noise_alt / max(A, 1e-12))
    # High-frequency residual after a 2 Hz zero-phase low-pass. This is a
    # more appropriate noise estimate for capnograms than MAD(diff).
    if len(finite) >= 30:
        from scipy.signal import butter, filtfilt
        b, a = butter(3, 2.0 / (FS / 2.0), btype="low")
        smooth = filtfilt(b, a, finite)
        resid_hf = finite - smooth
        noise_hf = float(1.4826 * np.median(np.abs(resid_hf - np.median(resid_hf))))
    else:
        noise_hf = 0.0
    noise_ratio_hf = float(noise_hf / max(A, 1e-12))
    A_invalid = (not np.isfinite(A)) or A < 1e-6
    flat_run_sec = None
    if not A_invalid:
        z = (finite - B) / A
        flat_run_sec = longest_true_run(z < 0.15) / FS
    maxv = float(np.max(finite))
    max_value_frac = float((finite == maxv).sum() / len(finite))
    return {
        "n": n,
        "nan_frac": nan_frac,
        "zero_frac": zero_frac,
        "max_nan_run_sec": float(max_nan_run),
        "A": A,
        "B": B,
        "noise": noise,
        "noise_ratio": noise_ratio,
        "noise_alt": noise_alt,
        "noise_ratio_alt": noise_ratio_alt,
        "noise_hf": noise_hf,
        "noise_ratio_hf": noise_ratio_hf,
        "zero_diff_frac": zero_diff_frac,
        "A_invalid": bool(A_invalid),
        "flat_run_sec": flat_run_sec,
        "min": float(np.min(finite)),
        "max": maxv,
        "p99": float(np.percentile(finite, 99)),
        "max_value_frac": max_value_frac,
    }


def percentile_summary(values: Iterable[float], qs=(0, 10, 25, 50, 75, 90, 100)) -> dict:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return {"n": 0}
    return {"n": int(len(arr)), **{f"p{q}": float(np.percentile(arr, q)) for q in qs}}


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
