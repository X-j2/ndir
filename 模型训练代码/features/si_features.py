# -*- coding: utf-8 -*-
"""
15-dimensional scale-invariant (SI) features for CF-REM Stage 1.

The feature definitions are frozen in ``features/feature_spec_si_v2.json``.
Only the uncalibrated CO2 waveform is used. Reference RR/EtCO2 tracks are
labels, never inputs.

The transform is invariant to x -> a*x + b for a > 0 (unknown positive gain
and offset). Negative gain is not silently flipped; the quality layer marks
it POLARITY_INVALID.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, peak_widths

SPEC_PATH = Path(__file__).with_name("feature_spec_si_v2.json")
with open(SPEC_PATH, "r", encoding="utf-8") as _f:
    SPEC = json.load(_f)

FS = int(SPEC["sampling_rate_hz"])
EPS = float(SPEC["normalization"]["eps"])
A_INVALID_EPS = float(SPEC["quality"]["a_invalid_eps"])
FEATURE_NAMES = tuple(SPEC["features"])
N_FEATURES = len(FEATURE_NAMES)

_LOWPASS_COEFFS = None


def _lowpass_coeffs():
    global _LOWPASS_COEFFS
    if _LOWPASS_COEFFS is None:
        from scipy.signal import butter
        _LOWPASS_COEFFS = butter(3, 2.0 / (FS / 2.0), btype="low")
    return _LOWPASS_COEFFS


def _smooth(x: np.ndarray, size: int) -> np.ndarray:
    return uniform_filter1d(np.asarray(x, dtype=float), size=size, mode="nearest")


def _fill_short_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linear-interpolate short NaN gaps, leaving long gaps as NaN."""
    x = np.asarray(x, dtype=float).copy()
    bad = ~np.isfinite(x)
    if not bad.any() or bad.all():
        return x
    valid = ~bad
    idx = np.arange(len(x))
    # Runs of NaN: interpolate only runs <= max_gap.
    d = np.diff(np.r_[False, bad, False].astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    for a, b in zip(starts, ends):
        if b - a <= max_gap and a > 0 and b < len(x):
            x[a:b] = np.linspace(x[a - 1], x[b], b - a + 2)[1:-1]
    return x


def _finite_fill(x: np.ndarray) -> np.ndarray:
    """Interpolate any remaining finite gaps; if none, return zeros."""
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if finite.all():
        return x
    if not finite.any():
        return np.zeros_like(x)
    idx = np.arange(len(x))
    return np.interp(idx, idx[finite], x[finite])


def _run_length(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0
    d = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return int(np.max(ends - starts))


def _acf_peak(z: np.ndarray, lag_min: int, lag_max: int) -> float:
    n = len(z)
    if n < max(3, lag_min + 1):
        return 0.0
    z0 = z - np.mean(z)
    denom = float(np.dot(z0, z0))
    if denom <= EPS:
        return 0.0
    nfft = 1 << int(np.ceil(np.log2(2 * n - 1)))
    spec = np.fft.rfft(z0, n=nfft)
    acf = np.fft.irfft(spec * np.conj(spec), n=nfft)[:n]
    acf = acf / denom
    hi = min(lag_max, n - 1)
    if hi < lag_min:
        return 0.0
    return float(np.clip(np.max(acf[lag_min:hi + 1]), -1.0, 1.0))


def _band_ratio(z: np.ndarray) -> float:
    n = len(z)
    if n < 4:
        return 0.0
    # Use the whole window so a 15 s window has the frequency resolution
    # described in the spec (about 0.067 Hz per bin).
    power = np.abs(np.fft.rfft(z)) ** 2
    freq = np.fft.rfftfreq(n, d=1.0 / FS)
    low = float(SPEC["band_ratio"]["low_hz"])
    high = float(SPEC["band_ratio"]["high_hz"])
    total_high = float(SPEC["band_ratio"]["total_high_hz"])
    total = float(np.sum(power[(freq > 0) & (freq <= total_high)]))
    band = float(np.sum(power[(freq >= low) & (freq <= high)]))
    return band / max(total, EPS)


def _polarity_invalid(x: np.ndarray) -> bool:
    """Detect an inverted capnogram without silently rectifying it.

    Detection is on the raw (not z-normalized) trace: quantile normalization
    maps both a waveform and its inversion to the same [0,1] scale, so z alone
    cannot reveal the sign. Prominence is scale-relative (0.12*A).
    """
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    if len(finite) < 30:
        return False
    # In the frozen VitalDB/device convention the raw CO2 trace is
    # non-negative. A wholly negative trace is the unambiguous negative-gain
    # case. A negative gain combined with a large positive offset is not
    # identifiable without a known gain sign; that limitation is documented.
    if float(np.median(finite)) >= 0.0:
        return False
    A = float(np.percentile(finite, 90) - np.percentile(finite, 10))
    if not np.isfinite(A) or A < EPS:
        return False
    sm = _smooth(x, int(SPEC["peaks"]["smooth_samples"]))
    prom = 0.12 * A
    dist = max(1, int(round(1.0 * FS)))
    pos, pp = find_peaks(sm, prominence=prom, distance=dist)
    neg, nprom = find_peaks(-sm, prominence=prom, distance=dist)
    if len(neg) == 0:
        return False
    return bool(len(neg) > len(pos)) or (len(pos) == 0 and len(neg) >= 2)


def _flat_run_sec(z: np.ndarray, a_invalid: bool, window_sec: float) -> float:
    if a_invalid:
        return float(window_sec)
    mask = z < float(SPEC["quality"]["flat_z_threshold"])
    return _run_length(mask) / FS


def _detect_peaks(z: np.ndarray, acf_peak: float, noise_ratio: float) -> dict:
    sm = _smooth(z, int(SPEC["peaks"]["smooth_samples"]))
    pk = SPEC["peaks"]

    def run(cfg):
        return find_peaks(
            sm,
            height=cfg["height_z"],
            prominence=cfg["prominence_z"],
            distance=max(1, int(round(cfg["distance_sec"] * FS))),
            width=(cfg["width_min_sec"] * FS, cfg["width_max_sec"] * FS),
            rel_height=cfg["rel_height"],
        )

    peaks, props = run(pk["pass1"])
    used_pass = 1
    if len(peaks) < 2:
        req = pk["pass2_requires"]
        if acf_peak > req["acf_peak_gt"] and noise_ratio < req["noise_ratio_lt"]:
            peaks2, props2 = run(pk["pass2"])
            if len(peaks2) >= 2:
                peaks, props, used_pass = peaks2, props2, 2
    return {"peaks": peaks, "props": props, "pass": used_pass, "smoothed": sm}


def normalize_si(window: np.ndarray) -> dict:
    """Return B/A/z and quality diagnostics for one 100 Hz window."""
    x = np.asarray(window, dtype=float)
    x = _fill_short_gaps(x, int(SPEC["normalization"]["missing_fill_max_gap_samples"]))
    x_f = _finite_fill(x)
    n = len(x_f)
    finite = x_f[np.isfinite(x_f)]
    if n == 0 or len(finite) == 0:
        B = A = 0.0
        z = np.zeros(max(n, 0), dtype=float)
        A_invalid = True
    else:
        B = float(np.percentile(finite, SPEC["normalization"]["lower_quantile"]))
        q90 = float(np.percentile(finite, SPEC["normalization"]["upper_quantile"]))
        A = float(q90 - B)
        # A=0 (or non-finite) is the only scale-free invalid case. Any finite
        # positive A is valid; classifying "low amplitude" is done relative
        # to causal history, never with a fixed absolute floor.
        A_invalid = (not np.isfinite(A)) or A <= A_INVALID_EPS
        z = np.zeros(n, dtype=float) if A_invalid else (x_f - B) / A
    finite_span = (float(np.max(finite) - np.min(finite))
                   if len(finite) else 0.0)
    # A=q90-q10 can be zero for two different reasons:
    #   1. a genuinely constant window (all finite samples equal);
    #   2. a quantile-degenerate window where >80% of samples sit on one
    #      level but sparse pulses or brief excursions still exist.
    # Only case 1 is a flat/apnea candidate. Case 2 is an artifact, not
    # evidence of absent breathing, and must not inherit the A=0 candidate
    # path. This test is exact and uses no absolute amplitude floor.
    constant_flat = bool(A_invalid and len(finite) > 0 and finite_span == 0.0)
    quantile_artifact = bool(A_invalid and finite_span > 0.0)
    if len(finite) >= 2:
        dabs = np.abs(np.diff(finite))
        zero_diff_frac = float((dabs == 0).mean())
        noise = float(1.4826 * np.median(dabs) / np.sqrt(2.0))
    else:
        zero_diff_frac = 1.0
        noise = 0.0
    noise_ratio = noise / max(A, EPS)
    if len(finite) >= 30:
        from scipy.signal import filtfilt
        b, a = _lowpass_coeffs()
        smooth = filtfilt(b, a, finite)
        resid = finite - smooth
        noise_hf = float(1.4826 * np.median(np.abs(resid - np.median(resid))))
    else:
        noise_hf = 0.0
    noise_ratio_hf = noise_hf / max(A, EPS)
    return {
        "x": x_f,
        "B": B,
        "A": A,
        "z": z,
        "A_invalid": bool(A_invalid),
        "finite_count": int(len(finite)),
        "finite_span": finite_span,
        "constant_flat": constant_flat,
        "quantile_artifact": quantile_artifact,
        "noise": noise,
        "noise_ratio": float(noise_ratio),
        "noise_hf": float(noise_hf),
        "noise_ratio_hf": float(noise_ratio_hf),
        "zero_diff_frac": zero_diff_frac,
    }


def extract_si_features(window: np.ndarray, fs: int = FS) -> dict:
    """Compute the frozen 15-dimensional SI vector plus diagnostics.

    If A=q90-q10 is invalid, all z-derived features are set to finite neutral
    values and ``flat_run_frac=1``. Diagnostics distinguish a genuinely
    constant window (``constant_flat=True``) from a quantile-degenerate
    window that still contains sparse pulses or other variation
    (``quantile_artifact=True``). The quality layer, not the feature layer,
    decides whether the window is a candidate or UNKNOWN.
    """
    window = np.asarray(window, dtype=float)
    if fs != FS:
        raise ValueError(f"SI features require {FS} Hz input, got {fs}")
    norm = normalize_si(window)
    A, z = norm["A"], norm["z"]
    a_invalid = norm["A_invalid"]
    window_sec = len(window) / FS

    acf_lag_min = int(round(SPEC["acf"]["lag_min_sec"] * FS))
    acf_lag_max = int(round(SPEC["acf"]["lag_max_sec"] * FS))
    acf_peak = _acf_peak(z, acf_lag_min, acf_lag_max)
    band_ratio = _band_ratio(z) if not a_invalid else 0.0
    flat_run_sec = _flat_run_sec(z, a_invalid, window_sec)
    polarity_invalid = _polarity_invalid(norm["x"]) if not a_invalid else False

    if a_invalid:
        peaks = np.array([], dtype=int)
        widths = np.array([], dtype=float)
        prominences = np.array([], dtype=float)
        peak_pass = 0
    else:
        pk = _detect_peaks(z, acf_peak, norm["noise_ratio"])
        peaks = np.asarray(pk["peaks"], dtype=int)
        prominences = np.asarray(pk["props"].get("prominences", []), dtype=float)
        if len(peaks):
            w = peak_widths(pk["smoothed"], peaks, rel_height=0.5)[0]
            widths = np.asarray(w, dtype=float) / FS
        else:
            widths = np.array([], dtype=float)
        peak_pass = int(pk["pass"])

    if len(peaks) >= 2:
        intervals = np.diff(peaks) / FS
        interval_mean = float(np.mean(intervals))
        interval_std = float(np.std(intervals))
        rhythm_conf = (max(0.0, 1.0 - interval_std / interval_mean)
                       if interval_mean > 0 else 0.0)
    else:
        interval_mean = 0.0
        interval_std = 0.0
        rhythm_conf = 0.0

    finite_x = norm["x"][np.isfinite(norm["x"])]
    if len(finite_x) and not a_invalid:
        rel_range = float((np.percentile(finite_x, 99) -
                           np.percentile(finite_x, 1)) / A)
        z_std = float(np.std(z))
        if np.isfinite(z_std) and z_std > EPS:
            skew = float(np.mean((z - np.mean(z)) ** 3) / (z_std ** 3))
        else:
            skew = 0.0
    else:
        rel_range = 0.0
        skew = 0.0

    feats = {
        "rel_mean": float(np.mean(z)) if not a_invalid else 0.0,
        "rel_std": float(np.std(z)) if not a_invalid else 0.0,
        "rel_range": rel_range,
        "skewness": skew,
        "peak_interval_mean": interval_mean,
        "peak_interval_std": interval_std,
        "peak_width_mean": float(np.mean(widths)) if len(widths) else 0.0,
        "plateau_ratio": float(np.mean(z >= 0.85)) if not a_invalid else 0.0,
        "peak_count": float(len(peaks)),
        "mean_prominence": float(np.mean(prominences)) if len(prominences) else 0.0,
        "peak_rate": float(len(peaks) * 60.0 / window_sec) if window_sec > 0 else 0.0,
        "rhythm_confidence": float(rhythm_conf),
        "band_ratio": float(band_ratio),
        "acf_peak": float(acf_peak),
        "flat_run_frac": float(np.clip(flat_run_sec / window_sec, 0.0, 1.0))
        if window_sec > 0 else 1.0,
    }
    nan_mask = ~np.isfinite(window)
    nan_frac = float(nan_mask.mean()) if len(window) else 1.0
    max_nan_run_sec = _run_length(nan_mask) / FS
    return {
        "features": feats,
        "vector": [float(feats[name]) for name in FEATURE_NAMES],
        "B": norm["B"],
        "A": norm["A"],
        "A_invalid": a_invalid,
        "finite_count": norm["finite_count"],
        "finite_span": norm["finite_span"],
        "constant_flat": norm["constant_flat"],
        "quantile_artifact": norm["quantile_artifact"],
        "nan_frac": nan_frac,
        "max_nan_run_sec": float(max_nan_run_sec),
        "noise": norm["noise"],
        "noise_ratio": norm["noise_ratio"],
        "noise_hf": norm["noise_hf"],
        "noise_ratio_hf": norm["noise_ratio_hf"],
        "zero_diff_frac": norm["zero_diff_frac"],
        "flat_run_sec": float(flat_run_sec),
        "flat_candidate": bool(flat_run_sec >=
                               SPEC["quality"]["min_flat_sec_for_candidate"]),
        "polarity_invalid": bool(polarity_invalid),
        "peak_pass": peak_pass,
        "peaks": peaks.tolist(),
        "window_sec": float(window_sec),
    }


def feature_vector(window: np.ndarray) -> np.ndarray:
    return np.asarray(extract_si_features(window)["vector"], dtype=np.float32)
