# -*- coding: utf-8 -*-
"""Stage 4 causal multi-scale consensus and its evaluation.

The fusion layer deliberately does not multiply probabilities or accumulate
unbounded log-odds.  At each 7.5 s tick every scale contributes at most one
latest decision if it is still valid.  Contributions are normalized weights
with a bounded recency decay, so overlapping windows cannot add evidence more
than once per tick.  The resulting ``(time, quality_state, fused_p)`` stream is
fed into the reviewed Stage 3 ``EventEngine`` unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from features.quality_states import UNKNOWN_STATES
from training.stage2_data import decode_states
from training.stage3_event_engine import EventParams, params_from_dict as _ev_from
from training.stage3_eval import (
    aggregate_stage3, evaluate_frames_case, param_grid,
)

SCALES = ("U15", "U30", "U60", "U90")
GRID_SEC = 7.5
EPS = 1e-9

WEIGHT_PROFILES = {
    "balanced": {"U15": 0.5, "U30": 0.8, "U60": 1.0, "U90": 1.2},
    "equal": {"U15": 1.0, "U30": 1.0, "U60": 1.0, "U90": 1.0},
    "early": {"U15": 1.2, "U30": 0.8, "U60": 0.5, "U90": 0.3},
    "confirm": {"U15": 0.4, "U30": 0.8, "U60": 1.0, "U90": 1.2},
}
LOW_CONF_WEIGHT = 0.5
REFERENCE_EVENT = EventParams(
    suspect_p=0.10, active_enter_p=0.30, active_exit_p=0.10,
    suspect_on_count=1, suspect_off_count=1,
    active_on_count=2, active_off_count=2)


@dataclass(frozen=True)
class FusionParams:
    weight_profile: str = "balanced"
    tau_sec: float = 60.0
    min_active_sources: int = 1
    grid_sec: float = GRID_SEC
    mode: str = "consensus"          # consensus / logodds (ablation only)
    event: EventParams = field(default_factory=EventParams)

    def as_dict(self) -> dict:
        return {
            "weight_profile": self.weight_profile,
            "tau_sec": float(self.tau_sec),
            "min_active_sources": int(self.min_active_sources),
            "grid_sec": float(self.grid_sec),
            "mode": self.mode,
            "event": self.event.as_dict(),
        }


def fusion_params_from_dict(d: dict) -> FusionParams:
    return FusionParams(
        weight_profile=str(d.get("weight_profile", "balanced")),
        tau_sec=float(d.get("tau_sec", 60.0)),
        min_active_sources=int(d.get("min_active_sources", 1)),
        grid_sec=float(d.get("grid_sec", GRID_SEC)),
        mode=str(d.get("mode", "consensus")),
        event=_ev_from(d.get("event", {})),
    )


def build_scale_series(data: dict, meta: dict, cid: int,
                       rows: np.ndarray, p_cal: np.ndarray,
                       scale: str) -> dict:
    """Sorted causal decision series for one scale and case."""
    rows = np.asarray(rows, dtype=int)
    if len(rows) == 0:
        return {"scale": scale, "times": np.empty(0, dtype=float),
                "states": (), "p": np.empty(0, dtype=float),
                "window_sec": float(meta["window_sec"]),
                "step_sec": float(meta["step_sec"])}
    order = rows[np.argsort(data["start"][rows], kind="mergesort")]
    times = (data["start"][order].astype(float) / 100.0 +
             float(meta["window_sec"]))
    states = tuple(str(x) for x in decode_states(data["state"][order]))
    p = (np.asarray(p_cal)[order, 2] if np.asarray(p_cal).ndim == 2
         else np.asarray(p_cal)[order])
    return {
        "scale": str(scale),
        "times": np.asarray(times, dtype=float),
        "states": states,
        "p": np.asarray(p, dtype=float),
        "window_sec": float(meta["window_sec"]),
        "step_sec": float(meta["step_sec"]),
    }


def build_series_by_case(data_by_scale: dict, meta_by_scale: dict,
                         rows_by_scale: dict, p_by_scale: dict,
                         case_ids: list[int]) -> dict:
    """``{case_id: {scale: series}}`` for the requested cases."""
    out = {int(c): {} for c in case_ids}
    for scale in SCALES:
        data = data_by_scale.get(scale)
        meta = meta_by_scale.get(scale)
        rows = rows_by_scale.get(scale, {})
        p = p_by_scale.get(scale)
        if data is None or meta is None or p is None:
            continue
        for cid in case_ids:
            r = np.asarray(rows.get(int(cid), []), dtype=int)
            out[int(cid)][scale] = build_scale_series(
                data, meta, int(cid), r, p, scale)
    return out


def _decay(age: float, tau: float) -> float:
    if tau is None or tau <= 0 or math.isinf(tau):
        return 1.0
    return float(math.exp(-max(0.0, float(age)) / float(tau)))


def fuse_at_time(series_by_scale: dict, t: float,
                 params: FusionParams,
                 scales: tuple[str, ...] = SCALES
                 ) -> tuple[str, float | None]:
    """One causal fusion tick using only decisions with time <= t.

    This is the streaming form used by the host runtime; ``fuse_case`` uses
    the same contribution rule over a whole case.
    """
    if params.mode != "consensus":
        raise ValueError("fuse_at_time supports consensus mode only")
    scales = tuple(s for s in scales if s in series_by_scale)
    weights = WEIGHT_PROFILES[params.weight_profile]
    contributors = []
    for s in scales:
        ser = series_by_scale[s]
        times = ser["times"]
        if len(times) == 0:
            continue
        idx = int(np.searchsorted(times, float(t), side="right") - 1)
        if idx < 0:
            continue
        age = float(t) - float(times[idx])
        step = float(ser["step_sec"])
        if age < 0.0 or age >= step - max(1e-6, step * 1e-6):
            continue
        state = ser["states"][idx]
        if state in UNKNOWN_STATES:
            continue
        if state == "VALID":
            p = float(ser["p"][idx])
            if not np.isfinite(p):
                continue
            v, high, low = p, True, False
        elif state == "FLAT_WITH_HISTORY":
            v, high, low = 1.0, True, False
        elif state == "FLAT_EVER_HISTORY":
            v, high, low = 1.0, False, True
        else:
            continue
        w = float(weights.get(s, 0.0))
        if w <= 0.0:
            continue
        if low:
            w *= LOW_CONF_WEIGHT
        contributors.append((w * _decay(age, params.tau_sec), v, high, low))
    if not contributors:
        return "GAP", None
    high_count = sum(1 for _, _, high, _ in contributors if high)
    if high_count == 0:
        return "FLAT_EVER_HISTORY", None
    denom = sum(w for w, _, _, _ in contributors)
    score = sum(w * v for w, v, _, _ in contributors) / denom
    if high_count < params.min_active_sources:
        score = min(score, params.event.active_enter_p - 1e-6)
    return "VALID", float(min(1.0, max(0.0, score)))


def fuse_case(series_by_scale: dict, case_end: float,
              params: FusionParams, scales: tuple[str, ...] = SCALES
              ) -> list[tuple[float, str, float | None]]:
    """Return bounded causal fusion decisions for one case."""
    scales = tuple(s for s in scales if s in series_by_scale)
    if not scales:
        return []
    weights = WEIGHT_PROFILES[params.weight_profile]
    ptr = {s: -1 for s in scales}
    series = {}
    for s in scales:
        ser = series_by_scale[s]
        series[s] = ser
        if len(ser["times"]):
            ptr[s] = 0  # candidate index; advanced below
    # Precompute next index semantics: ptr is the latest index already <= t.
    ptr = {s: -1 for s in scales}
    start = min(series[s]["window_sec"] for s in scales)
    end = float(case_end)
    frames: list[tuple[float, str, float | None]] = []
    logodds = 0.0
    previous_t = None
    t = float(start)
    while t <= end + EPS:
        contributors = []
        for s in scales:
            ser = series[s]
            times = ser["times"]
            if len(times) == 0:
                continue
            idx = ptr[s]
            while idx + 1 < len(times) and times[idx + 1] <= t + EPS:
                idx += 1
            ptr[s] = idx
            if idx < 0:
                continue
            age = t - float(times[idx])
            step = float(ser["step_sec"])
            if age < 0.0 or age >= step - max(1e-6, step * 1e-6):
                continue
            state = ser["states"][idx]
            if state in UNKNOWN_STATES:
                continue
            if state == "VALID":
                p = float(ser["p"][idx])
                if not np.isfinite(p):
                    continue
                v, high, low = p, True, False
            elif state == "FLAT_WITH_HISTORY":
                v, high, low = 1.0, True, False
            elif state == "FLAT_EVER_HISTORY":
                v, high, low = 1.0, False, True
            else:
                continue
            w = float(weights.get(s, 0.0))
            if w <= 0.0:
                continue
            if low:
                w *= LOW_CONF_WEIGHT
            d = _decay(age, params.tau_sec)
            contributors.append((w * d, v, high, low))

        if not contributors:
            quality, fused = "GAP", None
            logodds = 0.0
        else:
            high_count = sum(1 for _, _, high, _ in contributors if high)
            if high_count == 0:
                quality, fused = "FLAT_EVER_HISTORY", None
            else:
                denom = sum(w for w, _, _, _ in contributors)
                if params.mode == "logodds":
                    if previous_t is not None:
                        logodds *= _decay(t - previous_t, params.tau_sec)
                    for w, v, _, _ in contributors:
                        vv = min(max(float(v), 1e-6), 1.0 - 1e-6)
                        logodds += w * math.log(vv / (1.0 - vv))
                    logodds = min(10.0, max(-10.0, logodds))
                    fused = 1.0 / (1.0 + math.exp(-logodds))
                    quality = "VALID"
                else:
                    score = sum(w * v for w, v, _, _ in contributors) / denom
                    if high_count < params.min_active_sources:
                        score = min(score,
                                    params.event.active_enter_p - 1e-6)
                    fused = float(min(1.0, max(0.0, score)))
                    quality = "VALID"
                    logodds = 0.0
        frames.append((float(t), quality, fused))
        previous_t = float(t)
        t += params.grid_sec
    return frames


def _eval_frames_by_case(frames_by_case: dict, meta: dict, case_ids: list[int],
                         params: FusionParams) -> dict:
    per_case = {}
    for cid in case_ids:
        frames = frames_by_case.get(int(cid), [])
        per_case[int(cid)] = evaluate_frames_case(
            frames, meta, int(cid), params.event, "FUSION", params.grid_sec)
    return per_case


def build_frames_by_case(series_by_case: dict, meta: dict,
                         params: FusionParams,
                         scales: tuple[str, ...] = SCALES) -> dict:
    out = {}
    for cid, series in series_by_case.items():
        cm = meta["case_meta"][str(int(cid))]
        out[int(cid)] = fuse_case(series, float(cm["total_sec"]),
                                  params, scales=scales)
    return out


def evaluate_fusion(series_by_case: dict, meta: dict, case_ids: list[int],
                    params: FusionParams,
                    scales: tuple[str, ...] = SCALES) -> dict:
    frames = build_frames_by_case(series_by_case, meta, params, scales)
    per_case = _eval_frames_by_case(frames, meta, case_ids, params)
    return {
        "params": params.as_dict(),
        "scales": list(scales),
        "per_case": per_case,
        "aggregate": aggregate_stage3(per_case, case_ids),
    }


def _key(agg: dict, params: FusionParams) -> tuple:
    fa = agg.get("fa_per_hour_scorable")
    fa = float(fa) if fa is not None else 0.0
    return (float(agg["event_f1"]), float(agg["event_recall"]), -fa,
            -float(params.event.active_enter_p),
            int(params.event.active_on_count),
            int(params.event.active_off_count),
            -float(params.event.suspect_p))


def _weighting_grid() -> list[FusionParams]:
    out = []
    for profile in ("balanced", "equal", "early", "confirm"):
        for tau in (30.0, 60.0, 120.0):
            for mas in (1, 2):
                out.append(FusionParams(
                    weight_profile=profile, tau_sec=tau,
                    min_active_sources=mas, event=REFERENCE_EVENT))
    return out


def select_fusion_params(series_by_case: dict, meta: dict,
                         case_ids: list[int],
                         event_grid: list[EventParams] | None = None
                         ) -> dict:
    """Frozen two-stage inner selection of fusion weighting and thresholds."""
    event_grid = event_grid or param_grid()
    weight_rows = []
    best = None
    for fp in _weighting_grid():
        r = evaluate_fusion(series_by_case, meta, case_ids, fp)
        row = {
            "weight_profile": fp.weight_profile,
            "tau_sec": fp.tau_sec,
            "min_active_sources": fp.min_active_sources,
            "event_f1": r["aggregate"]["event_f1"],
            "event_recall": r["aggregate"]["event_recall"],
            "event_ppv": r["aggregate"]["event_ppv"],
            "fa_per_hour_scorable": r["aggregate"]["fa_per_hour_scorable"],
        }
        weight_rows.append(row)
        k = _key(r["aggregate"], fp)
        if best is None or k > best[0]:
            best = (k, fp, r)
    best_weight = best[1]
    # Stage B: fix weighting, tune the event state machine on inner OOF.
    frames = build_frames_by_case(series_by_case, meta, best_weight)
    event_rows = []
    best_event = None
    for ev in event_grid:
        fp = FusionParams(
            weight_profile=best_weight.weight_profile,
            tau_sec=best_weight.tau_sec,
            min_active_sources=best_weight.min_active_sources,
            event=ev)
        per_case = _eval_frames_by_case(frames, meta, case_ids, fp)
        agg = aggregate_stage3(per_case, case_ids)
        event_rows.append({
            "event": ev.as_dict(),
            "event_f1": agg["event_f1"],
            "event_recall": agg["event_recall"],
            "event_ppv": agg["event_ppv"],
            "fa_per_hour_scorable": agg["fa_per_hour_scorable"],
        })
        k = _key(agg, fp)
        if best_event is None or k > best_event[0]:
            best_event = (k, fp, per_case, agg)
    _, final_params, per_case, agg = best_event
    weight_rows.sort(key=lambda x: (x["event_f1"], x["event_recall"],
                                    -(x["fa_per_hour_scorable"] or 0.0)),
                     reverse=True)
    event_rows.sort(key=lambda x: (x["event_f1"], x["event_recall"],
                                   -(x["fa_per_hour_scorable"] or 0.0)),
                    reverse=True)
    return {
        "params": final_params,
        "weighting_best": best_weight,
        "per_case": per_case,
        "aggregate": agg,
        "weighting_grid_size": len(weight_rows),
        "event_grid_size": len(event_rows),
        "weighting_top": weight_rows[:20],
        "event_top": event_rows[:20],
    }
