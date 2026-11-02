# -*- coding: utf-8 -*-
"""Stage 3 event-engine evaluation against the frozen Stage 2 truth/scorable domain."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from training.stage2_data import decode_states  # noqa: E402
from training.stage2_eval import (  # noqa: E402
    aggregate_events, match_active_causal, match_suspect_causal,
)
from training.stage3_event_engine import EventEngine, EventParams  # noqa: E402

EPS = 1e-9


def _truth(meta_case: dict, cid: int) -> list[dict]:
    return [{"case_id": int(cid), "class": "U2",
             "start": float(e["start"]), "end": float(e["end"])}
            for e in meta_case.get("reliable_u2_events", [])]


def _clip(start: float, end: float, decision_time: float,
          scorable: list[tuple[float, float]]) -> tuple[list[dict], int, float]:
    segs = []
    for sid, (a, b) in enumerate(scorable):
        s = max(float(start), float(a))
        e = min(float(end), float(b))
        if e > s + EPS:
            segs.append({"start": s, "end": e, "decision_time": float(decision_time),
                         "scorable_id": sid})
    unscorable = 1 if not segs else 0
    duration = sum(s["end"] - s["start"] for s in segs)
    return segs, unscorable, duration


def run_engine_case(data: dict, meta: dict, cid: int, rows: np.ndarray,
                    p_cal: np.ndarray, params: EventParams,
                    scale: str, window_sec: float,
                    step_sec: float) -> EventEngine:
    engine = EventEngine(params, step_sec, scale)
    cm = meta["case_meta"][str(int(cid))]
    if len(rows):
        order = rows[np.argsort(data["start"][rows])]
        states = decode_states(data["state"][order])
        p = p_cal[order, 2] if p_cal.ndim == 2 else p_cal[order]
        for i, r in enumerate(order):
            t = float(data["start"][r]) / 100.0 + float(window_sec)
            state = str(states[i])
            prob = float(p[i]) if state == "VALID" else None
            engine.update(t, state, prob)
        engine.finish(float(cm["total_sec"]))
    else:
        engine.finish(float(cm["total_sec"]))
    return engine


def evaluate_engine_case(data: dict, meta: dict, cid: int,
                         rows: np.ndarray, p_cal: np.ndarray,
                         params: EventParams, scale: str,
                         window_sec: float, step_sec: float) -> dict:
    engine = run_engine_case(data, meta, cid, rows, p_cal, params, scale,
                             window_sec, step_sec)
    return _engine_metrics(engine, meta, cid, len(rows))


def evaluate_frames_case(frames: list[tuple[float, str, float | None]],
                         meta: dict, cid: int, params: EventParams,
                         scale: str, step_sec: float) -> dict:
    """Evaluate an already-causal decision stream with the Stage 3 engine."""
    engine = EventEngine(params, step_sec, scale)
    cm = meta["case_meta"][str(int(cid))]
    for t, state, p_u2 in frames:
        engine.update(float(t), str(state), p_u2)
    engine.finish(float(cm["total_sec"]))
    return _engine_metrics(engine, meta, cid, len(frames))


def _engine_metrics(engine: EventEngine, meta: dict, cid: int,
                    n_windows: int) -> dict:
    cm = meta["case_meta"][str(int(cid))]
    scorable = [(float(a), float(b)) for a, b in cm["scorable_intervals"]]
    truth = _truth(cm, cid)
    active_alarms = []
    unscorable_active = 0
    unscorable_active_duration = 0.0
    active_duration = 0.0
    for ev in engine.active_intervals():
        segs, us, dur = _clip(ev["start"], ev["end"], ev["decision_time"],
                              scorable)
        if us:
            unscorable_active += 1
            unscorable_active_duration += ev["end"] - ev["start"]
        active_duration += dur
        for s in segs:
            active_alarms.append({**ev, **s, "stream": "ACTIVE"})
    suspect_alarms = []
    unscorable_suspect = 0
    suspect_duration = 0.0
    for ev in engine.suspect_intervals():
        segs, us, dur = _clip(ev["start"], ev["end"], ev["decision_time"],
                              scorable)
        if us:
            unscorable_suspect += 1
        suspect_duration += dur
        for s in segs:
            suspect_alarms.append({**ev, **s, "stream": "SUSPECT"})
    ra = match_active_causal(truth, active_alarms)
    rs = match_suspect_causal(truth, suspect_alarms)
    return {
        "case_id": int(cid),
        "n_windows": int(n_windows),
        "n_truth": len(truth),
        "n_low_support_truth": len(cm.get("low_support_u2_events", [])),
        "n_truth_no_windows": int(len(truth) if n_windows == 0 else 0),
        "raw_alarm_episodes": int(len(engine.episodes)),
        "engine_episodes": int(len(engine.episodes)),
        "n_alarm_segments": len(active_alarms),
        "unscorable_alarm_episodes": int(unscorable_active),
        "unscorable_alarm_duration": float(unscorable_active_duration),
        "alarm_duration": float(active_duration),
        "tp": int(ra["tp"]), "fn": int(ra["fn"]), "fp": int(ra["fp"]),
        "ttds": [float(x) for x in ra["ttd_s"]],
        "n_suspect_segments": len(suspect_alarms),
        "suspect_unscorable_episodes": int(unscorable_suspect),
        "suspect_duration": float(suspect_duration),
        "suspect_pre": int(len(rs["pre_onset_matches"])),
        "suspect_post": int(len(rs["post_onset_matches"])),
        "suspect_any": int(len(rs["matched_truth"])),
        "suspect_unmatched": int(len(rs["unmatched_suspects"])),
        "suspect_leads": [float(x) for x in rs["leads_s"]],
        "suspect_ttds": [float(x) for x in rs["ttds_s"]],
        "unknown_decisions": int(engine.unknown_decisions),
        "total_decisions": int(engine.total_decisions),
        "interruptions": int(engine.interruptions),
        "gap_interruptions": int(engine.gap_interruptions),
        "episodes_ended_unknown": int(sum(
            1 for ep in engine.episodes if ep.get("end_reason") == "UNKNOWN")),
        "low_confidence_active_episodes": int(sum(
            1 for ep in engine.episodes
            if ep.get("active_intervals") and ep.get("low_confidence"))),
        "scorable_sec": float(cm.get("scorable_sec", 0.0)),
        "total_sec": float(cm.get("total_sec", 0.0)),
        "excluded_sec": float(cm.get("excluded_sec", 0.0)),
        "reference_unknown_excluded_sec": float(
            cm.get("reference_unknown_excluded_sec", 0.0)),
        "low_support_excluded_sec": float(
            cm.get("low_support_excluded_sec", 0.0)),
        "low_support_unknown_overlap_sec": float(
            cm.get("low_support_unknown_overlap_sec", 0.0)),
    }


def aggregate_stage3(per_case: dict, case_ids: list[int]) -> dict:
    base = aggregate_events(per_case, case_ids)
    rows = [per_case[int(c)] for c in case_ids if int(c) in per_case]
    total_dec = sum(r["total_decisions"] for r in rows)
    unknown_dec = sum(r["unknown_decisions"] for r in rows)
    base["engine"] = {
        "episodes": int(sum(r["engine_episodes"] for r in rows)),
        "episodes_ended_unknown": int(sum(
            r["episodes_ended_unknown"] for r in rows)),
        "low_confidence_active_episodes": int(sum(
            r["low_confidence_active_episodes"] for r in rows)),
        "unknown_decisions": int(unknown_dec),
        "total_decisions": int(total_dec),
        "unknown_decision_fraction": float(
            unknown_dec / total_dec if total_dec else 0.0),
        "interruptions": int(sum(r["interruptions"] for r in rows)),
        "gap_interruptions": int(sum(r["gap_interruptions"] for r in rows)),
    }
    return base


def evaluate_params(data: dict, meta: dict, rows: dict, p_cal: np.ndarray,
                    case_ids: list[int], params: EventParams,
                    scale: str) -> dict:
    window_sec = float(meta["window_sec"])
    step_sec = float(meta["step_sec"])
    per_case = {}
    for cid in case_ids:
        r = np.asarray(rows.get(int(cid), []), dtype=int)
        per_case[int(cid)] = evaluate_engine_case(
            data, meta, int(cid), r, p_cal, params, scale,
            window_sec, step_sec)
    return {
        "params": params.as_dict(),
        "per_case": per_case,
        "aggregate": aggregate_stage3(per_case, case_ids),
    }


def _param_sort_key(result: dict) -> tuple:
    a = result["aggregate"]
    fa = a.get("fa_per_hour_scorable")
    fa = float(fa) if fa is not None else 0.0
    p = result["params"]
    return (float(a["event_f1"]), float(a["event_recall"]), -fa,
            -float(p["active_enter_p"]), int(p["active_on_count"]),
            int(p["active_off_count"]), -float(p["suspect_p"]))


def param_grid() -> list[EventParams]:
    """Pre-registered small Stage 3 grid; all constraints enforced."""
    out = []
    for sp in (0.10, 0.15, 0.20):
        for ae in (0.30, 0.45, 0.60):
            if sp > ae:
                continue
            for ax in (0.10, 0.20):
                if ax >= ae:
                    continue
                for son in (1, 2):
                    for aon in (1, 2):
                        if son > aon:
                            continue
                        for aoff in (1, 2):
                            out.append(EventParams(
                                suspect_p=sp, active_enter_p=ae,
                                active_exit_p=ax, suspect_on_count=son,
                                suspect_off_count=1, active_on_count=aon,
                                active_off_count=aoff))
    return out


def select_params(data: dict, meta: dict, rows: dict, p_cal: np.ndarray,
                  case_ids: list[int], scale: str) -> dict:
    table = []
    best = None
    for params in param_grid():
        r = evaluate_params(data, meta, rows, p_cal, case_ids,
                            params, scale)
        row = {
            "params": params.as_dict(),
            "event_f1": r["aggregate"]["event_f1"],
            "event_recall": r["aggregate"]["event_recall"],
            "event_ppv": r["aggregate"]["event_ppv"],
            "fa_per_hour_scorable": r["aggregate"]["fa_per_hour_scorable"],
        }
        table.append(row)
        key = _param_sort_key(r)
        if best is None or key > best[0]:
            best = (key, r)
    table.sort(key=lambda x: (x["event_f1"], x["event_recall"],
                              -(x["fa_per_hour_scorable"] or 0.0)),
               reverse=True)
    return {
        "best": best[1],
        "grid_size": len(table),
        "top": table[:20],
    }
