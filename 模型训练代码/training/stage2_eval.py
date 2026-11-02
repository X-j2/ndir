# -*- coding: utf-8 -*-
"""Stage 2 v2 causal single-scale alarm proxy and window metrics.

Causal rule: a window ``[start, end)`` can only produce a prediction at
``end``.  Alarm output intervals start at that decision time, never at
``start``.  Positive decisions are held for one update step; a negative,
UNKNOWN, missing decision or data gap does not extend an alarm.

Low-support and reference-unknown time is unscorable.  Alarms are clipped to
the scorable time domain, split at unscorable intervals, and never merged
across them.  FA/h and alarm-duration fractions use the same scorable domain.

These metrics are a single-scale raw-alarm proxy, not the Stage 3 ACTIVE
event engine.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from features.quality_states import state_is_unknown  # noqa: E402
from training.stage2_data import decode_states  # noqa: E402

EPS = 1e-9


def _u2_score(p_u2) -> np.ndarray:
    p = np.asarray(p_u2, dtype=float)
    if p.ndim == 2:
        p = p[:, 2]
    return p


def raw_decision_episodes(decision_times_s: np.ndarray, positive: np.ndarray,
                          step_sec: float, case_end: float) -> list[dict]:
    """Turn positive decisions into causal held episodes.

    The decision time is the window end.  A decision is held for one update
    step, and only consecutive positive decisions (next decision exactly at
    the current hold end) are merged.
    """
    decision_times = np.asarray(decision_times_s, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    out: list[dict] = []
    for d in np.sort(decision_times[positive]):
        d = float(d)
        hold = min(d + float(step_sec), float(case_end))
        if hold <= d + EPS:
            continue
        if out and abs(d - out[-1]["end"]) <= 1e-6:
            out[-1]["end"] = max(out[-1]["end"], hold)
        else:
            out.append({"start": d, "end": hold, "decision_time": d})
    return out


def _clip_interval(start: float, end: float,
                   scorable: list[tuple[float, float]]) -> list[tuple[float, float, int]]:
    out = []
    for sid, (a, b) in enumerate(scorable):
        s = max(float(start), float(a))
        e = min(float(end), float(b))
        if e > s + EPS:
            out.append((s, e, sid))
    return out


def scored_decision_alarms(decision_times_s: np.ndarray,
                           positive: np.ndarray,
                           step_sec: float, case_end: float,
                           scorable: list[tuple[float, float]],
                           case_id: int, stream: str) -> dict:
    """Clip per-decision alarms to scorable time, preserving decision time."""
    decision_times = np.asarray(decision_times_s, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    raw = raw_decision_episodes(decision_times, positive, step_sec, case_end)
    scored = []
    for d in np.sort(decision_times[positive]):
        d = float(d)
        hold = min(d + float(step_sec), float(case_end))
        if hold <= d + EPS:
            continue
        for s, e, sid in _clip_interval(d, hold, scorable):
            scored.append({"start": s, "end": e, "decision_time": d,
                           "scorable_id": sid, "case_id": int(case_id),
                           "stream": stream})
    # Merge only contiguous scored segments within the same scorable interval.
    scored.sort(key=lambda x: (x["scorable_id"], x["start"], x["end"]))
    merged = []
    for seg in scored:
        if (merged and seg["scorable_id"] == merged[-1]["scorable_id"] and
                seg["start"] <= merged[-1]["end"] + 1e-6):
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
        else:
            merged.append(dict(seg))
    # Count raw episodes that have no scorable intersection at all.
    unscorable = 0
    unscorable_duration = 0.0
    for r in raw:
        if not _clip_interval(r["start"], r["end"], scorable):
            unscorable += 1
            unscorable_duration += r["end"] - r["start"]
    return {
        "alarms": merged,
        "raw_episodes": len(raw),
        "unscorable_episodes": int(unscorable),
        "unscorable_duration": float(unscorable_duration),
        "scored_duration": float(sum(a["end"] - a["start"] for a in merged)),
    }


# ---- causal event matching ------------------------------------------------

def _truth_events(meta_case: dict, case_id: int) -> list[dict]:
    return [{"case_id": int(case_id), "class": "U2",
             "start": float(ev["start"]), "end": float(ev["end"])}
            for ev in meta_case.get("reliable_u2_events", [])]


def _low_support_events(meta_case: dict, case_id: int) -> list[dict]:
    return [{"case_id": int(case_id), "class": "U2",
             "start": float(ev["start"]), "end": float(ev["end"])}
            for ev in meta_case.get("low_support_u2_events", [])]


def _overlap(t: dict, a: dict) -> float:
    return max(0.0, min(float(t["end"]), float(a["end"])) -
               max(float(t["start"]), float(a["start"])))


def _iou(t: dict, a: dict) -> float:
    inter = _overlap(t, a)
    union = (float(t["end"]) - float(t["start"]) +
             float(a["end"]) - float(a["start"]) - inter)
    return inter / union if union > 0 else 0.0


def match_active_causal(truth: list[dict], alarms: list[dict],
                        min_overlap_sec: float = 1.0,
                        min_overlap_frac: float = 0.05) -> dict:
    """Frozen overlap/IoU/one-to-one rules with causal decision-time order.

    Overlap is computed on scorable alarm segments.  Candidate ordering and
    TTD use ``decision_time``, never the input window start.
    """
    candidates = []
    for ti, t in enumerate(truth):
        for ai, a in enumerate(alarms):
            inter = _overlap(t, a)
            if inter <= 0.0:
                continue
            min_dur = min(float(t["end"]) - float(t["start"]),
                          float(a["end"]) - float(a["start"]))
            if inter < min_overlap_sec or inter < min_overlap_frac * min_dur:
                continue
            dt = float(a["decision_time"])
            onset_dist = abs(dt - float(t["start"]))
            candidates.append((inter, _iou(t, a), onset_dist,
                               float(t["start"]), dt, ti, ai))
    candidates.sort(key=lambda x: (-x[0], -x[1], x[2], x[3], x[4], x[5], x[6]))
    used_t, used_a = set(), set()
    matches = []
    for inter, iou, onset_dist, _, dt, ti, ai in candidates:
        if ti in used_t or ai in used_a:
            continue
        used_t.add(ti)
        used_a.add(ai)
        t, a = truth[ti], alarms[ai]
        matches.append({
            "truth_index": ti, "alarm_index": ai,
            "overlap_s": float(inter), "iou": float(iou),
            "ttd_s": float(dt - float(t["start"])),
            "decision_time": float(dt),
            "truth": t, "alarm": a,
        })
    tp = len(matches)
    fn = len(truth) - tp
    fp = len(alarms) - tp
    recall = tp / (tp + fn) if tp + fn else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {
        "matches": matches,
        "unmatched_truth": [i for i in range(len(truth)) if i not in used_t],
        "unmatched_alarms": [i for i in range(len(alarms)) if i not in used_a],
        "tp": int(tp), "fn": int(fn), "fp": int(fp),
        "recall": float(recall), "ppv": float(ppv), "f1": float(f1),
        "ttd_s": [m["ttd_s"] for m in matches],
        "iou": [m["iou"] for m in matches],
    }


def match_suspect_causal(truth: list[dict], suspects: list[dict],
                         pre_lead_allow: float = 10.0,
                         post_onset_allow: float = 30.0) -> dict:
    """One-to-one SUSPECT matching using actual decision time."""
    used = set()
    pre_matches, post_matches = [], []
    for ti, t in enumerate(truth):
        pre, post = [], []
        for ai, a in enumerate(suspects):
            if ai in used:
                continue
            dt = float(a["decision_time"])
            lead = float(t["start"]) - dt
            ttd = dt - float(t["start"])
            if 0.0 < lead <= pre_lead_allow:
                pre.append((lead, ai, dt))
            elif (0.0 <= ttd <= post_onset_allow and
                  dt <= float(t["end"])):
                post.append((ttd, ai, dt))
        if pre:
            pre.sort(key=lambda x: (x[0], x[1]))
            lead, ai, dt = pre[0]
            used.add(ai)
            pre_matches.append({"truth_index": ti, "alarm_index": ai,
                                "lead_s": float(lead), "decision_time": dt,
                                "truth": t, "alarm": suspects[ai]})
        elif post:
            post.sort(key=lambda x: (x[0], x[1]))
            ttd, ai, dt = post[0]
            used.add(ai)
            post_matches.append({"truth_index": ti, "alarm_index": ai,
                                 "ttd_s": float(ttd), "decision_time": dt,
                                 "truth": t, "alarm": suspects[ai]})
    n_truth = len(truth)
    return {
        "pre_onset_matches": pre_matches,
        "post_onset_matches": post_matches,
        "matched_truth": pre_matches + post_matches,
        "unmatched_suspects": [i for i in range(len(suspects)) if i not in used],
        "leads_s": [m["lead_s"] for m in pre_matches],
        "ttds_s": [m["ttd_s"] for m in post_matches],
        "suspect_recall_pre": float(len(pre_matches) / n_truth if n_truth else 0.0),
        "suspect_recall_post": float(len(post_matches) / n_truth if n_truth else 0.0),
        "suspect_recall_any": float(
            (len(pre_matches) + len(post_matches)) / n_truth
            if n_truth else 0.0),
        "n_suspects": len(suspects),
    }


# ---- alarm masks ----------------------------------------------------------

def rf_positive_masks(data: dict, rows: np.ndarray, p_u2,
                      threshold: float) -> tuple[np.ndarray, np.ndarray]:
    states = decode_states(data["state"][rows])
    valid = states == "VALID"
    flat_with = states == "FLAT_WITH_HISTORY"
    flat_ever = states == "FLAT_EVER_HISTORY"
    p = _u2_score(p_u2)
    active = flat_with | (valid & (p >= float(threshold)))
    suspect = active | flat_ever
    return active, suspect


def b0_predict(data: dict, rows: np.ndarray | None = None) -> np.ndarray:
    """Vectorized Stage 1 B0 rule, identical to features.rule_baseline."""
    states = decode_states(data["state"] if rows is None else data["state"][rows])
    feats = data["features"] if rows is None else data["features"][rows]
    from features.si_features import FEATURE_NAMES
    rate = feats[:, FEATURE_NAMES.index("peak_rate")]
    pred = np.full(len(states), -1, dtype=np.int64)
    unknown = np.asarray([state_is_unknown(s) for s in states], dtype=bool)
    flat = np.isin(states, ["FLAT_WITH_HISTORY", "FLAT_EVER_HISTORY"])
    valid = states == "VALID"
    pred[flat] = 2
    pred[valid & (rate >= 18.0)] = 1
    pred[valid & (rate >= 7.0) & (rate < 18.0)] = 0
    pred[valid & (rate >= 3.0) & (rate < 7.0)] = 3
    pred[unknown] = -1
    return pred


def b0_positive_masks(b0_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    active = np.asarray(b0_pred, dtype=int) == 2
    return active, active.copy()


# ---- per-case / aggregate evaluation -------------------------------------

def _scorable(case_meta: dict) -> list[tuple[float, float]]:
    iv = case_meta.get("scorable_intervals")
    if iv is None:
        raise KeyError("case metadata missing scorable_intervals")
    return [(float(a), float(b)) for a, b in iv]


def evaluate_case(case_id: int, data: dict, meta: dict,
                  rows: np.ndarray, active_pos: np.ndarray,
                  suspect_pos: np.ndarray, step_sec: float) -> dict:
    cm = meta["case_meta"][str(int(case_id))]
    case_end = float(cm["total_sec"])
    scorable = _scorable(cm)
    truth = _truth_events(cm, case_id)
    low_truth = _low_support_events(cm, case_id)
    decisions = (data["start"][rows] / 100.0 + float(meta["window_sec"])
                 if len(rows) else np.empty(0))
    active = scored_decision_alarms(
        decisions, active_pos, step_sec, case_end, scorable,
        case_id, "ACTIVE")
    suspects = scored_decision_alarms(
        decisions, suspect_pos, step_sec, case_end, scorable,
        case_id, "SUSPECT")
    ra = match_active_causal(truth, active["alarms"])
    rs = match_suspect_causal(truth, suspects["alarms"])
    return {
        "case_id": int(case_id),
        "n_windows": int(len(rows)),
        "n_truth": len(truth),
        "n_low_support_truth": len(low_truth),
        "n_truth_no_windows": int(len(truth) if len(rows) == 0 else 0),
        "raw_alarm_episodes": int(active["raw_episodes"]),
        "n_alarm_segments": len(active["alarms"]),
        "unscorable_alarm_episodes": int(active["unscorable_episodes"]),
        "unscorable_alarm_duration": float(active["unscorable_duration"]),
        "alarm_duration": float(active["scored_duration"]),
        "tp": int(ra["tp"]), "fn": int(ra["fn"]), "fp": int(ra["fp"]),
        "ttds": [float(x) for x in ra["ttd_s"]],
        "n_suspect_segments": len(suspects["alarms"]),
        "suspect_unscorable_episodes": int(suspects["unscorable_episodes"]),
        "suspect_duration": float(suspects["scored_duration"]),
        "suspect_pre": int(len(rs["pre_onset_matches"])),
        "suspect_post": int(len(rs["post_onset_matches"])),
        "suspect_any": int(len(rs["matched_truth"])),
        "suspect_unmatched": int(len(rs["unmatched_suspects"])),
        "suspect_leads": [float(x) for x in rs["leads_s"]],
        "suspect_ttds": [float(x) for x in rs["ttds_s"]],
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


def evaluate_events_for_cases(data: dict, meta: dict,
                              rows_by_case: dict, case_ids: list[int],
                              p_u2=None, threshold: float | None = None,
                              policy: str = "rf") -> dict:
    per_case = {}
    step_sec = float(meta["step_sec"])
    if policy == "b0":
        b0_pred_all = b0_predict(data)
    else:
        b0_pred_all = None
    for cid in case_ids:
        rows = np.asarray(rows_by_case.get(int(cid), []), dtype=int)
        if len(rows) == 0:
            active = np.empty(0, dtype=bool)
            suspect = np.empty(0, dtype=bool)
        elif policy == "b0":
            active, suspect = b0_positive_masks(b0_pred_all[rows])
        else:
            if threshold is None:
                raise ValueError("RF policy requires threshold")
            active, suspect = rf_positive_masks(
                data, rows, np.asarray(p_u2)[rows], threshold)
        per_case[int(cid)] = evaluate_case(
            int(cid), data, meta, rows, active, suspect, step_sec)
    return per_case


def _aggregate_per_case(per_case: dict, case_ids: list[int]) -> dict:
    rows = [per_case[int(c)] for c in case_ids if int(c) in per_case]
    tp = sum(r["tp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    n_truth = sum(r["n_truth"] for r in rows)
    n_truth_no_windows = sum(r["n_truth_no_windows"] for r in rows)
    n_low = sum(r["n_low_support_truth"] for r in rows)
    n_alarm = sum(r["n_alarm_segments"] for r in rows)
    n_raw = sum(r["raw_alarm_episodes"] for r in rows)
    n_unscorable = sum(r["unscorable_alarm_episodes"] for r in rows)
    unscorable_duration = sum(r["unscorable_alarm_duration"] for r in rows)
    scorable_sec = sum(r["scorable_sec"] for r in rows)
    total_sec = sum(r["total_sec"] for r in rows)
    excluded_sec = sum(r["excluded_sec"] for r in rows)
    ref_unknown_sec = sum(r["reference_unknown_excluded_sec"] for r in rows)
    low_excluded_sec = sum(r["low_support_excluded_sec"] for r in rows)
    low_unknown_sec = sum(r["low_support_unknown_overlap_sec"] for r in rows)
    alarm_duration = sum(r["alarm_duration"] for r in rows)
    suspect_duration = sum(r["suspect_duration"] for r in rows)
    ttds = [x for r in rows for x in r["ttds"]]
    early = [-x for x in ttds if x < 0.0]
    pos_ttds = [x for x in ttds if x >= 0.0]
    pre = sum(r["suspect_pre"] for r in rows)
    post = sum(r["suspect_post"] for r in rows)
    sus_any = sum(r["suspect_any"] for r in rows)
    sus_unmatched = sum(r["suspect_unmatched"] for r in rows)
    sus_n = sum(r["n_suspect_segments"] for r in rows)
    sus_unscorable = sum(r["suspect_unscorable_episodes"] for r in rows)
    sus_leads = [x for r in rows for x in r["suspect_leads"]]
    sus_ttds = [x for r in rows for x in r["suspect_ttds"]]
    recall = tp / (tp + fn) if tp + fn else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    scorable_h = scorable_sec / 3600.0
    total_h = total_sec / 3600.0

    def med(xs):
        return float(np.median(xs)) if xs else None

    def p90(xs):
        return float(np.percentile(xs, 90)) if xs else None

    def rate_within(sec):
        return float(sum(1 for x in ttds if x <= sec) / n_truth) if n_truth else 0.0

    return {
        "n_cases": len(rows),
        "n_truth_events": int(n_truth),
        "n_truth_events_no_window_cases": int(n_truth_no_windows),
        "n_low_support_events": int(n_low),
        "tp": int(tp), "fn": int(fn), "fp": int(fp),
        "event_recall": float(recall),
        "event_ppv": float(ppv),
        "event_f1": float(f1),
        "raw_alarm_episodes": int(n_raw),
        "n_alarm_segments": int(n_alarm),
        "unscorable_alarm_episodes": int(n_unscorable),
        "unscorable_alarm_duration": float(unscorable_duration),
        "alarms_per_hour": float(n_alarm / scorable_h) if scorable_h else None,
        "alarm_duration_fraction": (
            float(alarm_duration / scorable_sec) if scorable_sec else None),
        "fa_per_hour_scorable": float(fp / scorable_h) if scorable_h else None,
        "fa_per_hour_total": float(fp / total_h) if total_h else None,
        "scorable_hours": float(scorable_h),
        "total_hours": float(total_h),
        "excluded_hours": float(excluded_sec / 3600.0),
        "reference_unknown_excluded_hours": float(ref_unknown_sec / 3600.0),
        "low_support_excluded_hours": float(low_excluded_sec / 3600.0),
        "low_support_unknown_overlap_hours": float(low_unknown_sec / 3600.0),
        "ttd_s": {
            "n": len(ttds),
            "median": med(ttds),
            "p90": p90(ttds),
            "median_nonnegative": med(pos_ttds),
            "p90_nonnegative": p90(pos_ttds),
            "n_early": len(early),
            "median_early_lead": med(early),
            "p90_early_lead": p90(early),
            "detect_30s": rate_within(30.0),
            "detect_60s": rate_within(60.0),
            "detect_90s": rate_within(90.0),
        },
        "suspect": {
            "n_suspect_segments": int(sus_n),
            "unscorable_suspect_episodes": int(sus_unscorable),
            "pre_onset_matches": int(pre),
            "post_onset_matches": int(post),
            "matched_truth": int(sus_any),
            "unmatched_suspects": int(sus_unmatched),
            "suspect_recall_pre": float(pre / n_truth if n_truth else 0.0),
            "suspect_recall_post": float(post / n_truth if n_truth else 0.0),
            "suspect_recall_any": float(sus_any / n_truth if n_truth else 0.0),
            "suspects_per_hour": float(sus_n / scorable_h) if scorable_h else None,
            "false_suspects_per_hour": (
                float(sus_unmatched / scorable_h) if scorable_h else None),
            "suspect_duration_fraction": (
                float(suspect_duration / scorable_sec) if scorable_sec else None),
            "median_lead": med(sus_leads),
            "median_post_ttd": med(sus_ttds),
        },
        "per_case": {str(c): per_case[int(c)] for c in case_ids
                     if int(c) in per_case},
    }


def aggregate_events(per_case: dict, case_ids: list[int]) -> dict:
    return _aggregate_per_case(per_case, case_ids)


def bootstrap_event_metrics(per_case: dict, case_ids: list[int],
                            n_boot: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    cases = np.asarray([int(c) for c in case_ids if int(c) in per_case])
    if len(cases) == 0:
        return {}
    point = _aggregate_per_case(per_case, cases.tolist())
    keys = ("event_recall", "event_ppv", "event_f1",
            "fa_per_hour_scorable", "fa_per_hour_total",
            "alarm_duration_fraction")
    vals = {k: [] for k in keys}
    ttd_med = []
    early_n = []
    for _ in range(n_boot):
        sample = rng.choice(cases, size=len(cases), replace=True)
        agg = _aggregate_per_case(per_case, sample.tolist())
        for k in keys:
            v = agg[k]
            if v is not None and np.isfinite(v):
                vals[k].append(float(v))
        if agg["ttd_s"]["median"] is not None:
            ttd_med.append(agg["ttd_s"]["median"])
        early_n.append(agg["ttd_s"]["n_early"])

    def ci(xs):
        return ([float(np.percentile(xs, 2.5)),
                 float(np.percentile(xs, 97.5))] if xs else None)

    return {
        "n_boot": int(n_boot),
        "point": point,
        "ci95": {k: ci(vals[k]) for k in keys},
        "ttd_median_ci95": ci(ttd_med),
        "early_count_ci95": ci(early_n),
    }


def select_threshold(data: dict, meta: dict, rows_by_case: dict,
                     p_u2: np.ndarray, case_ids: list[int],
                     candidates: np.ndarray | None = None) -> dict:
    if candidates is None:
        candidates = np.r_[np.arange(0.05, 0.951, 0.05), 0.98]
    best = None
    table = []
    for t in candidates:
        per_case = evaluate_events_for_cases(
            data, meta, rows_by_case, case_ids,
            p_u2=p_u2, threshold=float(t), policy="rf")
        agg = aggregate_events(per_case, case_ids)
        score = (agg["event_f1"], agg["event_recall"],
                 -agg["fp"], -abs(float(t) - 0.5))
        table.append({"threshold": float(t), "event_f1": agg["event_f1"],
                      "event_recall": agg["event_recall"], "fp": agg["fp"],
                      "alarms": agg["n_alarm_segments"],
                      "fa_per_hour_scorable": agg["fa_per_hour_scorable"]})
        if best is None or score > best[0]:
            best = (score, float(t), agg)
    return {"threshold": best[1], "objective": list(best[0]),
            "inner_metrics": best[2], "table": table}


# ---- window-level metrics -------------------------------------------------

WINDOW_PRED_CODES = (-1, 0, 1, 2, 3)  # UNKNOWN, U0, U1, U2, U3
N_WINDOW_PRED_COLS = len(WINDOW_PRED_CODES)


def window_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """3 x 5 counts: true U0/U1/U2; pred UNKNOWN/U0/U1/U2/U3.

    The U3 prediction column is kept for B0 comparisons.  True U3 is never
    included (three-class task); a B0 U3 prediction remains a non-U2,
    non-UNKNOWN prediction rather than being silently discarded.
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    out = np.zeros((3, N_WINDOW_PRED_COLS), dtype=np.int64)
    for j, p in enumerate(WINDOW_PRED_CODES):
        for c in (0, 1, 2):
            out[c, j] = int(np.sum((yt == c) & (yp == p)))
    return out


def metrics_from_window_confusion(counts: np.ndarray) -> dict:
    c = np.asarray(counts, dtype=np.int64)
    n = int(c.sum())
    covered = int(c[:, 1:].sum())
    correct = int(c[0, 1] + c[1, 2] + c[2, 3])
    per = {}
    f1s = []
    for k, name in enumerate(("0", "1", "2")):
        tp = int(c[k, k + 1])
        fp = int(c[:, k + 1].sum() - tp)
        fn = int(c[k].sum() - tp)
        support = int(c[k].sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        per[name] = {"support": support, "tp": tp, "fp": fp, "fn": fn,
                     "precision": float(prec), "recall": float(rec),
                     "f1": float(f1)}
        f1s.append(f1)
    u2_tp = int(c[2, 3])
    u2_fp = int(c[:, 3].sum() - u2_tp)
    u2_fn = int(c[2].sum() - u2_tp)
    return {
        "n": n,
        "covered": covered,
        "coverage": float(covered / n if n else 0.0),
        "accuracy_all": float(correct / n if n else 0.0),
        "accuracy_covered": float(correct / covered if covered else 0.0),
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "per_class": per,
        "u2_binary": {
            "tp": u2_tp, "fp": u2_fp, "fn": u2_fn,
            "precision": float(u2_tp / (u2_tp + u2_fp)
                               if u2_tp + u2_fp else 0.0),
            "recall": float(u2_tp / (u2_tp + u2_fn)
                            if u2_tp + u2_fn else 0.0),
            "f1": float(2 * u2_tp / (2 * u2_tp + u2_fp + u2_fn)
                        if (2 * u2_tp + u2_fp + u2_fn) else 0.0),
        },
    }


def bootstrap_window_metrics(counts_by_case: dict, n_boot: int = 1000,
                             seed: int = 42) -> dict:
    cases = list(counts_by_case)
    if not cases:
        return {}
    rng = np.random.default_rng(seed)
    arr = np.stack([np.asarray(counts_by_case[c], dtype=np.int64)
                    for c in cases])
    point = metrics_from_window_confusion(arr.sum(axis=0))
    keys = ("accuracy_all", "accuracy_covered", "macro_f1", "coverage")
    vals = {k: [] for k in keys}
    u2 = {"precision": [], "recall": [], "f1": []}
    for _ in range(n_boot):
        idx = rng.integers(0, len(cases), size=len(cases))
        m = metrics_from_window_confusion(arr[idx].sum(axis=0))
        for k in keys:
            vals[k].append(float(m[k]))
        for k in u2:
            u2[k].append(float(m["u2_binary"][k]))

    def ci(xs):
        return ([float(np.percentile(xs, 2.5)),
                 float(np.percentile(xs, 97.5))] if xs else None)

    return {
        "n_boot": int(n_boot),
        "point": point,
        "ci95": {**{k: ci(vals[k]) for k in keys},
                 **{f"u2_{k}": ci(v) for k, v in u2.items()}},
    }


def multi_class_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        labels: tuple[int, ...] = (0, 1, 2),
                        unknown: int = -1) -> dict:
    return metrics_from_window_confusion(window_confusion(y_true, y_pred))
