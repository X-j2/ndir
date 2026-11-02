# -*- coding: utf-8 -*-
"""
Frozen Stage 0 event matching.

Rules:
  - ACTIVE TP requires temporal overlap > 0. Grace windows alone never make TP.
  - Matching is one-to-one, same case and class.
  - Temporal IoU is intersection / union, not intersection / max(duration).
  - SUSPECT is a separate early-warning stream; it never contributes to
    ACTIVE TP/PPV/F1/FA_h.
"""
from __future__ import annotations

from typing import Iterable

Event = dict

# Pre-frozen Stage 0 ACTIVE validity gate. Do not tune on Final Test.
MIN_ACTIVE_OVERLAP_SEC = 1.0
MIN_ACTIVE_OVERLAP_FRAC = 0.05   # of the shorter event


def _same_case_class(t: Event, a: Event) -> bool:
    return t.get("case_id") == a.get("case_id") and t.get("class") == a.get("class")


def intersection_sec(t: Event, a: Event) -> float:
    return max(0.0, min(float(t["end"]), float(a["end"])) - max(float(t["start"]), float(a["start"])))


def union_sec(t: Event, a: Event) -> float:
    return ((float(t["end"]) - float(t["start"])) +
            (float(a["end"]) - float(a["start"])) -
            intersection_sec(t, a))


def temporal_iou(t: Event, a: Event) -> float:
    """Intersection over union. Both intervals are half-open [start, end)."""
    inter = intersection_sec(t, a)
    union = union_sec(t, a)
    return float(inter / union) if union > 0 else 0.0


def match_active(truth: list[Event], alarms: list[Event],
                 pre_allow: float = 10.0,
                 post_allow: float = 30.0,
                 min_overlap_sec: float = MIN_ACTIVE_OVERLAP_SEC,
                 min_overlap_frac: float = MIN_ACTIVE_OVERLAP_FRAC) -> dict:
    """One-to-one ACTIVE matching. No-overlap grace alarms are not TP.

    A candidate must have a *valid* overlap:
      overlap >= min_overlap_sec AND
      overlap >= min_overlap_frac * min(truth_duration, alarm_duration).
    This rejects a 0.1 s touch as TP while still allowing a 1 s alarm
    overlapping a 10 s event.
    """
    alarms = [a for a in alarms if a.get("stream", "ACTIVE") == "ACTIVE"]
    candidates = []
    for ti, t in enumerate(truth):
        for ai, a in enumerate(alarms):
            if not _same_case_class(t, a):
                continue
            inter = intersection_sec(t, a)
            if inter <= 0.0:
                # Deliberate: an ACTIVE alarm with no overlap cannot be TP,
                # even if it lies inside the pre/post grace window.
                continue
            min_dur = min(float(t["end"]) - float(t["start"]),
                          float(a["end"]) - float(a["start"]))
            if inter < min_overlap_sec or inter < min_overlap_frac * min_dur:
                # 0.1 s touches and tiny fragments are not valid ACTIVE TP.
                continue
            iou = temporal_iou(t, a)
            onset_dist = abs(float(a["start"]) - float(t["start"]))
            candidates.append((inter, iou, onset_dist, ti, ai))
    candidates.sort(key=lambda x: (-x[0], -x[1], x[2], x[3], x[4]))

    matched_t = set()
    matched_a = set()
    matches = []
    for inter, iou, onset_dist, ti, ai in candidates:
        if ti in matched_t or ai in matched_a:
            continue
        matched_t.add(ti)
        matched_a.add(ai)
        t, a = truth[ti], alarms[ai]
        matches.append({
            "truth_index": ti,
            "alarm_index": ai,
            "overlap_s": float(inter),
            "iou": float(iou),
            "ttd_s": float(float(a["start"]) - float(t["start"])),
            "truth": t,
            "alarm": a,
        })

    unmatched_truth = [i for i in range(len(truth)) if i not in matched_t]
    unmatched_alarms = [i for i in range(len(alarms)) if i not in matched_a]

    # Diagnostics only: grace-zone alarms are FP, never TP.
    late, early = [], []
    for ai in unmatched_alarms:
        a = alarms[ai]
        for t in truth:
            if not _same_case_class(t, a):
                continue
            if float(t["end"]) <= float(a["start"]) <= float(t["end"]) + post_allow:
                late.append({"alarm_index": ai, "truth": t, "alarm": a})
            if float(t["start"]) - pre_allow <= float(a["end"]) <= float(t["start"]):
                early.append({"alarm_index": ai, "truth": t, "alarm": a})

    tp = len(matches)
    fn = len(unmatched_truth)
    fp = len(unmatched_alarms)
    recalls = tp / (tp + fn) if (tp + fn) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
    return {
        "matches": matches,
        "unmatched_truth": unmatched_truth,
        "unmatched_alarms": unmatched_alarms,
        "late_alarms": late,
        "early_alarms": early,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "recall": float(recalls),
        "ppv": float(ppv),
        "f1": float(f1),
        "ttd_s": [m["ttd_s"] for m in matches],
        "iou": [m["iou"] for m in matches],
    }


def match_suspect(truth: list[Event], suspects: list[Event],
                  pre_lead_allow: float = 10.0,
                  post_onset_allow: float = 30.0) -> dict:
    """Separate evaluation for SUSPECT.

    Two distinct streams:
      - pre-onset warning: 0 < (t.onset - a.onset) <= pre_lead_allow;
        reported as lead.
      - post-onset early detection: 0 <= (a.onset - t.onset) <= post_onset_allow
        and a.onset <= t.offset; reported as actual TTD.

    A truth prefers a pre-onset suspect if available; otherwise the earliest
    post-onset suspect. One suspect matches at most one truth. Neither stream
    contributes to ACTIVE TP/PPV/F1/FA_h.
    """
    suspects = [s for s in suspects if s.get("stream", "SUSPECT") == "SUSPECT"]
    used = set()
    pre_matches = []
    post_matches = []
    for ti, t in enumerate(truth):
        pre = []
        post = []
        for ai, a in enumerate(suspects):
            if ai in used or not _same_case_class(t, a):
                continue
            lead = float(t["start"]) - float(a["start"])
            ttd = float(a["start"]) - float(t["start"])
            if 0.0 < lead <= pre_lead_allow:
                pre.append((lead, ai, a))
            elif (0.0 <= ttd <= post_onset_allow and
                  float(a["start"]) <= float(t["end"])):
                post.append((ttd, ai, a))
        if pre:
            pre.sort(key=lambda x: (x[0], x[1]))
            lead, ai, a = pre[0]
            used.add(ai)
            pre_matches.append({"truth_index": ti, "alarm_index": ai,
                                "lead_s": float(lead), "truth": t, "alarm": a})
        elif post:
            post.sort(key=lambda x: (x[0], x[1]))
            ttd, ai, a = post[0]
            used.add(ai)
            post_matches.append({"truth_index": ti, "alarm_index": ai,
                                 "ttd_s": float(ttd), "truth": t, "alarm": a})

    leads = [m["lead_s"] for m in pre_matches]
    ttds = [m["ttd_s"] for m in post_matches]
    matched_truth = pre_matches + post_matches
    unmatched_suspects = [i for i in range(len(suspects)) if i not in used]
    n_truth = len(truth)
    return {
        "pre_onset_matches": pre_matches,
        "post_onset_matches": post_matches,
        "matched_truth": matched_truth,
        "unmatched_suspects": unmatched_suspects,
        "leads_s": leads,
        "ttds_s": ttds,
        "suspect_recall_pre": float(len(pre_matches) / n_truth) if n_truth else 0.0,
        "suspect_recall_post": float(len(post_matches) / n_truth) if n_truth else 0.0,
        "suspect_recall_any": float(len(matched_truth) / n_truth) if n_truth else 0.0,
        # Backward-compatible alias: any early/early-detection match.
        "suspect_recall": float(len(matched_truth) / n_truth) if n_truth else 0.0,
        "median_lead_s": float(sorted(leads)[len(leads) // 2]) if leads else None,
        "median_ttd_s": float(sorted(ttds)[len(ttds) // 2]) if ttds else None,
        "n_suspects": len(suspects),
        "n_matched_suspects": len(used),
    }
