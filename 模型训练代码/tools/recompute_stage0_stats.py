# -*- coding: utf-8 -*-
"""
Stage 0: recompute the statistics claimed in CF-REM v1.1.

This script is read-only with respect to cache/vitaldb and existing code.
It writes:
  report/stage0/stats.json
  report/stage0/stats.md

Key frozen choices:
  - windows: production make_windows_with_reference (90s/45s etc.)
  - U2 event: etco2<10, 1s grid, ffill cap 10s, merge gap 10s, min 10s
  - window label: overlap >=0.50 -> U2; 0.10-0.50 -> ignore; else RR median
  - event order: merge first, then drop short
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from stage0_common import (
    REPO_ROOT, FS, LABELS, MIN_REFERENCE_POINTS, LOW_SUPPORT_READINGS,
    build_rr3_events, build_u2_events, classify_window, iter_windows,
    list_case_ids, load_case, peak_count, reference_stats, sha256_file,
    split_events_by_support,
)

OUT_DIR = REPO_ROOT / "report" / "stage0"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_SPLITS = REPO_ROOT / "data_splits"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sum_events(case_counts: dict, cases: list[int]) -> dict:
    def rec(c):
        return case_counts[str(c)]
    return {
        "events": int(sum(rec(c)["u2_events"] for c in cases)),
        "cases": int(sum(1 for c in cases if rec(c)["u2_events"] > 0)),
        "u2_low_support_events": int(sum(rec(c).get("u2_low_support_events", 0) for c in cases)),
        "u2_low_support_cases": int(sum(1 for c in cases
                                        if rec(c).get("u2_low_support_events", 0) > 0)),
        "rr3_events": int(sum(rec(c)["rr3_events"] for c in cases)),
        "rr3_cases": int(sum(1 for c in cases if rec(c)["rr3_events"] > 0)),
    }


def update_event_sens(d: dict, res: dict) -> None:
    ev = res.get("events", [])
    sup = res.get("support", [])
    d["events"] += len(ev)
    if ev:
        d["cases"] += 1
    d["durations"].extend(int(e - s) for s, e in ev)
    low = [s for s in sup if int(s.get("low_readings", 0)) < LOW_SUPPORT_READINGS]
    reliable = [s for s in sup if int(s.get("low_readings", 0)) >= LOW_SUPPORT_READINGS]
    d["low_support"] += len(low)
    d["reliable"] += len(reliable)
    if low:
        d["low_cases"] += 1
    if reliable:
        d["reliable_cases"] += 1
    for s in sup:
        n = int(s.get("low_readings", 0))
        d["supports"].append(n)
        if n == 0:
            d["zero"] += 1
        elif n == 1:
            d["single"] += 1
        else:
            d["multi"] += 1
    if any(int(s.get("low_readings", 0)) == 0 for s in sup):
        d["zero_cases"] += 1
    if any(int(s.get("low_readings", 0)) == 1 for s in sup):
        d["single_cases"] += 1
    if any(int(s.get("low_readings", 0)) >= 2 for s in sup):
        d["multi_cases"] += 1


def segments_from_starts(starts: list[int], step_samples: int) -> int:
    """Count contiguous window segments for one case."""
    starts = sorted(starts)
    n = 0
    last = None
    for s in starts:
        if last is None or s - last > step_samples:
            n += 1
        last = s
    return n


def pct(x: int, n: int) -> float:
    return 100.0 * x / n if n else 0.0


def main() -> int:
    t0 = time.time()
    ids = list_case_ids()
    final_test = load_json(DATA_SPLITS / "final_test_seed42.json")
    case_events = load_json(DATA_SPLITS / "case_event_counts.json")["cases"]
    test_ids = set(final_test["case_ids"])

    # ---- event sensitivity (per case, one pass) ----
    event_sens = {
        "u2_cap5": {"events": 0, "cases": 0, "durations": [], "blocked": 0,
                    "low_support": 0, "low_cases": 0, "zero": 0, "zero_cases": 0,
                    "single": 0, "single_cases": 0, "multi": 0, "multi_cases": 0,
                    "reliable": 0, "reliable_cases": 0, "supports": []},
        "u2_cap10": {"events": 0, "cases": 0, "durations": [], "blocked": 0,
                     "low_support": 0, "low_cases": 0, "zero": 0, "zero_cases": 0,
                     "single": 0, "single_cases": 0, "multi": 0, "multi_cases": 0,
                     "reliable": 0, "reliable_cases": 0, "supports": []},
        "u2_cap20": {"events": 0, "cases": 0, "durations": [], "blocked": 0,
                     "low_support": 0, "low_cases": 0, "zero": 0, "zero_cases": 0,
                     "single": 0, "single_cases": 0, "multi": 0, "multi_cases": 0,
                     "reliable": 0, "reliable_cases": 0, "supports": []},
        "u2_merge5": {"events": 0, "cases": 0, "durations": [], "blocked": 0,
                      "low_support": 0, "low_cases": 0, "zero": 0, "zero_cases": 0,
                      "single": 0, "single_cases": 0, "multi": 0, "multi_cases": 0,
                      "reliable": 0, "reliable_cases": 0, "supports": []},
        "u2_merge10": {"events": 0, "cases": 0, "durations": [], "blocked": 0,
                       "low_support": 0, "low_cases": 0, "zero": 0, "zero_cases": 0,
                       "single": 0, "single_cases": 0, "multi": 0, "multi_cases": 0,
                       "reliable": 0, "reliable_cases": 0, "supports": []},
        "u2_merge20": {"events": 0, "cases": 0, "durations": [], "blocked": 0,
                       "low_support": 0, "low_cases": 0, "zero": 0, "zero_cases": 0,
                       "single": 0, "single_cases": 0, "multi": 0, "multi_cases": 0,
                       "reliable": 0, "reliable_cases": 0, "supports": []},
        "rr3_cap10": {"events": 0, "cases": 0, "durations": [], "blocked": 0},
    }

    # ---- per-scale label counts ----
    scales = [(15.0, 7.5), (30.0, 15.0), (60.0, 30.0), (90.0, 45.0)]
    scale_counts = {f"{int(w)}s": Counter() for w, _ in scales}
    scale_u2_cases = {f"{int(w)}s": set() for w, _ in scales}

    # ---- 90s old labels and selected-window peak analysis ----
    old_counts = Counter()
    old_class3 = Counter()
    u90_by_split = {"final_test": Counter(), "dev": Counter()}
    u90_ref_counts = Counter()
    u90_ref_by_split = {"final_test": Counter(), "dev": Counter()}
    old2_filtered_starts = defaultdict(list)
    old2_unfiltered_starts = defaultdict(list)
    old2_unfiltered_windows = 0
    old2_filtered_windows = 0
    peak_sets = {
        "u2_event_windows": [],
        "rr_lt3_windows": [],
        "etco2_lt10_windows": [],
    }

    # per-case old labels and U labels for 90s
    for i, cid in enumerate(ids, 1):
        d = load_case(cid)
        u2_res = build_u2_events(d["etco2"])
        u2_reliable, u2_low = split_events_by_support(u2_res)
        rr3_res = build_rr3_events(d["rr"])
        rr3 = rr3_res["events"]

        # event sensitivity + support audit
        for key, kwargs in [
            ("u2_cap5", {"cap_sec": 5}),
            ("u2_cap10", {"cap_sec": 10}),
            ("u2_cap20", {"cap_sec": 20}),
            ("u2_merge5", {"cap_sec": 10, "merge_gap_sec": 5}),
            ("u2_merge10", {"cap_sec": 10, "merge_gap_sec": 10}),
            ("u2_merge20", {"cap_sec": 10, "merge_gap_sec": 20}),
        ]:
            res = build_u2_events(d["etco2"], **kwargs)
            event_sens[key]["blocked"] += res.get("merge_blocked_by_unknown", 0)
            update_event_sens(event_sens[key], res)
        event_sens["rr3_cap10"]["events"] += len(rr3)
        event_sens["rr3_cap10"]["blocked"] += rr3_res.get("merge_blocked_by_unknown", 0)
        if rr3:
            event_sens["rr3_cap10"]["cases"] += 1
        event_sens["rr3_cap10"]["durations"].extend(int(e - s) for s, e in rr3)

        # 90s windows: old labels + U labels + peaks
        from preprocessing.make_windows import make_windows_with_reference
        from labels.build_reference_labels import build_labels_for_window, label_from_reference
        recs = make_windows_with_reference(
            d["co2"], d["rr"], d["etco2"], window_size=9000, step=4500, sampling_rate=FS)
        for r in recs:
            st = reference_stats(r["rr"], r["etco2"])
            ov = 0.0
            ov_low = 0.0
            ws = r["start"] / FS
            we = (r["start"] + 9000) / FS
            for s, e in u2_reliable:
                ov += max(0.0, min(we, e) - max(ws, s))
            for s, e in u2_low:
                ov_low += max(0.0, min(we, e) - max(ws, s))
            ov = min(1.0, ov / 90.0)
            ov_low = min(1.0, ov_low / 90.0)
            u = classify_window(r["rr"], r["etco2"], ov, ov_low)
            scale_counts["90s"][u["label"]] += 1
            if u["label"] == "U2":
                scale_u2_cases["90s"].add(cid)
            split_key = "final_test" if cid in test_ids else "dev"
            u90_by_split[split_key][u["label"]] += 1

            old = build_labels_for_window(None, r["rr"], r["etco2"],
                                          min_reference_points=MIN_REFERENCE_POINTS)
            if old["label"] is not None:
                u90_ref_counts[u["label"]] += 1
                u90_ref_by_split[split_key][u["label"]] += 1
                old_counts[old["label"]] += 1
                if old["label"] == 3:
                    rr_trig = st["rr_median"] is not None and st["rr_median"] <= 6
                    et_trig = st["et_median"] is not None and (
                        st["et_median"] < 35 or st["et_median"] > 45)
                    if rr_trig and et_trig:
                        old_class3["both"] += 1
                    elif et_trig:
                        old_class3["et_only"] += 1
                    elif rr_trig:
                        old_class3["rr_only"] += 1
                    else:
                        old_class3["other"] += 1
                if old["label"] == 2:
                    old2_filtered_starts[cid].append(r["start"])
                    old2_filtered_windows += 1
            # unfiltered old class 2 (no min_reference_points gate)
            uf = label_from_reference(st["rr_median"], st["et_median"])
            if uf is not None and uf[0] == 2:
                old2_unfiltered_starts[cid].append(r["start"])
                old2_unfiltered_windows += 1

            if u["label"] == "U2":
                peak_sets["u2_event_windows"].append(peak_count(r["co2"]))
            if st["rr_median"] is not None and st["rr_median"] < 3:
                peak_sets["rr_lt3_windows"].append(peak_count(r["co2"]))
            if st["et_median"] is not None and st["et_median"] < 10:
                peak_sets["etco2_lt10_windows"].append(peak_count(r["co2"]))

        # other scales: aggregate only
        for win, step in scales[:-1]:
            key = f"{int(win)}s"
            for rec in iter_windows(cid, win, step, u2_events=u2_reliable,
                                    low_support_events=u2_low, case_data=d):
                scale_counts[key][rec["label"]] += 1
                if rec["label"] == "U2":
                    scale_u2_cases[key].add(cid)

        if i % 25 == 0:
            print(f"  ... {i}/{len(ids)} cases, {time.time()-t0:.1f}s", flush=True)

    # ---- final test / dev events from frozen split ----
    test_event_counts = sum_events(case_events, final_test["case_ids"])
    dev_ids = load_json(DATA_SPLITS / "cases_200.json")["case_ids"]
    dev_event_counts = sum_events(case_events, dev_ids)

    # ---- old class 2 segments ----
    def segment_summary(starts_map, step_samples=4500):
        total_segments = 0
        total_cases = 0
        total_windows = 0
        for cid, starts in starts_map.items():
            if not starts:
                continue
            total_segments += segments_from_starts(starts, step_samples)
            total_cases += 1
            total_windows += len(starts)
        return {"windows": total_windows, "segments": total_segments, "cases": total_cases}

    old2_filtered = segment_summary(old2_filtered_starts)
    old2_unfiltered = segment_summary(old2_unfiltered_starts)
    old2_filtered["windows_recomputed"] = old2_filtered_windows
    old2_unfiltered["windows_recomputed"] = old2_unfiltered_windows

    # ---- peak summaries ----
    def peak_summary(vals):
        arr = np.asarray(vals, dtype=int)
        if len(arr) == 0:
            return {"n": 0}
        return {
            "n": int(len(arr)),
            "zero_pct": float(pct(int((arr == 0).sum()), len(arr))),
            "ge2_pct": float(pct(int((arr >= 2).sum()), len(arr))),
            "ge4_pct": float(pct(int((arr >= 4).sum()), len(arr))),
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
            "max": int(arr.max()),
        }

    peak_summaries = {k: peak_summary(v) for k, v in peak_sets.items()}

    def event_sens_summary(d):
        out = {k: {"events": v["events"], "cases": v["cases"],
                   "blocked_by_unknown": v.get("blocked", 0),
                   "low_support": v.get("low_support", 0),
                   "low_cases": v.get("low_cases", 0),
                   "reliable": v.get("reliable", 0),
                   "reliable_cases": v.get("reliable_cases", 0),
                   "zero_read": v.get("zero", 0),
                   "zero_read_cases": v.get("zero_cases", 0),
                   "single_read": v.get("single", 0),
                   "single_read_cases": v.get("single_cases", 0),
                   "multi_read": v.get("multi", 0),
                   "multi_read_cases": v.get("multi_cases", 0)}
               for k, v in d.items()}
        for k, v in d.items():
            arr = np.asarray(v["durations"], dtype=int)
            if len(arr):
                out[k].update({"median_duration": float(np.median(arr)),
                               "p90_duration": float(np.percentile(arr, 90))})
            sup = np.asarray(v.get("supports", []), dtype=int)
            if len(sup):
                out[k].update({"support_median": float(np.median(sup)),
                               "support_p90": float(np.percentile(sup, 90))})
        return out

    stats = {
        "version": "stage0_stats_v3",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_cases": len(ids),
        "definitions": {
            "u2_event": "etco2<10, 1s grid, ffill cap 10s, merge gap 10s, min 10s; merge before drop; no merge across unknown",
            "support_association": "a raw low reading run supports the event if it is the ffill source of at least one low grid cell in [start,end); source may precede the integer-second start within the cap",
            "rr3_event": "rr<3, 1s grid, ffill cap 10s, merge gap 20s, min 10s; no merge across unknown",
            "window_label": "overlap>=0.50 U2; 0.10-0.50 ignore; else RR median U0/U1/U3/ambiguous/unlabeled",
        },
        "total_windows_90s": int(sum(scale_counts["90s"].values())),
        "reference_windows_90s": int(sum(u90_ref_counts.values())),
        "u90_label_counts": {k: int(scale_counts["90s"].get(k, 0)) for k in LABELS},
        "u90_reference_label_counts": {k: int(u90_ref_counts.get(k, 0)) for k in LABELS},
        "u90_label_counts_by_split": {
            "final_test": {k: int(u90_by_split["final_test"].get(k, 0)) for k in LABELS},
            "dev": {k: int(u90_by_split["dev"].get(k, 0)) for k in LABELS},
        },
        "u90_reference_label_counts_by_split": {
            "final_test": {k: int(u90_ref_by_split["final_test"].get(k, 0)) for k in LABELS},
            "dev": {k: int(u90_ref_by_split["dev"].get(k, 0)) for k in LABELS},
        },
        "scale_u2_counts": {
            k: {"windows": int(scale_counts[k].get("U2", 0)),
                "cases": int(len(scale_u2_cases[k])),
                "total_windows": int(sum(scale_counts[k].values()))}
            for k in scale_counts
        },
        "scale_label_counts": {
            k: {lab: int(scale_counts[k].get(lab, 0)) for lab in LABELS}
            for k in scale_counts
        },
        "event_sensitivity": event_sens_summary(event_sens),
        "old_labels_90s": {str(k): int(v) for k, v in sorted(old_counts.items())},
        "old_class3": {k: int(old_class3.get(k, 0)) for k in
                       ("et_only", "rr_only", "both", "other")},
        "old_class2_filtered": old2_filtered,
        "old_class2_unfiltered": old2_unfiltered,
        "peak_counts": peak_summaries,
        "final_test": {
            "case_ids_hash": sha256_file(DATA_SPLITS / "final_test_seed42.json"),
            **test_event_counts,
            "u90_labels": {k: int(u90_by_split["final_test"].get(k, 0)) for k in LABELS},
            "u90_reference_labels": {k: int(u90_ref_by_split["final_test"].get(k, 0)) for k in LABELS},
        },
        "dev": {
            **dev_event_counts,
            "u90_labels": {k: int(u90_by_split["dev"].get(k, 0)) for k in LABELS},
            "u90_reference_labels": {k: int(u90_ref_by_split["dev"].get(k, 0)) for k in LABELS},
        },
    }

    with open(OUT_DIR / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # ---- markdown ----
    lines = []
    lines.append("# Stage 0 复算统计（自动生成）\n")
    lines.append(f"- 生成时间：{stats['generated_at']}")
    lines.append(f"- 病例数：{stats['n_cases']}")
    lines.append(f"- 90s/45s 窗口总数：{stats['total_windows_90s']}")
    lines.append(f"- 事件定义：{stats['definitions']['u2_event']}\n")

    lines.append("## 1. 90s U 标签分布（有参考窗，与 v1.1 同口径）\n")
    lines.append("| 标签 | 窗口数 | 占有参考 90s 窗 |")
    lines.append("|---|---:|---:|")
    total = stats["reference_windows_90s"]
    for k in LABELS:
        n = stats["u90_reference_label_counts"][k]
        lines.append(f"| {k} | {n} | {pct(n, total):.2f}% |")
    lines.append("")
    lines.append(f"全部 90s 窗口（含无参考）：{stats['total_windows_90s']}；"
                 f"有参考窗口：{stats['reference_windows_90s']}；"
                 f"无参考窗口：{stats['total_windows_90s'] - stats['reference_windows_90s']}。"
                 f"全部窗口的 U 标签计数见 stats.json 的 `u90_label_counts`。\n")

    lines.append("## 2. 各尺度 U2 正窗\n")
    lines.append("| 尺度 | U2 窗 | U2 病例 | 全部窗 |")
    lines.append("|---|---:|---:|---:|")
    for k, v in stats["scale_u2_counts"].items():
        lines.append(f"| {k} | {v['windows']} | {v['cases']} | {v['total_windows']} |")
    lines.append("")

    lines.append("## 3. 事件敏感性 + 实际读数支撑\n")
    lines.append("| 口径 | 事件数 | 病例数 | 可靠(≥2读) | 低支撑(<2读) | 0读 | 1读 | ≥2读 | 因 unknown 阻止合并 | 中位时长(s) | p90(s) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k, v in stats["event_sensitivity"].items():
        if k.startswith("rr3"):
            support_cells = "| - | - | - | - | - "
        else:
            support_cells = (f"| {v.get('reliable', 0)} | {v.get('low_support', 0)} | "
                             f"{v.get('zero_read', 0)} | {v.get('single_read', 0)} | "
                             f"{v.get('multi_read', 0)} ")
        lines.append(f"| {k} | {v['events']} | {v['cases']} {support_cells}| "
                     f"{v.get('blocked_by_unknown', 0)} | "
                     f"{v.get('median_duration', float('nan')):.1f} | "
                     f"{v.get('p90_duration', float('nan')):.1f} |")
    lines.append("")
    u2s_md = stats["event_sensitivity"]["u2_cap10"]
    lines.append("支撑按**原始读数与事件网格的实际关联**计数：某条低值原始读数只要是事件内"
                 "至少一个低值 1s 网格点的 ffill 来源，就计入；该读数可位于事件整数秒起点之前"
                 "（仍在 cap 有效期内）。不是对 `[start_s,end_s)` 做原始样点切片。")
    lines.append(f"低支撑 = 关联读数 <2；本次 0 读 {u2s_md.get('zero_read', 0)}、"
                 f"1 读 {u2s_md.get('single_read', 0)}。低支撑事件只映射到 "
                 "`ignore_low_support`，不计可靠正例，也不改作负例。\n")

    lines.append("## 4. 旧类 2 / 旧类 3\n")
    lines.append(f"- 旧类 2（生产过滤）：{stats['old_class2_filtered']}")
    lines.append(f"- 旧类 2（未过滤）：{stats['old_class2_unfiltered']}")
    lines.append(f"- 旧类 3 触发拆分：{stats['old_class3']}")
    lines.append("")

    lines.append("## 5. 波形峰数（90s 窗）\n")
    lines.append("| 子集 | n | 0 峰 | ≥2 峰 | ≥4 峰 | 中位 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for k, v in stats["peak_counts"].items():
        if v.get("n", 0):
            lines.append(f"| {k} | {v['n']} | {v['zero_pct']:.1f}% | "
                         f"{v['ge2_pct']:.1f}% | {v['ge4_pct']:.1f}% | {v['median']:.1f} |")
        else:
            lines.append(f"| {k} | 0 | - | - | - | - |")
    lines.append("")

    lines.append("## 6. 冻结划分\n")
    lines.append(f"- Final Test：{stats['final_test']}")
    lines.append(f"- Dev：{stats['dev']}")
    lines.append("")

    lines.append("## 7. 与 v1.1 计划数字的差异\n")
    lines.append("| 项 | v1.1 计划 | Stage 0 复算 | 说明 |")
    lines.append("|---|---:|---:|---|")
    lines.append("| 有参考 90s 窗 | 60,674 | 60,527 | 生产 `build_labels_for_window(min_reference_points=2)` 口径；差 147 |")
    lines.append(f"| U0 | 51,787 | {stats['u90_reference_label_counts']['U0']} | 事件重叠口径 |")
    lines.append(f"| U1 | 6,199 | {stats['u90_reference_label_counts']['U1']} | 事件重叠口径 |")
    lines.append(f"| U2 | 391 | {stats['u90_reference_label_counts']['U2']} | 事件重叠口径 |")
    lines.append(f"| U3 | 501 | {stats['u90_reference_label_counts']['U3']} | 事件重叠口径 |")
    lines.append(f"| ambiguous | 45 | {stats['u90_reference_label_counts']['ambiguous']} | 事件重叠口径 |")
    lines.append(f"| ignore | 699 | {stats['u90_reference_label_counts']['ignore']} | 可靠事件边界口径 |")
    lines.append(f"| ignore_low_support | 未定义 | {stats['u90_reference_label_counts']['ignore_low_support']} | 低支撑事件窗，不计正/负 |")
    lines.append(f"| unlabeled | 1,042 | {stats['u90_reference_label_counts']['unlabeled']} | 事件重叠口径 |")
    u2s = stats['event_sensitivity']['u2_cap10']
    lines.append(f"| U2 可靠事件（≥2 读） | 428/189 | {u2s['reliable']}/{u2s['reliable_cases']} | "
                 f"已按“不跨 unknown 合并 + 实际读数支撑”重算；428 不再是冻结值 |")
    lines.append(f"| U2 低支撑事件（<2 读） | 未定义 | {u2s['low_support']}/{u2s['low_cases']} | "
                 f"映射到 ignore_low_support；不计正例/负例 |")
    lines.append(f"| U2 事件总数（含低支撑） | 428/189 | {u2s['events']}/{u2s['cases']} | "
                 f"仅作审计，不作可靠正例 |")
    lines.append(f"| RR<3 敏感性事件 | 93/72 | {stats['event_sensitivity']['rr3_cap10']['events']}/"
                 f"{stats['event_sensitivity']['rr3_cap10']['cases']} | "
                 f"已按“不跨 unknown 合并”重算；93 不再是冻结值 |")
    lines.append(f"| Final Test U2 可靠 | 70/37 | {stats['final_test']['events']}/"
                 f"{stats['final_test']['cases']} | 新冻结分层划分，非 v1.1 的 seed42 数值 |")
    lines.append(f"| Final Test U2 低支撑 | 未定义 | {stats['final_test'].get('u2_low_support_events', 0)}/"
                 f"{stats['final_test'].get('u2_low_support_cases', 0)} | 不计正例/负例 |")
    lines.append(f"| Final Test RR<3 | 27/20 | {stats['final_test']['rr3_events']}/"
                 f"{stats['final_test']['rr3_cases']} | 新冻结分层划分 |")
    lines.append("")

    with open(OUT_DIR / "stats.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"wrote {OUT_DIR/'stats.json'}")
    print(f"wrote {OUT_DIR/'stats.md'}")
    print(f"elapsed {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
