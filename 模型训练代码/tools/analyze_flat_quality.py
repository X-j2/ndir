# -*- coding: utf-8 -*-
"""
Stage 0: distinguish flat-waveform apnea candidates from disconnect/saturation.

Causal only: a window is classified from the current window plus a summary of
the PAST 300 s. No future/case-level data is used. This directly addresses the
review requirement to delete case_has_breathing.

Outputs:
  report/stage0/quality_analysis.json
  report/stage0/quality_analysis.md
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from stage0_common import (
    FS, REPO_ROOT, build_u2_events, iter_windows, list_case_ids,
    load_case, quality_metrics, split_events_by_support,
)
from quality_state import (
    A_FLOOR_DEFAULT, HISTORY_LOOKBACK_SEC, MIN_HISTORY_SEC,
    breathing_bins, classify_quality_state, history_from_bins,
)

OUT_DIR = REPO_ROOT / "report" / "stage0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

A_MIN_PROVISIONAL = A_FLOOR_DEFAULT
NON_U2_SAMPLE = 2000
FLAT_STATES = {"FLAT_WITH_HISTORY", "FLAT_EVER_HISTORY",
               "FLAT_NO_HISTORY", "FLAT_HISTORY_UNKNOWN"}


def pct(n, d):
    return 100.0 * n / d if d else 0.0


def analyze_window(d: dict, rec: dict, window_sec: float,
                   bins: list[dict]) -> dict:
    q = quality_metrics(rec["co2"])
    s = rec["start"]
    history = history_from_bins(bins, s / FS, recent_sec=HISTORY_LOOKBACK_SEC,
                                min_history_sec=MIN_HISTORY_SEC)
    state = classify_quality_state(
        rec["co2"], history, a_floor=A_MIN_PROVISIONAL,
        saturated=False, polarity_invalid=False, noise_ok=True)
    flat_run = q["flat_run_sec"]
    if flat_run is None and q["A_invalid"]:
        flat_run = window_sec
    flat = state in FLAT_STATES
    saturated_proxy = bool(q["max_value_frac"] > 0.80 and q["max"] >= 40.0)
    return {
        "case_id": rec["case_id"],
        "start": int(s),
        "window_sec": float(window_sec),
        "state": state,
        "A": q["A"],
        "B": q["B"],
        "noise": q["noise"],
        "noise_ratio": q["noise_ratio"],
        "noise_alt": q["noise_alt"],
        "noise_ratio_alt": q["noise_ratio_alt"],
        "noise_hf": q["noise_hf"],
        "noise_ratio_hf": q["noise_ratio_hf"],
        "zero_diff_frac": q["zero_diff_frac"],
        "nan_frac": q["nan_frac"],
        "zero_frac": q["zero_frac"],
        "max_nan_run_sec": q["max_nan_run_sec"],
        "flat_run_sec": flat_run,
        "A_invalid": q["A_invalid"],
        "max_value_frac": q["max_value_frac"],
        "saturated_proxy": saturated_proxy,
        "history_available": bool(history.available),
        "history_has_breathing": history.has_breathing,
        "history_ever_has_breathing": history.ever_has_breathing,
        "history_lookback_sec": float(history.lookback_sec),
        "history_A": history.A,
        "history_peaks": history.peaks,
        "flat": bool(flat),
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "state_counts": {}, "history_counts": {}}
    A = np.array([r["A"] for r in rows], dtype=float)
    nr = np.array([r["noise_ratio"] for r in rows], dtype=float)
    nrh = np.array([r["noise_ratio_hf"] for r in rows], dtype=float)
    nra = np.array([r["noise_ratio_alt"] for r in rows], dtype=float)
    zdf = np.array([r["zero_diff_frac"] for r in rows], dtype=float)
    nan = np.array([r["nan_frac"] for r in rows], dtype=float)
    zero = np.array([r["zero_frac"] for r in rows], dtype=float)
    flat = np.array([r["flat_run_sec"] for r in rows if r["flat_run_sec"] is not None], dtype=float)
    states = Counter(r["state"] for r in rows)
    histories = Counter(
        "True" if r["history_has_breathing"] is True else
        "False" if r["history_has_breathing"] is False else
        "None" for r in rows)
    n = len(rows)
    return {
        "n": n,
        "state_counts": dict(states),
        "history_counts": dict(histories),
        "A": {f"p{q}": float(np.percentile(A, q)) for q in (0, 10, 25, 50, 75, 90, 100)},
        "noise_ratio": {f"p{q}": float(np.percentile(nr, q)) for q in (10, 25, 50, 75, 90)},
        "noise_ratio_alt": {f"p{q}": float(np.percentile(nra, q)) for q in (10, 25, 50, 75, 90)},
        "noise_ratio_hf": {f"p{q}": float(np.percentile(nrh, q)) for q in (10, 25, 50, 75, 90)},
        "zero_diff_frac_mean": float(zdf.mean()),
        "nan_frac_mean": float(nan.mean()),
        "zero_frac_mean": float(zero.mean()),
        "flat_run_sec": {f"p{q}": float(np.percentile(flat, q)) for q in (10, 25, 50, 75, 90)} if len(flat) else {},
        "A_invalid": int(sum(r["A_invalid"] for r in rows)),
        "A_lt_0.5": int(sum(r["A"] < 0.5 for r in rows)),
        "A_lt_1": int(sum(r["A"] < 1.0 for r in rows)),
        "A_lt_2": int(sum(r["A"] < 2.0 for r in rows)),
        "A_lt_3": int(sum(r["A"] < 3.0 for r in rows)),
        "flat_candidate": int(sum(r["flat"] for r in rows)),
        "flat_with_history": int(states.get("FLAT_WITH_HISTORY", 0)),
        "flat_ever_history": int(states.get("FLAT_EVER_HISTORY", 0)),
        "flat_no_history": int(states.get("FLAT_NO_HISTORY", 0)),
        "flat_history_unknown": int(states.get("FLAT_HISTORY_UNKNOWN", 0)),
        "saturated_proxy": int(sum(r["saturated_proxy"] for r in rows)),
        "v11_abstain": int(sum(r["A_invalid"] for r in rows)),
        "v11_abstain_pct": pct(int(sum(r["A_invalid"] for r in rows)), n),
    }


def main() -> int:
    t0 = time.time()
    ids = list_case_ids()
    rows = {"u2_90": [], "u2_15": [], "non_u2_90": []}
    non_u2_seen = 0
    for i, cid in enumerate(ids, 1):
        d = load_case(cid)
        bins = breathing_bins(d["co2"])
        u2_res = build_u2_events(d["etco2"])
        u2_reliable, u2_low = split_events_by_support(u2_res)
        for win, step, key in [(90.0, 45.0, "u2_90"), (15.0, 7.5, "u2_15")]:
            for rec in iter_windows(cid, win, step, u2_events=u2_reliable,
                                    low_support_events=u2_low,
                                    include_co2=True, case_data=d):
                if rec["label"] != "U2":
                    continue
                rows[key].append(analyze_window(d, rec, win, bins))
        # deterministic non-U2 sample from 90s windows
        for rec in iter_windows(cid, 90.0, 45.0, u2_events=u2_reliable,
                                low_support_events=u2_low,
                                include_co2=True, case_data=d):
            if rec["label"] == "U2":
                continue
            if non_u2_seen % 30 == 0 and len(rows["non_u2_90"]) < NON_U2_SAMPLE:
                rows["non_u2_90"].append(analyze_window(d, rec, 90.0, bins))
            non_u2_seen += 1
        if i % 25 == 0:
            print(f"  ... {i}/{len(ids)} cases, {time.time()-t0:.1f}s", flush=True)

    out = {
        "version": "stage0_quality_analysis_rev3",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "causality": "only current window + prior <=300s; no future/case-level data",
        "provisional": {
            "A_min": A_MIN_PROVISIONAL,
            "history_lookback_sec": HISTORY_LOOKBACK_SEC,
            "min_history_sec": MIN_HISTORY_SEC,
            "note": "Thresholds are descriptive for Stage 0; Stage 1 must tune them on dev cases only.",
        },
        "summaries": {k: summarize(v) for k, v in rows.items()},
        "rows": rows,
    }
    with open(OUT_DIR / "quality_analysis.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append("# Stage 0 平坦波形/质量分析（rev3，causal）\n")
    lines.append(f"- 生成时间：{out['generated_at']}")
    lines.append(f"- 临时阈值：A_min={A_MIN_PROVISIONAL} mmHg；历史回看 {HISTORY_LOOKBACK_SEC:.0f}s；"
                 f"最短历史 {MIN_HISTORY_SEC:.0f}s")
    lines.append("- **因果性**：每个窗只用当前窗 + 过去 ≤300s 的波形；不使用完整病例/未来数据。")
    lines.append("- 这些阈值仅用于 Stage 0 描述，Stage 1 必须在开发集上冻结，不能用 Final Test 调。\n")

    lines.append("## 1. 状态汇总\n")
    lines.append("| 子集 | n | A<1e-6 | A<1 | flat候选 | FLAT_WITH_HISTORY | FLAT_EVER_HISTORY | FLAT_NO_HISTORY | FLAT_HISTORY_UNKNOWN | GAP | VALID | 饱和代理 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k, s in out["summaries"].items():
        if not s.get("n"):
            lines.append(f"| {k} | 0 | - | - | - | - | - | - | - | - | - | - |")
            continue
        st = s["state_counts"]
        lines.append(
            f"| {k} | {s['n']} | {s['A_invalid']} | {s['A_lt_1']} | {s['flat_candidate']} | "
            f"{s['flat_with_history']} | {s['flat_ever_history']} | {s['flat_no_history']} | "
            f"{s['flat_history_unknown']} | {st.get('GAP', 0)} | {st.get('VALID', 0)} | "
            f"{s['saturated_proxy']} |")
    lines.append("")

    lines.append("## 2. 平坦窗按“已验证历史呼吸”拆分\n")
    lines.append("| 子集 | 平坦窗 | 近 300s 有呼吸 | 仅更早有呼吸 | 从未有呼吸 | 历史不足/未知 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for k, s in out["summaries"].items():
        if not s.get("n"):
            continue
        lines.append(f"| {k} | {s['flat_candidate']} | {s['flat_with_history']} | "
                     f"{s['flat_ever_history']} | {s['flat_no_history']} | "
                     f"{s['flat_history_unknown']} |")
    lines.append("")

    lines.append("## 3. 关键判读\n")
    u2_90 = out["summaries"]["u2_90"]
    u2_15 = out["summaries"]["u2_15"]
    if u2_90.get("n"):
        lines.append(f"- U2 90s：{u2_90['A_invalid']}/{u2_90['n']} "
                     f"({u2_90['v11_abstain_pct']:.1f}%) 会被 v1.1 的 `A<1e-6` 直接判 invalid；"
                     f"平坦窗 {u2_90['flat_candidate']} 个，其中近 300s 有呼吸 {u2_90['flat_with_history']} 个、"
                     f"仅更早有呼吸 {u2_90['flat_ever_history']} 个、从未有呼吸 {u2_90['flat_no_history']} 个、"
                     f"历史不足 {u2_90['flat_history_unknown']} 个。")
    if u2_15.get("n"):
        lines.append(f"- U2 15s：{u2_15['A_invalid']}/{u2_15['n']} "
                     f"({u2_15['v11_abstain_pct']:.1f}%) 会被直接判 invalid；"
                     f"平坦窗 {u2_15['flat_candidate']} 个，其中近 300s 有呼吸 {u2_15['flat_with_history']} 个、"
                     f"仅更早有呼吸 {u2_15['flat_ever_history']} 个、从未有呼吸 {u2_15['flat_no_history']} 个、"
                     f"历史不足 {u2_15['flat_history_unknown']} 个。")
    lines.append("- `FLAT_WITH_HISTORY` 是高置信屏息候选；`FLAT_EVER_HISTORY` 是长屏息/低置信候选；"
                 "`FLAT_NO_HISTORY`/`FLAT_HISTORY_UNKNOWN` 进入 UNKNOWN，不报警。")
    lines.append("- 不再使用 `case_has_breathing`：状态只使用当前窗和已结束的历史 bin；未来 bin 永远不参与。")
    lines.append("- `MAD(diff(x))` 在平台/量化段上会退化为 0：`zero_diff_frac` 均值见下表；"
                 "Stage 1 必须用平台鲁棒的噪声估计，不能直接沿用 v1.1 的 `noise_ratio` 阈值。")
    lines.append("")

    lines.append("## 4. 分布\n")
    for k, s in out["summaries"].items():
        if not s.get("n"):
            continue
        lines.append(f"### {k}\n")
        lines.append(f"- A: {s['A']}")
        lines.append(f"- noise_ratio (MAD diff): {s['noise_ratio']}")
        lines.append(f"- noise_ratio_alt (median-filter residual): {s['noise_ratio_alt']}")
        lines.append(f"- noise_ratio_hf (2 Hz residual): {s['noise_ratio_hf']}")
        lines.append(f"- zero_diff_frac 均值: {s['zero_diff_frac_mean']:.3f}")
        lines.append(f"- flat_run_sec: {s['flat_run_sec']}")
        lines.append(f"- nan_frac 均值: {s['nan_frac_mean']:.3f}; zero_frac 均值: {s['zero_frac_mean']:.3f}")
        lines.append(f"- 状态计数: {s['state_counts']}")
        lines.append(f"- 历史呼吸计数: {s['history_counts']}")
        lines.append("")

    with open(OUT_DIR / "quality_analysis.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {OUT_DIR/'quality_analysis.json'}")
    print(f"wrote {OUT_DIR/'quality_analysis.md'}")
    print(f"elapsed {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
