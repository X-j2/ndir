# -*- coding: utf-8 -*-
"""
Stage 1 B0 rule baseline on the frozen 200-case development split.

This script deliberately never loads or evaluates the 50-case Final Test.
It uses only cases_200.json, computes the 15-dim SI features and causal
quality state per window, applies the rule baseline, and writes aggregate
metrics plus a deterministic feature sample.

Usage:
  python tools/run_stage1_rule_baseline.py --window-sec 15 --step-sec 7.5
  python tools/run_stage1_rule_baseline.py --window-sec 30 --step-sec 15
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from stage0_common import (
    FS, REPO_ROOT, build_u2_events, iter_windows, load_case,
    split_events_by_support,
)
from features.quality_states import (
    FLAT_CANDIDATE_STATES, UNKNOWN_STATES, HistoryIndex,
    classify_signal_state, si_breathing_bins,
)
from features.si_features import FEATURE_NAMES, SPEC, extract_si_features
from features.rule_baseline import RULE_VERSION, UNKNOWN, predict_rule

DATA_SPLITS = REPO_ROOT / "data_splits"
OUT_DIR = REPO_ROOT / "report" / "stage1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LABELS = ("U0", "U1", "U2", "U3")
PREDS = LABELS + (UNKNOWN,)
QUALITY_STATES = (
    "VALID", "FLAT_WITH_HISTORY", "FLAT_EVER_HISTORY",
    "FLAT_NO_HISTORY", "FLAT_HISTORY_UNKNOWN", "GAP", "SATURATED",
    "POLARITY_INVALID", "QUANTILE_ARTIFACT", "NOISY",
)
SAMPLE_EVERY = 50       # deterministic feature sample among labeled windows
SAMPLE_U2_EVERY = 5     # denser sample for U2
QUALITY_SPEC_VERSION = "stage1_quality_rule_spec_v3"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dev_ids() -> list[int]:
    dev = load_json(DATA_SPLITS / "cases_200.json")
    test = load_json(DATA_SPLITS / "final_test_seed42.json")
    dev_ids = [int(c) for c in dev["case_ids"]]
    test_ids = {int(c) for c in test["case_ids"]}
    if len(dev_ids) != 200:
        raise RuntimeError(f"expected 200 dev cases, got {len(dev_ids)}")
    if set(dev_ids) & test_ids:
        raise RuntimeError("dev/final-test overlap")
    return dev_ids


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def init_metrics() -> dict:
    return {
        "confusion": {t: Counter() for t in LABELS},
        "state_by_truth": {t: Counter() for t in LABELS},
        "n": 0,
        "u2": {
            "n": 0, "pred": Counter(), "state": Counter(),
            "A_invalid": 0,
            "constant_flat": 0, "quantile_artifact": 0,
            "A_invalid_pred_u2": 0, "A_invalid_pred_unknown": 0,
        },
        "artifact": {"windows": 0, "by_truth": Counter()},
        "constant_flat": {"windows": 0, "by_truth": Counter()},
        "feature_sample": defaultdict(list),
        "feature_sample_u2": [],
        "feature_sample_total": 0,
        "flat_candidate_windows": 0,
        "flat_candidate_by_truth": Counter(),
        "flat_candidate_pred_u2": 0,
        "flat_candidate_pred_unknown": 0,
    }


def summarize(m: dict) -> dict:
    conf = m["confusion"]
    n = m["n"]
    correct = sum(conf[t][t] for t in LABELS)
    unknown = sum(conf[t][UNKNOWN] for t in LABELS)
    covered = n - unknown
    per_class = {}
    f1s = []
    for c in LABELS:
        tp = conf[c][c]
        fp = sum(conf[t][c] for t in LABELS if t != c)
        fn = sum(conf[c][p] for p in PREDS if p != c)
        support = sum(conf[c].values())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        f1s.append(f1)
        per_class[c] = {
            "support": int(support),
            "predicted": int(sum(conf[t][c] for t in LABELS)),
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": precision, "recall": recall, "f1": f1,
        }
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    u2 = m["u2"]
    u2_tp = conf["U2"]["U2"]
    u2_fp = sum(conf[t]["U2"] for t in LABELS if t != "U2")
    u2_fn = sum(conf["U2"][p] for p in PREDS if p != "U2")
    u2_precision = u2_tp / (u2_tp + u2_fp) if u2_tp + u2_fp else 0.0
    u2_recall = u2_tp / u2["n"] if u2["n"] else 0.0
    u2_f1 = (2 * u2_tp / (2 * u2_tp + u2_fp + u2_fn)
             if (2 * u2_tp + u2_fp + u2_fn) else 0.0)
    return {
        "n_labeled": n,
        "n_covered": covered,
        "coverage": covered / n if n else 0.0,
        "accuracy_all": correct / n if n else 0.0,
        "accuracy_covered": correct / covered if covered else 0.0,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion": {t: {p: int(conf[t].get(p, 0)) for p in PREDS}
                      for t in LABELS},
        "state_by_truth": {t: {s: int(m["state_by_truth"][t].get(s, 0))
                               for s in QUALITY_STATES} for t in LABELS},
        "u2_binary": {
            "support": int(u2["n"]),
            "precision": u2_precision,
            "recall": u2_recall,
            "f1": u2_f1,
            "predicted_counts": {p: int(u2["pred"].get(p, 0)) for p in PREDS},
            "state_counts": {s: int(u2["state"].get(s, 0))
                             for s in QUALITY_STATES},
            "A_invalid": int(u2["A_invalid"]),
            "constant_flat": int(u2["constant_flat"]),
            "quantile_artifact": int(u2["quantile_artifact"]),
            "A_invalid_pred_u2": int(u2["A_invalid_pred_u2"]),
            "A_invalid_pred_unknown": int(u2["A_invalid_pred_unknown"]),
        },
        "quantile_artifact": {
            "windows": int(m["artifact"]["windows"]),
            "by_truth": dict(m["artifact"]["by_truth"]),
        },
        "constant_flat": {
            "windows": int(m["constant_flat"]["windows"]),
            "by_truth": dict(m["constant_flat"]["by_truth"]),
        },
        "flat_candidate": {
            "windows": int(m["flat_candidate_windows"]),
            "by_truth": dict(m["flat_candidate_by_truth"]),
            "pred_u2": int(m["flat_candidate_pred_u2"]),
            "pred_unknown": int(m["flat_candidate_pred_unknown"]),
        },
        "feature_sample_n": int(m["feature_sample_total"]),
        "feature_summary": feature_summary(m),
    }


def feature_summary(m: dict) -> dict:
    out = {}
    for label, rows in m["feature_sample"].items():
        arr = np.asarray(rows, dtype=float)
        if len(arr) == 0:
            continue
        out[label] = {
            "n": int(len(arr)),
            "p10": {name: float(np.percentile(arr[:, i], 10))
                    for i, name in enumerate(FEATURE_NAMES)},
            "p50": {name: float(np.percentile(arr[:, i], 50))
                    for i, name in enumerate(FEATURE_NAMES)},
            "p90": {name: float(np.percentile(arr[:, i], 90))
                    for i, name in enumerate(FEATURE_NAMES)},
        }
    return out


def update_metrics(m: dict, truth: str, pred: str, state: str,
                   features: dict, sample_index: int) -> None:
    m["n"] += 1
    m["confusion"][truth][pred] += 1
    m["state_by_truth"][truth][state] += 1
    if features.get("quantile_artifact", False):
        m["artifact"]["windows"] += 1
        m["artifact"]["by_truth"][truth] += 1
    if features.get("constant_flat", False):
        m["constant_flat"]["windows"] += 1
        m["constant_flat"]["by_truth"][truth] += 1
    if truth == "U2":
        u = m["u2"]
        u["n"] += 1
        u["pred"][pred] += 1
        u["state"][state] += 1
        if features.get("constant_flat", False):
            u["constant_flat"] += 1
        if features.get("quantile_artifact", False):
            u["quantile_artifact"] += 1
        if features["A_invalid"]:
            u["A_invalid"] += 1
            if pred == "U2":
                u["A_invalid_pred_u2"] += 1
            elif pred == UNKNOWN:
                u["A_invalid_pred_unknown"] += 1
    if state in FLAT_CANDIDATE_STATES:
        m["flat_candidate_windows"] += 1
        m["flat_candidate_by_truth"][truth] += 1
        if pred == "U2":
            m["flat_candidate_pred_u2"] += 1
        elif pred == UNKNOWN:
            m["flat_candidate_pred_unknown"] += 1
    if sample_index % SAMPLE_EVERY == 0:
        m["feature_sample"][truth].append(list(features["vector"]))
        m["feature_sample_total"] += 1
        if truth == "U2" and sample_index % SAMPLE_U2_EVERY == 0:
            m["feature_sample_u2"].append(list(features["vector"]))


def run_scale(window_sec: float, step_sec: float, dev_ids: list[int]) -> dict:
    t0 = time.time()
    m = init_metrics()
    processed = 0
    for ci, cid in enumerate(dev_ids, 1):
        d = load_case(cid)
        u2_res = build_u2_events(d["etco2"])
        u2_reliable, u2_low = split_events_by_support(u2_res)
        hist = HistoryIndex(si_breathing_bins(d["co2"]))
        for rec in iter_windows(
                cid, window_sec, step_sec, u2_events=u2_reliable,
                low_support_events=u2_low, include_co2=True, case_data=d):
            truth = rec["label"]
            if truth not in LABELS:
                continue
            feat = extract_si_features(rec["co2"])
            state = classify_signal_state(
                rec["co2"], hist.summary(rec["start"] / FS),
                feature_diag=feat)
            rule = predict_rule(feat["features"], state)
            update_metrics(m, truth, rule["label"], state,
                           feat, processed)
            processed += 1
        if ci % 20 == 0:
            print(f"  ... {ci}/{len(dev_ids)} cases, {processed} windows, "
                  f"{time.time()-t0:.1f}s", flush=True)
    result = summarize(m)
    result.update({
        "rule_version": RULE_VERSION,
        "feature_spec_version": SPEC["version"],
        "quality_spec_version": QUALITY_SPEC_VERSION,
        "window_sec": float(window_sec),
        "step_sec": float(step_sec),
        "n_dev_cases": len(dev_ids),
        "sample_every": SAMPLE_EVERY,
        "sample_u2_every": SAMPLE_U2_EVERY,
        "elapsed_sec": time.time() - t0,
    })
    return result


def write_markdown(scale_key: str, r: dict) -> Path:
    path = OUT_DIR / f"rule_baseline_{scale_key}.md"
    lines = []
    lines.append(f"# Stage 1 B0 规则基线（{scale_key}）\n")
    lines.append(f"- 规则版本：`{r['rule_version']}`；SI 规范：`{SPEC['version']}`；"
                 f"质量规范：`{r.get('quality_spec_version', 'unknown')}`")
    lines.append(f"- 窗口：{r['window_sec']:.0f}s / 步长 {r['step_sec']:.1f}s；"
                 f"开发集 {r['n_dev_cases']} 例（Final Test 未加载）")
    lines.append(f"- 有标签窗：{r['n_labeled']}；覆盖：{r['n_covered']} "
                 f"({100*r['coverage']:.1f}%)；UNKNOWN："
                 f"{r['n_labeled']-r['n_covered']}")
    lines.append(f"- Accuracy（全部/覆盖）：{r['accuracy_all']:.4f} / "
                 f"{r['accuracy_covered']:.4f}；Macro-F1：{r['macro_f1']:.4f}\n")

    lines.append("## 1. 四类结果\n")
    lines.append("| 类别 | support | precision | recall | F1 |")
    lines.append("|---|---:|---:|---:|---:|")
    for c in LABELS:
        p = r["per_class"][c]
        lines.append(f"| {c} | {p['support']} | {p['precision']:.3f} | "
                     f"{p['recall']:.3f} | {p['f1']:.3f} |")
    lines.append("")

    lines.append("## 2. U2 二分类（UNKNOWN 不当作正例）\n")
    u = r["u2_binary"]
    lines.append(f"- U2 support：{u['support']}；precision {u['precision']:.3f}，"
                 f"recall {u['recall']:.3f}，F1 {u['f1']:.3f}")
    lines.append(f"- U2 预测分布：{u['predicted_counts']}")
    lines.append(f"- U2 质量状态：{u['state_counts']}")
    lines.append(f"- U2 中 A=0/non-finite（尺度无关 invalid）：{u['A_invalid']}；"
                 f"其中真正常数窗 {u['constant_flat']}、"
                 f"分位退化伪迹窗 {u['quantile_artifact']}；"
                 f"预测 U2 {u['A_invalid_pred_u2']}，"
                 f"UNKNOWN {u['A_invalid_pred_unknown']}")
    lines.append("- `A=0` 只表示 q90-q10=0，不等于常数窗。有限样本完全相等才是"
                 "真正常数窗并保留候选路径；仍含稀疏脉冲/变化的窗为 "
                 "`QUANTILE_ARTIFACT`/UNKNOWN。")
    lines.append("- 低幅判断按 `A < 0.05 * 因果历史 A_ref`，不再使用绝对 A 门限；"
                 "U2 的 `FLAT_*` 状态分布见上表。\n")

    lines.append("## 3. 分位退化伪迹与常数窗\n")
    q = r["quantile_artifact"]
    c = r["constant_flat"]
    lines.append(f"- 全部有标签窗中，分位退化伪迹：{q['windows']}；按真值 {q['by_truth']}")
    lines.append(f"- 全部有标签窗中，真正常数窗：{c['windows']}；按真值 {c['by_truth']}")
    lines.append("- 分位退化伪迹先于噪声/平坦判定，进入 UNKNOWN，不能成为 U2 候选。\n")

    lines.append("## 4. 平坦候选与 UNKNOWN\n")
    f = r["flat_candidate"]
    lines.append(f"- 平坦候选窗：{f['windows']}；按真值 {f['by_truth']}")
    lines.append(f"- 平坦候选中预测 U2：{f['pred_u2']}；UNKNOWN：{f['pred_unknown']}")
    lines.append("- `FLAT_NO_HISTORY` / `FLAT_HISTORY_UNKNOWN` / GAP / 饱和 / "
                 "极性错误 / 分位退化伪迹 / 噪声均进入 UNKNOWN，不被当作 U0。\n")

    lines.append("## 5. 质量状态分布\n")
    lines.append("| 真值 | " + " | ".join(QUALITY_STATES) + " |")
    lines.append("|---|" + "---:|" * len(QUALITY_STATES))
    for c in LABELS:
        vals = r["state_by_truth"][c]
        lines.append("| " + c + " | " +
                     " | ".join(str(vals.get(s, 0)) for s in QUALITY_STATES) + " |")
    lines.append("")
    lines.append("## 6. 混淆矩阵\n")
    lines.append("| 真值 \\ 预测 | " + " | ".join(PREDS) + " |")
    lines.append("|---|" + "---:|" * len(PREDS))
    for c in LABELS:
        lines.append("| " + c + " | " +
                     " | ".join(str(r["confusion"][c].get(p, 0)) for p in PREDS) + " |")
    lines.append("")
    lines.append(f"特征样本：每 {r['sample_every']} 个有标签窗取 1 个，"
                 f"共 {r['feature_sample_n']} 个；分位数见 JSON `feature_summary`。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-sec", type=float, default=15.0)
    ap.add_argument("--step-sec", type=float, default=7.5)
    ap.add_argument("--tag", type=str, default=None)
    args = ap.parse_args()
    dev_ids = load_dev_ids()
    print(f"dev cases: {len(dev_ids)}; window {args.window_sec}s / {args.step_sec}s")
    r = run_scale(args.window_sec, args.step_sec, dev_ids)
    tag = args.tag or f"{int(args.window_sec)}s"
    out = OUT_DIR / f"rule_baseline_{tag}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    md = write_markdown(tag, r)
    print(f"wrote {out}")
    print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
