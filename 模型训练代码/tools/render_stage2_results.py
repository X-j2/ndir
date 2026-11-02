# -*- coding: utf-8 -*-
"""Render Stage 2 v2 U15/U30 results as Markdown."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "stage2"


def load(name):
    with open(OUT / name, "r", encoding="utf-8") as f:
        return json.load(f)


def f(v, n=3, dash="—"):
    return dash if v is None else f"{float(v):.{n}f}"


def ci(pair, n=3):
    return "—" if not pair else f"[{pair[0]:.{n}f}, {pair[1]:.{n}f}]"


def window_block(title, m, b=None):
    lines = [f"### {title}\n"]
    lines.append(f"- n={m['n']}，覆盖 {100*m['coverage']:.2f}%；"
                 f"Macro-F1 {f(m['macro_f1'],4)}；"
                 f"U2 P/R/F1 {f(m['u2_binary']['precision'])}/"
                 f"{f(m['u2_binary']['recall'])}/{f(m['u2_binary']['f1'])}")
    if b is not None:
        lines.append(f"- 同口径 B0：Macro-F1 {f(b['macro_f1'],4)}；"
                     f"U2 P/R/F1 {f(b['u2_binary']['precision'])}/"
                     f"{f(b['u2_binary']['recall'])}/"
                     f"{f(b['u2_binary']['f1'])}")
    lines.append("")
    return lines


def event_block(title, e, b=None):
    lines = [f"### {title}\n"]
    lines.append(f"- 全量可靠 U2：{e['n_truth_events']} 事件；其中无窗口病例事件 "
                 f"{e['n_truth_events_no_window_cases']}；低支撑 {e['n_low_support_events']} "
                 "只计数")
    lines.append(f"- TP/FN/FP = {e['tp']}/{e['fn']}/{e['fp']}；"
                 f"Recall {f(e['event_recall'])}，PPV {f(e['event_ppv'])}，"
                 f"F1 {f(e['event_f1'])}")
    lines.append(f"- 可评分 FA/h {f(e['fa_per_hour_scorable'])}；"
                 f"总时长敏感性 {f(e['fa_per_hour_total'])}；"
                 f"可评分小时 {f(e['scorable_hours'],1)}，"
                 f"排除 {f(e['excluded_hours'],2)} h")
    lines.append(f"- 原始告警 {e['raw_alarm_episodes']}；可评分片段 "
                 f"{e['n_alarm_segments']}；完全不可评分 {e['unscorable_alarm_episodes']}；"
                 f"不可评分时长 {f(e['unscorable_alarm_duration'],1)} s")
    lines.append(f"- 告警持续占比 {100*e['alarm_duration_fraction']:.2f}%；"
                 f"告警/h {f(e['alarms_per_hour'])}")
    t = e["ttd_s"]
    lines.append(f"- TTD 中位 {f(t['median'])} s；非负中位 "
                 f"{f(t['median_nonnegative'])} s；早期 {t['n_early']}，"
                 f"提前中位 {f(t['median_early_lead'])} s；"
                 f"30/60/90 s 检出率 {f(t['detect_30s'])}/"
                 f"{f(t['detect_60s'])}/{f(t['detect_90s'])}")
    s = e["suspect"]
    lines.append(f"- SUSPECT pre/post/any = {f(s['suspect_recall_pre'])}/"
                 f"{f(s['suspect_recall_post'])}/{f(s['suspect_recall_any'])}；"
                 f"未匹配 {s['unmatched_suspects']}（"
                 f"{f(s['false_suspects_per_hour'])}/h）")
    if b is not None:
        lines.append(f"- 同口径 B0 事件：TP/FN/FP = {b['tp']}/{b['fn']}/{b['fp']}；"
                     f"Recall/PPV/F1 = {f(b['event_recall'])}/"
                     f"{f(b['event_ppv'])}/{f(b['event_f1'])}；"
                     f"可评分 FA/h {f(b['fa_per_hour_scorable'])}")
    lines.append("- 口径：窗口结束时理论可用时间；单尺度原始告警代理，不是 Stage 3 ACTIVE。\n")
    return lines


def render(tag: str, r: dict) -> list[str]:
    ws = int(r["window_sec"])
    lines = [f"## {ws} s 单尺度（U{ws}）\n"]
    lines.append(f"- dataset fingerprint：`{r['dataset_fingerprint']}`")
    lines.append(f"- 训练窗 {r['n_train_rows']}；U0/U1/U2 = "
                 f"{r['train_label_counts']['0']}/{r['train_label_counts']['1']}/"
                 f"{r['train_label_counts']['2']}")
    lines.append(f"- 外层窗口 n={r['outer_window']['n']}；"
                 f"U2 95% CI "
                 f"P {ci(r['outer_window_bootstrap']['ci95']['u2_precision'])} "
                 f"R {ci(r['outer_window_bootstrap']['ci95']['u2_recall'])} "
                 f"F1 {ci(r['outer_window_bootstrap']['ci95']['u2_f1'])}\n")
    lines += window_block("RF 窗口级", r["outer_window"], r["b0_window"])

    lines.append("### 概率校准（外层交叉拟合 OOF）\n")
    lines.append("| 指标 | raw RF | calibrated |")
    lines.append("|---|---:|---:|")
    raw, cal = r["calibration_raw"], r["calibration_calibrated"]
    lines.append(f"| log loss | {f(raw['log_loss'],4)} | {f(cal['log_loss'],4)} |")
    lines.append(f"| top ECE | {f(raw['top_label_ece'],4)} | "
                 f"{f(cal['top_label_ece'],4)} |")
    for k in ("0", "1", "2"):
        lines.append(f"| U{k} ECE | {f(raw['classwise'][k]['ece'],4)} | "
                     f"{f(cal['classwise'][k]['ece'],4)} |")
    lines.append("")

    lines += event_block("RF 事件级（全量 303 事件）", r["outer_event"],
                         r["b0_event"])
    lines.append("### 外层折选择\n")
    lines.append("| fold | weight | calibrator | threshold | test events | "
                 "Macro-F1 | U2 F1 | event F1 | FA/h |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|")
    for fr in r["folds"]:
        lines.append(f"| {fr['fold']+1} | {fr['weight']} | "
                     f"{fr['calibrator_method']} | {f(fr['threshold'],2)} | "
                     f"{fr['test_events']} | {f(fr['window']['macro_f1'],4)} | "
                     f"{f(fr['window']['u2_binary']['f1'],4)} | "
                     f"{f(fr['event']['event_f1'],4)} | "
                     f"{f(fr['event']['fa_per_hour_scorable'])} |")
    lines.append("")
    fc = r["final_config"]
    mp = ROOT / fc["model_path"]
    lines.append("### 最终部署配置（全开发集选择，非外部验证）\n")
    lines.append(f"- weight `{fc['weight']}`；calibrator `{fc['calibrator_method']}`；"
                 f"threshold {f(fc['threshold'],2)}")
    lines.append(f"- model `{Path(fc['model_path']).as_posix()}`，"
                 f"{mp.stat().st_size} B，SHA256 `{fc['model_sha256']}`")
    lines.append("")
    return lines


def main() -> int:
    parts = ["# Stage 2 v2 U15/U30 结果摘要\n",
             "由 `tools/render_stage2_results.py` 从结果 JSON 生成。"
             "事件时间为窗口结束时的理论可用时间；事件级为单尺度原始告警代理。\n"]
    for tag in ("u15", "u30"):
        parts.extend(render(tag, load(f"{tag}_results.json")))
    (OUT / "stage2_summary.md").write_text("\n".join(parts) + "\n",
                                           encoding="utf-8")
    print(OUT / "stage2_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
