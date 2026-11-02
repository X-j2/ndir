# -*- coding: utf-8 -*-
"""Reproducible synthetic demonstration of Stage 1 quality-state separation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stage0_common import FS, REPO_ROOT
from features.quality_states import (
    HistoryIndex, classify_signal_state, si_breathing_bins, state_is_unknown,
)
from features.si_features import extract_si_features
from features.rule_baseline import predict_rule

OUT = REPO_ROOT / "report" / "stage1" / "quality_state_demo.json"


def normal_wave(duration_sec: float, rr: float = 15.0,
                etco2: float = 40.0, base: float = 5.0) -> np.ndarray:
    t = np.arange(int(duration_sec * FS)) / FS
    return base + (etco2 - base) * (np.sin(2 * np.pi * rr / 60.0 * t) > 0.3)


def flat_wave(duration_sec: float, value: float = 5.0) -> np.ndarray:
    return np.full(int(duration_sec * FS), value, dtype=float)


def sparse_pulse_wave(duration_sec: float, value: float = 5.0,
                      amplitude: float = 40.0,
                      pulse_secs=(3.0, 8.0, 12.0),
                      pulse_len_sec: float = 0.2) -> np.ndarray:
    """Constant baseline plus a few brief pulses (q90-q10 still zero)."""
    x = flat_wave(duration_sec, value=value)
    width = int(pulse_len_sec * FS)
    for sec in pulse_secs:
        i = int(sec * FS)
        if i + width <= len(x):
            x[i:i + width] = value + amplitude
    return x


def row(name, window, prior, rail_high=None):
    diag = extract_si_features(window)
    hist = HistoryIndex(si_breathing_bins(prior)).summary(len(prior) / FS)
    state = classify_signal_state(
        window, hist, rail_high=rail_high, feature_diag=diag)
    rule = predict_rule(diag["features"], state)
    return {
        "case": name,
        "state": state,
        "unknown": state_is_unknown(state),
        "rule_label": rule["label"],
        "A": diag["A"],
        "finite_span": diag["finite_span"],
        "constant_flat": diag["constant_flat"],
        "quantile_artifact": diag["quantile_artifact"],
        "flat_run_sec": diag["flat_run_sec"],
        "peak_rate": diag["features"]["peak_rate"],
    }


def main() -> int:
    normal_prior = normal_wave(60)
    flat_prior = flat_wave(60)
    rows = [
        row("normal + recent history", normal_wave(15), normal_prior),
        row("flat A=0 + recent history", flat_wave(15), normal_prior),
        row("flat A=0 + no history", flat_wave(15), flat_prior),
        row("flat A=0 + insufficient history", flat_wave(15), flat_wave(10)),
    ]
    gap = normal_wave(15)
    gap[300:800] = np.nan
    rows.append(row("5s missing gap", gap, normal_prior))

    clipped = normal_wave(15)
    rail = float(np.percentile(clipped, 95))
    clipped[clipped >= rail] = rail
    rows.append(row("clipped at rail", clipped, normal_prior, rail_high=rail))
    rows.append(row("inverted trace", -normal_wave(15), normal_prior))

    t = np.arange(15 * FS) / FS
    noisy = normal_wave(15) + 20.0 * np.sin(2 * np.pi * 10.0 * t)
    rows.append(row("high-frequency noise", noisy, normal_prior))
    rng = np.random.default_rng(11)
    low_noisy = 0.01 * normal_wave(15) + rng.normal(0, 0.2, 15 * FS)
    rows.append(row("low-amplitude noise", low_noisy, normal_prior))
    sparse = sparse_pulse_wave(15)
    rows.append(row("sparse pulses + recent history", sparse, normal_prior))
    rows.append(row("sparse pulses + no history", sparse, flat_prior))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"synthetic_only": True, "rows": rows}, f,
                  ensure_ascii=False, indent=2)
    md = OUT.with_suffix(".md")
    lines = ["# Stage 1 质量状态合成演示\n",
             "仅用合成波形；不是设备实测。\n",
             "| case | state | UNKNOWN | rule | A | finite_span | "
             "constant_flat | quantile_artifact | flat_run_sec | peak_rate |",
             "|---|---|---|---|---:|---:|---|---|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['case']} | {r['state']} | {r['unknown']} | "
                     f"{r['rule_label']} | {r['A']:.3f} | "
                     f"{r['finite_span']:.3f} | {r['constant_flat']} | "
                     f"{r['quantile_artifact']} | {r['flat_run_sec']:.2f} | "
                     f"{r['peak_rate']:.1f} |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
