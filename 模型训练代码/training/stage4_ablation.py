# -*- coding: utf-8 -*-
"""Stage 4 ablations: scales, fusion/dedup, temporal engine, features."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from stage0_common import load_case  # noqa: E402
from training.stage2_data import (  # noqa: E402
    build_scale_dataset, cache_paths, training_mask, LABEL_CODE,
    STATE_CODE,
)
from training.stage2_calibration import apply_calibrator  # noqa: E402
from training.stage2_eval import (  # noqa: E402
    metrics_from_window_confusion, window_confusion,
)
from training.run_stage2 import (  # noqa: E402
    concat_rows, label_ints, predict_full, rows_by_case, train_rf,
)
from training.stage3_eval import aggregate_stage3  # noqa: E402
from training.stage3_event_engine import EventParams  # noqa: E402
from training.stage4_fusion import (  # noqa: E402
    build_series_by_case, evaluate_fusion, fusion_params_from_dict,
)
from training.stage4_perturb import perturb_signal  # noqa: E402
from features.si_features import (  # noqa: E402
    FEATURE_NAMES, extract_si_features,
)

OUT = ROOT / "report" / "stage4"
MODEL_DIR = ROOT / "models" / "stage4"
FOLD_DIR = MODEL_DIR / "folds"
SCALES = (("U15", 15.0, 7.5), ("U30", 30.0, 15.0),
          ("U60", 60.0, 30.0), ("U90", 90.0, 45.0))
SCALE_CFG = {s: (ws, st) for s, ws, st in SCALES}
WEIGHT = "u2x10"


def _load_fold_configs():
    with open(MODEL_DIR / "fold_configs.json", encoding="utf-8") as f:
        return json.load(f)["folds"]


def _load_asset(scale: str, fold: int):
    import joblib
    mp = FOLD_DIR / f"f{fold}_{scale.lower()}_rf.joblib"
    cp = FOLD_DIR / f"f{fold}_{scale.lower()}_calibrator.json"
    payload = joblib.load(mp)
    with open(cp, encoding="utf-8") as f:
        cal = json.load(f)
    return payload["model"], cal


def _load_meta(ws: float) -> dict:
    _, mp = cache_paths(ws)
    with open(mp, encoding="utf-8") as f:
        return json.load(f)


def build_oof_series():
    data_by_scale, meta_by_scale, rows_by_scale = {}, {}, {}
    p_oof = {}
    for scale, ws, st in SCALES:
        data, _ = build_scale_dataset(ws, st)
        meta = _load_meta(ws)
        data_by_scale[scale] = data
        meta_by_scale[scale] = meta
        rows_by_scale[scale] = rows_by_case(data)
        p_oof[scale] = np.zeros((len(data["label"]), 3), dtype=np.float32)
    folds = _load_fold_configs()
    for fr in folds:
        fi = int(fr["fold"])
        for scale, _, _ in SCALES:
            model, cal = _load_asset(scale, fi)
            rows = concat_rows(fr["test_cases"], rows_by_scale[scale])
            p_raw = predict_full(model, data_by_scale[scale]["features"][rows])
            p_oof[scale][rows] = apply_calibrator(cal, p_raw)
    all_cases = sorted(int(c) for c in meta_by_scale["U15"]["case_meta"])
    series = build_series_by_case(data_by_scale, meta_by_scale,
                                  rows_by_scale, p_oof, all_cases)
    return series, meta_by_scale["U15"], folds, all_cases


def _eval_variant(series, meta, folds, all_cases, variant: str,
                  scales=None) -> dict:
    per_case = {}
    for fr in folds:
        base = fusion_params_from_dict(fr["fusion"]["params"])
        if variant == "logodds":
            p = type(base)(**{**base.__dict__, "mode": "logodds"})
        elif variant == "no_decay":
            p = type(base)(**{**base.__dict__, "tau_sec": 1e9})
        elif variant == "no_hysteresis":
            ev = EventParams(
                suspect_p=base.event.suspect_p,
                active_enter_p=base.event.active_enter_p,
                active_exit_p=base.event.active_exit_p,
                suspect_on_count=1, suspect_off_count=1,
                active_on_count=1, active_off_count=1)
            p = type(base)(**{**base.__dict__, "event": ev})
        else:
            p = base
        ev = evaluate_fusion(series, meta, fr["test_cases"], p,
                             scales=scales or tuple(SCALES_NAMES))
        per_case.update(ev["per_case"])
    agg = aggregate_stage3(per_case, all_cases)
    return {k: v for k, v in agg.items() if k != "per_case"}


SCALES_NAMES = tuple(s for s, _, _ in SCALES)


def _feature_subset_ablation(scale: str, ws: float, st: float) -> dict:
    data, meta = build_scale_dataset(ws, st)
    rows = rows_by_case(data)
    # RF-only population: VALID quality and truth U0/U1/U2.  The same mask
    # is used for training and evaluation and for every feature subset.
    mask = training_mask(data)
    all_cases = sorted(int(c) for c in meta["case_meta"])
    from training.run_stage2 import case_folds
    folds = case_folds(meta, all_cases, 5, seed=42)
    eval_all = concat_rows(all_cases, rows)
    eval_all = eval_all[mask[eval_all]]
    eval_cases = sorted(set(int(c) for c in data["case_id"][eval_all].tolist()))
    eval_labels = label_ints(data, eval_all)
    supports = {str(c): int(np.sum(eval_labels == c)) for c in (0, 1, 2)}
    subsets = {
        "full": list(range(len(FEATURE_NAMES))),
        "no_band_ratio": [i for i, n in enumerate(FEATURE_NAMES)
                          if n != "band_ratio"],
        "no_acf_peak": [i for i, n in enumerate(FEATURE_NAMES)
                        if n != "acf_peak"],
        "no_flat_run_frac": [i for i, n in enumerate(FEATURE_NAMES)
                             if n != "flat_run_frac"],
        "no_three": [i for i, n in enumerate(FEATURE_NAMES)
                     if n not in ("band_ratio", "acf_peak",
                                  "flat_run_frac")],
    }
    out = {}
    for name, cols in subsets.items():
        counts = np.zeros((3, 5), dtype=np.int64)
        for tr_cases, te_cases in folds:
            tr = concat_rows(tr_cases, rows)
            te = concat_rows(te_cases, rows)
            fit = tr[mask[tr]]
            te_eval = te[mask[te]]
            if not len(fit) or not len(te_eval):
                continue
            clf = train_rf(data["features"][fit][:, cols],
                           label_ints(data, fit),
                           {0: 1.0, 1: 1.0, 2: 10.0})
            pred = np.argmax(predict_full(
                clf, data["features"][te_eval][:, cols]), axis=1)
            lab = label_ints(data, te_eval)
            counts += window_confusion(lab, pred)
        out[name] = {
            **metrics_from_window_confusion(counts),
            "n_cases": len(eval_cases),
            "n_eval_rows": int(counts.sum()),
            "supports": supports,
            "population": "RF-only: VALID and truth U0/U1/U2",
            "weight": WEIGHT,
            "calibration": "none",
        }
    return out


def band_ratio_stability_decision(conditions: dict) -> dict:
    """Apply the frozen gain/bias-only band_ratio criterion.

    Noise/Gamma/clipping values are reported separately and must not replace
    the gain/bias median-relative-change criterion.
    """
    gain_keys = ("gain0.5", "gain2")
    bias_keys = ("bias+0.5", "bias-0.5")
    rho_gain = min(float(conditions[k]["spearman"] or 0.0)
                   for k in gain_keys)
    rho_bias = min(float(conditions[k]["spearman"] or 0.0)
                   for k in bias_keys)
    median_gb = max(float(conditions[k]["median_relative_change"])
                    for k in gain_keys + bias_keys)
    stable = bool(rho_gain >= 0.8 and rho_bias >= 0.8 and
                  median_gb <= 0.20)
    return {
        "rho_gain_min": float(rho_gain),
        "rho_bias_min": float(rho_bias),
        "median_relative_change_gain_bias": float(median_gb),
        "stable": stable,
        "criterion": ("gain/bias rho>=0.8 and gain/bias median rel<=0.20; "
                      "other perturbations reported separately"),
        "other_perturbations": {
            k: conditions[k] for k in conditions
            if k not in set(gain_keys + bias_keys)},
    }


def _band_ratio_stability() -> dict:
    data, _ = build_scale_dataset(15.0, 7.5)
    rows = np.where((data["state"] == STATE_CODE["VALID"]) &
                    np.isin(data["label"], [LABEL_CODE["U0"],
                                            LABEL_CODE["U1"],
                                            LABEL_CODE["U2"]]))[0]
    rng = np.random.default_rng(42)
    rows = rng.choice(rows, size=min(5000, len(rows)), replace=False)
    bi = FEATURE_NAMES.index("band_ratio")
    cache = {}
    base, pert = [], {k: [] for k in ("gain0.5", "gain2", "bias+0.5",
                                      "bias-0.5", "noise1", "drop5",
                                      "gamma1.25", "clip")}
    for r in rows:
        cid = int(data["case_id"][r])
        start = int(data["start"][r])
        case = cache.get(cid)
        if case is None:
            case = load_case(cid)
            cache[cid] = case
        seg = case["co2"][start:start + 1500]
        b = extract_si_features(seg)["features"]["band_ratio"]
        specs = {
            "gain0.5": ("gain", 0.5), "gain2": ("gain", 2.0),
            "bias+0.5": ("bias", 0.5), "bias-0.5": ("bias", -0.5),
            "noise1": ("noise", 1.0), "drop5": ("drop", 0.05),
            "gamma1.25": ("gamma", 1.25), "clip": ("clip", 1.0),
        }
        base.append(float(b))
        for name, (kind, lev) in specs.items():
            y = perturb_signal(seg, kind, lev, 0)
            pert[name].append(float(
                extract_si_features(y)["features"]["band_ratio"]))
    base = np.asarray(base)
    out = {}
    for name, vals in pert.items():
        v = np.asarray(vals)
        rho = spearmanr(base, v).correlation
        rel = np.abs(v - base) / np.maximum(np.abs(base), 1e-9)
        out[name] = {"spearman": float(rho) if np.isfinite(rho) else None,
                     "median_relative_change": float(np.median(rel))}
    decision = band_ratio_stability_decision(out)
    return {"n_windows": int(len(rows)), "conditions": out, **decision}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-only", action="store_true",
                    help="rerun only RF-only feature ablation and band_ratio")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.feature_only:
        path = OUT / "ablation.json"
        out = json.load(open(path, encoding="utf-8")) if path.exists() else {}
        out["feature_ablation"] = {
            "U15": _feature_subset_ablation("U15", 15.0, 7.5),
            "U30": _feature_subset_ablation("U30", 30.0, 15.0),
        }
        out["band_ratio_stability_15s"] = _band_ratio_stability()
        ta = out.get("temporal_ablation", {})
        if "no_hysteresis" in ta:
            ta["no_persistence_counts_keep_dual_threshold"] = ta.pop(
                "no_hysteresis")
        out["temporal_ablation"] = ta
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(json.dumps({
            "feature_ablation": out["feature_ablation"],
            "band_ratio_stability_15s": out["band_ratio_stability_15s"],
            "temporal_keys": list(ta),
        }, ensure_ascii=False, indent=2)[:2500])
        return 0
    series, meta15, folds, all_cases = build_oof_series()
    out = {
        "scale_ablation": {
            "u15_u30": _eval_variant(series, meta15, folds, all_cases,
                                     "main", scales=("U15", "U30")),
            "u15_u30_u60": _eval_variant(series, meta15, folds, all_cases,
                                         "main",
                                         scales=("U15", "U30", "U60")),
            "all_four": _eval_variant(series, meta15, folds, all_cases,
                                      "main"),
        },
        "fusion_ablation": {
            "consensus": _eval_variant(series, meta15, folds, all_cases,
                                       "main"),
            "logodds_sum": _eval_variant(series, meta15, folds, all_cases,
                                         "logodds"),
            "no_decay": _eval_variant(series, meta15, folds, all_cases,
                                      "no_decay"),
        },
        "temporal_ablation": {
            "full_hysteresis": _eval_variant(series, meta15, folds, all_cases,
                                             "main"),
            # counts=1 removes persistence counting but KEEPS the dual
            # active_enter/active_exit thresholds.
            "no_persistence_counts_keep_dual_threshold":
                _eval_variant(series, meta15, folds, all_cases,
                              "no_hysteresis"),
        },
        "feature_ablation": {
            "U15": _feature_subset_ablation("U15", 15.0, 7.5),
            "U30": _feature_subset_ablation("U30", 30.0, 15.0),
        },
        "band_ratio_stability_15s": _band_ratio_stability(),
    }
    with open(OUT / "ablation.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
