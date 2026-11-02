# -*- coding: utf-8 -*-
"""Stage 4 robustness: waveform perturbations with fixed truth.

The final scheme is read from Stage 4 results.  If fusion passed the frozen
benefit rule, the full four-scale fusion is stressed; otherwise the
inner-selected best single scale per outer fold is stressed.  Models,
calibrators and event parameters are the outer-fold clean configurations; no
perturbed data is used for fitting or selection.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from stage0_common import load_case  # noqa: E402
from training.stage2_data import (  # noqa: E402
    STATE_CODE, build_scale_dataset, cache_paths, LABEL_CODE,
)
from training.stage2_calibration import apply_calibrator  # noqa: E402
from training.stage2_eval import (  # noqa: E402
    bootstrap_event_metrics, metrics_from_window_confusion, window_confusion,
)
from training.run_stage2 import predict_full  # noqa: E402
from training.stage3_eval import (  # noqa: E402
    aggregate_stage3, evaluate_engine_case,
)
from training.stage3_event_engine import params_from_dict  # noqa: E402
from training.stage4_fusion import (  # noqa: E402
    build_scale_series, evaluate_fusion, fusion_params_from_dict,
)
from training.stage4_perturb import (  # noqa: E402
    build_perturbed_case_rows, perturb_signal, perturb_signal_detail,
)
from features.si_features import extract_si_features, FEATURE_NAMES  # noqa: E402

OUT = ROOT / "report" / "stage4"
MODEL_DIR = ROOT / "models" / "stage4"
FOLD_DIR = MODEL_DIR / "folds"
SCALES = (("U15", 15.0, 7.5), ("U30", 30.0, 15.0),
          ("U60", 60.0, 30.0), ("U90", 90.0, 45.0))
SCALE_CFG = {s: (ws, st) for s, ws, st in SCALES}

EVENT_CONDITIONS = [
    {"name": "clean", "kind": "none", "level": 0.0, "seeds": [0]},
    {"name": "gain_0.5", "kind": "gain", "level": 0.5, "seeds": [0]},
    {"name": "gain_2.0", "kind": "gain", "level": 2.0, "seeds": [0]},
    {"name": "bias_p0.5A", "kind": "bias", "level": 0.5, "seeds": [0]},
    {"name": "bias_m0.5A", "kind": "bias", "level": -0.5, "seeds": [0]},
    {"name": "noise_2", "kind": "noise", "level": 2.0, "seeds": list(range(5))},
    {"name": "drop_5pct", "kind": "drop", "level": 0.05,
     "seeds": list(range(5))},
    {"name": "gamma_1.25", "kind": "gamma", "level": 1.25, "seeds": [0]},
    {"name": "clip_q1q99", "kind": "clip", "level": 1.0, "seeds": [0]},
    {"name": "lag_2s", "kind": "lag", "level": 2.0, "seeds": [0]},
    {"name": "drift_0.5A_min", "kind": "drift", "level": 0.5, "seeds": [0]},
    {"name": "negative_gain", "kind": "negative", "level": -1.0,
     "seeds": [0]},
]

WINDOW_CONDITIONS = [
    ("gain", [0.2, 0.5, 1.0, 2.0, 5.0], [0]),
    ("bias", [-0.5, -0.2, 0.0, 0.2, 0.5], [0]),
    ("noise", [0.0, 0.5, 1.0, 2.0, 5.0], [0, 1, 2, 3, 4]),
    ("drop", [0.0, 0.01, 0.05, 0.10], [0, 1, 2, 3, 4]),
    ("burst_drop", [0.2, 1.0, 5.0], [0, 1, 2, 3, 4]),
    ("lag", [0.0, 0.5, 1.0, 2.0], [0]),
    ("drift", [0.0, 0.1, 0.5], [0]),
    ("gamma", [0.5, 0.8, 1.0, 1.25, 2.0], [0]),
    ("clip", [1.0], [0]),
    ("negative", [-1.0], [0]),
    ("quantize", [8, 10, 12, 16], [0]),
]


def _log(msg=""):
    print(msg, flush=True)


def _load_fold_configs():
    with open(MODEL_DIR / "fold_configs.json", encoding="utf-8") as f:
        return json.load(f)["folds"]


def _load_comparison():
    with open(OUT / "comparison.json", encoding="utf-8") as f:
        return json.load(f)


def _load_meta(window_sec: float) -> dict:
    _, mp = cache_paths(window_sec)
    with open(mp, encoding="utf-8") as f:
        return json.load(f)


def _load_asset(scale: str, fold: int):
    import joblib
    mp = FOLD_DIR / f"f{fold}_{scale.lower()}_rf.joblib"
    cp = FOLD_DIR / f"f{fold}_{scale.lower()}_calibrator.json"
    payload = joblib.load(mp)
    with open(cp, encoding="utf-8") as f:
        cal = json.load(f)
    return payload["model"], cal


def _predict_local(data: dict, model, cal) -> np.ndarray:
    n = len(data["features"])
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)
    X = np.nan_to_num(np.asarray(data["features"], dtype=float),
                      nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -np.finfo(np.float32).max,
                np.finfo(np.float32).max).astype(np.float32)
    return apply_calibrator(cal, predict_full(model, X))


def _eval_single_case(data_local: dict, meta_scale: dict, cid: int,
                      p_cal: np.ndarray, params, scale: str,
                      window_sec: float, step_sec: float) -> dict:
    if len(data_local["label"]):
        rows = np.arange(len(data_local["label"]), dtype=int)
    else:
        rows = np.empty(0, dtype=int)
    return evaluate_engine_case(
        data_local, meta_scale, cid, rows, p_cal, params, scale,
        window_sec, step_sec)


def _eval_fusion_case(data_by_scale: dict, meta_by_scale: dict,
                      p_by_scale: dict, cid: int, params) -> dict:
    series = {}
    for scale, data in data_by_scale.items():
        n = len(data["label"])
        rows = np.arange(n, dtype=int)
        series[scale] = build_scale_series(
            data, meta_by_scale[scale], cid, rows,
            p_by_scale[scale], scale)
    return evaluate_fusion({cid: series}, meta_by_scale["U15"], [cid],
                           params)["per_case"][cid]


def _run_event_one(target: str, cond: dict, seed: int) -> tuple[str, dict]:
    folds = _load_fold_configs()
    metas = {s: _load_meta(SCALE_CFG[s][0]) for s in SCALE_CFG}
    needed = ([s for s, _, _ in SCALES] if target == "fusion"
              else sorted({fr["best_scale"] for fr in folds}))
    assets = {}
    for fr in folds:
        for scale in needed:
            assets[(fr["fold"], scale)] = _load_asset(scale, int(fr["fold"]))
    per_case = {}
    missing_samples = 0
    samples = 0
    for fr in folds:
        fi = int(fr["fold"])
        for cid in fr["test_cases"]:
            cid = int(cid)
            case = load_case(cid)
            scale_names = ([s for s, _, _ in SCALES] if target == "fusion"
                           else [fr["best_scale"]])
            data_by_scale, p_by_scale = {}, {}
            for scale in scale_names:
                ws, st = SCALE_CFG[scale]
                data_local, m = build_perturbed_case_rows(
                    case, cid, ws, st, cond["kind"],
                    float(cond["level"]), int(seed))
                model, cal = assets[(fi, scale)]
                p_cal = _predict_local(data_local, model, cal)
                data_by_scale[scale] = data_local
                p_by_scale[scale] = p_cal
                missing_samples += int(m["n_missing_samples"])
                samples += int(m["n_samples"])
            if target == "fusion":
                fparams = fusion_params_from_dict(fr["fusion"]["params"])
                p = _eval_fusion_case(
                    data_by_scale, metas, p_by_scale, cid, fparams)
            else:
                scale = scale_names[0]
                ws, st = SCALE_CFG[scale]
                params = params_from_dict(fr["scales"][scale]["params"])
                p = _eval_single_case(
                    data_by_scale[scale], metas[scale], cid,
                    p_by_scale[scale], params, scale, ws, st)
            per_case[cid] = p
    all_cases = sorted(int(c) for fr in folds for c in fr["test_cases"])
    agg = aggregate_stage3(per_case, all_cases)
    ci = bootstrap_event_metrics(per_case, all_cases, n_boot=500, seed=42)
    ci["point"] = {k: v for k, v in agg.items() if k != "per_case"}
    key = f"{cond['name']}__seed{seed}"
    return key, {
        "condition": cond, "seed": int(seed),
        "aggregate": {k: v for k, v in agg.items() if k != "per_case"},
        "bootstrap": ci,
        "missing_fraction": float(missing_samples / samples
                                  if samples else 0.0),
    }


def run_event_robustness(target: str, jobs: int = 1) -> dict:
    tasks = [(cond, seed) for cond in EVENT_CONDITIONS
             for seed in cond["seeds"]]
    results = {}
    if jobs > 1:
        ctx = __import__("multiprocessing").get_context("spawn")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
            futs = {ex.submit(_run_event_one, target, c, s): (c["name"], s)
                    for c, s in tasks}
            for fut in as_completed(futs):
                key, rec = fut.result()
                results[key] = rec
                a = rec["aggregate"]
                _log(f"{key}: F1={a['event_f1']:.4f} "
                     f"R={a['event_recall']:.4f} "
                     f"FA/h={(a['fa_per_hour_scorable'] or 0):.3f}")
    else:
        for cond, seed in tasks:
            key, rec = _run_event_one(target, cond, seed)
            results[key] = rec
            a = rec["aggregate"]
            _log(f"{key}: F1={a['event_f1']:.4f} "
                 f"R={a['event_recall']:.4f} "
                 f"FA/h={(a['fa_per_hour_scorable'] or 0):.3f}")
    folds = _load_fold_configs()
    all_cases = sorted(int(c) for fr in folds for c in fr["test_cases"])
    out = {"target": target, "n_cases": len(all_cases),
           "n_truth_events": 303, "conditions": results}
    summary = {}
    for cond in EVENT_CONDITIONS:
        keys = [f"{cond['name']}__seed{s}" for s in cond["seeds"]]
        vals = [results[k]["aggregate"] for k in keys]
        summary[cond["name"]] = {
            "n_seeds": len(keys),
            "mean_event_f1": float(np.mean([v["event_f1"] for v in vals])),
            "worst_event_f1": float(np.min([v["event_f1"] for v in vals])),
            "mean_event_recall": float(np.mean(
                [v["event_recall"] for v in vals])),
            "worst_event_recall": float(np.min(
                [v["event_recall"] for v in vals])),
            "mean_fa_per_hour": float(np.mean([
                (v["fa_per_hour_scorable"] or 0.0) for v in vals])),
            "worst_fa_per_hour": float(np.max([
                (v["fa_per_hour_scorable"] or 0.0) for v in vals])),
            "mean_unknown_fraction": float(np.mean([
                v["engine"]["unknown_decision_fraction"] for v in vals])),
        }
    out["summary"] = summary
    return out


def _window_sample(n: int = 10000, seed: int = 42):
    data, meta = build_scale_dataset(15.0, 7.5)
    states = data["state"]
    labels = data["label"]
    rng = np.random.default_rng(seed)
    rows = []
    per_class = n // 3
    chosen = {}
    for lab in (LABEL_CODE["U0"], LABEL_CODE["U1"], LABEL_CODE["U2"]):
        idx = np.where((states == STATE_CODE["VALID"]) &
                       (labels == lab))[0]
        take = min(per_class, len(idx))
        chosen[lab] = idx
        rows.extend(int(x) for x in rng.choice(idx, size=take,
                                               replace=False))
    # U2-VALID is often below n/3. Fill the target n from U0/U1 so the
    # pre-registered 10,000-window sample size is actually met.
    remaining = n - len(rows)
    for lab in (LABEL_CODE["U0"], LABEL_CODE["U1"]):
        idx = chosen[lab]
        already = min(per_class, len(idx))
        can = max(0, len(idx) - already)
        take = min(remaining, can)
        if take:
            pool = np.setdiff1d(idx, np.asarray(rows, dtype=int),
                                assume_unique=False)
            if len(pool):
                take = min(take, len(pool))
                rows.extend(int(x) for x in rng.choice(
                    pool, size=take, replace=False))
                remaining -= take
        if remaining <= 0:
            break
    rows = np.asarray(rows, dtype=int)
    fold_by_case = {}
    for fr in _load_fold_configs():
        for cid in fr["test_cases"]:
            fold_by_case[int(cid)] = int(fr["fold"])
    return data, meta, rows, fold_by_case


def run_window_robustness(sample_n: int = 10000,
                          kinds: set[str] | None = None) -> dict:
    data, meta, rows, fold_by_case = _window_sample(sample_n)
    assets = {fi: _load_asset("U15", fi) for fi in set(fold_by_case.values())}
    # Group by case so only one long waveform is resident at a time.
    by_case = {}
    for r in rows:
        by_case.setdefault(int(data["case_id"][r]), []).append(int(r))
    cond_keys = [(kind, float(level), int(seed))
                 for kind, levels, seeds in WINDOW_CONDITIONS
                 if kinds is None or kind in kinds
                 for level in levels for seed in seeds]
    feat_bank = {f"{k}={l}__seed{s}": [] for k, l, s in cond_keys}
    label_bank = {f"{k}={l}__seed{s}": [] for k, l, s in cond_keys}
    fold_bank = {f"{k}={l}__seed{s}": [] for k, l, s in cond_keys}
    burst_keys = {f"{k}={l}__seed{s}" for k, l, s in cond_keys
                  if k == "burst_drop"}
    burst_audit = {f"{kind}={level}__seed{seed}": {
        "kind": "burst_drop", "level_sec": level, "seed": seed,
        "mode": "window", "n_windows": 0, "n_insufficient": 0,
        "window_start": [], "gap_start": [], "gap_end": [], "gap_length": [],
        "new_missing_samples": [], "new_missing_fraction": [],
    } for kind, level, seed in cond_keys if kind == "burst_drop"}
    inv_set = set(int(x) for x in rows[:5000])
    inv_conditions = {("gain", 0.5, 0), ("gain", 2.0, 0),
                      ("bias", 0.5, 0), ("bias", -0.5, 0)}
    max_delta = 0.0
    n_bad = 0
    n_inv = 0
    for cid, case_rows in by_case.items():
        case = load_case(cid)
        fi = fold_by_case[cid]
        for r in case_rows:
            start = int(data["start"][r])
            lab = int(data["label"][r])
            seg = case["co2"][start:start + int(15 * 100)]
            base = (np.asarray(extract_si_features(seg)["vector"], dtype=float)
                    if r in inv_set else None)
            for kind, level, seed in cond_keys:
                key = f"{kind}={level}__seed{seed}"
                if kind == "burst_drop":
                    y, gap_info = perturb_signal_detail(
                        seg, kind, level, seed, mode="window")
                    rec = burst_audit[key]
                    rec["n_windows"] += 1
                    rec["window_start"].append(start)
                    if gap_info["status"] == "insufficient_window":
                        rec["n_insufficient"] += 1
                        rec["gap_start"].append(-1)
                        rec["gap_end"].append(-1)
                        rec["gap_length"].append(0)
                        rec["new_missing_samples"].append(0)
                        rec["new_missing_fraction"].append(0.0)
                    else:
                        g = gap_info["gaps"][0]
                        rec["gap_start"].append(start + int(g["start"]))
                        rec["gap_end"].append(start + int(g["end"]))
                        rec["gap_length"].append(int(g["length"]))
                        rec["new_missing_samples"].append(
                            int(gap_info["new_missing_samples"]))
                        rec["new_missing_fraction"].append(
                            float(gap_info["new_missing_fraction"]))
                else:
                    y = perturb_signal(seg, kind, level, seed)
                feat = extract_si_features(y)
                vec = np.nan_to_num(np.asarray(feat["vector"], dtype=float),
                                    nan=0.0, posinf=0.0, neginf=0.0)
                vec = np.clip(vec, -np.finfo(np.float32).max,
                              np.finfo(np.float32).max)
                feat_bank[key].append(vec.astype(np.float32))
                label_bank[key].append(lab)
                fold_bank[key].append(fi)
                if base is not None and (kind, level, seed) in inv_conditions:
                    d = float(np.max(np.abs(vec - base)))
                    max_delta = max(max_delta, d)
                    n_bad += int(d > 1e-6)
                    n_inv += 1
    out = {"scale": "U15", "n_windows": int(len(rows)),
           "note": "quality fixed at clean VALID; window-level feature/model stability only",
           "conditions": {}, "invariance": {},
           "burst_drop_audit": burst_audit}
    # Batch inference per fold; single-row sklearn calls dominated runtime.
    for kind, level, seed in cond_keys:
        key = f"{kind}={level}__seed{seed}"
        X = np.asarray(feat_bank[key], dtype=np.float32)
        y_true = np.asarray(label_bank[key], dtype=np.int64)
        fold = np.asarray(fold_bank[key], dtype=np.int64)
        pred = np.full(len(y_true), -1, dtype=np.int64)
        for fi in np.unique(fold):
            idx = np.flatnonzero(fold == fi)
            model, cal = assets[int(fi)]
            p = apply_calibrator(cal, predict_full(model, X[idx]))
            pred[idx] = np.argmax(p, axis=1)
        cnt = np.zeros((3, 5), dtype=np.int64)
        for lab, pr in zip(y_true.tolist(), pred.tolist()):
            cnt[int(lab), int(pr) + 1] += 1
        out["conditions"][key] = {
            "kind": kind, "level": float(level), "seed": int(seed),
            "metrics": metrics_from_window_confusion(cnt),
        }
    out["invariance"] = {
        "n_windows": int(len(inv_set)), "n_checks": int(n_inv),
        "max_abs_feature_delta": float(max_delta),
        "fraction_above_1e-6": float(n_bad / n_inv if n_inv else 0.0),
        "note": "positive gain/bias; quality/coverage separately in event stress",
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", action="store_true")
    ap.add_argument("--window", action="store_true")
    ap.add_argument("--burst-only", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.burst_only:
        wi = run_window_robustness(kinds={"burst_drop"})
        path = OUT / "robustness_window.json"
        if path.exists():
            old = json.load(open(path, encoding="utf-8"))
        else:
            old = {"scale": "U15", "n_windows": wi["n_windows"],
                   "note": wi["note"], "conditions": {}, "invariance": {}}
        old["conditions"].update(wi["conditions"])
        # Keep the large per-window audit in its own file, not duplicated
        # inside the summary results.
        old.pop("burst_drop_audit", None)
        json.dump(old, open(path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        with open(OUT / "burst_drop_audit.json", "w", encoding="utf-8") as f:
            json.dump({"n_windows": wi["n_windows"],
                       "conditions": wi["burst_drop_audit"]},
                      f, ensure_ascii=False, indent=2)
        return 0
    comp = _load_comparison()
    target = "fusion" if comp["benefit"].get("pass") else "best_single"
    _log(f"robustness target from frozen benefit rule: {target}")
    if args.event or not args.window:
        ev = run_event_robustness(target, jobs=args.jobs)
        with open(OUT / "robustness_event.json", "w", encoding="utf-8") as f:
            json.dump(ev, f, ensure_ascii=False, indent=2)
    if args.window or not args.event:
        wi = run_window_robustness()
        audit = wi.pop("burst_drop_audit", {})
        with open(OUT / "robustness_window.json", "w", encoding="utf-8") as f:
            json.dump(wi, f, ensure_ascii=False, indent=2)
        with open(OUT / "burst_drop_audit.json", "w", encoding="utf-8") as f:
            json.dump({"n_windows": wi["n_windows"], "conditions": audit},
                      f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
