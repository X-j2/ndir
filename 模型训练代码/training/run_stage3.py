# -*- coding: utf-8 -*-
"""Stage 3 nested evaluation and parameter selection for U15/U30 engines.

No Final Test waveform/label is loaded.  Each outer fold trains its own model
and calibrator on outer-training cases; event-engine parameters are selected
on inner OOF of those same outer-training cases.  The outer test fold is only
applied, never used for selection.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from training.stage2_data import build_scale_dataset, training_mask  # noqa: E402
from training.stage2_calibration import (  # noqa: E402
    apply_calibrator, select_calibrator_crossfit,
)
from training.stage2_eval import (  # noqa: E402
    bootstrap_event_metrics,
)
from training.run_stage2 import (  # noqa: E402
    CALIB_METHODS, WEIGHTS, case_folds, class_weight_score, concat_rows,
    label_ints, predict_full, rows_by_case, train_rf,
)
from training.stage3_eval import (  # noqa: E402
    aggregate_stage3, evaluate_params, select_params,
)
from training.stage3_event_engine import params_from_dict  # noqa: E402

OUT = ROOT / "report" / "stage3"
MODEL_DIR = ROOT / "models" / "stage3"


def log(msg: str = "") -> None:
    print(msg, flush=True)


def _strip(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if k != "per_case"}


def run_scale(scale: str, window_sec: float, step_sec: float,
              jobs: int, rebuild: bool) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data, meta = build_scale_dataset(window_sec, step_sec, jobs=jobs,
                                     rebuild=rebuild)
    rows = rows_by_case(data)
    train_mask = training_mask(data)
    all_cases = sorted(int(c) for c in meta["case_meta"])
    folds = case_folds(meta, all_cases, 5, seed=42)
    n_total = len(data["label"])
    p_cal_oof = np.zeros((n_total, 3), dtype=np.float32)
    fold_by_row = np.full(n_total, -1, dtype=np.int16)
    per_case_all = {}
    fold_results = []
    log(f"Stage3 {scale}: {len(all_cases)} cases, {n_total} windows; "
        f"fingerprint={meta['dataset_fingerprint'][:16]}")
    for fi, (tr_cases, te_cases) in enumerate(folds):
        tf = time.time()
        inner_folds = case_folds(meta, tr_cases, 3, seed=42 + fi)
        p_inner = {name: np.zeros((n_total, 3), dtype=np.float32)
                   for name, _ in WEIGHTS}
        inner_fold = np.full(n_total, -1, dtype=np.int16)
        for ii, (itr_cases, ival_cases) in enumerate(inner_folds):
            itr = concat_rows(itr_cases, rows)
            ival = concat_rows(ival_cases, rows)
            fit = itr[train_mask[itr]]
            ytr = label_ints(data, fit)
            for name, w in WEIGHTS:
                clf = train_rf(data["features"][fit], ytr, w)
                if len(ival):
                    p_inner[name][ival] = predict_full(
                        clf, data["features"][ival])
            inner_fold[ival] = ii
        inner_rows = concat_rows(tr_cases, rows)
        elig = inner_rows[train_mask[inner_rows] &
                          (inner_fold[inner_rows] >= 0)]
        y_elig = label_ints(data, elig)
        weight_scores = {
            name: class_weight_score(data, elig, y_elig,
                                     p_inner[name][elig])
            for name, _ in WEIGHTS
        }
        best_weight = max(
            WEIGHTS, key=lambda nw: (weight_scores[nw[0]],
                                     -WEIGHTS.index(nw)))[0]
        p_sel = p_inner[best_weight]
        cal = select_calibrator_crossfit(
            p_sel[elig], y_elig, inner_fold[elig],
            methods=CALIB_METHODS)
        p_cal_inner = apply_calibrator(cal["calibrator"], p_sel)
        sel = select_params(data, meta, rows, p_cal_inner, tr_cases, scale)
        best_params = sel["best"]["params"]
        inner_active = _strip(sel["best"]["aggregate"])

        outer_train = concat_rows(tr_cases, rows)
        outer_test = concat_rows(te_cases, rows)
        outer_fit = outer_train[train_mask[outer_train]]
        clf = train_rf(data["features"][outer_fit],
                       label_ints(data, outer_fit),
                       dict(WEIGHTS)[best_weight])
        p_raw_test = predict_full(clf, data["features"][outer_test])
        p_cal_test = apply_calibrator(cal["calibrator"], p_raw_test)
        p_cal_global = np.zeros((n_total, 3), dtype=np.float32)
        p_cal_global[outer_test] = p_cal_test
        ev = evaluate_params(data, meta, rows, p_cal_global, te_cases,
                             params_from_dict(best_params), scale)
        per_case_all.update(ev["per_case"])
        p_cal_oof[outer_test] = p_cal_test
        fold_by_row[outer_test] = fi
        fold_results.append({
            "fold": fi,
            "train_cases": [int(c) for c in tr_cases],
            "test_cases": [int(c) for c in te_cases],
            "train_events": int(sum(len(meta["case_meta"][str(c)][
                "reliable_u2_events"]) for c in tr_cases)),
            "test_events": int(sum(len(meta["case_meta"][str(c)][
                "reliable_u2_events"]) for c in te_cases)),
            "weight": best_weight,
            "weight_scores": {
                k: [float(x) for x in v] for k, v in weight_scores.items()},
            "calibrator_method": cal["method"],
            "calibrator_cv_log_loss": cal["cv_log_loss"],
            "params": best_params,
            "inner_grid_size": sel["grid_size"],
            "inner_top": sel["top"],
            "inner_active": _strip(inner_active),
            "outer_active": _strip(ev["aggregate"]),
            "elapsed_sec": float(time.time() - tf),
        })
        log(f"  fold {fi+1}/5: weight={best_weight} cal={cal['method']} "
            f"params={best_params} "
            f"innerF1={inner_active['event_f1']:.3f} "
            f"outerF1={ev['aggregate']['event_f1']:.3f} "
            f"outerFA/h={(ev['aggregate']['fa_per_hour_scorable'] or 0):.3f} "
            f"({time.time()-tf:.1f}s)")

    active = aggregate_stage3(per_case_all, all_cases)
    active_ci = bootstrap_event_metrics(per_case_all, all_cases,
                                        n_boot=1000, seed=42)
    active_ci["point"] = _strip(active_ci["point"])
    # Full-development OOF parameters for the host demo only.  These are not
    # used to report validation performance.
    final = select_params(data, meta, rows, p_cal_oof, all_cases, scale)
    final_params = final["best"]["params"]
    params_path = MODEL_DIR / f"{scale.lower()}_event_params.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump({
            "version": "stage3_event_params_v1",
            "scale": scale,
            "window_sec": float(window_sec),
            "step_sec": float(step_sec),
            "params": final_params,
            "dataset_fingerprint": meta["dataset_fingerprint"],
            "selection": "full development OOF; host demo only",
            "grid_size": final["grid_size"],
            "top": final["top"],
        }, f, ensure_ascii=False, indent=2)
    stage2_path = ROOT / "report" / "stage2" / f"{scale.lower()}_results.json"
    stage2_ref = None
    if stage2_path.exists():
        with open(stage2_path, encoding="utf-8") as f:
            s2 = json.load(f)
        stage2_ref = {
            "outer_event": s2.get("outer_event"),
            "outer_event_bootstrap": s2.get("outer_event_bootstrap"),
        }
    result = {
        "stage": 3,
        "scale": scale,
        "window_sec": float(window_sec),
        "step_sec": float(step_sec),
        "dataset_fingerprint": meta["dataset_fingerprint"],
        "n_cases": len(all_cases),
        "n_truth_events": int(active["n_truth_events"]),
        "n_low_support_events": int(active["n_low_support_events"]),
        "folds": fold_results,
        "active": _strip(active),
        "active_bootstrap": active_ci,
        "final_params": final_params,
        "final_params_path": str(params_path.relative_to(ROOT)),
        "stage2_reference": stage2_ref,
        "selection": ("outer folds: model/calibrator/engine params selected "
                      "inside outer training; final params on full dev OOF "
                      "only for host demo"),
        "final_test": "sealed; only case-id disjoint assertion",
        "elapsed_sec": float(time.time() - t0),
    }
    with open(OUT / f"{scale.lower()}_results.json", "w",
              encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log(f"  {scale} done: active F1={active['event_f1']:.4f}, "
        f"recall={active['event_recall']:.4f}, "
        f"FA/h={(active['fa_per_hour_scorable'] or 0):.3f}; "
        f"final params -> {params_path.name}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=("U15", "U30", "both"), default="both")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    scales = (("U15", 15.0, 7.5), ("U30", 30.0, 15.0))
    for scale, ws, step in scales:
        if args.scale in (scale, "both"):
            run_scale(scale, ws, step, args.jobs, args.rebuild)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
