# -*- coding: utf-8 -*-
"""Stage 4: U15/U30/U60/U90 single-scale evaluation and causal fusion.

All selection is inside each outer training fold:
  - class weight, calibrator, single-scale event params;
  - fusion weighting/decay/min-active-sources and fused event params;
  - best single scale.

The outer test fold is only applied.  Final Test is never loaded.  B0 and all
methods use the frozen 303 reliable / 60 low-support events and scorable
domain from Stage 2 v2.
"""
from __future__ import annotations

import argparse
import hashlib
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
    aggregate_events, bootstrap_event_metrics, evaluate_events_for_cases,
)
from training.run_stage2 import (  # noqa: E402
    CALIB_METHODS, WEIGHTS, case_folds, class_weight_score, concat_rows,
    label_ints, predict_full, rows_by_case, train_rf,
)
from training.stage3_eval import (  # noqa: E402
    aggregate_stage3, evaluate_params, select_params,
)
from training.stage3_event_engine import params_from_dict  # noqa: E402
from training.stage4_fusion import (  # noqa: E402
    FusionParams, build_series_by_case, evaluate_fusion, select_fusion_params,
)
from features.si_features import FEATURE_NAMES  # noqa: E402

SCALES = (("U15", 15.0, 7.5), ("U30", 30.0, 15.0),
          ("U60", 60.0, 30.0), ("U90", 90.0, 45.0))
OUT = ROOT / "report" / "stage4"
MODEL_DIR = ROOT / "models" / "stage4"
FOLD_DIR = MODEL_DIR / "folds"


def log(msg: str = "") -> None:
    print(msg, flush=True)


def _strip(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if k != "per_case"}


def _inner_folds(meta: dict, train_cases: list[int], seed: int):
    return case_folds(meta, train_cases, 3, seed=seed)


def _fit_scale_fold(scale: str, window_sec: float, step_sec: float,
                    data: dict, meta: dict, rows: dict, train_mask: np.ndarray,
                    train_cases: list[int], test_cases: list[int],
                    inner_folds, outer_fold: int) -> dict:
    t0 = time.time()
    n_total = len(data["label"])
    p_inner = {name: np.zeros((n_total, 3), dtype=np.float32)
               for name, _ in WEIGHTS}
    inner_fold_by_row = np.full(n_total, -1, dtype=np.int16)
    for ii, (itr_cases, ival_cases) in enumerate(inner_folds):
        itr = concat_rows(itr_cases, rows)
        ival = concat_rows(ival_cases, rows)
        fit = itr[train_mask[itr]]
        if not len(fit) or not len(ival):
            continue
        ytr = label_ints(data, fit)
        for name, w in WEIGHTS:
            clf = train_rf(data["features"][fit], ytr, w)
            p_inner[name][ival] = predict_full(clf, data["features"][ival])
        inner_fold_by_row[ival] = ii
    inner_rows = concat_rows(train_cases, rows)
    elig = inner_rows[train_mask[inner_rows] &
                      (inner_fold_by_row[inner_rows] >= 0)]
    if not len(elig):
        raise RuntimeError(f"{scale}: empty inner OOF rows")
    y_elig = label_ints(data, elig)
    weight_scores = {
        name: class_weight_score(data, elig, y_elig, p_inner[name][elig])
        for name, _ in WEIGHTS
    }
    best_weight = max(
        WEIGHTS, key=lambda nw: (weight_scores[nw[0]],
                                 -WEIGHTS.index(nw)))[0]
    p_sel = p_inner[best_weight]
    cal = select_calibrator_crossfit(
        p_sel[elig], y_elig, inner_fold_by_row[elig],
        methods=CALIB_METHODS)
    p_cal_inner = apply_calibrator(cal["calibrator"], p_sel)
    sel = select_params(data, meta, rows, p_cal_inner, train_cases, scale)
    best_params = params_from_dict(sel["best"]["params"])

    outer_train = concat_rows(train_cases, rows)
    outer_test = concat_rows(test_cases, rows)
    outer_fit = outer_train[train_mask[outer_train]]
    clf = train_rf(data["features"][outer_fit],
                   label_ints(data, outer_fit),
                   dict(WEIGHTS)[best_weight])
    p_raw_test = predict_full(clf, data["features"][outer_test])
    p_cal_test = apply_calibrator(cal["calibrator"], p_raw_test)
    p_cal_global = np.zeros((n_total, 3), dtype=np.float32)
    p_cal_global[outer_test] = p_cal_test
    ev = evaluate_params(data, meta, rows, p_cal_global, test_cases,
                         best_params, scale)

    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    model_path = FOLD_DIR / f"f{outer_fold}_{scale.lower()}_rf.joblib"
    import joblib
    joblib.dump({
        "model": clf,
        "feature_names": list(FEATURE_NAMES),
        "window_sec": float(window_sec),
        "classes": [0, 1, 2],
        "class_weight_name": best_weight,
    }, model_path)
    cal_path = FOLD_DIR / f"f{outer_fold}_{scale.lower()}_calibrator.json"
    with open(cal_path, "w", encoding="utf-8") as f:
        json.dump(cal["calibrator"], f, ensure_ascii=False, indent=2)
    return {
        "scale": scale,
        "window_sec": float(window_sec),
        "step_sec": float(step_sec),
        "weight": best_weight,
        "weight_scores": {k: [float(x) for x in v]
                          for k, v in weight_scores.items()},
        "calibrator_method": cal["method"],
        "calibrator": cal["calibrator"],
        "params": best_params.as_dict(),
        "inner": _strip(sel["best"]["aggregate"]),
        "outer": _strip(ev["aggregate"]),
        "per_case": ev["per_case"],
        "p_cal_inner": p_cal_inner,
        "p_cal_test": p_cal_test,
        "outer_test_rows": outer_test,
        "model_path": str(model_path.relative_to(ROOT)),
        "calibrator_path": str(cal_path.relative_to(ROOT)),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "elapsed_sec": float(time.time() - t0),
    }


def _best_scale(inner_by_scale: dict) -> str:
    order = {name: i for i, (name, _, _) in enumerate(SCALES)}

    def key(name):
        a = inner_by_scale[name]
        return (float(a["event_f1"]), float(a["event_recall"]),
                -(float(a["fa_per_hour_scorable"] or 0.0)),
                -order[name])

    return max(inner_by_scale, key=key)


def _series_payload(scale_data: dict, scale_meta: dict,
                    scale_rows: dict, p_by_scale: dict,
                    case_ids: list[int]) -> dict:
    return build_series_by_case(scale_data, scale_meta, scale_rows,
                                p_by_scale, case_ids)


def _paired_diff(per_a: dict, per_b: dict, cases: list[int],
                 n_boot: int = 1000, seed: int = 42) -> dict:
    a0 = aggregate_events(per_a, cases)
    b0 = aggregate_events(per_b, cases)
    keys = ("event_recall", "event_ppv", "event_f1",
            "fa_per_hour_scorable", "alarm_duration_fraction")
    point = {}
    for k in keys:
        va, vb = a0.get(k), b0.get(k)
        point[k] = (None if va is None or vb is None
                    else float(va) - float(vb))
    rng = np.random.default_rng(seed)
    cids = np.asarray([int(c) for c in cases], dtype=int)
    vals = {k: [] for k in keys}
    for _ in range(n_boot):
        sample = rng.choice(cids, size=len(cids), replace=True)
        aa = aggregate_events(per_a, sample.tolist())
        bb = aggregate_events(per_b, sample.tolist())
        for k in keys:
            va, vb = aa.get(k), bb.get(k)
            if va is not None and vb is not None:
                vals[k].append(float(va) - float(vb))

    def ci(xs):
        return ([float(np.percentile(xs, 2.5)),
                 float(np.percentile(xs, 97.5))] if xs else None)

    return {"point": point, "ci95": {k: ci(vals[k]) for k in keys},
            "n_boot": int(n_boot), "seed": int(seed)}


def _benefit(comparison: dict) -> dict:
    f1 = comparison["point"]["event_f1"]
    fa = comparison["point"]["fa_per_hour_scorable"]
    rec = comparison["point"]["event_recall"]
    f1_ci = comparison["ci95"]["event_f1"]
    fa_ratio = None
    # Point differences are enough for the FA rule; ratio is computed by caller.
    criteria = {
        "f1_delta_ge_0_05": bool(f1 is not None and f1 >= 0.05),
        "recall_delta_ge_minus_0_05": bool(rec is not None and rec >= -0.05),
        "f1_ci_lower_gt_zero": bool(f1_ci is not None and f1_ci[0] > 0.0),
    }
    return {"criteria": criteria, "f1_delta": f1, "recall_delta": rec,
            "fa_delta": fa,
            "pass_before_fa_ratio": bool(all(criteria.values()))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    data_by_scale, meta_by_scale, rows_by_scale, mask_by_scale = {}, {}, {}, {}
    for scale, ws, st in SCALES:
        log(f"loading {scale} {ws:.0f}s/{st:.1f}s")
        data, meta = build_scale_dataset(ws, st, jobs=args.jobs,
                                         rebuild=args.rebuild)
        data_by_scale[scale] = data
        meta_by_scale[scale] = meta
        rows_by_scale[scale] = rows_by_case(data)
        mask_by_scale[scale] = training_mask(data)
    meta15 = meta_by_scale["U15"]
    all_cases = sorted(int(c) for c in meta15["case_meta"])
    folds = case_folds(meta15, all_cases, 5, seed=42)
    p_cal_oof = {s: np.zeros((len(data_by_scale[s]["label"]), 3),
                             dtype=np.float32) for s, _, _ in SCALES}
    scale_per_case = {s: {} for s, _, _ in SCALES}
    best_per_case = {}
    fusion_per_case = {}
    fold_results = []

    for fi, (tr_cases, te_cases) in enumerate(folds):
        tf = time.time()
        log(f"outer fold {fi+1}/5: train={len(tr_cases)} "
            f"test={len(te_cases)} cases")
        inner_folds = _inner_folds(meta15, tr_cases, seed=42 + fi)
        scale_fold = {}
        inner_by_scale = {}
        for scale, ws, st in SCALES:
            r = _fit_scale_fold(
                scale, ws, st, data_by_scale[scale], meta_by_scale[scale],
                rows_by_scale[scale], mask_by_scale[scale], tr_cases,
                te_cases, inner_folds, fi)
            scale_fold[scale] = r
            inner_by_scale[scale] = r["inner"]
            scale_per_case[scale].update(r["per_case"])
            p_cal_oof[scale][r["outer_test_rows"]] = r["p_cal_test"]
            log(f"  {scale}: weight={r['weight']} cal={r['calibrator_method']} "
                f"params={r['params']} innerF1={r['inner']['event_f1']:.3f} "
                f"outerF1={r['outer']['event_f1']:.3f} "
                f"FA/h={(r['outer']['fa_per_hour_scorable'] or 0):.3f} "
                f"({r['elapsed_sec']:.1f}s)")
        best_scale = _best_scale(inner_by_scale)
        best_per_case.update(scale_fold[best_scale]["per_case"])
        log(f"  inner-selected best single: {best_scale} "
            f"(inner F1={inner_by_scale[best_scale]['event_f1']:.3f})")

        # Fusion is selected only on the outer-training inner OOF.
        inner_series = _series_payload(
            data_by_scale, meta_by_scale, rows_by_scale,
            {s: scale_fold[s]["p_cal_inner"] for s, _, _ in SCALES},
            tr_cases)
        fsel = select_fusion_params(inner_series, meta15, tr_cases)
        fparams = fsel["params"]
        outer_series = _series_payload(
            data_by_scale, meta_by_scale, rows_by_scale,
            {s: _full_test_p(scale_fold[s], p_cal_oof[s], len(
                data_by_scale[s]["label"])) for s, _, _ in SCALES},
            te_cases)
        fev = evaluate_fusion(outer_series, meta15, te_cases, fparams)
        fusion_per_case.update(fev["per_case"])
        log(f"  fusion: {fparams.weight_profile} tau={fparams.tau_sec} "
            f"min_active={fparams.min_active_sources} "
            f"innerF1={fsel['aggregate']['event_f1']:.3f} "
            f"outerF1={fev['aggregate']['event_f1']:.3f} "
            f"FA/h={(fev['aggregate']['fa_per_hour_scorable'] or 0):.3f}")

        fold_results.append({
            "fold": fi,
            "train_cases": [int(c) for c in tr_cases],
            "test_cases": [int(c) for c in te_cases],
            "scales": {
                s: {
                    "weight": scale_fold[s]["weight"],
                    "calibrator_method": scale_fold[s]["calibrator_method"],
                    "params": scale_fold[s]["params"],
                    "inner": scale_fold[s]["inner"],
                    "outer": scale_fold[s]["outer"],
                    "model_path": scale_fold[s]["model_path"],
                    "calibrator_path": scale_fold[s]["calibrator_path"],
                } for s, _, _ in SCALES
            },
            "inner_best_scale": best_scale,
            "fusion": {
                "params": fparams.as_dict(),
                "inner": _strip(fsel["aggregate"]),
                "outer": _strip(fev["aggregate"]),
                "weighting_top": fsel["weighting_top"],
                "event_top": fsel["event_top"],
            },
            "elapsed_sec": float(time.time() - tf),
        })
        log(f"  fold done in {time.time()-tf:.1f}s")

    # Outer OOF aggregates and CIs.
    scale_results = {}
    for scale, ws, st in SCALES:
        per = scale_per_case[scale]
        agg = aggregate_stage3(per, all_cases)
        ci = bootstrap_event_metrics(per, all_cases, n_boot=1000, seed=42)
        ci["point"] = _strip(agg)
        scale_results[scale] = {
            "scale": scale, "window_sec": ws, "step_sec": st,
            "outer": _strip(agg), "outer_ci": ci,
            "folds": [f["scales"][scale] for f in fold_results],
        }
        log(f"{scale} outer OOF: F1={agg['event_f1']:.4f} "
            f"R={agg['event_recall']:.4f} PPV={agg['event_ppv']:.4f} "
            f"FA/h={(agg['fa_per_hour_scorable'] or 0):.3f}")

    best_agg = aggregate_stage3(best_per_case, all_cases)
    best_ci = bootstrap_event_metrics(best_per_case, all_cases,
                                      n_boot=1000, seed=42)
    best_ci["point"] = _strip(best_agg)
    fusion_agg = aggregate_stage3(fusion_per_case, all_cases)
    fusion_ci = bootstrap_event_metrics(fusion_per_case, all_cases,
                                        n_boot=1000, seed=42)
    fusion_ci["point"] = _strip(fusion_agg)
    log(f"best single outer: F1={best_agg['event_f1']:.4f} "
        f"R={best_agg['event_recall']:.4f} PPV={best_agg['event_ppv']:.4f} "
        f"FA/h={(best_agg['fa_per_hour_scorable'] or 0):.3f}")
    log(f"fusion outer: F1={fusion_agg['event_f1']:.4f} "
        f"R={fusion_agg['event_recall']:.4f} PPV={fusion_agg['event_ppv']:.4f} "
        f"FA/h={(fusion_agg['fa_per_hour_scorable'] or 0):.3f}")

    # Same-cohort B0 raw proxies, one per scale.
    b0_results = {}
    for scale, _, _ in SCALES:
        per = evaluate_events_for_cases(
            data_by_scale[scale], meta_by_scale[scale],
            rows_by_scale[scale], all_cases, policy="b0")
        agg = aggregate_events(per, all_cases)
        ci = bootstrap_event_metrics(per, all_cases, n_boot=1000, seed=42)
        ci["point"] = _strip(agg)
        b0_results[scale] = {"outer": _strip(agg), "outer_ci": ci,
                             "per_case": per}

    comparisons = {}
    pairs = {
        "fusion_vs_best_single": (fusion_per_case, best_per_case),
        "fusion_vs_u15": (fusion_per_case, scale_per_case["U15"]),
        "fusion_vs_u30": (fusion_per_case, scale_per_case["U30"]),
        "best_single_vs_u15": (best_per_case, scale_per_case["U15"]),
        "best_single_vs_u30": (best_per_case, scale_per_case["U30"]),
        "u15_vs_u30": (scale_per_case["U15"], scale_per_case["U30"]),
    }
    for name, (a, b) in pairs.items():
        comparisons[name] = _paired_diff(a, b, all_cases)
    benefit = _benefit(comparisons["fusion_vs_best_single"])
    best_fa = best_agg.get("fa_per_hour_scorable")
    fus_fa = fusion_agg.get("fa_per_hour_scorable")
    fa_ratio = (float(fus_fa) / float(best_fa)
                if best_fa not in (None, 0) and fus_fa is not None else None)
    benefit["fa_ratio"] = fa_ratio
    benefit["criteria"]["fa_ratio_le_1_10"] = bool(
        fa_ratio is not None and fa_ratio <= 1.10)
    benefit["pass"] = bool(benefit["pass_before_fa_ratio"] and
                           benefit["criteria"]["fa_ratio_le_1_10"])
    benefit["rule"] = ("F1 +>=5.0 pp AND FA/h <=1.10x best AND "
                       "recall >= -5.0 pp AND paired F1 CI lower >0")

    # Final dev-only configs for the host experiment (not used for reporting).
    final = {}
    for scale, ws, st in SCALES:
        fp = select_params(data_by_scale[scale], meta_by_scale[scale],
                           rows_by_scale[scale], p_cal_oof[scale],
                           all_cases, scale)
        final[scale] = fp["best"]["params"]
        with open(MODEL_DIR / f"{scale.lower()}_event_params.json", "w",
                  encoding="utf-8") as f:
            json.dump({
                "version": "stage4_event_params_v1",
                "scale": scale, "window_sec": ws, "step_sec": st,
                "params": final[scale],
                "dataset_fingerprint": meta_by_scale[scale][
                    "dataset_fingerprint"],
                "selection": "full development OOF; host experiment only",
                "grid_size": fp["grid_size"],
                "top": fp["top"],
            }, f, ensure_ascii=False, indent=2)
    full_series = _series_payload(
        data_by_scale, meta_by_scale, rows_by_scale,
        {s: p_cal_oof[s] for s, _, _ in SCALES}, all_cases)
    final_fusion = select_fusion_params(full_series, meta15, all_cases)
    with open(MODEL_DIR / "fusion_event_params.json", "w",
              encoding="utf-8") as f:
        json.dump({
            "version": "stage4_fusion_params_v1",
            "experimental": True,
            "params": final_fusion["params"].as_dict(),
            "selection": "full development OOF; host experiment only",
            "benefit_passed": bool(benefit["pass"]),
            "weighting_top": final_fusion["weighting_top"],
            "event_top": final_fusion["event_top"],
        }, f, ensure_ascii=False, indent=2)

    # Fold configuration for robustness replay.
    fold_configs = []
    for fr in fold_results:
        fold_configs.append({
            "fold": fr["fold"],
            "test_cases": fr["test_cases"],
            "best_scale": fr["inner_best_scale"],
            "scales": fr["scales"],
            "fusion": {"params": fr["fusion"]["params"]},
        })
    with open(MODEL_DIR / "fold_configs.json", "w", encoding="utf-8") as f:
        json.dump({"folds": fold_configs}, f, ensure_ascii=False, indent=2)

    for scale in scale_results:
        with open(OUT / f"{scale.lower()}_results.json", "w",
                  encoding="utf-8") as f:
            json.dump({
                "stage": 4, **scale_results[scale],
                "n_truth_events": int(scale_results[scale]["outer"][
                    "n_truth_events"]),
                "final_params": final[scale],
                "final_test": "sealed; only case-id disjoint assertion",
            }, f, ensure_ascii=False, indent=2)
    with open(OUT / "b0_results.json", "w", encoding="utf-8") as f:
        json.dump({s: {"outer": r["outer"], "outer_ci": r["outer_ci"]}
                   for s, r in b0_results.items()},
                  f, ensure_ascii=False, indent=2)
    with open(OUT / "best_single_results.json", "w", encoding="utf-8") as f:
        json.dump({"outer": _strip(best_agg), "outer_ci": best_ci,
                   "folds": [{"fold": f["fold"],
                              "best_scale": f["inner_best_scale"]}
                             for f in fold_results]},
                  f, ensure_ascii=False, indent=2)
    with open(OUT / "fusion_results.json", "w", encoding="utf-8") as f:
        json.dump({"outer": _strip(fusion_agg), "outer_ci": fusion_ci,
                   "folds": [{"fold": f["fold"], "fusion": f["fusion"]}
                             for f in fold_results],
                   "final_params": final_fusion["params"].as_dict(),
                   "experimental": True}, f, ensure_ascii=False, indent=2)
    with open(OUT / "comparison.json", "w", encoding="utf-8") as f:
        json.dump({
            "methods": {
                "stage3_u15": _strip(scale_results["U15"]["outer"]),
                "stage3_u30": _strip(scale_results["U30"]["outer"]),
                "single_u60": _strip(scale_results["U60"]["outer"]),
                "single_u90": _strip(scale_results["U90"]["outer"]),
                "best_single": _strip(best_agg),
                "fusion": _strip(fusion_agg),
                "b0_u15": b0_results["U15"]["outer"],
                "b0_u30": b0_results["U30"]["outer"],
                "b0_u60": b0_results["U60"]["outer"],
                "b0_u90": b0_results["U90"]["outer"],
            },
            "paired_differences": comparisons,
            "benefit": benefit,
            "n_cases": len(all_cases),
            "n_truth_events": 303,
            "n_low_support_events": 60,
        }, f, ensure_ascii=False, indent=2)
    with open(OUT / "fold_selection.json", "w", encoding="utf-8") as f:
        json.dump(fold_results, f, ensure_ascii=False, indent=2)
    with open(OUT / "run.log", "w", encoding="utf-8") as f:
        f.write(f"Stage 4 complete in {time.time()-t_all:.1f}s\n")
    log(f"Stage 4 complete in {time.time()-t_all:.1f}s")
    log(f"benefit passed={benefit['pass']} criteria={benefit['criteria']}")
    return 0


def _full_test_p(fold_scale: dict, p_cal_oof: np.ndarray, n_total: int
                 ) -> np.ndarray:
    return p_cal_oof


if __name__ == "__main__":
    raise SystemExit(main())
