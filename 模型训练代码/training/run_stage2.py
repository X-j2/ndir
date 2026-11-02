# -*- coding: utf-8 -*-
"""Stage 2: U15/U30 RF, probability calibration, window and event evaluation.

Scope and anti-leakage rules:
  - only ``cases_200.json`` is loaded as waveforms;
  - ``final_test_seed42.json`` is read only for a disjoint-case assertion;
  - outer folds and inner folds are case-level (no window crosses a split);
  - class weight, calibrator and event threshold are selected inside each
    outer training fold using an inner case-level CV;
  - calibration metrics are computed from cross-fitted outer predictions;
  - event metrics are a single-scale raw-alarm proxy, not Stage 3 ACTIVE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from training.stage2_data import (  # noqa: E402
    build_scale_dataset, decode_labels, decode_states, event_eval_mask,
    model_inference_mask, training_mask,
)
from training.stage2_calibration import (  # noqa: E402
    apply_calibrator, calibration_metrics, fit_calibrator,
    select_calibrator_crossfit,
)
from training.stage2_eval import (  # noqa: E402
    aggregate_events, bootstrap_event_metrics, bootstrap_window_metrics,
    b0_predict, evaluate_events_for_cases, select_threshold,
    window_confusion, metrics_from_window_confusion,
)
from features.quality_states import UNKNOWN_STATES  # noqa: E402
from features.si_features import FEATURE_NAMES  # noqa: E402

LABEL_TO_INT = {"U0": 0, "U1": 1, "U2": 2, "U3": 3}
TRAIN_CLASSES = (0, 1, 2)
WEIGHTS = (
    ("none", None),
    ("u2x3", {0: 1.0, 1: 1.0, 2: 3.0}),
    ("u2x10", {0: 1.0, 1: 1.0, 2: 10.0}),
    ("u2x30", {0: 1.0, 1: 1.0, 2: 30.0}),
    ("balanced", "balanced"),
)
CALIB_METHODS = ("none", "temperature", "multinomial_logistic")
OUT_DIR = ROOT / "report" / "stage2"
MODEL_DIR = ROOT / "models" / "stage2"


def log(msg: str = "") -> None:
    print(msg, flush=True)


def rows_by_case(data: dict) -> dict[int, np.ndarray]:
    out: dict[int, list[int]] = {}
    for i, cid in enumerate(data["case_id"].tolist()):
        out.setdefault(int(cid), []).append(i)
    return {k: np.asarray(v, dtype=np.int64) for k, v in out.items()}


def concat_rows(cases: list[int], rows: dict[int, np.ndarray]) -> np.ndarray:
    parts = [rows[c] for c in cases if c in rows and len(rows[c])]
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(parts)


def label_ints(data: dict, rows: np.ndarray) -> np.ndarray:
    labels = decode_labels(data["label"][rows])
    return np.asarray([LABEL_TO_INT.get(str(x), -999) for x in labels],
                      dtype=np.int64)


def event_strata(meta: dict, case_ids: list[int]) -> np.ndarray:
    out = []
    for cid in case_ids:
        cm = meta["case_meta"][str(int(cid))]
        n = len(cm.get("reliable_u2_events", []))
        out.append(min(n, 3))
    return np.asarray(out, dtype=int)


def case_folds(meta: dict, case_ids: list[int], n_splits: int,
               seed: int = 42) -> list[tuple[list[int], list[int]]]:
    """Event-stratified, case-level folds.

    Each case is one sample; no window can cross a fold boundary.
    """
    ids = np.asarray([int(c) for c in case_ids], dtype=int)
    strata = event_strata(meta, case_ids)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tr, te in skf.split(ids, strata):
        folds.append((ids[tr].tolist(), ids[te].tolist()))
    return folds


def train_rf(X: np.ndarray, y: np.ndarray, weight) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=30, max_depth=8, random_state=42,
        class_weight=weight, n_jobs=-1)
    clf.fit(X, y)
    return clf


def predict_full(clf: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    p = clf.predict_proba(X)
    out = np.zeros((len(X), 3), dtype=np.float32)
    for j, c in enumerate(clf.classes_):
        out[:, int(c)] = p[:, j]
    return out


def hybrid_pred(data: dict, rows: np.ndarray, p_cal: np.ndarray) -> np.ndarray:
    states = decode_states(data["state"][rows])
    pred = np.full(len(rows), -1, dtype=np.int64)
    valid = states == "VALID"
    if valid.any():
        pred[valid] = np.argmax(p_cal[valid], axis=1)
    flat = np.isin(states, ["FLAT_WITH_HISTORY", "FLAT_EVER_HISTORY"])
    pred[flat] = 2
    return pred


def class_weight_score(data: dict, rows: np.ndarray, y: np.ndarray,
                       p_raw: np.ndarray) -> tuple[float, float, float]:
    pred = np.argmax(p_raw, axis=1)
    m = metrics_from_window_confusion(window_confusion(y, pred))
    return (float(m["macro_f1"]), float(m["u2_binary"]["f1"]),
            -float(log_loss(y, np.clip(p_raw, 1e-12, 1), labels=[0, 1, 2])))


def _strip_per_case(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if k != "per_case"}


def _fold_summary(fold: dict) -> dict:
    keep = ("fold", "train_cases", "test_cases", "train_events", "test_events",
            "weight", "weight_scores", "calibrator_method",
            "calibrator_cv_log_loss", "threshold", "threshold_table",
            "inner_event", "window", "calibration", "event", "elapsed_sec")
    return {k: fold[k] for k in keep if k in fold}


def run_outer_fold(data: dict, meta: dict, rows: dict[int, np.ndarray],
                   train_mask: np.ndarray, eval_mask: np.ndarray,
                   train_cases: list[int], test_cases: list[int],
                   outer_fold: int, n_total: int) -> dict:
    t0 = time.time()
    inner_folds = case_folds(meta, train_cases, 3, seed=42 + outer_fold)
    inner_fold_by_row = np.full(n_total, -1, dtype=np.int16)
    p_inner = {name: np.zeros((n_total, 3), dtype=np.float32)
               for name, _ in WEIGHTS}
    for fi, (itr_cases, ival_cases) in enumerate(inner_folds):
        itr = concat_rows(itr_cases, rows)
        ival = concat_rows(ival_cases, rows)
        itr_train = itr[train_mask[itr]]
        ytr = label_ints(data, itr_train)
        for name, w in WEIGHTS:
            clf = train_rf(data["features"][itr_train], ytr, w)
            if len(ival):
                p_inner[name][ival] = predict_full(clf, data["features"][ival])
        inner_fold_by_row[ival] = fi
    inner_rows = concat_rows(train_cases, rows)
    inner_train_rows = inner_rows[train_mask[inner_rows] &
                                  (inner_fold_by_row[inner_rows] >= 0)]
    y_inner = label_ints(data, inner_train_rows)
    scores = {}
    for name, _ in WEIGHTS:
        scores[name] = class_weight_score(
            data, inner_train_rows, y_inner, p_inner[name][inner_train_rows])
    best_weight = max(WEIGHTS, key=lambda nw: (scores[nw[0]], -WEIGHTS.index(nw)))[0]
    p_sel = p_inner[best_weight]
    sel = select_calibrator_crossfit(
        p_sel[inner_train_rows], y_inner,
        inner_fold_by_row[inner_train_rows], methods=CALIB_METHODS)
    p_cal_inner = apply_calibrator(sel["calibrator"], p_sel)
    thr = select_threshold(
        data, meta, rows, p_cal_inner, train_cases)
    inner_event = _strip_per_case(thr["inner_metrics"])

    outer_train = concat_rows(train_cases, rows)
    outer_test = concat_rows(test_cases, rows)
    outer_fit = outer_train[train_mask[outer_train]]
    clf = train_rf(data["features"][outer_fit],
                   label_ints(data, outer_fit), dict(WEIGHTS)[best_weight])
    p_raw_test = predict_full(clf, data["features"][outer_test])
    p_cal_test = apply_calibrator(sel["calibrator"], p_raw_test)
    p_cal_global = np.zeros((n_total, 3), dtype=np.float32)
    p_raw_global = np.zeros((n_total, 3), dtype=np.float32)
    p_cal_global[outer_test] = p_cal_test
    p_raw_global[outer_test] = p_raw_test
    per_case = evaluate_events_for_cases(
        data, meta, rows, test_cases, p_u2=p_cal_global,
        threshold=thr["threshold"], policy="rf")
    event = aggregate_events(per_case, test_cases)

    win_counts = {}
    for cid in test_cases:
        r = rows[cid] if cid in rows else np.empty(0, dtype=np.int64)
        lab = label_ints(data, r) if len(r) else np.empty(0, dtype=np.int64)
        use = np.isin(lab, TRAIN_CLASSES)
        if use.any():
            win_counts[cid] = window_confusion(
                lab[use], hybrid_pred(data, r[use], p_cal_global[r[use]]))
        else:
            win_counts[cid] = np.zeros((3, 5), dtype=np.int64)
    win = metrics_from_window_confusion(sum(win_counts.values()))
    cal_rows = outer_test[train_mask[outer_test]]
    cal = calibration_metrics(
        p_cal_global[cal_rows], label_ints(data, cal_rows))
    return {
        "fold": int(outer_fold),
        "train_cases": [int(c) for c in train_cases],
        "test_cases": [int(c) for c in test_cases],
        "train_events": int(sum(len(meta["case_meta"][str(c)]["reliable_u2_events"])
                                for c in train_cases)),
        "test_events": int(sum(len(meta["case_meta"][str(c)]["reliable_u2_events"])
                               for c in test_cases)),
        "weight": best_weight,
        "weight_scores": {k: list(v) for k, v in scores.items()},
        "calibrator_method": sel["method"],
        "calibrator_cv_log_loss": sel["cv_log_loss"],
        "threshold": float(thr["threshold"]),
        "threshold_table": thr["table"],
        "inner_event": _strip_per_case(inner_event),
        "window": win,
        "calibration": cal,
        "event": _strip_per_case(event),
        "per_case": per_case,
        "win_counts": win_counts,
        "p_raw_test": p_raw_test,
        "p_cal_test": p_cal_test,
        "outer_test_rows": outer_test,
        "calibrator": sel["calibrator"],
        "elapsed_sec": float(time.time() - t0),
    }


def run_scale(window_sec: float, step_sec: float, jobs: int,
              rebuild: bool) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    data, meta = build_scale_dataset(window_sec, step_sec, jobs=jobs,
                                     rebuild=rebuild)
    rows = rows_by_case(data)
    n_total = len(data["label"])
    train_mask = training_mask(data)
    eval_mask = event_eval_mask(data)
    b0_pred_all = b0_predict(data)
    all_cases = sorted(int(c) for c in meta["case_meta"])
    folds = case_folds(meta, all_cases, 5, seed=42)
    p_raw_oof = np.zeros((n_total, 3), dtype=np.float32)
    p_cal_oof = np.zeros((n_total, 3), dtype=np.float32)
    fold_by_row = np.full(n_total, -1, dtype=np.int16)
    per_case_all = {}
    fold_results = []
    win_counts_all = {cid: np.zeros((3, 5), dtype=np.int64)
                      for cid in all_cases}
    log(f"scale {window_sec:.0f}s: {len(all_cases)} cases, {n_total} windows; "
        f"train rows={int(train_mask.sum())}, event rows={int(eval_mask.sum())}")
    for fi, (tr_cases, te_cases) in enumerate(folds):
        log(f"  outer fold {fi+1}/5: train={len(tr_cases)} cases, "
            f"test={len(te_cases)} cases")
        fr = run_outer_fold(data, meta, rows, train_mask, eval_mask,
                            tr_cases, te_cases, fi, n_total)
        fold_results.append(_fold_summary(fr))
        p_raw_oof[fr["outer_test_rows"]] = fr["p_raw_test"]
        p_cal_oof[fr["outer_test_rows"]] = fr["p_cal_test"]
        fold_by_row[fr["outer_test_rows"]] = fi
        per_case_all.update(fr["per_case"])
        for cid, cnt in fr["win_counts"].items():
            win_counts_all[cid] += cnt
        log(f"    weight={fr['weight']} cal={fr['calibrator_method']} "
            f"thr={fr['threshold']:.2f} win_macroF1={fr['window']['macro_f1']:.4f} "
            f"U2F1={fr['window']['u2_binary']['f1']:.4f} "
            f"evF1={fr['event']['event_f1']:.4f} ({fr['elapsed_sec']:.1f}s)")

    win_all = metrics_from_window_confusion(sum(win_counts_all.values()))
    win_ci = bootstrap_window_metrics(win_counts_all, n_boot=1000, seed=42)
    # Same-cohort B0 window metrics: same OOF rows, truth classes, quality
    # state and UNKNOWN policy; B0's U3 prediction is retained as non-U2.
    b0_win_counts = {cid: np.zeros((3, 5), dtype=np.int64)
                     for cid in all_cases}
    for cid in all_cases:
        r = rows.get(cid, np.empty(0, dtype=np.int64))
        if not len(r):
            continue
        lab = label_ints(data, r)
        use = np.isin(lab, TRAIN_CLASSES)
        if use.any():
            b0_win_counts[cid] = window_confusion(
                lab[use], b0_pred_all[r[use]])
    b0_win = metrics_from_window_confusion(sum(b0_win_counts.values()))
    b0_win_ci = bootstrap_window_metrics(b0_win_counts, n_boot=1000, seed=42)
    # Same-cohort B0 event proxy: same decision time, scorable clipping and
    # full 303-event denominator; no probability threshold.
    b0_per_case = evaluate_events_for_cases(
        data, meta, rows, all_cases, policy="b0")
    b0_event = aggregate_events(b0_per_case, all_cases)
    b0_event_ci = bootstrap_event_metrics(
        b0_per_case, all_cases, n_boot=1000, seed=42)
    b0_event_ci["point"] = _strip_per_case(b0_event_ci["point"])
    event_all = aggregate_events(per_case_all, all_cases)
    event_ci = bootstrap_event_metrics(per_case_all, all_cases,
                                       n_boot=1000, seed=42)
    event_ci["point"] = _strip_per_case(event_ci["point"])
    elig_rows = np.where(train_mask & (fold_by_row >= 0))[0]
    y_elig = label_ints(data, elig_rows)
    cal_raw = calibration_metrics(p_raw_oof[elig_rows], y_elig)
    cal_cal = calibration_metrics(p_cal_oof[elig_rows], y_elig)

    # ---- final deployment configuration (dev-only selection) ------------
    log("  selecting final weight/calibrator/threshold on all development OOF")
    weight_oof = {}
    weight_scores = {}
    for name, w in WEIGHTS:
        p = np.zeros((n_total, 3), dtype=np.float32)
        for tr_cases, te_cases in folds:
            tr = concat_rows(tr_cases, rows)
            te = concat_rows(te_cases, rows)
            fit = tr[train_mask[tr]]
            clf = train_rf(data["features"][fit], label_ints(data, fit), w)
            if len(te):
                p[te] = predict_full(clf, data["features"][te])
        weight_oof[name] = p
        sel = select_calibrator_crossfit(
            p[elig_rows], y_elig, fold_by_row[elig_rows], methods=CALIB_METHODS)
        p_cal = apply_calibrator(sel["calibrator"], p)
        pred = hybrid_pred(data, np.arange(n_total), p_cal)
        lab = label_ints(data, np.arange(n_total))
        use = np.isin(lab, TRAIN_CLASSES)
        wt = metrics_from_window_confusion(window_confusion(lab[use], pred[use]))
        weight_scores[name] = {
            "macro_f1": wt["macro_f1"],
            "u2_f1": wt["u2_binary"]["f1"],
            "calibrator": sel["method"],
        }
        log(f"    final candidate {name}: macroF1={wt['macro_f1']:.4f} "
            f"U2F1={wt['u2_binary']['f1']:.4f} cal={sel['method']}")
    final_name = max(WEIGHTS, key=lambda nw: (
        weight_scores[nw[0]]["macro_f1"],
        weight_scores[nw[0]]["u2_f1"], -WEIGHTS.index(nw)))[0]
    p_final_oof = weight_oof[final_name]
    sel_final = select_calibrator_crossfit(
        p_final_oof[elig_rows], y_elig, fold_by_row[elig_rows],
        methods=CALIB_METHODS)
    p_final_cal = apply_calibrator(sel_final["calibrator"], p_final_oof)
    final_thr = select_threshold(
        data, meta, rows, p_final_cal, all_cases)
    all_fit_rows = np.where(train_mask)[0]
    final_model = train_rf(
        data["features"][all_fit_rows], label_ints(data, all_fit_rows),
        dict(WEIGHTS)[final_name])
    model_path = MODEL_DIR / f"u{int(window_sec)}_rf.joblib"
    import joblib
    joblib.dump({
        "model": final_model,
        "feature_names": list(FEATURE_NAMES),
        "window_sec": float(window_sec),
        "classes": [0, 1, 2],
        "class_weight_name": final_name,
    }, model_path)
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    log(f"    final: weight={final_name} cal={sel_final['method']} "
        f"threshold={final_thr['threshold']:.2f} model={model_path.name} "
        f"sha256={model_hash[:16]}...")
    calib_path = MODEL_DIR / f"u{int(window_sec)}_calibrator.json"
    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump(sel_final["calibrator"], f, ensure_ascii=False, indent=2)
    meta_path = MODEL_DIR / f"u{int(window_sec)}_meta.json"
    result = {
        "stage": 2,
        "window_sec": float(window_sec),
        "step_sec": float(step_sec),
        "dataset_fingerprint": meta["dataset_fingerprint"],
        "cache_schema_version": meta["version"],
        "n_cases": len(all_cases),
        "n_windows": int(n_total),
        "n_train_rows": int(train_mask.sum()),
        "train_label_counts": {
            c: int(np.sum(train_mask & (label_ints(data, np.arange(n_total)) == c)))
            for c in TRAIN_CLASSES},
        "train_state_counts": {
            s: int(np.sum(train_mask & (decode_states(data["state"]) == s)))
            for s in ("VALID",)},
        "folds": fold_results,
        "outer_window": win_all,
        "outer_window_bootstrap": win_ci,
        "outer_event": _strip_per_case(event_all),
        "outer_event_bootstrap": event_ci,
        "b0_window": b0_win,
        "b0_window_bootstrap": b0_win_ci,
        "b0_event": _strip_per_case(b0_event),
        "b0_event_bootstrap": b0_event_ci,
        "calibration_raw": cal_raw,
        "calibration_calibrated": cal_cal,
        "final_config": {
            "weight": final_name,
            "class_weight": dict(WEIGHTS)[final_name],
            "calibrator_method": sel_final["method"],
            "calibrator": sel_final["calibrator"],
            "threshold": float(final_thr["threshold"]),
            "threshold_table": final_thr["table"],
            "weight_scores": weight_scores,
            "model_path": str(model_path.relative_to(ROOT)),
            "model_sha256": model_hash,
            "calibrator_path": str(calib_path.relative_to(ROOT)),
        },
        "quality_policy": {
            "unknown_states": sorted(UNKNOWN_STATES),
            "flat_with_history": "direct U2 window; raw event alarm",
            "flat_ever_history": "direct U2 window; SUSPECT only",
            "event_alarm": ("decision at window end; hold one step; "
                            "clip to scorable time; no merge across "
                            "UNKNOWN/low-support gaps"),
            "time_basis": ("theoretical window-end decision time; "
                           "excludes runtime"),
        },
        "final_test": "sealed; only case-id disjoint assertion",
    }
    with open(OUT_DIR / f"u{int(window_sec)}_results.json", "w",
              encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / f"u{int(window_sec)}_folds.json", "w",
              encoding="utf-8") as f:
        json.dump({"window_sec": float(window_sec),
                   "folds": [{"fold": f["fold"],
                              "train_cases": f["train_cases"],
                              "test_cases": f["test_cases"]}
                             for f in fold_results]},
                  f, ensure_ascii=False, indent=2)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "version": "stage2_model_meta_v2",
            "feature_spec_version": meta["feature_spec_version"],
            "quality_spec_version": meta["quality_spec_version"],
            "dataset_fingerprint": meta["dataset_fingerprint"],
            "cache_schema_version": meta["version"],
            "window_sec": float(window_sec),
            "classes": [0, 1, 2],
            "weight": final_name,
            "calibrator_method": sel_final["method"],
            "threshold": float(final_thr["threshold"]),
            "model_sha256": model_hash,
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "final_test": "sealed",
        }, f, ensure_ascii=False, indent=2)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-sec", type=float, default=None)
    ap.add_argument("--step-sec", type=float, default=None)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    scales = ([(args.window_sec, args.step_sec)]
              if args.window_sec else [(15.0, 7.5), (30.0, 15.0)])
    for ws, st in scales:
        run_scale(ws, st, args.jobs, args.rebuild)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
