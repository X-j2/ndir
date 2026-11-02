# -*- coding: utf-8 -*-
"""Multiclass probability calibration and calibration metrics for Stage 2.

No model is trained here.  Calibrators are fitted on out-of-fold RF
probabilities only.  Candidates are:
  - ``none``: raw RF probabilities;
  - ``temperature``: one-parameter temperature scaling on log-probabilities;
  - ``multinomial_logistic``: multinomial logistic regression on
    pairwise log-probability ratios.

Calibrator selection is done by leave-one-inner-fold-out fitting, so a
calibration method is not selected on the same predictions used to score it.
"""
from __future__ import annotations

import json
from typing import Iterable

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

CLASSES = (0, 1, 2)
METHODS = ("none", "temperature", "multinomial_logistic")
EPS = 1e-12


def clip_probs(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, EPS, 1.0)
    return p / np.sum(p, axis=1, keepdims=True)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=float)
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def _log_prob_features(p: np.ndarray) -> np.ndarray:
    lp = np.log(clip_probs(p))
    # Two log ratios relative to class 0 avoid a redundant third column.
    return lp[:, 1:] - lp[:, :1]


def fit_calibrator(method: str, p_raw: np.ndarray, y: np.ndarray) -> dict:
    p_raw = clip_probs(p_raw)
    y = np.asarray(y, dtype=int)
    if method == "none":
        return {"method": "none"}
    if method == "temperature":
        lp = np.log(p_raw)

        def objective(log_t: float) -> float:
            p = _softmax(lp / np.exp(log_t))
            return float(log_loss(y, p, labels=list(CLASSES)))

        res = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
        return {"method": "temperature", "log_temperature": float(res.x),
                "temperature": float(np.exp(res.x))}
    if method == "multinomial_logistic":
        clf = LogisticRegression(
            C=1.0, max_iter=2000, solver="lbfgs", random_state=42)
        clf.fit(_log_prob_features(p_raw), y)
        return {
            "method": "multinomial_logistic",
            "classes": [int(c) for c in clf.classes_],
            "coef": np.asarray(clf.coef_, dtype=float).tolist(),
            "intercept": np.asarray(clf.intercept_, dtype=float).tolist(),
        }
    raise ValueError(f"unknown calibration method: {method}")


def apply_calibrator(cal: dict, p_raw: np.ndarray) -> np.ndarray:
    p_raw = clip_probs(p_raw)
    method = cal.get("method", "none")
    if method == "none":
        return p_raw
    if method == "temperature":
        lp = np.log(p_raw)
        return _softmax(lp / float(cal["temperature"]))
    if method == "multinomial_logistic":
        feat = _log_prob_features(p_raw)
        coef = np.asarray(cal["coef"], dtype=float)
        intercept = np.asarray(cal["intercept"], dtype=float)
        logits = feat @ coef.T + intercept
        # sklearn stores classes in sorted order; restore 0,1,2 if needed.
        classes = [int(c) for c in cal["classes"]]
        p = _softmax(logits)
        if classes != list(CLASSES):
            out = np.zeros_like(p)
            for j, c in enumerate(classes):
                out[:, c] = p[:, j]
            p = out
        return p
    raise ValueError(f"unknown calibrated method: {method}")


def select_calibrator_crossfit(p_oof: np.ndarray, y: np.ndarray,
                               inner_fold: np.ndarray,
                               methods: Iterable[str] = METHODS) -> dict:
    """Select method by leave-one-inner-fold-out log loss.

    ``p_oof`` must come from inner-fold models that never saw the fold being
    scored.  The returned calibrator is fitted on all inner OOF rows.
    """
    p_oof = clip_probs(p_oof)
    y = np.asarray(y, dtype=int)
    fold = np.asarray(inner_fold, dtype=int)
    scores = {}
    for method in methods:
        losses = []
        for f in sorted(set(fold.tolist())):
            train = fold != f
            test = fold == f
            if not train.any() or not test.any():
                continue
            cal = fit_calibrator(method, p_oof[train], y[train])
            p = apply_calibrator(cal, p_oof[test])
            losses.append(float(log_loss(y[test], p, labels=list(CLASSES))))
        scores[method] = float(np.mean(losses)) if losses else float("inf")
    best = min(scores, key=lambda m: (scores[m], METHODS.index(m)))
    final = fit_calibrator(best, p_oof, y)
    return {
        "method": best,
        "cv_log_loss": scores,
        "calibrator": final,
    }


def ece_binary(y_bin: np.ndarray, p: np.ndarray, n_bins: int = 15) -> dict:
    y = np.asarray(y_bin, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    value = 0.0
    bins = []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            bins.append({"bin": b, "n": 0, "mean_pred": None, "frac_pos": None})
            continue
        mp = float(np.mean(p[m]))
        fp = float(np.mean(y[m]))
        value += (n / len(p)) * abs(mp - fp)
        bins.append({"bin": b, "n": n, "mean_pred": mp, "frac_pos": fp})
    return {"ece": float(value), "bins": bins}


def calibration_metrics(p: np.ndarray, y: np.ndarray) -> dict:
    p = clip_probs(p)
    y = np.asarray(y, dtype=int)
    out = {
        "n": int(len(y)),
        "log_loss": float(log_loss(y, p, labels=list(CLASSES))),
        "classwise": {},
        "top_label_ece": None,
        "auc_macro": None,
    }
    aucs = []
    for k in CLASSES:
        yb = (y == k).astype(int)
        pk = p[:, k]
        out["classwise"][str(k)] = {
            "prevalence": float(np.mean(yb)),
            "brier": float(np.mean((pk - yb) ** 2)),
            **ece_binary(yb, pk, n_bins=15),
            "auc": (float(roc_auc_score(yb, pk))
                    if len(set(yb.tolist())) == 2 else None),
        }
        if len(set(yb.tolist())) == 2:
            aucs.append(float(roc_auc_score(yb, pk)))
    out["auc_macro"] = float(np.mean(aucs)) if aucs else None
    pred = np.argmax(p, axis=1)
    conf = p[np.arange(len(p)), pred]
    correct = (pred == y).astype(float)
    out["top_label_ece"] = float(ece_binary(correct, conf, n_bins=15)["ece"])
    return out


def calibrator_to_json(cal: dict) -> dict:
    return json.loads(json.dumps(cal))
