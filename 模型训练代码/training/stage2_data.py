# -*- coding: utf-8 -*-
"""Stage 2 U15/U30 development datasets.

Only ``data_splits/cases_200.json`` is used.  ``final_test_seed42.json`` is
opened only to assert that the 50 locked cases are disjoint; no Final Test
waveform or label is loaded.

The cache stores, per window:
  - 15 SI features (float32)
  - Stage 1 quality state (integer code)
  - frozen Stage 0 window label (integer code)
  - window start and reference-valid fraction
  - flat/artifact diagnostics needed by the quality policy

Per-case metadata stores total/evaluable hours and the frozen U2 event
timelines (reliable and low-support, kept separate).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from stage0_common import (  # noqa: E402
    FS, U2_CAP_SEC, build_u2_events, ffill_100, iter_windows, load_case,
    runs, sample_1s, split_events_by_support,
)
from features.si_features import SPEC, extract_si_features  # noqa: E402
from features.quality_states import (  # noqa: E402
    HistoryIndex, classify_signal_state, si_breathing_bins,
)

CACHE_DIR = ROOT / "cache" / "stage2"
DATA_SPLITS = ROOT / "data_splits"
CACHE_SCHEMA_VERSION = "stage2_dataset_v2"
FINGERPRINT_FILES = (
    "features/feature_spec_si_v2.json",
    "features/si_features.py",
    "features/quality_states.py",
    "tools/stage0_common.py",
    "preprocessing/make_windows.py",
    "preprocessing/signal_processing.py",
    "training/stage2_data.py",
    "docs/specs/labels_spec_v1.md",
    "docs/specs/event_spec_v1.md",
    "docs/specs/quality_abstention_spec_v1.md",
    "docs/specs/stage1_quality_rule_spec_v3.md",
    "docs/specs/stage2_model_eval_spec_v2.md",
    "data_splits/cases_200.json",
    "data_splits/final_test_seed42.json",
    "data_splits/case_event_counts.json",
    "data_splits/split_manifest.json",
)

LABEL_ORDER = (
    "U0", "U1", "U2", "U3", "ambiguous", "ignore",
    "ignore_low_support", "unlabeled", "other",
)
STATE_ORDER = (
    "VALID", "FLAT_WITH_HISTORY", "FLAT_EVER_HISTORY",
    "FLAT_NO_HISTORY", "FLAT_HISTORY_UNKNOWN", "GAP", "SATURATED",
    "POLARITY_INVALID", "QUANTILE_ARTIFACT", "NOISY",
)
LABEL_CODE = {name: i for i, name in enumerate(LABEL_ORDER)}
STATE_CODE = {name: i for i, name in enumerate(STATE_ORDER)}
TRAIN_LABELS = ("U0", "U1", "U2")
EVENT_STATES = ("VALID", "FLAT_WITH_HISTORY", "FLAT_EVER_HISTORY")


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dev_ids() -> list[int]:
    dev = load_json(DATA_SPLITS / "cases_200.json")
    test = load_json(DATA_SPLITS / "final_test_seed42.json")
    dev_ids = [int(c) for c in dev["case_ids"]]
    test_ids = {int(c) for c in test["case_ids"]}
    if len(dev_ids) != 200:
        raise RuntimeError(f"expected 200 development cases, got {len(dev_ids)}")
    if set(dev_ids) & test_ids:
        raise RuntimeError("development/Final-Test overlap")
    return dev_ids


def load_final_test_ids() -> list[int]:
    """Case ids only.  Final Test waveforms/labels are never loaded."""
    test = load_json(DATA_SPLITS / "final_test_seed42.json")
    return [int(c) for c in test["case_ids"]]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_components() -> dict:
    return {name: sha256_file(ROOT / name) for name in FINGERPRINT_FILES}


def compute_dataset_fingerprint(window_sec: float, step_sec: float,
                                dev_ids: list[int],
                                final_ids: list[int]) -> str:
    """Hash the split, specs, and every implementation file used by the cache."""
    h = hashlib.sha256()
    h.update(CACHE_SCHEMA_VERSION.encode())
    h.update(f"{float(window_sec):.6f}|{float(step_sec):.6f}".encode())
    h.update(json.dumps(sorted(int(c) for c in dev_ids)).encode())
    h.update(json.dumps(sorted(int(c) for c in final_ids)).encode())
    for name, digest in sorted(fingerprint_components().items()):
        h.update(name.encode())
        h.update(digest.encode())
    return h.hexdigest()


def validate_cache_meta(meta: dict, data: dict | None,
                        expected_fingerprint: str,
                        dev_ids: list[int], final_ids: list[int],
                        window_sec: float, step_sec: float) -> list[str]:
    """Return explicit cache-invalidation reasons (empty means valid)."""
    errors = []
    if meta.get("version") != CACHE_SCHEMA_VERSION:
        errors.append(f"schema {meta.get('version')} != {CACHE_SCHEMA_VERSION}")
    if float(meta.get("window_sec", -1)) != float(window_sec):
        errors.append("window_sec mismatch")
    if float(meta.get("step_sec", -1)) != float(step_sec):
        errors.append("step_sec mismatch")
    if meta.get("dataset_fingerprint") != expected_fingerprint:
        errors.append("dataset fingerprint mismatch")
    dev = [int(c) for c in meta.get("dev_case_ids", [])]
    if dev != [int(c) for c in dev_ids]:
        errors.append("cached dev_case_ids differ from frozen development set")
    if set(dev) & set(int(c) for c in final_ids):
        errors.append("cached development ids overlap Final Test")
    if data is not None:
        ids = {int(c) for c in np.asarray(data.get("case_id", [])).tolist()}
        if not ids.issubset(set(int(c) for c in dev)):
            errors.append("cached window case_id outside development set")
        if ids & set(int(c) for c in final_ids):
            errors.append("cached window case_id includes Final Test")
    return errors


def cache_paths(window_sec: float) -> tuple[Path, Path]:
    tag = f"u{int(window_sec)}"
    return CACHE_DIR / f"{tag}.npz", CACHE_DIR / f"{tag}_meta.json"




def verify_dev_event_counts(meta: dict, dev_ids: list[int]) -> None:
    """Check every cached case against the frozen per-case event table."""
    frozen = load_json(DATA_SPLITS / "case_event_counts.json")["cases"]
    total_rel = total_low = 0
    for cid in dev_ids:
        key = str(int(cid))
        cm = meta["case_meta"][key]
        n_rel = len(cm.get("reliable_u2_events", []))
        n_low = len(cm.get("low_support_u2_events", []))
        f = frozen[key]
        if n_rel != int(f["u2_events"]) or n_low != int(f["u2_low_support_events"]):
            raise RuntimeError(
                f"case {cid} event count mismatch: cache={n_rel}/{n_low}, "
                f"frozen={f['u2_events']}/{f['u2_low_support_events']}")
        total_rel += n_rel
        total_low += n_low
    if (total_rel, total_low) != (303, 60):
        raise RuntimeError(
            f"development event totals {total_rel}/{total_low} != 303/60")


def _window_arrays(rec: dict, d: dict, hist: HistoryIndex,
                   ref_valid: np.ndarray) -> tuple[list, dict]:
    feat = extract_si_features(rec["co2"])
    state = classify_signal_state(
        rec["co2"], hist.summary(rec["start"] / FS), feature_diag=feat)
    start = int(rec["start"])
    end = start + len(rec["co2"])
    ref_frac = float(np.mean(ref_valid[start:end])) if end > start else 0.0
    row = {
        "start": start,
        "label": LABEL_CODE[rec["label"]],
        "state": STATE_CODE[state],
        "features": [float(x) for x in feat["vector"]],
        "ref_valid_frac": ref_frac,
        "A_invalid": bool(feat["A_invalid"]),
        "constant_flat": bool(feat["constant_flat"]),
        "quantile_artifact": bool(feat["quantile_artifact"]),
    }
    return row, {
        "start_s": start / FS,
        "end_s": end / FS,
        "label": rec["label"],
        "state": state,
    }


def _event_mask_1s(length: int, intervals: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for s, e in intervals:
        mask[max(0, int(s)):min(length, int(e))] = True
    return mask


def _case_scorable_meta(d: dict, reliable: list[tuple[int, int]],
                        low: list[tuple[int, int]]) -> dict:
    """Build the common scorable time domain for alarms/FA_h.

    Low-support intervals are neither positive nor negative.  Reference
    unknown (ffill cap exceeded) is also unscorable.  The returned intervals
    are half-open seconds on the 1 s reference grid.
    """
    ref_1s = sample_1s(d["etco2"], U2_CAP_SEC)
    n = len(ref_1s)
    ref_ok = np.isfinite(ref_1s)
    low_mask = _event_mask_1s(n, low)
    scorable = ref_ok & ~low_mask
    scorable_iv = [[float(s), float(e)] for s, e in runs(scorable)]
    unknown_iv = [[float(s), float(e)] for s, e in runs(~ref_ok)]
    low_iv = [[float(s), float(e)] for s, e in runs(low_mask)]
    low_excl = int(np.sum(low_mask & ref_ok))
    low_unknown = int(np.sum(low_mask & ~ref_ok))
    ref_unknown = int(np.sum(~ref_ok))
    scorable_sec = int(np.sum(scorable))
    total_sec = len(d["co2"]) / FS
    return {
        "total_sec": float(total_sec),
        "scorable_intervals": scorable_iv,
        "reference_unknown_intervals": unknown_iv,
        "low_support_intervals": low_iv,
        "scorable_sec": float(scorable_sec),
        "evaluable_sec": float(scorable_sec),
        "excluded_sec": float(n - scorable_sec),
        "reference_unknown_excluded_sec": float(ref_unknown),
        "low_support_excluded_sec": float(low_excl),
        "low_support_unknown_overlap_sec": float(low_unknown),
        "reliable_u2_events": [
            {"start": float(s), "end": float(e)} for s, e in reliable],
        "low_support_u2_events": [
            {"start": float(s), "end": float(e)} for s, e in low],
    }


def _build_case(args: tuple[int, float, float]) -> dict:
    case_id, window_sec, step_sec = args
    d = load_case(case_id)
    u2_res = build_u2_events(d["etco2"])
    reliable, low = split_events_by_support(u2_res)
    meta = {"case_id": case_id, **_case_scorable_meta(d, reliable, low)}
    hist = HistoryIndex(si_breathing_bins(d["co2"]))
    ref_valid = np.isfinite(ffill_100(d["etco2"], U2_CAP_SEC))
    rows = []
    for rec in iter_windows(
            case_id, window_sec, step_sec, u2_events=reliable,
            low_support_events=low, include_co2=True, case_data=d):
        row, _ = _window_arrays(rec, d, hist, ref_valid)
        rows.append(row)
    meta["n_windows"] = len(rows)
    return {"case_id": case_id, "rows": rows, "meta": meta}


def _assemble(parts: list[dict], window_sec: float, step_sec: float,
              dev_ids: list[int], final_ids: list[int],
              fingerprint: str, components: dict) -> tuple[dict, dict]:
    by_case = {p["case_id"]: p for p in parts}
    starts, case_ids, labels, states = [], [], [], []
    feats, ref_fracs = [], []
    a_invalid, constant_flat, artifact = [], [], []
    case_meta = {}
    for cid in dev_ids:
        p = by_case[cid]
        for r in p["rows"]:
            starts.append(r["start"])
            case_ids.append(cid)
            labels.append(r["label"])
            states.append(r["state"])
            feats.append(r["features"])
            ref_fracs.append(r["ref_valid_frac"])
            a_invalid.append(r["A_invalid"])
            constant_flat.append(r["constant_flat"])
            artifact.append(r["quantile_artifact"])
        case_meta[str(cid)] = p["meta"]
    data = {
        "case_id": np.asarray(case_ids, dtype=np.int32),
        "start": np.asarray(starts, dtype=np.int32),
        "label": np.asarray(labels, dtype=np.int8),
        "state": np.asarray(states, dtype=np.int8),
        "features": np.asarray(feats, dtype=np.float32),
        "ref_valid_frac": np.asarray(ref_fracs, dtype=np.float32),
        "A_invalid": np.asarray(a_invalid, dtype=bool),
        "constant_flat": np.asarray(constant_flat, dtype=bool),
        "quantile_artifact": np.asarray(artifact, dtype=bool),
    }
    meta = {
        "version": CACHE_SCHEMA_VERSION,
        "window_sec": float(window_sec),
        "step_sec": float(step_sec),
        "n_cases": len(dev_ids),
        "n_windows": int(len(case_ids)),
        "feature_spec_version": SPEC["version"],
        "quality_spec_version": "stage1_quality_rule_spec_v3",
        "dataset_fingerprint": fingerprint,
        "fingerprint_components": components,
        "dev_case_ids": [int(c) for c in dev_ids],
        "final_test_case_ids": [int(c) for c in final_ids],
        "labels": list(LABEL_ORDER),
        "states": list(STATE_ORDER),
        "case_meta": case_meta,
    }
    return data, meta


def build_scale_dataset(window_sec: float, step_sec: float,
                        jobs: int = 1, rebuild: bool = False) -> tuple[dict, dict]:
    npz_path, meta_path = cache_paths(window_sec)
    dev_ids = load_dev_ids()
    final_ids = load_final_test_ids()
    expected_fp = compute_dataset_fingerprint(
        window_sec, step_sec, dev_ids, final_ids)
    if not rebuild and npz_path.exists() and meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            with np.load(npz_path) as z:
                data = {k: z[k] for k in z.files}
        except Exception as exc:  # corrupt/incompatible cache
            print(f"[stage2_data] cache unreadable ({exc}); rebuilding",
                  flush=True)
        else:
            errors = validate_cache_meta(
                meta, data, expected_fp, dev_ids, final_ids,
                window_sec, step_sec)
            if not errors:
                verify_dev_event_counts(meta, dev_ids)
                return data, meta
            print("[stage2_data] cache invalid: " + "; ".join(errors) +
                  "; rebuilding", flush=True)
    elif rebuild:
        print("[stage2_data] --rebuild requested; rebuilding", flush=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [(cid, float(window_sec), float(step_sec)) for cid in dev_ids]
    parts = []
    if jobs <= 1:
        for i, task in enumerate(tasks, 1):
            parts.append(_build_case(task))
            if i % 20 == 0:
                print(f"  ... {i}/{len(tasks)} cases", flush=True)
    else:
        ctx = __import__("multiprocessing").get_context("spawn")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
            futs = {ex.submit(_build_case, t): t[0] for t in tasks}
            done = 0
            for fut in as_completed(futs):
                parts.append(fut.result())
                done += 1
                if done % 20 == 0:
                    print(f"  ... {done}/{len(tasks)} cases", flush=True)
    parts.sort(key=lambda p: p["case_id"])
    components = fingerprint_components()
    fingerprint = compute_dataset_fingerprint(
        window_sec, step_sec, dev_ids, final_ids)
    if fingerprint != expected_fp:
        raise RuntimeError("implementation/spec changed while building cache")
    data, meta = _assemble(
        parts, window_sec, step_sec, dev_ids, final_ids,
        fingerprint, components)
    np.savez_compressed(npz_path, **data)
    verify_dev_event_counts(meta, dev_ids)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return data, meta


def decode_labels(arr: np.ndarray) -> np.ndarray:
    names = np.asarray(LABEL_ORDER, dtype=object)
    return names[np.asarray(arr, dtype=int)]


def decode_states(arr: np.ndarray) -> np.ndarray:
    names = np.asarray(STATE_ORDER, dtype=object)
    return names[np.asarray(arr, dtype=int)]


def training_mask(data: dict) -> np.ndarray:
    """Rows eligible for the RF: U0/U1/U2 and quality state VALID."""
    labels = decode_labels(data["label"])
    states = decode_states(data["state"])
    return (np.isin(labels, TRAIN_LABELS) & (states == "VALID"))


def model_inference_mask(data: dict) -> np.ndarray:
    """Rows that reach the RF in the hybrid policy (VALID only)."""
    return decode_states(data["state"]) == "VALID"


def event_eval_mask(data: dict, min_ref_valid_frac: float = 0.90) -> np.ndarray:
    """Rows usable for single-scale event proxies (non-UNKNOWN + reference)."""
    states = decode_states(data["state"])
    return (np.isin(states, EVENT_STATES) &
            (data["ref_valid_frac"] >= min_ref_valid_frac))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-sec", type=float, required=True)
    ap.add_argument("--step-sec", type=float, required=True)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    data, meta = build_scale_dataset(
        args.window_sec, args.step_sec, jobs=args.jobs, rebuild=args.rebuild)
    print(json.dumps({k: meta[k] for k in (
        "version", "window_sec", "step_sec", "n_cases", "n_windows",
        "feature_spec_version", "quality_spec_version")}, ensure_ascii=False))
    print("label counts", {LABEL_ORDER[i]: int((data["label"] == i).sum())
                           for i in range(len(LABEL_ORDER))})
    print("state counts", {STATE_ORDER[i]: int((data["state"] == i).sum())
                           for i in range(len(STATE_ORDER))})
