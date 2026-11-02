# -*- coding: utf-8 -*-
"""
Build frozen case lists and the Stage 0 event-stratified split.

Outputs (data_splits/):
  cases_250.json          all cached cases, numeric order
  cases_200.json          development cases (250 - final test)
  final_test_seed42.json  locked 50-case final test
  case_event_counts.json  per-case U2 / RR<3 event counts
  split_manifest.json     method, counts, strata, repair log, hashes

The split is deterministic and does not use any model result. Event strata
are based only on reference-track event counts, which are label-side data,
not model predictions.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from stage0_common import (
    REPO_ROOT, build_rr3_events, build_u2_events, list_case_ids,
    load_case, sha256_file, split_events_by_support,
)

OUT_DIR = REPO_ROOT / "data_splits"
TARGET_TEST_FRAC = 0.20
SEED = 42


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    return (x ^ (x >> 31)) & 0xFFFFFFFFFFFFFFFF


def u2_bin(n: int) -> int:
    return 0 if n == 0 else (1 if n <= 2 else 2)


def rr3_bin(n: int) -> int:
    return 0 if n == 0 else 1


def quota_allocate(strata: dict[tuple[int, int], list[int]], target: int) -> dict:
    total = sum(len(v) for v in strata.values())
    raw = {k: len(v) * target / total for k, v in strata.items()}
    quota = {k: int(math.floor(q)) for k, q in raw.items()}
    remainder = target - sum(quota.values())
    # largest fractional part first; tie by stratum key for determinism
    order = sorted(raw, key=lambda k: (-(raw[k] - math.floor(raw[k])), k))
    for k in order[:remainder]:
        quota[k] += 1
    # Do not exceed stratum size.
    for k in strata:
        quota[k] = min(quota[k], len(strata[k]))
    # If capping removed slots, redistribute to strata with room.
    while sum(quota.values()) < target:
        candidates = [k for k in strata if quota[k] < len(strata[k])]
        if not candidates:
            break
        k = max(candidates, key=lambda k: (len(strata[k]) - quota[k], k))
        quota[k] += 1
    while sum(quota.values()) > target:
        candidates = [k for k in strata if quota[k] > 0]
        if not candidates:
            break
        k = max(candidates, key=lambda k: (quota[k], k))
        quota[k] -= 1
    return quota


def select_from_stratum(cases: list[int], n: int) -> list[int]:
    """Deterministic hash-order selection within a stratum."""
    ordered = sorted(cases, key=lambda c: (splitmix64(c ^ SEED), c))
    return ordered[:n]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids = list_case_ids()
    if len(ids) != 250:
        print(f"WARNING: expected 250 cases, found {len(ids)}")

    u2_counts: dict[int, int] = {}
    u2_low_counts: dict[int, int] = {}
    rr3_counts: dict[int, int] = {}
    per_case = {}
    for cid in ids:
        d = load_case(cid)
        u2_res = build_u2_events(d["etco2"])
        u2_reliable, u2_low = split_events_by_support(u2_res)
        rr3 = build_rr3_events(d["rr"])["events"]
        u2_counts[cid] = len(u2_reliable)
        u2_low_counts[cid] = len(u2_low)
        rr3_counts[cid] = len(rr3)
        per_case[cid] = {
            "u2_events": len(u2_reliable),
            "u2_low_support_events": len(u2_low),
            "rr3_events": len(rr3),
            "u2_durations": [int(e - s) for s, e in u2_reliable],
            "u2_low_support_durations": [int(e - s) for s, e in u2_low],
            "u2_support": u2_res.get("support", []),
            "rr3_durations": [int(e - s) for s, e in rr3],
        }

    strata: dict[tuple[int, int], list[int]] = defaultdict(list)
    for cid in ids:
        strata[(u2_bin(u2_counts[cid]), rr3_bin(rr3_counts[cid]))].append(cid)

    target = int(round(len(ids) * TARGET_TEST_FRAC))
    quota = quota_allocate(strata, target)
    test = []
    for k in sorted(strata):
        test.extend(select_from_stratum(strata[k], quota[k]))
    test = sorted(set(test))
    # Repair if the proportional allocation is short on the minimum events.
    repair_log = []
    dev = [c for c in ids if c not in set(test)]

    def counts(cases):
        return (sum(u2_counts[c] for c in cases), sum(rr3_counts[c] for c in cases))

    u2_t, rr3_t = counts(test)
    for _ in range(100):
        if u2_t >= 50 and rr3_t >= 15:
            break
        # Prefer a swap that improves the deficient metric without losing the other.
        improved = False
        if rr3_t < 15:
            dev_candidates = sorted(dev, key=lambda c: (-rr3_counts[c], -u2_counts[c], c))
            test_candidates = sorted(test, key=lambda c: (rr3_counts[c], u2_counts[c], c))
            for dc in dev_candidates:
                if rr3_counts[dc] == 0:
                    break
                for tc in test_candidates:
                    if rr3_counts[dc] <= rr3_counts[tc]:
                        continue
                    new_test = [c for c in test if c != tc] + [dc]
                    nu2, nrr = counts(new_test)
                    if nu2 >= 50 and nrr >= 15:
                        test = sorted(new_test)
                        dev = [c for c in ids if c not in set(test)]
                        repair_log.append({"swap_out": tc, "swap_in": dc,
                                           "reason": "rr3_min", "u2": nu2, "rr3": nrr})
                        u2_t, rr3_t = nu2, nrr
                        improved = True
                        break
                if improved:
                    break
        if improved:
            continue
        if u2_t < 50:
            dev_candidates = sorted(dev, key=lambda c: (-u2_counts[c], c))
            test_candidates = sorted(test, key=lambda c: (u2_counts[c], c))
            for dc in dev_candidates:
                if u2_counts[dc] == 0:
                    break
                for tc in test_candidates:
                    if u2_counts[dc] <= u2_counts[tc]:
                        continue
                    new_test = [c for c in test if c != tc] + [dc]
                    nu2, nrr = counts(new_test)
                    if nu2 >= 50:
                        test = sorted(new_test)
                        dev = [c for c in ids if c not in set(test)]
                        repair_log.append({"swap_out": tc, "swap_in": dc,
                                           "reason": "u2_min", "u2": nu2, "rr3": nrr})
                        u2_t, rr3_t = nu2, nrr
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            break

    test = sorted(test)
    dev = sorted(c for c in ids if c not in set(test))
    u2_t, rr3_t = counts(test)
    u2_d, rr3_d = counts(dev)
    low_t = sum(u2_low_counts[c] for c in test)
    low_t_cases = sum(1 for c in test if u2_low_counts[c] > 0)
    low_d = sum(u2_low_counts[c] for c in dev)
    low_d_cases = sum(1 for c in dev if u2_low_counts[c] > 0)
    assert len(test) + len(dev) == len(ids)
    assert not (set(test) & set(dev))

    cases_250 = {
        "version": "cases_250_v1",
        "n": len(ids),
        "case_ids": ids,
        "source": "cache/vitaldb/*.npz, numeric case-id order",
        "split": "none; frozen superset",
    }
    final_test = {
        "version": "final_test_seed42_v2",
        "method": "event_stratified_quota_with_minimum_repair",
        "seed": SEED,
        "n": len(test),
        "case_ids": test,
        "event_counts": {"u2_events": u2_t, "u2_cases": sum(1 for c in test if u2_counts[c]),
                         "u2_low_support_events": low_t,
                         "u2_low_support_cases": low_t_cases,
                         "rr3_events": rr3_t, "rr3_cases": sum(1 for c in test if rr3_counts[c])},
        "note": "Locked final test; must not be used for training, threshold selection, or fusion tuning.",
    }
    cases_200 = {
        "version": "cases_200_dev_v1",
        "n": len(dev),
        "case_ids": dev,
        "role": "development (250 - final test)",
        "event_counts": {"u2_events": u2_d, "u2_cases": sum(1 for c in dev if u2_counts[c]),
                         "u2_low_support_events": low_d,
                         "u2_low_support_cases": low_d_cases,
                         "rr3_events": rr3_d, "rr3_cases": sum(1 for c in dev if rr3_counts[c])},
    }
    case_events = {
        "version": "case_event_counts_v1",
        "definitions": {
            "u2": "etco2<10, 1s grid, ffill cap 10s, merge gap 10s (no unknown), min duration 10s; reliable = >=2 actual low reading runs associated with event grid cells",
            "support_association": "a raw low run counts if it is the ffill source of at least one low grid cell inside [start,end); its source sample may precede the integer-second event start, within the cap",
            "u2_low_support": "<2 associated EtCO2<10 readings; ignore, not positive/negative",
            "rr3": "rr<3, 1s grid, ffill cap 10s, merge gap 20s (no unknown), min duration 10s",
        },
        "cases": {str(c): per_case[c] for c in ids},
    }

    for name, obj in [
        ("cases_250.json", cases_250),
        ("cases_200.json", cases_200),
        ("final_test_seed42.json", final_test),
        ("case_event_counts.json", case_events),
    ]:
        path = OUT_DIR / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    hash_targets = [
        "cases_250.json",
        "cases_200.json",
        "final_test_seed42.json",
        "case_event_counts.json",
    ]
    manifest = {
        "version": "split_manifest_v2",
        "seed": SEED,
        "target_test_frac": TARGET_TEST_FRAC,
        "n_total": len(ids),
        "n_test": len(test),
        "n_dev": len(dev),
        "test_event_counts": final_test["event_counts"],
        "dev_event_counts": cases_200["event_counts"],
        "strata": {f"u2bin{u}_rr3bin{r}": {
            "n": len(v), "quota": quota.get((u, r), 0),
            "u2_events": sum(u2_counts[c] for c in v),
            "rr3_events": sum(rr3_counts[c] for c in v),
        } for (u, r), v in sorted(strata.items())},
        "repair_log": repair_log,
        "support_definition": (
            "reliable U2 >=2 actual EtCO2<10 reading runs whose ffill source feeds "
            "at least one low grid cell in the event; the source may precede the "
            "integer-second event start within the cap; <2 -> low-support / "
            "ignore_low_support"
        ),
        "hash_algorithm": "sha256",
        "hash_targets": hash_targets,
        "hashes": {name: sha256_file(OUT_DIR / name) for name in hash_targets},
        "manifest_hash_file": "split_manifest.sha256",
        "self_hash_note": (
            "This manifest does not contain its own hash. The sidecar "
            "split_manifest.sha256 is the integrity record for this file; "
            "hash_targets lists exactly the files whose hashes are in hashes."
        ),
    }
    with open(OUT_DIR / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    manifest_hash = sha256_file(OUT_DIR / "split_manifest.json")
    with open(OUT_DIR / "split_manifest.sha256", "w", encoding="utf-8") as f:
        f.write(f"{manifest_hash}  split_manifest.json\n")

    print(f"cases: {len(ids)}; test: {len(test)}; dev: {len(dev)}")
    print(f"test events: U2={u2_t} in {final_test['event_counts']['u2_cases']} cases; "
          f"RR3={rr3_t} in {final_test['event_counts']['rr3_cases']} cases")
    print(f"dev events: U2={u2_d}; RR3={rr3_d}")
    print(f"repair swaps: {len(repair_log)}")
    print(f"manifest: {OUT_DIR / 'split_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
