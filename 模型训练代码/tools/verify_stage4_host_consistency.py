# -*- coding: utf-8 -*-
"""Verify Stage 4 host runtime against an independent 150 s offline reference.

Outputs report/stage4/host_consistency.json.  This is a development-data
verification; no Final Test data are loaded.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT.parent / "上位机代码"
for p in (ROOT, ROOT / "tools", HOST):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from stage0_common import load_case  # noqa: E402
from stage4_host_reference import build_reference, compare_host  # noqa: E402
from host_core.stage4_runtime import Stage4Runtime  # noqa: E402
from training.stage4_fusion import SCALES as SCALE_NAMES  # noqa: E402


def consistency_exit_code(comp: dict) -> int:
    """Return non-zero for any host-vs-reference consistency failure."""
    return 0 if comp.get("ok") else 1


def _ref_100_vs_25(r100: dict, r25: dict) -> dict:
    out = {"scales": {}, "fusion": {}}
    for scale in SCALE_NAMES:
        a = r100["scales"][scale]["decisions"]
        b = r25["scales"][scale]["decisions"]
        rec = {"n_100": len(a), "n_25": len(b),
               "state_changes": 0, "max_feature_abs_delta": None,
               "max_p_abs_delta": None}
        if len(a) == len(b):
            fd, pd = [], []
            for x, y in zip(a, b):
                if x["state"] != y["state"]:
                    rec["state_changes"] += 1
                fd.append(float(np.max(np.abs(
                    np.asarray(x["features"]) - np.asarray(y["features"])))))
                if x["p_u2"] is not None and y["p_u2"] is not None:
                    pd.append(abs(x["p_u2"] - y["p_u2"]))
            rec["max_feature_abs_delta"] = max(fd) if fd else None
            rec["max_p_abs_delta"] = max(pd) if pd else None
        out["scales"][scale] = rec
    t100 = [(x["kind"], x["time"]) for x in r100["fusion_transitions"]]
    t25 = [(x["kind"], x["time"]) for x in r25["fusion_transitions"]]
    f100 = r100["fusion_frames"]
    f25 = r25["fusion_frames"]
    qmis = sum(1 for a, b in zip(f100, f25)
               if abs(a[0] - b[0]) > 1e-9 or a[1] != b[1])
    pd = [abs(a[2] - b[2]) for a, b in zip(f100, f25)
          if a[2] is not None and b[2] is not None]
    out["fusion"] = {
        "transition_equal": t100 == t25,
        "transition_100": t100, "transition_25": t25,
        "quality_mismatch": int(qmis),
        "max_p_abs_delta": max(pd) if pd else None,
        "n_100": len(t100), "n_25": len(t25),
    }
    return out


def main() -> int:
    d = load_case(1)
    # Use a development segment containing UNKNOWN, VALID and an event;
    # 330 s is the start of the checked dev replay window.
    START_SEC = 330.0
    sig100 = np.asarray(
        d["co2"][int(START_SEC * 100):int((START_SEC + 150) * 100)],
        dtype=float)
    sig25 = sig100[::4]
    ref25 = build_reference(sig25, 25, 150.0, ROOT)
    ref100 = build_reference(sig100, 100, 150.0, ROOT)
    host = Stage4Runtime()
    if not host.available:
        raise RuntimeError(host.error)
    for i in range(0, len(sig25), 25):
        host.feed_samples(sig25[i:i + 25])
    host.finish(150.0)
    comp = compare_host(host, ref25)
    out = {
        "case_id": 1,
        "duration_sec": 150.0,
        "sample_timeline": "original 100 Hz -> 25 Hz [::4]",
        "host_vs_reference_25hz": comp,
        "reference_100hz_vs_25hz": _ref_100_vs_25(ref100, ref25),
    }
    path = ROOT / "report" / "stage4" / "host_consistency.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if comp["ok"]:
        print("HOST_VS_REFERENCE_25HZ: PASS")
    else:
        print("HOST_VS_REFERENCE_25HZ: FAIL")
        for reason in comp["failure_reasons"]:
            print(f"  - {reason}")
    return consistency_exit_code(comp)


if __name__ == "__main__":
    raise SystemExit(main())
