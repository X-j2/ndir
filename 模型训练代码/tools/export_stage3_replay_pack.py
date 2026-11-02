# -*- coding: utf-8 -*-
"""Export a dev-only replay pack for the isolated Stage 3 host workbench."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stage0_common import (
    FS, build_u2_events, load_case, split_events_by_support,
)

ROOT = Path(__file__).resolve().parents[1]
DEV_JSON = ROOT / "data_splits" / "cases_200.json"
FINAL_JSON = ROOT / "data_splits" / "final_test_seed42.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=240.0)
    ap.add_argument("--case-id", type=int, default=None)
    args = ap.parse_args()
    dev = [int(c) for c in json.load(open(DEV_JSON, encoding="utf-8"))["case_ids"]]
    final = {int(c) for c in json.load(open(FINAL_JSON, encoding="utf-8"))["case_ids"]}
    if set(dev) & final:
        raise RuntimeError("dev/Final Test overlap")
    candidates = [args.case_id] if args.case_id is not None else dev
    chosen = None
    for cid in candidates:
        if cid not in dev or cid in final:
            raise RuntimeError("only development cases are allowed")
        d = load_case(cid)
        res = build_u2_events(d["etco2"])
        reliable, low = split_events_by_support(res)
        if reliable or low:
            chosen = (cid, d, reliable, low)
            break
    if chosen is None:
        raise RuntimeError("no development case with U2 events")
    cid, d, reliable, low = chosen
    total = len(d["co2"]) / FS
    first = reliable[0] if reliable else low[0]
    start = max(0.0, min(float(first[0]) - 30.0, total - args.seconds))
    end = min(total, start + args.seconds)
    start_i = int(round(start * FS))
    end_i = int(round(end * FS))
    co2 = np.asarray(d["co2"][start_i:end_i], dtype=float)
    co2 = co2[::4]  # 100 -> 25 Hz
    events = []
    for s, e in reliable:
        if e > start and s < end:
            events.append({"start": max(0.0, s - start), "end": min(end, e) - start,
                           "class": "U2", "reliability": "reliable"})
    pack = {
        "version": "stage3_replay_pack_v1",
        "source": "dev",
        "stitched": False,
        "sample_rate_hz": 25.0,
        "case_ids": [cid],
        "segments": [{"case_id": cid, "start_s": start, "end_s": end}],
        "samples_mmhg": [float(x) for x in co2],
        "truth_events": events,
        "note": "dev-only replay; no Final Test waveforms or labels",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(pack, f, ensure_ascii=False)
    print(f"wrote {out} case={cid} source=dev samples={len(co2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
