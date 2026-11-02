"""Isolated Stage 3 replay workbench.

Replay uses only a self-contained dev/synthetic pack.  It never touches the
production DatabaseManager, tray notifications, beep, serial port or the live
Stage3Runtime.  All algorithm decisions are driven by signal position, not by
wall-clock time.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from host_core.paths import training_root

FS = 25.0


def _case_sets() -> tuple[set[int], set[int]]:
    """Load the reviewed development whitelist and Final Test blacklist."""
    root = training_root()
    if root is None:
        raise ValueError("无法定位训练仓库，不能校验回放病例集")
    dev_path = root / "data_splits" / "cases_200.json"
    final_path = root / "data_splits" / "final_test_seed42.json"
    try:
        with open(dev_path, encoding="utf-8") as f:
            dev = {int(x) for x in json.load(f)["case_ids"]}
        with open(final_path, encoding="utf-8") as f:
            final = {int(x) for x in json.load(f)["case_ids"]}
    except Exception as exc:
        raise ValueError(f"无法读取回放病例集: {exc}") from exc
    if dev & final:
        raise ValueError("开发集与 Final Test 病例集重叠，拒绝回放")
    return dev, final


@dataclass
class ReplayPack:
    version: str
    source: str
    stitched: bool
    sample_rate_hz: float
    samples_mmhg: list[float]
    case_ids: list[int] = field(default_factory=list)
    segments: list[dict] = field(default_factory=list)
    truth_events: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "ReplayPack":
        if str(path).lower().endswith("final_test_seed42.json"):
            raise ValueError("Final Test 不允许进入回放")
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        source = str(d.get("source", "")).lower()
        if source not in ("dev", "synthetic"):
            raise ValueError("回放来源必须是 dev 或 synthetic")
        if "final_test" in json.dumps(d.get("segments", []), ensure_ascii=False).lower():
            raise ValueError("Final Test 不允许进入回放")
        if abs(float(d.get("sample_rate_hz", 0)) - FS) > 1e-9:
            raise ValueError("Stage 3 replay pack 必须为 25 Hz")
        samples = [float(x) for x in d.get("samples_mmhg", [])]
        if not samples:
            raise ValueError("回放 pack 无样本")

        case_ids = [int(x) for x in d.get("case_ids", [])]
        segments = list(d.get("segments", []))
        truth_events = list(d.get("truth_events", []))
        segment_ids = [int(s["case_id"]) for s in segments
                       if "case_id" in s]
        truth_ids = [int(e["case_id"]) for e in truth_events
                     if "case_id" in e]
        dev_ids, final_ids = _case_sets()
        if any(x in final_ids for x in case_ids + segment_ids + truth_ids):
            raise ValueError("Final Test 病例号不允许进入回放")
        if source == "dev":
            if not case_ids:
                raise ValueError("dev 回放包必须声明开发集病例号")
            if any(x not in dev_ids for x in case_ids):
                raise ValueError("dev 回放包病例不在开发集白名单")
            if not segments:
                raise ValueError("dev 回放包必须包含分段信息")
            for seg in segments:
                if "case_id" not in seg:
                    raise ValueError("dev 回放包分段缺少 case_id")
                sid = int(seg["case_id"])
                if sid not in case_ids or sid not in dev_ids:
                    raise ValueError("dev 回放包分段病例不在开发集白名单")
                start_s = float(seg.get("start_s", 0.0))
                end_s = float(seg.get("end_s", -1.0))
                if start_s < 0 or end_s <= start_s:
                    raise ValueError("dev 回放包分段区间无效")
        else:
            if case_ids or segment_ids:
                raise ValueError("synthetic 回放包不得声明真实病例号")

        return cls(
            version=str(d.get("version", "stage3_replay_pack_v1")),
            source=source,
            stitched=bool(d.get("stitched", False)),
            sample_rate_hz=float(d["sample_rate_hz"]),
            samples_mmhg=samples,
            case_ids=case_ids,
            segments=segments,
            truth_events=truth_events,
        )

    @property
    def duration_sec(self) -> float:
        return len(self.samples_mmhg) / self.sample_rate_hz


class ReplaySession:
    """Signal-time replay driver.  The injected runtime must be isolated."""

    def __init__(self, pack: ReplayPack, runtime):
        self.pack = pack
        self.runtime = runtime
        self.position = 0
        self.playing = False
        self.events: list[dict] = []
        self.last_state = "WAITING"
        self._signal_accum = 0.0
        self._rebuild()

    def _rebuild(self) -> None:
        self.runtime.reset()
        self.position = 0
        self.events = []
        self.last_state = "WAITING"
        self._signal_accum = 0.0

    def _feed_range(self, end: int) -> list[dict]:
        end = max(0, min(int(end), len(self.pack.samples_mmhg)))
        out = []
        if end <= self.position:
            return out
        # Feed in small chunks so history/feature calculations never depend on
        # the UI timer's speed chunk size.
        while self.position < end:
            n = min(25, end - self.position)
            out.extend(self.runtime.feed_samples(
                self.pack.samples_mmhg[self.position:self.position + n]))
            self.position += n
        self.events.extend(out)
        states = self.runtime.states()
        if states:
            self.last_state = " | ".join(
                f"{s}:{states[s]['state']}" for s in ("U15", "U30"))
        return out

    def play(self) -> None:
        if self.position >= len(self.pack.samples_mmhg):
            self.restart()
        self.playing = True

    def pause(self) -> None:
        self.playing = False

    def restart(self) -> None:
        self._rebuild()

    def seek(self, signal_time: float) -> list[dict]:
        self.playing = False
        self._rebuild()
        target = max(0, min(int(round(float(signal_time) *
                                       self.pack.sample_rate_hz)),
                            len(self.pack.samples_mmhg)))
        out = self._feed_range(target)
        self._signal_accum = float(self.position)
        return out

    def advance(self, wall_dt: float, speed: float) -> list[dict]:
        if not self.playing:
            return []
        # Carry the fractional sample across ticks.  Rounding each tick
        # independently makes 10 x 0.1 s at 1x consume only 20 samples while
        # 0.5 s at 2x consumes 25.
        self._signal_accum += (float(wall_dt) * float(speed) *
                               self.pack.sample_rate_hz)
        target = min(len(self.pack.samples_mmhg),
                     max(self.position,
                         int(math.floor(self._signal_accum + 1e-9))))
        out = self._feed_range(target)
        if self.position >= len(self.pack.samples_mmhg):
            self.playing = False
        return out

    @property
    def signal_time(self) -> float:
        return self.position / self.pack.sample_rate_hz

    @property
    def stitched_label(self) -> str:
        return "拼接" if self.pack.stitched else "单段"
