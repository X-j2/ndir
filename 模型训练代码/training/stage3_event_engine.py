# -*- coding: utf-8 -*-
"""Stage 3 causal single-scale event engine (U15/U30).

The engine consumes decisions at window end.  It never back-dates an alarm to
the input window start.  UNKNOWN closes the current output and resets all
counters; evidence cannot silently continue across an interruption.

FLAT_EVER_HISTORY is suspect-only.  ACTIVE requires FLAT_WITH_HISTORY or a
VALID window whose calibrated U2 probability reaches the active threshold.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, asdict

from features.quality_states import UNKNOWN_STATES

LABEL = "呼气 CO2 低值/消失样事件"
SCALES = ("U15", "U30")
_UNSET = object()


@dataclass(frozen=True)
class EventParams:
    suspect_p: float = 0.15
    active_enter_p: float = 0.45
    active_exit_p: float = 0.15
    suspect_on_count: int = 1
    suspect_off_count: int = 1
    active_on_count: int = 2
    active_off_count: int = 2

    def as_dict(self) -> dict:
        return asdict(self)


def params_from_dict(d: dict) -> EventParams:
    return EventParams(
        suspect_p=float(d["suspect_p"]),
        active_enter_p=float(d["active_enter_p"]),
        active_exit_p=float(d["active_exit_p"]),
        suspect_on_count=int(d["suspect_on_count"]),
        suspect_off_count=int(d["suspect_off_count"]),
        active_on_count=int(d["active_on_count"]),
        active_off_count=int(d["active_off_count"]),
    )


def _close_episode(ep: dict, t: float, reason: str) -> None:
    ep["end_reason"] = reason
    if ep.get("active_intervals"):
        last = ep["active_intervals"][-1]
        if last["end"] is None:
            last["end"] = float(t)
    if ep.get("suspect_end") is None:
        ep["suspect_end"] = float(t)
    ep["end_time"] = float(t)


class EventEngine:
    """One scale's state machine.

    States: WAITING, UNKNOWN, NORMAL, SUSPECT, ACTIVE.
    """

    def __init__(self, params: EventParams, step_sec: float,
                 scale: str = "U15"):
        if params.active_exit_p >= params.active_enter_p:
            raise ValueError("active_exit_p must be below active_enter_p")
        if params.suspect_p > params.active_enter_p:
            raise ValueError("suspect_p must be <= active_enter_p")
        if params.suspect_on_count > params.active_on_count:
            raise ValueError("suspect_on_count must be <= active_on_count")
        self.params = params
        self.step_sec = float(step_sec)
        self.scale = str(scale)
        self.state = "WAITING"
        self.episodes: list[dict] = []
        self.transitions: list[dict] = []
        self._episode: dict | None = None
        self._episode_seq = 0
        self._suspect_run = 0
        self._suspect_clear = 0
        self._active_run = 0
        self._active_clear = 0
        self._suspect_open = False
        self._active_open = False
        self._high_seen = False
        self._last_decision_time: float | None = None
        self._transition_seq = 0
        self.unknown_decisions = 0
        self.total_decisions = 0
        self.interruptions = 0
        self.gap_interruptions = 0

    # ---- helpers -----------------------------------------------------
    def _snapshot(self, ep: dict | None = None) -> dict | None:
        ep = self._episode if ep is None else ep
        return copy.deepcopy(ep) if ep is not None else None

    def _emit(self, kind: str, t: float, ep=_UNSET,
              reason: str | None = None) -> None:
        target = self._episode if ep is _UNSET else ep
        snap = self._snapshot(target)
        self._transition_seq += 1
        self.transitions.append({
            "transition_seq": self._transition_seq,
            "kind": kind, "time": float(t), "scale": self.scale,
            "episode_id": (snap or {}).get("episode_id"),
            "state": self.state,
            "reason": reason,
            "episode": snap,
        })

    def _new_episode(self, t: float) -> dict:
        self._episode_seq += 1
        ep = {
            "episode_id": self._episode_seq,
            "scale": self.scale,
            "estimated_onset": float(t),
            "first_prompt_time": None,
            "first_active_time": None,
            "suspect_start": None,
            "suspect_end": None,
            "active_intervals": [],
            "end_time": None,
            "end_reason": None,
            "low_confidence": True,
        }
        self.episodes.append(ep)
        self._episode = ep
        return ep

    def _open_suspect(self, t: float) -> None:
        ep = self._episode or self._new_episode(t)
        if not self._suspect_open:
            self._suspect_open = True
            ep["first_prompt_time"] = float(t)
            ep["suspect_start"] = float(t)
            self.state = "SUSPECT"
            self._emit("SUSPECT_START", t, ep)

    def _open_active(self, t: float) -> None:
        ep = self._episode or self._new_episode(t)
        # If SUSPECT was not open yet, this same decision emits SUSPECT_START
        # first; ACTIVE_START must carry the snapshot after first_active_time.
        if not self._suspect_open:
            self._open_suspect(t)
            ep = self._episode
        if not self._active_open:
            self._active_open = True
            ep["first_active_time"] = (ep["first_active_time"]
                                       if ep["first_active_time"] is not None
                                       else float(t))
            ep["active_intervals"].append({"start": float(t), "end": None})
            self.state = "ACTIVE"
            self._emit("ACTIVE_START", t, ep)

    def _close_active(self, t: float, reason: str | None = None) -> None:
        if self._active_open and self._episode:
            ep = self._episode
            iv = ep["active_intervals"][-1]
            if iv["end"] is None:
                iv["end"] = float(t)
            self._active_open = False
            self._emit("ACTIVE_END", t, ep, reason)

    def _close_suspect(self, t: float, reason: str) -> None:
        if self._suspect_open and self._episode:
            ep = self._episode
            ep["suspect_end"] = float(t)
            self._suspect_open = False
            # end_time/end_reason must be in the SUSPECT_END snapshot.
            _close_episode(ep, t, reason)
            self._emit("SUSPECT_END", t, ep, reason)
            self._episode = None
            self.state = "NORMAL" if reason != "UNKNOWN" else "UNKNOWN"

    def _interrupt(self, t: float, reason: str,
                   count: bool = True) -> list[dict]:
        """End open output at t, clear all evidence, and report UNKNOWN."""
        before = len(self.transitions)
        ep = self._episode
        self._close_active(t, reason)
        if self._suspect_open:
            self._close_suspect(t, "UNKNOWN")
        elif self._episode is not None:
            # Evidence was accumulating but no output had opened yet.
            self._episode["end_time"] = float(t)
            self._episode["end_reason"] = "UNKNOWN"
            self._episode = None
        self._reset_runs()
        self.state = "UNKNOWN"
        if count:
            self.interruptions += 1
        self._emit("UNKNOWN", t, ep, reason)
        return self.transitions[before:]

    def _reset_runs(self) -> None:
        self._suspect_run = 0
        self._suspect_clear = 0
        self._active_run = 0
        self._active_clear = 0
        self._high_seen = False

    # ---- public API ---------------------------------------------------
    def update(self, decision_time: float, quality_state: str,
               p_u2: float | None = None) -> list[dict]:
        """Consume one window-end decision; return newly emitted transitions."""
        t = float(decision_time)
        before = len(self.transitions)
        tol = max(1e-6, self.step_sec * 1e-6)

        # A missing decision is a data interruption, not neutral evidence.
        # End the old output at the expected deadline, then process the late
        # observation as a fresh decision with zero carried counters.
        if self._last_decision_time is not None and \
                t - self._last_decision_time > self.step_sec + tol:
            deadline = self._last_decision_time + self.step_sec
            self.gap_interruptions += 1
            self._interrupt(deadline, "MISSING_DECISION", count=True)

        self.total_decisions += 1
        self._last_decision_time = t

        if quality_state in UNKNOWN_STATES:
            self.unknown_decisions += 1
            self._interrupt(t, "UNKNOWN", count=False)
            return self.transitions[before:]

        valid = quality_state == "VALID"
        flat_with = quality_state == "FLAT_WITH_HISTORY"
        flat_ever = quality_state == "FLAT_EVER_HISTORY"
        p = float(p_u2) if (valid and p_u2 is not None) else None

        suspect_cue = flat_with or flat_ever or (
            valid and p is not None and p >= self.params.suspect_p)
        active_cue = flat_with or (
            valid and p is not None and p >= self.params.active_enter_p)
        active_exit = valid and p is not None and p < self.params.active_exit_p

        if self.state in ("WAITING", "UNKNOWN"):
            self.state = "NORMAL"

        if suspect_cue or active_cue:
            if self._episode is None:
                self._new_episode(t)
            if flat_with or active_cue:
                self._high_seen = True
        if suspect_cue:
            self._suspect_run += 1
            self._suspect_clear = 0
        else:
            self._suspect_run = 0
            self._suspect_clear += 1

        # Strict consecutive counts: any decision that is not active evidence
        # breaks active_run; any decision that is not active-exit breaks
        # active_clear.  Neutral (L or exit <= p < enter) breaks BOTH runs.
        if active_cue:
            self._active_run += 1
            self._active_clear = 0
        elif active_exit:
            self._active_run = 0
            self._active_clear += 1
        else:
            self._active_run = 0
            self._active_clear = 0

        if not self._suspect_open and self._suspect_run >= self.params.suspect_on_count:
            self._open_suspect(t)
        if not self._active_open and self._active_run >= self.params.active_on_count:
            self._open_active(t)

        if self._active_open and self._active_clear >= self.params.active_off_count:
            self._close_active(t, "NORMAL")
            if self._suspect_open and self._suspect_clear >= self.params.suspect_off_count:
                self._close_suspect(t, "NORMAL")
            elif self._suspect_open:
                self.state = "SUSPECT"
            else:
                self.state = "NORMAL"

        if self._suspect_open and not self._active_open and \
                self._suspect_clear >= self.params.suspect_off_count:
            self._close_suspect(t, "NORMAL")

        if self._episode is not None and self._high_seen:
            self._episode["low_confidence"] = False
        return self.transitions[before:]

    def interrupt(self, at_time: float | None = None,
                  reason: str = "DATA_INTERRUPTION") -> list[dict]:
        """Close open output at the expected deadline and clear evidence.

        If ``at_time`` is omitted, use ``last_decision_time + step_sec``.
        This is the same interruption rule used for missing decisions.
        """
        t = at_time
        if t is None:
            t = ((self._last_decision_time + self.step_sec)
                 if self._last_decision_time is not None else 0.0)
        return self._interrupt(float(t), reason, count=True)

    def finish(self, case_end: float) -> list[dict]:
        """Close open evidence at the theoretical end of the last hold."""
        t = float(case_end)
        if self._last_decision_time is not None:
            t = min(t, self._last_decision_time + self.step_sec)
        before = len(self.transitions)
        self._close_active(t, "FINISH")
        if self._suspect_open:
            self._close_suspect(t, "FINISH")
        elif self._episode is not None:
            _close_episode(self._episode, t, "FINISH")
        self._episode = None
        self._reset_runs()
        if self.state != "UNKNOWN":
            self.state = "NORMAL"
        return self.transitions[before:]

    def suspect_intervals(self) -> list[dict]:
        out = []
        for ep in self.episodes:
            if ep["suspect_start"] is not None and ep["suspect_end"] is not None:
                out.append({
                    "scale": self.scale,
                    "episode_id": ep["episode_id"],
                    "start": ep["suspect_start"],
                    "end": ep["suspect_end"],
                    "decision_time": ep["first_prompt_time"],
                    "estimated_onset": ep["estimated_onset"],
                    "low_confidence": bool(ep["low_confidence"]),
                })
        return out

    def active_intervals(self) -> list[dict]:
        out = []
        for ep in self.episodes:
            for iv in ep["active_intervals"]:
                if iv["end"] is not None:
                    out.append({
                        "scale": self.scale,
                        "episode_id": ep["episode_id"],
                        "start": iv["start"],
                        "end": iv["end"],
                        "decision_time": iv["start"],
                        "first_active_time": ep["first_active_time"],
                        "estimated_onset": ep["estimated_onset"],
                        "first_prompt_time": ep["first_prompt_time"],
                        "low_confidence": bool(ep["low_confidence"]),
                    })
        return out

    def current_event(self) -> dict:
        return {"state": self.state,
                "episode": copy.deepcopy(self._episode)}
