# -*- coding: utf-8 -*-
"""B0 rule baseline: SI features + causal quality state, no learned model."""
from __future__ import annotations

from features.quality_states import (
    FLAT_CANDIDATE_STATES, LOW_CONFIDENCE_STATES, state_is_unknown,
)

RULE_VERSION = "b0_rule_si_v1"
UNKNOWN = "UNKNOWN"
PREDICT_LABELS = ("U0", "U1", "U2", "U3", UNKNOWN)
TACHYPNEA_RR = 18.0
NORMAL_RR_MIN = 7.0
SLOW_RR_MIN = 3.0


def predict_rule(features: dict, quality_state: str) -> dict:
    """Return one of U0/U1/U2/U3/UNKNOWN plus a rule reason.

    Flat windows are *not* discarded because A=0. A flat window with verified
    causal breathing history is a U2 candidate. Flat with no credible history,
    disconnect/GAP, saturation, polarity error, and excessive noise produce
    UNKNOWN, never a normal label.
    """
    if state_is_unknown(quality_state):
        return {"label": UNKNOWN, "reason": f"quality:{quality_state}",
                "confidence": "none", "flat_candidate": False}

    if quality_state in FLAT_CANDIDATE_STATES:
        low_conf = quality_state in LOW_CONFIDENCE_STATES
        return {"label": "U2", "reason": f"flat:{quality_state}",
                "confidence": "low" if low_conf else "high",
                "flat_candidate": True}

    if quality_state != "VALID":
        return {"label": UNKNOWN, "reason": f"unhandled_state:{quality_state}",
                "confidence": "none", "flat_candidate": False}

    rate = float(features.get("peak_rate", 0.0))
    if rate >= TACHYPNEA_RR:
        return {"label": "U1", "reason": "peak_rate>=18", "confidence": "rule",
                "flat_candidate": False}
    if rate >= NORMAL_RR_MIN:
        return {"label": "U0", "reason": "7<=peak_rate<18", "confidence": "rule",
                "flat_candidate": False}
    if rate >= SLOW_RR_MIN:
        return {"label": "U3", "reason": "3<=peak_rate<7", "confidence": "rule",
                "flat_candidate": False}
    # Not flat, no reliable peaks: RR<3 is ambiguous in the frozen labels.
    return {"label": UNKNOWN, "reason": "peak_rate<3_not_flat",
            "confidence": "none", "flat_candidate": False}
