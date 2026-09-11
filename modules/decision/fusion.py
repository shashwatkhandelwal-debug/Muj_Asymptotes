import sys
from pathlib import Path

# Ensure repo root is on sys.path
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from shared.contracts import (
    SignalResult, SignalId, Severity,
    SIGNAL_WEIGHTS, SIGNAL_TRIGGER, HARD_SIGNALS, TIERS
)

def fuse(existing_score: float, new_signals: list[SignalResult]) -> dict:
    """
    ALGORITHM:
    1. For each signal in new_signals:
       a. If signal.ok is False: skip (fail-open — do not penalise a detector crash).
       b. If signal.triggered is True:
          penalty = signal.penalty if signal.penalty is not None
                    else signal.raw_score * SIGNAL_WEIGHTS.get(signal.signal, 0)
       c. Else: penalty = 0.0
       d. Accumulate penalty. Record a line dict:
          {"signal": signal.signal.value, "penalty": penalty,
           "label": signal.label, "confidence": signal.confidence,
           "severity": signal.severity.value, "triggered": signal.triggered,
           "evidence": signal.evidence}

    2. raw_total = existing_score + sum(all penalties), clamped to [0, 100].

    3. FLOORS (apply AFTER summation, monotone — can only increase score):
       floors_applied = []
       a. For any triggered signal where signal.signal in HARD_SIGNALS:
          if raw_total < 70: raw_total = 70; floors_applied.append(signal.signal.value)
       b. If NAME_MATCH signal is triggered and raw_total < 31:
          raw_total = 31; floors_applied.append("name_match_review_floor")

    4. Determine tier from TIERS: first (lo, hi, label) where lo <= raw_total <= hi.

    5. Return:
       {
         "score": raw_total,
         "tier": tier_label,
         "lines": [list of line dicts from step 1],
         "floors_applied": floors_applied
       }
    """
    lines = []
    total_penalty = 0.0

    # 1. Process signals
    for signal in new_signals:
        if not signal.ok:
            # fail-open: skip crashed detector
            continue

        if signal.triggered:
            if signal.penalty is not None:
                penalty = float(signal.penalty)
            else:
                penalty = float(signal.raw_score * SIGNAL_WEIGHTS.get(signal.signal, 0.0))
        else:
            penalty = 0.0

        total_penalty += penalty
        lines.append({
            "signal": signal.signal.value,
            "penalty": penalty,
            "label": signal.label,
            "confidence": signal.confidence,
            "severity": signal.severity.value,
            "triggered": signal.triggered,
            "evidence": signal.evidence,
        })

    # 2. Raw total
    raw_total = float(min(100.0, max(0.0, existing_score + total_penalty)))

    # 3. Floors
    floors_applied: list[str] = []
    # a. Hard signals floor
    for signal in new_signals:
        if signal.ok and signal.triggered and signal.signal in HARD_SIGNALS:
            if raw_total < 70:
                raw_total = 70.0
                if signal.signal.value not in floors_applied:
                    floors_applied.append(signal.signal.value)

    # b. Name match floor
    for signal in new_signals:
        if signal.ok and signal.triggered and signal.signal == SignalId.NAME_MATCH:
            if raw_total < 31:
                raw_total = 31.0
                if "name_match_review_floor" not in floors_applied:
                    floors_applied.append("name_match_review_floor")

    # 4. Determine tier
    tier_label = "clear"
    for lo, hi, label in TIERS:
        if lo <= raw_total <= hi:
            tier_label = label
            break
    else:
        if raw_total > 69:
            tier_label = "flagged"
        elif raw_total >= 31:
            tier_label = "review"
        else:
            tier_label = "clear"

    return {
        "score": raw_total,
        "tier": tier_label,
        "lines": lines,
        "floors_applied": floors_applied,
    }

if __name__ == "__main__":
    # Test case 1: Single FACE_DEEPFAKE with raw_score=0.95, others clean
    df_signal = SignalResult(
        signal=SignalId.FACE_DEEPFAKE,
        raw_score=0.95,
        confidence=0.95,
        severity=Severity.HARD,
        triggered=True,
        label="Deepfake detected"
    )
    clean_signals = [
        SignalResult(signal=SignalId.GENAI_DOC, raw_score=0.0, confidence=0.0, triggered=False),
        SignalResult(signal=SignalId.ACTIVE_LIVENESS, raw_score=0.0, confidence=0.0, triggered=False),
        SignalResult(signal=SignalId.NAME_MATCH, raw_score=0.0, confidence=0.0, triggered=False),
        SignalResult(signal=SignalId.VOICE_SPOOF, raw_score=0.0, confidence=0.0, severity=Severity.HARD, triggered=False),
    ]
    all_signals = [df_signal] + clean_signals

    res1 = fuse(5.0, all_signals)
    assert res1["tier"] == "flagged", f"Expected flagged, got {res1['tier']}"
    assert "face_deepfake" in res1["floors_applied"], f"Expected face_deepfake in floors: {res1['floors_applied']}"
    assert res1["score"] == 70.0, f"Expected 70.0, got {res1['score']}"

    # Test case 2: Temporarily remove FACE_DEEPFAKE from HARD_SIGNALS
    try:
        HARD_SIGNALS.remove(SignalId.FACE_DEEPFAKE)
        res2 = fuse(5.0, all_signals)
        assert res2["score"] <= 30.0, f"Expected score <= 30, got {res2['score']}"
        assert res2["tier"] == "clear", f"Expected clear, got {res2['tier']}"
    finally:
        HARD_SIGNALS.add(SignalId.FACE_DEEPFAKE)

    print("FUSION OK")
