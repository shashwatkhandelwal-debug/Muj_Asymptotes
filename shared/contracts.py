from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional
import time

class Severity(str, Enum):
    SOFT = "soft"
    HARD = "hard"

class SignalId(str, Enum):
    ELA = "ela"
    EXIF = "exif"
    CROSSFIELD = "crossfield"
    FACE_MATCH = "face_match"
    PASSIVE_LIVENESS = "passive_liveness"
    WATCHLIST = "watchlist"
    GENAI_DOC = "genai_doc"
    FACE_DEEPFAKE = "face_deepfake"
    ACTIVE_LIVENESS = "active_liveness"
    NAME_MATCH = "name_match"
    VOICE_SPOOF = "voice_spoof"

@dataclass
class SignalResult:
    signal: SignalId
    raw_score: float
    confidence: float
    severity: Severity = Severity.SOFT
    triggered: bool = False
    penalty: Optional[float] = None
    label: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    ms: float = 0.0
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["signal"] = self.signal.value
        d["severity"] = self.severity.value
        return d

SIGNAL_WEIGHTS: dict[SignalId, float] = {
    SignalId.GENAI_DOC:       25.0,
    SignalId.FACE_DEEPFAKE:   25.0,
    SignalId.VOICE_SPOOF:     20.0,
    SignalId.NAME_MATCH:      15.0,
    SignalId.ACTIVE_LIVENESS: 15.0,
    SignalId.ELA:             15.0,
    SignalId.EXIF:            10.0,
    SignalId.CROSSFIELD:      15.0,
}

SIGNAL_TRIGGER: dict[SignalId, float] = {
    SignalId.GENAI_DOC:       0.60,
    SignalId.FACE_DEEPFAKE:   0.60,
    SignalId.VOICE_SPOOF:     0.55,
    SignalId.NAME_MATCH:      0.50,
    SignalId.ACTIVE_LIVENESS: 0.50,
    SignalId.ELA:             0.50,
    SignalId.EXIF:            0.50,
    SignalId.CROSSFIELD:      0.50,
}

HARD_SIGNALS: set[SignalId] = {
    SignalId.FACE_DEEPFAKE,
}

TIERS = [(0, 30, "clear"), (31, 69, "review"), (70, 100, "flagged")]
