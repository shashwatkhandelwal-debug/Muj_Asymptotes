import os
import sys
import time
import wave
import string
import re
import logging
from pathlib import Path

# Ensure repo root is on sys.path
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
from rapidfuzz.fuzz import token_set_ratio

from shared.contracts import SignalResult, SignalId, Severity, SIGNAL_TRIGGER
from modules.decision.calibration import apply_calibration

logger = logging.getLogger(__name__)

_WHISPER_MODEL = None
_WHISPER_TRIED = False

def _get_whisper_model():
    global _WHISPER_MODEL, _WHISPER_TRIED
    if _WHISPER_TRIED:
        return _WHISPER_MODEL
    _WHISPER_TRIED = True
    try:
        from faster_whisper import WhisperModel
        # Load local cached model strictly without network calls
        _WHISPER_MODEL = WhisperModel("tiny", device="cpu", compute_type="int8", local_files_only=True)
    except Exception as e:
        logger.info("faster-whisper local model not available: %s", e)
        _WHISPER_MODEL = None
    return _WHISPER_MODEL

def _normalize_text(text: str) -> str:
    # Lowercase and remove punctuation
    text = text.lower()
    return re.sub(r"[^\w\s]", " ", text)

def score_name_match(audio_path: str, ocr_name: str,
                     expected_nonce: str, expected_date: str) -> SignalResult:
    """
    IN:  audio_path = path to WAV or MP4 audio.
         ocr_name = name extracted from document OCR (e.g. "Priya Sharma").
         expected_nonce = 4-digit string from Challenge.
         expected_date = ISO date string from Challenge.
    OUT: SignalResult(
           signal=SignalId.NAME_MATCH,
           raw_score=mismatch_probability in [0,1],
           confidence=apply_calibration(raw_score, "name_match"),
           triggered=(raw_score > SIGNAL_TRIGGER[SignalId.NAME_MATCH]),
           label="Spoken name matches document" if not triggered
                 else "MISMATCH: spoken name does not match document",
           evidence={
             "transcript": str,
             "name_ratio": float,
             "nonce_present": bool,
             "date_present": bool
           }
         )
    Measure wall time and set result.ms.
    """
    t0 = time.perf_counter()
    transcript = ""
    error_msg = None

    if not os.path.exists(audio_path):
        elapsed_ms = max(0.1, (time.perf_counter() - t0) * 1000.0)
        return SignalResult(
            signal=SignalId.NAME_MATCH,
            raw_score=1.0,
            confidence=apply_calibration(1.0, "name_match"),
            severity=Severity.SOFT,
            triggered=True,
            label="MISMATCH: spoken name does not match document",
            evidence={
                "transcript": "",
                "name_ratio": 0.0,
                "nonce_present": False,
                "date_present": False,
                "error": f"Audio file not found: {audio_path}",
            },
            ok=False,
            ms=elapsed_ms,
        )

    model = _get_whisper_model()
    if model is not None:
        try:
            segments, _ = model.transcribe(audio_path, beam_size=1)
            transcript = " ".join(seg.text for seg in segments).strip()
        except Exception as e:
            logger.warning("Whisper transcription failed: %s", e)
            error_msg = str(e)
            transcript = ""
    else:
        error_msg = "faster_whisper_unavailable"
        transcript = ""

    norm_trans = _normalize_text(transcript)
    norm_ocr = _normalize_text(ocr_name)

    if norm_trans.strip():
        name_ratio = float(token_set_ratio(norm_trans, norm_ocr) / 100.0)
    else:
        name_ratio = 0.0

    raw_score = 1.0 - name_ratio

    # Anti-replay check
    norm_nonce = _normalize_text(expected_nonce).strip()
    norm_date = _normalize_text(expected_date).strip()
    clean_date = re.sub(r"\D", "", expected_date)

    nonce_present = bool(norm_nonce and norm_nonce in norm_trans)
    date_present = bool(
        (norm_date and norm_date in norm_trans)
        or (clean_date and clean_date in re.sub(r"\D", "", norm_trans))
    )

    if not nonce_present:
        raw_score += 0.3
    if not date_present:
        raw_score += 0.3

    raw_score = float(np.clip(raw_score, 0.0, 1.0))
    confidence = apply_calibration(raw_score, "name_match")
    triggered = bool(raw_score > SIGNAL_TRIGGER[SignalId.NAME_MATCH])
    label = (
        "Spoken name matches document"
        if not triggered
        else "MISMATCH: spoken name does not match document"
    )
    elapsed_ms = max(0.1, (time.perf_counter() - t0) * 1000.0)

    evidence = {
        "transcript": transcript,
        "name_ratio": name_ratio,
        "nonce_present": nonce_present,
        "date_present": date_present,
    }
    if error_msg:
        evidence["error"] = error_msg

    return SignalResult(
        signal=SignalId.NAME_MATCH,
        raw_score=raw_score,
        confidence=confidence,
        severity=Severity.SOFT,
        triggered=triggered,
        label=label,
        evidence=evidence,
        ok=True,
        ms=elapsed_ms,
    )

if __name__ == "__main__":
    tmp_dir = "/tmp" if os.name != "nt" else os.environ.get("TEMP", "C:/tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    test_wav = os.path.join(tmp_dir, "test_asr.wav")

    # Write 1-second silent 16kHz WAV
    with wave.open(test_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00" * 32000)

    res = score_name_match(test_wav, "Test User", "1234", "2026-01-01")
    assert isinstance(res, SignalResult)
    assert 0.0 <= res.raw_score <= 1.0
    assert res.triggered is True
    print("ASR NAME MATCH OK")
