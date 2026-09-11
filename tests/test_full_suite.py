"""
tests/test_full_suite.py
Full pytest suite for the HackMUJ 4.0 deepfake detection layer.
Covers: contracts, all 5 detectors, fusion security, audit chain,
pepper SSS, calibration, challenge orchestrator, orchestrator shape.
All tests are hermetic — no network calls, no real files required.
Fixtures generate synthetic inputs in temp directory.
"""
import os
import wave
import struct
import tempfile
import json
import sqlite3
import shutil
import pytest
import numpy as np
from pathlib import Path
from PIL import Image

from shared.contracts import (
    SignalResult, SignalId, Severity,
    SIGNAL_WEIGHTS, SIGNAL_TRIGGER, HARD_SIGNALS, TIERS
)
from modules.decision.fusion import fuse
from shared.audit_chain import chain_hash, append_entry, verify_chain
from shared.pepper_sss import generate_pepper, split_pepper, reconstruct_pepper
from modules.decision.calibration import fit_calibrator, apply_calibration
from modules.forensics.genai_detector import detect_genai_document
from modules.face.deepfake_detector import detect_face_deepfake
from modules.face.active_liveness import score_action
from modules.audio.asr import score_name_match
from modules.audio.antispoof import score_voice_spoof
from modules.face.challenge import issue_challenge, run_challenge, Challenge
from api.orchestrator import build_officer_summary
from shared.preprocessing import clahe_rgb


# ============================================================================
# GROUP 1 — Contracts (4 tests)
# ============================================================================

def test_signal_result_fields():
    res = SignalResult(
        signal=SignalId.GENAI_DOC,
        raw_score=0.75,
        confidence=0.82,
        severity=Severity.SOFT,
        triggered=True,
        penalty=18.75,
        label="Document appears AI-generated",
        evidence={"method": "fft_heuristic"},
        ok=True,
        ms=12.5,
        ts=1700000000.0,
    )
    d = res.to_dict()
    required_keys = [
        "signal", "raw_score", "confidence", "severity", "triggered",
        "penalty", "label", "evidence", "ok", "ms", "ts"
    ]
    for key in required_keys:
        assert key in d, f"Missing key '{key}' in to_dict()"

    assert isinstance(d["signal"], str)
    assert isinstance(d["severity"], str)
    assert d["signal"] == "genai_doc"
    assert d["severity"] == "soft"


def test_signal_result_hard_severity():
    res = SignalResult(
        signal=SignalId.FACE_DEEPFAKE,
        raw_score=0.9,
        confidence=0.9,
        severity=Severity.HARD,
    )
    d = res.to_dict()
    assert d["severity"] == "hard"


def test_hard_signals_set():
    assert SignalId.FACE_DEEPFAKE in HARD_SIGNALS
    assert SignalId.VOICE_SPOOF not in HARD_SIGNALS  # demoted to SOFT — insufficient OOS margin
    assert SignalId.GENAI_DOC not in HARD_SIGNALS


def test_signal_weights_complete():
    expected_signals = [
        SignalId.GENAI_DOC,
        SignalId.FACE_DEEPFAKE,
        SignalId.VOICE_SPOOF,
        SignalId.NAME_MATCH,
        SignalId.ACTIVE_LIVENESS,
    ]
    for sig in expected_signals:
        assert sig in SIGNAL_WEIGHTS, f"Missing {sig} in SIGNAL_WEIGHTS"
        assert SIGNAL_WEIGHTS[sig] > 0, f"Weight for {sig} must be > 0"


# ============================================================================
# GROUP 2 — Fusion security (6 tests)
# ============================================================================

def test_fusion_hard_floor_applied():
    deepfake_hit = SignalResult(
        signal=SignalId.FACE_DEEPFAKE,
        raw_score=0.95,
        confidence=0.95,
        severity=Severity.HARD,
        triggered=True,
    )
    result = fuse(5.0, [deepfake_hit])
    assert result["tier"] == "flagged"
    assert result["score"] >= 70.0
    assert "face_deepfake" in result["floors_applied"]


def test_fusion_floor_is_monotone():
    deepfake_hit = SignalResult(
        signal=SignalId.FACE_DEEPFAKE,
        raw_score=0.95,
        confidence=0.95,
        severity=Severity.HARD,
        triggered=True,
    )
    res_with_floor = fuse(5.0, [deepfake_hit])
    assert res_with_floor["score"] >= 70.0
    assert res_with_floor["tier"] == "flagged"

    try:
        HARD_SIGNALS.remove(SignalId.FACE_DEEPFAKE)
        res_without_floor = fuse(5.0, [deepfake_hit])
        assert res_without_floor["score"] < 31.0
        assert res_without_floor["tier"] == "clear"
    finally:
        HARD_SIGNALS.add(SignalId.FACE_DEEPFAKE)


def test_fusion_name_match_review_floor():
    name_mismatch = SignalResult(
        signal=SignalId.NAME_MATCH,
        raw_score=0.8,
        confidence=0.8,
        triggered=True,
    )
    result = fuse(2.0, [name_mismatch])
    assert result["score"] >= 31.0
    assert result["tier"] in ("review", "flagged")
    assert "name_match_review_floor" in result["floors_applied"]


def test_fusion_crashed_detector_no_penalty():
    crashed = SignalResult(
        signal=SignalId.GENAI_DOC,
        ok=False,
        raw_score=0.0,
        confidence=0.0,
        triggered=False,
    )
    result = fuse(10.0, [crashed])
    assert result["score"] == 10.0
    assert result["tier"] == "clear"


def test_fusion_soft_signal_no_floor():
    genai_hit = SignalResult(
        signal=SignalId.GENAI_DOC,
        raw_score=0.95,
        confidence=0.95,
        severity=Severity.SOFT,
        triggered=True,
    )
    result = fuse(5.0, [genai_hit])
    assert result["score"] < 70.0
    assert "face_deepfake" not in result["floors_applied"]
    assert "genai_doc" not in result["floors_applied"]


def test_fusion_voice_spoof_soft_signal():
    """After demotion, voice_spoof contributes weighted penalty but no hard floor."""
    voice_hit = SignalResult(
        signal=SignalId.VOICE_SPOOF,
        severity=Severity.SOFT,
        raw_score=0.9,
        confidence=0.9,
        triggered=True,
    )
    result = fuse(5.0, [voice_hit])
    # SOFT signal: no floor applied, weighted penalty only
    assert "voice_spoof" not in result["floors_applied"]
    # Score should reflect weighted contribution, not clamped to 70+
    assert result["score"] < 70.0


# ============================================================================
# GROUP 3 — Audit chain (5 tests)
# ============================================================================

def test_audit_chain_hash_deterministic():
    pepper = os.urandom(32)
    prev = "0" * 64
    payload = {"event": "verify", "score": 82.0}
    h1 = chain_hash(pepper, prev, payload)
    h2 = chain_hash(pepper, prev, payload)
    assert h1 == h2


def test_audit_chain_pepper_sensitivity():
    pepper_a = os.urandom(32)
    pepper_b = os.urandom(32)
    prev = "0" * 64
    payload = {"event": "verify", "score": 82.0}
    h1 = chain_hash(pepper_a, prev, payload)
    h2 = chain_hash(pepper_b, prev, payload)
    assert h1 != h2


def test_audit_append_and_verify_intact():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    pepper = os.urandom(32)
    append_entry(db, pepper, {"round": 1, "status": "clear"})
    append_entry(db, pepper, {"round": 2, "status": "review"})
    append_entry(db, pepper, {"round": 3, "status": "flagged"})

    intact, broken_row = verify_chain(db, pepper)
    assert intact is True
    assert broken_row == -1


def test_audit_tamper_detected():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    pepper = os.urandom(32)
    append_entry(db, pepper, {"round": 1, "status": "clear"})
    append_entry(db, pepper, {"round": 2, "status": "flagged"})
    append_entry(db, pepper, {"round": 3, "status": "clear"})

    intact, _ = verify_chain(db, pepper)
    assert intact is True

    cur = db.cursor()
    cur.execute("UPDATE audit_log SET entry_hash='deadbeef' WHERE rowid=2")
    db.commit()

    intact_after, broken_row = verify_chain(db, pepper)
    assert intact_after is False
    assert broken_row == 2


def test_audit_wrong_pepper_fails():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    pepper_a = os.urandom(32)
    pepper_b = os.urandom(32)
    append_entry(db, pepper_a, {"round": 1})
    append_entry(db, pepper_a, {"round": 2})

    intact, _ = verify_chain(db, pepper_b)
    assert intact is False


# ============================================================================
# GROUP 4 — Shamir SSS (4 tests)
# ============================================================================

def test_pepper_generate_length():
    p = generate_pepper()
    assert isinstance(p, bytes)
    assert len(p) == 32


def test_pepper_split_count():
    p = generate_pepper()
    shares = split_pepper(p, k=2, n=3)
    assert isinstance(shares, list)
    assert len(shares) == 3


def test_pepper_reconstruct_2_of_3():
    p = generate_pepper()
    shares = split_pepper(p, k=2, n=3)
    rec = reconstruct_pepper([shares[0], shares[2]])
    assert rec == p


def test_pepper_reconstruct_different_pair():
    p = generate_pepper()
    shares = split_pepper(p, k=2, n=3)
    rec = reconstruct_pepper([shares[1], shares[2]])
    assert rec == p


# ============================================================================
# GROUP 5 — Calibration (4 tests)
# ============================================================================

@pytest.fixture
def backup_calibration():
    p = Path("calibration.json")
    orig_content = p.read_text(encoding="utf-8") if p.exists() else None
    yield
    if orig_content is not None:
        p.write_text(orig_content, encoding="utf-8")
    elif p.exists():
        p.unlink()


def test_calibration_fit_platt(backup_calibration):
    np.random.seed(42)
    fakes = list(np.random.uniform(0.6, 1.0, 30))
    reals = list(np.random.uniform(0.0, 0.4, 30))
    scores = fakes + reals
    labels = [1] * 30 + [0] * 30

    fit_calibrator(scores, labels, "pytest_signal")
    cal = apply_calibration(0.8, "pytest_signal")
    assert 0.0 <= cal <= 1.0


def test_calibration_monotone(backup_calibration):
    np.random.seed(42)
    fakes = list(np.random.uniform(0.6, 1.0, 30))
    reals = list(np.random.uniform(0.0, 0.4, 30))
    scores = fakes + reals
    labels = [1] * 30 + [0] * 30

    fit_calibrator(scores, labels, "pytest_signal_mono")
    cal_high = apply_calibration(0.8, "pytest_signal_mono")
    cal_low = apply_calibration(0.2, "pytest_signal_mono")
    assert cal_high > cal_low


def test_calibration_signals_direction_positive():
    """
    Asserts sign and direction correctness for all core detector calibrations.
    Ensures that for every signal (genai_doc, face_deepfake, voice_spoof):
      1. The fitted Platt scaling coefficient is strictly positive (coef > 0).
      2. Calibration confidence is strictly increasing with raw score:
         apply_calibration(0.85) > apply_calibration(0.15), preventing inverted sigmoids.
    """
    calib_path = Path("calibration.json")
    assert calib_path.exists(), "calibration.json must exist"
    with open(calib_path, "r", encoding="utf-8") as f:
        calibs = json.load(f)

    signals_to_verify = ["genai_doc", "face_deepfake", "voice_spoof"]
    for sig in signals_to_verify:
        assert sig in calibs, f"Signal {sig} missing from calibration.json"
        entry = calibs[sig]
        assert "coef" in entry, f"Signal {sig} missing 'coef'"
        coef = entry["coef"]
        assert coef > 0, f"Signal {sig} has negative or zero coef: {coef} (must be positive to prevent inverted sigmoid)"

        # Assert monotonic increase: higher raw score MUST produce higher calibrated confidence
        cal_high = apply_calibration(0.85, sig)
        cal_low = apply_calibration(0.15, sig)
        assert cal_high > cal_low, (
            f"Signal {sig} failed monotonicity check: apply_calibration(0.85)={cal_high} "
            f"not greater than apply_calibration(0.15)={cal_low}"
        )


def test_calibration_threshold_fallback(backup_calibration):
    fit_calibrator([0.5, 0.6], [1, 0], "tiny_pytest")
    cal = apply_calibration(0.9, "tiny_pytest")
    assert 0.0 <= cal <= 1.0


def test_calibration_missing_json_returns_raw(backup_calibration):
    p = Path("calibration.json")
    if p.exists():
        p.unlink()
    raw = 0.7
    cal = apply_calibration(raw, "nonexistent_signal")
    assert cal == raw


# ============================================================================
# GROUP 6 — Detectors (6 tests)
# ============================================================================

def test_genai_detector_returns_signal_result():
    arr = np.random.randint(0, 255, (300, 200, 3), dtype=np.uint8)
    tmp_dir = os.environ.get("TEMP", tempfile.gettempdir())
    img_path = os.path.join(tmp_dir, f"test_genai_{os.getpid()}.jpg")
    Image.fromarray(arr).save(img_path)

    try:
        res = detect_genai_document(img_path)
        assert res.signal == SignalId.GENAI_DOC
        assert 0.0 <= res.raw_score <= 1.0
        assert res.ok is True
        assert res.ms > 0
        hm_path = res.evidence.get("heatmap_path")
        assert isinstance(hm_path, str) and len(hm_path) > 0
        assert os.path.exists(hm_path)
    finally:
        if os.path.exists(img_path):
            os.unlink(img_path)


def test_genai_detector_heatmap_method_recorded():
    arr = np.random.randint(0, 255, (300, 200, 3), dtype=np.uint8)
    tmp_dir = os.environ.get("TEMP", tempfile.gettempdir())
    img_path = os.path.join(tmp_dir, f"test_genai_m_{os.getpid()}.jpg")
    Image.fromarray(arr).save(img_path)

    try:
        res = detect_genai_document(img_path)
        assert bool(res.evidence.get("method"))
        assert res.evidence.get("method") in ("clip_probe", "clip_zeroshot", "fft_heuristic")
    finally:
        if os.path.exists(img_path):
            os.unlink(img_path)


def test_deepfake_detector_empty_frames():
    res = detect_face_deepfake([])
    assert res.ok is False


def test_deepfake_detector_random_frames():
    frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(20)]
    res = detect_face_deepfake(frames)
    assert res.signal == SignalId.FACE_DEEPFAKE
    assert res.severity == Severity.HARD
    assert 0.0 <= res.raw_score <= 1.0
    assert len(res.evidence.get("per_frame_scores", [])) <= 15


def test_antispoof_silent_wav():
    tmp_dir = os.environ.get("TEMP", tempfile.gettempdir())
    wav_path = os.path.join(tmp_dir, f"test_silent_{os.getpid()}.wav")
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<" + "h" * 16000, *([0] * 16000)))

    try:
        res = score_voice_spoof(wav_path)
        assert res.signal == SignalId.VOICE_SPOOF
        assert res.severity == Severity.SOFT  # demoted from HARD
        assert 0.0 <= res.raw_score <= 1.0
        assert res.ok is True
        assert res.evidence.get("method") in ("onnx_aasist", "mfcc_heuristic", "fft_fallback")
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)


def test_active_liveness_random_frames():
    frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(30)]
    res = score_action(frames, "blink_twice")
    assert res.ok is True
    assert 0.0 <= res.raw_score <= 1.0


# ============================================================================
# GROUP 7 — ASR / Name match (4 tests)
# ============================================================================

def test_asr_silent_wav_triggers_mismatch():
    tmp_dir = os.environ.get("TEMP", tempfile.gettempdir())
    wav_path = os.path.join(tmp_dir, f"test_asr_silent_{os.getpid()}.wav")
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<" + "h" * 16000, *([0] * 16000)))

    try:
        res = score_name_match(wav_path, "Test User", "1234", "2026-01-01")
        assert res.triggered is True
        assert isinstance(res.evidence.get("transcript"), str)
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)


def test_asr_nonce_absent_penalty():
    tmp_dir = os.environ.get("TEMP", tempfile.gettempdir())
    wav_path = os.path.join(tmp_dir, f"test_asr_nonce_{os.getpid()}.wav")
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<" + "h" * 16000, *([0] * 16000)))

    try:
        res = score_name_match(wav_path, "X", "9999", "2026-01-01")
        assert res.raw_score >= 0.5
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)


def test_asr_signal_id_correct():
    tmp_dir = os.environ.get("TEMP", tempfile.gettempdir())
    wav_path = os.path.join(tmp_dir, f"test_asr_sig_{os.getpid()}.wav")
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<" + "h" * 16000, *([0] * 16000)))

    try:
        res = score_name_match(wav_path, "Alice", "0000", "2026-09-11")
        assert res.signal == SignalId.NAME_MATCH
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)


def test_asr_ok_true_on_silent():
    tmp_dir = os.environ.get("TEMP", tempfile.gettempdir())
    wav_path = os.path.join(tmp_dir, f"test_asr_ok_{os.getpid()}.wav")
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<" + "h" * 16000, *([0] * 16000)))

    try:
        res = score_name_match(wav_path, "Alice", "0000", "2026-09-11")
        assert res.ok is True
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)


# ============================================================================
# GROUP 8 — Challenge orchestrator (3 tests)
# ============================================================================

def test_challenge_issue_fields():
    ch = issue_challenge("Priya Sharma")
    assert ch.action in ("blink_twice", "turn_left", "turn_right")
    assert len(ch.nonce) == 4 and ch.nonce.isdigit()
    assert "Priya Sharma" in ch.spoken_phrase
    assert ch.nonce in ch.spoken_phrase
    assert len(ch.date_str) == 10


def test_challenge_run_nonexistent_files():
    ch = issue_challenge("Priya Sharma")
    results = run_challenge("/nonexistent.mp4", "/nonexistent.wav", ch, "Priya Sharma")
    assert isinstance(results, list)
    assert len(results) == 3
    for r in results:
        assert isinstance(r, SignalResult)


def test_challenge_signal_ids():
    ch = issue_challenge("Priya Sharma")
    results = run_challenge("/nonexistent.mp4", "/nonexistent.wav", ch, "Priya Sharma")
    signals = {r.signal for r in results}
    assert signals == {SignalId.ACTIVE_LIVENESS, SignalId.NAME_MATCH, SignalId.VOICE_SPOOF}


# ============================================================================
# GROUP 9 — Orchestrator response shape (3 tests)
# ============================================================================

def test_orchestrator_summary_read_only():
    fused = {
        "score": 82.0,
        "tier": "flagged",
        "floors_applied": ["face_deepfake"],
        "lines": [
            {
                "signal": "face_deepfake",
                "triggered": True,
                "label": "Deepfake detected",
                "confidence": 0.91,
            }
        ],
    }
    summary = build_officer_summary(fused)
    assert isinstance(summary, str)
    assert "FLAGGED" in summary
    assert fused["score"] == 82.0
    assert fused["tier"] == "flagged"


def test_orchestrator_summary_not_triggered_excluded():
    fused = {
        "score": 10.0,
        "tier": "clear",
        "floors_applied": [],
        "lines": [
            {
                "signal": "face_deepfake",
                "triggered": False,
                "label": "Face appears genuine",
                "confidence": 0.1,
            }
        ],
    }
    summary = build_officer_summary(fused)
    assert "face_deepfake" not in summary


def test_contracts_not_modified_by_fusion():
    original_hard = set(HARD_SIGNALS)
    deepfake_hit = SignalResult(
        signal=SignalId.FACE_DEEPFAKE,
        raw_score=0.95,
        confidence=0.95,
        severity=Severity.HARD,
        triggered=True,
    )
    _ = fuse(5.0, [deepfake_hit])
    assert HARD_SIGNALS == original_hard


# ============================================================================
# GROUP 10 — Integration smoke (2 tests)
# ============================================================================

def test_all_new_signals_importable():
    from modules.forensics.genai_detector import detect_genai_document
    from modules.face.deepfake_detector import detect_face_deepfake
    from modules.face.active_liveness import score_action
    from modules.audio.asr import score_name_match
    from modules.audio.antispoof import score_voice_spoof
    from modules.decision.fusion import fuse
    from modules.decision.calibration import apply_calibration
    from shared.audit_chain import chain_hash, verify_chain
    from shared.pepper_sss import generate_pepper, split_pepper, reconstruct_pepper
    from shared.preprocessing import clahe_rgb

    assert callable(detect_genai_document)
    assert callable(detect_face_deepfake)
    assert callable(score_action)
    assert callable(score_name_match)
    assert callable(score_voice_spoof)
    assert callable(fuse)
    assert callable(apply_calibration)
    assert callable(chain_hash)
    assert callable(verify_chain)
    assert callable(generate_pepper)
    assert callable(split_pepper)
    assert callable(reconstruct_pepper)
    assert callable(clahe_rgb)


def test_signal_result_ms_set():
    arr = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    tmp_dir = os.environ.get("TEMP", tempfile.gettempdir())
    img_path = os.path.join(tmp_dir, f"test_ms_{os.getpid()}.jpg")
    Image.fromarray(arr).save(img_path)
    try:
        res = detect_genai_document(img_path)
        assert res.ms > 0
    finally:
        if os.path.exists(img_path):
            os.unlink(img_path)


# ============================================================================
# GROUP 11 — Enhancement tests (8 new tests)
# ============================================================================

def test_replay_attack_script_exists():
    assert Path("scripts/demo_replay_attack.py").exists()

def test_temporal_consistency_in_evidence():
    frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
              for _ in range(20)]
    r = detect_face_deepfake(frames)
    assert "temporal_consistency_score" in r.evidence
    assert 0 <= r.evidence["temporal_consistency_score"] <= 1

def test_provenance_module():
    from modules.forensics.provenance import check_provenance
    arr = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    Image.fromarray(arr).save(tmp.name)
    tmp.close()
    result = check_provenance(tmp.name)
    assert "provenance_score" in result
    assert "provenance_flags" in result
    assert 0 <= result["provenance_score"] <= 1
    os.unlink(tmp.name)

def test_genai_provenance_in_evidence():
    arr = np.random.randint(0, 255, (300, 200, 3), dtype=np.uint8)
    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    Image.fromarray(arr).save(tmp.name)
    tmp.close()
    r = detect_genai_document(tmp.name)
    assert "provenance_score" in r.evidence
    assert "provenance_flags" in r.evidence
    os.unlink(tmp.name)

def test_cross_modal_consistency_all_match():
    from modules.decision.cross_modal import check_cross_modal_consistency
    r = check_cross_modal_consistency("Priya Sharma", "priya sharma")
    assert r["all_consistent"] is True
    assert r["consistency_score"] == 0.0

def test_cross_modal_consistency_mismatch():
    from modules.decision.cross_modal import check_cross_modal_consistency
    r = check_cross_modal_consistency("Priya Sharma", "John Smith")
    assert r["all_consistent"] is False
    assert len(r["mismatches"]) >= 1

def test_temporal_jitter_higher_than_smooth():
    smooth = np.full((224, 224, 3), 128, dtype=np.uint8)
    smooth_frames = [smooth.copy() for _ in range(20)]
    jitter_frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                     for _ in range(20)]
    r_smooth = detect_face_deepfake(smooth_frames)
    r_jitter = detect_face_deepfake(jitter_frames)
    assert r_jitter.raw_score >= r_smooth.raw_score

def test_cross_modal_three_way():
    from modules.decision.cross_modal import check_cross_modal_consistency
    # All three match
    r = check_cross_modal_consistency("Priya Sharma", "priya sharma",
                                      "PRIYA SHARMA")
    assert r["all_consistent"] is True
    # Doc mismatch
    r2 = check_cross_modal_consistency("Priya Sharma", "Priya Sharma",
                                       "Rahul Verma")
    assert r2["all_consistent"] is False

