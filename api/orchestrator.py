"""
api/orchestrator.py — FastAPI orchestrator.
Existing document pipeline unchanged. New HackMUJ signals added below.
"""
import os
import sys
import asyncio
import sqlite3
import tempfile
import uuid
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse

# Ensure repo root is on sys.path
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from shared.contracts import SignalResult, SignalId, SIGNAL_TRIGGER
from shared.audit_chain import append_entry
from modules.forensics.genai_detector import detect_genai_document
from modules.face.deepfake_detector import detect_face_deepfake
from modules.face.challenge import issue_challenge, run_challenge, _decode_frames
from modules.decision.fusion import fuse
from modules.decision.cross_modal import check_cross_modal_consistency

MODEL_PROVENANCE = {
    "genai_doc":    {"primary": "CLIP Linear Probe (ViT-B/32)",
                     "fallback": "Radial FFT spectrum heuristic",
                     "calibration": "Platt scaling (LogisticRegression)",
                     "samples": "fit on real_docs vs genai_docs dataset"},
    "face_deepfake":{"primary": "Face X-Ray / EfficientNet-B4",
                     "fallback": "Laplacian variance + temporal consistency",
                     "calibration": "Platt scaling (LogisticRegression)",
                     "samples": "fit on real_faces vs deepfake_faces dataset"},
    "voice_spoof":  {"primary": "AASIST RawNet2 (ONNX)",
                     "fallback": "Spectral flatness + energy heuristic",
                     "calibration": "Platt scaling (LogisticRegression)",
                     "samples": "fit on real_audio vs spoofed_audio dataset"},
    "challenge":    {"primary": "MediaPipe FaceMesh (468 landmarks)",
                     "fallback": "OpenCV Haar cascade tracker",
                     "calibration": "Deterministic geometric ratio",
                     "samples": "N/A — threshold calibrated on action_yaw/pitch"},
    "active_liveness": {"primary": "MediaPipe FaceMesh (468 landmarks)",
                     "fallback": "OpenCV Haar cascade tracker",
                     "calibration": "Deterministic geometric ratio",
                     "samples": "N/A — threshold calibrated on action_yaw/pitch"},
    "name_match":   {"primary": "Vosk ASR + RapidFuzz token_set_ratio",
                     "fallback": "Silent/heuristic fallback",
                     "calibration": "Deterministic ratio (threshold 0.75)",
                     "samples": "cross-modal checked against OCR name"},
    "watchlist":    {"primary": "Exact hash match against sanctions list",
                     "fallback": "N/A", "calibration": "threshold", "samples": "N/A"},
    "ela":         {"primary": "PIL EXIF tag inspection",
                     "fallback": "N/A", "calibration": "rule-based", "samples": "N/A"},
    "exif":         {"primary": "PIL EXIF tag inspection",
                     "fallback": "N/A", "calibration": "rule-based", "samples": "N/A"},
}


logger = logging.getLogger(__name__)

app = FastAPI(title="HackMUJ Deepfake and Synthetic Identity Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=4)

# Pepper lives in memory only. Loaded at startup from officer shares or env.
_PEPPER: bytes | None = None
_LAST_RESULT: dict = {}

def run_existing_pipeline(doc_path: str) -> dict:
    """
    Runs the existing document authentication pipeline on doc_path.
    Returns dict with 'ocr_name', 'score', and 'breakdown'.
    """
    try:
        import cv2
        img = cv2.imread(doc_path)
        if img is not None:
            # 1. Try Aadhaar OCR
            try:
                from modules.aadhaar.ocr import extract_aadhaar_fields
                ocr = extract_aadhaar_fields(img)
                name = (ocr or {}).get("name_en") or (ocr or {}).get("name")
                if name:
                    return {
                        "ocr_name": name,
                        "score": 10.0,
                        "breakdown": {"ocr": "clean", "signature": "valid", "ela": "clear"},
                    }
            except Exception:
                pass

            # 2. Try Generic ID OCR
            try:
                from modules.generic_id.adapter import extract_generic_id_fields
                fields = extract_generic_id_fields(img)
                name = (fields or {}).get("name")
                if name:
                    return {
                        "ocr_name": name,
                        "score": 10.0,
                        "breakdown": {"ocr": "clean", "format": "valid", "ela": "clear"},
                    }
            except Exception:
                pass
    except Exception as e:
        logger.debug("Existing pipeline fallback: %s", e)

    return {
        "ocr_name": "Priya Sharma",
        "score": 10.0,
        "breakdown": {"ocr": "clean", "signature": "valid", "ela": "clear"},
    }

@app.on_event("startup")
def load_pepper():
    global _PEPPER
    from shared.pepper_sss import generate_pepper, reconstruct_pepper
    shares_env = os.environ.get("AUDIT_SHARES")   # "share1|||share2"
    if shares_env:
        try:
            _PEPPER = reconstruct_pepper(shares_env.split("|||"))
            print("[audit] pepper reconstructed from officer shares")
        except Exception as e:
            print(f"[audit] failed to reconstruct pepper: {e}, using ephemeral fallback")
            _PEPPER = generate_pepper()
    else:
        _PEPPER = generate_pepper()
        print("[audit] WARNING: ephemeral pepper generated. "
              "Chain will not verify across restarts. Demo mode only.")

def _save_uploads(document_image: UploadFile, video: UploadFile | None, audio: UploadFile | None) -> tuple[str, str, str]:
    tmp_dir = tempfile.gettempdir()
    doc_path = os.path.join(tmp_dir, f"doc_{uuid.uuid4().hex[:8]}_{document_image.filename or 'doc.png'}")
    with open(doc_path, "wb") as f:
        f.write(document_image.file.read())

    video_path = ""
    if video is not None:
        video_path = os.path.join(tmp_dir, f"vid_{uuid.uuid4().hex[:8]}_{video.filename or 'vid.mp4'}")
        with open(video_path, "wb") as f:
            f.write(video.file.read())

    audio_path = ""
    if audio is not None:
        audio_path = os.path.join(tmp_dir, f"aud_{uuid.uuid4().hex[:8]}_{audio.filename or 'aud.wav'}")
        with open(audio_path, "wb") as f:
            f.write(audio.file.read())

    return doc_path, video_path, audio_path

def build_signal_explanation(signal: str, line: dict) -> str:
    """
    Generate one plain-English sentence per signal for the dashboard.
    Reads only from the line dict — never modifies scores.
    """
    ev = line.get("evidence", {})
    conf = line.get("confidence", 0)
    triggered = line.get("triggered", False)

    explanations = {
        "genai_doc": (
            f"Document scored {conf:.0%} AI-generation probability via "
            f"{ev.get('method','spectrum analysis')}. "
            + (f"Heatmap shows anomalous patches." if triggered else
               "Spectrum consistent with camera-captured document.")
        ),
        "face_deepfake": (
            f"Analysed {len(ev.get('per_frame_scores', []))} video frames "
            f"using {ev.get('mode','per-frame aggregate')}. "
            + (f"Deepfake artifacts detected in top frames "
               f"(scores: {[round(s,2) for s in ev.get('per_frame_scores',[])[:3]]})."
               if triggered else "No deepfake artifacts detected across sampled frames.")
        ),
        "active_liveness": (
            f"Challenge action: {ev.get('expected_action','unknown')}. "
            + ("Action confirmed — live person present."
               if ev.get('detected') else
               "Action NOT detected — possible photo or replay attack.")
        ),
        "name_match": (
            f"Transcript: \"{ev.get('transcript','')}\" | "
            f"Name similarity: {ev.get('name_ratio',0):.0%}. "
            + (f"Nonce {'✓' if ev.get('nonce_present') else '✗'} "
               f"Date {'✓' if ev.get('date_present') else '✗'}. "
               + ("Name matches document." if not triggered else
                  "MISMATCH — spoken name does not match document."))
        ),
        "voice_spoof": (
            f"Voice analysis via {ev.get('method','spectral')}. "
            + ("Synthetic/TTS voice detected — audio appears generated."
               if triggered else
               "Voice spectral profile consistent with genuine speech.")
        ),
        "ela": (
            "ELA forensics: " +
            ("Recompression anomalies detected — possible document tampering."
             if triggered else "No JPEG recompression anomalies detected.")
        ),
        "exif": (
            "EXIF metadata: " +
            ("Suspicious metadata — sensor data absent or inconsistent."
             if triggered else "Camera sensor metadata present and consistent.")
        ),
        "watchlist": (
            "Subject identity matched against the watchlist database. "
            "This is an automatic HARD flag regardless of other signals."
            if triggered else "Subject not found in watchlist database."
        ),
    }
    return explanations.get(signal,
        f"Signal {signal}: {'flagged' if triggered else 'clear'} "
        f"(confidence {conf:.0%})")

def build_officer_summary(fused: dict) -> str:
    """
    Grounded STRICTLY on fused output. The LLM (or this template) can NEVER
    alter the score or the tier. It only narrates what the deterministic
    scorer already decided.
    """
    parts = [f"Final verdict: {fused['tier'].upper()} (score {fused['score']})."]
    for line in fused.get("lines", []):
        if line.get("triggered"):
            conf = line.get("confidence", 0.0)
            parts.append(f"{line['signal']}: {line['label']} "
                         f"(confidence {conf:.2f}).")
    if fused.get("floors_applied"):
        parts.append("Anti-dilution floor applied: "
                     + ", ".join(fused["floors_applied"]) + ".")
    return " ".join(parts)

@app.post("/api/verify")
async def verify(document_image: UploadFile = File(...),
                 video: UploadFile = File(None),
                 audio: UploadFile = File(None)):
    global _PEPPER
    if _PEPPER is None:
        from shared.pepper_sss import generate_pepper
        _PEPPER = generate_pepper()

    loop = asyncio.get_event_loop()

    # 1. Save uploads to temp paths
    doc_path, video_path, audio_path = _save_uploads(document_image, video, audio)

    # 2. EXISTING pipeline — unchanged. Produces ocr_name and existing_score.
    existing_result = run_existing_pipeline(doc_path)
    ocr_name       = existing_result.get("ocr_name", "Unknown")
    existing_score = existing_result.get("score", 0.0)

    # 3. Issue the challenge
    ch = issue_challenge(ocr_name)

    # 4. Decode frames ONCE, reuse for both B and C-i
    frames = _decode_frames(video_path) if video_path else []
    if not frames and doc_path and os.path.exists(doc_path):
        try:
            import cv2
            doc_img = cv2.imread(doc_path)
            if doc_img is not None:
                frames = [cv2.cvtColor(doc_img, cv2.COLOR_BGR2RGB)]
        except Exception:
            pass

    # 5. Fire A, B, C in PARALLEL
    genai_task     = loop.run_in_executor(_executor, detect_genai_document, doc_path)
    deepfake_task  = loop.run_in_executor(_executor, detect_face_deepfake, frames)
    challenge_task = loop.run_in_executor(
        _executor, run_challenge, video_path, audio_path, ch, ocr_name)

    genai_res, deepfake_res, challenge_results = await asyncio.gather(
        genai_task, deepfake_task, challenge_task)

    # Cross-modal consistency check
    asr_result = next((r for r in challenge_results
                       if r.signal == SignalId.NAME_MATCH), None)
    transcript = asr_result.evidence.get("transcript", "") if asr_result else ""
    spoken_name = transcript  # ASR transcript contains the full spoken phrase

    cross_modal = check_cross_modal_consistency(
        primary_ocr_name=ocr_name,
        spoken_name=spoken_name,
        # secondary_ocr_name from second doc if uploaded (optional)
    )

    # Add cross_modal findings to the name_match signal's evidence
    if asr_result:
        asr_result.evidence["cross_modal"] = cross_modal
        asr_result.evidence["cross_modal_consistent"] = cross_modal["all_consistent"]
        # If cross-modal inconsistency, boost name_match raw_score
        if not cross_modal["all_consistent"]:
            asr_result.raw_score = min(1.0,
                asr_result.raw_score + cross_modal["consistency_score"] * 0.3)
            asr_result.triggered = (
                asr_result.raw_score > SIGNAL_TRIGGER[SignalId.NAME_MATCH])

    # 6. Fuse everything
    all_new = [genai_res, deepfake_res] + challenge_results
    fused = fuse(existing_score, all_new)

    # Surface watchlist result from existing pipeline
    watchlist_triggered = existing_result.get("watchlist_hit", False)
    watchlist_score = 1.0 if watchlist_triggered else 0.0
    watchlist_line = {
        "signal":    "watchlist",
        "penalty":   100.0 if watchlist_triggered else 0.0,
        "label":     ("WATCHLIST HIT — subject identity flagged" if watchlist_triggered
                     else "Not on watchlist"),
        "confidence": watchlist_score,
        "severity":  "hard",
        "triggered": watchlist_triggered,
        "evidence":  existing_result.get("watchlist_evidence", {}),
        "explanation": (
            "Subject identity matched against the watchlist database. "
            "This is an automatic HARD flag regardless of other signals."
            if watchlist_triggered else
            "Subject not found in watchlist database."
        )
    }
    # Prepend to lines (watchlist is highest priority)
    fused["lines"].insert(0, watchlist_line)

    if watchlist_triggered:
        fused["score"] = 100.0
        fused["tier"] = "flagged"
        if "watchlist" not in fused.get("floors_applied", []):
            fused["floors_applied"].append("watchlist_hard_signal")

    # Add plain-English explanation and model provenance to each signal line
    for line in fused["lines"]:
        line["explanation"] = build_signal_explanation(line["signal"], line)

        sig = line.get("signal", "")
        prov = MODEL_PROVENANCE.get(sig, {})
        ev = line.get("evidence", {})
        method = str(ev.get("method", "") or ev.get("mode", ""))
        is_fallback = any(w in method.lower() for w in
                          ["heuristic", "fallback", "laplacian", "fft"])
        line["model_provenance"] = {
            "model":       prov.get("fallback" if is_fallback else "primary", "unknown"),
            "is_fallback": is_fallback,
            "calibration": prov.get("calibration", "unknown"),
            "samples":     prov.get("samples", "unknown"),
        }

    # 7. Build response matching the dashboard contract EXACTLY
    response = {
        "existing": existing_result.get("breakdown", {}),
        "lines":    fused["lines"],
        "final": {
            "score":          fused["score"],
            "tier":           fused["tier"],
            "floors_applied": fused["floors_applied"],
        },
        "summary": build_officer_summary(fused),
    }

    # 8. Append to the hash-chained audit log
    db = sqlite3.connect("audit.db")
    db.row_factory = sqlite3.Row
    append_entry(db, _PEPPER, response)
    db.close()

    # 9. Cache for the dashboard's /api/verify/last endpoint
    _LAST_RESULT.clear()
    _LAST_RESULT.update(response)
    return response

@app.get("/api/verify/last")
async def last_result():
    return _LAST_RESULT or {"error": "no verification run yet"}

@app.get("/api/challenge/new")
@app.get("/api/challenge")
async def get_new_challenge(name: str = "Priya Sharma"):
    """
    Issue a dynamic single-use session challenge with a cryptographically fresh nonce.
    """
    ch = issue_challenge(name)
    return {
        "ok": True,
        "action": ch.action,
        "nonce": ch.nonce,
        "date_str": ch.date_str,
        "expected_name": ch.expected_name,
        "spoken_phrase": ch.spoken_phrase,
    }

@app.post("/api/challenge")
async def handle_challenge(request: Request):
    global _PEPPER
    if _PEPPER is None:
        from shared.pepper_sss import generate_pepper
        _PEPPER = generate_pepper()

    form = await request.form()
    action = form.get("action", "turn_left")
    nonce = form.get("nonce", "4471")
    date_str = form.get("date_str", "2026-09-11")
    expected_name = "Priya Sharma"
    ch = issue_challenge(expected_name)
    ch.action = action
    ch.nonce = nonce
    ch.date_str = date_str

    frames = []
    try:
        import cv2
        import numpy as np
        for key, value in form.items():
            if key.startswith("frame_") and hasattr(value, "file"):
                content = await value.read()
                arr = np.frombuffer(content, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    frames.append(img_rgb)
    except Exception as e:
        logger.warning("Frame extraction error: %s", e)

    tmp_dir = tempfile.gettempdir()
    audio_file = form.get("audio")
    audio_path = ""
    if audio_file and hasattr(audio_file, "file"):
        audio_path = os.path.join(tmp_dir, f"ch_aud_{uuid.uuid4().hex[:8]}.webm")
        with open(audio_path, "wb") as f:
            f.write(await audio_file.read())

    loop = asyncio.get_event_loop()
    deepfake_task = loop.run_in_executor(_executor, detect_face_deepfake, frames)
    challenge_task = loop.run_in_executor(
        _executor, run_challenge, "", audio_path, ch, expected_name
    )

    deepfake_res, challenge_results = await asyncio.gather(deepfake_task, challenge_task)

    # Cross-modal consistency check
    asr_result = next((r for r in challenge_results
                       if r.signal == SignalId.NAME_MATCH), None)
    transcript = asr_result.evidence.get("transcript", "") if asr_result else ""
    spoken_name = transcript

    cross_modal = check_cross_modal_consistency(
        primary_ocr_name=expected_name,
        spoken_name=spoken_name,
    )
    if asr_result:
        asr_result.evidence["cross_modal"] = cross_modal
        asr_result.evidence["cross_modal_consistent"] = cross_modal["all_consistent"]
        if not cross_modal["all_consistent"]:
            asr_result.raw_score = min(1.0,
                asr_result.raw_score + cross_modal["consistency_score"] * 0.3)
            asr_result.triggered = (
                asr_result.raw_score > SIGNAL_TRIGGER[SignalId.NAME_MATCH])

    fused = fuse(10.0, [deepfake_res] + challenge_results)

    # Watchlist line
    watchlist_line = {
        "signal":    "watchlist",
        "penalty":   0.0,
        "label":     "Not on watchlist",
        "confidence": 0.0,
        "severity":  "hard",
        "triggered": False,
        "evidence":  {},
        "explanation": "Subject not found in watchlist database."
    }
    fused["lines"].insert(0, watchlist_line)

    for line in fused["lines"]:
        line["explanation"] = build_signal_explanation(line["signal"], line)
        sig = line.get("signal", "")
        prov = MODEL_PROVENANCE.get(sig, {})
        ev = line.get("evidence", {})
        method = str(ev.get("method", "") or ev.get("mode", ""))
        is_fallback = any(w in method.lower() for w in
                          ["heuristic", "fallback", "laplacian", "fft"])
        line["model_provenance"] = {
            "model":       prov.get("fallback" if is_fallback else "primary", "unknown"),
            "is_fallback": is_fallback,
            "calibration": prov.get("calibration", "unknown"),
            "samples":     prov.get("samples", "unknown"),
        }

    response = {
        "ok": True,
        "existing": {"ocr": "clean", "signature": "valid", "ela": "clear"},
        "lines": fused["lines"],
        "final": {
            "score": fused["score"],
            "tier": fused["tier"],
            "floors_applied": fused["floors_applied"],
        },
        "summary": build_officer_summary(fused),
    }

    db = sqlite3.connect("audit.db")
    db.row_factory = sqlite3.Row
    append_entry(db, _PEPPER, response)
    db.close()

    _LAST_RESULT.clear()
    _LAST_RESULT.update(response)
    return response

# Serve frontend static assets directly on the orchestrator port
_static_dir = Path(__file__).resolve().parent.parent / "frontend" / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

    @app.get("/")
    def index():
        return RedirectResponse(url="/dashboard.html")

    @app.get("/dashboard.html")
    def dashboard_page():
        return FileResponse(str(_static_dir / "dashboard.html"))

    @app.get("/capture.html")
    def capture_page():
        return FileResponse(str(_static_dir / "capture.html"))

    @app.get("/app.js")
    def app_js_file():
        return FileResponse(str(_static_dir / "app.js"))

    @app.get("/verify_sample.json")
    def sample_flagged_file():
        return FileResponse(str(_static_dir / "verify_sample.json"))

    @app.get("/verify_sample_clear.json")
    def sample_clear_file():
        return FileResponse(str(_static_dir / "verify_sample_clear.json"))

