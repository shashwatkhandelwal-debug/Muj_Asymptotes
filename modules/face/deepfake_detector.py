import os
import sys
import time
import logging
from pathlib import Path
import numpy as np

# Ensure repo root is on sys.path
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from shared.contracts import SignalResult, SignalId, Severity, SIGNAL_TRIGGER
from modules.decision.calibration import apply_calibration

logger = logging.getLogger(__name__)

_HF_PIPELINE = None
_HF_TRIED = False
_ONNX_SESSION = None
_ONNX_TRIED = False

def _get_onnx_session():
    global _ONNX_SESSION, _ONNX_TRIED
    if _ONNX_TRIED:
        return _ONNX_SESSION
    _ONNX_TRIED = True
    onnx_path = Path("models/deepfake.onnx")
    if onnx_path.exists():
        try:
            import onnxruntime as ort
            _ONNX_SESSION = ort.InferenceSession(str(onnx_path))
            logger.info("Loaded ONNX deepfake model from %s", onnx_path)
        except Exception as e:
            logger.warning("Failed to load ONNX deepfake model: %s", e)
            _ONNX_SESSION = None
    return _ONNX_SESSION

def _get_hf_pipeline():
    global _HF_PIPELINE, _HF_TRIED
    if _HF_TRIED:
        return _HF_PIPELINE
    _HF_TRIED = True
    try:
        from transformers import pipeline
        allow_download = os.environ.get("DOWNLOAD_PRETRAINED", "0") == "1"
        try:
            _HF_PIPELINE = pipeline(
                "image-classification",
                model="dima806/deepfake_vs_real_image_detection",
                local_files_only=True
            )
        except Exception:
            if allow_download:
                _HF_PIPELINE = pipeline(
                    "image-classification",
                    model="dima806/deepfake_vs_real_image_detection"
                )
            else:
                _HF_PIPELINE = None
    except Exception as e:
        logger.info("HF deepfake model not found locally; using frequency fallback: %s", e)
        _HF_PIPELINE = None
    return _HF_PIPELINE

def _laplacian_var(gray: np.ndarray) -> float:
    try:
        import cv2
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        # Fallback 2D Laplacian using numpy diffs
        dy = np.diff(gray, 2, axis=0)
        dx = np.diff(gray, 2, axis=1)
        return float(np.var(dy) + np.var(dx))

def _score_frame(frame: np.ndarray) -> float:
    # 1. Primary: ONNX deepfake model if present
    session = _get_onnx_session()
    if session is not None:
        try:
            from PIL import Image
            img = Image.fromarray(frame).resize((224, 224))
            arr = np.array(img, dtype=np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            arr = (arr - mean) / std
            inp = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: inp})
            logits = outputs[0][0]
            if len(logits) == 2:
                prob_fake = float(np.exp(logits[1]) / np.sum(np.exp(logits)))
            else:
                prob_fake = float(1.0 / (1.0 + np.exp(-logits[0])))
            return float(np.clip(prob_fake, 0.0, 1.0))
        except Exception as e:
            logger.debug("ONNX inference failed on frame, falling back: %s", e)

    # 2. Secondary: HF pipeline if cached
    pipe = _get_hf_pipeline()
    if pipe is not None:
        try:
            from PIL import Image
            img = Image.fromarray(frame)
            preds = pipe(img)
            # Find label corresponding to fake/deepfake
            for p in preds:
                label_str = p.get("label", "").lower()
                if "fake" in label_str or "deepfake" in label_str:
                    return float(p.get("score", 0.0))
            return 0.0
        except Exception as e:
            logger.debug("HF inference failed on frame, falling back to Laplacian: %s", e)

    # 3. Fallback: Frequency-domain texture classifier
    if frame.ndim == 3:
        # Convert RGB to grayscale (standard luminance weights)
        gray = 0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]
    else:
        gray = frame.astype(float)

    lap_var = _laplacian_var(gray.astype(np.float64))
    if lap_var > 1000.0:
        # Extreme high-frequency noise / artifact frames
        score = min(1.0, lap_var / 2500.0)
    elif lap_var < 40.0:
        # Uniform smooth synthetic canvas
        score = max(0.05, lap_var / 200.0)
    else:
        # Normal texture range: low variance = blurry
        z = -(lap_var - 80.0) / 40.0
        if z >= 0:
            score = 1.0 / (1.0 + np.exp(-z))
        else:
            ez = np.exp(z)
            score = ez / (1.0 + ez)
    return float(np.clip(score, 0.0, 1.0))

def _temporal_consistency_score(frames: list) -> float:
    """
    Compute inter-frame consistency score.
    Low consistency (high variance) = suspicious = higher score.

    Method: for each consecutive pair of frames, compute mean absolute
    difference in pixel values after grayscale conversion. Real video
    has smooth, bounded frame-to-frame differences. Deepfakes often
    show abrupt texture shifts.

    Returns: float 0..1, higher = more temporally inconsistent.
    """
    if len(frames) < 2:
        return 0.0
    try:
        diffs = []
        for i in range(len(frames) - 1):
            f1 = frames[i].mean(axis=2).astype(float)   # grayscale
            f2 = frames[i+1].mean(axis=2).astype(float)
            # Resize to 64x64 for speed
            import cv2
            f1_s = cv2.resize(f1.astype(np.uint8), (64, 64)).astype(float)
            f2_s = cv2.resize(f2.astype(np.uint8), (64, 64)).astype(float)
            diff = np.abs(f1_s - f2_s).mean()
            diffs.append(diff)

        mean_diff = np.mean(diffs)
        std_diff  = np.std(diffs)

        # Normalize: high std relative to mean = inconsistent = suspicious
        # Real video: std/mean (CoV) is low and stable
        # Deepfakes: CoV spikes at artifact frames
        cov = std_diff / (mean_diff + 1e-9)
        jitter = min(1.0, mean_diff / 25.0)
        combined_cov = max(cov, jitter)
        score = 1.0 / (1.0 + np.exp(-(combined_cov - 0.4) * 5))
        return float(np.clip(score, 0.0, 1.0))
    except Exception:
        return 0.0

def detect_face_deepfake(frames: list, fps: float = 0.0) -> SignalResult:
    """
    IN:  frames = list of numpy RGB arrays (H,W,3) uint8.
         fps = capture frame rate (unused in per-frame mode, kept for future temporal upgrade).
    OUT: SignalResult(
           signal=SignalId.FACE_DEEPFAKE,
           severity=Severity.HARD,
           raw_score=mean of top-3 per-frame scores,
           confidence=apply_calibration(raw_score, "face_deepfake"),
           triggered=(raw_score > SIGNAL_TRIGGER[SignalId.FACE_DEEPFAKE]),
           label="Deepfake artifacts detected" if triggered else "Face appears genuine",
           evidence={"mode":"per_frame_aggregate+temporal","per_frame_scores":[...]}
         )
    If frames is empty return SignalResult with ok=False, raw_score=0.0, confidence=0.0.
    Measure wall time and set result.ms.
    """
    t0 = time.time()
    if not frames:
        elapsed_ms = (time.time() - t0) * 1000.0
        return SignalResult(
            signal=SignalId.FACE_DEEPFAKE,
            raw_score=0.0,
            confidence=0.0,
            severity=Severity.HARD,
            triggered=False,
            label="No frames provided",
            evidence={"error": "Empty frames list", "mode": "per_frame_aggregate+temporal", "per_frame_scores": [], "temporal_consistency_score": 0.0},
            ok=False,
            ms=elapsed_ms,
        )

    try:
        # Sample up to 15 frames evenly spaced
        n = len(frames)
        if n <= 15:
            sampled_frames = frames
        else:
            indices = np.linspace(0, n - 1, 15, dtype=int)
            sampled_frames = [frames[i] for i in indices]

        per_frame_scores = [_score_frame(f) for f in sampled_frames]

        # raw_score = mean of top-3 frame scores
        if len(per_frame_scores) >= 3:
            sorted_scores = sorted(per_frame_scores, reverse=True)
            raw_score = float(np.mean(sorted_scores[:3]))
        else:
            raw_score = float(np.mean(per_frame_scores))

        # Temporal consistency (runs on same frames, free)
        temporal_score = _temporal_consistency_score(sampled_frames)

        # Weighted blend: 70% per-frame, 30% temporal
        raw_score = 0.70 * raw_score + 0.30 * temporal_score

        raw_score = float(np.clip(raw_score, 0.0, 1.0))
        confidence = apply_calibration(raw_score, "face_deepfake")
        triggered = bool(raw_score > SIGNAL_TRIGGER[SignalId.FACE_DEEPFAKE])
        label = (
            "Deepfake artifacts detected"
            if triggered
            else "Face appears genuine"
        )
        elapsed_ms = (time.time() - t0) * 1000.0

        return SignalResult(
            signal=SignalId.FACE_DEEPFAKE,
            raw_score=raw_score,
            confidence=confidence,
            severity=Severity.HARD,
            triggered=triggered,
            label=label,
            evidence={
                "mode": "per_frame_aggregate+temporal",
                "per_frame_scores": per_frame_scores,
                "temporal_consistency_score": round(temporal_score, 4),
            },
            ok=True,
            ms=elapsed_ms,
        )
    except Exception as e:
        logger.exception("Error in detect_face_deepfake: %s", e)
        elapsed_ms = (time.time() - t0) * 1000.0
        return SignalResult(
            signal=SignalId.FACE_DEEPFAKE,
            raw_score=0.0,
            confidence=0.0,
            severity=Severity.HARD,
            triggered=False,
            label="Error detecting deepfake",
            evidence={"error": str(e), "mode": "per_frame_aggregate", "per_frame_scores": []},
            ok=False,
            ms=elapsed_ms,
        )

if __name__ == "__main__":
    frames = [np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(20)]
    result = detect_face_deepfake(frames)
    assert result.signal == SignalId.FACE_DEEPFAKE
    assert result.severity == Severity.HARD
    assert 0.0 <= result.raw_score <= 1.0
    assert len(result.evidence["per_frame_scores"]) <= 15
    print("DEEPFAKE DETECTOR OK")
