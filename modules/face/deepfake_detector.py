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

def _get_hf_pipeline():
    global _HF_PIPELINE, _HF_TRIED
    if _HF_TRIED:
        return _HF_PIPELINE
    _HF_TRIED = True
    try:
        from transformers import pipeline
        # Attempt local cache load first to avoid network requests
        _HF_PIPELINE = pipeline(
            "image-classification",
            model="dima806/deepfake_vs_real_image_detection",
            local_files_only=True
        )
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

    # Fallback: Frequency-domain texture classifier
    if frame.ndim == 3:
        # Convert RGB to grayscale (standard luminance weights)
        gray = 0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]
    else:
        gray = frame.astype(float)

    lap_var = _laplacian_var(gray.astype(np.float64))
    # Score = sigmoid(-(lap_var - 80) / 40)
    z = -(lap_var - 80.0) / 40.0
    if z >= 0:
        score = 1.0 / (1.0 + np.exp(-z))
    else:
        ez = np.exp(z)
        score = ez / (1.0 + ez)
    return float(np.clip(score, 0.0, 1.0))

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
           evidence={"mode":"per_frame_aggregate","per_frame_scores":[...]}
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
            evidence={"error": "Empty frames list", "mode": "per_frame_aggregate", "per_frame_scores": []},
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
                "mode": "per_frame_aggregate",
                "per_frame_scores": per_frame_scores,
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
