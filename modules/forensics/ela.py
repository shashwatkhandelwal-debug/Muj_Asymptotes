"""
Error Level Analysis (ELA) - JPEG Compression Forensics.

Detects regions of a document image that have a different JPEG compression
history than the surrounding image. This acts as evidence of digital splicing or editing.

Theory:
  JPEG compression is lossy via DCT quantization. Recompressing an image at
  a fixed quality Q produces near-zero pixel differences in regions already
  at quality Q. It produces elevated differences in regions edited or saved at a
  different quality. Examples include a spliced photo or modified text block.

Three modes:
  1. full_document  - whole image ELA (detects large-scale edits)
  2. region         - ELA restricted to a supplied bounding box (photo/QR/stamp)
  3. heatmap        - returns amplified RGB difference array for visualization
"""

import os
import sys
import io
import time
import uuid
import tempfile
import logging
from typing import Optional, Union
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Ensure repo root is on sys.path
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from shared.contracts import SignalResult, SignalId, Severity, SIGNAL_TRIGGER
from modules.decision.calibration import apply_calibration

logger = logging.getLogger(__name__)


def run_ela(
    image: np.ndarray,
    quality: int = 95,
    region: Optional[tuple] = None,
) -> dict:
    """
    Run ELA on an image or a sub-region.

    Args:
        image:   BGR numpy array of the document.
        quality: JPEG recompression quality (0-95). Default 95.
        region:  Optional (x1, y1, x2, y2) tuple to restrict analysis.
                 If None, analyzes the full image.

    Returns:
        dict with keys:
          mean_variance (float)  - statistical variance of pixel-level difference
          max_variance  (float)  - maximum variance across channels
          max_patch_variance (float) - highest variance across localized patches
          heatmap       (np.ndarray) - amplified difference map (RGB uint8)
          suspicious    (bool)   - True if variance exceeds threshold
          threshold     (float)  - threshold used
    """
    if image is None or getattr(image, "size", 0) == 0:
        return {
            "mean_variance": 0.0,
            "max_variance": 0.0,
            "max_patch_variance": 0.0,
            "heatmap": None,
            "suspicious": False,
            "threshold": 15.0,
            "region": region,
        }

    # Ensure 3-channel BGR
    img = image.copy()
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    pil_img = Image.fromarray(img[..., ::-1]).convert("RGB")  # BGR to RGB

    if region:
        x1, y1, x2, y2 = region
        pil_img = pil_img.crop((x1, y1, x2, y2))

    # Recompress at fixed quality
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")

    orig_arr = np.array(pil_img, dtype=np.float32)
    recomp_arr = np.array(recompressed, dtype=np.float32)

    diff = np.abs(orig_arr - recomp_arr)

    # Statistical variance computation
    mean_var = float(np.var(diff))
    max_var = float(diff.max())

    # Local patch analysis (8x8 grid) to detect localized splicing / text tampering
    h, w, _ = diff.shape
    gh, gw = 8, 8
    sh, sw = max(1, h // gh), max(1, w // gw)
    patch_vars = [
        float(np.var(diff[r * sh : (r + 1) * sh, c * sw : (c + 1) * sw]))
        for r in range(gh)
        for c in range(gw)
    ]
    max_patch_var = float(np.max(patch_vars)) if patch_vars else mean_var

    # Amplify by alpha=10 scale factor
    heatmap = np.clip(diff * 10.0, 0, 255).astype(np.uint8)

    # Threshold: empirically set
    threshold = _adaptive_threshold(image, quality)
    suspicious = bool(mean_var > threshold or max_patch_var > threshold * 2.2)

    return {
        "mean_variance": round(mean_var, 4),
        "max_variance": round(max_var, 4),
        "max_patch_variance": round(max_patch_var, 4),
        "heatmap": heatmap,
        "suspicious": suspicious,
        "threshold": threshold,
        "region": region,
    }


def _adaptive_threshold(image: np.ndarray, recomp_quality: int) -> float:
    """
    Estimate adaptive ELA threshold from the image's estimated JPEG quality.
    Higher original quality leads to lower expected ELA variance and tighter threshold.
    Lower original quality leads to higher baseline variance and looser threshold.

    Returns a variance threshold float.
    """
    try:
        # Encode to JPEG, read back quality from quantization tables
        pil = Image.fromarray(image[..., ::-1])
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=95)  # Save at near-lossless
        buf.seek(0)
        img_back = Image.open(buf)

        # Luma channel variance
        luma = np.array(img_back.convert("L"), dtype=np.float32)
        luma_var = float(np.var(luma))

        # Scale threshold inversely with estimated quality
        if luma_var > 3000:
            return 8.0  # High-quality scan
        elif luma_var > 1000:
            return 12.0  # Medium quality
        else:
            return 18.0  # Low quality
    except Exception:
        return 12.0


def score_document_ela(image_or_path: Union[str, np.ndarray]) -> SignalResult:
    """
    Extracts ELA forensics and returns a standard SignalResult.
    IN:  image_or_path = file path (str) or BGR numpy array.
    OUT: SignalResult(
           signal=SignalId.ELA,
           raw_score=P(tampered) in [0,1],
           confidence=calibrated score,
           severity=Severity.SOFT,
           triggered=bool,
           label=str,
           evidence={...}
         )
    """
    t0 = time.perf_counter()

    if isinstance(image_or_path, str):
        if not os.path.exists(image_or_path):
            elapsed_ms = max(0.1, (time.perf_counter() - t0) * 1000.0)
            return SignalResult(
                signal=SignalId.ELA,
                raw_score=0.0,
                confidence=0.0,
                severity=Severity.SOFT,
                triggered=False,
                label="Document image not found",
                evidence={"error": f"File not found: {image_or_path}"},
                ok=False,
                ms=elapsed_ms,
            )
        img = cv2.imread(image_or_path)
    else:
        img = image_or_path

    if img is None or getattr(img, "size", 0) == 0:
        elapsed_ms = max(0.1, (time.perf_counter() - t0) * 1000.0)
        return SignalResult(
            signal=SignalId.ELA,
            raw_score=0.0,
            confidence=0.0,
            severity=Severity.SOFT,
            triggered=False,
            label="Invalid document image",
            evidence={"error": "Empty or corrupted image"},
            ok=False,
            ms=elapsed_ms,
        )

    try:
        ela_res = run_ela(img)
        mean_var = ela_res["mean_variance"]
        max_pvar = ela_res.get("max_patch_variance", mean_var)
        threshold = ela_res["threshold"]

        score_global = mean_var / (threshold * 1.5)
        score_local = max_pvar / (threshold * 2.5)
        raw_score = float(np.clip(max(score_global, score_local), 0.0, 1.0))

        confidence = apply_calibration(raw_score, "ela")
        trigger_val = SIGNAL_TRIGGER.get(SignalId.ELA, 0.50)
        triggered = bool(raw_score > trigger_val or ela_res["suspicious"])

        label = (
            "Recompression anomalies detected — possible document tampering (ELA)"
            if triggered
            else "No ELA compression anomalies detected"
        )

        heatmap_path = ""
        if ela_res.get("heatmap") is not None:
            try:
                tmp_dir = tempfile.gettempdir()
                heatmap_path = os.path.join(
                    tmp_dir, f"ela_heatmap_{uuid.uuid4().hex[:8]}.png"
                )
                Image.fromarray(ela_res["heatmap"]).save(heatmap_path)
            except Exception as e:
                logger.debug("Failed to save ELA heatmap: %s", e)

        elapsed_ms = max(0.1, (time.perf_counter() - t0) * 1000.0)
        return SignalResult(
            signal=SignalId.ELA,
            raw_score=raw_score,
            confidence=confidence,
            severity=Severity.SOFT,
            triggered=triggered,
            label=label,
            evidence={
                "mean_variance": ela_res["mean_variance"],
                "max_variance": ela_res["max_variance"],
                "max_patch_variance": max_pvar,
                "threshold": threshold,
                "heatmap_path": heatmap_path,
                "method": "jpeg_error_level_analysis",
            },
            ok=True,
            ms=elapsed_ms,
        )
    except Exception as e:
        logger.exception("Error in score_document_ela: %s", e)
        elapsed_ms = max(0.1, (time.perf_counter() - t0) * 1000.0)
        return SignalResult(
            signal=SignalId.ELA,
            raw_score=0.0,
            confidence=0.0,
            severity=Severity.SOFT,
            triggered=False,
            label="Error running ELA analysis",
            evidence={"error": str(e)},
            ok=False,
            ms=elapsed_ms,
        )


if __name__ == "__main__":
    tmp_dir = tempfile.gettempdir()
    test_img_path = os.path.join(tmp_dir, f"test_ela_{os.getpid()}.jpg")
    blank = np.ones((200, 300, 3), dtype=np.uint8) * 200
    cv2.imwrite(test_img_path, blank)

    res = score_document_ela(test_img_path)
    assert res.signal == SignalId.ELA
    assert res.severity == Severity.SOFT
    assert 0.0 <= res.raw_score <= 1.0
    print("ELA FORENSICS OK")
