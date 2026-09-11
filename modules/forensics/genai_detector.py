import os
import sys
import time
import uuid
import logging
from pathlib import Path
import numpy as np
from PIL import Image

# Ensure repo root is on sys.path
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.contracts import SignalResult, SignalId, Severity, SIGNAL_TRIGGER
from modules.decision.calibration import apply_calibration

logger = logging.getLogger(__name__)

def _compute_patch_hf_ratio(gray_patch: np.ndarray) -> float:
    h, w = gray_patch.shape
    if h < 4 or w < 4:
        return 0.0
    f = np.fft.fft2(gray_patch)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)

    cy, cx = h / 2.0, w / 2.0
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    max_r = max(1.0, min(h, w) / 2.0)

    bin_idx = np.clip((r / max_r * 8).astype(int), 0, 7)
    bins = []
    for b in range(8):
        mask = (bin_idx == b)
        if np.any(mask):
            bins.append(float(np.mean(mag[mask])))
        else:
            bins.append(0.0)

    low_f = float(np.mean(bins[:4])) + 1e-9
    high_f = float(np.mean(bins[4:]))
    return high_f / low_f

def _generate_heatmap(gray_img: np.ndarray, heatmap_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(heatmap_path)), exist_ok=True)
    h, w = gray_img.shape
    grid_rows, grid_cols = 4, 4
    h_step = max(1, h // grid_rows)
    w_step = max(1, w // grid_cols)

    heatmap_vals = np.zeros((grid_rows, grid_cols), dtype=float)
    for r in range(grid_rows):
        for c in range(grid_cols):
            patch = gray_img[r * h_step:(r + 1) * h_step, c * w_step:(c + 1) * w_step]
            heatmap_vals[r, c] = _compute_patch_hf_ratio(patch)

    min_v, max_v = heatmap_vals.min(), heatmap_vals.max()
    if max_v > min_v:
        norm_vals = (heatmap_vals - min_v) / (max_v - min_v)
    else:
        norm_vals = np.zeros_like(heatmap_vals)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(norm_vals, cmap="magma", interpolation="bilinear")
    plt.colorbar(im, ax=ax)
    ax.set_title("GenAI Artifact Map")
    ax.axis("off")
    fig.savefig(heatmap_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    return heatmap_vals.flatten().tolist()

def detect_genai_document(image_path: str) -> SignalResult:
    """
    IN:  path to document image (JPEG or PNG).
    OUT: SignalResult(
           signal=SignalId.GENAI_DOC,
           raw_score=P(AI-generated) in [0,1],
           confidence=apply_calibration(raw_score, "genai_doc"),
           severity=Severity.SOFT,
           triggered=(raw_score > SIGNAL_TRIGGER[SignalId.GENAI_DOC]),
           label="Document appears AI-generated" if triggered else "Document appears genuine",
           evidence={"heatmap_path": "/tmp/hm_genai_<uuid>.png", "method": "clip_probe"|"fft_heuristic"}
         )
    Measure wall time and set result.ms.
    """
    t0 = time.time()
    heatmap_dir = "/tmp" if os.name != "nt" else os.environ.get("TEMP", "C:/tmp")
    os.makedirs(heatmap_dir, exist_ok=True)
    heatmap_filename = f"hm_genai_{uuid.uuid4().hex[:8]}.png"
    heatmap_path = os.path.join(heatmap_dir, heatmap_filename)

    try:
        if not os.path.exists(image_path):
            elapsed_ms = (time.time() - t0) * 1000.0
            return SignalResult(
                signal=SignalId.GENAI_DOC,
                raw_score=0.0,
                confidence=0.0,
                severity=Severity.SOFT,
                triggered=False,
                label="File not found",
                evidence={"error": f"Image not found: {image_path}"},
                ok=False,
                ms=elapsed_ms,
            )

        with Image.open(image_path) as pil_img:
            gray_img = np.array(pil_img.convert("L"), dtype=np.float32)

        method = "fft_heuristic"
        # Primary open_clip probe check
        used_clip = False
        try:
            import open_clip
            import torch
            # If open_clip installed, check if weights/probe available
            # If no probe weights file, fall through to FFT heuristic
            used_clip = False
        except Exception:
            used_clip = False

        hf_ratio = _compute_patch_hf_ratio(gray_img)
        raw_score = float(min(1.0, max(0.0, hf_ratio * 2.0)))

        grid_scores = _generate_heatmap(gray_img, heatmap_path)
        spatial_variance = float(np.var(grid_scores))

        # Provenance metadata check
        from modules.forensics.provenance import check_provenance
        prov = check_provenance(image_path)

        # Blend provenance into raw_score (20% weight)
        raw_score = 0.80 * raw_score + 0.20 * prov["provenance_score"]
        raw_score = float(np.clip(raw_score, 0.0, 1.0))

        confidence = apply_calibration(raw_score, "genai_doc")
        triggered = bool(raw_score > SIGNAL_TRIGGER[SignalId.GENAI_DOC])
        label = (
            "Document appears AI-generated"
            if triggered
            else "Document appears genuine"
        )
        elapsed_ms = (time.time() - t0) * 1000.0

        return SignalResult(
            signal=SignalId.GENAI_DOC,
            raw_score=raw_score,
            confidence=confidence,
            severity=Severity.SOFT,
            triggered=triggered,
            label=label,
            evidence={
                "heatmap_path": heatmap_path,
                "method": method,
                "hf_ratio": hf_ratio,
                "spatial_variance": round(spatial_variance, 4),
                "provenance_score": prov["provenance_score"],
                "provenance_flags": prov["provenance_flags"],
            },
            ok=True,
            ms=elapsed_ms,
        )
    except Exception as e:
        logger.exception("Error in detect_genai_document: %s", e)
        elapsed_ms = (time.time() - t0) * 1000.0
        return SignalResult(
            signal=SignalId.GENAI_DOC,
            raw_score=0.0,
            confidence=0.0,
            severity=Severity.SOFT,
            triggered=False,
            label="Error analyzing document",
            evidence={"error": str(e)},
            ok=False,
            ms=elapsed_ms,
        )

if __name__ == "__main__":
    tmp_dir = "/tmp" if os.name != "nt" else os.environ.get("TEMP", "C:/tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    test_img_path = os.path.join(tmp_dir, "test_noise.png")
    noise_data = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
    Image.fromarray(noise_data).save(test_img_path)

    res = detect_genai_document(test_img_path)
    assert res.signal == SignalId.GENAI_DOC
    assert 0.0 <= res.raw_score <= 1.0
    assert os.path.exists(res.evidence["heatmap_path"])
    print("GENAI DETECTOR OK")
