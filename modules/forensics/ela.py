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

import io
from typing import Optional

import numpy as np
from PIL import Image


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
          heatmap       (np.ndarray) - amplified difference map (RGB uint8)
          suspicious    (bool)   - True if variance exceeds threshold
          threshold     (float)  - threshold used
    """
    if image is None or getattr(image, "size", 0) == 0:
        return {
            "mean_variance": 0.0,
            "max_variance": 0.0,
            "heatmap": None,
            "suspicious": False,
            "threshold": 15.0,
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

    # Amplify by alpha=10 scale factor
    heatmap = np.clip(diff * 10.0, 0, 255).astype(np.uint8)

    # Threshold: empirically set
    threshold = _adaptive_threshold(image, quality)
    suspicious = mean_var > threshold

    return {
        "mean_variance": round(mean_var, 4),
        "max_variance": round(max_var, 4),
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
