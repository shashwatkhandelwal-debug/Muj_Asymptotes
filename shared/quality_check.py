"""
Document Quality Pre-Check.

Validates that an image meets minimum quality requirements before
running OCR or forensics. Prevents silent errors from blurry
or low-resolution inputs.
"""

import cv2
import numpy as np

# Minimum resolution boundaries (orientation independent)
MIN_LONG_EDGE = 300
MIN_SHORT_EDGE = 200

# Laplacian variance threshold
BLUR_THRESHOLD = 15.0


def check_quality(image: np.ndarray) -> dict:
    """
    Check document image quality.

    Args:
        image: BGR numpy array.

    Returns:
        dict with keys:
          acceptable (bool)
          issues     (list[str])
          blur_score (float)
          resolution (tuple)
    """
    # Standardize image array to uint8 [0, 255] range
    if image.dtype.kind == "f":
        if np.max(image) <= 1.01:
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    issues = []

    # Handle single channel (grayscale) images by replicating channels
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    h, w = image.shape[:2]

    # Orientation-independent resolution check
    long_edge = max(w, h)
    short_edge = min(w, h)

    if long_edge < MIN_LONG_EDGE or short_edge < MIN_SHORT_EDGE:
        issues.append(
            f"Resolution too low: {w}x{h}px. "
            f"Requires at least {MIN_LONG_EDGE}px on the long edge and {MIN_SHORT_EDGE}px on the short edge."
        )

    # Blur check (Laplacian variance)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_score < BLUR_THRESHOLD:
        issues.append(
            f"Image too blurry (sharpness score: {blur_score:.1f}, minimum required: {BLUR_THRESHOLD:.1f})"
        )

    return {
        "acceptable": len(issues) == 0,
        "issues": issues,
        "blur_score": round(blur_score, 2),
        "resolution": (w, h),
    }
