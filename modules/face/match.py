"""
Face Match  -  Document Photo vs Live Capture.

Compares the face on the document (Aadhaar card photo or passport bio-page photo)
against a live capture of the traveler standing at the checkpoint.

Two inputs:
  1. Document image (full card/page)  -  face extracted by cropping known region
     or from chip DG2 (if NFC was successful, chip photo is used  -  authenticated)
  2. Live capture frame from Streamlit camera widget

The same ArcFace embedder is used for both to ensure the embedding spaces
are compatible (comparing apples to apples).
"""

from typing import Optional

import cv2
import numpy as np

from .embedder import cosine_similarity, get_embedding, similarity_to_verdict

# Approximate face regions for cropping (relative to card/page dimensions)
# These are fallback regions if face detection fails on the full image
_AADHAAR_FACE_REGION = (0.03, 0.15, 0.30, 0.60)  # (x1%, y1%, x2%, y2%)
_PASSPORT_FACE_REGION = (0.05, 0.10, 0.35, 0.65)


def match_face_to_document(
    doc_image: np.ndarray,
    live_image: np.ndarray,
    doc_type: str,
    chip_face_bytes: Optional[bytes] = None,
) -> dict:
    """
    Compare document face photo against live face capture.

    Args:
        doc_image:        BGR numpy array of the full document image.
        live_image:       BGR numpy array from live camera capture.
        doc_type:         'Aadhaar' | 'Passport' | 'Visa'
        chip_face_bytes:  If provided (from DG2), use chip JPEG photo instead
                          of cropping from doc_image. Chip photo is authenticated.

    Returns:
        dict with keys:
          similarity (float), verdict (str), score (float),
          doc_embedding_source (str), error (str|None)
    """
    # Step 1: Get document face embedding
    if chip_face_bytes is not None:
        doc_face = _decode_chip_face(chip_face_bytes)
        embedding_source = "chip_dg2_authenticated"
    else:
        doc_face = _crop_face_region(doc_image, doc_type)
        embedding_source = "ocr_crop"

    if doc_face is None:
        return {
            "similarity": None,
            "verdict": "UNKNOWN",
            "score": 0.5,
            "doc_embedding_source": embedding_source,
            "error": "Could not extract face from document",
        }

    doc_embedding = get_embedding(doc_face)
    live_embedding = get_embedding(live_image)

    if doc_embedding is None:
        return {
            "similarity": None,
            "verdict": "UNKNOWN",
            "score": 0.5,
            "doc_embedding_source": embedding_source,
            "error": "Face not detected in document photo",
        }

    if live_embedding is None:
        return {
            "similarity": None,
            "verdict": "UNKNOWN",
            "score": 0.5,
            "doc_embedding_source": embedding_source,
            "error": "Face not detected in live capture",
        }

    # Step 2: Compare
    sim = cosine_similarity(doc_embedding, live_embedding)
    verdict, score = similarity_to_verdict(sim)

    return {
        "similarity": round(sim, 4),
        "verdict": verdict,
        "score": score,
        "doc_embedding_source": embedding_source,
        "error": None,
    }


# ── Helpers ────────────────────────────────────────────────────────────────────


def _crop_face_region(image: np.ndarray, doc_type: str) -> Optional[np.ndarray]:
    """Crop the expected face region from document image using Haar detection with layout ROI fallback."""
    if image is None or getattr(image, "size", 0) == 0:
        return None

    h, w = image.shape[:2]

    # 1. Attempt dynamic frontal face detection
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
        if len(faces) > 0:
            # Pick largest detected face
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            # Add 15% boundary margin
            margin_x = int(0.15 * fw)
            margin_y = int(0.15 * fh)
            nx1 = max(0, fx - margin_x)
            ny1 = max(0, fy - margin_y)
            nx2 = min(w, fx + fw + margin_x)
            ny2 = min(h, fy + fh + margin_y)
            crop = image[ny1:ny2, nx1:nx2]
            if crop.size > 0:
                return crop
    except Exception:
        pass

    # 2. Fallback to percentage-based layout regions
    dt = doc_type.upper()
    if dt == "AADHAAR":
        x1f, y1f, x2f, y2f = _AADHAAR_FACE_REGION
    elif dt in ("PASSPORT", "VISA"):
        x1f, y1f, x2f, y2f = _PASSPORT_FACE_REGION
    else:
        # Default top-left or center-left quadrant for DL / Generic IDs
        x1f, y1f, x2f, y2f = (0.05, 0.15, 0.40, 0.70)

    x1, y1 = int(x1f * w), int(y1f * h)
    x2, y2 = int(x2f * w), int(y2f * h)

    crop = image[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def _decode_chip_face(jpeg_bytes: bytes) -> Optional[np.ndarray]:
    """Decode chip DG2 JPEG/JPEG2000 bytes to numpy BGR array."""
    try:
        buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None
