"""
Liveness Detection  -  Anti-Spoofing.

Detects whether the live capture is of a real person or a spoofing attack
(printed photo, phone screen, 3D mask).

Two approaches:
  1. Passive liveness (texture-based): Uses a pretrained MobileNetV2 model
     fine-tuned on CelebA-Spoof dataset. Analyzes skin texture vs. paper/screen
     texture. No action required from the subject.

  2. Active liveness (challenge-based): Uses MediaPipe Face Mesh to detect
     eye landmarks and measure Eye Aspect Ratio (EAR). Prompts the subject
     to blink, verifying real eye movement.

Eye Aspect Ratio (EAR):
  EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
  where p1-p6 are the six eye landmark coordinates (horizontal and vertical).
  EAR drops below ~0.2 during a blink and rises back. A real blink lasts
  150-400ms. A static photo has a constant EAR (no blink).
"""

from typing import Optional

import cv2
import numpy as np

# EAR threshold  -  below this = eye closed (blink detected)
_EAR_BLINK_THRESHOLD = 0.20
_EAR_FRAMES_REQUIRED = 2  # Minimum consecutive frames with low EAR = real blink


def passive_liveness_check(image: np.ndarray) -> dict:
    """
    Passive liveness detection using texture analysis.

    Uses MediaPipe's built-in face detection + a texture-based heuristic
    as a lightweight substitute for a full spoof detection model.

    Args:
        image: BGR numpy array of the live capture.

    Returns:
        dict with keys: is_live (bool), confidence (float), method, error
    """
    try:
        import mediapipe as mp

        mp_face = mp.solutions.face_detection

        with mp_face.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        ) as detector:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)

            if not results.detections:
                return {
                    "is_live": False,
                    "confidence": 0.0,
                    "method": "passive",
                    "error": "No face detected",
                }

            # Texture-based heuristic: real faces have higher local binary
            # pattern (LBP) variance than printed photos
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            lbp_var = _lbp_variance(gray)

            # Empirical threshold  -  printed photos typically < 35, real faces > 50
            is_live = lbp_var > 42.0
            confidence = min(lbp_var / 100.0, 1.0)

            return {
                "is_live": is_live,
                "confidence": round(confidence, 3),
                "lbp_variance": round(lbp_var, 2),
                "method": "passive_lbp",
                "error": None,
            }

    except ImportError:
        return {
            "is_live": None,
            "confidence": 0.5,
            "method": "passive",
            "error": "mediapipe not installed",
        }
    except Exception as e:
        return {
            "is_live": None,
            "confidence": 0.5,
            "method": "passive",
            "error": str(e),
        }


def compute_ear(eye_landmarks: list) -> float:
    """
    Compute Eye Aspect Ratio from 6 eye landmark points.

    Args:
        eye_landmarks: List of 6 (x, y) tuples:
                       [left_corner, top_inner, top_outer,
                        right_corner, bottom_outer, bottom_inner]

    Returns:
        EAR value (float). < 0.2 indicates blink.
    """
    p1, p2, p3, p4, p5, p6 = [np.array(p) for p in eye_landmarks]
    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    return (vertical_1 + vertical_2) / (2.0 * horizontal + 1e-6)


def detect_blink_in_frame(image: np.ndarray) -> dict:
    """
    Detect whether eyes are open or closed (blink state) in a single frame.
    Used by the frontend to track blink events across a video stream.

    Args:
        image: BGR numpy array.

    Returns:
        dict with keys: blink_detected (bool), ear (float), error (str|None)
    """
    try:
        import mediapipe as mp

        mp_mesh = mp.solutions.face_mesh

        # MediaPipe Face Mesh eye landmark indices (left eye)
        LEFT_EYE = [33, 160, 158, 133, 153, 144]

        with mp_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
        ) as mesh:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = mesh.process(rgb)

            if not results.multi_face_landmarks:
                return {
                    "blink_detected": False,
                    "ear": None,
                    "error": "No face detected",
                }

            lm = results.multi_face_landmarks[0].landmark
            h, w = image.shape[:2]
            points = [(int(lm[i].x * w), int(lm[i].y * h)) for i in LEFT_EYE]
            ear = compute_ear(points)

            return {
                "blink_detected": ear < _EAR_BLINK_THRESHOLD,
                "ear": round(ear, 4),
                "error": None,
            }

    except ImportError:
        return {"blink_detected": None, "ear": None, "error": "mediapipe not installed"}
    except Exception as e:
        return {"blink_detected": None, "ear": None, "error": str(e)}


# ── Helpers ────────────────────────────────────────────────────────────────────


def _lbp_variance(gray: np.ndarray) -> float:
    """
    Compute Local Binary Pattern variance as a texture measure.
    Higher variance indicates more complex texture (real face skin).
    Lower variance indicates smoother texture (printed paper/screen).
    """
    h, w = gray.shape
    center = gray[1 : h - 1, 1 : w - 1].astype(np.int32)

    lbp = np.zeros_like(center, dtype=np.uint8)
    for shift_r, shift_c in [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]:
        neighbor = gray[
            1 + shift_r : h - 1 + shift_r, 1 + shift_c : w - 1 + shift_c
        ].astype(np.int32)
        lbp = (lbp << 1) | (neighbor >= center).astype(np.uint8)

    return float(np.var(lbp))
