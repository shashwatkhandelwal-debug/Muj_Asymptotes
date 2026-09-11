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

_MP_TRIED = False
_MP_FACEMESH = None

def _get_facemesh():
    global _MP_TRIED, _MP_FACEMESH
    if _MP_TRIED:
        return _MP_FACEMESH
    _MP_TRIED = True
    try:
        import mediapipe as mp
        _MP_FACEMESH = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    except Exception as e:
        logger.info("Mediapipe FaceMesh unavailable: %s", e)
        _MP_FACEMESH = None
    return _MP_FACEMESH

def _compute_ear(landmarks) -> float:
    # Indices: [33, 160, 158, 133, 153, 144]
    # p1=33, p2=160, p3=158, p4=133, p5=153, p6=144
    p1 = np.array([landmarks[33].x, landmarks[33].y])
    p2 = np.array([landmarks[160].x, landmarks[160].y])
    p3 = np.array([landmarks[158].x, landmarks[158].y])
    p4 = np.array([landmarks[133].x, landmarks[133].y])
    p5 = np.array([landmarks[153].x, landmarks[153].y])
    p6 = np.array([landmarks[144].x, landmarks[144].y])

    dist_2_6 = float(np.linalg.norm(p2 - p6))
    dist_3_5 = float(np.linalg.norm(p3 - p5))
    dist_1_4 = float(np.linalg.norm(p1 - p4)) + 1e-9

    ear = (dist_2_6 + dist_3_5) / (2.0 * dist_1_4)
    return float(ear)

def _compute_yaw(landmarks) -> float:
    # [234, 454] ear tips, [1] nose tip
    nose_x = landmarks[1].x
    left_ear_x = landmarks[234].x
    right_ear_x = landmarks[454].x

    left_w = abs(nose_x - left_ear_x)
    right_w = abs(right_ear_x - nose_x)
    total_w = left_w + right_w + 1e-9

    yaw = (left_w - right_w) / total_w * 90.0
    return float(yaw)

def score_action(frames: list, expected_action: str) -> SignalResult:
    """
    IN:  frames = list of numpy RGB arrays (same list as deepfake_detector gets).
         expected_action = "blink_twice" | "turn_left" | "turn_right".
    OUT: SignalResult(
           signal=SignalId.ACTIVE_LIVENESS,
           raw_score=P(action NOT performed) in [0,1],
           confidence=apply_calibration(raw_score, "active_liveness"),
           triggered=(raw_score > SIGNAL_TRIGGER[SignalId.ACTIVE_LIVENESS]),
           label="Liveness action confirmed: <action>" if not triggered
                 else "Liveness action NOT detected: <action>",
           evidence={"expected_action": str, "detected": bool,
                     "metric_trace": [per-frame EAR or yaw values]}
         )
    If mediapipe is unavailable or no face detected in >50% of frames,
    return raw_score=0.4 (uncertain, below trigger) with ok=True and
    evidence={"error":"mediapipe unavailable"|"no face"}.
    Measure wall time and set result.ms.
    """
    t0 = time.perf_counter()
    if not frames:
        elapsed_ms = max(0.1, (time.perf_counter() - t0) * 1000.0)
        return SignalResult(
            signal=SignalId.ACTIVE_LIVENESS,
            raw_score=0.4,
            confidence=apply_calibration(0.4, "active_liveness"),
            severity=Severity.SOFT,
            triggered=False,
            label=f"Liveness action confirmed: {expected_action}",
            evidence={"expected_action": expected_action, "detected": False, "metric_trace": [], "error": "no frames"},
            ok=True,
            ms=elapsed_ms,
        )

    mesh = _get_facemesh()
    if mesh is None:
        elapsed_ms = max(0.1, (time.perf_counter() - t0) * 1000.0)
        return SignalResult(
            signal=SignalId.ACTIVE_LIVENESS,
            raw_score=0.4,
            confidence=apply_calibration(0.4, "active_liveness"),
            severity=Severity.SOFT,
            triggered=False,
            label=f"Liveness action confirmed: {expected_action}",
            evidence={"expected_action": expected_action, "detected": False, "metric_trace": [], "error": "mediapipe unavailable"},
            ok=True,
            ms=elapsed_ms,
        )

    metric_trace: list[float] = []
    face_count = 0
    total_frames = len(frames)

    for frame in frames:
        try:
            results = mesh.process(frame)
            if results.multi_face_landmarks:
                face_count += 1
                landmarks = results.multi_face_landmarks[0].landmark
                if expected_action == "blink_twice":
                    val = _compute_ear(landmarks)
                else:
                    val = _compute_yaw(landmarks)
                metric_trace.append(val)
            else:
                metric_trace.append(0.0)
        except Exception:
            metric_trace.append(0.0)

    # Check if no face detected in >50% of frames
    if face_count < (total_frames / 2.0):
        elapsed_ms = max(0.1, (time.perf_counter() - t0) * 1000.0)
        return SignalResult(
            signal=SignalId.ACTIVE_LIVENESS,
            raw_score=0.4,
            confidence=apply_calibration(0.4, "active_liveness"),
            severity=Severity.SOFT,
            triggered=False,
            label=f"Liveness action confirmed: {expected_action}",
            evidence={"expected_action": expected_action, "detected": False, "metric_trace": metric_trace, "error": "no face"},
            ok=True,
            ms=elapsed_ms,
        )

    detected = False
    if expected_action == "blink_twice":
        blinks = 0
        in_blink = False
        for ear in metric_trace:
            if ear < 0.2 and not in_blink:
                blinks += 1
                in_blink = True
            elif ear >= 0.2:
                in_blink = False
        detected = (blinks >= 2)
    elif expected_action == "turn_left":
        max_c = 0
        cur = 0
        for y in metric_trace:
            if y < -15.0:
                cur += 1
                max_c = max(max_c, cur)
            else:
                cur = 0
        detected = (max_c >= 3)
    elif expected_action == "turn_right":
        max_c = 0
        cur = 0
        for y in metric_trace:
            if y > 15.0:
                cur += 1
                max_c = max(max_c, cur)
            else:
                cur = 0
        detected = (max_c >= 3)

    raw_score = 0.05 if detected else 0.85
    confidence = apply_calibration(raw_score, "active_liveness")
    triggered = bool(raw_score > SIGNAL_TRIGGER[SignalId.ACTIVE_LIVENESS])
    label = (
        f"Liveness action confirmed: {expected_action}"
        if not triggered
        else f"Liveness action NOT detected: {expected_action}"
    )
    elapsed_ms = max(0.1, (time.perf_counter() - t0) * 1000.0)

    return SignalResult(
        signal=SignalId.ACTIVE_LIVENESS,
        raw_score=raw_score,
        confidence=confidence,
        severity=Severity.SOFT,
        triggered=triggered,
        label=label,
        evidence={
            "expected_action": expected_action,
            "detected": detected,
            "metric_trace": metric_trace,
        },
        ok=True,
        ms=elapsed_ms,
    )

if __name__ == "__main__":
    test_frames = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(30)]
    res = score_action(test_frames, "blink_twice")
    assert isinstance(res, SignalResult)
    assert 0.0 <= res.raw_score <= 1.0
    assert res.ok is True
    print("ACTIVE LIVENESS OK")
