import os
import sys
import random
import datetime
import time
import logging
from dataclasses import dataclass
from pathlib import Path

# Ensure repo root is on sys.path
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from shared.contracts import SignalResult

logger = logging.getLogger(__name__)

NAMES_POOL = [
    "Priya Sharma", "Aarav Patel", "Rohan Gupta", "Ananya Iyer",
    "Vikram Singh", "Sneha Reddy", "Aditya Verma", "Meera Nair",
    "Rajesh Kumar", "Kavita Rao", "Karan Malhotra", "Pooja Desai"
]

NONCE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # 32 characters; excludes ambiguous 0, O, 1, I, L

def generate_nonce(length: int = 5) -> str:
    """Generate a speakable length-N uppercase alphanumeric nonce."""
    return "".join(random.choices(NONCE_ALPHABET, k=length))

@dataclass
class Challenge:
    action: str           # "blink_twice" | "turn_left" | "turn_right"
    nonce: str            # 5-character mixed alphanumeric string (e.g. 'A7K2M')
    date_str: str         # ISO date string
    expected_name: str    # from document OCR or randomly sampled from pool

    @property
    def spoken_phrase(self) -> str:
        return f"{self.expected_name} {self.date_str} {self.nonce}"

def issue_challenge(ocr_name: str = "") -> Challenge:
    name = ocr_name.strip() if ocr_name and ocr_name.strip() and ocr_name.strip().lower() != "unknown" else random.choice(NAMES_POOL)
    return Challenge(
        action=random.choice(["blink_twice", "turn_left", "turn_right"]),
        nonce=generate_nonce(5),
        date_str=datetime.date.today().isoformat(),
        expected_name=name,
    )

def _decode_frames(video_path: str, max_frames: int = 30) -> list:
    """
    Decode up to max_frames from video_path using cv2.VideoCapture.
    Sample evenly: stride = total_frames // max_frames.
    Return list of numpy RGB arrays (H,W,3). If cv2 unavailable or file not
    found, return []. Do NOT raise.
    """
    if not os.path.exists(video_path):
        return []
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stride = max(1, total_frames // max_frames) if total_frames > 0 else 1

        frames = []
        idx = 0
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % stride == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)
            idx += 1

        cap.release()
        return frames
    except Exception as e:
        logger.debug("Failed to decode video frames: %s", e)
        return []

def run_challenge(video_path: str, audio_path: str, ch: Challenge,
                  ocr_name: str) -> list[SignalResult]:
    """
    1. frames = _decode_frames(video_path)
    2. active_liveness  = score_action(frames, ch.action)
    3. name_match       = score_name_match(audio_path, ocr_name, ch.nonce, ch.date_str)
    4. voice_spoof      = score_voice_spoof(audio_path)
    5. Return [active_liveness, name_match, voice_spoof]
       (detect_face_deepfake is called separately in the orchestrator and
       receives the same frames via the API layer — do NOT call it here)
    """
    from modules.face.active_liveness import score_action
    from modules.audio.asr import score_name_match
    from modules.audio.antispoof import score_voice_spoof

    if isinstance(video_path, list):
        frames = video_path
    else:
        frames = _decode_frames(video_path)
    active_liveness = score_action(frames, ch.action)
    name_match = score_name_match(audio_path, ocr_name, ch.nonce, ch.date_str)
    voice_spoof = score_voice_spoof(audio_path)

    return [active_liveness, name_match, voice_spoof]

if __name__ == "__main__":
    ch = issue_challenge("Priya Sharma")
    assert ch.action in ["blink_twice", "turn_left", "turn_right"]
    assert len(ch.nonce) == 4
    assert ch.expected_name == "Priya Sharma"

    results = run_challenge("/nonexistent.mp4", "/nonexistent.wav", ch, "Priya Sharma")
    assert isinstance(results, list)
    assert len(results) == 3
    assert all(isinstance(r, SignalResult) for r in results)
    print("CHALLENGE OK")
