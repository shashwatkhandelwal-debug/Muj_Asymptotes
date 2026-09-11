"""
shared/preprocessing.py
Shared image/frame preprocessing. Applied before ANY detector sees a frame.
CLAHE (Contrast Limited Adaptive Histogram Equalization) normalises exposure so
a dim checkpoint or a bright window does not change detector behaviour.
"""
import numpy as np

def clahe_rgb(frame_rgb: np.ndarray) -> np.ndarray:
    """
    Normalise exposure on an RGB uint8 frame using CLAHE on the L channel.
    Returns a new RGB uint8 array. On any failure, returns the input unchanged.
    """
    try:
        import cv2
        lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    except Exception:
        return frame_rgb

def mean_brightness(frame_rgb: np.ndarray) -> float:
    """Mean pixel brightness 0..255. Used for the low-light warning."""
    try:
        return float(np.mean(frame_rgb))
    except Exception:
        return 128.0

LOW_LIGHT_THRESHOLD = 60.0

if __name__ == "__main__":
    import numpy as np
    dark = np.full((100, 100, 3), 30, dtype=np.uint8)
    bright = np.full((100, 100, 3), 200, dtype=np.uint8)
    assert mean_brightness(dark) < LOW_LIGHT_THRESHOLD
    assert mean_brightness(bright) > LOW_LIGHT_THRESHOLD
    out = clahe_rgb(dark)
    assert out.shape == dark.shape and out.dtype == np.uint8
    print("PREPROCESSING OK")
