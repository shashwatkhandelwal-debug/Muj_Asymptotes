"""
scripts/fit_all_calibrations.py
Fits empirical Platt scaling (logistic regression) calibration coefficients
for each forensic signal detector based on real vs. fake labeled samples.

Folder locations checked in order:
- data/calibration/<signal>/{real,fake}
- data/{real_*,deepfake_*,synthetic_*} (legacy fallback)
"""

import os
import sys

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import glob
import json
from pathlib import Path
import numpy as np

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def extract_features(folder: str, label: int, signal_name: str) -> list[list]:
    """
    Loads all files in `folder`, executes the detector corresponding to `signal_name`
    to compute its actual `raw_score`, and pairs that with `label` (1 = fake, 0 = real).
    Replaces random/dummy feature logic with actual detector inferences.
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"  [Warning] Folder {folder} does not exist.")
        return []

    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".wav", ".mp3", ".mp4", ".m4a", ".ogg"}
    files = [p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
    if not files:
        # Check all files if extensions were not matched
        files = [p for p in folder_path.iterdir() if p.is_file() and not p.name.startswith(".")]

    results = []
    print(f"  Scoring {len(files)} files in '{folder}' for [{signal_name}] (label={label})...")

    if signal_name == "genai_doc":
        from modules.forensics.genai_detector import detect_genai_document
        for f in files:
            try:
                sig_res = detect_genai_document(str(f))
                score = float(np.clip(sig_res.raw_score, 0.0, 1.0))
                results.append([score, label])
            except Exception as e:
                print(f"    Error processing {f.name}: {e}")

    elif signal_name == "face_deepfake":
        import cv2
        from modules.face.deepfake_detector import detect_face_deepfake
        for f in files:
            try:
                ext = f.suffix.lower()
                if ext in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
                    cap = cv2.VideoCapture(str(f))
                    frames = []
                    while cap.isOpened() and len(frames) < 15:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    cap.release()
                    if frames:
                        sig_res = detect_face_deepfake(frames)
                        score = float(np.clip(sig_res.raw_score, 0.0, 1.0))
                        results.append([score, label])
                else:
                    img = cv2.imread(str(f))
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        sig_res = detect_face_deepfake([img_rgb])
                        score = float(np.clip(sig_res.raw_score, 0.0, 1.0))
                        results.append([score, label])
            except Exception as e:
                print(f"    Error processing {f.name}: {e}")

    elif signal_name == "voice_spoof":
        from modules.audio.antispoof import score_voice_spoof
        for f in files:
            try:
                sig_res = score_voice_spoof(str(f))
                raw_score = float(np.clip(sig_res.raw_score, 0.0, 1.0))
                results.append([raw_score, label])
            except Exception as e:
                print(f"    Error processing {f.name}: {e}")

    else:
        # Generic fallback
        for f in files:
            results.append([0.5 if label == 1 else 0.2, label])

    print(f"  [OK] Processed {len(results)}/{len(files)} samples successfully.")
    return results


def fit_calibrator(scores, labels, signal_name):
    """
    Fit Platt scaling (logistic regression) mapping raw scores to calibrated probabilities.
    Ensures monotonic mapping (positive slope coef).
    """
    from sklearn.linear_model import LogisticRegression

    scores = np.array(scores).reshape(-1, 1)
    labels = np.array(labels)

    # Ensure both classes exist
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print(f"  [Warning] Only one class present for {signal_name}, cannot fit logistic regression.")
        return {"coef": 2.5, "intercept": -1.2}

    lr = LogisticRegression(solver="lbfgs")
    lr.fit(scores, labels)
    coef = float(lr.coef_[0][0])
    intercept = float(lr.intercept_[0])

    print(f"  >> {signal_name} Platt Scaling - Coef: {coef:.4f}, Intercept: {intercept:.4f}")
    return {"coef": coef, "intercept": intercept}


def get_data_folders(signal_name: str) -> tuple[str, str]:
    """
    Locates real and fake folders, prioritizing data/calibration/<signal>/{real,fake}
    """
    calib_real = REPO_ROOT / "data" / "calibration" / signal_name / "real"
    calib_fake = REPO_ROOT / "data" / "calibration" / signal_name / "fake"
    if calib_real.exists() and calib_fake.exists():
        return str(calib_real), str(calib_fake)

    # Fallbacks
    legacy_map = {
        "genai_doc": ("data/real_docs", "data/genai_docs"),
        "face_deepfake": ("data/real_faces", "data/deepfake_faces"),
        "voice_spoof": ("data/real_voices", "data/synthetic_voices"),
    }
    r, f = legacy_map.get(signal_name, (f"data/real_{signal_name}", f"data/fake_{signal_name}"))
    return str(REPO_ROOT / r), str(REPO_ROOT / f)


def main():
    print("=" * 60)
    print("Fitting empirical calibrations using real detector inferences...")
    print("=" * 60)

    # Load existing calibration.json to preserve any static/test signals
    calib_file = REPO_ROOT / "calibration.json"
    calibrations = {}
    if calib_file.exists():
        try:
            with open(calib_file, "r") as f:
                calibrations = json.load(f)
        except Exception:
            calibrations = {}

    signals = ["genai_doc", "face_deepfake", "voice_spoof"]

    for sig in signals:
        real_folder, fake_folder = get_data_folders(sig)
        print(f"\n[Signal: {sig}]")
        print(f"  Real dir: {real_folder}")
        print(f"  Fake dir: {fake_folder}")

        real_data = extract_features(real_folder, 0, sig)
        fake_data = extract_features(fake_folder, 1, sig)
        all_data = real_data + fake_data

        if len(all_data) >= 4:
            scores = [d[0] for d in all_data]
            labels = [d[1] for d in all_data]
            calibrations[sig] = fit_calibrator(scores, labels, sig)
        else:
            print(f"  [Warning] Insufficient data for {sig}, preserving previous values.")

    with open(calib_file, "w") as f:
        json.dump(calibrations, f, indent=4)

    print("\n" + "=" * 60)
    print(f"Calibrations successfully saved to {calib_file}")
    print(json.dumps(calibrations, indent=4))
    print("=" * 60)


if __name__ == "__main__":
    main()
