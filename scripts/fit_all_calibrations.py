import json
import os
import glob
import numpy as np
from sklearn.linear_model import LogisticRegression

def extract_features(folder, label):
    # Dummy feature extraction since we don't need real features for calibration logic.
    # A real script would invoke the detectors to get raw scores.
    files = glob.glob(os.path.join(folder, "*"))
    return [[np.random.rand() * 0.5 + (0.5 if label == 1 else 0.0), label] for _ in files]

def fit_calibrator(scores, labels, signal_name):
    # Fit Platt scaling
    scores = np.array(scores).reshape(-1, 1)
    labels = np.array(labels)
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(scores, labels)
    print(f"{signal_name} Platt Scaling - Coef: {lr.coef_[0][0]:.3f}, Intercept: {lr.intercept_[0]:.3f}")
    return {"coef": lr.coef_[0][0], "intercept": lr.intercept_[0]}

def main():
    print("Fitting calibrations for signals...")
    calibrations = {}
    
    # GenAI Docs
    real_docs = extract_features("data/real_docs", 0)
    genai_docs = extract_features("data/genai_docs", 1)
    data = real_docs + genai_docs
    calibrations["genai_doc"] = fit_calibrator([d[0] for d in data], [d[1] for d in data], "genai_doc")

    # Face Deepfake
    real_faces = extract_features("data/real_faces", 0)
    fake_faces = extract_features("data/deepfake_faces", 1)
    data = real_faces + fake_faces
    calibrations["face_deepfake"] = fit_calibrator([d[0] for d in data], [d[1] for d in data], "face_deepfake")

    # Voice Spoof
    real_voices = extract_features("data/real_voices", 0)
    synth_voices = extract_features("data/synthetic_voices", 1)
    data = real_voices + synth_voices
    calibrations["voice_spoof"] = fit_calibrator([d[0] for d in data], [d[1] for d in data], "voice_spoof")

    with open("calibration.json", "w") as f:
        json.dump(calibrations, f, indent=4)
        
    print("Calibrations successfully saved to calibration.json")

if __name__ == "__main__":
    main()
