import json
import logging
import sys
from pathlib import Path

# Ensure repo root is on sys.path
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from sklearn.linear_model import LogisticRegression
import numpy as np

from shared.contracts import SIGNAL_TRIGGER, SignalId

logger = logging.getLogger(__name__)

CALIBRATION_FILE = Path("calibration.json")

def fit_calibrator(raw_scores: list[float], labels: list[int],
                   signal_name: str, method: str = "platt") -> dict:
    """
    Fit Platt scaling: LogisticRegression(C=1e10) on raw_scores reshaped to (-1,1).
    labels: 0 = real/genuine, 1 = fake/suspicious.
    Persist {signal_name: {"coef": float, "intercept": float}} to calibration.json
    (merge, don't overwrite other signals).
    Returns the params dict for this signal.
    If len(raw_scores) < 10, log a warning and use a fixed threshold: calibrated
    score = 1.0 if raw_score > SIGNAL_TRIGGER[SignalId(signal_name)] else 0.0.
    Persist {"method": "threshold"} in that case.
    """
    data = {}
    if CALIBRATION_FILE.exists():
        try:
            with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("Could not read existing calibration file: %s", e)

    if len(raw_scores) < 10 or len(set(labels)) < 2:
        logger.warning(
            "Insufficient data (%d samples, %d classes) for %s. Using fixed threshold.",
            len(raw_scores), len(set(labels)), signal_name
        )
        params = {"method": "threshold"}
    else:
        X = np.array(raw_scores, dtype=float).reshape(-1, 1)
        y = np.array(labels, dtype=int)
        clf = LogisticRegression(C=1e10)
        clf.fit(X, y)
        params = {
            "coef": float(clf.coef_[0][0]),
            "intercept": float(clf.intercept_[0])
        }

    data[signal_name] = params
    try:
        with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.warning("Could not persist calibration to %s: %s", CALIBRATION_FILE, e)

    return params

def apply_calibration(raw_score: float, signal_name: str) -> float:
    """
    Load calibration.json. If signal_name has {"coef","intercept"}, apply sigmoid:
      z = coef * raw_score + intercept
      return 1 / (1 + exp(-z))
    If {"method": "threshold"}, apply the fixed-threshold rule.
    If signal_name not in file, return raw_score unchanged (no calibration yet).
    """
    if not CALIBRATION_FILE.exists():
        return float(raw_score)

    try:
        with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("Could not read calibration file: %s", e)
        return float(raw_score)

    if signal_name not in data:
        return float(raw_score)

    entry = data[signal_name]
    if "coef" in entry and "intercept" in entry:
        z = entry["coef"] * float(raw_score) + entry["intercept"]
        # Stable sigmoid
        if z >= 0:
            return float(1.0 / (1.0 + np.exp(-z)))
        else:
            ez = np.exp(z)
            return float(ez / (1.0 + ez))
    elif entry.get("method") == "threshold":
        try:
            trig = SIGNAL_TRIGGER[SignalId(signal_name)]
        except Exception:
            trig = 0.5
        return 1.0 if float(raw_score) > trig else 0.0

    return float(raw_score)

if __name__ == "__main__":
    np.random.seed(42)
    scores = np.random.uniform(0.0, 1.0, 50).tolist()
    labels = [1 if s > 0.5 else 0 for s in scores]
    params = fit_calibrator(scores, labels, "test_signal")
    calibrated = apply_calibration(0.75, "test_signal")
    assert 0.0 <= calibrated <= 1.0, f"Calibrated score out of range: {calibrated}"
    print("CALIBRATION OK")
