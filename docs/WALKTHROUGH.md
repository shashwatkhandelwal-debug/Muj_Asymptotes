# Project Status Walkthrough

## Completed Steps
1. **Environment Verification:** All original core tests and HackMUJ detectors are functioning perfectly on their heuristic fallbacks.
2. **Setup:** `models/` and `notebooks/` directories have been initialized with appropriate `.gitignore` tracking.
3. **Synthetic Voices:** 15 synthetic TTS voices have been generated and converted to 16kHz mono WAV format in `data/synthetic_voices/`.
4. **Demo Machine Test:** The HuggingFace `deepfake_detector` pipeline works on the demo machine, eliminating the need for complex ONNX exports.

## Next Steps (Safe Execution Plan)
To proceed without altering or breaking any existing core logic (especially `shared/contracts.py`), we will perform the following safely isolated steps:

### 1. Generate Placeholder Data
We will create a standalone script (`scripts/seed_dummy_data.py`) to generate basic programmatic images and videos. This will populate the `data/` directories without touching any application code.

### 2. Train the GenAI Model (Step 4)
We will run a script to train a Logistic Regression probe on the dummy images and save it to `models/clip_genai_probe.pt`. The existing `genai_detector.py` will automatically detect and load this file.

### 3. Fit Calibration (Step 7)
We will run a new script (`scripts/fit_all_calibrations.py`) that reads the `data/` folders and outputs `calibration.json`. This calibrates the confidence scores without modifying detector logic.

### 4. Robustness & Adversarial Testing (Steps 8 & 9)
We will execute the existing `tests/robustness/harness.py` and `tests/adversarial/selftest.py` scripts. These will read the data and output their results, which we will format into `docs/ROBUSTNESS_RESULTS.md` and `docs/LIMITATIONS.md`.
