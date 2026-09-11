# 🛡️ Muj_Asymptotes — Enterprise-Grade Multi-Modal Fraud Detection & Identity Verification

> **HackMUJ 4.0 Submission** | **PS#3 — Deepfake and Synthetic Identity Detection**
> **Theme:** Cybersecurity & Defence

---

## 📸 Executive Summary & Strategic Pitch

Modern identity fraud in high stakes KYC (such as identity cards, generic IDs, video KYC, and biometric onboarding) has evolved far beyond manual image editing or simple document forgery. Today's threat landscape leverages generative AI models, real time video manipulation, and voice cloning to bypass standard security filters.

**Muj_Asymptotes** is an end-to-end, multi-modal fraud detection and identity verification platform engineered to detect, cross-check, and fuse multiple attack vectors concurrently under strict, tamper-evident auditing.

---

## 🎯 Threat Model & Core Capabilities

| Vector | Attack Methodology | Detection Strategy |
| --- | --- | --- |
| **GenAI Documents** | Diffusion models (Midjourney, Stable Diffusion) generating realistic fake IDs.

 | Radial FFT frequency spectrum analysis & EXIF provenance checks.

 |
| **Face Deepfakes** | Real time virtual camera injection and face swapping (DeepFaceLab, SimSwap).

 | Spatial boundary blur (Laplacian variance) & temporal jitter analysis.

 |
| **Replay Attacks** | Playback of previously recorded legitimate KYC video sessions.

 | Dynamic cryptographic challenge (active liveness + single-use nonce).

 |
| **Voice Cloning / TTS** | Synthetic voice synthesis (ElevenLabs, VITS, Bark) mimicking victims.

 | Vocoder artifact detection, MFCC delta variance, & spectral flatness.

 |

---

## 🏗️ System Architecture & Data Pipeline

```
                            [ Web Camera & Mic / User Upload ]
                                            │
                                            ▼
                                   [ frontend/static/ ]
                            (app.js & dashboard.html)[cite: 1]
                                            │
                        POST /api/verify or POST /api/challenge[cite: 1]
                                            │
                                            ▼
                                  [ api/orchestrator.py ]
                        (FastAPI Controller & ThreadPoolExecutor)[cite: 1]
                                            │
        ┌───────────────────┬───────────────┴───────────────┬───────────────────┐
        │                   │                               │                   │
        ▼                   ▼                               ▼                   ▼
  [ Signal A ]        [ Signal B ]                    [ Signal C-i ]      [ Signals C-ii & C-iii ]
   GenAI Doc          Face Deepfake                  Active Liveness       ASR & Voice Spoof
 (genai_detector)   (deepfake_detector)                (challenge)          (asr & antispoof)[cite: 1]
        │                   │                               │                   │
        └───────────────────┼───────────────────────────────┴───────────────────┘
                            ▼
                   [ modules/decision/calibration.py ]
                   (Platt Scaling via calibration.json)[cite: 1]
                            │
                            ▼
                     [ modules/decision/fusion.py ]
             (Score Aggregation + Anti-Dilution Hard Floors)[cite: 1]
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
      [ final verdict ]          [ shared/audit_chain.py ]
  CLEAR / REVIEW / FLAGGED    (SHA-256 HMAC Chained Ledger
   (0-30)   (31-69)   (70-100)   backed by 2-of-3 Shamir Shares)[cite: 1]

```

### Key Components

1. **Edge Client (`frontend/static/app.js`)**:
* Ambient brightness checking (rejects feeds with low lux).


* MediaPipe FaceDetection extracts face crops browser side to optimize bandwidth.


* Synchronizes audio/video capture with dynamic challenge phrases.




2. **Challenge Engine (`modules/face/challenge.py`)**:
* Generates single-use session tokens: dynamic physical gesture (`turn_left`, `blink_twice`) combined with a spoken phrase containing a random 4-digit nonce and timestamp.


* Eliminates session replay attacks.




3. **Parallel Forensics Layer (`api/orchestrator.py`)**:
* **`genai_doc`**: Radial FFT spectrum analysis detecting checkerboard/upscaling grid artifacts + EXIF metadata verification.


* **`face_deepfake`**: Laplacian frame variance & inter-frame temporal warping analysis.


* **`active_liveness`**: Action verification for requested gestures.


* **`name_match`**: Vosk ASR + RapidFuzz token matching verifying spoken name, date, and nonce against extracted document OCR text.


* **`voice_spoof`**: Spectral flatness, high-frequency variance, and vocoder pattern inspection.




4. **Decision & Fusion Engine (`modules/decision/fusion.py`)**:
* Calibrates raw detector outputs via Platt Scaling logistic curves into standardized statistical probabilities.


* **Anti-Dilution Hard Floor**: If a critical check (such as `face_deepfake` or `voice_spoof`) triggers, the overall score is instantly floored to **$\ge 70.0$ (FLAGGED)**, preventing clean signals from masking a severe threat.




5. **Tamper Proof Cryptographic Ledger (`shared/audit_chain.py`)**:
* All verification events are appended to a SHA-256 HMAC hash chained SQLite ledger (`audit.db`).


* The HMAC secret pepper is protected using a **2-of-3 Shamir Secret Sharing (SSS)** scheme across compliance officers (`shared/pepper_sss.py`), ensuring historical records cannot be altered by a single actor.





---

## 🔧 Voice Anti-Spoofing & Platt Scaling Mechanics

The platform integrates a native spoof-likelihood convention across all audio channels:

* **Source Mechanics (`modules/audio/antispoof.py`)**: Voice signals calculate native spoof probability directly at the source, ensuring accurate triggering during live inference:


```python
prob_spoof = 1.0 - score
raw_score = float(np.clip(prob_spoof, 0.0, 1.0))
confidence = apply_calibration(raw_score, "voice_spoof")
triggered = bool(raw_score > SIGNAL_TRIGGER[SignalId.VOICE_SPOOF])

```


* **Unified Calibration (`calibration.json`)**: Unified Platt scaling curves across all forensic models:


* `genai_doc`: slope $= +0.6483$, intercept $= +0.1104$

* `face_deepfake`: slope $= +0.6946$, intercept $= -0.0774$

* `voice_spoof`: slope $= +1.5440$, intercept $= -1.0992$



* **Enforcement**: Synthetic voice inputs trigger `voice_spoof`, engage the anti-dilution hard floor, and assign the final decision to **FLAGGED** (score $70.0$).



---

## 🚀 Quickstart & Deployment Guide

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/shashwatkhandelwal-debug/Muj_Asymptotes.git
cd Muj_Asymptotes

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt

```

### 2. Launching the Platform

```bash
# Start the FastAPI orchestrator backend
uvicorn api.orchestrator:app --reload --port 8000

```

Access the web portal interfaces:

* **KYC Capture Portal**: `http://localhost:8000/static/capture.html`

* **Auditor Dashboard**: `http://localhost:8000/static/dashboard.html`


---