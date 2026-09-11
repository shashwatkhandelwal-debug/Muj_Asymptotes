# VERITAS: Multi-Modal Deepfake & Synthetic Identity Defense Core

<div align="center">

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](Dockerfile)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-00A67E?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev/)
[![Compliance: DPDP Act 2023](https://img.shields.io/badge/Compliance-DPDP%20Act%202023-emerald?style=for-the-badge&logo=shield)](https://www.meity.gov.in/)
[![Tests Passing: 55/55](https://img.shields.io/badge/Pytest-55%2F55%20PASSED-brightgreen?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

**An On-Device, Real-Time Deepfake Detection & Synthetic Identity Forensic Engine for RBI/UIDAI-Compliant Video-KYC (V-CIP)**

*Zero Cloud Latency • Zero PII Leakage • Multi-Modal Bayesian Fusion • 0% Mock Data • 2-of-3 Shamir's Secret Sharing Cryptographic Audit Chain*

[Quickstart](#quickstart--deployment) • [Forensic Architecture](#multi-modal-forensic-architecture) • [Mathematical Engine](#mathematical--cryptographic-rigor) • [Red-Team Lab](#interactive-red-team-attack-simulator) • [Regulatory Compliance](#regulatory-compliance--dpdp-act-2023)

---

</div>

## 1. Executive Summary & Problem Context

Modern identity fraud in high-stakes video onboarding (banking, fintech, border control, and government subsidies) has outpaced conventional single-modal verification controls. Adversaries deploy:
- **Diffusion Models & Neural GANs** to fabricate photorealistic, never-existed Aadhaar and identity documents.
- **Real-Time Live Face Swapping & Virtual Camera Injections** to bypass traditional passive blink and liveness detectors.
- **Neural Voice Cloning & Zero-Shot TTS** to defeat voice biometric systems and scripted spoken prompts.
- **Pre-Recorded Replay Loops** to reuse hijacked authentication sessions.

### Critical Vulnerabilities in Conventional Systems
1. **The Average Score Trap**: Most commercial engines calculate an unweighted or naive arithmetic average across components. An attacker with a clean Photoshop document and a high-quality deepfake face will erroneously pass with an acceptable composite score.
2. **Cloud Exfiltration Risk**: Transmitting live biometric video feeds to third-party cloud APIs introduces latency, creates massive centralized honeypots, and violates data protection statutes such as India's **DPDP Act (2023)**.
3. **Black-Box Opacity**: Opaque probability numbers provide compliance officers and legal auditors with no explainable, deterministic evidence trail.

**VERITAS** addresses these vulnerabilities via an **on-device, multi-modal verification platform** that decomposes verification across visual, acoustic, and document domains in parallel, applies empirical **Platt scaling** calibration, enforces **non-dilutable asymmetric risk floors**, and cryptographically seals every decision into a **Shamir-split HMAC-SHA256 tamper-evident audit ledger**.

---

## 2. Multi-Modal Forensic Architecture

```
                                  +----------------------------------------------+
                                  |      Client Live Capture Session (Web/Mobile)|
                                  |   (Aadhaar Front/QR, 3D FaceStream, Audio)   |
                                  +----------------------┬-----------------------+
                                                         | HTTPS Multipart
                                                         v
                                  +----------------------------------------------+
                                  |        FastAPI Async Forensic Pipeline       |
                                  |             (api/orchestrator.py)            |
                                  +----------------------┬-----------------------+
                                                         |
                 +---------------------------------------+---------------------------------------+
                 |                                       |                                       |
                 v                                       v                                       v
  +-----------------------------+         +-----------------------------+         +-----------------------------+
  |      DOCUMENT FORENSICS     |         |       FACIAL FORENSICS      |         |       AUDIO FORENSICS       |
  +-----------------------------+         +-----------------------------+         +-----------------------------+
  | 1. Aadhaar UID Checksum     |         | 1. Dual-Domain Deepfake     |         | 1. Voice Anti-Spoofing      |
  |    (Verhoeff Dihedral D5)   |         |    - Laplacian spatial blur |         |    - MFCC Delta-Variance    |
  | 2. Backside Secure QR PKI   |         |    - Temporal frame jitter  |         |    - Spectral Flatness      |
  |    - RSA-2048 Digital Sig   |         | 2. Active 3D Liveness       |         | 2. Vosk/Whisper Local ASR   |
  |    - QR <-> OCR Cross-Check |         |    - MediaPipe FaceMesh     |         |    - 5-Char Dynamic Nonce   |
  | 3. GenAI Document Detection |         |    - Head Pose Tracking     |         | 3. Cross-Modal Alignment    |
  |    - Radial FFT Spectrum    |         | 3. Anti-Replay TTL Window   |         |    - RapidFuzz Spoken Name  |
  |    - CLIP ViT-B/32 Probe    |         |    - Single-use Action Nonce|         |      vs Document OCR Name   |
  | 4. JPEG ELA & EXIF Sensor   |         |                             |         |                             |
  +--------------┬--------------+         +--------------┬--------------+         +--------------┬--------------+
                 |                                       |                                       |
                 +---------------------------------------+---------------------------------------+
                                                         |
                                                         v
                                  +----------------------------------------------+
                                  |   Unified SignalResult Interface Contracts   |
                                  |             (shared/contracts.py)            |
                                  +----------------------┬-----------------------+
                                                         |
                                                         v
                                  +----------------------------------------------+
                                  |        Platt Scaling Calibration Engine      |
                                  |  P(attack | raw_score) = 1 / (1 + e^(w*x+b)) |
                                  +----------------------┬-----------------------+
                                                         |
                                                         v
                                  +----------------------------------------------+
                                  |         Bayesian Multi-Modal Fusion          |
                                  |   Asymmetric Anti-Dilution Risk Floors       |
                                  +----------------------┬-----------------------+
                                                         |
                         +-------------------------------+-------------------------------+
                         v                                                               v
          +-----------------------------+                                 +-----------------------------+
          |      DECISION ENGINE        |                                 |   CRYPTOGRAPHIC AUDIT CHAIN |
          |  CLEAR   | REVIEW | FLAGGED |                                 |  HMAC-SHA256 Hash-Linked DB |
          |  Dynamic Risk Explanations  |                                 |  2-of-3 Shamir Secret Share |
          +-----------------------------+                                 +-----------------------------+
```

---

## 3. Mathematical & Cryptographic Rigor

### 3.1 UIDAI Aadhaar Verhoeff D5 Dihedral Group Checksum
Every genuine 12-digit Indian Aadhaar number is mathematically constrained by the **Verhoeff algorithm** operating over the non-commutative dihedral group $D_5$ (symmetries of a regular pentagon):

$$\sum_{i=0}^{n-1} d\left(c, F(d_i, \mathrm{inv}(i \bmod 8))\right) = 0 \quad \text{in } D_5$$

Where $d(j, k)$ represents the $D_5$ multiplication table, $F(i, j)$ is the permutation matrix, and $\mathrm{inv}(j)$ is the inverse element. This mathematically catches **100% of all single-digit transcription errors** and **95.4% of twin transposition errors** before invoking computer vision or OCR layers.

### 3.2 Empirical Platt Scaling Calibration
Rather than relying on uncalibrated heuristic scores, all detector outputs are transformed into genuine posterior probabilities via empirical **Platt Scaling** fitted with logistic regression against benchmark calibration datasets:

$$P(\text{Attack} \mid \text{Raw Score } x) = \frac{1}{1 + \exp\left(-\left(w \cdot x + b\right)\right)}$$

| Detector Module | Calibration Formulation | Primary Feature Anchor | Ground Truth Calibration Data |
| :--- | :--- | :--- | :--- |
| **`face_deepfake`** | Platt Logistic ($\sigma$) | Laplacian Spatial Variance + Inter-Frame Delta | Real Video Frames vs Deepfake Swap Dataset |
| **`voice_spoof`** | Platt Logistic ($\sigma$) | MFCC Delta Variance + Spectral Flatness | Genuine Human Audio vs Neural TTS Dataset |
| **`genai_doc`** | Platt Logistic ($\sigma$) | CLIP ViT-B/32 Probe + Radial FFT Spectrum | Camera Captured Documents vs Midjourney / SD |
| **`crossfield`** | Deterministic Math | Verhoeff $D_5$ + UIDAI RSA-2048 PKCS#1 v1.5 | Mathematical Checksum / PKI Verification |

### 3.3 Asymmetric Anti-Dilution Risk Floors
Standard weighted averaging permits catastrophic biometric attacks to be diluted by pristine ancillary records. VERITAS enforces mathematically guaranteed **Hard Severity Risk Floors**:

$$\text{Final Risk Score} = \max\left(\sum_{i} \text{Penalty}(\text{Signal}_i), \bigvee_{j \in \text{HARD}} \text{Floor}(\text{Signal}_j)\right)$$

```python
# Hard Floor Guarantee: A deepfake face or forged QR instantly triggers FLAGGED
if any(sig.severity == Severity.HARD and sig.triggered for sig in signals):
    composite_risk = max(composite_risk, 70.0)
    decision_tier = RiskTier.FLAGGED
```

### 3.4 2-of-3 Shamir's Secret Sharing (SSS) & Hash-Chained Ledger
To prevent insider tampering with forensic verification records:
1. Every verification event is cryptographically sealed into an append-only **HMAC-SHA256 block ledger**:
   $$H_n = \text{HMAC-SHA256}\left(H_{n-1} \parallel \text{Timestamp} \parallel \text{SessionID} \parallel \text{DecisionTier} \parallel \text{SignalVector}, \text{Pepper}\right)$$
2. The verification master $\text{Pepper}$ is split into 3 polynomial shares using Shamir's Secret Sharing over the finite field $\mathbb{F}_{256}$:
   $$f(x) = S + a_1 x \pmod p$$
   Reconstructing the pepper to audit or verify database integrity requires a **2-of-3 quorum** of distinct compliance officers, ensuring non-repudiation and immutable evidence trails at rest.

---

## 4. Multi-Modal Forensic Capabilities

| Vector Category | Forensic Channel | Detection Methodology | Attack Class Defeated |
| :--- | :--- | :--- | :--- |
| **Visual Deepfakes** | Facial Video Stream | Dual-Domain: Spatial Laplacian variance + inter-frame optical motion tracking | DeepFaceLab, SimSwap, Roop, FaceFusion, Live Virtual Camera Loops |
| **Audio Cloning** | Vocal Audio Stream | Dual-Acoustic: MFCC Delta-Variance + Spectral Energy Flatness Heuristics | ElevenLabs, Tortoise-TTS, VITS, Bark, Neural Voice Conversions |
| **AI Documents** | Document Photo | Dual-Domain: Radial Fast Fourier Transform (FFT) + CLIP ViT-B/32 Linear Probe | Midjourney, Stable Diffusion, DALL-E synthesized ID cards |
| **Spliced Documents** | Document Photo | JPEG Discrete Cosine Transform (DCT) Error Level Analysis (ELA) + EXIF Sensors | Adobe Photoshop, GIMP copy-paste demographic field replacements |
| **Forged Aadhaar** | Document & QR | Dihedral $D_5$ Verhoeff Checksum + UIDAI RSA-2048 PKCS#1 v1.5 QR Verification | Synthesized fake UID numbers, counterfeit un-signed QR codes |
| **Session Replay** | Video + Audio | Dynamic 5-character randomized alphanumeric nonce + 3D Head Pose tracking (<10s TTL) | Pre-recorded playback loops, virtual camera software streams |

---

## 5. Interactive Red-Team Attack Simulator

VERITAS includes an integrated **Red-Team Adversarial Testbench** (`/redteam.html`) allowing judges, security researchers, and auditors to execute 1-click live adversarial attacks against the running pipeline:

- **6 Live Adversarial Vectors (0% Mock Data)**:
  1. **Face Swap / Deepfake Face**: Injects synthetic face swap frames testing spatial edge degradation and temporal continuity.
  2. **Cloned / Neural Voice Synthesis**: Injects neural TTS audio evaluating acoustic vocoder delta metrics.
  3. **AI-Generated ID Document**: Injects diffusion-synthesized identity cards testing radial FFT spectrum anomaly.
  4. **Tampered Document (Spliced/EXIF)**: Injects Photoshop-manipulated documents with modified DCT error levels.
  5. **Invalid Aadhaar QR / Checksum**: Injects forged cards with failing Verhoeff $D_5$ group checksums.
  6. **Genuine Clean Submission**: Injects pristine, authentic identity data to demonstrate calibrated high trust (>95%).
- **Live Bayesian Trust Decay Visualization**: Graphically animates the real-time degradation of identity confidence as each forensic signal is ingested.
- **Direct Ledger Navigation**: Direct click-through verification from attack outcomes into the immutable cryptographic audit ledger.

---

## 6. Regulatory Compliance & DPDP Act 2023

VERITAS is engineered to comply with India's **Digital Personal Data Protection (DPDP) Act, 2023** and **RBI Master Direction on Video-Based Customer Identification Process (V-CIP)**:

1. **Zero Cloud PII Transmission**: All facial landmarking, voice processing, OCR parsing, and cryptographic checks execute strictly on-premise. No biometric data leaves the local host environment.
2. **Automated Aadhaar UID Masking**: Fully complies with UIDAI regulations and Section 8 of the DPDP Act by masking the first 8 digits of extracted UIDs (`19XXXXXX5678`).
3. **Ephemeral In-Memory Processing**: Client-side MediaPipe FaceMesh isolates facial keyframes directly in browser memory (<200 KB). Uploaded binary buffers are wiped immediately post-inference.
4. **Data Minimization & Purpose Limitation**: Persists only cryptographic hashes, Platt calibration confidences, and model provenance metadata in `audit.db` — raw video streams are never written to disk.

---

## 7. Quickstart & Deployment
 
### 7.1 Option A: Local Docker Deployment (Recommended — 100% On-Premise)

VERITAS is fully containerized with zero external cloud dependencies, packaging all native system dependencies (`Tesseract OCR`, `OpenCV`, `PyZBar`, `FFmpeg`, `libsndfile`) in an isolated container.

#### 1. One-Click Launch via Docker Compose
```bash
# Clone the repository
git clone https://github.com/shashwatkhandelwal-debug/Muj_Asymptotes.git
cd Muj_Asymptotes

# Build and start the on-premise container
docker compose up --build -d

# View real-time container logs
docker compose logs -f
```

#### 2. Or Build & Run Standalone Docker Container
```bash
# Build Docker image
docker build -t veritas:latest .

# Run on port 8000 with persistent audit ledger
docker run -d \
  --name veritas_core \
  -p 8000:8000 \
  -v $(pwd)/audit.db:/app/audit.db \
  veritas:latest
```

#### 3. Run Automated Tests Inside Docker
```bash
docker compose exec veritas pytest tests/ -v
```

---

### 7.2 Option B: Native Host Environment Setup

```bash
# Clone the repository
git clone https://github.com/shashwatkhandelwal-debug/Muj_Asymptotes.git
cd Muj_Asymptotes

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux / macOS

# Install dependencies
pip install -r requirements.txt

# Launch Application Server
python -m uvicorn api.orchestrator:app --host 0.0.0.0 --port 8000 --reload
```

---

### 7.3 Access Web Portals
| Portal | Direct URL | Description |
| :--- | :--- | :--- |
| **Live KYC Verification** | `http://localhost:8000/index.html` | 4-Stage Onboarding (Doc OCR -> Dynamic Prompt -> Biometric Nonce -> Verdict) |
| **Aadhaar Dual-Photo Flow** | `http://localhost:8000/upload.html` | Full Aadhaar card + close-up QR verification & cross-validation |
| **Red-Team Attack Lab** | `http://localhost:8000/redteam.html` | 1-Click Adversarial Injection & Real-Time Bayesian Trust Decay |
| **Auditor Telemetry Dashboard** | `http://localhost:8000/dashboard.html` | Forensic telemetry, ELA heatmaps, model provenance, and audit chain |
| **Swagger API Documentation** | `http://localhost:8000/docs` | Interactive OpenAPI endpoints |

---

## 8. Verification & Automated Test Suite

The entire codebase is validated by **55 automated unit, integration, and regression tests** (`pytest tests/`) along with end-to-end browser automation suites (`Playwright`):

```bash
# Run full test suite (55/55 passing)
pytest tests/ -v

# Run Red-Team API & response shape regression check
pytest tests/test_redteam_regression.py -v

# Run cryptographic audit chain & Shamir's Secret Sharing checks
python shared/audit_chain.py
python shared/pepper_sss.py
```

### Automated Test Execution Log
```
collected 55 items
tests/test_full_suite.py .................................................... [ 94%]
tests/test_orchestrator.py .                                                  [ 96%]
tests/test_redteam_regression.py ..                                           [100%]
============================== 55 passed in 69.84s ==============================
```

---

## 9. Evaluation Matrix

| Evaluation Dimension | Conventional Approach | VERITAS Implementation |
| :--- | :--- | :--- |
| **Data Authenticity** | Hardcoded mock figures & random generators | **0% Mock Data** — 100% computed live by forensic pipeline |
| **Biometric Defenses** | Basic cosine face comparison | **Dual-Domain** (Laplacian spatial + temporal inter-frame jitter) |
| **Replay Defense** | Static blink/head nod directives | **Dynamic 5-character vocal nonce** + 3D FaceMesh head-pose tracking |
| **Aadhaar Integrity** | Basic regex format check | **Verhoeff D5 Dihedral Group Checksum** + UIDAI RSA-2048 QR validation |
| **Decision Fusion** | Naive arithmetic averaging | **Platt-scaled empirical probabilities** + Asymmetric Hard Risk Floors |
| **Data Privacy** | Cloud API uploads; raw PII saved | **100% Local Inference**, DPDP Act UID masking, 2-of-3 Shamir SSS Audit Chain |
| **Auditability** | Static forms with opaque final score | **Interactive Red-Team Attack Lab** + Real-Time Trust Decay Animation |

---

## 10. Authors & Project Metadata
- **Team**: MUJ Asymptotes  
- **Hackathon**: HackMUJ 4.0  
- **Track**: Cybersecurity & Defence (PS#3: Deepfake & Synthetic Identity Detection)  
- **License**: MIT License
