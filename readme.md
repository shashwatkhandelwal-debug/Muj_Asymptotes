# 🛡️ AEGIS-SENTINEL: Multi-Modal Deepfake & Synthetic Identity Defense Core

<div align="center">

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-00A67E?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev/)
[![Compliance: DPDP Act 2023](https://img.shields.io/badge/Compliance-DPDP%20Act%202023-emerald?style=for-the-badge&logo=shield)](https://www.meity.gov.in/)
[![Tests Passing: 55/55](https://img.shields.io/badge/Pytest-55%2F55%20PASSED-brightgreen?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

**An On-Device, Real-Time Deepfake Detection & Synthetic Identity Forensic Engine for RBI/UIDAI-Compliant Video-KYC (V-CIP)**

*Zero Cloud Latency • Zero PII Leakage • Multi-Modal Bayesian Fusion • 0% Mock Data • 2-of-3 Shamir's Secret Sharing Cryptographic Audit Chain*

[🚀 Quickstart](#-quickstart--deployment) • [🔬 Forensic Architecture](#-multi-modal-forensic-architecture) • [🧮 Mathematical Engine](#-mathematical--cryptographic-rigor) • [⚔️ Red-Team Lab](#-interactive-red-team-attack-simulator) • [⚖️ Regulatory Compliance](#-regulatory-compliance--dpdp-act-2023)

---

</div>

## 📌 Executive Summary & Problem Context

Modern identity fraud in high-stakes video onboarding (banking, fintech, border control, and government subsidies) has outpaced conventional single-modal verification controls. Adversaries deploy:
- **Diffusion Models & Neural GANs** to fabricate photorealistic, never-existed Aadhaar and identity documents.
- **Real-Time Live Face Swapping & Virtual Camera Injections** to bypass traditional passive blink/liveness detectors.
- **Neural Voice Cloning & Zero-Shot TTS** to defeat voice biometric systems and scripted spoken prompts.
- **Pre-Recorded Replay Loops** to reuse hijacked authentication sessions.

### Why Conventional Solutions Fail
1. **The "Average Score" Trap**: Most commercial engines average component scores. An attacker with a clean Photoshop document and a high-quality deepfake face will "pass" with a composite score of 80%.
2. **Cloud Exfiltration Risk**: Sending live biometric video feeds to cloud APIs introduces latency and violates data sovereignty laws like India's **DPDP Act (2023)**.
3. **Black-Box Opacity**: Opaque probability scores give compliance auditors no actionable forensic trail for regulatory inquiries.

**AEGIS-SENTINEL** resolves these fatal vulnerabilities through an **on-device, multi-modal verification platform** that decomposes verification across visual, acoustic, and document domains in parallel, applies empirical **Platt scaling** calibration, enforces **non-dilutable asymmetric risk floors**, and cryptographically seals every decision into a **Shamir-split HMAC-SHA256 tamper-evident audit ledger**.

---

## 🏗️ Multi-Modal Forensic Architecture

```
                                  ┌──────────────────────────────────────────────┐
                                  │      Client Live Capture Session (Web/Mobile)│
                                  │   (Aadhaar Front/QR, 3D FaceStream, Audio)   │
                                  └──────────────────────┬───────────────────────┘
                                                         │ HTTPS Multipart
                                                         ▼
                                  ┌──────────────────────────────────────────────┐
                                  │        FastAPI Async Forensic Pipeline       │
                                  │             (api/orchestrator.py)            │
                                  └──────────────────────┬───────────────────────┘
                                                         │
                 ┌───────────────────────────────────────┼───────────────────────────────────────┐
                 │                                       │                                       │
                 ▼                                       ▼                                       ▼
  ┌─────────────────────────────┐         ┌─────────────────────────────┐         ┌─────────────────────────────┐
  │      DOCUMENT FORENSICS     │         │       FACIAL FORENSICS      │         │       AUDIO FORENSICS       │
  ├─────────────────────────────┤         ├─────────────────────────────┤         ├─────────────────────────────┤
  │ 1. Aadhaar UID Checksum     │         │ 1. Dual-Domain Deepfake     │         │ 1. Voice Anti-Spoofing      │
  │    (Verhoeff Dihedral D5)   │         │    • Laplacian spatial blur │         │    • MFCC Delta-Variance    │
  │ 2. Backside Secure QR PKI   │         │    • Temporal frame jitter  │         │    • Spectral Flatness      │
  │    • RSA-2048 Digital Sig   │         │ 2. Active 3D Liveness       │         │ 2. Vosk/Whisper Local ASR   │
  │    • QR ↔ OCR Cross-Check   │         │    • MediaPipe FaceMesh     │         │    • 5-Char Dynamic Nonce   │
  │ 3. GenAI Document Detection │         │    • Head Pose Tracking     │         │ 3. Cross-Modal Alignment    │
  │    • Radial FFT Spectrum    │         │ 3. Anti-Replay TTL Window   │         │    • RapidFuzz Spoken Name  │
  │    • CLIP ViT-B/32 Probe    │         │    • Single-use Action Nonce│         │      vs Document OCR Name   │
  │ 4. JPEG ELA & EXIF Sensor   │         │                             │         │                             │
  └──────────────┬──────────────┘         └──────────────┬──────────────┘         └──────────────┬──────────────┘
                 │                                       │                                       │
                 └───────────────────────────────────────┼───────────────────────────────────────┘
                                                         │
                                                         ▼
                                  ┌──────────────────────────────────────────────┐
                                  │   Unified SignalResult Interface Contracts   │
                                  │             (shared/contracts.py)            │
                                  └──────────────────────┬───────────────────────┘
                                                         │
                                                         ▼
                                  ┌──────────────────────────────────────────────┐
                                  │        Platt Scaling Calibration Engine      │
                                  │  P(attack | raw_score) = 1 / (1 + e^(w*x+b)) │
                                  └──────────────────────┬───────────────────────┘
                                                         │
                                                         ▼
                                  ┌──────────────────────────────────────────────┐
                                  │         Bayesian Multi-Modal Fusion          │
                                  │   Asymmetric Anti-Dilution Risk Floors       │
                                  └──────────────────────┬───────────────────────┘
                                                         │
                         ┌───────────────────────────────┴───────────────────────────────┐
                         ▼                                                               ▼
          ┌─────────────────────────────┐                                 ┌─────────────────────────────┐
          │      DECISION ENGINE        │                                 │   CRYPTOGRAPHIC AUDIT CHAIN │
          │  CLEAR   | REVIEW | FLAGGED │                                 │  HMAC-SHA256 Hash-Linked DB │
          │  Dynamic Risk Explanations  │                                 │  2-of-3 Shamir Secret Share │
          └─────────────────────────────┘                                 └─────────────────────────────┘
```

---

## 🧮 Mathematical & Cryptographic Rigor

### 1. UIDAI Aadhaar Verhoeff $D_5$ Dihedral Group Checksum
Every authentic 12-digit Indian Aadhaar number is mathematically constrained by the **Verhoeff algorithm** operating over the non-commutative dihedral group $D_5$ (symmetries of a regular pentagon):

$$\sum_{i=0}^{n-1} d\left(c, F(d_i, \operatorname{inv}(i mod 8))ight) = 0 \quad 	ext{in } D_5$$

Where $d(j, k)$ is the $D_5$ multiplication table, $p(i, j)$ is the permutation table, and $\operatorname{inv}(j)$ is the inverse element. This mathematically catches **100% of all single-digit transcription errors** and **95.4% of twin transposition errors** before any computer vision or OCR processing.

### 2. Empirical Platt Scaling (Defensible Calibration)
Rather than raw heuristics or arbitrary linear mappings, all detector outputs are calibrated into posterior probabilities using empirical **Platt Scaling** fitted via logistic regression against ground-truth calibration datasets:

$$P(	ext{Attack} \mid 	ext{Raw Score } x) = rac{1}{1 + \exp\left(-\left(w \cdot x + bight)ight)}$$

| Detector Module | Calibration Method | Primary Anchor | Sample Benchmark |
| :--- | :--- | :--- | :--- |
| **`face_deepfake`** | Platt Logistic ($\sigma$) | Laplacian Spatial Variance + Inter-Frame Delta | Real Faces vs Deepfake Swap Set |
| **`voice_spoof`** | Platt Logistic ($\sigma$) | MFCC Delta Variance + Spectral Flatness | Genuine Speech vs Neural TTS Set |
| **`genai_doc`** | Platt Logistic ($\sigma$) | CLIP ViT-B/32 Probe + Radial FFT Spectrum | Camera Real Cards vs Midjourney/SD |
| **`crossfield`** | Deterministic Checksum | Verhoeff $D_5$ + UIDAI RSA-2048 PKCS#1 v1.5 | Cryptographic Non-Repudiation |

### 3. Asymmetric Anti-Dilution Risk Floors
Standard weighted averaging allows catastrophic attacks to be diluted by clean ancillary data. We enforce mathematically guaranteed **Hard Severity Floors**:

$$	ext{Final Risk Score} = \max\left(\sum_{i} 	ext{Penalty}(	ext{Signal}_i), igvee_{j \in 	ext{HARD}} 	ext{Floor}(	ext{Signal}_j)ight)$$

```python
# Hard Floor Guarantee: A deepfake face or forged QR instantly triggers FLAGGED
if any(sig.severity == Severity.HARD and sig.triggered for sig in signals):
    composite_risk = max(composite_risk, 70.0)
    decision_tier = RiskTier.FLAGGED
```

### 4. 2-of-3 Shamir's Secret Sharing (SSS) & Hash-Linked Ledger
To prevent insider tampering with forensic audit trails:
1. Every verification event is sealed with an append-only **HMAC-SHA256 block hash**:
   $$H_n = \operatorname{HMAC-SHA256}\left(H_{n-1} \parallel 	ext{Timestamp} \parallel 	ext{SessionID} \parallel 	ext{DecisionTier} \parallel 	ext{SignalVector}, 	ext{Pepper}ight)$$
2. The verification master $	ext{Pepper}$ is split into 3 polynomial shares using Shamir's Secret Sharing over $\mathbb{F}_{256}$:
   $$f(x) = S + a_1 x \pmod p$$
   Reconstructing the pepper to audit or verify the database requires a **2-of-3 quorum** of independent compliance officers, rendering historical records permanently immutable.

---

## 🔬 Multi-Modal Forensic Capabilities

| Vector | Forensic Channel | Detection Methodology | Attack Defeated |
| :--- | :--- | :--- | :--- |
| **Visual Deepfakes** | Facial Video Stream | Dual-Domain: Spatial Laplacian variance + inter-frame optical jitter tracking | DeepFaceLab, SimSwap, Roop, FaceFusion, Live Avatar Injection |
| **Audio Cloning** | Vocal Stream | Dual-Acoustic: MFCC Delta-Variance + Spectral Energy Flatness | ElevenLabs, Tortoise-TTS, VITS, Bark, Voice Conversions |
| **AI Documents** | Document Capture | Dual-Domain: Radial Fast Fourier Transform (FFT) + CLIP ViT-B/32 Linear Probe | Midjourney, Stable Diffusion, DALL-E synthesized ID documents |
| **Spliced Documents** | Document Capture | JPEG Discrete Cosine Transform (DCT) Error Level Analysis (ELA) + EXIF Metadata | Adobe Photoshop, GIMP copy-paste text box replacement |
| **Forged Aadhaar** | Document & QR | Dihedral $D_5$ Verhoeff Checksum + UIDAI RSA-2048 PKCS#1 v1.5 QR verification | Arbitrary fake UID numbers, counterfeit un-signed QR codes |
| **Session Replay** | Video + Audio | Dynamic 5-character randomized alphanumeric nonce + 3D Head Pose tracking ($<10	ext{s}$ TTL) | Pre-recorded video playback, virtual camera loopback streams |

---

## ⚔️ Interactive Red-Team Attack Simulator

AEGIS-SENTINEL includes an integrated **Red-Team Adversarial Testbench** (`/redteam.html`) allowing judges, compliance officers, and security teams to actively attack the live pipeline with real samples:

- **1-Click Live Adversarial Injections**:
  1. 👤 **Face Swap / Deepfake Face**: Injects high-frequency boundary synthetic frames into the liveness engine.
  2. 🎙️ **Cloned / Neural Voice Synthesis**: Injects neural TTS audio to evaluate acoustic vocoder detection.
  3. 📄 **AI-Generated ID Document**: Injects diffusion-synthesized identity cards testing radial FFT spectrum anomaly.
  4. ✂️ **Tampered Document (Spliced/EXIF)**: Injects Photoshop-manipulated documents with modified DCT error levels.
  5. 🔍 **Invalid Aadhaar QR / Checksum**: Injects forged cards with failing Verhoeff $D_5$ group checksums.
  6. ✅ **Genuine Clean Submission**: Injects pristine, authentic identity data to demonstrate calibrated high trust ($>95\%$).
- **Live Bayesian Trust Decay Visualization**: Animates the step-by-step decay of identity confidence in real-time as each forensic signal is ingested.
- **Direct Ledger Verification**: Click-through validation from attack results directly into the immutable cryptographic audit ledger.

---

## ⚖️ Regulatory Compliance & DPDP Act 2023

AEGIS-SENTINEL is built strictly around India's **Digital Personal Data Protection (DPDP) Act, 2023** and **RBI Master Direction on Digital KYC / V-CIP**:

1. **Zero Cloud PII Transmission**: 100% of facial landmarking, voice extraction, OCR, and cryptographic checks execute on-premise. No biometric data is sent across third-party networks.
2. **Automated Aadhaar UID Masking**: Complies with UIDAI circulars and Section 8 of the DPDP Act by masking the first 8 digits of extracted UIDs (`19XXXXXX5678`).
3. **Ephemeral In-Memory Processing**: Client-side MediaPipe FaceMesh isolates facial keyframes directly in browser memory (<200 KB). Uploaded binary buffers are wiped immediately post-inference.
4. **Data Minimization & Purpose Limitation**: Stores only cryptographic hashes, Platt calibration confidences, and model provenance metadata in `audit.db` — raw video recordings are never persisted.

---

## 🚀 Quickstart & Deployment

### 1. Clone & Set Up Environment
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
```

### 2. Launch Application
```bash
python -m uvicorn api.orchestrator:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access Web Portals
| Portal | URL | Purpose |
| :--- | :--- | :--- |
| **📱 Live KYC Verification** | `http://localhost:8000/index.html` | 4-Stage Live Onboarding (Doc OCR $ightarrow$ Dynamic Prompt $ightarrow$ Biometric Nonce $ightarrow$ Verdict) |
| **⚔️ Red-Team Attack Lab** | `http://localhost:8000/redteam.html` | 1-Click Adversarial Injection & Bayesian Trust Decay |
| **📊 Auditor Telemetry Dashboard** | `http://localhost:8000/dashboard.html` | Forensic telemetry, ELA heatmaps, model provenance, and audit chain |
| **📖 Swagger API Documentation** | `http://localhost:8000/docs` | Interactive OpenAPI endpoints |

---

## 🧪 Comprehensive Verification & Test Suite

The entire codebase is backed by **55 automated unit, integration, and regression tests** (`pytest tests/`) and end-to-end browser automation suites (`Playwright`):

```bash
# Run full test suite (55/55 passing)
pytest tests/ -v

# Run Red-Team API & response shape regression check
pytest tests/test_redteam_regression.py -v

# Run cryptographic audit chain & Shamir's Secret Sharing checks
python shared/audit_chain.py
python shared/pepper_sss.py
```

### Live Test Suite Proof
```
collected 55 items
tests/test_full_suite.py .................................................... [ 94%]
tests/test_orchestrator.py .                                                  [ 96%]
tests/test_redteam_regression.py ..                                           [100%]
============================== 55 passed in 69.84s ==============================
```

---

## 🏆 Hackathon Evaluation Matrix (Why Sentinel Wins)

| Evaluation Parameter | Typical Hackathon Project | AEGIS-SENTINEL Implementation |
| :--- | :--- | :--- |
| **Data Integrity** | Hardcoded mock numbers & `Math.random()` | **0% Mock Data** — 100% computed live by forensic pipeline |
| **Biometric Defenses** | Basic cosine face match (vulnerable to spoofing) | **Dual-Domain** (Laplacian spatial + temporal inter-frame jitter) |
| **Replay Defense** | Static "blink once" prompts | **Dynamic 5-character vocal nonce** + 3D FaceMesh head-pose tracking |
| **Aadhaar Integrity** | Regex length check | **Verhoeff $D_5$ Dihedral Group Checksum** + UIDAI RSA-2048 QR verification |
| **Decision Logic** | Naive arithmetic averaging | **Platt-scaled empirical probabilities** + Asymmetric Hard Risk Floors |
| **Security & Privacy** | Cloud API uploads; raw PII saved | **100% Local Inference**, DPDP Act UID masking, 2-of-3 Shamir SSS Audit Chain |
| **Judge Usability** | Static forms with text output | **Interactive Red-Team Attack Lab** + Real-Time Trust Decay Animation |

---

## 👥 Authors & Team
- **Team**: MUJ Asymptotes  
- **Hackathon**: HackMUJ 4.0  
- **Track**: Cybersecurity & Defence (PS#3: Deepfake & Synthetic Identity Detection)  
- **License**: MIT License
