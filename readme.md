# 🛡️ Asymptotes: Multi-Modal Deepfake & Synthetic Identity Detection Core

> **HackMUJ 4.0 Submission** | **PS#3: Deepfake & Synthetic Identity Detection**  
> **Theme:** Cybersecurity & Defence | **Deployment Model:** 100% On-Device / Local Web Application  
> **Compliance Standard:** DPDP Act 2023 (Digital Personal Data Protection Act, India)

---

## 📌 Executive Summary

Modern identity fraud in high-stakes KYC (banking onboarding, government subsidies, border security, and remote identity checks) has outpaced conventional security controls. Attackers no longer rely on simplistic image retouching; instead, they deploy:
- **Diffusion models & GANs** to fabricate synthetic identity documents.
- **Real-time face swapping & virtual camera injection** to bypass video liveness checks.
- **Neural voice cloning & TTS models** to defeat voice biometric systems.
- **Pre-recorded replay loops** to reuse previously authenticated sessions.

**Asymptotes** is an on-device, multi-modal verification and forensic analysis web platform engineered to detect, cross-validate, and fuse multiple attack vectors concurrently. Unlike conventional black-box verification systems that rely on opaque scores or cloud APIs, Asymptotes operates **entirely locally**, decomposes verification across visual, auditory, and document channels in parallel, calibrates raw suspicion into defensible empirical probabilities via **Platt scaling**, enforces **anti-dilution risk floors**, and records forensic evidence into a **Shamir-split, hash-chained tamper-evident audit ledger**.

---

## 🌟 Key Highlights & Architectural Differentiators

- **100% Local, Zero-Cloud Inference**: Runs entirely on-premise with zero runtime third-party API calls. Biometric data never leaves the host machine, eliminating data exfiltration risks and external vendor latency.
- **DPDP Act (2023) Privacy-by-Design**: Strict compliance with India's Digital Personal Data Protection Act — featuring automated Aadhaar UID masking (`XXXXXXXX1234`), on-device ephemeral memory handling, zero biometric data retention, and mathematically auditable access logs.
- **Dual-Domain Visual Forensics**: Evaluates both per-frame spatial boundary degradation (Laplacian variance) and inter-frame temporal warping to expose live face-swap injection tools in real time.
- **Aadhaar Authenticity & Backside QR PKI Verification**: Validates the 12-digit printed UID using the **Verhoeff D5 dihedral group checksum**, decodes the **Secure QR code on the backside of the Aadhaar card**, cryptographically verifies the **UIDAI RSA-2048 digital signature**, and cross-checks QR fields against printed OCR to catch copy-paste tampering.
- **Non-Dilutable Risk Fusion**: Catastrophic biometric failures (such as a detected deepfake face or cloned voice) trigger **anti-dilution floors** that instantly force the overall session into the **FLAGGED** tier ($\ge 70.0$), preventing clean peripheral paperwork from averaging away a severe attack.
- **Calibrated Probabilistic Confidence**: Replaces arbitrary magic numbers with empirical **Platt scaling** ($P(\text{attack}\mid\text{score})$), presenting defensible forensic evidence for compliance officers and auditors.
- **Dynamic Challenge-Response Anti-Replay**: Generates unpredictable physical actions (e.g., `turn_left`, `nod`, `smile`) and 5-character vocal nonces per session, defeating pre-recorded replay loops and generative puppetry.
- **Cryptographic Accountability (2-of-3 Shamir's Secret Sharing)**: Audit records are sealed into an append-only, peppered SHA-256 hash-chain protected by a 2-of-3 SSS scheme, requiring multi-party quorum to reconstruct and preventing unilateral tampering at rest.
- **Interactive Red-Team Attack Simulator**: Built-in interactive attack suite allowing judges and auditors to launch 6 live adversarial attack vectors with one click and observe real-time Bayesian trust decay.

---

## 🏗️ Architectural Workflow

```
                                 [ Client / Capture Session ]
                          (Front Document, Backside QR, Video, Audio)
                                              │
                                              ▼
                                   [ FastAPI Orchestrator ]
                                (api/orchestrator.py - Async)
                                              │
         ┌────────────────────────────────────┼────────────────────────────────────┐
         │                                    │                                    │
         ▼                                    ▼                                    ▼
┌────────────────────────────────┐   ┌────────────────────────────────┐   ┌────────────────────────────────┐
│       Document Forensics       │   │   Live Face Feed / Forensics   │   │        Audio Forensics         │
├────────────────────────────────┤   ├────────────────────────────────┤   ├────────────────────────────────┤
│ 1. Aadhaar Verhoeff Checksum   │   │ 1. Dual-Domain Visual Forensics│   │ 1. AASIST ONNX & MFCC Delta    │
│    (Dihedral D5 printed UID)   │   │    • Per-frame spatial blur    │   │    Vocoder anti-spoofing       │
│ 2. Backside Secure QR Validation│  │      (Laplacian variance)      │   │ 2. Whisper/Vosk Local ASR      │
│    • RSA-2048 UIDAI digital sig│   │    • Temporal jitter tracking  │   │    Dynamic spoken nonce & date │
│    • QR ↔ OCR copy-paste cross │   │      (Inter-frame warping)     │   │ 3. Cross-Modal Name Alignment  │
│ 3. GenAI Diffusion Detection   │   │ 2. Active Liveness Validation  │   │    Spoken name vs document OCR │
│    (CLIP ViT-B/32 & Radial FFT)│   │    Single-use random gestures  │   │ 4. Acoustic Spectral Flatness  │
│ 4. JPEG ELA & EXIF Provenance  │   │ 3. Local Video Evaluation      │   │    Synthetic voice detection   │
│    Recompression error mapping │   │    Pre-recorded benchmark store│   │                                │
└───────────────┬────────────────┘   └───────────────┬────────────────┘   └───────────────┬────────────────┘
                │                                    │                                    │
                └────────────────────────────────────┼────────────────────────────────────┘
                                                     │
                                                     ▼
                                    [ Unified SignalResult Contracts ]
                                      (shared/contracts.py - Frozen)
                                                     │
                                                     ▼
                                     [ Platt Scaling Calibration ]
                                     (modules/decision/calibration)
                                                     │
                                                     ▼
                                       [ Multi-Modal Risk Fusion ]
                                      (Anti-Dilution Risk Floors)
                                                     │
                   ┌─────────────────────────────────┴─────────────────────────────────┐
                   ▼                                                                   ▼
     [ Deterministic Officer Summary ]                                   [ Peppered SHA-256 Audit Chain ]
   (Narrative Explanation & Web UI View)                                  (2-of-3 Shamir's Secret Share)
```

---

## 🔬 Core Forensic Pillars

### 1. Dual Domain Visual Forensics (Spatial Blur + Temporal Jitter)

* **Beyond Static Frame Analysis**: Standard deepfake detectors evaluate frames independently and often miss dynamic inconsistencies. Our visual deepfake engine (`modules/face/deepfake_detector.py`) combines per-frame spatial boundary blur (via Laplacian variance) with frame-to-frame temporal warping and motion jitter tracking.
* **Exposes Live Injection Tools**: Real-time face swapping software (e.g., DeepFaceLab, SimSwap, or virtual camera streams) struggles to maintain high-frequency boundary sharpening across movements. Our dual spatial-temporal approach detects boundary warping and edge degradation during live video streams.
* **Local Video Storage & Replay Defense**: Pre-recorded videos are stored locally in the verification test bench (`data/test/`, `data/real_faces/`, `data/deepfake_faces/`). This enables zero-latency offline testing, reproducible evaluation benchmarks, and immediate detection of pre-recorded replay injection attacks when compared against single-use session nonces.
* **Unpredictable Challenge-Response**: Prompts the user to perform randomized physical gestures (`turn_left`, `turn_right`, `nod`, `smile`) tracked via 3D facial landmarks within an isolated temporal window.

---

### 2. Document Forensics & Aadhaar Cryptographic Verification

* **Aadhaar Verhoeff Checksum**: Every 12-digit Aadhaar UID contains a check digit calculated using the Verhoeff algorithm over the dihedral group $D_5$. Our validator (`modules/aadhaar/verhoeff.py`) validates the printed UID directly, instantly flagging synthetic, typo-riddled, or arbitrarily hallucinated Aadhaar numbers.
* **Backside Secure QR Cryptographic Validation**: Rather than treating document images as passive pictures, Asymptotes decodes the high-density Secure QR code located on the **backside of the Aadhaar card**:
  - **UIDAI RSA-2048 PKCS#1 v1.5 Signature Verification**: The decompressed QR payload is cryptographically authenticated against UIDAI's official public certificate (`shared/certs/uidai_offline_pub.cer`). Any altered byte or forged QR invalidates the cryptographic signature.
  - **QR ↔ OCR Printed Field Cross-Check**: Decoded demographic data (Name, UID, DOB, Gender) is cross-checked against OCR text extracted from the front of the card using Levenshtein distance matching (`modules/aadhaar/consistency.py`), immediately catching copy-paste tampering and photo-replacement attacks.
* **GenAI Diffusion & GAN Detection (`genai_doc`)**: Uses a CLIP ViT-B/32 linear probe paired with a radial Fast Fourier Transform (FFT) fallback to catch high-frequency checkerboard grid patterns and spectral decay characteristics inherent to diffusion and GAN generation.
* **Error Level Analysis (ELA) & EXIF**: Performs JPEG DCT recompression error mapping to isolate digitally spliced text boxes or stamps, complemented by EXIF camera sensor provenance audits.

---

### 3. Audio Forensics & Cross-Modal Nonce Binding

* **Voice Anti-Spoofing (`voice_spoof`)**: Deploys an AASIST ONNX model backed by MFCC delta-variance analysis and spectral flatness heuristics to detect vocoder artifacts, text-to-speech (TTS) synthesis (ElevenLabs, Bark, VITS), and voice cloning models.
* **Local Whisper / Vosk ASR Challenge**: Transcribes user audio completely offline using faster-whisper/Vosk to verify that the subject spoke their name, today's date, and a single-use 5-character nonce.
* **Cross-Modal Identity Alignment**: Matches the transcribed spoken name against the OCR-extracted document name using fuzzy token matching, ensuring modal coherence between audio and document channels.

---

### 4. Calibrated Decision Engine & Anti-Dilution Risk Floors

* **Empirical Platt Scaling**: Converts raw model outputs into calibrated posterior probabilities ($P(\text{attack}\mid\text{score})$) using pre-computed logistic parameters (`calibration.json`), producing defensible statistical metrics rather than arbitrary heuristics.
* **Anti-Dilution Hard Floors**: Standard weighted averages allow an attacker with clean peripheral paperwork to mask a severe attack. In Asymptotes:
  - Critical signals (`face_deepfake`, `voice_spoof`, `crossfield`) are categorized as **HARD** signals.
  - If any HARD signal triggers, the composite session score is mathematically floored at **$\ge 70.0$ (FLAGGED)**. A deepfake face or forged Aadhaar signature cannot be averaged away.

```python
# Anti-dilution floor guarantee
if any(signal.severity == Severity.HARD and signal.triggered for signal in results):
    composite_score = max(composite_score, 70.0)
    decision_tier = RiskTier.FLAGGED
```

---

### 5. Cryptographic Audit Trail & 2-of-3 Shamir's Secret Sharing

* **Tamper-Evident Hash Chain**: Every verification session is committed to an append-only SQLite hash ledger (`audit.db`). Each block contains:
  $$\text{Hash}_n = \text{SHA-256}(\text{Hash}_{n-1} \parallel \text{Timestamp} \parallel \text{Payload} \parallel \text{Pepper})$$
* **Shamir's 2-of-3 Secret Sharing (SSS)**: The HMAC secret pepper is split into three polynomial shares distributed among distinct compliance officers. Reconstructing the pepper to audit or verify ledger integrity requires at least 2 authorized officers, preventing rogue administrators from unilaterally forging historical logs.

---

## 🔒 Compliance & Privacy-by-Design: DPDP Act (2023)

Asymptotes is architected from the ground up to comply with India's **Digital Personal Data Protection (DPDP) Act, 2023**:

1. **100% On-Device / Local Inference**: Biometric face streams, voice recordings, and identity card scans are processed strictly within the local host environment. No PII is ever transmitted to third-party clouds or external APIs.
2. **Aadhaar UID Masking**: In compliance with UIDAI regulations and Section 8 of the DPDP Act, the first 8 digits of extracted 12-digit Aadhaar numbers are automatically masked (`XXXXXXXX1234`). Only the last four digits are preserved for consistency cross-checking.
3. **Ephemeral Processing & Zero PII Retention**: Client-side face extraction (via MediaPipe) isolates cropped facial keyframes and compressed Opus audio in-browser (<200 KB total). Uploaded media files are processed in ephemeral memory and discarded immediately after score calculation.
4. **Purpose Limitation & Data Minimization**: The platform extracts only the minimal cryptographic features (frequency histograms, landmark ratios, spectral variances) required to determine liveness and authenticity.
5. **Auditable Integrity**: The SSS-protected hash-chained audit ledger guarantees tamper-evident logging of all verification outcomes without storing raw biometric video feeds at rest.

---

## 🖥️ Web Applications & Interactive Interfaces

Asymptotes provides three web portals built with vanilla CSS and dynamic JavaScript:

| Portal | URL | Purpose |
| --- | --- | --- |
| **KYC Verification Portal** | `http://localhost:8000/` or `/static/upload.html` | End-to-end KYC session capture: dual-photo Aadhaar front + backside QR, live webcam feed, dynamic gesture challenge, microphone recording, and instant verdict. |
| **Forensic Auditor Dashboard** | `http://localhost:8000/static/dashboard.html` | 3-column real-time forensic workspace displaying multi-modal score breakdowns, per-frame sparkline visualizations, Platt probabilities, and the Shamir-protected audit chain. |
| **Red-Team Attack Simulator** | `http://localhost:8000/static/redteam.html` | Interactive attack suite allowing judges to launch 6 adversarial vectors (Replay, Deepfake Face Swap, Voice Cloning, GenAI Document, UID Checksum Tampering, QR Signature Forgery) and watch real-time Bayesian trust decay. |

---

## 🚀 Quickstart & Running the WebApp

### 1. Installation

```bash
# Clone repository
git clone https://github.com/shashwatkhandelwal-debug/Muj_Asymptotes.git
cd Muj_Asymptotes

# Initialize virtual environment
python -m venv venv
# Activate virtual environment (e.g., source venv/bin/activate or venv\Scripts\activate)

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Application Server

```bash
# Start the FastAPI orchestrator
uvicorn api.orchestrator:app --reload --port 8000
```

Once running, access the web portals directly:
- **Interactive KYC Portal**: `http://localhost:8000/`
- **Forensic Auditor Dashboard**: `http://localhost:8000/static/dashboard.html`
- **Red-Team Simulator**: `http://localhost:8000/static/redteam.html`
- **Interactive Swagger API Docs**: `http://localhost:8000/docs`

### 3. Run Self-Tests & Verifications

```bash
# Execute the full integration smoke test
python smoke_test.py

# Verify the cryptographic hash chain and Shamir's Secret Sharing
python shared/audit_chain.py
python shared/pepper_sss.py
```

---

## 📋 Contract Specification & API Reference

All forensic detectors communicate via the frozen contract in `shared/contracts.py`:

```python
class Severity(str, Enum):
    SOFT = "soft"   # Weighted additive risk contributor
    HARD = "hard"   # Immediate tier escalation floor; cannot be diluted

class RiskTier(str, Enum):
    CLEAR   = "clear"    # Score 0 - 30: Verified authentic
    REVIEW  = "review"   # Score 31 - 69: Borderline anomalies, requires human review
    FLAGGED = "flagged"  # Score 70 - 100: Fraud detected or hard floor engaged
```

### Key API Endpoints

- `POST /api/verify`: Multi-modal verification accepting `document_image`, optional `qr_image` (backside QR close-up), `video`, and `audio`.
- `GET /api/challenge`: Generates a dynamic single-use session challenge (physical action + 5-character vocal nonce + timestamp).
- `POST /api/redteam/run`: Executes pre-configured adversarial test scenarios with live detector responses.
- `GET /api/audit`: Retrieves the latest entries from the tamper-evident hash-chained audit ledger.

---

## ⚖️ Hackathon Evaluation Checklist (Why Asymptotes Wins)

| Evaluation Criteria | Asymptotes Implementation |
| --- | --- |
| **Real-Time Defense** | Sub-3 second multi-modal execution via asynchronous parallel thread pooling. |
| **Multi-Modal Rigor** | Visual (Laplacian + temporal), Auditory (AASIST + ASR), and Document (Verhoeff + RSA-2048 QR + CLIP) channels. |
| **Adversarial Resilience** | Anti-dilution risk floors prevent clean signals from masking catastrophic biometric attacks. |
| **Zero Data Leakage** | 100% on-device local execution; fully compliant with India's DPDP Act 2023. |
| **Legal & Audit Defense** | Platt-calibrated empirical probabilities and 2-of-3 Shamir's Secret Sharing hash-chain. |
| **Judge Usability** | Interactive 1-click Red-Team Attack Simulator and responsive forensic auditor dashboard. |

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.