# HackMUJ 4.0 — 30-Minute Pre-Judging Demo Rehearsal Checklist

> **Problem Statement #3:** Deepfake and Synthetic Identity Detection  
> **Theme:** Cybersecurity & Defence  
> **Team:** Asymptotes (VAJRA-SHIELD)  
> **Target Demo Duration:** 60 to 90 seconds max

---

## 1. Browser Tabs to Have Open (In Exact Order)

Open these tabs in a clean Chrome profile before judges arrive:

1. **Tab 1: Live Capture**  
   `http://localhost:8080/capture.html`  
   *Pre-grant Camera & Microphone permissions ahead of time so no prompt interrupts the demo flow.*
2. **Tab 2: Defense Dashboard**  
   `http://localhost:8080/dashboard.html`  
   *Pre-loaded with `verify_sample_clear.json` (Green / Clear state).*
3. **Tab 3: Offline Fixture Backup (Fail-Safe Tab)**  
   `http://localhost:8080/dashboard.html`  
   *Ready to trigger `[LOAD FIXTURE]` (Flagged attack state).*
4. **Tab 4: Tamper-Evident Audit Chain Log**  
   Terminal or viewer displaying the hash chain verification (`shared/audit_chain.py`).

---

## 2. Preloaded State
- Ensure Tab 2 has the **CLEAR** fixture pre-loaded:
  - Score reads **8 / 100 (CLEAR)** in emerald green.
  - Zero floors applied.
  - All biometric channels show PASS / SAFE.
- Keep the room lighting check completed so the brightness meter reads green (> 60 LUX).

---

## 3. Exact Click Sequence: Clear Demo (0:15 – 0:45)

1. Switch to **Tab 2 (Dashboard)**.
2. Point at the emerald green score **8 / 100 (CLEAR)**.
3. Highlight:
   - "All 5 identity channels verified: Document OCR, ArcFace 1:1, spoken name matching, and voice acoustics."
   - "Deterministic baseline: zero false positive penalties on genuine subjects."
4. If doing a live capture:
   - Click `[Live Capture]` (Tab 1).
   - Click `[START CAPTURE]`.
   - Subject turns head left, speaks: `"Priya Sharma 2026-09-11 4471"`.
   - The 5-second countdown finishes, face crops (<200KB payload) are sent, auto-redirects to Tab 2.
   - Dashboard flashes green, verdict remains **CLEAR**.

---

## 4. Exact Click Sequence: Flagged Demo (0:45 – 1:15)

1. Click the red button `[LOAD FIXTURE]` on top right (or run the deepfake attack capture).
2. The dashboard flashes red with pulse animation:
   - **Score shoots to 82 / 100 (FLAGGED)**.
   - Point immediately to the **ANTI-DILUTION FLOORS** panel:
     - **Red Pill:** `face_deepfake → FLAGGED`.
     - *Key Pitch Line:* "Even if an attacker passes document OCR and spoken name matching, our anti-dilution floor ensures a HARD forensic hit cannot be averaged or diluted away."
   - Point to the **Face Deepfake panel**:
     - Red `HARD` badge, `91% confidence`.
     - 5-bar sparkline showing per-frame artifact spikes above 0.70.
   - Point to the **Voice Spoof panel**:
     - Grey `HARD` label with green `SAFE` status, showing that innocent channels are never unfairly triggered.
   - Point to the **Officer Summary**:
     - "Final verdict: FLAGGED (score 82.0). Anti-dilution floor applied: face_deepfake."

---

## 5. Troubleshooting & Contingency Protocols

### A. Camera / Mic permission prompt does not appear
- **Cause:** Chrome permissions blocked or cached.
- **Fix:** Click the lock icon in the Chrome URL bar -> Site settings -> explicitly set Camera & Microphone to "Allow".
- **Instant Fallback:** Do not spend judge time debugging hardware. Immediately pivot:
  > *"We have the full live client pipeline running, but let us jump straight to the real-time forensic verdict station."*
  Click `[DEMO: CLEAR]` then `[LOAD FIXTURE]`.

### B. Venue Wi-Fi drops or is completely offline
- **Status:** **Zero impact.**
- **Architecture Guarantee:** All inference, web serving, and client assets run 100% on-device on localhost (`http://localhost:8080`).
- Tailwind CSS includes fallback offline `<style>` blocks for all defense colors and typography.
- MediaPipe falls back automatically to local canvas frame crop.
- Never connect to any external cloud API during judging.

### C. Live backend pipeline busy or restarting
- Click `[REFRESH]`. If backend is restarting, the frontend automatically falls back to `verify_sample.json`.
- Both `[LOAD FIXTURE]` and `[DEMO: CLEAR]` work fully offline without any network calls.

---

## 6. Speaker Division (Who Says What, In What Order)

### Speaker 1: Yash (0:00 – 0:25) — Problem & Core Architecture
> *"Good morning judges. Current identity systems have a fatal blind spot: generative diffusion documents and deepfake video bypass traditional OCR and passive liveness checks.*  
> *We built **VAJRA-SHIELD**, an edge-first defense station that detects deepfakes, synthetic speech, and GenAI documents in under 2.5 seconds with zero cloud reliance."*

### Speaker 2: Teammate AI/ML (0:25 – 0:55) — Forensic Channels & Fusion
> *"Notice the multi-modal defense here: we evaluate radial FFT and CLIP linear probes for diffusion artifacts, per-frame facial warping, and AASIST voice anti-spoofing.*  
> *Notice our anti-dilution floor in action: when a deepfake attack occurs, a HARD hit locks the verdict to FLAGGED. In a military or immigration checkpoint, an attacker cannot dilute high-confidence deepfakes by providing clean documents."*

### Speaker 3: Teammate Frontend (0:55 – 1:25) — Edge Performance & Officer Audit Trail
> *"For low-bandwidth field deployment, we extract face crops client-side at under 200KB per round, instead of streaming megabytes of video.*  
> *Every frame score is calibrated via Platt scaling, and the entire audit trail is peppered with SHA-256 hash chaining split via Shamir's Secret Sharing—so no single compromised officer can rewrite past verdicts."*

### Wrap-Up (1:25 – 1:30)
> *"Questions?"*
