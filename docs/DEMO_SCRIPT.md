# 90-Second Judge Presentation & Demo Script

This script is timed for a 90-second checkpoint presentation at HackMUJ 4.0. Follow these exact spoken sentences and physical actions.

---

### [0:00 – 0:15] The Hook & The Problem

**Say:**
> "Generative AI and real-time deepfakes have broken traditional identity checkpoints because synthetic documents bypass ELA recompression checks, and injected face swaps easily fool static selfies. Our platform, Asymptotes, combines cryptographically chained hardware-independent active challenges, multi-modal forensics, and mathematical anti-dilution risk scoring to detect synthetic identities on-device in under three seconds."

---

### [0:15 – 0:45] Live Demonstration: Genuine Subject (Clear Tier)

**Action:**
1. Click **`[DEMO: CLEAR]`** (or upload the genuine identity verification sample).
2. The dashboard updates instantaneously with a green score bar.

**Say:**
> "Here, a genuine applicant presents their government ID and completes a randomized single-pass active liveness challenge — in this round, turning left while speaking their name, today's date, and a one-time four-digit nonce. All five forensic detectors run in parallel: document diffusion analysis confirms natural frequency decay, the whisper transcriber confirms the spoken name matches the OCR record, and voice anti-spoofing confirms biological vocal tract acoustics. The fused risk score is 8 out of 100, landing cleanly in the green 'CLEAR' tier."

---

### [0:45 – 1:15] Live Demonstration: The Attack (Anti-Dilution Floor)

**Action:**
1. Click **`[LOAD FIXTURE]`** (or submit the live deepfake video capture attack).
2. The dashboard flashes red, score leaps to 82, and the red **`HARD`** floor badge illuminates.

**Say:**
> "Now watch what happens during a sophisticated deepfake presentation attack where an attacker attempts to dilute a face swap with a clean, stolen document. Notice that even though the document forensics and passive checks passed, our frequency-domain texture classifier caught deepfake artifacts across the video frames with 91% confidence. Because face deepfakes and voice clones are classified as HARD signals, our anti-dilution security floor automatically forces the risk score up to 70 and locks the verdict into 'FLAGGED' — mathematically guaranteeing that a catastrophic biometric violation cannot be averaged away by clean paperwork."

---

### [1:15 – 1:30] The Defense & Audit Chain

**Action:**
1. Point to the audit log status badge at the bottom of the dashboard.

**Say:**
> "Finally, every verification event is committed to a hash-chained SQLite audit ledger peppered with an external key split two-of-three across officers using Shamir's Secret Sharing: tampering with any past entry immediately breaks the chain, and no single rogue officer can forge or re-chain the evidence alone."

---

### Quick Reference for Q&A

- **Q: Why not use cloud APIs for inference?**
  *A: Border checkpoints and defense installations cannot rely on external connectivity or leak biometric PII to third-party endpoints. All inference runs strictly on-device.*
- **Q: What happens if a detector crashes?**
  *A: The system fails open on individual detector errors to avoid DoS denial of legitimate traffic, but records the crash in the immutable audit log for manual review.*
- **Q: How does the cross-modal challenge work?**
  *A: We bind the physical video action, ASR transcription of the document name, and voice anti-spoofing into a single 5-second capture, preventing pre-recorded replay attacks.*
