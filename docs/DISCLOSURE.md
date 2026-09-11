# Disclosure — Prior Work

Reuse of prior competition work has been explicitly confirmed as permitted by
the HackMUJ 4.0 organizing team.

## Carried forward (SIH-VAJRA / SIH26188)
Multi-document OCR, UIDAI RSA-2048 signature verification, Verhoeff and ICAO MRZ
checksums, cross-field consistency, ELA and EXIF forensics, ArcFace 1:1 biometric
matching, passive liveness, FAISS watchlist search, deterministic risk scoring,
LLM officer summaries, SQLite audit logging.

This is a validated, tested foundation, and we present it as such.

## Built for HackMUJ 4.0
- **A** Generative-forgery document detection, closing a blind spot our own prior
  documentation identified: diffusion and GAN forgeries produce no JPEG
  recompression artifacts and were invisible to ELA.
- **B** Face deepfake detection across live video capture
- **C** Active liveness plus cross-modal identity challenge: randomised physical
  action, spoken-name-to-OCR-name matching, and voice anti-spoofing, all from a
  single capture
- **D** Peppered, hash-chained, Shamir-split audit log preventing single-party tampering
- **E** Fusion with anti-dilution floors
- **F** Robustness harness against compression, low light, and low bandwidth
- **G** Confidence calibration
