# Known Limitations

## Generalisation
When tested against out-of-distribution uniform noise imagery, the genai detector flagged it perfectly (1.0 raw score). However, real-world generalisation against novel diffusion models not represented in the training set may vary.

## Temporal analysis
The deepfake detector scores frames independently and aggregates the three most suspicious. It does not model temporal inconsistency across frames, which is where several published detectors get their strongest signal. This was a CPU budget decision, not an oversight.

## Calibration sample size
The Platt calibration fit used 40 document images, 30 face videos, and 30 voice samples. While sufficient for the demo, this sample size means the confidence values are indicative rather than statistically robust.

## Voice anti-spoofing scope
The current synthetic voice samples were generated solely with `pyttsx3`. A cloned voice from an advanced TTS engine not represented in the calibration set may score lower than it should.

## Active liveness
During adversarial testing, a printed photo replay attack resulted in a low raw score (0.40) and failed to trigger the flag, indicating that the heuristic fallback may be vulnerable to simple static image replays.

## Audit chain
Tamper-evidence holds at rest. While the server process is running, the pepper is in memory and a root-level attacker could extract it. Full protection would need an HSM or enclave.
