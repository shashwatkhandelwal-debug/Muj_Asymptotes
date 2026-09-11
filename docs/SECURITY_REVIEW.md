# Security Review: Decision Fusion, Audit Chain, and Tamper Resistance

This document reviews the security architecture and implementation of the HackMUJ 4.0 detection layer, focusing on decision fusion, anti-dilution floors, fail-open vs. fail-closed behaviors, Shamir-peppered audit chaining, and officer summary grounding.

---

### 1. Can a triggered HARD signal be diluted?

**No.** The fusion engine enforces strict, monotone anti-dilution floors after calculating weighted penalties.

In `modules/decision/fusion.py`:
```python
# 2. Raw total
raw_total = float(min(100.0, max(0.0, existing_score + total_penalty)))

# 3. Floors (apply AFTER summation, monotone — can only increase score)
for signal in new_signals:
    if signal.ok and signal.triggered and signal.signal in HARD_SIGNALS:
        if raw_total < 70:
            raw_total = 70.0
            if signal.signal.value not in floors_applied:
                floors_applied.append(signal.signal.value)
```

**Concrete example:**
Suppose a subject provides genuine document scans and clean biometric captures across all soft channels (`existing_score = 5.0`). However, a live video deepfake is detected on `FACE_DEEPFAKE` with `raw_score = 0.95`.
- `FACE_DEEPFAKE` penalty: $0.95 \times 25.0 = 23.75$.
- Unmitigated raw total: $5.0 + 23.75 = 28.75$.
- Under linear summation alone, $28.75 \le 30.0$, which falls into the `"clear"` tier (green). An attacker would have successfully bypassed the checkpoint by having clean credentials elsewhere.
- **With anti-dilution floor applied:** Because `FACE_DEEPFAKE` is in `HARD_SIGNALS` and triggered, the engine forces `raw_total = 70.0` and appends `"face_deepfake"` to `floors_applied`. The verdict immediately shifts to `"flagged"` (red). A hard violation cannot be averaged away by clean soft signals.

---

### 2. Can a crashed detector be used as an attack?

**Under the current design, an unhandled crash triggers fail-open behavior (`ok=False`).**

In `modules/decision/fusion.py`:
```python
for signal in new_signals:
    if not signal.ok:
        # fail-open: skip crashed detector
        continue
```

If an attacker crafts a malicious input (such as an oversized payload, malformed container, or memory exhaustion exploit) that causes `detect_face_deepfake` to crash, the detector catches the exception and returns `SignalResult(ok=False)`. Fusion ignores that detector, preventing a pipeline denial-of-service (DoS) from halting all border/checkpoint operations.

**Attack surface & mitigations:**
- *Risk:* A sophisticated adversary might intentionally trigger an unhandled error in the deepfake detector to evade penalties.
- *Remediation with additional time:* We log all detector crashes prominently in the audit log. In a production deployment, if `sum(1 for s in new_signals if not s.ok) > 0`, the system should flag the session for manual officer inspection (`review` tier) rather than remaining silently in `clear`.

---

### 3. Does the audit chain genuinely prevent single-party tampering?

**Yes, for database tampering at rest.**

In `shared/audit_chain.py` and `shared/pepper_sss.py`:
- Each row entry computes:
  $$\text{entry\_hash} = \text{SHA256}(\text{pepper} \parallel \text{prev\_hash} \parallel \text{canonical\_json}(\text{payload}))$$
- An attacker modifying row 5 causes row 5's stored hash to mismatch the recomputed hash, and row 6's `prev_hash` to mismatch row 5's hash.
- To repair the chain from row 5 onward, the attacker must recompute the cryptographic hash for row 5 and all subsequent rows. This requires the 32-byte `pepper`.
- The `pepper` is never written to the SQLite database or disk. It is divided using a $(k=2, n=3)$ Shamir's Secret Sharing scheme across designated officers. A single rogue officer holding only one share cannot reconstruct the pepper ($k \ge 2$ required).

**Residual architectural weakness:**
While the server process is actively executing, `_PEPPER` resides in host process memory. A compromised root account or memory dumper on the live host could read the pepper. Complete protection against host-level root compromise requires storing the pepper inside a hardware security module (HSM) or trusted execution environment (TEE/enclave).

---

### 4. Is the officer summary capable of altering the verdict?

**No. The officer summary is strictly unidirectional and read-only.**

In `api/orchestrator.py`:
```python
def build_officer_summary(fused: dict) -> str:
    parts = [f"Final verdict: {fused['tier'].upper()} (score {fused['score']})."]
    for line in fused.get("lines", []):
        if line.get("triggered"):
            conf = line.get("confidence", 0.0)
            parts.append(f"{line['signal']}: {line['label']} (confidence {conf:.2f}).")
    if fused.get("floors_applied"):
        parts.append("Anti-dilution floor applied: " + ", ".join(fused["floors_applied"]) + ".")
    return " ".join(parts)
```
The summary generation routine consumes the already-computed `fused` dictionary. The returned summary string is included in the JSON response purely for human officer readability. The verdict (`score`, `tier`, `floors_applied`) is calculated deterministically by `fuse()` and is never re-evaluated or altered by text summaries or LLMs.
