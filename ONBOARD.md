# ONBOARD.md — PASTE THIS ENTIRE FILE INTO ANTIGRAVITY AFTER `git pull`

# HOW TO USE THIS FILE
#
# 1. git pull
# 2. Open Antigravity in the repo root
# 3. Paste this ENTIRE file as one message
# 4. Add ONE line at the very end saying who you are:
#       "I am Yash, architecture and security"
#       "I am the teammate doing AI/ML"
#       "I am the teammate doing frontend"
# 5. Antigravity reads your role, jumps to your section, and runs.
#
# Antigravity: read the role line at the end of this message FIRST, then execute
# ONLY that role's section. Ignore the other two sections entirely.

---

# PROJECT CONTEXT (all roles read this)

**What this is:** HackMUJ 4.0 submission, PS#3 — Deepfake and Synthetic Identity
Detection. Theme: Cybersecurity & Defence.

**What already exists and works:** A document-authentication and biometric core
carried over from a prior Smart India Hackathon submission (SIH-VAJRA / SIH26188).
Reuse has been explicitly confirmed as permitted by the HackMUJ organizing team.
That core handles: multi-document OCR, UIDAI RSA-2048 signature verification,
Verhoeff and ICAO MRZ checksums, cross-field consistency, ELA forensics, EXIF
analysis, ArcFace 1:1 face matching, passive liveness, FAISS watchlist search,
deterministic risk scoring, LLM officer summaries, and SQLite audit logging.

**What we built for HackMUJ (12 files, all passing acceptance tests):**

| File | Component | What it does |
|---|---|---|
| `shared/contracts.py` | — | FROZEN. The SignalResult contract every detector returns. |
| `shared/audit_chain.py` | D | Peppered SHA-256 hash-chained audit log |
| `shared/pepper_sss.py` | D | Shamir 2-of-3 secret sharing for the audit pepper |
| `modules/decision/calibration.py` | G | Platt scaling — raw scores become defensible confidence |
| `modules/forensics/genai_detector.py` | A | Detects diffusion/GAN-generated documents |
| `modules/face/deepfake_detector.py` | B | Detects deepfake face artifacts on video |
| `modules/face/active_liveness.py` | C-i | Verifies the random physical action was performed |
| `modules/audio/asr.py` | C-ii | Cross-modal: spoken name must match OCR name |
| `modules/audio/antispoof.py` | C-iii | Detects TTS/cloned/synthetic voice |
| `modules/face/challenge.py` | C | Issues challenge, decodes frames once, fans out |
| `modules/decision/fusion.py` | E | Anti-dilution floors — a HARD hit cannot be averaged away |
| `tests/robustness/harness.py` | F | Degradation testing: compression, darkness, low bandwidth |
| `tests/adversarial/selftest.py` | J | Deliberate attacks on our own detectors |

**Architecture rules that are NOT negotiable:**

1. `shared/contracts.py` is FROZEN. Nobody modifies it. Not for any reason.
2. Every detector returns exactly one `SignalResult`. No other return type.
3. All inference is on-device. Zero cloud API calls at runtime. Ever.
4. Colab GPU is for weight EXPORT only, never runtime inference.
5. Weights and thresholds live only in `contracts.py`. Never hardcode them inside a detector.
6. A detector that crashes returns `ok=False`, it never raises to the caller.
7. Every `SignalResult` must have `ms` set (wall time in milliseconds).

**The frozen contract, for reference:**

```python
class Severity(str, Enum):
    SOFT = "soft"   # weighted penalty only
    HARD = "hard"   # a hit forces the tier up, cannot be diluted

class SignalId(str, Enum):
    ELA = "ela"; EXIF = "exif"; CROSSFIELD = "crossfield"
    FACE_MATCH = "face_match"; PASSIVE_LIVENESS = "passive_liveness"
    WATCHLIST = "watchlist"
    GENAI_DOC = "genai_doc"              # A
    FACE_DEEPFAKE = "face_deepfake"      # B
    ACTIVE_LIVENESS = "active_liveness"  # C-i
    NAME_MATCH = "name_match"            # C-ii
    VOICE_SPOOF = "voice_spoof"          # C-iii

@dataclass
class SignalResult:
    signal: SignalId
    raw_score: float          # 0..1, higher = more suspicious
    confidence: float         # 0..1 calibrated
    severity: Severity = Severity.SOFT
    triggered: bool = False
    penalty: Optional[float] = None
    label: str = ""
    evidence: dict = field(default_factory=dict)
    ok: bool = True
    ms: float = 0.0
    ts: float = field(default_factory=time.time)

SIGNAL_WEIGHTS = {GENAI_DOC: 25.0, FACE_DEEPFAKE: 25.0, VOICE_SPOOF: 20.0,
                  NAME_MATCH: 15.0, ACTIVE_LIVENESS: 15.0}
SIGNAL_TRIGGER = {GENAI_DOC: 0.60, FACE_DEEPFAKE: 0.60, VOICE_SPOOF: 0.55,
                  NAME_MATCH: 0.50, ACTIVE_LIVENESS: 0.50}
HARD_SIGNALS = {FACE_DEEPFAKE, VOICE_SPOOF}
TIERS = [(0, 30, "clear"), (31, 69, "review"), (70, 100, "flagged")]
```

---

# ANTIGRAVITY OPERATING INSTRUCTIONS (all roles)

**Work autonomously by default.** Do not ask permission to write code, install a
package, create a file, or run a test. Just do it and report.

**Only stop and ask the human when one of these is true:**
- You need a physical-world action only they can take (record audio, point a camera, plug in a device)
- You need a credential, API key, or account login
- You need a decision that changes the demo strategy, not just the code
- A test fails twice in a row after you have already tried a fix
- You are about to delete or overwrite work that is not yours

**When you do need input, format it exactly like this:**

```
⏸  INPUT NEEDED
    What I need:   <one line>
    Why:           <one line>
    Options:       <if there are choices, list them>
    What I'll do:  <what happens after they answer>
```

Then stop and wait. Do not proceed with a guess.

**After each completed step, report like this:**

```
✅ STEP <n> DONE — <file or task name>
    Result:    <what the test printed>
    Next:      <what you are doing now>
```

**Never do these:**
- Never modify `shared/contracts.py`
- Never modify files owned by another role (the ownership table is in each section)
- Never `git push` without being told explicitly
- Never skip an acceptance test because it looks like it will pass
- Never mark a step done if its test did not print its OK string

---
---
---

# ═══════════════════════════════════════════════════════════
# ROLE A — "I am Yash, architecture and security"
# ═══════════════════════════════════════════════════════════

## Your ownership

You own and may modify:
- `api/orchestrator.py`
- `shared/audit_chain.py`, `shared/pepper_sss.py`
- `modules/decision/fusion.py`
- `modules/face/challenge.py`, `modules/face/active_liveness.py`
- `docs/DISCLOSURE.md`, `docs/DEMO_SCRIPT.md`
- `shared/preprocessing.py` (you will create this)

You do NOT touch: any file in `modules/forensics/`, `modules/audio/`,
`tests/`, or `frontend/`. Those belong to the other two.

## Your steps, in order

### STEP 1 — Verify the environment is sane

Run:
```bash
python smoke_test.py
python modules/decision/fusion.py
python shared/audit_chain.py
python shared/pepper_sss.py
```

All four must print their OK strings. If any fails, fix it before continuing.
Report what each printed.

---

### STEP 2 — Create `shared/preprocessing.py`

This does not exist yet. Every detector that touches an image or frame needs
CLAHE normalisation so low-light captures do not silently destroy accuracy.

Create the file with exactly this content:

```python
"""
shared/preprocessing.py
Shared image/frame preprocessing. Applied before ANY detector sees a frame.
CLAHE (Contrast Limited Adaptive Histogram Equalization) normalises exposure so
a dim checkpoint or a bright window does not change detector behaviour.
"""
import numpy as np

def clahe_rgb(frame_rgb: np.ndarray) -> np.ndarray:
    """
    Normalise exposure on an RGB uint8 frame using CLAHE on the L channel.
    Returns a new RGB uint8 array. On any failure, returns the input unchanged.
    """
    try:
        import cv2
        lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    except Exception:
        return frame_rgb

def mean_brightness(frame_rgb: np.ndarray) -> float:
    """Mean pixel brightness 0..255. Used for the low-light warning."""
    try:
        return float(np.mean(frame_rgb))
    except Exception:
        return 128.0

LOW_LIGHT_THRESHOLD = 60.0

if __name__ == "__main__":
    import numpy as np
    dark = np.full((100, 100, 3), 30, dtype=np.uint8)
    bright = np.full((100, 100, 3), 200, dtype=np.uint8)
    assert mean_brightness(dark) < LOW_LIGHT_THRESHOLD
    assert mean_brightness(bright) > LOW_LIGHT_THRESHOLD
    out = clahe_rgb(dark)
    assert out.shape == dark.shape and out.dtype == np.uint8
    print("PREPROCESSING OK")
```

Run `python shared/preprocessing.py`. Must print `PREPROCESSING OK`.

---

### STEP 3 — Wire the new signals into `api/orchestrator.py`

This is the integration seam. The existing `verify()` handler runs the old
pipeline. You are adding the new signals and the fusion step, without touching
the existing checks.

**Critical performance requirement:** A, B, and C must run in PARALLEL, not
sequentially. Sequential takes 4-6 seconds and reads as slow at a judge's table.
Parallel brings it to 2-2.5 seconds.

Implement this structure:

```python
"""
api/orchestrator.py — FastAPI orchestrator.
Existing SIH pipeline unchanged. New HackMUJ signals added below.
"""
import asyncio, sqlite3, os
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, File, Form

from shared.contracts import SignalResult, SignalId
from shared.audit_chain import append_entry
from modules.forensics.genai_detector import detect_genai_document
from modules.face.deepfake_detector import detect_face_deepfake
from modules.face.challenge import issue_challenge, run_challenge, _decode_frames
from modules.decision.fusion import fuse

_executor = ThreadPoolExecutor(max_workers=4)

# Pepper lives in memory only. Loaded at startup from officer shares or env.
_PEPPER: bytes | None = None
```

**Startup pepper loading:**
```python
@app.on_event("startup")
def load_pepper():
    global _PEPPER
    from shared.pepper_sss import generate_pepper, reconstruct_pepper
    shares_env = os.environ.get("AUDIT_SHARES")   # "share1|||share2"
    if shares_env:
        _PEPPER = reconstruct_pepper(shares_env.split("|||"))
        print("[audit] pepper reconstructed from officer shares")
    else:
        _PEPPER = generate_pepper()
        print("[audit] WARNING: ephemeral pepper generated. "
              "Chain will not verify across restarts. Demo mode only.")
```

**The verify endpoint:**
```python
@app.post("/api/verify")
async def verify(document_image: UploadFile = File(...),
                 video: UploadFile = File(None),
                 audio: UploadFile = File(None)):
    loop = asyncio.get_event_loop()

    # 1. Save uploads to temp paths
    doc_path, video_path, audio_path = _save_uploads(document_image, video, audio)

    # 2. EXISTING pipeline — unchanged. Produces ocr_name and existing_score.
    existing_result = run_existing_pipeline(doc_path)
    ocr_name       = existing_result["ocr_name"]
    existing_score = existing_result["score"]

    # 3. Issue the challenge
    ch = issue_challenge(ocr_name)

    # 4. Decode frames ONCE, reuse for both B and C-i
    frames = _decode_frames(video_path) if video_path else []

    # 5. Fire A, B, C in PARALLEL
    genai_task     = loop.run_in_executor(_executor, detect_genai_document, doc_path)
    deepfake_task  = loop.run_in_executor(_executor, detect_face_deepfake, frames)
    challenge_task = loop.run_in_executor(
        _executor, run_challenge, video_path, audio_path, ch, ocr_name)

    genai_res, deepfake_res, challenge_results = await asyncio.gather(
        genai_task, deepfake_task, challenge_task)

    # 6. Fuse everything
    all_new = [genai_res, deepfake_res] + challenge_results
    fused = fuse(existing_score, all_new)

    # 7. Build response matching the dashboard contract EXACTLY
    response = {
        "existing": existing_result.get("breakdown", {}),
        "lines":    fused["lines"],
        "final": {
            "score":          fused["score"],
            "tier":           fused["tier"],
            "floors_applied": fused["floors_applied"],
        },
        "summary": build_officer_summary(fused),
    }

    # 8. Append to the hash-chained audit log
    db = sqlite3.connect("audit.db"); db.row_factory = sqlite3.Row
    append_entry(db, _PEPPER, response)
    db.close()

    # 9. Cache for the dashboard's /api/verify/last endpoint
    _LAST_RESULT.update(response)
    return response

@app.get("/api/verify/last")
async def last_result():
    return _LAST_RESULT or {"error": "no verification run yet"}
```

**Officer summary rules — this is a security requirement, not a nicety:**
```python
def build_officer_summary(fused: dict) -> str:
    """
    Grounded STRICTLY on fused output. The LLM (or this template) can NEVER
    alter the score or the tier. It only narrates what the deterministic
    scorer already decided.
    """
    parts = [f"Final verdict: {fused['tier'].upper()} (score {fused['score']})."]
    for line in fused["lines"]:
        if line["triggered"]:
            parts.append(f"{line['signal']}: {line['label']} "
                         f"(confidence {line['confidence']:.2f}).")
    if fused["floors_applied"]:
        parts.append("Anti-dilution floor applied: "
                     + ", ".join(fused["floors_applied"]) + ".")
    return " ".join(parts)
```

**Acceptance test:** create `tests/test_orchestrator.py`:
```python
def test_response_shape():
    """Verify the response matches the dashboard contract exactly."""
    # build a fake fused result, call build_officer_summary, assert keys present
    # assert "lines", "final", "summary" in response
    # assert every line has signal/penalty/label/confidence/severity/triggered
    # assert final has score/tier/floors_applied
```
Run it. Must pass.

⏸ **INPUT NEEDED at this step:** The existing `run_existing_pipeline()` function
lives in the original SIH codebase and I do not have its exact signature. Ask
Yash: "What is the exact function name and return shape of the existing verify
pipeline, and which key holds the OCR-extracted name?" Do not guess this.

---

### STEP 4 — Security review of fusion (write it up, do not just check it)

Read `modules/decision/fusion.py` line by line and produce
`docs/SECURITY_REVIEW.md` answering these four questions with evidence from the code:

1. **Can a triggered HARD signal be diluted?** Show the floor logic. Demonstrate
   with a concrete number: existing_score=5, deepfake raw=0.95, everything else
   clean. Show the score before the floor and after.

2. **Can a crashed detector be used as an attack?** If an attacker crashes the
   deepfake detector, `ok=False` means it is skipped and adds no penalty. Is that
   fail-open behaviour exploitable? State the risk honestly and what you would do
   with more time (answer: log the crash prominently, and a crash count above a
   threshold should itself raise the tier).

3. **Does the audit chain genuinely prevent single-party tampering?** Walk through:
   attacker edits row 5's payload. Chain breaks at row 5. To re-chain they need
   the pepper. The pepper is split 2-of-3. One officer cannot reconstruct it.
   State the remaining weakness: if the server is running, the pepper is in
   process memory and a root-level attacker could extract it.

4. **Is the officer summary capable of altering the verdict?** Show that
   `build_officer_summary` only reads from `fused` and returns a string that is
   never fed back into scoring.

Write these as short, honest paragraphs. Judges respect a team that knows its own
weak points.

---

### STEP 5 — Update `docs/DISCLOSURE.md`

Replace the existing content with:

```markdown
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
```

---

### STEP 6 — Write `docs/DEMO_SCRIPT.md`

A judge gives you 30 to 90 seconds. Write the exact script.

Structure it as:
- **0:00–0:15** One sentence on the problem. One sentence on what the system does.
- **0:15–0:45** Live run: genuine subject. Show the dashboard turning green.
- **0:45–1:15** Live run: attack. Pick the most visually obvious one (a deepfake
  video, or a TTS voice reading the challenge phrase). Show the dashboard turning
  red and point at the floor badge.
- **1:15–1:30** One line on the audit chain: "tampering any entry breaks the chain,
  and no single officer can repair it."

Include the exact sentences to say. Not bullet points, actual sentences.

⏸ **INPUT NEEDED:** Ask Yash which attack demo is most reliable on their hardware
before writing the 0:45 section. Options: deepfake video file, TTS voice, wrong
person reading the phrase, or an AI-generated document image.

---

### STEP 7 — Final integration check

```bash
python smoke_test.py
python -m pytest tests/ -v
```

Report the results. If the orchestrator cannot start because the existing
pipeline import is missing, that is expected until Yash supplies the answer from
Step 3.

---
---
---

# ═══════════════════════════════════════════════════════════
# ROLE B — "I am the teammate doing AI/ML"
# ═══════════════════════════════════════════════════════════

## Your ownership

You own and may modify:
- `modules/forensics/genai_detector.py`
- `modules/face/deepfake_detector.py`
- `modules/audio/asr.py`, `modules/audio/antispoof.py`
- `tests/robustness/harness.py`, `tests/adversarial/selftest.py`
- `docs/LIMITATIONS.md`
- `data/` (all subfolders)
- `models/` (you will create this)
- `notebooks/` (Colab exports, you will create this)

You do NOT touch: `shared/`, `modules/decision/fusion.py`, `api/`, `frontend/`.

## The situation you are inheriting

All four of your detectors WORK and pass their tests. But all four are currently
running on **heuristic fallbacks**, not real models:

| Detector | Currently running | Should be running |
|---|---|---|
| genai_detector | Radial FFT high-frequency ratio | CLIP ViT-B/32 linear probe |
| deepfake_detector | Laplacian variance sigmoid | Xception / HF deepfake classifier |
| antispoof | MFCC delta variance + spectral flatness | AASIST ONNX (ASVspoof2019) |
| asr | faster-whisper tiny (this one is already correct) | no change needed |

Your job is to upgrade the first three from heuristic to real, and to seed the
test data so the robustness numbers mean something.

**Constraint you must respect:** Colab GPU is for weight EXPORT only. Runtime
inference happens on the demo laptop CPU. Do not build anything that requires a
network call at demo time.

---

### STEP 1 — Verify what you inherited

```bash
python modules/forensics/genai_detector.py
python modules/face/deepfake_detector.py
python modules/audio/asr.py
python modules/audio/antispoof.py
python tests/robustness/harness.py
python tests/adversarial/selftest.py
```

All six must print their OK strings. Report each one.

Then run this and report the output:
```bash
python -c "
from modules.forensics.genai_detector import detect_genai_document
import numpy as np, tempfile
from PIL import Image
arr = np.random.randint(0,255,(300,200,3),dtype=np.uint8)
f = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
Image.fromarray(arr).save(f.name)
r = detect_genai_document(f.name)
print('genai method:', r.evidence.get('method'))
"
```
This tells you which path each detector is actually taking right now.

---

### STEP 2 — Create `models/` and the loader convention

```bash
mkdir -p models notebooks
```

Create `models/README.md`:
```markdown
# Model weights — NOT committed to git

These files are exported from Colab and copied here manually.
`.gitignore` excludes this folder except for this README.

| File | Used by | Export notebook |
|---|---|---|
| `clip_genai_probe.pt` | genai_detector.py | notebooks/01_clip_probe.ipynb |
| `deepfake.onnx` | deepfake_detector.py | notebooks/02_deepfake_export.ipynb |
| `antispoof.onnx` | antispoof.py | notebooks/03_antispoof_export.ipynb |

If a file is absent, its detector falls back to a heuristic automatically.
The system never breaks on a missing model, it just gets weaker.
```

Add to `.gitignore`:
```
models/*
!models/README.md
notebooks/*.ipynb_checkpoints
```

---

### STEP 3 — Seed the test data

This is the critical path. Nothing downstream means anything without it. The
robustness harness is currently reporting numbers computed on random noise.

**Target counts:**

| Folder | What | Count | Source |
|---|---|---|---|
| `data/real_docs/` | Genuine ID scans | 20 | Phone photos of sample/test IDs |
| `data/genai_docs/` | AI-generated ID images | 20 | Stable Diffusion / Midjourney |
| `data/real_faces/` | Genuine face clips (3-5s) | 15 | Webcam, varied lighting |
| `data/deepfake_faces/` | Deepfake clips | 15 | FaceForensics++ / DFDC / face-swap tool |
| `data/real_voices/` | Genuine name reads | 15 | Phone recordings, varied accents |
| `data/synthetic_voices/` | TTS/cloned reads | 15 | ElevenLabs, Coqui TTS, Piper |

**What you can do autonomously right now:**

Generate the synthetic voices. Write and run `scripts/gen_synthetic_voices.py`:
```python
"""
Generate TTS samples for data/synthetic_voices/.
Uses pyttsx3 (offline, no API key) as the baseline generator.
If Coqui TTS is installed, use it instead for higher quality.
Each sample reads a name + date + 4-digit nonce, matching the challenge format.
"""
NAMES = ["Priya Sharma", "Rahul Verma", "Anjali Nair", "Vikram Singh",
         "Meera Iyer", "Arjun Patel", "Kavya Reddy", "Rohan Desai",
         "Sneha Kulkarni", "Aditya Rao", "Divya Menon", "Karan Malhotra",
         "Neha Joshi", "Siddharth Bose", "Pooja Gupta"]
# For each name: generate f"{name} 2026-09-11 {random 4 digits}"
# Save to data/synthetic_voices/{i:02d}_{slug}.wav at 16kHz mono
```
Run it. Report how many files were created.

Also write `scripts/gen_genai_docs.py` **if** a local Stable Diffusion is
available (check for `diffusers`). If not, skip and flag it for the human.

⏸ **INPUT NEEDED after the autonomous part:**
```
⏸  INPUT NEEDED
    What I need:  Real-world samples I cannot generate.
                  1. 15 voice recordings: each person says a name, then
                     "2026-09-11", then a 4-digit number. Phone voice memo is fine.
                     Vary the accents. Save as WAV or M4A to data/real_voices/.
                  2. 15 short webcam clips (3-5s) of real faces, some in dim
                     light, to data/real_faces/.
                  3. 20 photos of documents to data/real_docs/.
    Why:          Every AUC number in the robustness report is meaningless
                  without real samples on both sides of the classifier.
    What I'll do: Once these land, I run the full harness and produce the
                  real accuracy table for the pitch.
```

For deepfake_faces and genai_docs, tell them:
- Deepfakes: FaceForensics++ requires a signed academic request form which will
  not arrive in time. Faster options: use a face-swap app on the real_faces
  clips you just recorded, or download from the DFDC public sample set.
- GenAI docs: generate with any image model. Prompt shape:
  "a photograph of a national identity card, front view, printed text, photo of a person".
  Twenty images takes about ten minutes.

---

### STEP 4 — Colab notebook 1: CLIP probe for genai_detector

Create `notebooks/01_clip_probe.ipynb` with these cells, ready to paste into Colab:

**Cell 1 — setup**
```python
!pip install open_clip_torch scikit-learn pillow -q
import torch, open_clip, numpy as np
from PIL import Image
from pathlib import Path
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
```

**Cell 2 — upload data**
```python
from google.colab import files
# Zip data/real_docs and data/genai_docs locally, upload here, unzip
!unzip -q docs_dataset.zip -d /content/data
```

**Cell 3 — extract features**
```python
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai")
model = model.to(device).eval()

def embed(path):
    img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        return model.encode_image(img).cpu().numpy()[0]

X, y = [], []
for p in Path("/content/data/real_docs").glob("*"):
    X.append(embed(p)); y.append(0)
for p in Path("/content/data/genai_docs").glob("*"):
    X.append(embed(p)); y.append(1)
X = np.array(X); y = np.array(y)
print("features:", X.shape)
```

**Cell 4 — fit and evaluate**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
auc = roc_auc_score(yte, clf.predict_proba(Xte)[:,1])
print("held-out AUC:", auc)
assert auc >= 0.75, "AUC below target — need more or more varied data"
```

**Cell 5 — export**
```python
import torch.nn as nn
head = nn.Linear(512, 1)
head.weight.data = torch.tensor(clf.coef_, dtype=torch.float32)
head.bias.data   = torch.tensor(clf.intercept_, dtype=torch.float32)
torch.save({"state_dict": head.state_dict(), "auc": float(auc)},
           "clip_genai_probe.pt")
files.download("clip_genai_probe.pt")
```

**Cell 6 — also export the calibration data**
```python
# Save the held-out raw scores + labels so calibration.py can fit Platt scaling
np.savez("genai_calibration_data.npz",
         scores=clf.predict_proba(Xte)[:,1], labels=yte)
files.download("genai_calibration_data.npz")
```

Then update `genai_detector.py` so its primary path loads
`models/clip_genai_probe.pt` if present. The FFT fallback stays as-is for when
the file is absent. Re-run its acceptance test.

---

### STEP 5 — Colab notebook 2: deepfake model

Create `notebooks/02_deepfake_export.ipynb`.

First, test whether the HF model downloads on the demo machine at all:
```bash
python -c "
from transformers import pipeline
p = pipeline('image-classification',
             model='dima806/deepfake_vs_real_image_detection', device=-1)
print('HF model loads OK')
"
```

⏸ **DECISION POINT — resolve this before hour 4:**
```
⏸  INPUT NEEDED
    What I need:  Does the HF model download and run on the DEMO machine
                  (not this dev machine)?
    Why:          If yes, deepfake_detector needs no change at all.
                  If no, I export ONNX on Colab and change the loader.
                  This blocks final demo-machine setup and cannot slip.
    Options:      (a) It works — nothing to do
                  (b) It fails / no internet at venue — I build the ONNX path
    What I'll do: Either mark this step done, or write the ONNX export notebook
                  and update the loader.
```

If ONNX is needed, the notebook exports to `models/deepfake.onnx` and you add an
onnxruntime path to `deepfake_detector.py` ahead of the Laplacian fallback.

---

### STEP 6 — Colab notebook 3: AASIST antispoof

Create `notebooks/03_antispoof_export.ipynb`.

```python
# Clone AASIST, load ASVspoof2019-LA pretrained weights
!git clone https://github.com/clovaai/aasist.git
# Load the model, trace it, export to ONNX with input shape (1, 1, 64000)
torch.onnx.export(model, dummy_input, "antispoof.onnx",
                  input_names=["audio"], output_names=["score"],
                  opset_version=14, dynamic_axes={"audio": {0: "batch"}})
files.download("antispoof.onnx")
```

`antispoof.py` already has the ONNX loader path written. It checks for
`models/antispoof.onnx` and uses it automatically if present. You only need to
produce the file and verify the input shape matches what the loader sends
(`(1, 1, 64000)` float32).

Re-run `python modules/audio/antispoof.py` after dropping the file in. Confirm
`evidence["method"]` now reports `onnx_aasist` and not `mfcc_heuristic`.

---

### STEP 7 — Fit calibration against real data

Once Step 3 data exists and models are upgraded, run calibration for real:

Write `scripts/fit_all_calibrations.py`:
```python
"""
Run every detector across the seeded dataset, collect raw scores and true labels,
fit Platt scaling per signal, persist to calibration.json.
This is what makes the reported confidence defensible to a judge who asks
'how did you get that number?'
"""
# for genai_doc:       real_docs -> label 0, genai_docs -> label 1
# for face_deepfake:   real_faces -> 0, deepfake_faces -> 1
# for voice_spoof:     real_voices -> 0, synthetic_voices -> 1
# call fit_calibrator(scores, labels, signal_name) for each
# print the resulting AUC per signal
```

Run it. Report the AUC per signal. Targets from the original spec:
genai_doc >= 0.75, face_deepfake >= 0.70, voice_spoof >= 0.70.

If any misses its target, say so plainly and note it for LIMITATIONS.md. Do not
quietly tune the threshold to make the number look better.

---

### STEP 8 — Run the real robustness harness

```bash
python tests/robustness/harness.py
```

Now that real data exists, this produces meaningful numbers. Take the markdown
table it prints and save it to `docs/ROBUSTNESS_RESULTS.md` with a one-paragraph
intro explaining the degradations tested.

This directly answers the PS requirement for "robustness testing against
compression/lighting/low bandwidth" with evidence rather than a claim.

---

### STEP 9 — Adversarial self-test and honest limitations

```bash
python tests/adversarial/selftest.py
```

Read `tests/adversarial/results.json`. Then write `docs/LIMITATIONS.md` properly.
Do not write marketing copy. Write what a hostile reviewer would find:

```markdown
# Known Limitations

## Generalisation
<What happened when tested against a generator not in the training set. Give the
actual number.>

## Temporal analysis
The deepfake detector scores frames independently and aggregates the three most
suspicious. It does not model temporal inconsistency across frames, which is
where several published detectors get their strongest signal. This was a CPU
budget decision, not an oversight.

## Calibration sample size
<State how many validation samples the Platt fit used. If under 50, say the
confidence values are indicative rather than statistically robust.>

## Voice anti-spoofing scope
<State which TTS engines were in the test set. A cloned voice from an engine not
represented may score lower than it should.>

## Active liveness
<State the printed-photo replay result from the adversarial test honestly.>

## Audit chain
Tamper-evidence holds at rest. While the server process is running, the pepper is
in memory and a root-level attacker could extract it. Full protection would need
an HSM or enclave.
```

Numbers, not adjectives. A judge who finds a weakness you already disclosed
trusts everything else you said.

---

### STEP 10 — Report

Summarise: which detectors are on real models vs heuristics, what the AUCs are,
what the robustness deltas are, and what is in LIMITATIONS.md.

---
---
---

# ═══════════════════════════════════════════════════════════
# ROLE C — "I am the teammate doing frontend"
# ═══════════════════════════════════════════════════════════

## Your ownership

You own and may modify:
- `frontend/static/dashboard.html`
- `frontend/static/app.js`
- `frontend/static/capture.html`
- `frontend/static/verify_sample.json` (the fixture)
- `docs/DEMO_REHEARSAL.md`

You do NOT touch: `shared/`, `modules/`, `api/`, `tests/`, `data/`.

## Your advantage

You are never blocked. The backend contract is already frozen and there is a
fixture file to develop against. Everything you build works before the backend
is wired, and keeps working after.

## The contract you consume — memorise this shape

`POST /api/verify` and `GET /api/verify/last` both return:

```json
{
  "existing": {},
  "lines": [
    {
      "signal": "genai_doc",
      "penalty": 3.75,
      "label": "Document appears genuine",
      "confidence": 0.15,
      "severity": "soft",
      "triggered": false,
      "evidence": {"heatmap_path": "/static/hm_genai_demo.png", "method": "fft_heuristic"}
    },
    {
      "signal": "face_deepfake",
      "penalty": 25.0,
      "label": "Deepfake artifacts detected",
      "confidence": 0.91,
      "severity": "hard",
      "triggered": true,
      "evidence": {"mode": "per_frame_aggregate", "per_frame_scores": [0.88,0.92,0.95,0.71,0.85]}
    },
    {
      "signal": "active_liveness",
      "penalty": 0.0,
      "label": "Liveness action confirmed: turn_left",
      "confidence": 0.08,
      "severity": "soft",
      "triggered": false,
      "evidence": {"expected_action": "turn_left", "detected": true}
    },
    {
      "signal": "name_match",
      "penalty": 0.0,
      "label": "Spoken name matches document",
      "confidence": 0.05,
      "severity": "soft",
      "triggered": false,
      "evidence": {"transcript": "priya sharma 2026-09-11 4471", "name_ratio": 0.98,
                   "nonce_present": true, "date_present": true}
    },
    {
      "signal": "voice_spoof",
      "penalty": 0.0,
      "label": "Voice appears genuine",
      "confidence": 0.18,
      "severity": "hard",
      "triggered": false,
      "evidence": {"method": "mfcc_heuristic"}
    }
  ],
  "final": {"score": 82.0, "tier": "flagged", "floors_applied": ["face_deepfake"]},
  "summary": "Final verdict: FLAGGED (score 82.0). face_deepfake: Deepfake artifacts detected (confidence 0.91). Anti-dilution floor applied: face_deepfake."
}
```

**Signals that may appear in `lines`:** `ela`, `exif`, `crossfield`, `face_match`,
`passive_liveness`, `watchlist`, `genai_doc`, `face_deepfake`, `active_liveness`,
`name_match`, `voice_spoof`. Render whatever arrives. Never assume a fixed count.

---

### STEP 1 — Verify the fixture loads

```bash
cd frontend/static && python -m http.server 8080
```
Open `http://localhost:8080/verify_sample.json` in a browser. Confirm it parses.
Report what you see.

---

### STEP 2 — Build `dashboard.html`

**This is the single most important artifact for winning.** A judge spends 30
seconds at your table. Everything the system does must be legible in that time.

**Layout — 3 columns, responsive down to 1 column on narrow screens:**

```
┌──────────────────┬──────────────────┬──────────────────┐
│ FUSED SCORE      │ ELA Heatmap      │ GenAI Heatmap    │
│  82   FLAGGED    │ (img or grey box)│ (img or grey box)│
│  ████████░░      │                  │                  │
│  ┊30      ┊70    ├──────────────────┼──────────────────┤
│                  │ Face Deepfake    │ Active Liveness  │
│ FLOORS APPLIED   │ 91%  [HARD]      │ ✓ turn_left      │
│ ● face_deepfake  │ ▁▃█▇█ sparkline  │ Confirmed        │
│   → FLAGGED      ├──────────────────┼──────────────────┤
├──────────────────┤ Name Match       │ Voice Spoof      │
│ ALL SIGNALS      │ ✓ 98% match      │ 18%  [HARD]      │
│ (table, 1 row    │ "priya sharma    │ Genuine          │
│  per line)       │  2026-09-11..."  │ mfcc_heuristic   │
└──────────────────┴──────────────────┴──────────────────┘
┌─────────────────────────────────────────────────────────┐
│ OFFICER SUMMARY                                          │
│ <the `summary` string>                                   │
└─────────────────────────────────────────────────────────┘
```

**Colour rules — apply with zero exceptions:**

| Condition | Colour | Hex |
|---|---|---|
| tier `clear` | green | `#22c55e` |
| tier `review` | amber | `#f59e0b` |
| tier `flagged` | red | `#ef4444` |
| `triggered && severity=="hard"` | red badge reading `HARD` | `#ef4444` |
| `triggered && severity=="soft"` | amber badge | `#f59e0b` |
| `!triggered` | green badge | `#22c55e` |
| A `HARD` signal not triggered | green fill, but still show the `HARD` label in grey |

That last row matters. `voice_spoof` in the fixture is HARD but not triggered.
It must read as safe while still showing it is a high-severity channel.

**Score bar:** horizontal, 0 to 100, filled to `score/100`, coloured by tier.
Draw thin vertical tick marks at 30% and 70% showing the tier boundaries. Label
them `30` and `70` in small grey text.

**Sparkline (face_deepfake panel):** if `evidence.per_frame_scores` exists, draw
an inline SVG bar chart, 180x40px. One bar per frame score. Bar colour: green
below 0.5, amber 0.5 to 0.7, red above 0.7. This makes per-frame analysis visible
rather than a claim.

**Signal table (bottom of column 1):** one row per entry in `lines`. Columns:
Signal | Confidence % | Penalty | Status. Colour each row background by triggered
state at 10% opacity.

**Heatmap panels:** if `evidence.heatmap_path` is a non-empty string, render
`<img src={path}>`. Otherwise a grey box reading "No heatmap available". Never
render a broken image icon.

**Data loading:**
```javascript
async function loadData(forceFixture = false) {
    if (!forceFixture) {
        try {
            const res = await fetch("/api/verify/last");
            if (res.ok) {
                const d = await res.json();
                if (!d.error) return d;
            }
        } catch (e) { /* backend not up, fall through */ }
    }
    const res = await fetch("verify_sample.json");
    return await res.json();
}
```

**Buttons, top right:** `[REFRESH]` calls `loadData()`. `[LOAD FIXTURE]` calls
`loadData(true)`. Both re-render.

**Auto-refresh:** poll `/api/verify/last` every 3 seconds. If the response
changes (compare `final.score` and the summary string), re-render with a brief
flash animation on the score panel. This makes the live demo feel alive without
anyone clicking.

**Technical constraints:**
- Tailwind via CDN: `<script src="https://cdn.tailwindcss.com"></script>`
- Vanilla JS only. No React, no build step.
- A single `render(data)` function does all DOM updates.
- Include `<style>` fallbacks for the six colour classes in case the CDN is
  unreachable at the venue. Do not assume venue wifi works.
- `<meta charset="utf-8">` and viewport meta required.

**Acceptance test — check every one of these manually:**
1. Score reads `82`, bar is red, tier reads `FLAGGED`
2. `face_deepfake` panel: red `HARD` badge, `91%`, sparkline with 5 bars
3. `voice_spoof` panel: grey `HARD` label but green status, `18%`
4. `name_match` panel shows the full transcript string
5. `active_liveness` shows `turn_left` and a checkmark
6. Floors section shows one red pill: `face_deepfake → FLAGGED`
7. Signal table has one row per line in the fixture
8. Officer summary text renders at the bottom
9. `[LOAD FIXTURE]` re-renders with no error
10. Browser console is clean, zero errors
11. Resize to 400px wide: layout collapses to one column, nothing overflows

Report which of the 11 pass.

---

### STEP 3 — Build a second fixture for the "clear" case

Create `frontend/static/verify_sample_clear.json`: same shape, but everything
genuine. `score: 8.0`, `tier: "clear"`, `floors_applied: []`, every line
`triggered: false`, confidences all under 0.20.

Add a third button `[DEMO: CLEAR]` that loads it. During the live demo you will
show clear then flagged back to back, and having both fixtures means you can
rehearse the visual transition without needing the backend or a live capture.

---

### STEP 4 — Build `capture.html` and `app.js`

The capture flow. Low bandwidth by design: never send full video.

**Bandwidth target: under 200KB per challenge round.** Full 480p 5-second video
is roughly 3-8MB. Five face crops plus audio is about 150KB. That is the
difference between working and not working on venue wifi.

**Capture constraints:**
```javascript
const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: {ideal: 854}, height: {ideal: 480}, frameRate: {ideal: 15} },
    audio: { sampleRate: 16000, channelCount: 1,
             echoCancellation: true, noiseSuppression: true }
});
```

**Challenge display:** the backend provides
```javascript
{action: "turn_left", spoken_phrase: "Priya Sharma 2026-09-11 4471",
 nonce: "4471", date_str: "2026-09-11"}
```
Show it large and unmissable:
```
  PERFORM:  Turn your head LEFT
  SAY:      "Priya Sharma 2026-09-11 4471"
            [ START CAPTURE ]
```

**Brightness check before recording starts:**
```javascript
function meanBrightness(ctx, w, h) {
    const d = ctx.getImageData(0, 0, w, h).data;
    let sum = 0;
    for (let i = 0; i < d.length; i += 4) sum += (d[i]+d[i+1]+d[i+2])/3;
    return sum / (d.length / 4);   // 0..255
}
```
Threshold 60. Below it: show an amber warning, "Lighting is too dim. Move to a
brighter area." Show a live brightness meter so they can see it improve. After 5
seconds of waiting, also show an `[Override]` button. Never permanently block them.

We deliberately do NOT use the camera flash or torch. `ImageCapture.setOptions`
with `fillLightMode` works on Chrome Android only and silently does nothing on
iOS and desktop. Server-side CLAHE handles the rest.

**Face crop extraction:**
Load MediaPipe FaceDetection from CDN
(`https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/face_detection.js`),
`model_selection=0` for short range.

For each keyframe: detect face, take `boundingBox`, add 20% padding each side,
crop from the canvas, resize to 224x224, encode JPEG at quality 0.85. Roughly
25KB per crop.

If MediaPipe fails to load: log `"MediaPipe unavailable — sending full frame"`
and send the full 480p frame instead. Degraded but functional.

**Five keyframes:** sample at t = 0.5, 1.5, 2.5, 3.5, 4.5 seconds during the
5-second window. Skip any frame where no face is detected. Never send more than
five.

**Audio:** `MediaRecorder` with `audio/webm;codecs=opus` across the whole window.

**Send:**
```javascript
const fd = new FormData();
crops.forEach((b, i) => fd.append(`frame_${i}`, b, `frame_${i}.jpg`));
fd.append("audio", audioBlob, "audio.webm");
fd.append("action", ch.action);
fd.append("nonce", ch.nonce);
fd.append("date_str", ch.date_str);
await fetch("/api/challenge", {method: "POST", body: fd});
```

**UI states — implement each visibly:**

| State | Shows |
|---|---|
| `idle` | Challenge card + START button |
| `checking_brightness` | Preview + live brightness meter |
| `warning_brightness` | Amber warning + Override button after 5s |
| `recording` | Preview + big countdown 5,4,3,2,1 + the phrase to say |
| `processing` | Spinner, "Analysing signals..." |
| `result` | Auto-redirect to `dashboard.html` |
| `error` | Red message + TRY AGAIN |

**Cleanup — do not skip this:** `stream.getTracks().forEach(t => t.stop())` must
run after send completes AND in every error path. A camera light left on during
judging looks careless.

**Payload size assertion:** before sending, compute total blob size and log it.
If it exceeds 300KB, log a warning. This is your own guard against the
low-bandwidth requirement silently regressing.

**Acceptance test (manual, no backend needed):**
1. Open `capture.html`, challenge card renders
2. START triggers camera and mic permission prompt
3. Preview appears, brightness meter moves when you cover the lens
4. Cover the lens: amber warning appears
5. Uncover: countdown starts
6. After 5s: "Analysing..." appears
7. DevTools Network tab: the POST to `/api/challenge` contains `frame_0`
   through `frame_4`, `audio`, `action`, `nonce`, `date_str`
8. Console logs the total payload size, under 300KB
9. Camera light goes off after the flow ends
10. Zero console errors

Report which pass.

---

### STEP 5 — Write `docs/DEMO_REHEARSAL.md`

You own rehearsal. Write the checklist the team runs 30 minutes before judging:

- Browser tabs to have open, in order
- Which fixture to have preloaded
- Exact click sequence for the clear demo
- Exact click sequence for the flagged demo
- What to do if the camera permission prompt does not appear
- What to do if venue wifi drops (answer: everything runs locally, load the
  fixtures)
- Who says what, in what order, for the three of you

---

### STEP 6 — Report

Summarise: which of the dashboard's 11 acceptance checks pass, which of the
capture flow's 10 pass, and whether both fixtures render correctly.

---
---
---

# ═══════════════════════════════════════════════════════════
# END OF ROLE SECTIONS
# ═══════════════════════════════════════════════════════════
