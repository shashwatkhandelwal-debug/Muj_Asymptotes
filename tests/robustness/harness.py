import os
import sys
import json
import uuid
import wave
import logging
from pathlib import Path
import numpy as np
from PIL import Image

# Ensure repo root is on sys.path
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from shared.contracts import SignalId, SIGNAL_TRIGGER
from modules.forensics.genai_detector import detect_genai_document
from modules.audio.antispoof import score_voice_spoof

logger = logging.getLogger(__name__)

DEGRADATIONS = {
    "jpeg_q40":    {"type": "image", "op": "jpeg", "q": 40},
    "jpeg_q20":    {"type": "image", "op": "jpeg", "q": 20},
    "dark_0.4":    {"type": "image", "op": "darken", "factor": 0.4},
    "audio_8k":    {"type": "audio", "op": "downsample", "target_sr": 8000},
    "audio_noise": {"type": "audio", "op": "noise", "db": -20},
}

def _get_tmp_dir() -> str:
    tmp_dir = "/tmp" if os.name != "nt" else os.environ.get("TEMP", "C:/tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir

def apply_image_degradation(image_path: str, op: dict) -> str:
    """Return path to degraded copy in /tmp/."""
    tmp_dir = _get_tmp_dir()
    out_name = f"deg_{uuid.uuid4().hex[:8]}.jpg"
    out_path = os.path.join(tmp_dir, out_name)

    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        if op.get("op") == "jpeg":
            q = op.get("q", 40)
            img_rgb.save(out_path, "JPEG", quality=q)
        elif op.get("op") == "darken":
            factor = op.get("factor", 0.4)
            arr = (np.array(img_rgb, dtype=np.float32) * factor).clip(0, 255).astype(np.uint8)
            Image.fromarray(arr).save(out_path, "JPEG", quality=95)
        else:
            img_rgb.save(out_path, "JPEG")

    return out_path

def apply_audio_degradation(audio_path: str, op: dict) -> str:
    """Return path to degraded copy in /tmp/."""
    tmp_dir = _get_tmp_dir()
    out_name = f"deg_{uuid.uuid4().hex[:8]}.wav"
    out_path = os.path.join(tmp_dir, out_name)

    # Read WAV
    with wave.open(audio_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    op_type = op.get("op")
    if op_type == "downsample":
        target_sr = op.get("target_sr", 8000)
        # Resample
        try:
            from scipy import signal
            num_samples = int(len(data) * target_sr / sr)
            resampled = signal.resample(data, num_samples)
        except Exception:
            ratio = max(1, sr // target_sr)
            resampled = data[::ratio]
        out_sr = target_sr
        out_data = (np.clip(resampled, -1.0, 1.0) * 32767.0).astype(np.int16)
    elif op_type == "noise":
        db = op.get("db", -20)
        sig_amp = float(np.std(data))
        if sig_amp < 1e-4:
            sig_amp = 0.5
        noise_amp = sig_amp * (10.0 ** (db / 20.0))
        noise = np.random.normal(0, noise_amp, size=len(data))
        noisy = np.clip(data + noise, -1.0, 1.0)
        out_sr = sr
        out_data = (noisy * 32767.0).astype(np.int16)
    else:
        out_sr = sr
        out_data = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)

    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(out_sr)
        wf.writeframes(out_data.tobytes())

    return out_path

def _ensure_synthetic_data(data_path: Path):
    # Creates synthetic data for testing if data/ is empty
    img_dir = data_path / "images"
    audio_dir = data_path / "audio"
    img_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    # 1 real-style image (smooth gradient)
    real_img_path = img_dir / "real_doc_1.png"
    if not real_img_path.exists():
        x = np.linspace(0, 255, 200)
        y = np.linspace(0, 255, 300)
        xx, yy = np.meshgrid(x, y)
        grad = ((xx + yy) / 2).astype(np.uint8)
        Image.fromarray(grad).convert("RGB").save(real_img_path)

    # 1 fake-style image (high freq noise)
    fake_img_path = img_dir / "genai_doc_1.png"
    if not fake_img_path.exists():
        noise = np.random.randint(0, 256, (300, 200, 3), dtype=np.uint8)
        Image.fromarray(noise).save(fake_img_path)

    # 1 real audio (sine wave)
    real_audio_path = audio_dir / "real_voice_1.wav"
    if not real_audio_path.exists():
        t = np.linspace(0, 1, 16000)
        tone = (np.sin(2 * np.pi * 440 * t) * 30000).astype(np.int16)
        with wave.open(str(real_audio_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(tone.tobytes())

    # 1 fake audio (flat / silent)
    fake_audio_path = audio_dir / "spoof_voice_1.wav"
    if not fake_audio_path.exists():
        with wave.open(str(fake_audio_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00" * 32000)

def run_robustness(dataset_dir: str = "data") -> dict:
    """
    For each degradation:
      1. Apply to all files in relevant data subfolder.
      2. Run the appropriate detector on each degraded file.
      3. Compute accuracy: # correct classifications / total.
         A real doc/face/voice scoring below trigger = correct (genuine).
         A genai/deepfake/synthetic scoring above trigger = correct (fake).
      4. Record baseline accuracy (no degradation) and degraded accuracy.
    Return {"degradation_name": {"baseline": float, "degraded": float, "delta": float}, ...}
    Also write this dict to tests/robustness/report.json.
    Also print a markdown table of results to stdout.
    """
    data_path = Path(dataset_dir)
    _ensure_synthetic_data(data_path)

    image_files = list((data_path / "images").glob("*.*"))
    audio_files = list((data_path / "audio").glob("*.wav"))

    genai_trigger = SIGNAL_TRIGGER[SignalId.GENAI_DOC]
    voice_trigger = SIGNAL_TRIGGER[SignalId.VOICE_SPOOF]

    report = {}

    for deg_name, deg_info in DEGRADATIONS.items():
        deg_type = deg_info["type"]
        if deg_type == "image":
            files = image_files
            trigger = genai_trigger

            def run_det(path):
                return detect_genai_document(str(path)).raw_score

            def is_fake(p):
                name = p.name.lower()
                return "genai" in name or "fake" in name
        else:
            files = audio_files
            trigger = voice_trigger

            def run_det(path):
                return score_voice_spoof(str(path)).raw_score

            def is_fake(p):
                name = p.name.lower()
                return "spoof" in name or "fake" in name or "synthetic" in name

        if not files:
            report[deg_name] = {"baseline": 1.0, "degraded": 1.0, "delta": 0.0}
            continue

        base_correct = 0
        deg_correct = 0
        total = len(files)

        for f in files:
            fake_label = is_fake(f)

            # Baseline
            base_score = run_det(f)
            base_pred_fake = (base_score > trigger)
            if base_pred_fake == fake_label:
                base_correct += 1

            # Degraded
            if deg_type == "image":
                deg_path = apply_image_degradation(str(f), deg_info)
            else:
                deg_path = apply_audio_degradation(str(f), deg_info)

            try:
                deg_score = run_det(deg_path)
                deg_pred_fake = (deg_score > trigger)
                if deg_pred_fake == fake_label:
                    deg_correct += 1
            finally:
                if os.path.exists(deg_path):
                    try:
                        os.remove(deg_path)
                    except Exception:
                        pass

        base_acc = float(base_correct / total)
        deg_acc = float(deg_correct / total)
        delta = float(deg_acc - base_acc)

        report[deg_name] = {
            "baseline": base_acc,
            "degraded": deg_acc,
            "delta": delta,
        }

    # Write report.json
    out_dir = Path("tests/robustness")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "report.json"
    with open(report_file, "w", encoding="utf-8") as rf:
        json.dump(report, rf, indent=2)

    # Print markdown table
    print("\n| Degradation | Baseline Acc | Degraded Acc | Delta |")
    print("|---|---|---|---|")
    for k, v in report.items():
        print(f"| {k} | {v['baseline']:.2f} | {v['degraded']:.2f} | {v['delta']:+.2f} |")
    print("")

    return report

if __name__ == "__main__":
    rep = run_robustness()
    assert isinstance(rep, dict)
    for k in ["jpeg_q40", "jpeg_q20", "dark_0.4", "audio_8k", "audio_noise"]:
        assert k in rep, f"Missing key {k} in report"
        assert "baseline" in rep[k]
        assert "degraded" in rep[k]
        assert "delta" in rep[k]
    print("ROBUSTNESS HARNESS OK")
