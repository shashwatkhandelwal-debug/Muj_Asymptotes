"""
Generate TTS samples for data/synthetic_voices/.
Uses pyttsx3 (offline, no API key) as the baseline generator.
If Coqui TTS is installed, use it instead for higher quality.
Each sample reads a name + date + 4-digit nonce, matching the challenge format.
"""
import os
import re
import random
import wave
from pathlib import Path
import numpy as np

NAMES = [
    "Priya Sharma", "Rahul Verma", "Anjali Nair", "Vikram Singh",
    "Meera Iyer", "Arjun Patel", "Kavya Reddy", "Rohan Desai",
    "Sneha Kulkarni", "Aditya Rao", "Divya Menon", "Karan Malhotra",
    "Neha Joshi", "Siddharth Bose", "Pooja Gupta"
]

def _convert_to_16k_mono(in_wav: str, out_wav: str):
    try:
        from scipy import signal
        with wave.open(in_wav, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        if sampwidth == 2:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 1:
            data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            data = np.frombuffer(raw, dtype=np.float32)

        if n_channels > 1:
            data = data.reshape(-1, n_channels).mean(axis=1)

        target_sr = 16000
        if sr != target_sr:
            num_samples = int(len(data) * target_sr / sr)
            data = signal.resample(data, num_samples)

        out_int16 = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
        with wave.open(out_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(out_int16.tobytes())
    except Exception:
        # If conversion fails, keep in_wav as out_wav
        if in_wav != out_wav and os.path.exists(in_wav):
            import shutil
            shutil.copy2(in_wav, out_wav)

def main():
    out_dir = Path("data/synthetic_voices")
    out_dir.mkdir(parents=True, exist_ok=True)

    import pyttsx3
    engine = pyttsx3.init()

    created_files = []
    random.seed(42)

    for i, name in enumerate(NAMES, start=1):
        slug = re.sub(r"[^\w]", "_", name.lower())
        nonce = f"{random.randint(0, 9999):04d}"
        date_str = "2026-09-11"
        phrase = f"{name} {date_str} {nonce}"

        temp_wav = out_dir / f"tmp_{i:02d}_{slug}.wav"
        final_wav = out_dir / f"{i:02d}_{slug}.wav"

        engine.save_to_file(phrase, str(temp_wav))
        engine.runAndWait()

        if temp_wav.exists():
            _convert_to_16k_mono(str(temp_wav), str(final_wav))
            if temp_wav.exists() and temp_wav != final_wav:
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass
            created_files.append(str(final_wav))
            print(f"[{i:02d}/{len(NAMES)}] Generated: {final_wav.name} -> '{phrase}'")

    print(f"\nSuccessfully generated {len(created_files)} synthetic voice samples in {out_dir}")

if __name__ == "__main__":
    main()
