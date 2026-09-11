import os
import sys
import time
import wave
import logging
from pathlib import Path
import numpy as np

# Ensure repo root is on sys.path
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from shared.contracts import SignalResult, SignalId, Severity, SIGNAL_TRIGGER
from modules.decision.calibration import apply_calibration

logger = logging.getLogger(__name__)

def _load_audio_signal(path: str):
    # Try wave first
    try:
        with wave.open(path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw_bytes = wf.readframes(n_frames)
            if sampwidth == 2:
                data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 1:
                data = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            elif sampwidth == 4:
                data = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                data = np.frombuffer(raw_bytes, dtype=np.float32)
            if n_channels > 1:
                data = data.reshape(-1, n_channels).mean(axis=1)
            return data, sr
    except Exception:
        pass

    # Try scipy.io.wavfile
    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(path)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            data = (data.astype(np.float32) - 128.0) / 128.0
        else:
            data = data.astype(np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, sr
    except Exception:
        pass

    # Try soundfile
    try:
        import soundfile as sf
        data, sr = sf.read(path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data.astype(np.float32), sr
    except Exception:
        pass

    # Try PyAV
    try:
        import av
        container = av.open(path)
        audio_stream = next(s for s in container.streams if s.type == "audio")
        chunks = []
        for frame in container.decode(audio_stream):
            arr = frame.to_ndarray()
            chunks.append(arr)
        if chunks:
            data = np.concatenate(chunks, axis=-1)
            if data.ndim > 1:
                data = data.mean(axis=0)
            return data.astype(np.float32), audio_stream.rate
    except Exception:
        pass

    return None, None

def score_voice_spoof(audio_path: str) -> SignalResult:
    """
    IN:  audio_path = path to WAV or MP4 audio.
    OUT: SignalResult(
           signal=SignalId.VOICE_SPOOF,
           severity=Severity.SOFT,
           raw_score=P(synthetic voice) in [0,1],
           confidence=apply_calibration(raw_score, "voice_spoof"),
           triggered=(raw_score > SIGNAL_TRIGGER[SignalId.VOICE_SPOOF]),
           label="Synthetic/TTS voice detected" if triggered else "Voice appears genuine",
           evidence={"method": "onnx_aasist"|"mfcc_heuristic"|"fft_fallback",
                     "features": {dict of computed values}}
         )
    If audio_path does not exist or cannot be decoded, return ok=False,
    raw_score=0.0, confidence=0.0.
    Measure wall time and set result.ms.
    """
    t0 = time.perf_counter()
    if not os.path.exists(audio_path):
        elapsed_ms = max(0.1, (time.perf_counter() - t0) * 1000.0)
        return SignalResult(
            signal=SignalId.VOICE_SPOOF,
            severity=Severity.SOFT,
            raw_score=0.0,
            confidence=0.0,
            triggered=False,
            label="Audio file not found",
            evidence={"error": f"Audio file not found: {audio_path}"},
            ok=False,
            ms=elapsed_ms,
        )

    y, sr = _load_audio_signal(audio_path)
    if y is None or len(y) == 0:
        elapsed_ms = max(0.1, (time.perf_counter() - t0) * 1000.0)
        return SignalResult(
            signal=SignalId.VOICE_SPOOF,
            severity=Severity.SOFT,
            raw_score=0.0,
            confidence=0.0,
            triggered=False,
            label="Audio decode error",
            evidence={"error": "Cannot decode audio stream"},
            ok=False,
            ms=elapsed_ms,
        )

    # 1. Primary: ONNX AASIST / RawGRU model if present
    onnx_path = Path("models/antispoof.onnx")
    if onnx_path.exists():
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(onnx_path))
            input_name = session.get_inputs()[0].name
            # Prepare input tensor: shape typically [1, num_samples]
            inp = np.expand_dims(y[:64600], axis=0).astype(np.float32)
            outputs = session.run(None, {input_name: inp})
            logits = outputs[0][0]
            # Softmax or sigmoid probability of spoof
            if len(logits) == 2:
                prob_spoof = float(np.exp(logits[1]) / np.sum(np.exp(logits)))
            else:
                prob_spoof = float(1.0 / (1.0 + np.exp(-logits[0])))
            raw_score = float(np.clip(prob_spoof, 0.0, 1.0))
            confidence = apply_calibration(raw_score, "voice_spoof")
            triggered = bool(raw_score > SIGNAL_TRIGGER[SignalId.VOICE_SPOOF])
            elapsed_ms = max(0.1, (time.perf_counter() - t0) * 1000.0)
            return SignalResult(
                signal=SignalId.VOICE_SPOOF,
                severity=Severity.SOFT,
                raw_score=raw_score,
                confidence=confidence,
                triggered=triggered,
                label="Synthetic/TTS voice detected" if triggered else "Voice appears genuine",
                evidence={"method": "onnx_aasist", "features": {"prob_spoof": prob_spoof}},
                ok=True,
                ms=elapsed_ms,
            )
        except Exception as e:
            logger.warning("ONNX antispoof failed: %s", e)

    # 2. Fallback: MFCC or FFT spectral classifier
    method = "fft_fallback"
    used_librosa = False
    try:
        import librosa
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        delta = librosa.feature.delta(mfccs)
        mfcc_delta_var = float(np.var(delta))
        spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        zcr_mean = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
        method = "mfcc_heuristic"
        used_librosa = True
    except Exception:
        used_librosa = False

    if not used_librosa:
        # np.fft fallback
        frame_len = 1024
        hop_len = 512
        num_frames = max(1, (len(y) - frame_len) // hop_len + 1)
        flatness_list = []
        diffs = []
        prev_spec = None

        for i in range(num_frames):
            frame = y[i * hop_len: i * hop_len + frame_len]
            if len(frame) < frame_len:
                frame = np.pad(frame, (0, frame_len - len(frame)))
            ps = np.abs(np.fft.rfft(frame)) ** 2 + 1e-12
            gm = np.exp(np.mean(np.log(ps)))
            am = np.mean(ps)
            flatness_list.append(gm / am)
            if prev_spec is not None:
                diffs.append(np.var(ps - prev_spec))
            prev_spec = ps

        spectral_flatness = float(np.mean(flatness_list)) if flatness_list else 0.0
        mfcc_delta_var = float(np.mean(diffs)) if diffs else 0.0
        zcr_mean = float(np.mean(np.abs(np.diff(np.sign(y)))) / 2.0) if len(y) > 1 else 0.0
        method = "fft_fallback"

    # Compute genuineness score via sigmoid, with path-aware coefficients.
    # Librosa MFCCs: fake speech has HIGHER delta variance than real.
    # FFT fallback:  real speech has HIGHER spectral-diff variance than fake.
    log_v = float(np.log1p(mfcc_delta_var))
    if used_librosa:
        # Librosa: real log_v ≈ 1.8, fake log_v ≈ 2.7; center ≈ 2.1
        # Higher log_v → more synthetic → z goes negative → low genuineness
        z = (2.1 - log_v) * 5.0 + (spectral_flatness - 0.003) * 40.0
    else:
        # FFT fallback: real log_v ≈ 4-5, fake log_v ≈ 2-3; center ≈ 5.8
        z = (log_v - 5.8) * 0.8 + (spectral_flatness - 0.010) * 80.0

    if z >= 0:
        genuineness = 1.0 / (1.0 + np.exp(-z))
    else:
        ez = np.exp(z)
        genuineness = ez / (1.0 + ez)

    # raw_score = P(synthetic voice): invert genuineness
    raw_score = float(np.clip(1.0 - genuineness, 0.0, 1.0))
    confidence = apply_calibration(raw_score, "voice_spoof")
    triggered = bool(raw_score > SIGNAL_TRIGGER[SignalId.VOICE_SPOOF])
    label = (
        "Synthetic/TTS voice detected"
        if triggered
        else "Voice appears genuine"
    )
    elapsed_ms = max(0.1, (time.perf_counter() - t0) * 1000.0)

    return SignalResult(
        signal=SignalId.VOICE_SPOOF,
        severity=Severity.SOFT,
        raw_score=raw_score,
        confidence=confidence,
        triggered=triggered,
        label=label,
        evidence={
            "method": method,
            "features": {
                "mfcc_delta_var": mfcc_delta_var,
                "log_v": log_v,
                "spectral_flatness": spectral_flatness,
                "zcr_mean": zcr_mean,
            },
        },
        ok=True,
        ms=elapsed_ms,
    )

if __name__ == "__main__":
    tmp_dir = "/tmp" if os.name != "nt" else os.environ.get("TEMP", "C:/tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    test_wav = os.path.join(tmp_dir, "test_voice.wav")

    # 1-second silent WAV
    with wave.open(test_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00" * 32000)

    res = score_voice_spoof(test_wav)
    assert isinstance(res, SignalResult)
    assert res.severity == Severity.SOFT
    assert 0.0 <= res.raw_score <= 1.0
    print("VOICE ANTISPOOF OK")
