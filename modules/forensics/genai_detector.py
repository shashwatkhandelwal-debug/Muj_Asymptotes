import os
import sys
import time
import uuid
import logging
from pathlib import Path
import numpy as np
from PIL import Image

# Ensure repo root is on sys.path
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.contracts import SignalResult, SignalId, Severity, SIGNAL_TRIGGER
from modules.decision.calibration import apply_calibration

logger = logging.getLogger(__name__)

def _compute_patch_hf_ratio(gray_patch: np.ndarray) -> float:
    h, w = gray_patch.shape
    if h < 4 or w < 4:
        return 0.0
    f = np.fft.fft2(gray_patch)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)

    cy, cx = h / 2.0, w / 2.0
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    max_r = max(1.0, min(h, w) / 2.0)

    bin_idx = np.clip((r / max_r * 8).astype(int), 0, 7)
    bins = []
    for b in range(8):
        mask = (bin_idx == b)
        if np.any(mask):
            bins.append(float(np.mean(mag[mask])))
        else:
            bins.append(0.0)

    low_f = float(np.mean(bins[:4])) + 1e-9
    high_f = float(np.mean(bins[4:]))
    return high_f / low_f

def _generate_heatmap(gray_img: np.ndarray, heatmap_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(heatmap_path)), exist_ok=True)
    h, w = gray_img.shape
    grid_rows, grid_cols = 4, 4
    h_step = max(1, h // grid_rows)
    w_step = max(1, w // grid_cols)

    heatmap_vals = np.zeros((grid_rows, grid_cols), dtype=float)
    for r in range(grid_rows):
        for c in range(grid_cols):
            patch = gray_img[r * h_step:(r + 1) * h_step, c * w_step:(c + 1) * w_step]
            heatmap_vals[r, c] = _compute_patch_hf_ratio(patch)

    min_v, max_v = heatmap_vals.min(), heatmap_vals.max()
    if max_v > min_v:
        norm_vals = (heatmap_vals - min_v) / (max_v - min_v)
    else:
        norm_vals = np.zeros_like(heatmap_vals)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(norm_vals, cmap="magma", interpolation="bilinear")
    plt.colorbar(im, ax=ax)
    ax.set_title("GenAI Artifact Map")
    ax.axis("off")
    fig.savefig(heatmap_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    return heatmap_vals.flatten().tolist()
_PROBE_HEAD = None
_PROBE_TRIED = False
_CLIP_PROCESSOR = None
_CLIP_MODEL = None
_CLIP_TRIED = False

def _load_probe_head():
    global _PROBE_HEAD, _PROBE_TRIED
    if _PROBE_TRIED:
        return _PROBE_HEAD
    _PROBE_TRIED = True
    probe_path = Path("models/clip_genai_probe.pt")
    if probe_path.exists():
        try:
            import torch
            import torch.nn as nn
            state = torch.load(str(probe_path), map_location="cpu")
            head = nn.Linear(512, 1)
            if isinstance(state, dict) and "state_dict" in state:
                head.load_state_dict(state["state_dict"])
            elif isinstance(state, dict):
                head.load_state_dict(state)
            head.eval()
            _PROBE_HEAD = head
            logger.info("Loaded custom CLIP probe head from %s", probe_path)
        except Exception as e:
            logger.warning("Failed to load clip_genai_probe.pt: %s", e)
            _PROBE_HEAD = None
    return _PROBE_HEAD

def _load_clip_model():
    global _CLIP_PROCESSOR, _CLIP_MODEL, _CLIP_TRIED
    if _CLIP_TRIED:
        return _CLIP_PROCESSOR, _CLIP_MODEL
    _CLIP_TRIED = True
    try:
        from transformers import CLIPProcessor, CLIPModel
        model_id = "openai/clip-vit-base-patch32"
        allow_download = os.environ.get("DOWNLOAD_PRETRAINED", "0") == "1"
        try:
            _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
            _CLIP_MODEL = CLIPModel.from_pretrained(model_id, local_files_only=True)
        except Exception:
            if allow_download:
                _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(model_id)
                _CLIP_MODEL = CLIPModel.from_pretrained(model_id)
        if _CLIP_MODEL is not None:
            _CLIP_MODEL.eval()
            logger.info("Loaded pre-trained CLIP model: %s", model_id)
    except Exception as e:
        logger.info("Pre-trained CLIP model not available locally or offline: %s", e)
        _CLIP_PROCESSOR, _CLIP_MODEL = None, None
    return _CLIP_PROCESSOR, _CLIP_MODEL

def detect_genai_document(image_path: str) -> SignalResult:
    """
    IN:  path to document image (JPEG or PNG).
    OUT: SignalResult(
           signal=SignalId.GENAI_DOC,
           raw_score=P(AI-generated) in [0,1],
           confidence=apply_calibration(raw_score, "genai_doc"),
           severity=Severity.SOFT,
           triggered=(raw_score > SIGNAL_TRIGGER[SignalId.GENAI_DOC]),
           label="Document appears AI-generated" if triggered else "Document appears genuine",
           evidence={"heatmap_path": "/tmp/hm_genai_<uuid>.png", "method": "clip_probe"|"fft_heuristic"}
         )
    Measure wall time and set result.ms.
    """
    t0 = time.time()
    heatmap_dir = "/tmp" if os.name != "nt" else os.environ.get("TEMP", "C:/tmp")
    os.makedirs(heatmap_dir, exist_ok=True)
    heatmap_filename = f"hm_genai_{uuid.uuid4().hex[:8]}.png"
    heatmap_path = os.path.join(heatmap_dir, heatmap_filename)

    try:
        if not os.path.exists(image_path):
            elapsed_ms = (time.time() - t0) * 1000.0
            return SignalResult(
                signal=SignalId.GENAI_DOC,
                raw_score=0.0,
                confidence=0.0,
                severity=Severity.SOFT,
                triggered=False,
                label="File not found",
                evidence={"error": f"Image not found: {image_path}"},
                ok=False,
                ms=elapsed_ms,
            )

        with Image.open(image_path) as pil_img:
            gray_img = np.array(pil_img.convert("L"), dtype=np.float32)

        method = "fft_heuristic"
        hf_ratio = _compute_patch_hf_ratio(gray_img)
        raw_score = float(min(1.0, max(0.0, hf_ratio * 2.0)))

        # Tier 1: Check for custom-trained probe head
        probe_head = _load_probe_head()
        processor, clip_model = _load_clip_model()

        if probe_head is not None and clip_model is not None and processor is not None:
            try:
                import torch
                inputs = processor(images=pil_img, return_tensors="pt")
                with torch.no_grad():
                    image_features = clip_model.get_image_features(**inputs)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    logit = probe_head(image_features.float())
                    raw_score = float(torch.sigmoid(logit).item())
                    method = "clip_probe"
            except Exception as e:
                logger.debug("CLIP probe evaluation failed, falling back: %s", e)

        # Tier 2: Zero-Shot Foundation CLIP (no training required!)
        elif clip_model is not None and processor is not None:
            try:
                import torch
                prompts = [
                    "a photograph of an authentic real official national identity document card",
                    "a synthetic artificial AI generated fake document with diffusion artifacts"
                ]
                inputs = processor(text=prompts, images=pil_img, return_tensors="pt", padding=True)
                with torch.no_grad():
                    outputs = clip_model(**inputs)
                    probs = outputs.logits_per_image.softmax(dim=1)
                    raw_score = float(probs[0, 1].item())
                    method = "clip_probe"
            except Exception as e:
                logger.debug("Zero-shot CLIP failed, falling back to FFT: %s", e)

        grid_scores = _generate_heatmap(gray_img, heatmap_path)
        spatial_variance = float(np.var(grid_scores))

        # Provenance metadata check
        from modules.forensics.provenance import check_provenance
        prov = check_provenance(image_path)

        # Blend provenance into raw_score (20% weight)
        raw_score = 0.80 * raw_score + 0.20 * prov["provenance_score"]
        raw_score = float(np.clip(raw_score, 0.0, 1.0))

        confidence = apply_calibration(raw_score, "genai_doc")
        triggered = bool(raw_score > SIGNAL_TRIGGER[SignalId.GENAI_DOC])
        label = (
            "Document appears AI-generated"
            if triggered
            else "Document appears genuine"
        )
        elapsed_ms = (time.time() - t0) * 1000.0

        return SignalResult(
            signal=SignalId.GENAI_DOC,
            raw_score=raw_score,
            confidence=confidence,
            severity=Severity.SOFT,
            triggered=triggered,
            label=label,
            evidence={
                "heatmap_path": heatmap_path,
                "method": method,
                "hf_ratio": hf_ratio,
                "spatial_variance": round(spatial_variance, 4),
                "provenance_score": prov["provenance_score"],
                "provenance_flags": prov["provenance_flags"],
            },
            ok=True,
            ms=elapsed_ms,
        )
    except Exception as e:
        logger.exception("Error in detect_genai_document: %s", e)
        elapsed_ms = (time.time() - t0) * 1000.0
        return SignalResult(
            signal=SignalId.GENAI_DOC,
            raw_score=0.0,
            confidence=0.0,
            severity=Severity.SOFT,
            triggered=False,
            label="Error analyzing document",
            evidence={"error": str(e)},
            ok=False,
            ms=elapsed_ms,
        )

if __name__ == "__main__":
    tmp_dir = "/tmp" if os.name != "nt" else os.environ.get("TEMP", "C:/tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    test_img_path = os.path.join(tmp_dir, "test_noise.png")
    noise_data = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
    Image.fromarray(noise_data).save(test_img_path)

    res = detect_genai_document(test_img_path)
    assert res.signal == SignalId.GENAI_DOC
    assert 0.0 <= res.raw_score <= 1.0
    assert os.path.exists(res.evidence["heatmap_path"])
    print("GENAI DETECTOR OK")
