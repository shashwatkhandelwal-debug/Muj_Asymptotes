"""
modules/forensics/provenance.py
Document image provenance check.
Detects screenshots, screen-captures, and AI-generated documents
by inspecting image metadata signatures.

Real camera-captured documents:
  - Have EXIF with camera model, focal length, exposure
  - DPI typically 72-300 from camera sensor
  - Software field absent or shows camera firmware

Suspicious provenance signatures:
  - No EXIF at all (stripped — common in AI tools)
  - Software field shows: 'Adobe', 'GIMP', 'Photoshop', 'Midjourney',
    'Stable Diffusion', 'PIL', 'Paint', 'Screenshot'
  - DPI exactly 72 or 96 (display resolution — typical of screenshots)
  - Image dimensions are exact multiples of 16 with no sensor data
    (diffusion model output is typically 512x512, 768x512, 1024x1024 etc)
  - No MakerNote (camera-specific EXIF block)
"""
import sys
import time
from pathlib import Path
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from shared.contracts import SignalResult, SignalId, Severity, SIGNAL_TRIGGER
from modules.decision.calibration import apply_calibration

# Software strings that suggest non-camera origin
SUSPICIOUS_SOFTWARE = [
    'adobe', 'photoshop', 'gimp', 'paint', 'screenshot',
    'midjourney', 'stable diffusion', 'dall-e', 'dalle',
    'firefly', 'canva', 'snagit', 'lightshot', 'pil',
    'python', 'imagemagick', 'inkscape'
]

# Display DPI values (screenshots)
DISPLAY_DPI = {72, 96}

# Common diffusion output dimensions
DIFFUSION_SIZES = {
    (512, 512), (768, 512), (512, 768),
    (1024, 1024), (1024, 768), (768, 1024),
    (1280, 720), (1920, 1080),
}


def check_provenance(image_path: str) -> dict:
    """
    IN:  path to document image
    OUT: dict with:
      provenance_score: float 0..1
      provenance_flags: list of strings
      provenance_ms: float
    """
    t0 = time.time()
    flags = []
    suspicion_score = 0.0

    try:
        from PIL import Image
        from PIL.ExifTags import TAGS

        img = Image.open(image_path)
        width, height = img.size

        # Check 1: diffusion output dimensions
        if (width, height) in DIFFUSION_SIZES:
            flags.append(f"dimensions {width}x{height} match common AI output size")
            suspicion_score += 0.3

        # Check 2: EXIF presence
        exif_data = {}
        try:
            raw_exif = img._getexif()
            if raw_exif:
                exif_data = {TAGS.get(k, k): v for k, v in raw_exif.items()}
            else:
                flags.append("no EXIF data — camera-captured documents typically have EXIF")
                suspicion_score += 0.25
        except Exception:
            flags.append("EXIF unreadable")
            suspicion_score += 0.1

        if exif_data:
            # Check 3: Software field
            software = str(exif_data.get('Software', '')).lower()
            for sus in SUSPICIOUS_SOFTWARE:
                if sus in software:
                    flags.append(f"Software field contains '{sus}'")
                    suspicion_score += 0.5
                    break

            # Check 4: Camera model
            make = exif_data.get('Make', '')
            model = exif_data.get('Model', '')
            if not make and not model:
                flags.append("no camera Make/Model — screenshot or AI likely")
                suspicion_score += 0.2

            # Check 5: DPI
            x_res = exif_data.get('XResolution')
            if x_res:
                try:
                    dpi = float(x_res)
                    if int(dpi) in DISPLAY_DPI:
                        flags.append(f"DPI={int(dpi)} matches display resolution (screenshot indicator)")
                        suspicion_score += 0.3
                except Exception:
                    pass

            # Check 6: MakerNote (camera-specific block)
            if 'MakerNote' not in exif_data:
                flags.append("no MakerNote — typical of non-camera sources")
                suspicion_score += 0.1

        suspicion_score = min(1.0, suspicion_score)

    except Exception as e:
        flags.append(f"provenance check error: {e}")
        suspicion_score = 0.0

    elapsed = (time.time() - t0) * 1000

    return {
        "provenance_score": round(suspicion_score, 4),
        "provenance_flags": flags,
        "provenance_ms":    round(elapsed, 1),
    }


if __name__ == "__main__":
    import tempfile, os
    import numpy as np
    from PIL import Image

    # Test 1: random noise image (no EXIF — should flag)
    arr = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    Image.fromarray(arr).save(tmp.name)
    tmp.close()

    result = check_provenance(tmp.name)
    print("No-EXIF image provenance:")
    print("  score:", result["provenance_score"])
    print("  flags:", result["provenance_flags"])
    assert result["provenance_score"] > 0, "No-EXIF should have non-zero suspicion"
    os.unlink(tmp.name)

    print("PROVENANCE CHECK OK")
