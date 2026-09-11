"""
modules/forensics/exif.py — EXIF Metadata Inspection.

Detects editing-tool signatures left in JPEG EXIF metadata by software
such as Photoshop, GIMP, Lightroom, Canva, etc.

Note: A sophisticated forger can strip EXIF with exiftool. Missing EXIF
is NOT a positive signal  -  we only flag *presence* of editing-tool tags.
"""

import os
import sys
import time
import logging
from typing import Optional
from pathlib import Path

from PIL import Image, ExifTags

# Ensure repo root is on sys.path
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from shared.contracts import SignalResult, SignalId, Severity, SIGNAL_TRIGGER
from modules.decision.calibration import apply_calibration

logger = logging.getLogger(__name__)

EDITING_TOOL_KEYWORDS = [
    "photoshop",
    "gimp",
    "lightroom",
    "affinity",
    "paint.net",
    "snapseed",
    "pixelmator",
    "canva",
    "fotor",
    "adobe",
    "paint",
    "corel",
]

# EXIF tag IDs of interest
TAG_SOFTWARE = 0x0131  # Software used to process/create the image
TAG_MAKE = 0x010F  # Camera make
TAG_MODEL = 0x0110  # Camera model
TAG_DATETIME = 0x0132  # Date/time of last modification


def inspect_exif(image_path: str) -> dict:
    """
    Inspect EXIF metadata for signs of editing software.

    Args:
        image_path: Path to the JPEG/PNG document image.

    Returns:
        dict with keys:
          suspicious (bool)
          software   (str|None)   -  software tag value if present
          flags      (list[str])  -  list of suspicious findings
          raw_tags   (dict)       -  all readable EXIF tags
    """
    if not os.path.exists(image_path):
        return {
            "suspicious": False,
            "software": None,
            "flags": [],
            "raw_tags": {},
            "error": "File not found",
        }

    try:
        img = Image.open(image_path)
        exif = img.getexif()

        software = None
        make = None
        model = None
        raw_tags = {}

        if exif:
            for tag_id, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                raw_tags[tag_name] = str(value)
                if tag_id == TAG_SOFTWARE or tag_name == "Software":
                    software = _decode_tag(value)
                elif tag_id == TAG_MAKE or tag_name == "Make":
                    make = _decode_tag(value)
                elif tag_id == TAG_MODEL or tag_name == "Model":
                    model = _decode_tag(value)

        # Fallback to piexif if present
        if not software:
            try:
                import piexif

                exif_bytes = img.info.get("exif")
                if exif_bytes:
                    exif_dict = piexif.load(exif_bytes)
                    ifd0 = exif_dict.get("0th", {})
                    software = software or _decode_tag(ifd0.get(TAG_SOFTWARE))
                    make = make or _decode_tag(ifd0.get(TAG_MAKE))
                    model = model or _decode_tag(ifd0.get(TAG_MODEL))
            except Exception:
                pass

        flags = []
        if software and _is_editing_tool(software):
            flags.append(f"Editing software detected: '{software}'")

        if software and not make and not model:
            flags.append(
                f"Processed by software '{software}' without camera Make/Model tags"
            )

        return {
            "suspicious": len(flags) > 0,
            "software": software,
            "flags": flags,
            "raw_tags": raw_tags,
        }

    except Exception as e:
        logger.debug("EXIF extraction error on %s: %s", image_path, e)
        return {
            "suspicious": False,
            "software": None,
            "flags": [],
            "raw_tags": {},
            "error": str(e),
        }


def _decode_tag(value) -> Optional[str]:
    """Decode bytes or str EXIF tag value."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip("\x00")
    return str(value).strip()


def _is_editing_tool(software: str) -> bool:
    """Check if software string matches known editing tools."""
    sl = software.lower()
    return any(kw in sl for kw in EDITING_TOOL_KEYWORDS)


def score_document_exif(image_path: str) -> SignalResult:
    """
    Evaluates EXIF metadata for editing software signatures and returns SignalResult.
    IN:  image_path = path to document image.
    OUT: SignalResult(
           signal=SignalId.EXIF,
           raw_score=0.0 or 0.90,
           confidence=0.0 or 0.90,
           severity=Severity.SOFT,
           triggered=bool,
           label=str,
           evidence={...}
         )
    """
    t0 = time.perf_counter()
    res = inspect_exif(image_path)
    elapsed_ms = max(0.1, (time.perf_counter() - t0) * 1000.0)

    if res.get("error"):
        return SignalResult(
            signal=SignalId.EXIF,
            raw_score=0.0,
            confidence=0.0,
            severity=Severity.SOFT,
            triggered=False,
            label="Error inspecting EXIF metadata",
            evidence=res,
            ok=False,
            ms=elapsed_ms,
        )

    suspicious = res["suspicious"]
    raw_score = 0.90 if suspicious else 0.0
    confidence = apply_calibration(raw_score, "exif")
    trigger_thresh = SIGNAL_TRIGGER.get(SignalId.EXIF, 0.50)
    triggered = bool(raw_score > trigger_thresh or suspicious)

    label = (
        f"Editing tool metadata detected: {res.get('software') or 'software signature'}"
        if triggered
        else "Camera sensor metadata present or consistent"
    )

    return SignalResult(
        signal=SignalId.EXIF,
        raw_score=raw_score,
        confidence=confidence,
        severity=Severity.SOFT,
        triggered=triggered,
        label=label,
        evidence=res,
        ok=True,
        ms=elapsed_ms,
    )


if __name__ == "__main__":
    import tempfile

    tmp_dir = tempfile.gettempdir()
    test_img = os.path.join(tmp_dir, f"test_exif_{os.getpid()}.jpg")
    Image.new("RGB", (100, 100)).save(test_img)

    r = score_document_exif(test_img)
    assert r.signal == SignalId.EXIF
    assert r.severity == Severity.SOFT
    assert 0.0 <= r.raw_score <= 1.0
    print("EXIF INSPECT OK")
