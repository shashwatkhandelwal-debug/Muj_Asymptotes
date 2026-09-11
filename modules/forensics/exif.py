"""
EXIF Metadata Inspection.

Detects editing-tool signatures left in JPEG EXIF metadata by software
such as Photoshop, GIMP, Lightroom, etc.

Note: A sophisticated forger can strip EXIF with exiftool. Missing EXIF
is NOT a positive signal  -  we only flag *presence* of editing-tool tags.
"""

from typing import Optional

import piexif
from PIL import Image

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
    try:
        img = Image.open(image_path)
        exif_bytes = img.info.get("exif")
        if not exif_bytes:
            return {
                "suspicious": False,
                "software": None,
                "flags": [],
                "raw_tags": {},
                "note": "No EXIF data present",
            }

        exif_dict = piexif.load(exif_bytes)
        ifd0 = exif_dict.get("0th", {})

        software = _decode_tag(ifd0.get(TAG_SOFTWARE))
        make = _decode_tag(ifd0.get(TAG_MAKE))
        model = _decode_tag(ifd0.get(TAG_MODEL))

        flags = []
        if software and _is_editing_tool(software):
            flags.append(f"Editing software detected: '{software}'")

        # If software tag overrides make/model (image saved by software, not camera)
        if software and not make and not model:
            flags.append(
                "Software tag present but no camera Make/Model  -  image processed by software"
            )

        raw_tags = {
            "Software": software,
            "Make": make,
            "Model": model,
        }

        return {
            "suspicious": len(flags) > 0,
            "software": software,
            "flags": flags,
            "raw_tags": raw_tags,
        }

    except Exception as e:
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
