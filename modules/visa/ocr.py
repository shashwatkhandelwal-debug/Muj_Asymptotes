"""
Visa OCR  -  Extract fields from visa sticker stamp image.

Extracts:
  - Visa number
  - Visa type (Tourist / Business / Employment / Student / Transit)
  - Date of issue
  - Date of expiry (last entry valid date)
  - Duration of stay (days)
  - Number of entries (Single / Double / Multiple)
  - Passport number the visa is issued against
  - Applicant name
"""

import os
import re
from typing import Optional

import easyocr
import numpy as np

_reader: Optional[easyocr.Reader] = None


def _clean_corrupt_easyocr_models():
    """Remove partial or corrupted model files from EasyOCR cache directory."""
    try:
        model_dir = os.path.expanduser("~/.EasyOCR/model")
        if os.path.exists(model_dir):
            for fname in os.listdir(model_dir):
                fpath = os.path.join(model_dir, fname)
                if os.path.isfile(fpath) and (fname.endswith(".pth") or fname.endswith(".tmp")):
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass
    except Exception:
        pass


import cv2


def _preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Multi-stage contrast, scaling and edge enhancement for OCR."""
    if image is None or image.size == 0:
        return image

    if len(image.shape) == 2:
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img = image.copy()

    h, w = img.shape[:2]
    target_w = 1280
    if w > 0 and w != target_w:
        scale = target_w / float(w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(
            img,
            (new_w, new_h),
            interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA,
        )

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced_lab = cv2.merge((l_enhanced, a, b))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    gaussian = cv2.GaussianBlur(enhanced_bgr, (0, 0), 2.0)
    unsharp = cv2.addWeighted(enhanced_bgr, 1.4, gaussian, -0.4, 0)
    return unsharp


def _get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        try:
            _reader = easyocr.Reader(["en"], gpu=False)
        except (AssertionError, Exception):
            _clean_corrupt_easyocr_models()
            _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def extract_visa_fields(image: np.ndarray, reader: Optional[easyocr.Reader] = None) -> dict:
    """
    Extract all relevant fields from a visa stamp image.

    Args:
        image: BGR numpy array of the visa stamp.
        reader: Optional EasyOCR reader instance.

    Returns:
        dict with extracted fields and per-field confidence.
    """
    if reader is None:
        reader = _get_reader()
    enhanced = _preprocess_for_ocr(image)
    results = reader.readtext(enhanced, detail=1)
    raw_text = "\n".join([r[1] for r in results])
    confidences = [r[2] for r in results]
    avg_conf = float(np.mean(confidences)) if confidences else 0.0

    return {
        "visa_number": _extract_visa_number(raw_text),
        "visa_type": _extract_visa_type(raw_text),
        "date_of_issue": _extract_date(
            raw_text, ["date of issue", "issued on", "issue date"]
        ),
        "date_of_expiry": _extract_date(
            raw_text, ["valid until", "expiry", "valid till", "date of expiry"]
        ),
        "duration_days": _extract_duration(raw_text),
        "num_entries": _extract_entries(raw_text),
        "passport_number": _extract_passport_number(raw_text),
        "applicant_name": _extract_name(raw_text),
        "raw_text": raw_text,
        "confidence": round(avg_conf, 4),
    }


# ── Extraction helpers ─────────────────────────────────────────────────────────


def _extract_visa_number(text: str) -> Optional[str]:
    """Visa numbers are typically alphanumeric, 8–14 chars."""
    match = re.search(
        r"(?:visa\s*no|visa\s*number)[:\s]+([A-Z0-9]{6,16})", text, re.IGNORECASE
    )
    if match:
        return match.group(1).strip()
    # Fallback: look for standalone alphanumeric code
    match = re.search(r"\b([A-Z]{1,3}\d{6,10})\b", text)
    return match.group(1) if match else None


def _extract_visa_type(text: str) -> Optional[str]:
    """Match against known Indian visa type keywords."""
    types = [
        "tourist",
        "business",
        "employment",
        "student",
        "transit",
        "medical",
        "conference",
        "journalist",
        "research",
        "entry",
    ]
    text_lower = text.lower()
    for vtype in types:
        if vtype in text_lower:
            return vtype.title()
    return None


def _extract_date(text: str, labels: list) -> Optional[str]:
    for label in labels:
        pattern = rf"(?:{label})[:\s]+(\d{{2}}[/\-\.]\d{{2}}[/\-\.]\d{{4}}|\d{{2}}\s+\w{{3}}\s+\d{{4}})"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _normalize_date(match.group(1))
    # Fallback: any date-like pattern near the label
    for label in labels:
        idx = text.lower().find(label.split()[0])
        if idx != -1:
            snippet = text[idx : idx + 50]
            dm = re.search(r"(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})", snippet)
            if dm:
                return _normalize_date(dm.group(1))
    return None


def _extract_duration(text: str) -> Optional[int]:
    """Extract stay duration in days (e.g. '90 days', 'Duration: 180')."""
    match = re.search(r"(\d{1,4})\s*(?:days?|d\b)", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"(?:duration|stay)[:\s]+(\d{1,4})", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _extract_entries(text: str) -> Optional[str]:
    text_lower = text.lower()
    if "multiple" in text_lower:
        return "Multiple"
    if "double" in text_lower or "two" in text_lower:
        return "Double"
    if "single" in text_lower or "one" in text_lower:
        return "Single"
    return None


def _extract_passport_number(text: str) -> Optional[str]:
    """Indian passport number: 1 letter + 7 digits."""
    match = re.search(r"\b([A-Z]\d{7})\b", text)
    return match.group(1) if match else None


def _extract_name(text: str) -> Optional[str]:
    match = re.search(
        r"(?:name|applicant)[:\s]+([A-Z][A-Za-z\s]{2,40})", text, re.IGNORECASE
    )
    return match.group(1).strip() if match else None


def _normalize_date(s: str) -> str:
    s = s.strip()
    m = re.match(r"(\d{2})[/\-\.](\d{2})[/\-\.](\d{4})", s)
    if m:
        return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
    MONTHS = {
        "jan": "01",
        "feb": "02",
        "mar": "03",
        "apr": "04",
        "may": "05",
        "jun": "06",
        "jul": "07",
        "aug": "08",
        "sep": "09",
        "oct": "10",
        "nov": "11",
        "dec": "12",
    }
    m2 = re.match(r"(\d{2})\s+([A-Za-z]{3})\s+(\d{4})", s)
    if m2:
        mon = MONTHS.get(m2.group(2).lower(), "00")
        return f"{m2.group(1)}/{mon}/{m2.group(3)}"
    return s
