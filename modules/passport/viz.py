"""
Passport Visual Inspection Zone (VIZ) OCR Extractor.

Primary engine: OpenBharatOCR (passport).
Fallback: EasyOCR + regex pattern matching.
"""

import logging
import os
import re
import tempfile
from typing import Optional

import cv2
import easyocr
import numpy as np

logger = logging.getLogger(__name__)
_reader: Optional[easyocr.Reader] = None


def _clean_corrupt_easyocr_models():
    try:
        model_dir = os.path.expanduser("~/.EasyOCR/model")
        if os.path.exists(model_dir):
            for fname in os.listdir(model_dir):
                if fname.endswith(".pth") or fname.endswith(".py"):
                    fpath = os.path.join(model_dir, fname)
                    if os.path.getsize(fpath) < 1000:
                        os.remove(fpath)
    except Exception:
        pass


def _get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        try:
            _reader = easyocr.Reader(["en"], gpu=False)
        except Exception:
            _clean_corrupt_easyocr_models()
            _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def _try_openbharatocr_passport(image: np.ndarray) -> Optional[dict]:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        temp_path = f.name
    try:
        cv2.imwrite(temp_path, image)
        import openbharatocr
        res = openbharatocr.passport(temp_path)
        if isinstance(res, dict) and any(res.values()):
            return res
        return None
    except Exception as e:
        logger.debug("OpenBharatOCR passport extraction exception: %s", e)
        return None
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def extract_viz_fields(image: np.ndarray, reader: Optional[easyocr.Reader] = None) -> dict:
    """
    Extract VIZ fields from passport biographical page using OpenBharatOCR (with EasyOCR fallback).
    """
    if image is None or getattr(image, "size", 0) == 0:
        return {}

    # 1. Primary: OpenBharatOCR
    obo_res = _try_openbharatocr_passport(image)
    if obo_res:
        p_num = obo_res.get("Passport Number") or obo_res.get("passport_number") or obo_res.get("Passport No")
        s_name = obo_res.get("Surname") or obo_res.get("surname") or obo_res.get("Last Name")
        g_name = obo_res.get("Given Name") or obo_res.get("given_name") or obo_res.get("First Name")
        dob = obo_res.get("Date of Birth") or obo_res.get("dob")
        doi = obo_res.get("Date of Issue") or obo_res.get("doi")
        doe = obo_res.get("Date of Expiry") or obo_res.get("doe") or obo_res.get("expiry")
        nat = obo_res.get("Nationality") or obo_res.get("nationality") or "IND"

        if p_num or s_name or g_name or dob:
            return {
                "surname": s_name,
                "given_names": g_name,
                "dob": dob,
                "doi": doi,
                "doe": doe,
                "pob": obo_res.get("Place of Birth") or obo_res.get("pob"),
                "passport_number": p_num,
                "nationality": nat,
                "raw_text": str(obo_res),
                "confidence": 0.92,
                "ocr_engine": "openbharatocr",
            }

    # 2. Fallback: EasyOCR
    ocr_reader = reader or _get_reader()
    enhanced = _preprocess_for_ocr(image)
    results = ocr_reader.readtext(enhanced, detail=1)
    raw_text = "\n".join([r[1] for r in results])
    confidences = [r[2] for r in results if len(r) > 2]
    avg_conf = float(np.mean(confidences)) if confidences else 0.0

    return {
        "surname": _extract_surname(raw_text),
        "given_names": _extract_given_names(raw_text),
        "dob": _extract_date_field(raw_text, ["date of birth", "birth", "dob"]),
        "doi": _extract_date_field(raw_text, ["date of issue", "issue"]),
        "doe": _extract_date_field(raw_text, ["date of expiry", "expiry", "valid till"]),
        "pob": _extract_pob(raw_text),
        "passport_number": _extract_passport_number(raw_text),
        "nationality": _extract_nationality(raw_text),
        "raw_text": raw_text,
        "confidence": round(avg_conf, 4),
        "ocr_engine": "easyocr",
    }


def _preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(denoised)


def _extract_surname(text: str) -> Optional[str]:
    match = re.search(r"(?:surname|last\s*name)[:\s]+([A-Z][A-Za-z\s\-]+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else None


def _extract_given_names(text: str) -> Optional[str]:
    match = re.search(r"(?:given\s*name[s]?|first\s*name)[:\s]+([A-Z][A-Za-z\s\-]+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else None


def _extract_date_field(text: str, keywords: list) -> Optional[str]:
    for kw in keywords:
        pattern = rf"(?:{kw})[:\s]+(\d{{2}}[\/\-\.]\d{{2}}[\/\-\.]\d{{4}})"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            raw = match.group(1)
            parts = re.split(r"[\/\-\.]", raw)
            if len(parts) == 3:
                return f"{parts[0]}/{parts[1]}/{parts[2]}"
    return None


def _extract_pob(text: str) -> Optional[str]:
    match = re.search(r"(?:place\s*of\s*birth|pob)[:\s]+([A-Z][A-Za-z\s,]+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else None


def _extract_passport_number(text: str) -> Optional[str]:
    match = re.search(r"\b([A-Z][0-9]{7})\b", text)
    return match.group(1) if match else None


def _extract_nationality(text: str) -> Optional[str]:
    match = re.search(r"(?:nationality)[:\s]+([A-Z]{3}|[A-Za-z]+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else None
