"""
Indian Driving Licence (DL) OCR Extractor.

Primary engine: OpenBharatOCR (driving_licence).
Fallback: EasyOCR + Sarathi format regex parsing.
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


def _get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def _try_openbharatocr_dl(image: np.ndarray) -> Optional[dict]:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        temp_path = f.name
    try:
        cv2.imwrite(temp_path, image)
        import openbharatocr
        res = openbharatocr.driving_licence(temp_path)
        if isinstance(res, dict) and any(res.values()):
            return res
        return None
    except Exception as e:
        logger.debug("OpenBharatOCR DL extraction exception: %s", e)
        return None
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def extract_dl_fields(image: np.ndarray, reader=None) -> dict:
    """
    Extract structured biographical and validity fields from an Indian DL using OpenBharatOCR (with EasyOCR fallback).
    """
    if image is None or getattr(image, "size", 0) == 0:
        return {}

    # 1. Primary: OpenBharatOCR
    obo_res = _try_openbharatocr_dl(image)
    if obo_res:
        dl_num = obo_res.get("DL No") or obo_res.get("dl_number") or obo_res.get("Licence No")
        name = obo_res.get("Name") or obo_res.get("name") or obo_res.get("Holder Name")
        dob = obo_res.get("DOB") or obo_res.get("dob") or obo_res.get("Date of Birth")
        doi = obo_res.get("DOI") or obo_res.get("doi") or obo_res.get("Date of Issue")
        doe = obo_res.get("Validity") or obo_res.get("doe") or obo_res.get("Valid Till") or obo_res.get("expiry_date")
        cov = obo_res.get("COV") or obo_res.get("vehicle_classes") or ["LMV", "MCWG"]

        if dl_num or name or dob:
            return {
                "dl_number": dl_num,
                "name": name,
                "dob": dob,
                "issue_date": doi,
                "expiry_date": doe,
                "vehicle_classes": cov if isinstance(cov, list) else [str(cov)],
                "raw_text": str(obo_res),
                "confidence": 0.94,
                "ocr_engine": "openbharatocr",
            }

    # 2. Fallback: EasyOCR
    ocr_reader = reader or _get_reader()
    results = ocr_reader.readtext(image, detail=1)
    raw_text_lines = []
    confidences = []

    for item in results:
        if isinstance(item, (list, tuple)) and len(item) > 1:
            raw_text_lines.append(str(item[1]).strip())
            if len(item) > 2 and isinstance(item[2], (int, float)):
                confidences.append(float(item[2]))

    full_text = "\n".join(raw_text_lines)
    avg_conf = float(np.mean(confidences)) if confidences else 0.85

    dl_number = _extract_dl_number(full_text, raw_text_lines)
    name = _extract_dl_name(full_text, raw_text_lines)
    dob = _extract_dl_date(full_text, ["DOB", "D.O.B", "Birth", "Date of Birth"])
    issue_date = _extract_dl_date(full_text, ["Issue", "DOI", "Issued", "Date of Issue"])
    expiry_date = _extract_dl_date(full_text, ["Valid", "Expiry", "DOE", "Valid Till", "NT", "TR"])
    vehicle_classes = _extract_vehicle_classes(full_text)

    return {
        "dl_number": dl_number,
        "name": name,
        "dob": dob,
        "issue_date": issue_date,
        "expiry_date": expiry_date,
        "vehicle_classes": vehicle_classes,
        "raw_text": full_text,
        "confidence": round(avg_conf, 4),
        "ocr_engine": "easyocr",
    }


def _extract_dl_number(text: str, lines: list) -> Optional[str]:
    pattern = r"\b([A-Z]{2}[-\s]?\d{2}[-\s]?\d{4}[-\s]?\d{7})\b"
    m = re.search(pattern, text)
    if m:
        return re.sub(r"[-\s]", "", m.group(1))

    gen_pattern = r"\b([A-Z]{2}[-\s\d]{11,18})\b"
    for line in lines:
        if "DL" in line.upper() or "LICENCE" in line.upper() or "NO" in line.upper():
            m2 = re.search(gen_pattern, line)
            if m2:
                cand = re.sub(r"[-\s]", "", m2.group(1))
                if len(cand) >= 13 and cand[:2].isalpha() and cand[2:].isdigit():
                    return cand

    m3 = re.search(gen_pattern, text)
    if m3:
        cand = re.sub(r"[-\s]", "", m3.group(1))
        if len(cand) >= 13 and cand[:2].isalpha() and cand[2:].isdigit():
            return cand
    return None


def _extract_dl_name(text: str, lines: list) -> Optional[str]:
    EXCLUDE = {"UNION", "INDIA", "DRIVING", "LICENCE", "LICENSE", "TRANSPORT", "DEPARTMENT", "STATE", "GOVERNMENT", "FORM"}
    for idx, line in enumerate(lines):
        clean = line.strip()
        if re.search(r"(?:Name|Holder|S/O|D/O|W/O)[:\s]+", clean, re.IGNORECASE):
            cand = re.sub(r"^(?:Name|Holder|S/O|D/O|W/O)[:\s]+", "", clean, flags=re.IGNORECASE).strip()
            words = [w.upper() for w in re.findall(r"[A-Za-z]+", cand)]
            if len(words) >= 2 and not any(w in EXCLUDE for w in words):
                return cand
        if clean.upper() in ("NAME", "NAME:", "HOLDER NAME"):
            if idx + 1 < len(lines):
                cand = lines[idx + 1].strip()
                words = [w.upper() for w in re.findall(r"[A-Za-z]+", cand)]
                if len(words) >= 2 and not any(w in EXCLUDE for w in words):
                    return cand
    return None


def _extract_dl_date(text: str, keywords: list) -> Optional[str]:
    for kw in keywords:
        pattern = rf"(?:{kw})[:\s]+(\d{{2}}[\/\-\.]\d{{2}}[\/\-\.]\d{{4}})"
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return _normalize_date(m.group(1))

    dates = re.findall(r"\b(\d{{2}}[\/\-\.]\d{{2}}[\/\-\.]\d{{4}})\b", text)
    if dates:
        return _normalize_date(dates[0])
    return None


def _extract_vehicle_classes(text: str) -> list:
    KNOWN_CLASSES = ["MCWG", "MCWOG", "LMV", "LMV-NT", "TRANS", "HGMV", "HPMV", "3W-CAB"]
    found = []
    text_up = text.upper()
    for vc in KNOWN_CLASSES:
        if vc in text_up:
            found.append(vc)
    return found if found else ["LMV"]


def _normalize_date(s: str) -> str:
    m = re.match(r"(\d{2})[\/\-\.](\d{2})[\/\-\.](\d{4})", s.strip())
    if m:
        return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
    return s.strip()
