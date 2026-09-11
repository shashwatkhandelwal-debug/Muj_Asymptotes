"""
Generic National Identity Card Adapter.

Primary engine: OpenBharatOCR (voter_id_front).
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


def _get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def _try_openbharatocr_voter(image: np.ndarray) -> Optional[dict]:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        temp_path = f.name
    try:
        cv2.imwrite(temp_path, image)
        import openbharatocr
        res = openbharatocr.voter_id_front(temp_path)
        if isinstance(res, dict) and any(res.values()):
            return res
        return None
    except Exception as e:
        logger.debug("OpenBharatOCR Voter ID extraction exception: %s", e)
        return None
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def extract_generic_id_fields(image: np.ndarray, reader=None) -> dict:
    if image is None or getattr(image, "size", 0) == 0:
        return {}

    # 1. Primary: OpenBharatOCR
    obo_res = _try_openbharatocr_voter(image)
    if obo_res:
        epic_num = obo_res.get("EPIC No") or obo_res.get("voter_id") or obo_res.get("Card No")
        name = obo_res.get("Name") or obo_res.get("name")
        dob = obo_res.get("DOB") or obo_res.get("dob") or obo_res.get("Age")
        if epic_num or name:
            return {
                "id_number": epic_num,
                "name": name,
                "dob": dob,
                "expiry_date": None,
                "raw_text": str(obo_res),
                "confidence": 0.93,
                "ocr_engine": "openbharatocr",
            }

    # 2. Fallback: EasyOCR
    ocr_reader = reader or _get_reader()
    results = ocr_reader.readtext(image, detail=0)
    full_text = "\n".join(results)

    id_number = _extract_id_number(results, full_text)
    name = _extract_name(results)
    dob = _extract_date(full_text, ["dob", "birth", "date of birth"])
    expiry = _extract_date(full_text, ["expiry", "valid until", "valide"])
    if expiry == dob:
        expiry = None

    return {
        "id_number": id_number,
        "name": name,
        "dob": dob,
        "expiry_date": expiry,
        "raw_text": full_text,
        "confidence": 0.85,
        "ocr_engine": "easyocr",
    }


def _extract_id_number(lines: list, full_text: str) -> Optional[str]:
    # Pattern 1: Modern 10-char EPIC (3 letters + 7 digits, e.g. TDF1928374 or TDF 1928374)
    m = re.search(r"\b([A-Z]{3})\s*(\d{7})\b", full_text.upper())
    if m:
        return m.group(1) + m.group(2)

    # Pattern 2: Old 4-part state format (e.g. DL/01/001/012345 or WB/01/001/000123)
    m = re.search(r"\b([A-Z]{2}\s*\/\s*\d{1,3}\s*\/\s*\d{1,4}\s*\/\s*\d{4,7})\b", full_text.upper())
    if m:
        return re.sub(r"\s+", "", m.group(1))

    # Pattern 3: Fallback 2-4 letters + 6-8 digits excluding common header words
    EXCLUDE = {
        "IDENTITY", "ELECTION", "COMMISSION", "GOVERNMENT", "NATIONAL",
        "CARD", "INDIA", "ELECTOR", "ELECTORS", "VOTER", "FATHER", "REPUBLIC"
    }
    for line in lines:
        for token in re.findall(r"\b[A-Z0-9/-]{6,16}\b", line.upper()):
            if token in EXCLUDE:
                continue
            # Must contain both letters and digits
            if re.search(r"[A-Z]", token) and re.search(r"\d", token):
                return token
    return None


def _extract_name(lines: list) -> Optional[str]:
    EXCLUDE = {
        "ELECTION", "COMMISSION", "IDENTITY", "CARD", "GOVERNMENT", "NATIONAL",
        "REPUBLIC", "CITIZEN", "INDIA", "BHARAT", "FATHER", "HUSBAND", "MOTHER",
        "SEX", "AGE", "DOB", "ADDRESS", "MALE", "FEMALE", "PHOTO", "ELECTOR",
        "ELECTORS", "NAME"
    }
    
    # 1. Search for explicit 'Name' / 'Elector Name' prefix with actual name attached
    for i, line in enumerate(lines):
        m = re.search(r"(?:ELECTOR[\'S]*\s*NAME|ELECTOR|NAME)\s*[:\-\s]+\s*([A-Za-z\s\.]+)", line, re.IGNORECASE)
        if m:
            cand = re.sub(r"[^A-Za-z\s]", "", m.group(1)).strip()
            words = [w for w in cand.split() if w.upper() not in EXCLUDE]
            if len(words) >= 1 and all(len(w) >= 2 for w in words):
                return " ".join(words)
        
        # If the line is literally just "Elector's Name :" or "Elector Name", check next line
        if re.search(r"^(?:ELECTOR[\'S]*\s*NAME|NAME)\s*[:\-\s]*$", line.strip(), re.IGNORECASE):
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                cand = re.sub(r"[^A-Za-z\s]", "", next_line).strip()
                words = [w for w in cand.split() if w.upper() not in EXCLUDE]
                if len(words) >= 1 and all(len(w) >= 2 for w in words):
                    return " ".join(words)

    # 2. Search clean alphabetic lines (reject noise symbols/Devanagari OCR artifacts)
    for line in lines:
        clean_letters = re.findall(r"[A-Za-z]", line)
        if len(clean_letters) < 4:
            continue
        ratio = len(clean_letters) / max(1, len(line.strip()))
        if ratio < 0.75:  # Skip noisy non-Latin artifact lines
            continue
        words = [w for w in re.findall(r"[A-Za-z]+", line) if len(w) >= 2 and w.upper() not in EXCLUDE]
        if len(words) >= 2:
            return " ".join(words)
    return None


def _extract_date(text: str, keywords: list) -> Optional[str]:
    for kw in keywords:
        m = re.search(rf"(?:{kw})[:\s]+(\d{{2}}[\/\-\.]\d{{2}}[\/\-\.]\d{{4}})", text, re.IGNORECASE)
        if m:
            return m.group(1)
    # Generic date fallback
    m = re.search(r"\b(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})\b", text)
    if m:
        return m.group(1)
    return None
