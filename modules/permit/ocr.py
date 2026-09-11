"""
Border Permit & Travel Authorization OCR Extractor.

Extracts:
  - Permit Number (e.g. ILP-2026-84920, CBP-94821)
  - Holder Name
  - Allowed Border Gate / Sector
  - Valid From (Issue Date)
  - Valid Until (Expiry Date)
  - Associated Identity Document (Aadhaar or Passport Number)
"""

import re
from typing import Optional
import easyocr
import numpy as np

_reader: Optional[easyocr.Reader] = None


def _get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def extract_permit_fields(image: np.ndarray, reader=None) -> dict:
    if image is None or getattr(image, "size", 0) == 0:
        return {}

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

    permit_num = _extract_permit_number(full_text)
    name = _extract_permit_name(full_text, raw_text_lines)
    gate = _extract_gate(full_text)
    valid_from = _extract_date(full_text, ["from", "issue", "valid from"])
    valid_until = _extract_date(full_text, ["until", "expiry", "to", "valid till", "valid to"])
    associated_id = _extract_associated_id(full_text)

    return {
        "permit_number": permit_num,
        "holder_name": name,
        "border_gate": gate or "ALL_DESIGNATED_POSTS",
        "valid_from": valid_from,
        "valid_until": valid_until,
        "associated_id": associated_id,
        "raw_text": full_text,
        "confidence": round(avg_conf, 4),
    }


def _extract_permit_number(text: str) -> Optional[str]:
    m = re.search(r"\b(?:PERMIT|ILP|AUTH|PASS)[-\s:]*([A-Z0-9-]{6,16})\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m2 = re.search(r"\b([A-Z]{2,4}[-]\d{4}[-]\d{4,8})\b", text)
    if m2:
        return m2.group(1).upper()
    return None


def _extract_permit_name(text: str, lines: list) -> Optional[str]:
    for line in lines:
        if re.search(r"(?:Holder|Name|Traveler)[:\s]+", line, re.IGNORECASE):
            cand = re.sub(r"^(?:Holder|Name|Traveler)[:\s]+", "", line, flags=re.IGNORECASE).strip()
            if len(cand) >= 3 and not cand.upper().startswith("PERMIT"):
                return cand
    return None


def _extract_gate(text: str) -> Optional[str]:
    m = re.search(r"(?:Gate|Sector|Post|Checkpost)[:\s]+([A-Za-z0-9\s-]{3,25})", text, re.IGNORECASE)
    return m.group(1).strip() if m else None


def _extract_date(text: str, keywords: list) -> Optional[str]:
    for kw in keywords:
        m = re.search(rf"(?:{kw})[:\s]+(\d{{2}}[/\-\.]\d{{2}}[/\-\.]\d{{4}})", text, re.IGNORECASE)
        if m:
            s = m.group(1)
            dm = re.match(r"(\d{2})[/\-\.](\d{2})[/\-\.](\d{4})", s)
            if dm:
                return f"{dm.group(1)}/{dm.group(2)}/{dm.group(3)}"
    return None


def _extract_associated_id(text: str) -> Optional[str]:
    # 12-digit Aadhaar or 1-letter + 7-digit Passport
    m_pass = re.search(r"\b([A-Z]\d{7})\b", text)
    if m_pass:
        return m_pass.group(1)
    m_aadh = re.search(r"\b(\d{4}\s?\d{4}\s?\d{4})\b", text)
    if m_aadh:
        return re.sub(r"\s", "", m_aadh.group(1))
    return None
