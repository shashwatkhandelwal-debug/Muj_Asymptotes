"""
QR ↔ OCR Consistency Check.

Compares fields extracted from the Aadhaar Secure QR payload against
OCR-extracted printed fields. Defends against copy-paste attacks where
a genuine signed QR is applied to a fake card with different printed text.

A genuine UIDAI signature on the QR does NOT mean the printed fields match.
This check is the second line of defense after signature verification.

Missing fields (match is None) are treated as uncertain — not mismatches.
"""

from typing import Optional

from Levenshtein import distance as levenshtein_distance

# Maximum edit distance allowed for name fuzzy match
_NAME_MAX_EDIT_DISTANCE = 3


def check_qr_ocr_consistency(qr_fields: dict, ocr_fields: dict) -> dict:
    """
    Cross-check QR-decoded fields against OCR-printed fields.

    Args:
        qr_fields:  Fields parsed from the Aadhaar Secure QR payload.
        ocr_fields: Fields extracted by OCR from the printed card.

    Returns:
        dict with keys:
          consistent (bool|None), mismatches (list[str]), details (dict)
          consistent is None when no comparable fields are available.
    """
    mismatches = []
    details = {}
    compared = 0

    # Extract fields with fallback for nested/alternate keys
    qr_name = qr_fields.get("name") or qr_fields.get("name_en")
    ocr_name = ocr_fields.get("name") or ocr_fields.get("name_en") or (ocr_fields.get("fields", {}) if isinstance(ocr_fields.get("fields"), dict) else {}).get("name")

    qr_dob = qr_fields.get("dob")
    ocr_dob = ocr_fields.get("dob") or (ocr_fields.get("fields", {}) if isinstance(ocr_fields.get("fields"), dict) else {}).get("dob")

    qr_gender = qr_fields.get("gender")
    ocr_gender = ocr_fields.get("gender") or (ocr_fields.get("fields", {}) if isinstance(ocr_fields.get("fields"), dict) else {}).get("gender")

    # 1. Name match (fuzzy — OCR errors expected)
    name_result = _compare_name(qr_name, ocr_name)
    details["name"] = name_result
    if name_result["match"] is True:
        compared += 1
    elif name_result["match"] is False:
        compared += 1
        mismatches.append("name")

    # 2. Date of Birth (normalized exact match)
    dob_result = _compare_dob(qr_dob, ocr_dob)
    details["dob"] = dob_result
    if dob_result["match"] is True:
        compared += 1
    elif dob_result["match"] is False:
        compared += 1
        mismatches.append("dob")

    # 3. Gender (exact)
    gender_result = _compare_gender(qr_gender, ocr_gender)
    details["gender"] = gender_result
    if gender_result["match"] is True:
        compared += 1
    elif gender_result["match"] is False:
        compared += 1
        mismatches.append("gender")

    if compared == 0:
        return {
            "consistent": None,
            "mismatches": [],
            "details": details,
            "error": "insufficient_fields_for_comparison",
        }

    return {
        "consistent": len(mismatches) == 0,
        "mismatches": mismatches,
        "details": details,
    }


# ── Comparison helpers ─────────────────────────────────────────────────────────


def _compare_name(qr_name: Optional[str], ocr_name: Optional[str]) -> dict:
    if not qr_name or not ocr_name:
        return {"match": None, "reason": "missing data", "edit_distance": None}

    qr_norm = _normalize_name(qr_name)
    ocr_norm = _normalize_name(ocr_name)
    dist = levenshtein_distance(qr_norm, ocr_norm)
    match = dist <= _NAME_MAX_EDIT_DISTANCE or (qr_norm in ocr_norm) or (ocr_norm in qr_norm)

    return {
        "match": match,
        "qr_value": qr_name,
        "ocr_value": ocr_name,
        "edit_distance": dist,
        "threshold": _NAME_MAX_EDIT_DISTANCE,
    }


def _compare_dob(qr_dob: Optional[str], ocr_dob: Optional[str]) -> dict:
    if not qr_dob or not ocr_dob:
        return {"match": None, "reason": "missing data"}

    qr_norm = _normalize_date(qr_dob)
    ocr_norm = _normalize_date(ocr_dob)
    match = (qr_norm == ocr_norm) or (len(qr_norm) >= 4 and len(ocr_norm) >= 4 and qr_norm[-4:] == ocr_norm[-4:])

    return {"match": match, "qr_value": qr_dob, "ocr_value": ocr_dob}


def _compare_gender(qr_gender: Optional[str], ocr_gender: Optional[str]) -> dict:
    if not qr_gender or not ocr_gender:
        return {"match": None, "reason": "missing data"}

    qr_g = qr_gender.strip().upper()[0] if qr_gender.strip() else ""
    ocr_g = ocr_gender.strip().upper()[0] if ocr_gender.strip() else ""
    match = qr_g == ocr_g and qr_g in ("M", "F", "T")

    return {"match": match, "qr_value": qr_gender, "ocr_value": ocr_gender}


def _normalize_name(name: str) -> str:
    """Uppercase, strip extra spaces."""
    return " ".join(name.upper().split())


def _normalize_date(date_str: str) -> str:
    """Normalize DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD to digit form."""
    import re

    digits = re.sub(r"\D", "", date_str)
    if len(digits) == 8:
        # Prefer DDMMYYYY if day looks plausible; also accept YYYYMMDD
        dd_first = int(digits[0:2])
        if 1 <= dd_first <= 31:
            return digits  # DDMMYYYY
        return digits[6:8] + digits[4:6] + digits[0:4]  # YYYYMMDD → DDMMYYYY
    return digits
