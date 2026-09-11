"""
MRZ <-> VIZ Consistency Check.

Cross-checks fields extracted from the MRZ (machine-readable zone)
against OCR'd VIZ (visual inspection zone) fields.

Defends against: generating a valid MRZ for a fake identity and pasting
it over a genuine biographical page  -  the printed VIZ fields won't match.

Also catches: altering the printed DOB/expiry while leaving the MRZ intact,
since the MRZ check digits would still be valid but the fields now disagree.

LIMITATION: Live cross-referencing against a central government passport database
or INTERPOL's SLTD (Stolen and Lost Travel Documents) database is NOT implemented.
This requires institutional access (diplomatic/immigration API integration) that is
outside the current scope. This module is an architected extension point for such lookups.
"""

from datetime import datetime
from typing import Optional

from Levenshtein import distance as levenshtein_distance

_NAME_EDIT_THRESHOLD = 4  # Allows for OCR noise in longer names
_DATE_EXACT = True  # Dates must match exactly after normalization


def check_mrz_viz_consistency(mrz: dict, viz: dict) -> dict:
    """
    Compare MRZ-parsed fields against VIZ OCR fields.

    Args:
        mrz: Output from modules.passport.mrz.parse_mrz()
        viz: Output from modules.passport.viz.extract_viz_fields()

    Returns:
        dict with keys: consistent (bool), mismatches (list[str]), details (dict)
    """
    mismatches = []
    details = {}

    # 1. Surname
    sn = _compare_name(mrz.get("surname"), viz.get("surname"), "surname")
    details["surname"] = sn
    if sn.get("match") is False:
        mismatches.append("surname")

    # 2. Given names
    gn = _compare_name(mrz.get("given_names"), viz.get("given_names"), "given_names")
    details["given_names"] = gn
    if gn.get("match") is False:
        mismatches.append("given_names")

    # 3. Date of birth
    dob = _compare_date(mrz.get("dob"), viz.get("dob"), "dob")
    details["dob"] = dob
    if dob.get("match") is False:
        mismatches.append("dob")

    # 4. Date of expiry
    doe = _compare_date(mrz.get("expiry"), viz.get("doe"), "expiry")
    details["expiry"] = doe
    if doe.get("match") is False:
        mismatches.append("expiry")

    # 5. Passport number
    pn = _compare_exact(
        mrz.get("passport_number"), viz.get("passport_number"), "passport_number"
    )
    details["passport_number"] = pn
    if pn.get("match") is False:
        mismatches.append("passport_number")

    return {
        "consistent": len(mismatches) == 0,
        "mismatches": mismatches,
        "details": details,
    }


# ── Comparison helpers ─────────────────────────────────────────────────────────


def _compare_name(mrz_val: Optional[str], viz_val: Optional[str], field: str) -> dict:
    if not mrz_val or not viz_val:
        return {"match": None, "reason": "missing data", "field": field}
    m = " ".join(mrz_val.upper().split())
    v = " ".join(viz_val.upper().split())
    dist = levenshtein_distance(m, v)
    return {
        "match": dist <= _NAME_EDIT_THRESHOLD,
        "mrz_value": mrz_val,
        "viz_value": viz_val,
        "edit_distance": dist,
        "field": field,
    }


def _compare_date(mrz_val: Optional[str], viz_val: Optional[str], field: str) -> dict:
    if not mrz_val or not viz_val:
        return {"match": None, "reason": "missing data", "field": field}
    m = _strip_date(mrz_val)
    v = _strip_date(viz_val)
    return {
        "match": m == v,
        "mrz_value": mrz_val,
        "viz_value": viz_val,
        "normalized_mrz": m,
        "normalized_viz": v,
        "field": field,
    }


def _compare_exact(mrz_val: Optional[str], viz_val: Optional[str], field: str) -> dict:
    if not mrz_val or not viz_val:
        return {"match": None, "reason": "missing data", "field": field}
    match = mrz_val.strip().upper() == viz_val.strip().upper()
    return {"match": match, "mrz_value": mrz_val, "viz_value": viz_val, "field": field}


def _strip_date(d: str) -> str:
    """Remove separators, return 8-digit string DDMMYYYY."""
    import re

    return re.sub(r"\D", "", d)
