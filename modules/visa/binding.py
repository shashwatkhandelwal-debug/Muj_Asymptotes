"""
Visa ↔ Passport Binding Check.

Compares the passport number extracted from the visa stamp against
the passport number from the MRZ of the presented passport.

Attack this defends against:
  A genuine visa stamp (issued for Passport A) is presented together
  with Passport B. The visa itself is real and internally consistent.
  Only by cross-referencing the visa's stated passport number against
  the actual passport's MRZ number is the mismatch detected.

This is the direct defense against "tampered visa stamps" named in the PS.
"""

import re
from typing import Optional

# Characters commonly confused by OCR on visa stamps
_OCR_SUBSTITUTIONS = {"O": "0", "0": "O", "I": "1", "1": "I", "l": "1"}


def check_visa_passport_binding(visa_fields: dict, mrz_fields: dict) -> dict:
    """
    Check that the visa's stated passport number matches the presented passport.

    Args:
        visa_fields: Output from modules.visa.ocr.extract_visa_fields()
        mrz_fields:  Output from modules.passport.mrz.parse_mrz()

    Returns:
        dict with keys:
          bound (bool)            -  True if passport numbers match
          visa_passport_num (str)
          mrz_passport_num  (str)
          normalized_match  (bool)
          score (float)
          error (str|None)
    """
    visa_pn = visa_fields.get("passport_number")
    mrz_pn = mrz_fields.get("passport_number")

    if not visa_pn:
        return {
            "bound": None,
            "error": "Passport number not found on visa stamp (OCR confidence too low)",
            "score": 0.5,
        }

    if not mrz_pn:
        return {
            "bound": None,
            "error": "Passport number not available from MRZ",
            "score": 0.5,
        }

    # Exact match first
    visa_norm = _normalize_passport_number(visa_pn)
    mrz_norm = _normalize_passport_number(mrz_pn)

    exact_match = visa_norm == mrz_norm

    # OCR-tolerant match (O↔0, I↔1 substitutions)
    tolerant_match = _tolerant_compare(visa_norm, mrz_norm)

    bound = exact_match or tolerant_match

    return {
        "bound": bound,
        "visa_passport_number": visa_pn,
        "mrz_passport_number": mrz_pn,
        "normalized_visa": visa_norm,
        "normalized_mrz": mrz_norm,
        "exact_match": exact_match,
        "tolerant_match": tolerant_match,
        "score": 1.0 if bound else 0.0,
        "error": (
            None
            if bound
            else "Passport number on visa does not match presented passport MRZ"
        ),
    }


# ── Helpers ────────────────────────────────────────────────────────────────────


def _normalize_passport_number(pn: str) -> str:
    """Uppercase, strip spaces/dashes/fillers."""
    return re.sub(r"[\s\-<]", "", pn.upper())


def _tolerant_compare(a: str, b: str) -> bool:
    """
    Compare two passport number strings tolerating O/0 and I/1 confusion.
    Both strings must have the same length after normalization.
    """
    if len(a) != len(b):
        return False
    for ca, cb in zip(a, b):
        if ca != cb:
            # Check if it's a known OCR substitution pair
            if _OCR_SUBSTITUTIONS.get(ca) != cb and _OCR_SUBSTITUTIONS.get(cb) != ca:
                return False
    return True
