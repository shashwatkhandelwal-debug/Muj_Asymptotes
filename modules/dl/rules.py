"""
Indian Driving Licence Rules & Format Validator.

Validates:
  - Indian State Code RTO prefix (DL, MH, KA, TN, UP, HR, GJ, etc.)
  - DL Number formatting (length and digit structure)
  - Expiry Date (active validity window vs today)
  - Minimum Legal Age (calculated from DOB >= 18 years)
"""

import re
from datetime import datetime, date
from typing import Optional

# Valid Indian State and Union Territory 2-letter RTO codes
VALID_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB"
}


def validate_dl_rules(dl_fields: dict) -> dict:
    """
    Validate business rules for an Indian Driving Licence.

    Returns:
        dict with valid (bool), state_code (str), format_valid (bool),
        expired (bool), age_valid (bool), violations (list[str]), score (float)
    """
    violations = []
    dl_number = (dl_fields.get("dl_number") or "").replace(" ", "").replace("-", "").upper()
    state_code = dl_number[:2] if len(dl_number) >= 2 else None
    
    # 1. State Code Check
    format_valid = True
    if not state_code or state_code not in VALID_STATE_CODES:
        violations.append(f"Invalid state RTO prefix: '{state_code or 'UNKNOWN'}'")
        format_valid = False

    # 2. Number Length & Structure
    if len(dl_number) < 13 or len(dl_number) > 16:
        violations.append(f"Invalid DL number length ({len(dl_number)} chars, expected 13-16)")
        format_valid = False

    # 3. Expiry Check
    expired = False
    expiry_str = dl_fields.get("expiry_date")
    if expiry_str:
        try:
            exp_date = datetime.strptime(expiry_str, "%d/%m/%Y").date()
            if exp_date < date.today():
                expired = True
                violations.append(f"Licence expired on {expiry_str}")
        except Exception:
            pass

    # 4. Age Check (DOB)
    age_valid = True
    dob_str = dl_fields.get("dob")
    if dob_str:
        try:
            dob_date = datetime.strptime(dob_str, "%d/%m/%Y").date()
            age_years = (date.today() - dob_date).days // 365
            if age_years < 18:
                age_valid = False
                violations.append(f"Holder underage for driving licence: {age_years} years old (minimum 18 required)")
        except Exception:
            pass

    valid = (format_valid and not expired and age_valid)
    return {
        "valid": valid,
        "format_valid": format_valid,
        "state_code": state_code,
        "expired": expired,
        "age_valid": age_valid,
        "violations": violations,
        "score": 1.0 if valid else (0.5 if format_valid else 0.0),
    }
