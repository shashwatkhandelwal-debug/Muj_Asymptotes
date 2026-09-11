"""
Visa Rule Validation  -  Logical consistency checks.

No cryptographic verification exists for visa stamps.
Validation is entirely rule-based: internal logical consistency
of the extracted fields against known visa type rules.
"""

from datetime import date, datetime
from typing import Optional

VISA_TYPE_RULES = {
    "Tourist": {"max_days": 180, "min_days": 1},
    "Business": {"max_days": 365, "min_days": 1},
    "Employment": {"max_days": 1825, "min_days": 180},  # up to 5 years
    "Student": {"max_days": 1825, "min_days": 90},
    "Transit": {"max_days": 3, "min_days": 1},
    "Medical": {"max_days": 365, "min_days": 7},
}


def validate_visa_rules(fields: dict) -> dict:
    """
    Validate extracted visa fields for logical consistency.

    Args:
        fields: Output from modules.visa.ocr.extract_visa_fields()

    Returns:
        dict with keys:
          valid (bool), violations (list[str]), checks (dict), score (float)
    """
    violations = []
    checks = {}
    today = date.today()

    # 1. Expiry after issue date
    doi = _parse_date(fields.get("date_of_issue"))
    doe = _parse_date(fields.get("date_of_expiry"))

    if doi and doe:
        checks["expiry_after_issue"] = doe > doi
        if not checks["expiry_after_issue"]:
            violations.append("Expiry date is before or on issue date")
    else:
        checks["expiry_after_issue"] = None

    # 2. Issue date not in the future
    if doi:
        checks["issue_not_future"] = doi <= today
        if not checks["issue_not_future"]:
            violations.append("Issue date is in the future")
    else:
        checks["issue_not_future"] = None

    # 3. Not expired
    if doe:
        checks["not_expired"] = doe >= today
        if not checks["not_expired"]:
            violations.append(f"Visa expired on {doe.strftime('%d/%m/%Y')}")
    else:
        checks["not_expired"] = None

    # 4. Duration fits visa type
    visa_type = fields.get("visa_type")
    duration = fields.get("duration_days")
    if visa_type and duration and visa_type in VISA_TYPE_RULES:
        rules = VISA_TYPE_RULES[visa_type]
        fits = rules["min_days"] <= duration <= rules["max_days"]
        checks["duration_fits_type"] = fits
        if not fits:
            violations.append(
                f"Stay duration {duration} days doesn't fit {visa_type} visa "
                f"(expected {rules['min_days']}–{rules['max_days']} days)"
            )
    else:
        checks["duration_fits_type"] = None

    # 5. Duration consistent with date range
    if doi and doe and duration:
        actual_days = (doe - doi).days
        tolerance = 7  # 1-week tolerance for calendar rounding
        consistent = abs(actual_days - duration) <= tolerance or actual_days >= duration
        checks["duration_date_consistent"] = consistent
        if not consistent:
            violations.append(
                f"Declared duration ({duration} days) inconsistent with date range ({actual_days} days)"
            )
    else:
        checks["duration_date_consistent"] = None

    valid = len(violations) == 0
    # Score: fraction of non-None checks that passed
    non_none = {k: v for k, v in checks.items() if v is not None}
    score = sum(non_none.values()) / len(non_none) if non_none else 0.5

    return {
        "valid": valid,
        "violations": violations,
        "checks": checks,
        "score": round(score, 2),
    }


def _parse_date(date_str: Optional[str]) -> Optional[date]:
    if not date_str:
        return None
    formats = ["%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    return None
