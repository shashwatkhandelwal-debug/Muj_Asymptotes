"""
Border Permit Rules & Binding Validator.
"""

from datetime import datetime, date


def validate_permit_rules(permit_fields: dict, presented_id: str = None) -> dict:
    violations = []
    
    # 1. Number presence
    p_num = permit_fields.get("permit_number")
    if not p_num:
        violations.append("Permit number missing or unreadable")

    # 2. Validity Window
    expired = False
    valid_until = permit_fields.get("valid_until")
    if valid_until:
        try:
            exp_date = datetime.strptime(valid_until, "%d/%m/%Y").date()
            if exp_date < date.today():
                expired = True
                violations.append(f"Permit expired on {valid_until}")
        except Exception:
            pass

    # 3. ID Binding (if presented ID provided)
    bound = True
    assoc_id = permit_fields.get("associated_id")
    if presented_id and assoc_id:
        p_clean = presented_id.replace(" ", "").upper()
        a_clean = assoc_id.replace(" ", "").upper()
        if p_clean != a_clean:
            bound = False
            violations.append(f"Permit issued for ID {assoc_id}, but presented ID is {presented_id}")

    valid = (bool(p_num) and not expired and bound)
    return {
        "valid": valid,
        "expired": expired,
        "bound": bound,
        "violations": violations,
        "score": 1.0 if valid else 0.0,
    }
