"""
Visa Module
-----------
ocr.py      - Extract fields from visa stamp image
rules.py    - Logical rule validation (type/date/duration consistency)
binding.py  - Visa ↔ Passport number binding check
"""

from .binding import check_visa_passport_binding
from .ocr import extract_visa_fields
from .rules import validate_visa_rules
