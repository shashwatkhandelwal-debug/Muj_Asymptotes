"""
Aadhaar Module
--------------
ocr.py          - Extract printed fields via OpenBharatOCR (EasyOCR + spaCy fallback)
qr.py           - Decode Secure QR (PyZBar → big-int → gzip → 0xFF fields)
verhoeff.py     - Verhoeff checksum validation (dihedral group D5)
signature.py    - UIDAI RSA-2048 QR signature verification
consistency.py  - QR fields vs OCR printed fields cross-check
"""

from .consistency import check_qr_ocr_consistency
from .ocr import extract_aadhaar_fields
from .qr import decode_aadhaar_qr
from .signature import verify_uidai_signature
from .verhoeff import verhoeff_validate
