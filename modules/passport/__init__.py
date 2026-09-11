"""
Passport Module
---------------
mrz.py          - ICAO 9303 TD3 MRZ parsing + 7-3-1 check digits
viz.py          - VIZ OCR (biographical page printed fields)
consistency.py  - MRZ ↔ VIZ cross-check
nfc.py          - BAC handshake (3DES key derivation, mutual auth)
passive_auth.py - Passive Authentication (ICAO Master List cert chain + DG hashes)
active_auth.py  - Active Authentication (anti-chip-clone challenge-response)
"""

from .active_auth import perform_active_auth
from .consistency import check_mrz_viz_consistency
from .mrz import icao_check_digit, parse_mrz
from .nfc import derive_bac_keys, perform_bac
from .passive_auth import perform_passive_auth
from .viz import extract_viz_fields
