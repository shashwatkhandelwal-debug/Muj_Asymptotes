"""
Active Authentication  -  ICAO 9303 Part 11.

Proves the physical chip is the ORIGINAL chip  -  not a clone.

How cloning is stopped:
  During chip manufacturing, the chip generates an RSA key pair internally.
  The public key is stored in DG15 (authenticated by Passive Auth).
  The private key is stored in tamper-resistant secure enclave hardware and
  NEVER exposed via any APDU command  -  it can only be USED to sign data.

  A chip clone can copy every data group byte-for-byte, but cannot copy
  the private key. When the terminal sends a random challenge and asks the
  chip to sign it, only the original chip can produce a valid signature.

Protocol:
  1. Read chip's public key from DG15 (already authenticated by Passive Auth)
  2. Send 8-byte random challenge to chip via GET INTERNAL AUTHENTICATE (0x88)
  3. Chip signs challenge with its private key (never leaves the chip)
  4. Terminal verifies signature against the DG15 public key

References:
  - ICAO 9303 Part 11, Section 6 (Active Authentication)
"""

import os
from typing import Optional

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey

    _CRYPTO_AVAILABLE = True
except ImportError:
    _CRYPTO_AVAILABLE = False


def perform_active_auth(chip_dg15: bytes, clf=None) -> dict:
    """
    Perform Active Authentication challenge-response.

    Args:
        chip_dg15: Raw bytes of Data Group 15 (contains chip's AA public key).
        clf:       nfcpy ContactlessFrontend with active BAC session.
                   If None, returns a fallback indicating hardware unavailable.

    Returns:
        dict with keys:
          valid (bool)        -  True if chip answered correctly (original chip)
          challenge (bytes)   -  The random nonce sent to the chip
          error (str|None)
    """
    if not _CRYPTO_AVAILABLE:
        return {
            "valid": False,
            "error": "cryptography not installed",
            "challenge": None,
        }

    if clf is None:
        return {
            "valid": None,
            "error": "NFC hardware not available  -  Active Auth skipped",
            "challenge": None,
        }

    try:
        # Step 1: Parse public key from DG15
        public_key = _parse_dg15_public_key(chip_dg15)
        if public_key is None:
            return {
                "valid": False,
                "error": "Could not parse DG15 public key",
                "challenge": None,
            }

        # Step 2: Generate 8-byte random challenge
        challenge = os.urandom(8)

        # Step 3: Send GET INTERNAL AUTHENTICATE APDU
        # INS=0x88, P1=0x00, P2=0x00, Lc=len(challenge), Le=0x00 (any length)
        apdu = [0x00, 0x88, 0x00, 0x00, len(challenge)] + list(challenge) + [0x00]
        resp = _send_apdu_secure(clf, apdu)
        chip_signature = bytes(resp)

        # Step 4: Verify chip's signature against DG15 public key
        try:
            public_key.verify(
                chip_signature,
                challenge,
                padding.PKCS1v15(),
                hashes.SHA1(),  # ICAO originally specifies SHA-1 for AA
            )
            return {"valid": True, "challenge": challenge.hex(), "error": None}
        except InvalidSignature:
            return {
                "valid": False,
                "challenge": challenge.hex(),
                "error": "Active Auth failed  -  chip could not answer challenge. Possible cloned chip.",
            }

    except Exception as e:
        return {"valid": False, "error": f"Active Auth error: {e}", "challenge": None}


# ── Helpers ────────────────────────────────────────────────────────────────────


def _parse_dg15_public_key(dg15_bytes: bytes) -> Optional["RSAPublicKey"]:
    """
    Parse RSA public key from DG15 raw bytes.
    DG15 encodes a SubjectPublicKeyInfo ASN.1 structure.
    """
    try:
        from cryptography.hazmat.primitives.serialization import load_der_public_key

        # DG15 starts with a tag (0x6F or similar), need to find SubjectPublicKeyInfo
        # The actual SPKi starts after the DG tag/length bytes
        # Try parsing directly first
        try:
            return load_der_public_key(dg15_bytes)
        except Exception:
            # Skip the DG header bytes (usually 2–4 bytes) and retry
            for offset in [2, 3, 4, 6]:
                try:
                    return load_der_public_key(dg15_bytes[offset:])
                except Exception:
                    continue
        return None
    except Exception:
        return None


def _send_apdu_secure(clf, apdu: list) -> list:
    """Send APDU within a BAC-secured session."""
    resp = clf.send(bytes(apdu))
    if resp[-2:] not in [bytes([0x90, 0x00]), bytes([0x61, len(resp) - 2])]:
        raise RuntimeError(f"APDU error: SW={resp[-2:].hex()}")
    return list(resp[:-2])
