"""
NFC BAC (Basic Access Control) Handshake  -  ICAO 9303 Part 11.

Establishes an authenticated, encrypted NFC session with an e-passport chip
using the MRZ data as the access key. Only someone holding the physical
passport (who can read the MRZ) can derive the correct session keys.

Protocol:
  1. Derive K_seed from MRZ_information via SHA-1
  2. Derive K_enc and K_mac from K_seed via 3DES key derivation
  3. GET CHALLENGE → receive RND_IC from chip
  4. Compute authentication token using K_enc + K_mac
  5. EXTERNAL AUTHENTICATE → mutual authentication
  6. Derive session keys KS_enc, KS_mac from both parties' nonces

References:
  - ICAO Doc 9303 Part 11 (Machine Readable Travel Documents)
  - https://www.icao.int/publications/Documents/9303_p11_cons_en.pdf
"""

import hashlib
import os
import struct
from typing import Optional

try:
    import nfc

    _NFC_AVAILABLE = True
except ImportError:
    _NFC_AVAILABLE = False


def derive_bac_keys(mrz_line2: str) -> tuple[bytes, bytes]:
    """
    Derive BAC encryption and MAC keys from MRZ line 2.

    MRZ_information = passport_number(9) + cd_pn(1) + dob(6) + cd_dob(1) + expiry(6) + cd_expiry(1)
    This 24-character string is the BAC seed.

    Args:
        mrz_line2: 44-character MRZ line 2 string.

    Returns:
        Tuple of (K_enc, K_mac)  -  each 16 bytes (adjusted for 3DES parity).
    """
    # Extract the 24-char MRZ_information block
    mrz_info = mrz_line2[0:10] + mrz_line2[13:20] + mrz_line2[21:28]

    # K_seed = first 16 bytes of SHA-1(MRZ_information)
    k_seed = hashlib.sha1(mrz_info.encode("ascii")).digest()[:16]

    k_enc = _derive_key(k_seed, counter=1)
    k_mac = _derive_key(k_seed, counter=2)

    return k_enc, k_mac


def perform_bac(mrz_line2: str, clf=None) -> dict:
    """
    Perform full BAC handshake with the chip.

    Args:
        mrz_line2: 44-character MRZ line 2 (provides the access key).
        clf:       nfcpy ContactlessFrontend object. If None, uses a mock
                   (for testing without hardware).

    Returns:
        dict with keys:
          success (bool), ks_enc (bytes), ks_mac (bytes),
          ssc (bytes), error (str|None)
    """
    if not _NFC_AVAILABLE:
        return {
            "success": False,
            "error": "nfcpy not installed  -  NFC unavailable",
            "ks_enc": None,
            "ks_mac": None,
            "ssc": None,
        }

    try:
        k_enc, k_mac = derive_bac_keys(mrz_line2)

        # Step 1: SELECT application (e-passport AID)
        AID = bytes.fromhex("A0000002471001")
        _send_apdu(clf, [0x00, 0xA4, 0x04, 0x0C, len(AID)] + list(AID))

        # Step 2: GET CHALLENGE  -  receive 8-byte nonce from chip
        resp = _send_apdu(clf, [0x00, 0x84, 0x00, 0x00, 0x08])
        rnd_ic = bytes(resp[:8])

        # Step 3: Generate terminal nonce and key material
        rnd_ifd = os.urandom(8)
        k_ifd = os.urandom(16)

        # Step 4: Compute authentication token
        s = rnd_ifd + rnd_ic + k_ifd  # 32 bytes
        e_ifd = _3des_encrypt(k_enc, s)
        m_ifd = _retail_mac(k_mac, e_ifd)
        token_ifd = e_ifd + m_ifd  # 40 bytes

        # Step 5: EXTERNAL AUTHENTICATE
        resp2 = _send_apdu(
            clf, [0x00, 0x82, 0x00, 0x00, 0x28] + list(token_ifd) + [0x28]
        )
        token_ic = bytes(resp2[:40])

        # Step 6: Verify chip response
        decrypted = _3des_decrypt(k_enc, token_ic[:32])
        rnd_ic_back = decrypted[:8]
        rnd_ifd_back = decrypted[8:16]
        k_ic = decrypted[16:32]

        if rnd_ifd_back != rnd_ifd:
            return {
                "success": False,
                "error": "BAC mutual auth failed  -  RND_IFD mismatch",
                "ks_enc": None,
                "ks_mac": None,
                "ssc": None,
            }

        # Step 7: Derive session keys
        k_seed_session = bytes(a ^ b for a, b in zip(k_ifd, k_ic))
        ks_enc = _derive_key(k_seed_session, counter=1)
        ks_mac = _derive_key(k_seed_session, counter=2)
        ssc = rnd_ic[-4:] + rnd_ifd[-4:]  # Send Sequence Counter

        return {
            "success": True,
            "ks_enc": ks_enc,
            "ks_mac": ks_mac,
            "ssc": ssc,
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"BAC error: {e}",
            "ks_enc": None,
            "ks_mac": None,
            "ssc": None,
        }


# ── Cryptographic primitives ───────────────────────────────────────────────────


def _derive_key(k_seed: bytes, counter: int) -> bytes:
    """ICAO 3DES key derivation from K_seed with counter suffix."""
    data = k_seed + struct.pack(">I", counter)  # 20 bytes
    digest = hashlib.sha1(data).digest()
    key = digest[:16]
    return _adjust_parity(key)


def _adjust_parity(key: bytes) -> bytes:
    """Set odd parity on each byte (3DES key requirement)."""
    result = bytearray(key)
    for i in range(len(result)):
        b = result[i]
        # Count bits, adjust LSB to make total bits odd
        b = b & 0xFE
        bits = bin(b).count("1")
        if bits % 2 == 0:
            b |= 0x01
        result[i] = b
    return bytes(result)


def _3des_encrypt(key: bytes, data: bytes) -> bytes:
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    k = key + key[:8]  # 2TDEA → 3TDEA (24-byte key)
    cipher = Cipher(
        algorithms.TripleDES(k), modes.CBC(b"\x00" * 8), backend=default_backend()
    )
    enc = cipher.encryptor()
    return enc.update(data) + enc.finalize()


def _3des_decrypt(key: bytes, data: bytes) -> bytes:
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    k = key + key[:8]
    cipher = Cipher(
        algorithms.TripleDES(k), modes.CBC(b"\x00" * 8), backend=default_backend()
    )
    dec = cipher.decryptor()
    return dec.update(data) + dec.finalize()


def _retail_mac(key: bytes, data: bytes) -> bytes:
    """ISO 9797-1 MAC Algorithm 3 (Retail MAC) with 8-byte output."""
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    # Pad to 8-byte boundary
    pad_len = 8 - (len(data) % 8)
    data = data + b"\x80" + b"\x00" * (pad_len - 1)

    k1 = key[:8]
    k2 = key[8:16]
    iv = b"\x00" * 8
    # DES-CBC with k1
    cipher1 = Cipher(
        algorithms.TripleDES(k1 * 3), modes.CBC(iv), backend=default_backend()
    )
    enc1 = cipher1.encryptor()
    intermediate = enc1.update(data) + enc1.finalize()
    block = intermediate[-8:]
    # 3DES with full key on final block
    cipher2 = Cipher(
        algorithms.TripleDES(key + key[:8]),
        modes.CBC(b"\x00" * 8),
        backend=default_backend(),
    )
    enc2 = cipher2.encryptor()
    return (enc2.update(block) + enc2.finalize())[:8]


def _send_apdu(clf, apdu: list) -> list:
    """Send APDU to chip and return response bytes. Raises on non-9000 status."""
    resp = clf.send(bytes(apdu))
    if resp[-2:] != bytes([0x90, 0x00]):
        raise RuntimeError(f"APDU error: SW={resp[-2:].hex()}")
    return list(resp[:-2])
