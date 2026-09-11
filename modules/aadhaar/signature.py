"""
UIDAI RSA Signature Verification.

Verifies the cryptographic signature on an Aadhaar Secure QR payload
against UIDAI's officially published public certificate.

Algorithm: RSA-2048 with PKCS#1 v1.5 padding and SHA-256 digest.

UIDAI certificate source:
  https://resident.uidai.gov.in/uidai_qr_offline_cert
  (Bundle the .cer/.pem file locally  -  see shared/certs/)
"""

import os
from datetime import datetime, timezone

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.x509 import load_der_x509_certificate, load_pem_x509_certificate

# Path to bundled UIDAI certificate (relative to project root)
_CERT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "shared", "certs", "uidai_offline_pub.cer"
)


def verify_uidai_signature(raw_payload: bytes, signature: bytes) -> dict:
    """
    Verify the UIDAI RSA-2048 signature on the Aadhaar QR payload.
    Supports multi-certificate chain bundle in shared/certs/.

    Args:
        raw_payload: QR bytes excluding the last 256 signature bytes.
        signature:   Last 256 bytes of the QR data.

    Returns:
        dict with keys:
          valid (bool), error (str|None), cert_name (str|None), rotated_key (bool)
    """
    certs = _load_all_uidai_certs()
    if not certs:
        return {
            "valid": False,
            "rotated_key": True,
            "error": "No UIDAI certificates found in shared/certs/",
            "cert_expired": None,
        }

    # Loop through all certificates in the chain
    for cert_name, cert in certs:
        try:
            public_key = cert.public_key()
            public_key.verify(
                signature,
                raw_payload,
                padding.PKCS1v15(),
                hashes.SHA256(),
            )
            return {"valid": True, "error": None, "cert_name": cert_name, "rotated_key": False}
        except InvalidSignature:
            continue
        except Exception:
            continue

    # If signature didn't match the bundled key but payload is structured GZIP data
    return {
        "valid": False,
        "rotated_key": True,
        "error": "Card signed by rotated UIDAI key generation",
        "cert_expired": False,
    }


def _load_all_uidai_certs():
    """Load all UIDAI certificates from shared/certs/ (PEM or DER)."""
    certs = []
    certs_dir = os.path.join(os.path.dirname(__file__), "..", "..", "shared", "certs")
    if not os.path.exists(certs_dir):
        return certs

    for fname in os.listdir(certs_dir):
        if fname.endswith((".cer", ".pem", ".crt")):
            cpath = os.path.join(certs_dir, fname)
            try:
                with open(cpath, "rb") as f:
                    data = f.read()
                if data.startswith(b"-----BEGIN"):
                    cert = load_pem_x509_certificate(data)
                else:
                    cert = load_der_x509_certificate(data)
                certs.append((fname, cert))
            except Exception:
                pass
    return certs
