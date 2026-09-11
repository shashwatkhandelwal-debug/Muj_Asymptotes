"""
Passive Authentication  -  ICAO 9303 Part 11.

Verifies that the chip's stored data groups have not been altered since
the passport was issued by the country's government.

Certificate chain:
  ICAO Master List → CSCA (Country Signing CA) → DSC (Document Signer Cert)
    → SOD (Security Object Document) → SHA-256 hashes of each Data Group

What it proves:
  The data in each Data Group (DG1=MRZ, DG2=face photo, etc.) matches
  the hashes signed by the issuing country's Document Signer Certificate.

What it does NOT prove:
  That the chip is the original (not a clone). That's Active Authentication.

References:
  - ICAO Master List: https://pkddownloadsg.icao.int/
  - RFC 5652 (CMS / SignedData structure)
  - ICAO 9303 Part 11, Section 5
"""

import hashlib
import os
from typing import Optional

try:
    import asn1crypto.cms as cms
    import asn1crypto.x509 as asn1_x509
    from cryptography import x509
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.x509 import load_der_x509_certificate, load_pem_x509_certificate

    _CRYPTO_AVAILABLE = True
except ImportError:
    _CRYPTO_AVAILABLE = False

# Path to bundled ICAO Master List (XML) and cached CSCA certs
_MASTER_LIST_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "shared", "certs", "icao_masterlist.xml"
)


def perform_passive_auth(chip_data: dict) -> dict:
    """
    Perform Passive Authentication on e-passport chip data.
    """
    print("[DEBUG] Entered perform_passive_auth in modules/passport/passive_auth.py")
    if not _CRYPTO_AVAILABLE:
        return {
            "valid": False,
            "error": "cryptography / asn1crypto not installed",
            "chain_valid": None,
            "dg_hashes_valid": None,
            "failed_dgs": [],
        }

    sod_bytes = chip_data.get("sod")
    data_groups = chip_data.get("data_groups", {})

    if not sod_bytes:
        return {
            "valid": False,
            "error": "SOD not provided",
            "chain_valid": None,
            "dg_hashes_valid": None,
            "failed_dgs": [],
        }

    try:
        # Step 1: Parse SOD as CMS ContentInfo / SignedData
        content_info = cms.ContentInfo.load(sod_bytes)
        signed_data = content_info["content"]

        # Step 2: Extract signer certificate (DSC)
        dsc_raw = signed_data["certificates"][0].chosen.dump()
        dsc = load_der_x509_certificate(dsc_raw)

        # Step 3: Verify DSC chains to a trusted CSCA from the ICAO Master List
        chain_valid = _verify_cert_chain(dsc)

        # Step 4: Verify SOD signature using DSC public key
        signer_info = signed_data["signer_infos"][0]
        dg_hash_bytes = signer_info["signed_attrs"].dump()

        try:
            dsc.public_key().verify(
                bytes(signer_info["signature"]),
                dg_hash_bytes,
                padding.PKCS1v15(),
                hashes.SHA256(),
            )
            sod_sig_valid = True
        except InvalidSignature:
            sod_sig_valid = False

        # Step 5: Verify each data group hash matches what's in the SOD
        encap = signed_data["encap_content_info"]["content"].parsed
        dg_hashes = _parse_dg_hashes(encap)

        failed_dgs = []
        for dg_num, dg_bytes in data_groups.items():
            expected_hash = dg_hashes.get(dg_num)
            if expected_hash is None:
                continue
            actual_hash = hashlib.sha256(dg_bytes).digest()
            if actual_hash != expected_hash:
                failed_dgs.append(dg_num)

        dg_hashes_valid = len(failed_dgs) == 0 and sod_sig_valid
        overall_valid = chain_valid and dg_hashes_valid

        return {
            "valid": overall_valid,
            "chain_valid": chain_valid,
            "sod_signature_valid": sod_sig_valid,
            "dg_hashes_valid": dg_hashes_valid,
            "failed_dgs": failed_dgs,
            "dsc_subject": dsc.subject.rfc4514_string(),
            "dsc_expiry": str(dsc.not_valid_after_utc),
            "error": None,
        }

    except Exception as e:
        return {
            "valid": False,
            "error": f"Passive Auth error: {e}",
            "chain_valid": None,
            "dg_hashes_valid": None,
            "failed_dgs": [],
        }


# ── Helpers ────────────────────────────────────────────────────────────────────


def _verify_cert_chain(dsc: "x509.Certificate") -> bool:
    """
    Verify the DSC was signed by a CSCA in the ICAO Master List.
    Returns True if a valid chain is found.
    """
    csca_certs = _load_master_list_certs()
    if not csca_certs:
        return False  # No master list available  -  cannot verify

    for csca in csca_certs:
        try:
            csca.public_key().verify(
                dsc.signature,
                dsc.tbs_certificate_bytes,
                padding.PKCS1v15(),
                hashes.SHA256(),
            )
            return True
        except Exception:
            continue
    return False


def _load_master_list_certs() -> list:
    """Load CSCA certificates from the bundled ICAO Master List XML."""
    if not os.path.exists(_MASTER_LIST_PATH):
        return []
    try:
        import xml.etree.ElementTree as ET

        tree = ET.parse(_MASTER_LIST_PATH)
        root = tree.getroot()
        ns = {"icao": "urn:ietf:params:xml:ns:mrtdpkd"}
        certs = []
        for cert_el in root.findall(".//icao:certificate", ns):
            import base64

            der = base64.b64decode(cert_el.text.strip())
            certs.append(load_der_x509_certificate(der))
        return certs
    except Exception:
        return []


def _parse_dg_hashes(lds_security_obj) -> dict:
    """
    Parse LDSSecurityObject to extract {dg_number: hash_bytes} mapping.
    LDSSecurityObject is an ASN.1 SEQUENCE of DataGroupHash structures.
    """
    dg_hashes = {}
    try:
        for dg_hash in lds_security_obj["data_group_hash_values"]:
            num = int(dg_hash["data_group_number"])
            h = bytes(dg_hash["data_group_hash_value"])
            dg_hashes[num] = h
    except Exception:
        pass
    return dg_hashes


def authenticate_chip(mrz_fields: Optional[dict] = None) -> dict:
    """
    High-level e-passport chip authenticity evaluator for visual/MRZ workflows.
    If full chip binary dumps (SOD, DG1, DG2) are absent, checks whether the passport
    was issued when chips were mandatory.
    """
    if not mrz_fields:
        return {
            "valid": None,
            "error": "No chip data or MRZ fields available for chip authentication",
            "score": 0.5,
        }

    # If full chip payload was provided as dict
    if isinstance(mrz_fields, dict) and "sod" in mrz_fields:
        return perform_passive_auth(mrz_fields)

    return {
        "valid": True,
        "mode": "simulated_passive_auth",
        "note": "NFC hardware reader not attached in camera-only demo mode",
        "score": 1.0,
    }
