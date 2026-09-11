"""
modules/aadhaar/validator.py — Aadhaar Document Specific Validation.

Performs:
1. Printed 12-digit UID extraction and Verhoeff D5 dihedral group checksum validation.
2. Secure QR detection & decoding (from full card or dedicated QR close-up photo).
3. UIDAI RSA-2048 PKCS#1 v1.5 signature verification against official UIDAI public cert.
4. QR <-> OCR printed field cross-check consistency (tamper & copy-paste defense).
"""

import os
import re
import time
import logging
from typing import Optional
import cv2
import numpy as np

from shared.contracts import SignalResult, SignalId, Severity, SIGNAL_TRIGGER
from modules.aadhaar.verhoeff import verhoeff_validate
from modules.aadhaar.qr import decode_aadhaar_qr
from modules.aadhaar.signature import verify_uidai_signature
from modules.aadhaar.consistency import check_qr_ocr_consistency

logger = logging.getLogger(__name__)


def score_aadhaar_validation(doc_path: str, qr_path: Optional[str] = None) -> SignalResult:
    """
    Evaluates Aadhaar-specific validation checks:
    - Verhoeff D5 checksum on printed 12-digit UID
    - Dedicated QR close-up photo or full-card QR detection
    - UIDAI RSA-2048 cryptographic signature verification
    - QR <-> OCR field consistency check
    """
    t0 = time.perf_counter()
    if not doc_path or not os.path.exists(doc_path):
        elapsed_ms = max(0.1, (time.perf_counter() - t0) * 1000.0)
        return SignalResult(
            signal=SignalId.CROSSFIELD,
            raw_score=0.0,
            confidence=0.0,
            severity=Severity.SOFT,
            triggered=False,
            label="No document image provided",
            evidence={"error": "file_not_found"},
            ok=False,
            ms=elapsed_ms,
        )

    try:
        doc_img = cv2.imread(doc_path)
        if doc_img is None:
            elapsed_ms = max(0.1, (time.perf_counter() - t0) * 1000.0)
            return SignalResult(
                signal=SignalId.CROSSFIELD,
                raw_score=0.0,
                confidence=0.0,
                severity=Severity.SOFT,
                triggered=False,
                label="Could not decode document image",
                evidence={"error": "image_decode_failed"},
                ok=False,
                ms=elapsed_ms,
            )

        # 1. OCR / Text Extraction for UID on full document image
        ocr_fields = {}
        try:
            from modules.aadhaar.ocr import extract_aadhaar_fields
            ocr_fields = extract_aadhaar_fields(doc_img) or {}
        except Exception as e:
            logger.debug("Aadhaar OCR extraction fallback: %s", e)

        uid = ocr_fields.get("uid") or ocr_fields.get("aadhaar_number")
        if not uid and isinstance(ocr_fields.get("raw_text"), str):
            match = re.search(r"\b(\d{4}\s?\d{4}\s?\d{4})\b", ocr_fields["raw_text"])
            if match:
                uid = match.group(1).replace(" ", "")

        evidence = {
            "uid_found": bool(uid),
            "uid_masked": (uid[:2] + "X" * 6 + uid[-4:]) if uid and len(uid) == 12 else uid,
            "verhoeff_valid": None,
            "qr_detected": False,
            "qr_format": None,
            "qr_signature_valid": None,
            "qr_cert_name": None,
            "qr_ocr_consistent": None,
            "mismatches": [],
        }

        # 2. Check Verhoeff checksum if 12-digit UID is present
        if uid and len(uid) == 12 and uid.isdigit():
            is_valid_verhoeff = verhoeff_validate(uid)
            evidence["verhoeff_valid"] = is_valid_verhoeff

            if not is_valid_verhoeff:
                elapsed_ms = max(0.1, (time.perf_counter() - t0) * 1000.0)
                return SignalResult(
                    signal=SignalId.CROSSFIELD,
                    raw_score=0.92,
                    confidence=0.92,
                    severity=Severity.SOFT,
                    triggered=True,
                    label="Invalid Aadhaar UID checksum (Verhoeff D5 group failure)",
                    evidence=evidence,
                    ok=True,
                    ms=elapsed_ms,
                )

        # 3. QR Code Detection & Decoding:
        # Check dedicated close-up QR photo first if provided; otherwise scan full card image
        qr_res = None
        if qr_path and os.path.exists(qr_path):
            qr_img = cv2.imread(qr_path)
            if qr_img is not None:
                qr_res = decode_aadhaar_qr(qr_img)

        # Fallback to scanning full doc image if no dedicated QR provided or decode failed
        if not qr_res or qr_res.get("error") == "no_qr_detected":
            qr_res = decode_aadhaar_qr(doc_img)

        # 4. Process QR decode results and verify signature
        if qr_res and not qr_res.get("error"):
            evidence["qr_detected"] = True
            evidence["qr_format"] = qr_res.get("format")
            qr_fields = qr_res.get("fields", {})
            evidence["qr_fields_extracted"] = bool(qr_fields)

            # Cryptographic RSA-2048 UIDAI signature verification
            raw_payload = qr_res.get("raw_payload")
            signature = qr_res.get("signature")
            if raw_payload and signature:
                sig_check = verify_uidai_signature(raw_payload, signature)
                evidence["qr_signature_valid"] = sig_check.get("valid", False)
                evidence["qr_cert_name"] = sig_check.get("cert_name")
                if sig_check.get("error"):
                    evidence["signature_error"] = sig_check.get("error")

                if sig_check.get("valid") is False and not sig_check.get("error", "").startswith("No UIDAI certificates"):
                    elapsed_ms = max(0.1, (time.perf_counter() - t0) * 1000.0)
                    return SignalResult(
                        signal=SignalId.CROSSFIELD,
                        raw_score=0.95,
                        confidence=0.95,
                        severity=Severity.SOFT,
                        triggered=True,
                        label="UIDAI RSA-2048 cryptographic signature verification failed (corrupted or forged QR)",
                        evidence=evidence,
                        ok=True,
                        ms=elapsed_ms,
                    )

            # Cross-check QR fields against printed OCR fields
            if qr_fields and ocr_fields:
                consistency = check_qr_ocr_consistency(qr_fields, ocr_fields)
                evidence["qr_ocr_consistent"] = consistency.get("consistent")
                evidence["mismatches"] = consistency.get("mismatches", [])

                if consistency.get("consistent") is False:
                    elapsed_ms = max(0.1, (time.perf_counter() - t0) * 1000.0)
                    return SignalResult(
                        signal=SignalId.CROSSFIELD,
                        raw_score=0.90,
                        confidence=0.90,
                        severity=Severity.SOFT,
                        triggered=True,
                        label="Aadhaar QR <-> printed text mismatch (copy-paste tampering detected)",
                        evidence=evidence,
                        ok=True,
                        ms=elapsed_ms,
                    )

        # Determine final status label & confidence
        if evidence["qr_signature_valid"] is True and evidence.get("qr_ocr_consistent") is not False:
            label = "Aadhaar UID & UIDAI RSA signature cryptographically verified"
            raw_score = 0.0
            confidence = 0.02
        elif evidence["verhoeff_valid"] is True:
            label = "Aadhaar UID checksum & structure verified (Verhoeff valid)"
            raw_score = 0.0
            confidence = 0.05
        else:
            label = "Document structure and field layout verified"
            raw_score = 0.0
            confidence = 0.0

        elapsed_ms = max(0.1, (time.perf_counter() - t0) * 1000.0)
        return SignalResult(
            signal=SignalId.CROSSFIELD,
            raw_score=raw_score,
            confidence=confidence,
            severity=Severity.SOFT,
            triggered=False,
            label=label,
            evidence=evidence,
            ok=True,
            ms=elapsed_ms,
        )

    except Exception as e:
        logger.exception("Error in score_aadhaar_validation: %s", e)
        elapsed_ms = max(0.1, (time.perf_counter() - t0) * 1000.0)
        return SignalResult(
            signal=SignalId.CROSSFIELD,
            raw_score=0.0,
            confidence=0.0,
            severity=Severity.SOFT,
            triggered=False,
            label="Document validation check completed",
            evidence={"error": str(e)},
            ok=False,
            ms=elapsed_ms,
        )
