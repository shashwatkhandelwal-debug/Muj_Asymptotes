"""
Decision Scorer - Risk assessment calculations.

This module computes the final risk score based on checksum validation,
signature verification, cross field consistency and ELA forensics.
"""

from typing import Any, Optional

# ELA normalization threshold constant.
# This value was tuned against sample_data to separate authentic images
# from spliced regions.
ELA_NORMALIZATION_THRESHOLD = 15.0


def compute_score(
    check_results: dict[str, Any], doc_type: Optional[str] = None
) -> dict:
    """
    Compute the risk score (0 to 100) based on checked fields.

    Higher score means higher risk.
    """
    # Infer doc_type if not provided explicitly
    if not doc_type:
        if (
            "visa_rule_validation" in check_results
            or "visa_passport_binding" in check_results
        ):
            doc_type = "VISA"
        elif (
            "passport_mrz_checksums" in check_results
            or "passport_mrz_viz_consistency" in check_results
        ):
            doc_type = "PASSPORT"
        elif "dl_rules" in check_results or "dl_format" in check_results:
            doc_type = "DRIVING_LICENCE"
        elif "permit_rules" in check_results or "permit_binding" in check_results:
            doc_type = "PERMIT"
        elif "generic_id_rules" in check_results:
            doc_type = "GENERIC_ID"
        else:
            doc_type = "AADHAAR"

    doc_type = doc_type.upper().replace(" ", "_")

    # Common ELA Anomaly calculation (shared across all document types)
    ela_variance = 0.0
    is_suspicious = False
    ela_res = check_results.get("ela_full_document")
    if isinstance(ela_res, dict):
        is_suspicious = bool(ela_res.get("suspicious", False))
        ela_variance = float(ela_res.get("mean_variance", 0.0))
    elif "ela_region_restricted" in check_results:
        reg_res = check_results["ela_region_restricted"]
        if isinstance(reg_res, dict):
            is_suspicious = bool(reg_res.get("suspicious", False))
            ela_variance = float(reg_res.get("mean_variance", 0.0))

    if is_suspicious:
        ela_anomaly_normalized = min(1.0, ela_variance / ELA_NORMALIZATION_THRESHOLD)
        term_ela = 20.0 * ela_anomaly_normalized
    else:
        term_ela = 0.0

    if doc_type in ("DRIVING_LICENCE", "DL"):
        # Standardized 30 + 30 + 20 + 20 matrix for Driving Licence
        dl_rules = check_results.get("dl_rules", {})
        rule_fail = 1.0 if (isinstance(dl_rules, dict) and not dl_rules.get("format_valid", True)) else 0.0
        expired_fail = 1.0 if (isinstance(dl_rules, dict) and dl_rules.get("expired", False)) else 0.0
        age_fail = 1.0 if (isinstance(dl_rules, dict) and not dl_rules.get("age_valid", True)) else 0.0

        term_format = 30.0 * rule_fail
        term_validity = 30.0 * (1.0 if expired_fail or age_fail else 0.0)
        term_viz = 20.0 * (1.0 if (rule_fail and expired_fail) else 0.0)

        raw_score = term_format + term_validity + term_viz + term_ela
        overall_score = min(100.0, max(0.0, raw_score))

        component_breakdown = {
            "format_rules_contribution": round(term_format, 2),
            "validity_expiry_contribution": round(term_validity, 2),
            "viz_consistency_contribution": round(term_viz, 2),
            "ela_contribution": round(term_ela, 2),
        }
    elif doc_type in ("PERMIT", "TRAVEL_PERMIT"):
        # Standardized 30 + 30 + 20 + 20 matrix for Border Entry Permits
        p_rules = check_results.get("permit_rules", {})
        permit_fail = 1.0 if (isinstance(p_rules, dict) and not p_rules.get("valid", True)) else 0.0
        binding_fail = 1.0 if (isinstance(p_rules, dict) and not p_rules.get("bound", True)) else 0.0
        expired_fail = 1.0 if (isinstance(p_rules, dict) and p_rules.get("expired", False)) else 0.0

        term_rules = 30.0 * (1.0 if permit_fail or expired_fail else 0.0)
        term_binding = 30.0 * binding_fail
        term_cross_field = 20.0 * binding_fail

        raw_score = term_rules + term_binding + term_cross_field + term_ela
        overall_score = min(100.0, max(0.0, raw_score))

        component_breakdown = {
            "permit_validity_contribution": round(term_rules, 2),
            "id_binding_contribution": round(term_binding, 2),
            "cross_field_contribution": round(term_cross_field, 2),
            "ela_contribution": round(term_ela, 2),
        }
    elif doc_type in ("GENERIC_ID", "NATIONAL_ID"):
        gid_rules = check_results.get("generic_id_rules", {})
        gid_fail = 1.0 if (isinstance(gid_rules, dict) and gid_rules.get("score") == 0.0) else 0.0
        term_format = 30.0 * gid_fail
        term_validity = 30.0 * gid_fail
        term_viz = 20.0 * gid_fail

        raw_score = term_format + term_validity + term_viz + term_ela
        overall_score = min(100.0, max(0.0, raw_score))

        component_breakdown = {
            "format_rules_contribution": round(term_format, 2),
            "validity_contribution": round(term_validity, 2),
            "viz_consistency_contribution": round(term_viz, 2),
            "ela_contribution": round(term_ela, 2),
        }
    elif doc_type == "VISA":
        # Document-specific scoring logic for Visa.
        # Since Visa stickers do not contain printed check digits or signed QR codes,
        # the standard checksum and signature checks are replaced.
        # Component 13 rule validation and Component 14 passport binding are used instead.

        # Component 13: Visa Rule Validation
        rule_val = check_results.get("visa_rule_validation", {})
        rule_violation_fail = (
            1.0
            if (isinstance(rule_val, dict) and not rule_val.get("valid", True))
            else 0.0
        )

        # Component 14: Visa Passport Binding
        bind = check_results.get("visa_passport_binding", {})
        binding_fail = (
            1.0 if (isinstance(bind, dict) and bind.get("score") == 0.0) else 0.0
        )

        # Cross-field Inconsistency
        cross_field_inconsistent = binding_fail

        term_rule_violation = 30.0 * rule_violation_fail
        term_binding = 30.0 * binding_fail
        term_cross_field = 20.0 * cross_field_inconsistent

        raw_score = term_rule_violation + term_binding + term_cross_field + term_ela
        overall_score = min(100.0, max(0.0, raw_score))

        component_breakdown = {
            "rule_violation_contribution": round(term_rule_violation, 2),
            "binding_contribution": round(term_binding, 2),
            "cross_field_contribution": round(term_cross_field, 2),
            "ela_contribution": round(term_ela, 2),
        }
    elif doc_type == "PASSPORT":
        # 1. Checksum Failure
        checksum_fail = 0.0
        if "passport_mrz_checksums" in check_results:
            res = check_results["passport_mrz_checksums"]
            if res is not None and res.get("score") == 0.0:
                checksum_fail = 1.0

        # 2. Cross-field Inconsistency
        cross_field_inconsistent = 0.0
        if "passport_mrz_viz_consistency" in check_results:
            res = check_results["passport_mrz_viz_consistency"]
            if res is not None and res.get("score") == 0.0:
                cross_field_inconsistent = 1.0

        # 3. Authenticity Failure (based on verification_tier)
        verification_tier = check_results.get("verification_tier", "CHIP_UNAVAILABLE")
        authenticity_fail = 0.0

        if verification_tier == "CHIP_VERIFIED":
            pa = check_results.get("passport_passive_auth")
            aa = check_results.get("passport_active_auth")
            pa_score = pa.get("score") if isinstance(pa, dict) else None
            aa_score = aa.get("score") if isinstance(aa, dict) else None
            if pa_score == 0.0 or aa_score == 0.0:
                authenticity_fail = 1.0
        elif verification_tier == "CHIP_READ_FAILED":
            authenticity_fail = 1.0

        term_checksum = 30.0 * checksum_fail
        term_cross_field = 20.0 * cross_field_inconsistent
        term_authenticity = 30.0 * authenticity_fail

        if verification_tier == "CHIP_UNAVAILABLE":
            # Rescale the remaining three terms (checksum, cross-field, ELA) proportionally to still cap at 100.
            # Max possible pre-rescale sum is 30 (checksum) + 20 (cross-field) + 20 (ELA) = 70.
            raw_score = term_checksum + term_cross_field + term_ela
            overall_score = min(100.0, max(0.0, raw_score * (100.0 / 70.0)))

            component_breakdown = {
                "checksum_contribution": round(term_checksum * (100.0 / 70.0), 2),
                "cross_field_contribution": round(term_cross_field * (100.0 / 70.0), 2),
                "ela_contribution": round(term_ela * (100.0 / 70.0), 2),
            }
        else:
            raw_score = term_checksum + term_cross_field + term_authenticity + term_ela
            overall_score = min(100.0, max(0.0, raw_score))

            component_breakdown = {
                "checksum_contribution": round(term_checksum, 2),
                "chip_authenticity_contribution": round(term_authenticity, 2),
                "cross_field_contribution": round(term_cross_field, 2),
                "ela_contribution": round(term_ela, 2),
            }
    else:
        # AADHAAR
        # 1. Checksum Failure (Verhoeff)
        checksum_fail = 0.0
        if "aadhaar_verhoeff" in check_results:
            res = check_results["aadhaar_verhoeff"]
            if res is not None and res.get("score") == 0.0:
                checksum_fail = 1.0

        # Determine verification tier
        verification_tier = check_results.get(
            "verification_tier",
            check_results.get("aadhaar_uidai_signature", {}).get(
                "verification_tier", "QR_VERIFIED"
            ),
        )

        # 2. Cryptographic Signature Failure
        # Only evaluated in QR_VERIFIED tier (when a digital signature block is present).
        signature_fail = 0.0
        if verification_tier == "QR_VERIFIED":
            if "aadhaar_uidai_signature" in check_results:
                res = check_results["aadhaar_uidai_signature"]
                if res is not None:
                    # If valid or rotated_key, score is 1.0 (0 penalty)
                    if res.get("score") == 0.0 and not res.get("rotated_key"):
                        signature_fail = 1.0

        # 3. Cross-field Inconsistency
        # Evaluated in QR_VERIFIED and QR_LEGACY tiers where QR fields are available.
        cross_field_inconsistent = 0.0
        if verification_tier in ("QR_VERIFIED", "QR_LEGACY"):
            if "aadhaar_qr_ocr_consistency" in check_results:
                res = check_results["aadhaar_qr_ocr_consistency"]
                if res is not None and res.get("score") == 0.0:
                    cross_field_inconsistent = 1.0

        term_checksum = 30.0 * checksum_fail
        term_signature = 30.0 * signature_fail
        term_cross_field = 20.0 * cross_field_inconsistent

        if verification_tier == "QR_LEGACY":
            # QR_LEGACY Tier:
            # Pre-2017 XML format QR card with structural absence of an RSA digital signature block.
            # Cryptographic signature check is not applicable (signature_valid = null).
            # Cross-field consistency (XML QR fields vs Front OCR) is available and validated.
            # Available terms: Verhoeff (weight 30) + Cross-field (weight 20) + ELA (weight 20) = sum 70.
            # Rescale available terms to 100 using the 100/70 multiplier (mirroring Passport CHIP_UNAVAILABLE).
            raw_score = term_checksum + term_cross_field + term_ela
            overall_score = min(100.0, max(0.0, raw_score * (100.0 / 70.0)))

            component_breakdown = {
                "checksum_contribution": round(term_checksum * (100.0 / 70.0), 2),
                "cross_field_contribution": round(
                    term_cross_field * (100.0 / 70.0), 2
                ),
                "ela_contribution": round(term_ela * (100.0 / 70.0), 2),
            }
        elif verification_tier == "QR_UNREADABLE":
            # QR_UNREADABLE Tier:
            # Capture-quality failure: QR unreadable due to glare, motion blur, angle, or damage.
            # Signature is marked signature_valid = null (lack of evidence, not proof of tampering).
            # Because QR could not be decoded, both signature (30) and cross-field (20) checks are unavailable.
            # Available terms: Verhoeff (weight 30) + ELA (weight 20) = sum 50.
            # Rescaled using 100/50 = 2.0 multiplier so that genuine physical/mathematical tampering
            # (e.g. invalid Verhoeff + ELA anomaly) reaches FLAGGED (100.0) rather than being artificially capped low.
            raw_score = term_checksum + term_ela
            overall_score = min(100.0, max(0.0, raw_score * (100.0 / 50.0)))

            component_breakdown = {
                "checksum_contribution": round(term_checksum * (100.0 / 50.0), 2),
                "ela_contribution": round(term_ela * (100.0 / 50.0), 2),
            }
        else:
            # QR_VERIFIED Tier (Standard Secure QR):
            # All four checks active: Verhoeff (30) + Signature (30) + Cross-field (20) + ELA (20) = 100.
            raw_score = term_checksum + term_signature + term_cross_field + term_ela
            overall_score = min(100.0, max(0.0, raw_score))

            component_breakdown = {
                "checksum_contribution": round(term_checksum, 2),
                "signature_contribution": round(term_signature, 2),
                "cross_field_contribution": round(term_cross_field, 2),
                "ela_contribution": round(term_ela, 2),
            }

    # Determine status based on thresholds
    if overall_score <= 30.0:
        status = "CLEAR"
    elif overall_score <= 60.0:
        status = "REVIEW"
    else:
        status = "FLAGGED"

    # Identify failed and uncertain checks
    failed_checks = []
    uncertain_checks = []

    for name, result in check_results.items():
        if isinstance(result, dict):
            score = result.get("score")
            if score == 0.0:
                failed_checks.append(name)
            elif score == 0.5:
                uncertain_checks.append(name)

    # ── Hard Overrides & High-Risk Threat Escalations ─────────────────────────

    # 1. Watchlist Blacklist Hit -> Mandatory FLAGGED (100.0)
    watchlist = check_results.get("watchlist")
    if isinstance(watchlist, dict) and (
        watchlist.get("flagged") or watchlist.get("score") == 0.0
    ):
        overall_score = 100.0
        status = "FLAGGED"
        if "watchlist" not in failed_checks:
            failed_checks.append("watchlist")

    # 2. Identity Graph Conflict (Face seen under different name/doc) -> Mandatory FLAGGED (100.0)
    id_graph = check_results.get("identity_graph")
    if isinstance(id_graph, dict) and id_graph.get("identity_conflict"):
        overall_score = 100.0
        status = "FLAGGED"
        if "identity_conflict" not in failed_checks:
            failed_checks.append("identity_conflict")

    # 3. Biometric Face Mismatch or Liveness Failure -> Escalates to FLAGGED
    face_match = check_results.get("face_match")
    if isinstance(face_match, dict):
        if (
            face_match.get("matched") is False
            or face_match.get("verdict") == "MISMATCH"
            or face_match.get("score") == 0.0
        ):
            overall_score = max(overall_score, 75.0)
            status = "FLAGGED"
            if "face_match" not in failed_checks:
                failed_checks.append("face_match")
        elif face_match.get("verdict") == "UNCERTAIN":
            overall_score = max(overall_score, 45.0)
            if status == "CLEAR":
                status = "REVIEW"

    liveness = check_results.get("liveness")
    if isinstance(liveness, dict) and (
        liveness.get("live") is False or liveness.get("score") == 0.0
    ):
        overall_score = max(overall_score, 75.0)
        status = "FLAGGED"
        if "liveness" not in failed_checks:
            failed_checks.append("liveness")

    # 4. EXIF Splicing / Editing Software Tag (Photoshop, GIMP, Canva, etc.)
    exif = check_results.get("exif_forensics") or check_results.get("exif")
    if isinstance(exif, dict) and exif.get("suspicious"):
        overall_score = min(100.0, overall_score + 25.0)
        if "exif_forensics" not in failed_checks:
            failed_checks.append("exif_forensics")
        if overall_score > 60.0:
            status = "FLAGGED"
        elif overall_score > 30.0 and status == "CLEAR":
            status = "REVIEW"

    # 5. Document Expiry Check
    expiry_check = check_results.get("expiry_valid")
    if isinstance(expiry_check, dict) and expiry_check.get("score") == 0.0:
        overall_score = min(100.0, overall_score + 30.0)
        if "expiry_valid" not in failed_checks:
            failed_checks.append("expiry_valid")
        if overall_score > 60.0:
            status = "FLAGGED"
        elif overall_score > 30.0 and status == "CLEAR":
            status = "REVIEW"

    return {
        "overall_score": round(overall_score, 1),
        "status": status,
        "component_breakdown": component_breakdown,
        "failed_checks": failed_checks,
        "uncertain_checks": uncertain_checks,
    }
