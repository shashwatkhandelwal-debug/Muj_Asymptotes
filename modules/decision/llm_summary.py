"""
LLM Plain-Language Summary Generator.

Translates structured check results into one concise sentence
for the border officer, focusing only on the most critical finding.

Uses Google Gemini Flash (fast, low-latency, sufficient for this task).
Falls back to a rule-based template summary if the API call fails or is offline.
"""

import json
import os
from typing import Optional

try:
    import google.generativeai as genai

    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


def generate_summary(check_results: dict, score_result: dict) -> str:
    """
    Generate a plain-language officer summary.

    Args:
        check_results:  Raw check results dict (check_name -> result).
        score_result:   Output from scorer.compute_score().

    Returns:
        One sentence describing the most critical finding and recommended action.
    """
    tier = check_results.get("verification_tier")
    failed = score_result.get("failed_checks", [])

    # Strict policy gate: If QR is unreadable and no other check failed, never call Gemini (prevent hallucinating QR mismatch)
    if tier == "QR_UNREADABLE" and not failed:
        return "Format and physical checks passed, but QR unreadable: recommend camera recapture or secondary manual verification."

    if _GENAI_AVAILABLE and os.getenv("GEMINI_API_KEY"):
        try:
            return _llm_summary(check_results, score_result)
        except Exception:
            pass  # Fall through to rule-based

    return _rule_based_summary(check_results, score_result)


# ── LLM path ──────────────────────────────────────────────────────────────────


def _llm_summary(check_results: dict, score_result: dict) -> str:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")

    failed = list(score_result.get("failed_checks", []))
    status = score_result.get("status") or score_result.get("risk_level", "UNKNOWN")
    overall_score = score_result.get("overall_score") or score_result.get(
        "total_score", 0
    )
    tier = check_results.get("verification_tier", "STANDARD")

    face_match = check_results.get("face_match")
    if face_match and (
        face_match.get("matched") is False
        or face_match.get("verdict") == "MISMATCH"
    ):
        if "face_match" not in failed:
            failed.append("face_match")

    id_graph = check_results.get("identity_graph")
    if id_graph and id_graph.get("identity_conflict"):
        if "identity_conflict" not in failed:
            failed.append("identity_conflict")

    prompt = f"""You are a document security assistant at a border checkpoint.
A traveler has presented identity documents. Here is the verification result:

Document Verification Tier: {tier}
Risk Status: {status}
Risk Score: {overall_score:.0f}/100
Failed checks: {', '.join(failed) if failed else 'None'}
Check details: {json.dumps({k: v.get('score') for k, v in check_results.items() if isinstance(v, dict)}, indent=2)}

IMPORTANT RULES:
- If Verification Tier is QR_UNREADABLE, the QR was not decoded due to camera blur/glare. NEVER claim the QR was forged, tampered, or mismatched with text.
- If Verification Tier is QR_LEGACY, the card is pre-2017 without digital signature. Do not claim signature failure.
- State EXACTLY ONE concise sentence (maximum 28 words) stating the single most critical finding and recommended officer action (proceed / manual check / hold).
"""

    response = model.generate_content(prompt)
    return response.text.strip()


# ── Rule-based fallback ────────────────────────────────────────────────────────

_CHECK_PRIORITY = [
    "aadhaar_uidai_signature",
    "passport_active_auth",
    "passport_passive_auth",
    "identity_conflict",
    "face_match",
    "visa_passport_binding",
    "aadhaar_qr_ocr_consistency",
    "passport_mrz_viz_consistency",
    "aadhaar_verhoeff",
    "passport_mrz_checksums",
    "visa_rule_validation",
    "liveness",
    "ela_full_document",
    "ela_region_restricted",
]

_CRITICAL_CHECK_MESSAGES = {
    "aadhaar_uidai_signature": "Aadhaar QR signature invalid: document not issued by UIDAI.",
    "passport_passive_auth": "Passport digital signature cannot be verified against ICAO Master List: possible tampering.",
    "passport_active_auth": "Passport chip failed anti-cloning check: possible cloned chip.",
    "identity_conflict": "Identity conflict detected: traveler face previously seen under a different name or document.",
    "face_match": "Traveler face does not match document photo: possible identity mismatch.",
    "visa_passport_binding": "Visa is not issued for this passport booklet: possible document swap.",
    "aadhaar_qr_ocr_consistency": "Aadhaar printed text does not match cryptographically signed QR data.",
    "passport_mrz_viz_consistency": "Passport MRZ and biographical page printed data do not match.",
    "aadhaar_verhoeff": "Aadhaar Verhoeff checksum validation failed: invalid UID number.",
    "passport_mrz_checksums": "Passport MRZ check digits failed: possible counterfeit document.",
    "visa_rule_validation": "Visa rule violation detected: stay duration or dates exceed category limits.",
    "liveness": "Liveness check failed: possible photo presentation attack.",
    "ela_full_document": "Digital tampering detected via image error level analysis.",
    "ela_region_restricted": "Digital tampering detected in photo or QR region.",
}


def _rule_based_summary(check_results: dict, score_result: dict) -> str:
    failed = list(score_result.get("failed_checks", []))
    status = score_result.get("status") or score_result.get("risk_level", "CLEAR")
    overall_score = float(
        score_result.get("overall_score")
        or score_result.get("total_score", 0.0)
    )
    verification_tier = check_results.get("verification_tier")

    face_match = check_results.get("face_match")
    face_mismatched = bool(
        face_match
        and (
            face_match.get("matched") is False
            or face_match.get("verdict") == "MISMATCH"
        )
    )
    if face_mismatched and "face_match" not in failed:
        failed.append("face_match")

    id_graph = check_results.get("identity_graph")
    if id_graph and id_graph.get("identity_conflict") and "identity_conflict" not in failed:
        failed.append("identity_conflict")

    # Do not treat unreadable/legacy null signatures as cryptographic forgery
    sig_check = check_results.get("aadhaar_uidai_signature", {})
    if sig_check.get("valid") is None and "aadhaar_uidai_signature" in failed:
        failed.remove("aadhaar_uidai_signature")

    # QR unavailable is not a printed-vs-QR mismatch — never use that message
    consistency = check_results.get("aadhaar_qr_ocr_consistency") or {}
    if (
        verification_tier == "QR_UNREADABLE"
        or consistency.get("error") == "qr_data_unavailable"
        or consistency.get("consistent") is None
    ):
        if "aadhaar_qr_ocr_consistency" in failed:
            failed.remove("aadhaar_qr_ocr_consistency")

    # QR_UNREADABLE: always prefer recapture / secondary inspection wording
    if verification_tier == "QR_UNREADABLE":
        return (
            "Format and physical checks available, but QR unreadable: "
            "recommend camera recapture or secondary manual verification."
        )

    # If all checks passed and face matched
    if status == "CLEAR" and not face_mismatched and not failed:
        return "All checks passed: document appears genuine and traveler face matched, proceed normally."

    # If document passed but face mismatched
    if status == "CLEAR" and face_mismatched:
        return "Document checks passed, but traveler face does not match document photo: secondary manual identity verification required."

    # If document is flagged or in review, pick highest priority failed check
    for check_key in _CHECK_PRIORITY:
        if check_key in failed:
            msg = _CRITICAL_CHECK_MESSAGES.get(check_key, f"Check '{check_key}' failed.")
            if status == "FLAGGED":
                return f"{msg} Hold traveler for secondary inspection."
            else:
                return f"{msg} Recommend manual verification."

    if status == "FLAGGED":
        return f"Document FLAGGED (Risk Score: {overall_score:.0f}/100): multiple anomalies detected, hold traveler for inspection."

    if status == "REVIEW":
        return f"Document marked for REVIEW (Risk Score: {overall_score:.0f}/100): anomalies detected pending manual verification."

    return "All checks passed: document appears genuine, proceed normally."
