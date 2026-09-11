"""
modules/decision/cross_modal.py
Cross-modal identity consistency check.
Compares subject names across all available modalities:
  - Primary document OCR name (e.g. Aadhaar)
  - Spoken name from challenge audio ASR
  - Secondary document OCR name (e.g. Passport) if present

Uses RapidFuzz token_set_ratio. If any pair differs significantly,
flags the discrepancy as a possible identity impersonation attempt.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rapidfuzz.fuzz import token_set_ratio


def check_cross_modal_consistency(
    primary_ocr_name: str,
    spoken_name: str,
    secondary_ocr_name: str = "",
    asr_transcript: str = "",
) -> dict:
    """
    IN:
      primary_ocr_name:   name from primary document (e.g. Aadhaar)
      spoken_name:        what the subject said (from ASR transcript)
      secondary_ocr_name: name from secondary document (e.g. Passport) if present
      asr_transcript:     full ASR transcript for reference

    OUT: dict with:
      consistency_score:   0..1, higher = more inconsistent (more suspicious)
      pairs_checked:       list of name pairs compared
      mismatches:          list of failing pairs with ratios
      all_consistent:      bool
    """
    def _norm(s):
        import re
        return re.sub(r"[^a-z ]", "", (s or "").lower()).strip()

    def _similarity(a, b):
        if not a or not b:
            return 0.0
        norm_a = _norm(a)
        norm_b = _norm(b)
        if not norm_a or not norm_b:
            return 0.0
        return token_set_ratio(norm_a, norm_b) / 100.0

    pairs = []
    mismatches = []

    # Pair 1: primary doc vs spoken
    sim1 = _similarity(primary_ocr_name, spoken_name)
    pairs.append({
        "pair": "primary_doc vs spoken",
        "names": [primary_ocr_name, spoken_name],
        "similarity": round(sim1, 3),
        "consistent": sim1 >= 0.75
    })
    if sim1 < 0.75:
        mismatches.append(f"primary_doc '{primary_ocr_name}' vs "
                         f"spoken '{spoken_name}' ({sim1:.0%})")

    # Pair 2: primary doc vs secondary doc (if provided)
    if secondary_ocr_name:
        sim2 = _similarity(primary_ocr_name, secondary_ocr_name)
        pairs.append({
            "pair": "primary_doc vs secondary_doc",
            "names": [primary_ocr_name, secondary_ocr_name],
            "similarity": round(sim2, 3),
            "consistent": sim2 >= 0.75
        })
        if sim2 < 0.75:
            mismatches.append(f"primary_doc '{primary_ocr_name}' vs "
                             f"secondary_doc '{secondary_ocr_name}' ({sim2:.0%})")

        # Pair 3: secondary doc vs spoken
        sim3 = _similarity(secondary_ocr_name, spoken_name)
        pairs.append({
            "pair": "secondary_doc vs spoken",
            "names": [secondary_ocr_name, spoken_name],
            "similarity": round(sim3, 3),
            "consistent": sim3 >= 0.75
        })
        if sim3 < 0.75:
            mismatches.append(f"secondary_doc '{secondary_ocr_name}' vs "
                             f"spoken '{spoken_name}' ({sim3:.0%})")

    all_consistent = len(mismatches) == 0
    # Consistency score: fraction of pairs that failed, weighted by severity
    n_pairs = len(pairs)
    n_fail = len(mismatches)
    consistency_score = n_fail / max(n_pairs, 1)

    return {
        "consistency_score": round(consistency_score, 3),
        "pairs_checked":     pairs,
        "mismatches":        mismatches,
        "all_consistent":    all_consistent,
    }


if __name__ == "__main__":
    # Test 1: all match
    r1 = check_cross_modal_consistency(
        "Priya Sharma", "priya sharma", "PRIYA SHARMA")
    assert r1["all_consistent"] is True, f"Should be consistent: {r1}"
    print("Test 1 (all match): PASS")

    # Test 2: name mismatch between docs
    r2 = check_cross_modal_consistency(
        "Priya Sharma", "Priya Sharma", "Rahul Verma")
    assert r2["all_consistent"] is False
    assert len(r2["mismatches"]) >= 1
    print("Test 2 (doc mismatch): PASS")

    # Test 3: spoken name mismatch
    r3 = check_cross_modal_consistency(
        "Priya Sharma", "John Smith", "Priya Sharma")
    assert r3["all_consistent"] is False
    print("Test 3 (spoken mismatch): PASS")

    # Test 4: single document (no secondary)
    r4 = check_cross_modal_consistency("Alice Kumar", "alice kumar")
    assert r4["all_consistent"] is True
    print("Test 4 (single doc match): PASS")

    print("CROSS MODAL CONSISTENCY OK")

