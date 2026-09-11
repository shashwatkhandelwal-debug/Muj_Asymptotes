"""
MRZ Parser  -  ICAO 9303 TD3 format (full passport, 2 × 44 chars).

Parses both lines, extracts all fields and validates check digits
using the ICAO 7-3-1 cyclic weighting algorithm.
"""

import re
from datetime import datetime
from typing import Optional

WEIGHTS = [7, 3, 1]

CHAR_VALUES = {c: i + 10 for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
CHAR_VALUES.update({str(i): i for i in range(10)})
CHAR_VALUES["<"] = 0


def icao_check_digit(field: str) -> int:
    """Compute ICAO 7-3-1 weighted check digit for a field string."""
    total = sum(CHAR_VALUES.get(c, 0) * WEIGHTS[i % 3] for i, c in enumerate(field))
    return total % 10


def parse_mrz(mrz_line1: str, mrz_line2: str) -> dict:
    """
    Parse a TD3 MRZ (2 × 44 chars) and validate all check digits.

    Returns dict with:
      doc_type, issuing_country, surname, given_names,
      passport_number, nationality, dob, sex, expiry,
      personal_number, check_digits (all pass/fail),
      valid (overall bool), raw (dict of raw field strings)
    """
    l1 = mrz_line1.upper().replace(" ", "<")
    l2 = mrz_line2.upper().replace(" ", "<")

    if len(l1) != 44 or len(l2) != 44:
        return {
            "valid": False,
            "error": f"MRZ line length invalid: {len(l1)}, {len(l2)}",
        }

    # ── Line 1 ──
    doc_type = l1[0:2].replace("<", "").strip()
    issuing_ctry = l1[2:5]
    name_field = l1[5:44]
    surname, *given = name_field.split("<<")
    given_names = " ".join(" ".join(g.split("<")) for g in given).strip()
    surname = surname.replace("<", " ").strip()

    # ── Line 2 ──
    passport_num = l2[0:9]
    cd_pn = int(l2[9])
    nationality = l2[10:13]
    dob_raw = l2[13:19]
    cd_dob = int(l2[19])
    sex = l2[20]
    expiry_raw = l2[21:27]
    cd_expiry = int(l2[27])
    personal_num = l2[28:42]
    cd_personal = int(l2[42])
    cd_overall = int(l2[43])
    overall_field = l2[0:10] + l2[13:20] + l2[21:43]

    # ── Check digit validation ──
    checks = {
        "passport_number": icao_check_digit(passport_num) == cd_pn,
        "dob": icao_check_digit(dob_raw) == cd_dob,
        "expiry": icao_check_digit(expiry_raw) == cd_expiry,
        "personal_number": icao_check_digit(personal_num) == cd_personal,
        "overall": icao_check_digit(overall_field) == cd_overall,
    }
    all_valid = all(checks.values())

    return {
        "valid": all_valid,
        "doc_type": doc_type,
        "issuing_country": issuing_ctry,
        "surname": surname,
        "given_names": given_names,
        "passport_number": passport_num.replace("<", ""),
        "nationality": nationality,
        "dob": _parse_date(dob_raw),
        "sex": sex,
        "expiry": _parse_date(expiry_raw),
        "personal_number": personal_num.replace("<", ""),
        "check_digits": checks,
        "raw": {
            "line1": l1,
            "line2": l2,
            "passport_number": passport_num,
            "dob": dob_raw,
            "expiry": expiry_raw,
        },
    }


def _parse_date(yymmdd: str) -> Optional[str]:
    """Convert YYMMDD to DD/MM/YYYY. Handles century pivot at 30."""
    try:
        yy, mm, dd = int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
        year = 2000 + yy if yy < 30 else 1900 + yy
        return f"{dd:02d}/{mm:02d}/{year}"
    except Exception:
        return None


def parse_mrz_from_image(image, reader=None) -> Optional[dict]:
    """
    Extract and parse MRZ lines directly from the bottom strip of a passport bio-page image.
    Uses multi-pass bounding box reconstruction and OCR preprocessing.
    """
    if image is None or getattr(image, "size", 0) == 0:
        return None

    try:
        import cv2
        from modules.passport.viz import _get_reader

        ocr_reader = reader or _get_reader()

        # Try multiple passes: (1) Full Image (2) Grayscale Threshold (3) Bottom 40% Crop
        passes = [image]
        h, w = image.shape[:2]
        if h > 100 and w > 100:
            passes.append(image[int(h * 0.55):, :])

        for img_variant in passes:
            results = ocr_reader.readtext(img_variant, detail=1)
            # Sort detected boxes vertically from top to bottom
            results.sort(key=lambda item: item[0][0][1] if (isinstance(item, (list, tuple)) and len(item) > 0) else 0)

            raw_lines = [str(item[1]).replace(" ", "").upper() for item in results if len(item) > 1]
            
            # Find candidate MRZ lines (contains '<' or starts with 'P<' / 'P' + country code)
            mrz_candidates = []
            for line in raw_lines:
                # Replace common OCR misreads in MRZ chevrons
                clean = line.replace("«", "<").replace("(", "<").replace(")", "<").replace("{", "<").replace("}", "<")
                if "<" in clean or clean.startswith("P") or len(clean) >= 28:
                    mrz_candidates.append(clean)

            # If we have 2 or more candidates at the end of the text
            if len(mrz_candidates) >= 2:
                # Take the last two candidates
                l1_cand = mrz_candidates[-2]
                l2_cand = mrz_candidates[-1]

                # Ensure length 44 by padding with <
                l1 = l1_cand[:44].ljust(44, "<")
                l2 = l2_cand[:44].ljust(44, "<")

                # If l1 doesn't start with P<, try prefixing P<
                if not l1.startswith("P"):
                    if l1_cand.startswith("IND") or l1_cand.startswith("USA") or l1_cand.startswith("GBR"):
                        l1 = ("P<" + l1_cand)[:44].ljust(44, "<")

                parsed = parse_mrz(l1, l2)
                if parsed and parsed.get("valid"):
                    return parsed
                elif parsed and not parsed.get("error"):
                    return parsed

        # Fallback to simple line scan if multi-pass didn't validate
        full_res = ocr_reader.readtext(image, detail=0)
        full_candidates = [
            r.replace(" ", "").replace("«", "<").upper()
            for r in full_res
            if len(r.replace(" ", "")) >= 25
        ]
        if len(full_candidates) >= 2:
            l1 = full_candidates[-2][:44].ljust(44, "<")
            l2 = full_candidates[-1][:44].ljust(44, "<")
            return parse_mrz(l1, l2)
    except Exception as e:
        return {"valid": False, "error": f"MRZ image extraction failed: {str(e)}"}
    return None
