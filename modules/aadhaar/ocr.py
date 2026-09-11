"""
Aadhaar OCR  -  Extract printed fields from Aadhaar card image.

Primary engine: OpenBharatOCR.
Fallback: EasyOCR + spaCy NER (en_core_web_sm only).

Extracts:
  - UID number (12 digits)
  - Name (English)
  - Name (Hindi / Devanagari)
  - Date of Birth (DD/MM/YYYY)
  - Gender
  - Address
"""

import logging
import os
import re
import tempfile
from typing import Optional

import cv2
import easyocr
import numpy as np

logger = logging.getLogger(__name__)

_reader: Optional[easyocr.Reader] = None
_nlp = None
_nlp_failed = False

_HINDI_SLOGAN_MARKERS = (
    "मेरा आधार",
    "मेरी पहचान",
    "मेरी पहिचान",
    "आधार पहचान",
    "भारत सरकार",
)


def _clean_corrupt_easyocr_models():
    """Remove partial or corrupted model files from EasyOCR cache directory."""
    try:
        model_dir = os.path.expanduser("~/.EasyOCR/model")
        if os.path.exists(model_dir):
            for fname in os.listdir(model_dir):
                fpath = os.path.join(model_dir, fname)
                if os.path.isfile(fpath) and (
                    fname.endswith(".pth") or fname.endswith(".tmp")
                ):
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass
    except Exception:
        pass


def _preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Multi-stage contrast, scaling and edge enhancement for OCR."""
    if image is None or image.size == 0:
        return image

    if len(image.shape) == 2:
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img = image.copy()

    h, w = img.shape[:2]
    target_w = 1280
    if w > 0 and w != target_w:
        scale = target_w / float(w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(
            img,
            (new_w, new_h),
            interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA,
        )

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced_lab = cv2.merge((l_enhanced, a, b))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    gaussian = cv2.GaussianBlur(enhanced_bgr, (0, 0), 2.0)
    unsharp = cv2.addWeighted(enhanced_bgr, 1.4, gaussian, -0.4, 0)
    return unsharp


def _get_reader() -> easyocr.Reader:
    """Lazy-init EasyOCR reader with robust fallback and corrupted cache recovery."""
    global _reader
    if _reader is None:
        try:
            _reader = easyocr.Reader(["en", "hi"], gpu=False)
        except (AssertionError, Exception):
            _clean_corrupt_easyocr_models()
            try:
                _reader = easyocr.Reader(["en"], gpu=False)
            except (AssertionError, Exception):
                _clean_corrupt_easyocr_models()
                _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def _try_openbharatocr(image: np.ndarray) -> Optional[dict]:
    """Attempt extraction using OpenBharatOCR library."""
    temp_path = None
    try:
        import openbharatocr

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
            cv2.imwrite(temp_path, image)

        res = openbharatocr.front_aadhaar(temp_path)
        if not isinstance(res, dict):
            logger.warning("OpenBharatOCR returned non-dict result: %s", type(res))
            return None

        uid = res.get("Aadhaar Number") or res.get("aadhaar_number") or ""
        uid_clean = re.sub(r"[\s\-]", "", str(uid)) if uid else None
        name_en = res.get("Name") or res.get("name") or None
        dob = res.get("DOB") or res.get("dob") or res.get("Date of Birth") or None
        gender = res.get("Gender") or res.get("gender") or None
        address = res.get("Address") or res.get("address") or None
        name_hi = res.get("Name (Hindi)") or res.get("name_hindi") or None

        if not (
            (uid_clean and len(uid_clean) == 12 and uid_clean.isdigit())
            or name_en
            or dob
            or gender
        ):
            return None

        if name_en:
            name_en = str(name_en).strip() or None
        if name_hi and _is_hindi_slogan(str(name_hi)):
            name_hi = None

        return {
            "uid": (
                uid_clean
                if (uid_clean and len(uid_clean) == 12 and uid_clean.isdigit())
                else None
            ),
            "name_en": name_en,
            "name_hi": name_hi,
            "dob": dob,
            "gender": gender,
            "address": address,
            "raw_text": str(res),
            "confidence": 0.95,
            "ocr_engine": "openbharatocr",
        }
    except ImportError:
        logger.info("OpenBharatOCR not installed; falling back to EasyOCR")
        return None
    except Exception as e:
        logger.warning("OpenBharatOCR extraction failed: %s", e)
        return None
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def _get_spacy_nlp():
    """Lazy-load spaCy en_core_web_sm only (blank model has no PERSON NER)."""
    global _nlp, _nlp_failed
    if _nlp_failed:
        return None
    if _nlp is None:
        try:
            import spacy

            _nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.warning(
                "spaCy en_core_web_sm unavailable (%s). "
                "Run: python -m spacy download en_core_web_sm",
                e,
            )
            _nlp_failed = True
            return None
    return _nlp


def extract_aadhaar_fields(image: np.ndarray) -> dict:
    """
    Extract all printed fields from an Aadhaar card image using OpenBharatOCR
    and multi-pass OCR with spaCy NER.

    Returns:
        dict with keys: uid, name_en, name_hi, dob, gender, address,
        raw_text, confidence, ocr_engine
    """
    obo_result = _try_openbharatocr(image)
    if (
        obo_result
        and obo_result.get("uid")
        and obo_result.get("name_en")
        and obo_result.get("dob")
        and obo_result.get("gender")
    ):
        return obo_result

    # Prefer OpenBharat when UID or name present; fill gaps from EasyOCR
    need_easy = True
    if obo_result and (obo_result.get("uid") or obo_result.get("name_en")):
        need_easy = not (
            obo_result.get("uid")
            and obo_result.get("name_en")
            and obo_result.get("dob")
            and obo_result.get("gender")
        )

    uid = obo_result.get("uid") if obo_result else None
    name_en = obo_result.get("name_en") if obo_result else None
    name_hi = obo_result.get("name_hi") if obo_result else None
    dob = obo_result.get("dob") if obo_result else None
    gender = obo_result.get("gender") if obo_result else None
    address = obo_result.get("address") if obo_result else None
    raw_text = (obo_result.get("raw_text") if obo_result else "") or ""
    avg_confidence = float(obo_result.get("confidence", 0.0)) if obo_result else 0.0
    engine = "openbharatocr" if obo_result else "easyocr"

    if need_easy or not (uid and name_en):
        reader = _get_reader()
        enhanced = _preprocess_for_ocr(image)
        results = reader.readtext(enhanced, detail=1)

        easy_raw = " ".join([r[1] for r in results])
        confidences = [r[2] for r in results]
        easy_conf = float(np.mean(confidences)) if confidences else 0.0
        raw_text = f"{raw_text} {easy_raw}".strip()
        if not obo_result:
            avg_confidence = easy_conf
        else:
            avg_confidence = max(avg_confidence, easy_conf)
            engine = "openbharatocr+easyocr"

        if not uid:
            uid = _extract_uid(easy_raw)
        if not dob:
            dob = _extract_dob(easy_raw)
        if not gender:
            gender = _extract_gender(easy_raw)
        if not name_en:
            name_en = _extract_name_english(easy_raw, results)
        if not name_hi:
            name_hi = _extract_name_hindi(results)
        if not address:
            address = _extract_address(easy_raw)

        if not uid or not name_en:
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7
            )
            p2_results = reader.readtext(thresh, detail=1)
            p2_raw_text = " ".join([r[1] for r in p2_results])
            if not uid:
                uid = _extract_uid(p2_raw_text)
            if not name_en:
                name_en = _extract_name_english(p2_raw_text, p2_results)
            if not dob:
                dob = _extract_dob(p2_raw_text)
            if not gender:
                gender = _extract_gender(p2_raw_text)
            if not name_hi:
                name_hi = _extract_name_hindi(p2_results)
            raw_text = f"{raw_text} {p2_raw_text}".strip()

    return {
        "uid": uid,
        "name": name_en,
        "name_en": name_en,
        "dob": dob,
        "gender": gender,
        "address": address,
        "raw_text": raw_text,
        "confidence": round(avg_confidence, 4),
        "ocr_engine": engine,
    }


# ── Private helpers ────────────────────────────────────────────────────────────


def _extract_uid(text: str) -> Optional[str]:
    """
    Extract 12-digit UID from explicit 4-4-4 blocks or isolated 12-digit numbers.
    Does NOT concatenate unrelated numeric blocks (e.g. dates or phone numbers).
    """
    match = re.search(r"\b(\d{4}[\s\-]\d{4}[\s\-]\d{4})\b", text)
    if match:
        candidate = re.sub(r"[\s\-]", "", match.group(1))
        if len(candidate) == 12 and candidate.isdigit():
            return candidate

    cleaned = re.sub(r"[Oo]", "0", text)
    cleaned = re.sub(r"[Il|]", "1", cleaned)
    cleaned = re.sub(
        r"[SsbB]",
        lambda m: {"S": "5", "s": "5", "b": "6", "B": "8"}.get(
            m.group(0), m.group(0)
        ),
        cleaned,
    )
    match = re.search(r"\b(\d{4}[\s\-]\d{4}[\s\-]\d{4})\b", cleaned)
    if match:
        candidate = re.sub(r"[\s\-]", "", match.group(1))
        if len(candidate) == 12 and candidate.isdigit():
            return candidate

    match = re.search(r"\b(\d{12})\b", cleaned)
    if match:
        return match.group(1)

    return None


def _extract_dob(text: str) -> Optional[str]:
    """Extract DOB in DD/MM/YYYY format with flexible separators and prefixes."""
    match = re.search(
        r"(?:DOB|Birth|Date|जन्म)?[:\s]*(\d{2}[/.\-\s]\d{2}[/.\-\s]\d{4})\b",
        text,
        re.IGNORECASE,
    )
    if match:
        raw_dob = match.group(1)
        return re.sub(r"[/.\-\s]+", "/", raw_dob)
    match = re.search(
        r"(?:Year of Birth|YOB|जन्म वर्ष)[:\s]+(\d{4})", text, re.IGNORECASE
    )
    if match:
        return match.group(1)
    return None


def _extract_gender(text: str) -> Optional[str]:
    """Detect gender from printed text in English or Hindi."""
    text_upper = text.upper()
    if re.search(r"\b(FEMALE|महिला|WOMAN|WOMEN)\b", text_upper):
        return "FEMALE"
    if re.search(r"\b(MALE|पुरुष|MAN|MEN)\b", text_upper):
        return "MALE"
    if re.search(r"\b(TRANSGENDER|किन्नर)\b", text_upper):
        return "TRANSGENDER"
    return None


def _looks_like_ocr_junk(cand: str) -> bool:
    """Reject garbled OCR strings that are unlikely to be real names."""
    c = cand.strip()
    if not c or len(c) < 4:
        return True
    # Mixed case noise with digit-like OCR artifacts (e.g. PAPunAYNN RclOOO)
    if re.search(r"\d", c):
        return True
    letters = re.sub(r"[^A-Za-z]", "", c)
    if not letters:
        return True
    # Too many consecutive consonants without vowels often indicates OCR garbage
    if re.search(r"[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{6,}", letters):
        return True
    # Alternating case mid-token like PApunAYNN
    tokens = c.split()
    for t in tokens:
        if len(t) >= 4 and re.search(r"[a-z][A-Z]", t) and re.search(r"[A-Z]{2,}[a-z]", t):
            return True
    return False


def _extract_name_english(
    text: str, line_results: Optional[list] = None
) -> Optional[str]:
    """
    Extract English name from line bounding boxes or raw text.
    Handles ALL CAPS names and Title Case names accurately.
    """
    EXCLUDE = {
        "GOVERNMENT",
        "INDIA",
        "UIDAI",
        "AADHAAR",
        "MALE",
        "FEMALE",
        "DATE",
        "BIRTH",
        "ADDRESS",
        "YEAR",
        "DOB",
        "HELP",
        "ENROLMENT",
        "MERI",
        "PEHCHAN",
        "BHARAT",
        "SARKAR",
        "MY",
        "CARD",
        "UNIQUE",
        "IDENTIFICATION",
        "AUTHORITY",
        "FATHER",
        "NAME",
        "HUSBAND",
        "VID",
        "OF",
        "TO",
        "THE",
        "IN",
        "AND",
        "OR",
        "NO",
        "PH",
        "PIN",
    }

    import Levenshtein

    HEADER_STOPWORDS = [
        "GOVERNMENT OF INDIA",
        "UNIQUE IDENTIFICATION AUTHORITY OF INDIA",
        "BHARAT SARKAR",
        "DATE OF BIRTH",
        "YEAR OF BIRTH",
        "MERI AADHAAR",
        "ENROLMENT NO",
        "HELP LINE",
        "GOVERNMENT",
        "INDIA",
        "AADHAAR",
        "UIDAI",
    ]

    def is_header_noise(cand: str) -> bool:
        c_up = cand.upper().strip()
        for hw in HEADER_STOPWORDS:
            if hw in c_up:
                return True
            if len(c_up) >= 4:
                hw_prefix = hw[: min(len(hw), len(c_up))]
                if Levenshtein.distance(c_up, hw_prefix) <= 2:
                    return True
        return False

    def is_valid_name(cand: str) -> bool:
        if not cand or is_header_noise(cand) or _looks_like_ocr_junk(cand):
            return False
        words = [w.upper() for w in re.findall(r"[A-Za-z]+", cand)]
        if len(words) < 2 or any(w in EXCLUDE for w in words):
            return False
        return bool(re.match(r"^[A-Za-z\s.'-]+$", cand.strip()) and len(cand.strip()) >= 4)

    if line_results:
        for item in line_results:
            line_str = (
                item[1]
                if isinstance(item, (list, tuple)) and len(item) > 1
                else str(item)
            )
            cleaned_line = line_str.strip()
            cleaned_line = re.sub(
                r"^(?:Name|नाम|Applicant Name)[:\s]+",
                "",
                cleaned_line,
                flags=re.IGNORECASE,
            ).strip()
            if is_valid_name(cleaned_line):
                return cleaned_line

    filtered_text = text
    for stop_word in HEADER_STOPWORDS:
        filtered_text = re.sub(stop_word, " ", filtered_text, flags=re.IGNORECASE)

    nlp = _get_spacy_nlp()
    if nlp:
        try:
            doc = nlp(filtered_text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    clean_ent = re.sub(r"[^A-Za-z\s.'-]", "", ent.text).strip()
                    if is_valid_name(clean_ent):
                        return clean_ent
        except Exception:
            pass

    words_with_case = re.findall(r"\b[A-Za-z.'-]+\b", filtered_text)
    for i in range(len(words_with_case) - 1):
        for length in [2, 3, 4]:
            if i + length <= len(words_with_case):
                cand_words = words_with_case[i : i + length]
                if not any(w.upper() in EXCLUDE for w in cand_words):
                    candidate = " ".join(cand_words)
                    if (
                        all(w[0].isupper() for w in cand_words)
                        and is_valid_name(candidate)
                    ):
                        return candidate
    return None


def _is_hindi_slogan(text: str) -> bool:
    return any(m in text for m in _HINDI_SLOGAN_MARKERS)


def _extract_name_hindi(results: list) -> Optional[str]:
    """
    Extract Hindi name — Devanagari script characters (U+0900–U+097F).
    Excludes Aadhaar slogan / government header lines.
    """
    candidates = []
    for item in results:
        text = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else str(item)
        text = text.strip()
        if not re.search(r"[\u0900-\u097F]", text):
            continue
        if _is_hindi_slogan(text):
            continue
        # Prefer short person-name lines over long address blocks
        if len(text) > 60:
            continue
        candidates.append(text)

    if not candidates:
        return None
    # Prefer the shortest Devanagari line (names are typically short)
    candidates.sort(key=len)
    return candidates[0]


def _extract_address(text: str) -> Optional[str]:
    """
    Address extraction: text following 'Address:' or 'S/O', 'D/O', 'C/O' markers.
    """
    match = re.search(
        r"(?:Address|S/O|D/O|C/O|पता)[:\s]+(.+)", text, re.IGNORECASE | re.DOTALL
    )
    if match:
        return match.group(1).strip()[:300]
    return None
