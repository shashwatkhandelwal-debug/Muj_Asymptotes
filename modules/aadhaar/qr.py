"""
Aadhaar Secure QR Decoder.

Handles both:
  - New Secure QR (post-2017): base-10 integer → bytes → gzip → 0xFF-delimited fields
  - Old XML format (pre-2017): plain XML string

UIDAI Secure QR pipeline (User Manual for QR Code):
  1. Decode QR (PyZBar) → large base-10 digit string
  2. int(digits).to_bytes(...) → compressed byte array
  3. gzip / zlib decompress
  4. Split signature (last 256 bytes) from signed data
  5. Parse 0xFF-delimited fields: flag, name, DOB, gender, address…, photo
"""

import gzip
import logging
import struct
import xml.etree.ElementTree as ET
import zlib
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _pyzbar_decode(pil_img):
    """Lazy import so Secure QR parse helpers work without system libzbar in unit tests."""
    from pyzbar.pyzbar import decode as pyzbar_decode

    return pyzbar_decode(pil_img)

_DELIMITER = 0xFF
_SIGNATURE_LEN = 256


def _detect_and_decode_qr(
    image: np.ndarray,
) -> Tuple[Optional[bytes], Optional[Tuple[int, int, int, int]]]:
    """
    Multi-pass high-density QR code detector & decoder:
    1. Full image multi-filter sweeps (RGB, Gray, CLAHE, Otsu, Adaptive, Sharpened, Inverted)
    2. Regional crop sweeps (Right half, Left half, Center, 4 Quadrants)
    3. Multi-scale cubic interpolation upscaling (1.5x, 2.0x, 2.5x) for dense Version 14+ Aadhaar QRs
    4. OpenCV QRCodeDetector fallback
    """
    if image is None or image.size == 0:
        return None, None

    def try_decode(img_arr):
        if img_arr is None or getattr(img_arr, "size", 0) == 0:
            return None, None

        # 1. Try zxing-cpp (Fast C++ Reed-Solomon engine for dense Version 14+ QR codes)
        try:
            import zxingcpp
            res = zxingcpp.read_barcode(img_arr)
            if res and res.format == zxingcpp.BarcodeFormat.QRCode:
                raw_bytes = getattr(res, "bytes", None)
                if not raw_bytes and hasattr(res, "text") and res.text:
                    raw_bytes = res.text.encode("utf-8", errors="replace")
                if raw_bytes:
                    reg = None
                    if hasattr(res, "position"):
                        try:
                            pos = res.position
                            reg = (
                                min(pos.top_left.x, pos.bottom_left.x),
                                min(pos.top_left.y, pos.top_right.y),
                                max(pos.bottom_right.x, pos.top_right.x),
                                max(pos.bottom_right.y, pos.bottom_left.y),
                            )
                        except Exception:
                            pass
                    return bytes(raw_bytes), reg
        except Exception:
            pass

        # 2. Try PyZBar
        try:
            if isinstance(img_arr, np.ndarray):
                if len(img_arr.shape) == 2:
                    pil_img = Image.fromarray(img_arr)
                elif len(img_arr.shape) == 3 and img_arr.shape[2] == 3:
                    pil_img = Image.fromarray(cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB))
                elif len(img_arr.shape) == 3 and img_arr.shape[2] == 4:
                    pil_img = Image.fromarray(cv2.cvtColor(img_arr, cv2.COLOR_BGRA2RGB))
                else:
                    pil_img = Image.fromarray(img_arr)
            else:
                pil_img = img_arr
            objs = _pyzbar_decode(pil_img)
            for obj in objs:
                if obj.type == "QRCODE" and obj.data:
                    r = obj.rect
                    reg = (r.left, r.top, r.left + r.width, r.top + r.height)
                    return obj.data, reg
        except Exception:
            pass

        return None, None

    # Step 1: Direct full image passes
    data, reg = try_decode(image)
    if data:
        return data, reg

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    data, reg = try_decode(gray)
    if data:
        return data, reg

    # Filters on full image
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    data, reg = try_decode(enhanced)
    if data:
        return data, reg

    # Sharpening filter
    sharp_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, sharp_kernel)
    data, reg = try_decode(sharpened)
    if data:
        return data, reg

    # Otsu & Adaptive
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    data, reg = try_decode(otsu)
    if data:
        return data, reg

    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 11)
    data, reg = try_decode(adaptive)
    if data:
        return data, reg

    # Step 2: Regional crops for dense Aadhaar QR on card back/front
    h, w = gray.shape[:2]
    candidate_crops = [
        (int(0.30 * w), 0, w, h),               # Right 70% (standard card back)
        (0, int(0.30 * h), w, h),               # Bottom 70% (standard card front)
        (0, 0, int(0.70 * w), h),               # Left 70%
        (int(0.10 * w), int(0.10 * h), int(0.90 * w), int(0.90 * h)), # Center 80%
        (int(0.40 * w), 0, w, int(0.60 * h)),     # Top-Right quadrant
        (int(0.40 * w), int(0.40 * h), w, h),     # Bottom-Right quadrant
        (0, 0, int(0.60 * w), int(0.60 * h)),     # Top-Left quadrant
        (0, int(0.40 * h), int(0.60 * w), h),     # Bottom-Left quadrant
    ]

    for (x1, y1, x2, y2) in candidate_crops:
        crop_gray = gray[y1:y2, x1:x2]
        if crop_gray.size == 0:
            continue
        
        # Test crop directly
        data, sub_reg = try_decode(crop_gray)
        if data:
            abs_reg = (x1 + sub_reg[0], y1 + sub_reg[1], x1 + sub_reg[2], y1 + sub_reg[3]) if sub_reg else (x1, y1, x2, y2)
            return data, abs_reg

        # Test CLAHE + Upscale crop (up to 1400px) for high-density Version 14+ QR codes
        crop_enh = clahe.apply(crop_gray)
        ch, cw = crop_enh.shape[:2]
        if max(ch, cw) < 1400:
            scale = 1400.0 / float(max(ch, cw))
            upscaled = cv2.resize(crop_enh, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_CUBIC)
            data, _ = try_decode(upscaled)
            if data:
                return data, (x1, y1, x2, y2)

            # Test Otsu on upscaled crop
            _, crop_otsu = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            data, _ = try_decode(crop_otsu)
            if data:
                return data, (x1, y1, x2, y2)

            # Test Sharpening on upscaled crop
            crop_sharp = cv2.filter2D(upscaled, -1, sharp_kernel)
            data, _ = try_decode(crop_sharp)
            if data:
                return data, (x1, y1, x2, y2)

    # Step 3: Rotation sweeps (90 deg, 180 deg, 270 deg) for phone camera orientations
    for rot in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
        rot_img = cv2.rotate(gray, rot)
        data, _ = try_decode(rot_img)
        if data:
            return data, (0, 0, w, h)
        rot_enh = clahe.apply(rot_img)
        data, _ = try_decode(rot_enh)
        if data:
            return data, (0, 0, w, h)

    # Step 4: OpenCV QRCodeDetector fallback
    try:
        detector = cv2.QRCodeDetector()
        val, points, _ = detector.detectAndDecode(gray)
        if val:
            reg = None
            if points is not None and len(points) > 0:
                pts = points[0]
                x_min, y_min = int(np.min(pts[:, 0])), int(np.min(pts[:, 1]))
                x_max, y_max = int(np.max(pts[:, 0])), int(np.max(pts[:, 1]))
                reg = (x_min, y_min, x_max, y_max)
            return val.encode("utf-8", errors="replace"), reg
    except Exception:
        pass

    return None, None


def decode_aadhaar_qr(image: np.ndarray) -> dict:
    """
    Detect and decode the Aadhaar Secure QR from a card image.

    Returns:
        dict with keys: format ('xml'|'secure'), fields (dict), raw_payload (bytes),
        signature (bytes), error (str|None), region
    """
    logger.debug("Starting QR detection across multi-pass pipeline...")
    qr_data, region = _detect_and_decode_qr(image)

    if qr_data is None:
        logger.debug("No QR code detected")
        return {
            "error": "no_qr_detected",
            "fields": {},
            "region": None,
            "format": None,
            "raw_payload": None,
            "signature": None,
        }

    payload_len = len(qr_data)
    logger.debug("QR detected, raw payload length: %s bytes", payload_len)

    # XML (pre-2017) if UTF-8 and starts with '<'
    try:
        text = qr_data.decode("utf-8")
        if text.strip().startswith("<"):
            logger.debug("Detected format: XML (pre-2017)")
            res = _parse_xml_format(text, qr_data)
            res["region"] = region
            return res
        # Secure QR is a large decimal string
        if text.strip().isdigit() and len(text.strip()) > 50:
            logger.debug("Detected format: Secure QR (base-10 integer)")
            res = _parse_secure_qr_decimal(text.strip())
            res["region"] = region
            return res
    except UnicodeDecodeError:
        pass

    # Raw bytes that are actually ASCII digits
    try:
        as_text = qr_data.decode("ascii")
        if as_text.strip().isdigit() and len(as_text.strip()) > 50:
            res = _parse_secure_qr_decimal(as_text.strip())
            res["region"] = region
            return res
    except Exception:
        pass

    # Already-decompressed or non-digit binary — try gzip then delimiter parse
    logger.debug("Attempting secure QR parse on raw bytes")
    res = _parse_secure_qr_bytes(qr_data)
    res["region"] = region
    return res


# ── XML format (pre-2017) ──────────────────────────────────────────────────────


def _parse_xml_format(text: str, raw: bytes) -> dict:
    """Parse old-format Aadhaar QR (XML string + appended signature)."""
    try:
        root = ET.fromstring(text)
        fields = {
            "uid_last4": root.get("uid", ""),
            "name": root.get("name", ""),
            "dob": root.get("dob", ""),
            "gender": root.get("gender", ""),
            "co": root.get("co", ""),
            "house": root.get("house", ""),
            "street": root.get("street", ""),
            "lm": root.get("lm", ""),
            "loc": root.get("loc", ""),
            "vtc": root.get("vtc", ""),
            "subdist": root.get("subdist", ""),
            "dist": root.get("dist", ""),
            "state": root.get("state", ""),
            "pc": root.get("pc", ""),
            "po": root.get("po", ""),
        }
        signature = raw[-_SIGNATURE_LEN:] if len(raw) > _SIGNATURE_LEN else b""
        payload = raw[:-_SIGNATURE_LEN] if len(raw) > _SIGNATURE_LEN else raw
        return {
            "format": "xml",
            "fields": fields,
            "raw_payload": payload,
            "signature": signature if len(signature) == _SIGNATURE_LEN else None,
            "error": None,
        }
    except ET.ParseError as e:
        return {
            "format": "xml",
            "fields": {},
            "raw_payload": None,
            "signature": None,
            "error": f"secure_qr_parse_failed: XML parse error: {e}",
        }


# ── Secure QR (post-2017) ──────────────────────────────────────────────────────


def parse_secure_qr_payload(data) -> dict:
    """Parse Aadhaar Secure QR payload (either decimal string, XML string, or raw bytes)."""
    if isinstance(data, str):
        data_str = data.strip()
        if data_str.startswith("<"):
            return _parse_xml_format(data_str, data_str.encode("utf-8"))
        return _parse_secure_qr_decimal(data_str)
    elif isinstance(data, (bytes, bytearray)):
        return _parse_secure_qr_bytes(bytes(data))
    return {"error": "invalid_payload_type", "fields": {}}


def _parse_secure_qr_decimal(decimal_str: str) -> dict:
    """Convert base-10 Secure QR string → bytes → gzip → fields."""
    try:
        value = int(decimal_str)
        # Minimum byte length so that to_bytes does not raise
        byte_len = (value.bit_length() + 7) // 8
        if byte_len < 1:
            return _secure_parse_error("empty integer payload")
        compressed = value.to_bytes(byte_len, byteorder="big")
        return _parse_secure_qr_bytes(compressed)
    except Exception as e:
        return _secure_parse_error(f"decimal conversion failed: {e}")


def _decompress_secure_payload(data: bytes) -> bytes:
    """Gzip/zlib decompress Secure QR compressed bytes."""
    # Try gzip module first
    try:
        return gzip.decompress(data)
    except Exception:
        pass
    # zlib with gzip header (wbits=16+MAX_WBITS)
    try:
        return zlib.decompress(data, wbits=16 + zlib.MAX_WBITS)
    except Exception:
        pass
    # Raw zlib
    try:
        return zlib.decompress(data)
    except Exception:
        pass
    # Some scanners leave a leading null / padding byte
    if len(data) > 2 and data[0] == 0x00:
        return _decompress_secure_payload(data[1:])
    raise ValueError("gzip/zlib decompression failed")


def _parse_secure_qr_bytes(data: bytes) -> dict:
    """Decompress (if needed) and parse 0xFF-delimited Secure QR fields."""
    try:
        # Detect gzip magic 1f 8b, else try decompress anyway for Secure QR
        if len(data) >= 2 and data[0] == 0x1F and data[1] == 0x8B:
            decompressed = _decompress_secure_payload(data)
        else:
            try:
                decompressed = _decompress_secure_payload(data)
            except Exception:
                # Already decompressed delimiter payload
                decompressed = data

        if len(decompressed) <= _SIGNATURE_LEN:
            return _secure_parse_error(
                f"decompressed payload too short ({len(decompressed)} bytes)"
            )

        signature = decompressed[-_SIGNATURE_LEN:]
        signed_data = decompressed[:-_SIGNATURE_LEN]
        fields = _parse_delimiter_fields(signed_data)

        return {
            "format": "secure",
            "fields": fields,
            "raw_payload": signed_data,
            "signature": signature,
            "error": None,
        }
    except Exception as e:
        return _secure_parse_error(str(e))


def _secure_parse_error(msg: str) -> dict:
    return {
        "format": "secure",
        "fields": {},
        "raw_payload": None,
        "signature": None,
        "error": f"secure_qr_parse_failed: {msg}",
    }


def _split_delimiter_fields(data: bytes) -> List[bytes]:
    """Split signed data on 0xFF delimiters."""
    parts: List[bytes] = []
    start = 0
    for i, b in enumerate(data):
        if b == _DELIMITER:
            parts.append(data[start:i])
            start = i + 1
    if start <= len(data):
        parts.append(data[start:])
    # Drop empty trailing chunk if present
    while parts and parts[-1] == b"":
        parts.pop()
    return parts


def _decode_iso(chunk: bytes) -> str:
    return chunk.decode("iso-8859-1", errors="replace").strip("\x00").strip()


def _parse_delimiter_fields(signed_data: bytes) -> dict:
    """
    Parse UIDAI Secure QR fields from 0xFF-delimited signed data.
    Dynamically aligns V1 and V2 (post-2018 Reference ID) index offsets.
    """
    import re
    parts = _split_delimiter_fields(signed_data)
    if not parts:
        return {}

    # First field: email/mobile present bit indicator (0-3)
    flag_raw = _decode_iso(parts[0]) if parts else "0"
    try:
        email_mobile_flag = int(flag_raw) if flag_raw.isdigit() else (parts[0][0] if parts[0] else 0)
    except Exception:
        email_mobile_flag = 0

    def get(i: int) -> str:
        if 0 <= i < len(parts):
            return _decode_iso(parts[i])
        return ""

    # Find the Date of Birth index as the anchor
    # DOB matches DD-MM-YYYY, DD/MM/YYYY, or YYYY-MM-DD
    dob_idx = None
    for idx in range(1, min(len(parts), 6)):
        val = get(idx)
        if re.search(r"\b\d{2}[-/]\d{2}[-/]\d{4}\b", val) or re.search(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b", val):
            dob_idx = idx
            break

    if dob_idx is None:
        # Fallback to standard V1 offset if no date pattern found
        dob_idx = 2

    # In V2 (dob_idx == 3): parts[1] is Reference ID, parts[2] is Name, parts[3] is DOB, parts[4] is Gender
    # In V1 (dob_idx == 2): parts[1] is Name, parts[2] is DOB, parts[3] is Gender
    ref_id = get(1) if dob_idx > 2 else ""
    name_idx = dob_idx - 1
    gender_idx = dob_idx + 1

    name = get(name_idx)
    dob = get(dob_idx)
    gender = get(gender_idx)
    care_of = get(gender_idx + 1)
    district = get(gender_idx + 2)
    landmark = get(gender_idx + 3)
    house = get(gender_idx + 4)
    location = get(gender_idx + 5)
    pincode = get(gender_idx + 6)
    postoffice = get(gender_idx + 7)
    state = get(gender_idx + 8)
    street = get(gender_idx + 9)
    subdistrict = get(gender_idx + 10)
    vtc = get(gender_idx + 11)

    # Normalize gender to single letter / word
    g = gender.upper()[:1] if gender else ""
    gender_norm = {"M": "M", "F": "F", "T": "T"}.get(g, gender)

    # Photo is typically after VTC
    photo = None
    photo_idx = gender_idx + 12
    if len(parts) > photo_idx:
        photo_bytes = parts[photo_idx]
        if len(photo_bytes) > 100:
            photo = photo_bytes

    # Extract last 4 digits from ref_id if available (ref_id starts with 4 digits)
    uid_last4 = ref_id[:4] if (ref_id and ref_id[:4].isdigit()) else ""

    fields = {
        "email_mobile_flag": email_mobile_flag,
        "ref_id": ref_id,
        "name": name,
        "dob": dob,
        "gender": gender_norm,
        "care_of": care_of,
        "district": district,
        "landmark": landmark,
        "house": house,
        "location": location,
        "pincode": pincode,
        "postoffice": postoffice,
        "state": state,
        "street": street,
        "subdistrict": subdistrict,
        "vtc": vtc,
        "uid_last4": uid_last4,
        "has_photo": bool(photo),
    }
    return fields


# Backwards-compatible alias used by older call sites / tests
def _parse_binary_format(data: bytes) -> dict:
    """Deprecated alias — routes to Secure QR parser."""
    return _parse_secure_qr_bytes(data)


def _read_lp_string(data: bytes, offset: int) -> tuple:
    """Legacy helper retained for tests; reads 2-byte BE length-prefixed UTF-8."""
    length = struct.unpack_from(">H", data, offset)[0]
    offset += 2
    text = data[offset : offset + length].decode("utf-8", errors="replace")
    return text, offset + length
