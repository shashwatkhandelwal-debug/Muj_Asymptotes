import os
import random
import logging

logger = logging.getLogger(__name__)

# Precompute GF(256) tables with generator 2 and primitive polynomial 0x11d
EXP = [0] * 512
LOG = [0] * 256
_x = 1
for _i in range(255):
    EXP[_i] = _x
    EXP[_i + 255] = _x
    LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x11d

def _gf_mul(a: int, b: int) -> int:
    return 0 if a == 0 or b == 0 else EXP[LOG[a] + LOG[b]]

def _gf_div(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError("GF(256) division by zero")
    return 0 if a == 0 else EXP[(LOG[a] - LOG[b] + 255) % 255]

_USE_SECRETSHARING = False
try:
    from secretsharing import PlaintextToHexSecretSharer
    # Test if it runs in this Python environment without Python 2 'long' error
    _t_shares = PlaintextToHexSecretSharer.split_secret("aa", 2, 3)
    _USE_SECRETSHARING = True
except Exception:
    _USE_SECRETSHARING = False

def generate_pepper() -> bytes:
    """Return os.urandom(32)."""
    return os.urandom(32)

def split_pepper(pepper: bytes, k: int = 2, n: int = 3) -> list[str]:
    """
    Split pepper into n shares, any k sufficient to reconstruct.
    Return list of n hex strings (one per officer).
    Use PlaintextToHexSecretSharer from the secretsharing package:
      from secretsharing import PlaintextToHexSecretSharer
      shares = PlaintextToHexSecretSharer.split_secret(pepper.hex(), k, n)
    If secretsharing is unavailable, implement GF(256) Shamir and return
    shares as "i-hexvalue" strings (e.g. "1-a3f2...").
    """
    if _USE_SECRETSHARING:
        try:
            return PlaintextToHexSecretSharer.split_secret(pepper.hex(), k, n)
        except Exception as e:
            logger.warning("secretsharing failed, using GF(256) fallback: %s", e)

    # GF(256) Shamir implementation
    shares: list[str] = []
    pepper_len = len(pepper)
    coeffs = [
        [pepper[j]] + [random.randint(1, 255) for _ in range(k - 1)]
        for j in range(pepper_len)
    ]
    for i in range(1, n + 1):
        share_bytes = bytearray(pepper_len)
        for j in range(pepper_len):
            val = 0
            x_pow = 1
            for deg in range(k):
                val ^= _gf_mul(coeffs[j][deg], x_pow)
                x_pow = _gf_mul(x_pow, i)
            share_bytes[j] = val
        shares.append(f"{i}-{bytes(share_bytes).hex()}")
    return shares

def reconstruct_pepper(shares: list[str]) -> bytes:
    """
    Reconstruct from any k shares.
    Use PlaintextToHexSecretSharer.recover_secret(shares) and bytes.fromhex().
    If hand-rolled, parse "i-hexvalue" format and do Lagrange interpolation in GF(256).
    Zero out any intermediate bytes objects before returning (use ctypes or bytearray + memset).
    """
    if not shares:
        raise ValueError("No shares provided for reconstruction")

    # If shares are in standard secretsharing format (no hyphen index prefix like '1-hex')
    if _USE_SECRETSHARING and "-" not in shares[0]:
        try:
            recovered_hex = PlaintextToHexSecretSharer.recover_secret(shares)
            return bytes.fromhex(recovered_hex)
        except Exception as e:
            logger.warning("secretsharing recover failed: %s", e)

    # GF(256) Shamir reconstruction
    xs: list[int] = []
    ys: list[bytes] = []
    for s in shares:
        if "-" in s:
            idx_str, hex_val = s.split("-", 1)
            xs.append(int(idx_str))
            ys.append(bytes.fromhex(hex_val))
        else:
            # Fallback format: single share or hex
            xs.append(1)
            ys.append(bytes.fromhex(s))

    k = len(xs)
    num_bytes = len(ys[0])
    recovered = bytearray(num_bytes)

    for j in range(num_bytes):
        secret_byte = 0
        for m in range(k):
            num = 1
            den = 1
            for p in range(k):
                if p != m:
                    num = _gf_mul(num, xs[p])
                    den = _gf_mul(den, xs[m] ^ xs[p])
            l_m = _gf_div(num, den)
            secret_byte ^= _gf_mul(ys[m][j], l_m)
        recovered[j] = secret_byte

    result = bytes(recovered)
    # Zero out intermediate bytearray
    for idx in range(len(recovered)):
        recovered[idx] = 0

    return result

if __name__ == "__main__":
    pepper = generate_pepper()
    shares = split_pepper(pepper, 2, 3)
    reconstructed = reconstruct_pepper([shares[0], shares[2]])
    assert reconstructed == pepper, f"Expected {pepper.hex()}, got {reconstructed.hex()}"
    print("PEPPER SSS OK")
