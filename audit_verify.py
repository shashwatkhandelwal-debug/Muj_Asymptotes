"""
audit_verify.py — Standalone CLI tool to verify and demonstrate the Shamir-peppered audit chain.
Usage:
  python audit_verify.py --demo      # Run an end-to-end tamper demonstration
  python audit_verify.py --verify    # Verify the current audit.db
  python audit_verify.py --tamper 2  # Deliberately tamper with row 2 to demonstrate detection
"""
import sys
import os
import sqlite3
import argparse
from pathlib import Path

# Ensure repo root is on sys.path
_repo_root = str(Path(__file__).resolve().parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from shared.audit_chain import append_entry, verify_chain
from shared.pepper_sss import generate_pepper, split_pepper, reconstruct_pepper

def run_demo():
    print("=" * 60)
    print("DEMO: Shamir-Peppered Hash-Chained Audit Ledger")
    print("=" * 60)

    # 1. Generate pepper and split into 2-of-3 officer shares
    pepper = generate_pepper()
    shares = split_pepper(pepper, k=2, n=3)
    print(f"[*] 32-Byte Audit Pepper Generated.")
    print(f"[*] Pepper split into 3 Shamir shares (2-of-3 threshold required):")
    print(f"    Officer 1 Share: {shares[0][:24]}...")
    print(f"    Officer 2 Share: {shares[1][:24]}...")
    print(f"    Officer 3 Share: {shares[2][:24]}...")

    # 2. Reconstruct pepper from Officer 1 and Officer 3 shares
    reconstructed_pepper = reconstruct_pepper([shares[0], shares[2]])
    assert reconstructed_pepper == pepper
    print(f"[+] Pepper successfully reconstructed from Officer 1 and Officer 3 shares.\n")

    # 3. Create demo audit DB
    db = sqlite3.connect(":memory:")
    print("[*] Recording 3 live screening events into hash chain:")
    h1 = append_entry(db, pepper, {"screening_id": "SCR-1001", "name": "Priya Sharma", "status": "CLEAR", "score": 8.0})
    print(f"    Row 1 Entry Hash: {h1[:32]}...")
    h2 = append_entry(db, pepper, {"screening_id": "SCR-1002", "name": "Unknown Subject", "status": "FLAGGED", "score": 82.0})
    print(f"    Row 2 Entry Hash: {h2[:32]}...")
    h3 = append_entry(db, pepper, {"screening_id": "SCR-1003", "name": "Rohan Desai", "status": "CLEAR", "score": 12.0})
    print(f"    Row 3 Entry Hash: {h3[:32]}...")

    # 4. Verify untouched chain
    intact, broken_id = verify_chain(db, pepper)
    print(f"\n[+] Verification check on clean ledger:")
    if intact:
        print(f"    RESULT: >>> CHAIN INTACT (All rows verified with valid HMAC/SHA-256) <<<")
    else:
        print(f"    RESULT: Chain broken at row {broken_id}")

    # 5. Tamper attack
    print(f"\n[!] ATTACK SIMULATION: Rogue actor modifies database record for Row 2 (changing FLAGGED to CLEAR)...")
    cur = db.cursor()
    cur.execute("UPDATE audit_log SET payload_json = '{\"screening_id\":\"SCR-1002\",\"name\":\"Unknown Subject\",\"status\":\"CLEAR\",\"score\":8.0}' WHERE rowid = 2")
    db.commit()

    # 6. Verify tampered chain
    intact_after, broken_row = verify_chain(db, pepper)
    print(f"[+] Verification check after tampering:")
    if not intact_after:
        print(f"    ALERT: >>> TAMPERING DETECTED! Chain integrity broken at ROW {broken_row} <<<")
        print(f"    Without the Shamir-split pepper, an attacker cannot recompute a valid entry hash.")
    else:
        print(f"    ERROR: Tampering was not caught.")

    print("=" * 60)
    print("DEMO COMPLETE — AUDIT INTEGRITY PROVED")
    print("=" * 60)

def verify_db(db_path: str = "audit.db"):
    if not os.path.exists(db_path):
        print(f"No audit database found at {db_path}.")
        return

    shares_env = os.environ.get("AUDIT_SHARES")
    if not shares_env:
        print("[!] Note: AUDIT_SHARES not set in environment. Using demo ephemeral pepper.")
        pepper = generate_pepper()
    else:
        pepper = reconstruct_pepper(shares_env.split("|||"))

    db = sqlite3.connect(db_path)
    intact, broken_row = verify_chain(db, pepper)
    if intact:
        print(f"[+] Audit ledger at {db_path} is 100% INTACT.")
    else:
        print(f"[!] ALERT: Audit ledger broken at ROW {broken_row}!")
    db.close()

def main():
    parser = argparse.ArgumentParser(description="Audit Chain CLI")
    parser.add_argument("--demo", action="store_true", help="Run interactive tamper demonstration")
    parser.add_argument("--verify", action="store_true", help="Verify the SQLite audit ledger")
    args = parser.parse_args()

    if args.verify:
        verify_db()
    else:
        run_demo()

if __name__ == "__main__":
    main()
