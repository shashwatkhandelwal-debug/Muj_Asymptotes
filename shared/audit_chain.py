import hashlib
import json
import sqlite3
import time
import os

def chain_hash(pepper: bytes, prev_hash: str, payload: dict) -> str:
    """
    Returns SHA256(pepper + prev_hash.encode() + canonical_json(payload)).
    canonical_json = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    """
    canonical_json = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode("utf-8")
    data = pepper + prev_hash.encode("utf-8") + canonical_json
    return hashlib.sha256(data).hexdigest()

def append_entry(db: sqlite3.Connection, pepper: bytes, payload: dict) -> str:
    """
    1. Query: SELECT entry_hash FROM audit_log ORDER BY rowid DESC LIMIT 1
       If no rows exist, prev_hash = '0' * 64  (genesis sentinel)
    2. new_hash = chain_hash(pepper, prev_hash, payload)
    3. INSERT INTO audit_log (prev_hash, entry_hash, payload_json, ts)
       VALUES (prev_hash, new_hash, json.dumps(payload), time.time())
       Create the table first if it does not exist (use CREATE TABLE IF NOT EXISTS).
    4. Return new_hash.
    """
    cursor = db.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            prev_hash TEXT NOT NULL,
            entry_hash TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            ts REAL NOT NULL
        )
    """)
    cursor.execute("SELECT entry_hash FROM audit_log ORDER BY rowid DESC LIMIT 1")
    row = cursor.fetchone()
    if row is None:
        prev_hash = "0" * 64
    else:
        prev_hash = row[0]

    new_hash = chain_hash(pepper, prev_hash, payload)
    cursor.execute(
        "INSERT INTO audit_log (prev_hash, entry_hash, payload_json, ts) VALUES (?, ?, ?, ?)",
        (prev_hash, new_hash, json.dumps(payload), time.time()),
    )
    db.commit()
    return new_hash

def verify_chain(db: sqlite3.Connection, pepper: bytes) -> tuple[bool, int]:
    """
    SELECT rowid, prev_hash, entry_hash, payload_json FROM audit_log ORDER BY rowid ASC.
    For each row recompute chain_hash(pepper, prev_hash, json.loads(payload_json))
    and compare to stored entry_hash.
    Also verify that each row's prev_hash equals the previous row's entry_hash.
    Returns (True, -1) if the chain is intact.
    Returns (False, first_broken_rowid) on the first mismatch found.
    """
    cursor = db.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            prev_hash TEXT NOT NULL,
            entry_hash TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            ts REAL NOT NULL
        )
    """)
    cursor.execute("SELECT rowid, prev_hash, entry_hash, payload_json FROM audit_log ORDER BY rowid ASC")
    rows = cursor.fetchall()
    if not rows:
        return (True, -1)

    expected_prev = "0" * 64
    for rowid, prev_hash, entry_hash, payload_json in rows:
        if prev_hash != expected_prev:
            return (False, rowid)
        recomputed = chain_hash(pepper, prev_hash, json.loads(payload_json))
        if recomputed != entry_hash:
            return (False, rowid)
        expected_prev = entry_hash

    return (True, -1)

if __name__ == "__main__":
    db = sqlite3.connect(":memory:")
    pepper = os.urandom(32)
    append_entry(db, pepper, {"event": "login", "user": "alice"})
    append_entry(db, pepper, {"event": "verify", "doc_id": "12345"})
    append_entry(db, pepper, {"event": "logout", "user": "alice"})

    intact, broken_id = verify_chain(db, pepper)
    assert intact is True and broken_id == -1, f"Chain verify failed: {intact}, {broken_id}"

    # Corrupt entry_hash of row 2
    cursor = db.cursor()
    cursor.execute("UPDATE audit_log SET entry_hash = 'corruptedhash' WHERE rowid = 2")
    db.commit()

    intact_after, broken_id_after = verify_chain(db, pepper)
    assert intact_after is False and broken_id_after == 2, f"Expected broken rowid 2, got {broken_id_after}"

    print("AUDIT CHAIN OK")
