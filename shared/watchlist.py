"""
Local Watchlist  -  Flagged document lookup.

Addresses PS requirement: "Expired or blacklisted travel documents."

Maintains a local SQLite database of flagged document numbers.
In production this would sync with a government database.
For the hackathon, seeded with simulated demo data.

No network call required  -  works offline at a checkpoint.
"""

import os
import sqlite3
from datetime import datetime
from typing import Optional

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "watchlist.db")


def check_watchlist(doc_number: str, doc_type: str) -> dict:
    """
    Check if a document number is on the watchlist.

    Args:
        doc_number: Passport number, Aadhaar UID or visa number.
        doc_type:   'AADHAAR' | 'PASSPORT' | 'VISA'

    Returns:
        dict with keys: flagged (bool), reason (str|None), added_date (str|None)
    """
    db = _get_db()
    cur = db.execute(
        "SELECT reason, added_date, severity FROM watchlist WHERE doc_number=? AND doc_type=?",
        (doc_number.strip().upper(), doc_type.strip().upper()),
    )
    row = cur.fetchone()
    db.close()

    if row:
        return {
            "flagged": True,
            "reason": row[0],
            "added_date": row[1],
            "severity": row[2],
        }
    return {"flagged": False, "reason": None, "added_date": None, "severity": None}


def add_to_watchlist(
    doc_number: str, doc_type: str, reason: str, severity: str = "HIGH"
) -> bool:
    """Add a document to the watchlist."""
    db = _get_db()
    try:
        db.execute(
            "INSERT OR REPLACE INTO watchlist (doc_number, doc_type, reason, severity, added_date) VALUES (?,?,?,?,?)",
            (
                doc_number.strip().upper(),
                doc_type.strip().upper(),
                reason,
                severity,
                datetime.utcnow().isoformat(),
            ),
        )
        db.commit()
        return True
    except Exception:
        return False
    finally:
        db.close()


def seed_demo_data():
    """Seed watchlist with demo/simulated flagged documents for demonstration."""
    demo_entries = [
        ("X9999999", "PASSPORT", "Reported stolen - Interpol red notice", "HIGH"),
        ("123456789012", "AADHAAR", "Associated with identity fraud case", "HIGH"),
        ("TV1234567", "VISA", "Visa revoked - overstay record", "MEDIUM"),
        ("A0000001", "PASSPORT", "Demo flagged passport for testing", "LOW"),
    ]
    db = _get_db()
    for doc_number, doc_type, reason, severity in demo_entries:
        db.execute(
            "INSERT OR IGNORE INTO watchlist (doc_number, doc_type, reason, severity, added_date) VALUES (?,?,?,?,?)",
            (doc_number, doc_type, reason, severity, "2026-01-01T00:00:00"),
        )
    db.commit()
    db.close()


def _get_db() -> sqlite3.Connection:
    db = sqlite3.connect(_DB_PATH)
    db.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            doc_number TEXT NOT NULL,
            doc_type   TEXT NOT NULL,
            reason     TEXT,
            severity   TEXT DEFAULT 'HIGH',
            added_date TEXT,
            PRIMARY KEY (doc_number, doc_type)
        )
    """)
    db.commit()
    # Migrate existing entries to uppercase doc_types
    try:
        db.execute("UPDATE watchlist SET doc_type = UPPER(doc_type)")
        db.commit()
    except Exception:
        pass
    return db
