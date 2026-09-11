"""
Identity Graph  -  Cross-Visit, Cross-Document Identity Detection.

Detects when the same person appears at the checkpoint under different
names or document numbers across separate screening sessions.

Directly addresses the PS requirement: "Multiple identities used by the
same person."

Storage:
  - FAISS flat index: stores 512-dim ArcFace embedding vectors
  - SQLite: stores metadata (name, doc_number, doc_type, timestamp)
    per embedding. Index position in FAISS maps 1:1 to SQLite row_id.

Zero-image-storage principle:
  Only embedding vectors and field-value metadata are stored.
  No face photos are retained. Embeddings cannot be reverse-engineered
  to reconstruct a face image.
"""

import os
import sqlite3
from datetime import datetime
from typing import Optional

import numpy as np

try:
    import faiss

    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "identity_graph.db")
_INDEX_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "identity_graph.faiss"
)
_DIM = 512
_MATCH_THRESHOLD = 0.65  # Cosine similarity threshold for "same person"


# ── Public API ─────────────────────────────────────────────────────────────────


def search_and_store(
    embedding: np.ndarray,
    name: str,
    doc_number: str,
    doc_type: str,
    checkpoint_id: str = "default",
) -> dict:
    """
    Search the identity graph for the given embedding, then store it.

    Args:
        embedding:     512-dim unit vector from the face embedder.
        name:          Name as extracted from the document.
        doc_number:    Document number (passport/Aadhaar/visa).
        doc_type:      'Aadhaar' | 'Passport' | 'Visa'
        checkpoint_id: Identifier for this checkpoint (for multi-post deployments).

    Returns:
        dict with keys:
          matches (list[dict])      -  previous screenings of the same face
          identity_conflict (bool)  -  True if same face seen under different identity
          conflict_details (list)   -  details of conflicting records
          stored_id (int)           -  row ID of newly stored record
    """
    name = str(name or "UNKNOWN").strip()
    doc_number = str(doc_number or "UNKNOWN").strip()
    doc_type = str(doc_type or "UNKNOWN").strip()
    checkpoint_id = str(checkpoint_id or "default").strip()

    if not _FAISS_AVAILABLE:
        return {
            "matches": [],
            "identity_conflict": False,
            "conflict_details": [],
            "stored_id": None,
            "error": "faiss not installed",
        }

    index = _load_index()
    db = _get_db()

    matches = []
    conflicts = []

    if index.ntotal > 0:
        # Search k=5 nearest neighbors
        q = embedding.reshape(1, -1).astype(np.float32)
        D, I = index.search(q, k=min(5, index.ntotal))

        for dist, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            similarity = max(0.0, 1.0 - (dist * dist) / 2.0)

            if similarity >= _MATCH_THRESHOLD:
                record = _fetch_record(db, idx)
                if record:
                    match = {**record, "similarity": round(similarity, 4)}
                    matches.append(match)

                    # Identity conflict: same face, different name or document number
                    name_mismatch = (
                        record["name"].upper() != "UNKNOWN"
                        and name.upper() != "UNKNOWN"
                        and record["name"].upper() != name.upper()
                    )
                    docnum_mismatch = (
                        record["doc_number"] != "UNKNOWN"
                        and doc_number != "UNKNOWN"
                        and record["doc_number"] != doc_number
                    )
                    if name_mismatch or docnum_mismatch:
                        conflicts.append(
                            {
                                "previous_name": record["name"],
                                "current_name": name,
                                "previous_doc_number": record["doc_number"],
                                "current_doc_number": doc_number,
                                "previous_timestamp": record["timestamp"],
                                "similarity": round(similarity, 4),
                                "name_mismatch": name_mismatch,
                                "doc_number_mismatch": docnum_mismatch,
                            }
                        )

    # Store new record
    stored_id = _store_record(
        db, index, embedding, name, doc_number, doc_type, checkpoint_id
    )
    _save_index(index)

    db.close()

    return {
        "matches": matches,
        "identity_conflict": len(conflicts) > 0,
        "conflict_details": conflicts,
        "stored_id": stored_id,
        "error": None,
    }


def get_stats() -> dict:
    """Return basic statistics about the identity graph."""
    if not _FAISS_AVAILABLE:
        return {"total_embeddings": 0, "error": "faiss not installed"}
    try:
        index = _load_index()
        db = _get_db()
        cur = db.execute("SELECT COUNT(*) FROM identity_records")
        count = cur.fetchone()[0]
        db.close()
        return {"total_embeddings": index.ntotal, "total_records": count}
    except Exception as e:
        return {"error": str(e)}


# ── Storage helpers ────────────────────────────────────────────────────────────


def _load_index():
    """Load or create FAISS flat L2 index."""
    if os.path.exists(_INDEX_PATH):
        return faiss.read_index(_INDEX_PATH)
    return faiss.IndexFlatL2(_DIM)


def _save_index(index):
    faiss.write_index(index, _INDEX_PATH)


def _get_db() -> sqlite3.Connection:
    db = sqlite3.connect(_DB_PATH)
    db.execute("""
        CREATE TABLE IF NOT EXISTS identity_records (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            name         TEXT NOT NULL,
            doc_number   TEXT NOT NULL,
            doc_type     TEXT NOT NULL,
            checkpoint   TEXT NOT NULL,
            timestamp    TEXT NOT NULL,
            faiss_index  INTEGER NOT NULL
        )
    """)
    db.commit()
    return db


def _store_record(
    db,
    index,
    embedding: np.ndarray,
    name: str,
    doc_number: str,
    doc_type: str,
    checkpoint_id: str,
) -> Optional[int]:
    name = str(name or "UNKNOWN").strip()
    doc_number = str(doc_number or "UNKNOWN").strip()
    doc_type = str(doc_type or "UNKNOWN").strip()
    checkpoint_id = str(checkpoint_id or "default").strip()

    faiss_idx = index.ntotal
    index.add(embedding.reshape(1, -1).astype(np.float32))
    try:
        cur = db.execute(
            "INSERT INTO identity_records (name, doc_number, doc_type, checkpoint, timestamp, faiss_index) VALUES (?,?,?,?,?,?)",
            (
                name,
                doc_number,
                doc_type,
                checkpoint_id,
                datetime.utcnow().isoformat(),
                faiss_idx,
            ),
        )
        db.commit()
        return cur.lastrowid
    except Exception:
        return None


def _fetch_record(db, faiss_idx: int) -> Optional[dict]:
    cur = db.execute("SELECT * FROM identity_records WHERE faiss_index=?", (faiss_idx,))
    row = cur.fetchone()
    if not row:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))
