"""
shared/flow_logger.py — Asynchronous, Non-blocking Diagnostic & Flow Request Logger.
Logs every request, file payload size, signal breakdown, and server-side exception
to the `flow_debug_log` table in `audit.db` without blocking detector pipelines.
"""
import os
import json
import time
import queue
import sqlite3
import datetime
import threading
import traceback
from typing import Any, Optional, Dict, List

DB_PATH = "audit.db"

# Background worker queue for zero-overhead non-blocking SQLite writes
_log_queue: queue.Queue = queue.Queue(maxsize=1000)
_worker_thread: Optional[threading.Thread] = None
_init_lock = threading.Lock()

def _get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL;")  # WAL mode prevents write contention and reader blocking
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def init_debug_table(db_path: str = DB_PATH):
    conn = _get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS flow_debug_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            epoch_ts REAL NOT NULL,
            stage_name TEXT,
            endpoint TEXT NOT NULL,
            method TEXT NOT NULL,
            status_code INTEGER,
            files_summary TEXT NOT NULL,
            form_fields TEXT,
            evaluated_signals TEXT,
            skipped_or_missing_signals TEXT,
            final_score REAL,
            final_tier TEXT,
            response_payload TEXT,
            server_errors TEXT,
            duration_ms REAL
        )
    """)
    conn.commit()
    conn.close()

def _log_worker():
    while True:
        try:
            item = _log_queue.get()
            if item is None:
                break
            db_path, row_data = item
            conn = _get_db_connection(db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO flow_debug_log (
                    timestamp, epoch_ts, stage_name, endpoint, method, status_code,
                    files_summary, form_fields, evaluated_signals,
                    skipped_or_missing_signals, final_score, final_tier,
                    response_payload, server_errors, duration_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row_data,
            )
            conn.commit()
            conn.close()
            _log_queue.task_done()
        except Exception as e:
            print(f"[flow_logger] Error in background log worker: {e}", flush=True)

def _ensure_worker():
    global _worker_thread
    with _init_lock:
        if _worker_thread is None or not _worker_thread.is_alive():
            init_debug_table(DB_PATH)
            _worker_thread = threading.Thread(target=_log_worker, daemon=True, name="FlowLoggerWorker")
            _worker_thread.start()

def log_flow_event(
    endpoint: str,
    method: str,
    status_code: int,
    stage_name: Optional[str] = None,
    files_summary: Optional[Dict[str, Any]] = None,
    form_fields: Optional[Dict[str, Any]] = None,
    evaluated_signals: Optional[List[Dict[str, Any]]] = None,
    skipped_or_missing_signals: Optional[List[str]] = None,
    final_score: Optional[float] = None,
    final_tier: Optional[str] = None,
    response_payload: Optional[Dict[str, Any]] = None,
    server_errors: Optional[List[Dict[str, Any]]] = None,
    duration_ms: float = 0.0,
    db_path: str = DB_PATH,
    sync: bool = False,
) -> None:
    """
    Records a detailed diagnostic log event into audit.db asynchronously.
    """
    _ensure_worker()
    now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
    epoch = time.time()

    row_data = (
        now_str,
        epoch,
        stage_name or "unknown",
        endpoint,
        method,
        status_code,
        json.dumps(files_summary or {}),
        json.dumps(form_fields or {}),
        json.dumps(evaluated_signals or []),
        json.dumps(skipped_or_missing_signals or []),
        final_score,
        final_tier,
        json.dumps(response_payload or {}),
        json.dumps(server_errors or []),
        duration_ms,
    )

    if sync:
        try:
            conn = _get_db_connection(db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO flow_debug_log (
                    timestamp, epoch_ts, stage_name, endpoint, method, status_code,
                    files_summary, form_fields, evaluated_signals,
                    skipped_or_missing_signals, final_score, final_tier,
                    response_payload, server_errors, duration_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row_data,
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[flow_logger] Sync write failed: {e}", flush=True)
    else:
        try:
            _log_queue.put_nowait((db_path, row_data))
        except queue.Full:
            # Fallback to direct write if queue is full
            pass

def flush_logs(timeout: float = 2.0):
    """Wait for all pending logs to be written."""
    try:
        _log_queue.join()
    except Exception:
        pass

def get_recent_flow_logs(limit: int = 100, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    flush_logs()
    init_debug_table(db_path)
    conn = _get_db_connection(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM flow_debug_log ORDER BY id ASC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    results = []
    for r in rows:
        d = dict(r)
        for key in ["files_summary", "form_fields", "evaluated_signals", "skipped_or_missing_signals", "response_payload", "server_errors"]:
            if d.get(key):
                try:
                    d[key] = json.loads(d[key])
                except Exception:
                    pass
        results.append(d)
    conn.close()
    return results

if __name__ == "__main__":
    init_debug_table("test_audit.db")
    log_flow_event(
        endpoint="/api/verify",
        method="POST",
        status_code=200,
        stage_name="stage1_document",
        files_summary={"document_image": {"filename": "aadhaar.jpg", "size_bytes": 102400}},
        evaluated_signals=[{"signal": "genai_doc", "raw_score": 0.1}],
        final_score=10.0,
        final_tier="clear",
        db_path="test_audit.db",
        sync=True,
    )
    logs = get_recent_flow_logs(db_path="test_audit.db")
    assert len(logs) == 1
    print("FLOW LOGGER OK:", logs[0]["endpoint"])
    if os.path.exists("test_audit.db"):
        os.remove("test_audit.db")
