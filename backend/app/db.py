"""SQLite storage for scan history.

Kept deliberately tiny: one table, plain sqlite3 from the stdlib so there is no
extra dependency and it runs anywhere the container runs. Every diagnosis the app
makes is written here so the frontend can show a history screen.
"""

import sqlite3
import os
import json
from datetime import datetime, timezone
from contextlib import contextmanager

DB_PATH = os.environ.get("FASAL_DB_PATH", os.path.join(os.path.dirname(__file__), "scans.db"))


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                crop        TEXT NOT NULL,
                disease     TEXT NOT NULL,
                is_healthy  INTEGER NOT NULL DEFAULT 0,
                confidence  REAL NOT NULL,
                uncertain   INTEGER NOT NULL DEFAULT 0,
                advice      TEXT,
                created_at  TEXT NOT NULL
            )
            """
        )


def save_scan(crop, disease, is_healthy, confidence, uncertain, advice):
    """Persist one diagnosis. `advice` is stored as a JSON string (or None)."""
    with _conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO scans (crop, disease, is_healthy, confidence, uncertain, advice, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                crop,
                disease,
                1 if is_healthy else 0,
                float(confidence),
                1 if uncertain else 0,
                json.dumps(advice, ensure_ascii=False) if advice is not None else None,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        return cur.lastrowid


def recent_scans(limit=20):
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM scans ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    out = []
    for r in rows:
        out.append(
            {
                "id": r["id"],
                "crop": r["crop"],
                "disease": r["disease"],
                "is_healthy": bool(r["is_healthy"]),
                "confidence": r["confidence"],
                "uncertain": bool(r["uncertain"]),
                "advice": json.loads(r["advice"]) if r["advice"] else None,
                "created_at": r["created_at"],
            }
        )
    return out
