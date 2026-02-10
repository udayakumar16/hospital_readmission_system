"""SQLite helper for storing prediction history."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def get_conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db(db_path: Path) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                probability REAL NOT NULL,
                risk TEXT NOT NULL,
                input_json TEXT NOT NULL
            );
            """
        )
        conn.commit()


def insert_prediction(
    db_path: Path,
    created_at: str,
    probability: float,
    risk: str,
    input_payload: dict[str, Any],
) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO predictions (created_at, probability, risk, input_json) VALUES (?, ?, ?, ?)",
            (created_at, float(probability), str(risk), json.dumps(input_payload)),
        )
        conn.commit()
