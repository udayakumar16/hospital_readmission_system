"""Print SQLite prediction history summary.

Run:
  D:/anti_project/.venv/Scripts/python.exe backend/check_db.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def main() -> int:
    db_path = Path(__file__).resolve().parent / "predictions.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("select name from sqlite_master where type='table' order by name")
    tables = cur.fetchall()
    print("db=", db_path)
    print("tables=", tables)

    cur.execute("select count(*) from predictions")
    print("rows=", cur.fetchone()[0])

    cur.execute(
        "select id, created_at, round(probability, 4), risk from predictions order by id desc limit 20"
    )
    print("last20=")
    for row in cur.fetchall():
        print(row)

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
