#!/usr/bin/env python3
# storage.py — persistence layer (SQLite or Supabase)

from __future__ import annotations
from typing import Any, Dict, List
from pathlib import Path
import json
import datetime as dt
import sqlite3
import os

# Supabase (optional)
try:
    from supabase import create_client
except ImportError:
    create_client = None


def _now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# ---------- SQLite (local) ----------
def _ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def _sqlite_connect(db_path: Path) -> sqlite3.Connection:
    _ensure_dir(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def _sqlite_init(conn: sqlite3.Connection, table: str):
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,
            final_url TEXT,
            status INTEGER,
            fetched_at TEXT,
            record_json TEXT
        )
    """)
    conn.commit()


# ---------- Supabase (remote) ----------
def _supabase_insert(records: List[Dict[str, Any]], cfg: Dict[str, Any], meta: Dict[str, Any]):
    if create_client is None:
        raise ImportError("Install supabase-py: pip install supabase")

    url = os.getenv("SUPABASE_URL") or cfg["storage"]["options"].get("url")
    key = os.getenv("SUPABASE_KEY") or cfg["storage"]["options"].get("key")
    if not url or not key:
        raise ValueError("Supabase credentials missing. Set SUPABASE_URL and SUPABASE_KEY.")

    supabase = create_client(url, key)
    table = cfg["storage"]["options"].get("table", "records")
    fetched_at = _now_iso()

    data = [
        {
            "url": meta.get("url"),
            "final_url": meta.get("final_url"),
            "status": int(meta.get("status", 0)),
            "fetched_at": fetched_at,
            "record_json": rec,
        }
        for rec in records
    ]

    res = supabase.table(table).insert(data).execute()
    if res.get("error"):
        raise RuntimeError(res["error"])


# ---------- Dispatcher ----------
def save(records: List[Dict[str, Any]], cfg: Dict[str, Any], meta: Dict[str, Any]) -> None:
    backend = cfg.get("storage", {}).get("backend", "none")

    if backend == "none":
        return
    elif backend == "sqlite":
        opts = cfg["storage"].get("options", {}) or {}
        db_path = Path(opts.get("path", "Webscraper/data/scraper.sqlite"))
        table = opts.get("table", "records")
        conn = _sqlite_connect(db_path)
        _sqlite_init(conn, table)
        fetched_at = _now_iso()
        conn.executemany(
            f"INSERT INTO {table} (url, final_url, status, fetched_at, record_json) VALUES (?, ?, ?, ?, ?)",
            [
                (meta.get("url"), meta.get("final_url"), meta.get("status"), fetched_at, json.dumps(r))
                for r in records
            ],
        )
        conn.commit()
        conn.close()
    elif backend == "supabase":
        _supabase_insert(records, cfg, meta)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
