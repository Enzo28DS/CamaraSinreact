# learned_db.py  (PYTHON 3.9 compatible)
import sqlite3
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "learned_objects.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS learned_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    label TEXT NOT NULL,
    emb BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_learned_label ON learned_embeddings(label);
CREATE INDEX IF NOT EXISTS idx_learned_ts ON learned_embeddings(ts);
"""

def init_db(db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()

def save_object(label: str, emb, ts: Optional[str] = None, db_path: Path = DB_PATH) -> None:
    init_db(db_path)
    if ts is None:
        ts = datetime.now().isoformat(timespec="seconds")

    emb = np.asarray(emb, dtype="float32")
    blob = emb.tobytes()

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "INSERT INTO learned_embeddings (ts, label, emb) VALUES (?, ?, ?)",
            (ts, label, blob)
        )
        conn.commit()
    finally:
        conn.close()

def load_embeddings_grouped(db_path: Path = DB_PATH) -> Dict[str, List[np.ndarray]]:
    init_db(db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT label, emb FROM learned_embeddings").fetchall()
        groups: Dict[str, List[np.ndarray]] = {}
        for label, blob in rows:
            vec = np.frombuffer(blob, dtype="float32")
            groups.setdefault(label, []).append(vec)
        return groups
    finally:
        conn.close()

def get_last_learned(n: int = 5, db_path: Path = DB_PATH) -> List[Tuple[str, str]]:
    init_db(db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT label, ts FROM learned_embeddings ORDER BY id DESC LIMIT ?",
            (int(n),)
        ).fetchall()
        return [(r[0], r[1]) for r in rows]
    finally:
        conn.close()

def get_label_counts(limit: int = 15, db_path: Path = DB_PATH) -> List[Tuple[str, int]]:
    init_db(db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            """
            SELECT label, COUNT(*) as c
            FROM learned_embeddings
            GROUP BY label
            ORDER BY c DESC
            LIMIT ?
            """,
            (int(limit),)
        ).fetchall()
        return [(r[0], int(r[1])) for r in rows]
    finally:
        conn.close()
